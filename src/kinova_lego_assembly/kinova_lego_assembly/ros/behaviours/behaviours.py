"""Leaf behaviours for the LEGO assembly behaviour tree.

Each behaviour is a thin ROS adapter: it pulls inputs from the shared
:class:`OrchestratorContext` and the ``assembly`` blackboard, performs one
blocking ROS operation, writes any outputs back to the blackboard, and
returns ``SUCCESS`` or ``FAILURE``. Business logic (blueprint lookups,
scene validation, placement geometry) lives in ``core/``.

The tree is fail-fast: no retries yet. Behaviours are still small and
independently tickable so retry decorators can wrap them later.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from time import monotonic

import py_trees
import rclpy
from geometry_msgs.msg import PoseStamped
from kinova_lego_assembly_interfaces.msg import AssemblyStatus
from rclpy.exceptions import InvalidHandle

from kinova_lego_assembly.core.blueprint import ModelStep
from kinova_lego_assembly.core.orchestration import Phase
from kinova_lego_assembly.core.placement import PlacementError
from kinova_lego_assembly.core.scene_validation import count_detected, verify
from kinova_lego_assembly.ros.behaviours.context import OrchestratorContext
from kinova_lego_assembly.ros.clients import DEFAULT_GOAL_TOLERANCE, GoalTolerance, MotionPlanner
from kinova_lego_assembly.ros.grasp import compute_grasp_pregrasp_candidates, shift_candidate_x
from kinova_lego_assembly.ros.placement import (
    FkPlacementProvider,
    PlacementRecord,
    PlaceSlot,
)

try:
    import tf2_geometry_msgs  # noqa: F401  (registers PoseStamped tf support)

    _TF2_GEOMETRY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on system ROS installation
    _TF2_GEOMETRY_AVAILABLE = False


_PROGRESS_KEYS = ("phase", "brick_index", "total_bricks", "message")


class OrchestratorBehaviour(py_trees.behaviour.Behaviour):
    """Base class wiring context, blackboard, and progress reporting."""

    def __init__(
        self,
        name: str,
        context: OrchestratorContext,
        phase: Phase,
        brick_index: int = -1,
    ) -> None:
        """Store context and register the shared progress blackboard keys."""
        super().__init__(name)
        self.context = context
        self._phase = phase
        self._brick_index = brick_index
        self.bb = self.attach_blackboard_client(name=name, namespace="assembly")
        for key in _PROGRESS_KEYS:
            self.bb.register_key(key=key, access=py_trees.common.Access.WRITE)

    def _set_progress(self, message: str) -> None:
        """Publish this behaviour's phase/brick/message onto the blackboard."""
        self.bb.phase = self._phase.value
        self.bb.brick_index = self._brick_index
        self.bb.message = message
        self.context.node.get_logger().info(f"[{self._phase.value}] {message}")
        if self.context.status_pub is None:
            return
        # Guard against a torn-down node: on shutdown the publisher handle may
        # already be destroyed while a tick is still in flight, so skip (rather
        # than crash the tick) if the context is gone or the handle is invalid.
        if not rclpy.ok():
            return
        status = AssemblyStatus()
        status.phase = self._phase.value
        status.brick_index = int(self._brick_index)
        status.total_bricks = int(self.context.total_bricks)
        status.message = message
        try:
            self.context.status_pub.publish(status)
        except InvalidHandle:
            self.context.node.get_logger().warning(
                "status publisher unavailable (node shutting down); dropped status update"
            )

    @property
    def _log(self):
        """Shortcut to the orchestrator node logger."""
        return self.context.node.get_logger()


class MoveToJointGoal(OrchestratorBehaviour):
    """Plan+execute a joint-space goal to a fixed joint configuration.

    The arm ends in exactly the given configuration instead of whichever IK
    solution a pose goal would pick. Fail-fast: an empty or mismatched
    configuration or a planning/execution failure returns ``FAILURE``.
    """

    def __init__(
        self,
        context: OrchestratorContext,
        name: str,
        joint_names: list[str],
        joint_positions: list[float],
        velocity_scaling: float,
        planner: MotionPlanner,
        description: str,
        phase: Phase = Phase.PICKING,
    ) -> None:
        """Create the joint-goal move behaviour.

        Parameters
        ----------
        context : OrchestratorContext
            Shared runtime context.
        name : str
            Behaviour name.
        joint_names : list[str]
            Joint names of the goal configuration.
        joint_positions : list[float]
            Goal joint positions in radians, in ``joint_names`` order.
        velocity_scaling : float
            Velocity scaling factor for the move.
        planner : MotionPlanner
            Pipeline and planner for the move.
        description : str
            Human-readable progress description.
        phase : Phase, optional
            Orchestration phase reported while running.
        """
        super().__init__(name, context, phase)
        self._joint_names = list(joint_names)
        self._joint_positions = [float(p) for p in joint_positions]
        self._velocity_scaling = float(velocity_scaling)
        self._planner = planner
        self._description = description

    def update(self) -> py_trees.common.Status:
        """Move to the joint configuration."""
        if not self._joint_names or len(self._joint_names) != len(self._joint_positions):
            self._set_progress(f"{self._description} failed: joint goal is empty or mismatched")
            return py_trees.common.Status.FAILURE

        self._set_progress(self._description)
        ok, message = self.context.move_client.move_to_joint_goal(
            joint_names=self._joint_names,
            joint_positions=self._joint_positions,
            planner=self._planner,
            max_velocity=self._velocity_scaling,
        )
        if not ok:
            self._set_progress(f"{self._description} failed: {message}")
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class SetGripper(OrchestratorBehaviour):
    """Open the gripper, or close it for a grasp."""

    def __init__(
        self,
        context: OrchestratorContext,
        mode: str,
        brick_index: int = -1,
        position: float | None = None,
        phase: Phase | None = None,
    ) -> None:
        """Create a gripper behaviour.

        Parameters
        ----------
        context : OrchestratorContext
            Shared runtime context.
        mode : str
            Either ``"open"`` or ``"close"``.
        brick_index : int, optional
            Brick index for progress reporting. Defaults to ``-1``.
        position : float or None, optional
            Explicit joint position (rad) to drive to. When given it
            overrides the configured open/close position for either mode
            (e.g. a full clamp to snap a stack together) and skips the
            closed-band check. ``None`` (the default) uses the standard
            open/close behaviour.
        phase : Phase or None, optional
            Phase reported for this gripper action. ``None`` (the default)
            falls back to ``Phase.PICKING``/``Phase.MOVING_HOME`` based on
            ``mode``.
        """
        if mode not in ("open", "close"):
            raise ValueError(f"SetGripper mode must be 'open' or 'close', got {mode!r}")
        if phase is None:
            phase = Phase.PICKING if mode == "close" else Phase.MOVING_HOME
        super().__init__(f"Gripper_{mode}", context, phase, brick_index)
        self._mode = mode
        self._position = position

    def update(self) -> py_trees.common.Status:
        """Command the gripper and block until it settles."""
        timeout = self.context.params.gripper_timeout_s

        if self._position is not None:
            self._set_progress(f"driving gripper to {self._position:.3f} rad")
            ok = self.context.gripper.move_to(self._position, timeout_s=timeout)
            if not ok:
                self._set_progress("gripper position command failed")
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.SUCCESS

        if self._mode == "open":
            self._set_progress("opening gripper")
            ok = self.context.gripper.open(timeout_s=timeout)
            if not ok:
                self._set_progress("gripper open command failed")
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.SUCCESS

        self._set_progress("closing gripper slowly")
        reached = self.context.gripper.close(timeout_s=timeout)
        if not reached:
            self._set_progress(
                "gripper did not reach the expected closed band "
                "(position-only check; brick presence not verified)"
            )
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class WaitForArmSettled(OrchestratorBehaviour):
    """Wait until the arm joints have stopped moving before sensing."""

    def __init__(self, context: OrchestratorContext, phase: Phase, brick_index: int = -1) -> None:
        """Create the settle-wait behaviour for the given progress phase."""
        super().__init__("WaitForArmSettled", context, phase, brick_index)

    def update(self) -> py_trees.common.Status:
        """Block until joint velocities stay near zero; fail on timeout."""
        params = self.context.params
        self._set_progress("waiting for the arm to settle")
        settled = self.context.move_client.wait_until_settled(
            velocity_tolerance=params.settle_velocity_tolerance,
            hold_s=params.settle_hold_s,
            timeout_s=params.settle_timeout_s,
        )
        if not settled:
            self._set_progress("arm did not settle in time")
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class CaptureImage(OrchestratorBehaviour):
    """Grab a single colour frame and store it on the blackboard."""

    def __init__(self, context: OrchestratorContext) -> None:
        """Create the capture behaviour and register the image key."""
        super().__init__("CaptureImage", context, Phase.VALIDATING_SCENE)
        self.bb.register_key(key="scene_image", access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        """Wait for the next colour frame; fail if none arrives in time."""
        self._set_progress("capturing colour image for scene validation")
        image = self.context.sensor.get_color_image_ros(
            self.context.params.camera_color_topic,
            timeout_s=self.context.params.image_timeout_s,
        )
        if image is None:
            self._set_progress("no colour image received")
            return py_trees.common.Status.FAILURE
        self.bb.scene_image = image
        return py_trees.common.Status.SUCCESS


class ValidateScene(OrchestratorBehaviour):
    """Segment the captured frame and check enough bricks exist for the recipe."""

    def __init__(self, context: OrchestratorContext) -> None:
        """Create the validation behaviour and register blackboard keys."""
        super().__init__("ValidateScene", context, Phase.VALIDATING_SCENE)
        self.bb.register_key(key="scene_image", access=py_trees.common.Access.READ)
        self.bb.register_key(key="recipe", access=py_trees.common.Access.READ)
        self.bb.register_key(key="placement_record", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        """Run segmentation and hard-fail on any brick shortfall.

        When resuming a partial assembly, only the bricks not yet placed are
        demanded from the scene.
        """
        self._set_progress("validating scene against recipe demand")
        response = self.context.segment_client.segment(
            self.bb.scene_image,
            timeout_s=self.context.params.segment_timeout_s,
        )
        if response is None:
            self._set_progress("segmentation service call failed")
            return py_trees.common.Status.FAILURE

        detected_names = [det.class_name for det in response.detections]
        counts = count_detected(detected_names)
        self._log.info(f"segmentation returned {len(detected_names)} detection(s); tally={counts}")
        slot = (self.bb.get("placement_record") or {}).get(self.context.place_index)
        placed = slot.placed_steps() if slot else set()
        recipe = self.bb.recipe
        remaining = replace(recipe, steps=[s for s in recipe.steps if s.index not in placed])
        report = verify(counts, remaining)
        self._set_progress(report.summary())
        if not report.enough:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class DetectObjects(OrchestratorBehaviour):
    """Trigger the MMOR object detection action for the current step."""

    def __init__(self, context: OrchestratorContext, step: ModelStep) -> None:
        """Create the detection behaviour and register the detection-time key."""
        super().__init__(f"Detect_brick{step.index}", context, Phase.PICKING, step.index)
        self._step = step
        self.bb.register_key(key="detection_time", access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        """Record the request time then call the detection action."""
        self._set_progress(f"beginning pick of brick '{self._step.brick_name}': detecting objects")
        self.bb.detection_time = monotonic()
        ok = self.context.detection_client.detect(timeout_s=self.context.params.detection_timeout_s)
        if not ok:
            self._set_progress("object detection reported failure")
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class RefineDetectedObjectPoses(OrchestratorBehaviour):
    """Select matching detections and refine their poses against a point cloud."""

    def __init__(self, context: OrchestratorContext, step: ModelStep) -> None:
        """Create the selection/refinement behaviour and register candidate keys."""
        super().__init__(
            f"RefinePoses_brick{step.index}",
            context,
            Phase.PICKING,
            step.index,
        )
        self._step = step
        self.bb.register_key(key="detection_time", access=py_trees.common.Access.READ)
        self.bb.register_key(key="brick_pose_candidates", access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        """Fetch fresh detections and store probability-ranked refined poses."""
        params = self.context.params
        object_list = self.context.object_list_reader.get_fresh(
            self.bb.detection_time,
            timeout_s=params.object_list_timeout_s,
        )
        if object_list is None:
            self._set_progress("no object list published after detection")
            return py_trees.common.Status.FAILURE

        matches = [obj for obj in object_list.objects if obj.name == self._step.brick_name]
        if not matches:
            self._set_progress(
                f"no detected object matches brick '{self._step.brick_name}' "
                f"(names seen: {sorted({o.name for o in object_list.objects})})"
            )
            return py_trees.common.Status.FAILURE
        matches.sort(key=lambda obj: float(obj.probability), reverse=True)

        if not params.icp_refinement.enabled:
            candidates = []
            for obj in matches:
                pose = self._to_base_frame(obj.pose)
                if pose is not None:
                    candidates.append((pose, float(obj.probability)))
            if not candidates:
                self._set_progress("no matching pose could be transformed to the base frame")
                return py_trees.common.Status.FAILURE
            self.bb.brick_pose_candidates = candidates
            self._set_progress("ICP disabled; using detected object poses")
            return py_trees.common.Status.SUCCESS

        if self.context.icp_pose_refiner is None:
            self._set_progress("ICP pose refiner unavailable")
            return py_trees.common.Status.FAILURE
        cloud = self.context.sensor.get_point_cloud_ros(
            params.icp_refinement.point_cloud_topic,
            timeout_s=params.icp_refinement.point_cloud_timeout_s,
        )
        if cloud is None:
            self._set_progress("no point cloud received for ICP pose refinement")
            return py_trees.common.Status.FAILURE

        refinements = self.context.icp_pose_refiner.refine_candidates(
            cloud,
            [obj.pose for obj in matches],
            self._step.brick_name.split("_", 1)[0],
        )
        if not refinements:
            self._set_progress(f"ICP rejected every '{self._step.brick_name}' candidate")
            return py_trees.common.Status.FAILURE
        self.bb.brick_pose_candidates = [
            (result.pose, float(matches[result.candidate_index].probability))
            for result in refinements
        ]
        best = refinements[0]
        self._set_progress(
            f"refined {len(refinements)}/{len(matches)} candidate pose(s); "
            f"best fitness={best.fitness:.3f}, "
            f"RMSE={best.inlier_rmse_m:.4f} m"
        )
        return py_trees.common.Status.SUCCESS

    def _to_base_frame(self, pose: PoseStamped) -> PoseStamped | None:
        """Return ``pose`` in the configured base frame."""
        base_frame = self.context.params.base_frame
        source_frame = str(pose.header.frame_id).strip()
        if source_frame == base_frame:
            return pose
        if self.context.tf_buffer is None or not _TF2_GEOMETRY_AVAILABLE:
            self._log.error(
                f"object pose is in '{source_frame}' not '{base_frame}' "
                "and tf transform is unavailable"
            )
            return None
        try:
            return self.context.tf_buffer.transform(pose, base_frame)
        except Exception as exc:
            self._log.error(
                f"failed to transform object pose from '{source_frame}' to '{base_frame}': {exc}"
            )
            return None


class GenerateGrasp(OrchestratorBehaviour):
    """Generate grasp poses from probability-ranked refined brick poses."""

    def __init__(
        self,
        context: OrchestratorContext,
        step: ModelStep,
        planner: MotionPlanner,
        seed_joint_positions: dict[str, float] | None = None,
        tolerance: GoalTolerance = DEFAULT_GOAL_TOLERANCE,
    ) -> None:
        """Create grasp generation and register candidate/output pose keys.

        Parameters
        ----------
        context : OrchestratorContext
            Shared runtime context.
        step : ModelStep
            Recipe step whose brick is grasped.
        planner : MotionPlanner
            Planner for the pregrasp reachability check. Should match the
            planner of the pregrasp move.
        seed_joint_positions : dict of str to float or None, optional
            IK seed tried when no unshifted candidate is plannable from the
            current state, before any x shift. On success the pregrasp move
            becomes a joint goal (``pregrasp_joint_positions``). ``None``
            disables the seeded attempt.
        tolerance : GoalTolerance, optional
            Goal tolerance for the pregrasp reachability check. Should match
            the tolerance of the pregrasp move. Defaults to
            ``DEFAULT_GOAL_TOLERANCE``.
        """
        super().__init__(
            f"GenerateGrasp_brick{step.index}",
            context,
            Phase.PICKING,
            step.index,
        )
        self._step = step
        self._planner = planner
        self._seed_joint_positions = seed_joint_positions
        self._tolerance = tolerance
        self.bb.register_key(key="brick_pose_candidates", access=py_trees.common.Access.READ)
        for key in ("pregrasp_pose", "grasp_pose", "retreat_pose", "pregrasp_joint_positions"):
            self.bb.register_key(key=key, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        """Generate and store a reachable grasp from refined candidates."""
        for brick_pose, probability in self.bb.brick_pose_candidates:
            if self.context.tf_buffer is None:
                self._log.warning("grasp generation skipped: tf buffer unavailable")
                continue
            try:
                candidates = compute_grasp_pregrasp_candidates(
                    brick_pose,
                    self._step.brick_name.split("_")[0],
                    self.context.tf_buffer,
                )
            except RuntimeError as exc:
                self._log.warning(
                    f"grasp generation rejected candidate (prob={probability:.2f}): {exc}"
                )
                continue
            params = self.context.params
            attempts = [(0, None)]
            if self._seed_joint_positions:
                attempts.append((0, self._seed_joint_positions))
            attempts += [(shift, None) for shift in range(1, params.grasp_x_offset_steps + 1)]
            for shift, seed in attempts:
                x_offset_m = shift * params.grasp_x_offset_step_m
                if seed:
                    self._log.info("all grasp candidates failed; retrying with the IK seed")
                elif shift:
                    self._log.info(
                        f"all grasp candidates failed; retrying with x offset {x_offset_m:+.4f} m"
                    )
                for candidate in candidates:
                    if shift:
                        candidate = shift_candidate_x(candidate, x_offset_m)
                    self._log.info(
                        f"checking grasp candidate rank={candidate.rank} "
                        f"z_rotation_rad={candidate.z_rotation:+.6f} "
                        f"angular_distance_rad={candidate.angular_distance:.6f} "
                        f"x_offset_m={x_offset_m:+.4f} seeded={seed is not None}"
                    )
                    plan_ok, message, pregrasp_joints = self._check_pregrasp(candidate, seed)
                    if not plan_ok:
                        self._log.warning(
                            f"grasp candidate rank={candidate.rank} has no valid pregrasp "
                            f"plan: {message}"
                        )
                        continue

                    self._log.info(f"grasp candidate rank={candidate.rank} is plannable: {message}")
                    self.bb.pregrasp_pose = candidate.pregrasp_pose
                    self.bb.grasp_pose = candidate.grasp_pose
                    self.bb.retreat_pose = deepcopy(candidate.pregrasp_pose)
                    self.bb.pregrasp_joint_positions = pregrasp_joints
                    self._set_progress(
                        f"selected '{self._step.brick_name}' (prob={probability:.2f}, "
                        f"x_offset={x_offset_m:+.4f} m, seeded={seed is not None}); "
                        "grasp poses generated"
                    )
                    return py_trees.common.Status.SUCCESS

        self._set_progress(f"no plannable grasp for any '{self._step.brick_name}' candidate")
        return py_trees.common.Status.FAILURE

    def _check_pregrasp(
        self, candidate, seed: dict[str, float] | None
    ) -> tuple[bool, str, dict[str, float] | None]:
        """Check the pregrasp is plannable, optionally via a seeded IK solution.

        Returns
        -------
        tuple of (bool, str, dict of str to float or None)
            Success flag, status message, and the seeded pregrasp joint
            configuration (``None`` for the unseeded check).
        """
        params = self.context.params
        move_client = self.context.move_client
        if seed is None:
            ok, message = move_client.can_plan_to_pose(
                target_pose=candidate.pregrasp_pose,
                child_link=params.pose_link,
                planner=self._planner,
                max_velocity=params.fast_velocity_scaling,
                tolerance=self._tolerance,
            )
            return ok, message, None

        goal, message = move_client.solve_pose_from_seed(
            target_pose=candidate.pregrasp_pose,
            child_link=params.pose_link,
            planner=self._planner,
            seed_joint_positions=seed,
            tolerance=self._tolerance,
        )
        if goal is None:
            return False, f"no IK solution near the seed: {message}", None
        names = [name for name in params.home_joint_names if name in goal]
        ok, message = move_client.move_to_joint_goal(
            joint_names=names,
            joint_positions=[goal[name] for name in names],
            planner=self._planner,
            plan_only=True,
            max_velocity=params.fast_velocity_scaling,
        )
        if not ok:
            return False, f"seeded IK solution not reachable: {message}", None
        return True, message, {name: goal[name] for name in names}


class MovePoseFromBlackboard(OrchestratorBehaviour):
    """Plan+execute a MoveIt pose goal read from a blackboard key."""

    def __init__(
        self,
        context: OrchestratorContext,
        name: str,
        pose_key: str,
        velocity_scaling: float,
        planner: MotionPlanner,
        description: str,
        step: ModelStep | None = None,
        phase: Phase = Phase.PICKING,
        tolerance: GoalTolerance = DEFAULT_GOAL_TOLERANCE,
        joint_goal_key: str | None = None,
    ) -> None:
        """Create a pose-move behaviour bound to ``pose_key``.

        Parameters
        ----------
        context : OrchestratorContext
            Shared runtime context.
        name : str
            Behaviour name.
        pose_key : str
            Blackboard key holding the target ``PoseStamped``.
        velocity_scaling : float
            Velocity scaling factor for the move.
        planner : MotionPlanner
            Pipeline and planner for the move.
        description : str
            Human-readable progress description.
        step : ModelStep or None, optional
            Recipe step for progress reporting. ``None`` for step-less
            moves such as home. Defaults to ``None``.
        phase : Phase, optional
            Phase reported for this move. Defaults to ``Phase.PICKING``.
        tolerance : GoalTolerance, optional
            Goal tolerance. Defaults to ``DEFAULT_GOAL_TOLERANCE``.
        joint_goal_key : str or None, optional
            Blackboard key holding a ``{joint_name: position}`` dict. When the
            value is set, move there as a joint goal instead of the pose.
            Defaults to ``None``.
        """
        brick_index = step.index if step is not None else -1
        super().__init__(name, context, phase, brick_index)
        self._pose_key = pose_key
        self._velocity_scaling = float(velocity_scaling)
        self._planner = planner
        self._tolerance = tolerance
        self._joint_goal_key = joint_goal_key
        self._description = description
        self.bb.register_key(key=pose_key, access=py_trees.common.Access.READ)
        if joint_goal_key:
            self.bb.register_key(key=joint_goal_key, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        """Move to the stored pose with the configured speed/planner."""
        params = self.context.params
        self._set_progress(self._description)
        joint_goal = getattr(self.bb, self._joint_goal_key) if self._joint_goal_key else None
        if joint_goal:
            ok, message = self.context.move_client.move_to_joint_goal(
                joint_names=list(joint_goal),
                joint_positions=list(joint_goal.values()),
                planner=self._planner,
                max_velocity=self._velocity_scaling,
            )
            if not ok:
                self._set_progress(f"{self._description} failed: {message}")
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.SUCCESS

        pose: PoseStamped = getattr(self.bb, self._pose_key)
        position = (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
        orientation = (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        )
        ok, message = self.context.move_client.move_to_pose(
            position=position,
            orientation=orientation,
            parent_link=str(pose.header.frame_id) or params.base_frame,
            child_link=params.pose_link,
            planner=self._planner,
            plan_only=False,
            max_velocity=self._velocity_scaling,
            tolerance=self._tolerance,
        )
        if not ok:
            self._set_progress(f"{self._description} failed: {message}")
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class SelectPlaceIndex(OrchestratorBehaviour):
    """Choose the holder-board place index for this assembly.

    ``placement_record`` maps a one-based place index to a :class:`PlaceSlot`
    and persists across goals. Indices are checked in ascending order:

    1. The first slot holding an incomplete build of the requested model is
       resumed, so the tree continues on top of the bricks already there.
    2. Otherwise the first free index is claimed with an empty slot.

    Slots of other models, complete or partial, stay blocked. The choice is
    written to ``place_index`` (and ``context.place_index``). Fails when no
    slot can be resumed or claimed.
    """

    def __init__(self, context: OrchestratorContext) -> None:
        """Create the selector and register the placement keys."""
        super().__init__("SelectPlaceIndex", context, Phase.VALIDATING_SCENE)
        self.bb.register_key(key="placement_record", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="place_index", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="recipe", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        """Resume a partial build of this model, else claim a free index."""
        record = dict(self.bb.get("placement_record") or {})
        recipe = self.bb.recipe
        count = len(self.context.params.place_joint_positions)
        indices = range(1, count + 1)

        for index in indices:
            slot = record.get(index)
            if slot is not None and not slot.complete and slot.model_name == recipe.name:
                self._select(index)
                self._set_progress(
                    f"resuming '{recipe.name}' at place index {index} "
                    f"({len(slot.records)}/{len(recipe.steps)} bricks already placed)"
                )
                return py_trees.common.Status.SUCCESS

        for index in indices:
            if index in record:
                continue
            record[index] = PlaceSlot(model_name=recipe.name)
            self.bb.placement_record = record
            self._select(index)
            self._set_progress(f"assembling '{recipe.name}' at place index {index} of {count}")
            return py_trees.common.Status.SUCCESS

        self._set_progress(
            f"all {count} place indices (1-{count}) are occupied by other assemblies; "
            "call ~/clear_placement_record to free one"
        )
        return py_trees.common.Status.FAILURE

    def _select(self, index: int) -> None:
        """Publish the chosen place index to the context and blackboard."""
        self.context.place_index = index
        self.bb.place_index = index


class BrickAlreadyPlaced(OrchestratorBehaviour):
    """Succeed if this step was already placed at the selected place index.

    Guards each brick subtree so a resumed assembly skips bricks placed by an
    earlier, partially successful goal. Returns ``FAILURE`` otherwise so the
    guarding selector runs the brick subtree.
    """

    def __init__(self, context: OrchestratorContext, step: ModelStep) -> None:
        """Create the guard for one recipe step."""
        super().__init__(
            f"BrickAlreadyPlaced_brick{step.index}", context, Phase.VALIDATING_SCENE, step.index
        )
        self._step = step
        self.bb.register_key(key="placement_record", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        """Check the selected slot for this step."""
        slot = (self.bb.get("placement_record") or {}).get(self.context.place_index)
        if slot is not None and self._step.index in slot.placed_steps():
            self._set_progress(f"brick '{self._step.brick_name}' already placed; skipping")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class RecordPlacement(OrchestratorBehaviour):
    """Append the just-snapped brick to its slot in ``placement_record``.

    Ticked at the end of the snap subtree, so a brick only counts as placed
    once it has been released and pressed into the stack. The record carries
    the ``place_pose`` later bricks stack on.
    """

    def __init__(self, context: OrchestratorContext, step: ModelStep) -> None:
        """Create the recorder for one recipe step."""
        super().__init__(f"RecordPlacement_brick{step.index}", context, Phase.SNAPPING, step.index)
        self._step = step
        self.bb.register_key(key="placement_record", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="place_pose", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        """Record the placement of this step."""
        record = dict(self.bb.get("placement_record") or {})
        slot = record.get(self.context.place_index)
        if slot is None:
            self._set_progress(f"place index {self.context.place_index} has no slot to record into")
            return py_trees.common.Status.FAILURE
        placement = PlacementRecord(
            step_index=self._step.index,
            brick_id=self._step.brick_id,
            brick_name=self._step.brick_name,
            place_pose=deepcopy(self.bb.place_pose),
        )
        record[self.context.place_index] = replace(slot, records=(*slot.records, placement))
        self.bb.placement_record = record
        self._set_progress(f"recorded brick '{self._step.brick_name}' as placed")
        return py_trees.common.Status.SUCCESS


class ComputePlacePoses(OrchestratorBehaviour):
    """Compute preplace/place end-effector poses via FK and store them.

    The placement targets are hardcoded joint configurations for now. The
    provider turns them into ``base_frame`` poses with ``/compute_fk`` and
    this behaviour writes them to the ``preplace_pose``/``place_pose``
    blackboard keys so the standard pose-move behaviour can drive the arm.
    Fail-fast: an FK failure returns ``FAILURE``.
    """

    def __init__(self, context: OrchestratorContext, step: ModelStep) -> None:
        """Create the place-pose behaviour and register blackboard keys."""
        super().__init__(f"ComputePlacePoses_brick{step.index}", context, Phase.PLACING, step.index)
        self._step = step
        self._provider = FkPlacementProvider(context)
        for key in ("preplace_pose", "place_pose"):
            self.bb.register_key(key=key, access=py_trees.common.Access.WRITE)
        # Read this assembly's slot so subsequent bricks stack on top of the
        # last placed brick.
        self.bb.register_key(key="placement_record", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        """Compute the preplace/place poses.

        The first brick uses FK; every later brick stacks on the last place
        pose recorded in this assembly's slot. Recording happens later in
        :class:`RecordPlacement`, once the brick is snapped.
        """
        place_index = self.context.place_index
        slot = (self.bb.get("placement_record") or {}).get(place_index)
        records = slot.records if slot else ()
        prior_place_pose = records[-1].place_pose if records else None

        if prior_place_pose is None:
            self._set_progress(
                f"computing preplace/place poses for brick '{self._step.brick_name}' via FK"
            )
        else:
            self._set_progress(f"stacking brick '{self._step.brick_name}' on previous placement")

        try:
            preplace_pose, place_pose = self._provider.compute_poses(
                self._step,
                prior_place_pose=prior_place_pose,
                place_index=place_index,
            )
        except PlacementError as exc:
            self._set_progress(str(exc))
            return py_trees.common.Status.FAILURE

        self.bb.preplace_pose = preplace_pose
        self.bb.place_pose = place_pose
        return py_trees.common.Status.SUCCESS


class ComputePreSnapPose(OrchestratorBehaviour):
    """Compute the top-surface pre-snap pose for the just-placed brick."""

    def __init__(self, context: OrchestratorContext, step: ModelStep | None = None) -> None:
        """Create the pre-snap pose behaviour and register the blackboard keys."""
        super().__init__("ComputePreSnapPose", context, Phase.SNAPPING, step.index if step else -1)
        self.bb.register_key(key="place_pose", access=py_trees.common.Access.READ)
        self.bb.register_key(key="pre_snap_pose", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="snap_pose", access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        """Write a pose above the brick top surface where the closed gripper can snap from."""
        if not hasattr(self.bb, "place_pose"):
            self._set_progress("no place pose available; cannot compute pre-snap pose")
            return py_trees.common.Status.FAILURE

        pre_snap_pose = deepcopy(self.bb.place_pose)
        pre_snap_pose.pose.position.z += self.context.params.pre_snap_offset_m
        self.bb.pre_snap_pose = pre_snap_pose

        snap_pose = deepcopy(self.bb.place_pose)
        snap_pose.pose.position.z += self.context.params.snap_pose_offset_m
        self.bb.snap_pose = snap_pose

        self._set_progress(
            "computing pre-snap pose above the brick top surface to snap it into the stack"
        )
        return py_trees.common.Status.SUCCESS
