"""Blocking client wrappers for the orchestrator's external dependencies.

Each endpoint wrapper hosts one ROS action/service client on a background
:class:`SpinningNode` so behaviour ``update()`` methods can make synchronous
calls without blocking the behaviour-tree executor. Clients do not compose
each other; behaviours combine them to perform a task.
"""

from __future__ import annotations

from threading import Lock
from time import monotonic, sleep
from typing import NamedTuple

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, PoseStamped
from kinova_lego_assembly_interfaces.srv import SegmentImage
from mir_interfaces.action import ObjectDetection
from mir_interfaces.msg import ObjectList
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
    WorkspaceParameters,
)
from moveit_msgs.srv import GetPositionFK
from rclpy.action import ActionClient
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_parameters,
)
from sensor_msgs.msg import Image, JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from kinova_lego_assembly.ros.background import SpinningNode

# Full z extent of the pose goal region (m). A zero-height box has no volume
# for MoveIt's goal sampler, so keep a thin sliver instead.
_GOAL_REGION_HEIGHT_M = 1e-4


class GoalTolerance(NamedTuple):
    """Pose goal tolerance used for one request.

    Attributes
    ----------
    position_m : float
        Half-width in x and y of the flat goal region (no z slack).
    orientation_rad : float
        Allowed error about each of the x, y and z axes.
    """

    position_m: float
    orientation_rad: float


DEFAULT_GOAL_TOLERANCE = GoalTolerance(position_m=0.02, orientation_rad=0.01)


class ObjectDetectionClient:
    """Client for the MMOR ``/object_detection`` action."""

    def __init__(self, spinner: SpinningNode, action_name: str = "/object_detection") -> None:
        """Create the action client on the background node."""
        self._spinner = spinner
        self._node = spinner.node
        self._action_name = action_name
        self._client = ActionClient(self._node, ObjectDetection, action_name)

    def wait_for_server(self, timeout_s: float = 5.0) -> bool:
        """Wait for the object detection action server to be available."""
        return self._client.wait_for_server(timeout_sec=float(timeout_s))

    def detect(
        self,
        obj_category: str = "",
        obj_name: str = "",
        workstation: str = "",
        timeout_s: float = 30.0,
    ) -> bool:
        """Trigger a detection and return whether it reported success.

        Returns
        -------
        bool
            The action result's ``result`` flag, or ``False`` on server
            unavailability, rejection, or timeout.
        """
        if not self._client.wait_for_server(timeout_sec=3.0):
            self._node.get_logger().error(
                f"Object detection action '{self._action_name}' is not available"
            )
            return False

        goal = ObjectDetection.Goal()
        goal.obj_category = str(obj_category)
        goal.obj_name = str(obj_name)
        goal.workstation = str(workstation)

        send_future = self._client.send_goal_async(goal)
        if not self._spinner.wait_for_future(send_future, 5.0):
            self._node.get_logger().error("Timed out sending object detection goal")
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._node.get_logger().error("Object detection goal was rejected")
            return False

        result_future = goal_handle.get_result_async()
        if not self._spinner.wait_for_future(result_future, timeout_s):
            self._node.get_logger().error(f"Object detection timed out after {timeout_s:.1f}s")
            return False

        wrapped = result_future.result()
        if wrapped is None:
            return False
        return bool(wrapped.result.result)


class ObjectListReader:
    """Subscribes to the world-model object list and serves fresh snapshots."""

    def __init__(
        self,
        spinner: SpinningNode,
        topic: str = "/world_model_node/object_list",
    ) -> None:
        """Create a subscription on the background node."""
        self._node = spinner.node
        self._topic = topic
        self._latest: ObjectList | None = None
        self._latest_time: float = 0.0
        self._sub = self._node.create_subscription(
            ObjectList,
            topic,
            self._on_object_list,
            qos_profile_parameters,
        )

    def _on_object_list(self, msg: ObjectList) -> None:
        """Cache the newest object list with its receive time."""
        self._latest = msg
        self._latest_time = monotonic()

    def get_fresh(self, after_time: float, timeout_s: float = 10.0) -> ObjectList | None:
        """Return the first object list received strictly after ``after_time``.

        Parameters
        ----------
        after_time : float
            A ``time.monotonic()`` timestamp; only a message received
            after this is accepted (ensures the list reflects the latest
            detection, not a stale publication).
        timeout_s : float, optional
            Maximum time to wait for a fresh message. Defaults to ``10.0``.

        Returns
        -------
        mir_interfaces.msg.ObjectList or None
            The fresh object list, or ``None`` on timeout.
        """
        deadline = monotonic() + max(0.1, float(timeout_s))
        while monotonic() < deadline:
            if self._latest is not None and self._latest_time > after_time:
                return self._latest
            sleep(0.02)
        self._node.get_logger().error(
            f"Timed out after {timeout_s:.1f}s waiting for a fresh object list on '{self._topic}'"
        )
        return None


class SegmentImageClient:
    """Client for the ``/segment_image`` segmentation service."""

    def __init__(self, spinner: SpinningNode, service_name: str) -> None:
        """Create the service client on the background node."""
        self._spinner = spinner
        self._node = spinner.node
        self._service_name = service_name
        self._client = self._node.create_client(SegmentImage, service_name)

    def wait_for_service(self, timeout_s: float = 5.0) -> bool:
        """Wait for the segmentation service to be available."""
        return self._client.wait_for_service(timeout_sec=float(timeout_s))

    def segment(self, image: Image, timeout_s: float = 30.0) -> SegmentImage.Response | None:
        """Run segmentation on ``image`` and return the response.

        Returns
        -------
        kinova_lego_assembly_interfaces.srv.SegmentImage.Response or None
            The response, or ``None`` on service unavailability or timeout.
        """
        if not self._client.wait_for_service(timeout_sec=3.0):
            self._node.get_logger().error(
                f"Segmentation service '{self._service_name}' is not available"
            )
            return None

        request = SegmentImage.Request()
        request.image = image
        future = self._client.call_async(request)
        if not self._spinner.wait_for_future(future, timeout_s):
            if self._spinner.active:
                self._node.get_logger().error(
                    f"Segmentation service timed out after {timeout_s:.1f}s"
                )
            else:
                self._node.get_logger().warning(
                    "Segmentation wait aborted before the deadline: orchestrator is shutting down"
                )
            return None
        return future.result()


def _moveit_error_to_message(error_code: int) -> str:
    """Map raw MoveIt error codes to user-facing failure messages."""
    if error_code == MoveItErrorCodes.NO_IK_SOLUTION:
        return "Unable to find IK solution"
    if error_code == MoveItErrorCodes.PLANNING_FAILED:
        return "Planning failed"
    if error_code == MoveItErrorCodes.INVALID_MOTION_PLAN:
        return "Invalid motion plan"
    if error_code == MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE:
        return "Motion plan invalidated by environment change"
    return f"MoveGroup failed with error code {error_code}"


class MotionPlanner(NamedTuple):
    """MoveIt planning pipeline and planner used for one request.

    Attributes
    ----------
    pipeline_id : str
        Planning pipeline loaded by ``move_group`` (``request.pipeline_id``).
    planner_id : str
        Planner within that pipeline (``request.planner_id``).
    """

    pipeline_id: str
    planner_id: str


class MoveActionClient:
    """Client for MoveIt's ``/move_action`` MoveGroup action server.

    Owns the planning configuration and ``MotionPlanRequest`` construction
    for pose and joint goals sent to this one action server. It is planner
    agnostic: every goal names its :class:`MotionPlanner`.
    """

    def __init__(
        self,
        spinner: SpinningNode,
        group_name: str,
        joint_names: list[str],
        max_velocity: float = 0.1,
        max_acceleration: float = 0.1,
        planning_timeout_s: float = 10.0,
        execution_timeout_s: float = 30.0,
        workspace_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        workspace_size: tuple[float, float, float] = (2.0, 2.0, 2.0),
        workspace_frame: str = "base_link",
        action_name: str = "/move_action",
    ) -> None:
        """Create motion planning helpers and clients on the background node.

        Parameters
        ----------
        spinner : SpinningNode
            Background node that hosts the action client and joint-state
            subscription so calls can block without stalling the executor.
        group_name : str
            Planning group to plan for (``request.group_name``).
        joint_names : list of str
            Names of the joints in the planning group; used to detect when a
            complete joint state has been received for the start state.
        max_velocity : float, optional
            Default velocity scaling factor, overridable per call. Defaults
            to ``0.1``.
        max_acceleration : float, optional
            Default acceleration scaling factor. Defaults to ``0.1``.
        planning_timeout_s : float, optional
            Allowed planning time and the wait budget for planning results.
            Defaults to ``10.0``.
        execution_timeout_s : float, optional
            Extra wait budget added on top of planning time for executed
            (non ``plan_only``) goals. Defaults to ``30.0``.
        workspace_center : tuple of float, optional
            Centre ``(x, y, z)`` of the planning workspace box, in the goal
            frame. Defaults to ``(0.0, 0.0, 0.0)``.
        workspace_size : tuple of float, optional
            Full extent ``(x, y, z)`` of the planning workspace box. Combined
            with ``workspace_center`` to derive the min/max corners passed as
            ``WorkspaceParameters`` on every request. Defaults to
            ``(2.0, 2.0, 2.0)``.
        workspace_frame : str, optional
            Frame the workspace box is expressed in, used as the header frame
            for the debug workspace marker published on ``/planning_workspace``
            so it can be visualised in RViz. Should match the frame pose goals
            are planned in. Defaults to ``"base_link"``.
        action_name : str, optional
            MoveGroup action name. Defaults to ``"/move_action"``.
        """
        self._spinner = spinner
        self._node = spinner.node
        self._action_name = action_name
        self._group_name = group_name
        self._joint_names = list(joint_names)
        self._max_velocity = float(max_velocity)
        self._max_acceleration = float(max_acceleration)
        self._planning_timeout_s = float(planning_timeout_s)
        self._execution_timeout_s = float(execution_timeout_s)

        # Convert centre + full size into the min/max corners that
        # WorkspaceParameters expects (min = centre - size/2, max = centre + size/2).
        center = tuple(float(v) for v in workspace_center)
        half = tuple(float(v) / 2.0 for v in workspace_size)
        self._workspace_min = (
            center[0] - half[0],
            center[1] - half[1],
            center[2] - half[2],
        )
        self._workspace_max = (
            center[0] + half[0],
            center[1] + half[1],
            center[2] + half[2],
        )
        self._workspace_frame = str(workspace_frame)

        self._client = ActionClient(
            node=self._node,
            action_type=MoveGroup,
            action_name=action_name,
        )

        # Debug visualisation of the planning workspace box. Latched
        # (transient-local, depth 1) so an RViz Marker display started after
        # the node still receives the last marker without a periodic republish.
        self._workspace_marker_pub = self._node.create_publisher(
            Marker,
            "/planning_workspace",
            QoSProfile(
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self._publish_workspace_marker()

        self._joint_state_lock = Lock()
        self._latest_joint_state: JointState | None = None
        self._joint_state_sub = self._node.create_subscription(
            JointState,
            "joint_states",
            self._on_joint_state,
            QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )

    def wait_for_server(self, timeout_s: float = 5.0) -> bool:
        """Wait for the MoveGroup action server to be available."""
        return self._client.wait_for_server(timeout_sec=float(timeout_s))

    def can_plan_to_pose(
        self,
        target_pose: PoseStamped,
        child_link: str,
        planner: MotionPlanner,
        max_velocity: float | None = None,
        tolerance: GoalTolerance = DEFAULT_GOAL_TOLERANCE,
    ) -> tuple[bool, str]:
        """Check whether MoveIt can plan to ``target_pose`` without executing.

        Parameters
        ----------
        target_pose : PoseStamped
            Goal pose of ``child_link``.
        child_link : str
            The link that must reach the goal.
        planner : MotionPlanner
            Pipeline and planner to plan with. Should match the move this
            check gates.
        max_velocity : float, optional
            Override the configured velocity scaling factor for this call.
        tolerance : GoalTolerance, optional
            Goal tolerance. Defaults to ``DEFAULT_GOAL_TOLERANCE``.

        Returns
        -------
        tuple of (bool, str)
            Success flag and a human-readable status message.
        """
        position = (
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z,
        )
        orientation = (
            target_pose.pose.orientation.x,
            target_pose.pose.orientation.y,
            target_pose.pose.orientation.z,
            target_pose.pose.orientation.w,
        )
        constraints = self._build_pose_constraints(
            position,
            orientation,
            target_pose.header.frame_id,
            child_link,
            tolerance=tolerance,
        )
        motion_request = self._build_motion_plan_request(
            [constraints],
            frame_id=target_pose.header.frame_id,
            planner=planner,
            max_velocity=max_velocity,
            max_acceleration=1.0,
        )
        return self._call_move_action(motion_request, plan_only=True)

    def move_to_pose(
        self,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
        parent_link: str,
        child_link: str,
        planner: MotionPlanner,
        plan_only: bool = False,
        max_velocity: float | None = None,
        tolerance: GoalTolerance = DEFAULT_GOAL_TOLERANCE,
    ) -> tuple[bool, str]:
        """Plan (and optionally execute) a pose goal.

        Parameters
        ----------
        position : tuple of float
            Target position ``(x, y, z)`` in ``parent_link``.
        orientation : tuple of float
            Target orientation quaternion ``(x, y, z, w)`` in ``parent_link``.
        parent_link : str
            Reference frame for the goal (``header.frame_id``).
        child_link : str
            The link that must reach the goal (``link_name``).
        planner : MotionPlanner
            Pipeline and planner to plan with.
        plan_only : bool, optional
            If ``True``, only plan; do not execute. Defaults to ``False``.
        max_velocity : float, optional
            Override the configured velocity scaling factor for this call.
        tolerance : GoalTolerance, optional
            Goal tolerance. Defaults to ``DEFAULT_GOAL_TOLERANCE``.

        Returns
        -------
        tuple of (bool, str)
            Success flag and a human-readable status message.
        """
        constraints = self._build_pose_constraints(
            position,
            orientation,
            parent_link,
            child_link,
            tolerance=tolerance,
        )
        motion_request = self._build_motion_plan_request(
            [constraints],
            frame_id=parent_link,
            planner=planner,
            max_velocity=max_velocity,
            max_acceleration=None,
        )
        return self._call_move_action(motion_request, plan_only)

    def move_to_joint_goal(
        self,
        joint_names: list[str],
        joint_positions: list[float],
        planner: MotionPlanner,
        plan_only: bool = False,
        max_velocity: float | None = None,
        joint_tolerance_rad: float = 0.001,
    ) -> tuple[bool, str]:
        """Plan (and optionally execute) a joint-space goal.

        Parameters
        ----------
        joint_names : list of str
            Names of the joints to command.
        joint_positions : list of float
            Target positions, aligned with ``joint_names``.
        planner : MotionPlanner
            Pipeline and planner to plan with.
        plan_only : bool, optional
            If ``True``, only plan; do not execute. Defaults to ``False``.
        max_velocity : float, optional
            Override the configured velocity scaling factor for this call.
        joint_tolerance_rad : float, optional
            Symmetric position tolerance per joint. Defaults to ``0.001``.

        Returns
        -------
        tuple of (bool, str)
            Success flag and a human-readable status message.
        """
        if len(joint_names) != len(joint_positions):
            return (
                False,
                "joint_names and joint_positions length mismatch "
                f"({len(joint_names)} != {len(joint_positions)})",
            )
        if not joint_names:
            return False, "joint goal has no joints"

        constraints = self._build_joint_constraints(
            joint_names, joint_positions, joint_tolerance_rad
        )
        motion_request = self._build_motion_plan_request(
            [constraints],
            frame_id="",
            planner=planner,
            max_velocity=max_velocity,
            max_acceleration=None,
        )
        return self._call_move_action(motion_request, plan_only)

    def solve_pose_from_seed(
        self,
        target_pose: PoseStamped,
        child_link: str,
        planner: MotionPlanner,
        seed_joint_positions: dict[str, float],
        tolerance: GoalTolerance = DEFAULT_GOAL_TOLERANCE,
    ) -> tuple[dict[str, float] | None, str]:
        """Find a joint configuration reaching ``target_pose`` near a seed.

        Plans (plan-only) from ``seed_joint_positions`` instead of the current
        state. Planners that seed IK from the start state (Pilz) then return a
        goal configuration close to the seed. The trajectory itself starts at
        the seed and is not executable; use the returned configuration as a
        joint goal from the current state.

        Parameters
        ----------
        target_pose : PoseStamped
            Goal pose of ``child_link``.
        child_link : str
            The link that must reach the goal.
        planner : MotionPlanner
            Pipeline and planner to plan with.
        seed_joint_positions : dict of str to float
            Joint positions overriding the current state in the start state.
        tolerance : GoalTolerance, optional
            Goal tolerance. Defaults to ``DEFAULT_GOAL_TOLERANCE``.

        Returns
        -------
        tuple of (dict of str to float or None, str)
            Goal joint positions by name (``None`` on failure) and a status
            message.
        """
        position = (
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z,
        )
        orientation = (
            target_pose.pose.orientation.x,
            target_pose.pose.orientation.y,
            target_pose.pose.orientation.z,
            target_pose.pose.orientation.w,
        )
        constraints = self._build_pose_constraints(
            position,
            orientation,
            target_pose.header.frame_id,
            child_link,
            tolerance=tolerance,
        )
        motion_request = self._build_motion_plan_request(
            [constraints],
            frame_id=target_pose.header.frame_id,
            planner=planner,
            max_velocity=None,
            max_acceleration=None,
            seed_joint_positions=seed_joint_positions,
        )
        ok, message, result = self._send_move_goal(motion_request, plan_only=True)
        if not ok:
            return None, message
        trajectory = result.planned_trajectory.joint_trajectory
        if not trajectory.points:
            return None, "seeded plan returned an empty trajectory"
        goal = dict(zip(trajectory.joint_names, trajectory.points[-1].positions, strict=True))
        return goal, message

    def _call_move_action(
        self,
        motion_request: MotionPlanRequest,
        plan_only: bool,
    ) -> tuple[bool, str]:
        """Send a MoveGroup goal and decode the planning/execution outcome."""
        ok, message, _ = self._send_move_goal(motion_request, plan_only)
        return ok, message

    def _send_move_goal(
        self,
        motion_request: MotionPlanRequest,
        plan_only: bool,
    ) -> tuple[bool, str, MoveGroup.Result | None]:
        """Send a MoveGroup goal; also return the raw result on success."""
        if not self._client.wait_for_server(timeout_sec=3.0):
            message = f"MoveGroup action '{self._action_name}' is not available"
            self._node.get_logger().error(message)
            return False, message, None

        goal = MoveGroup.Goal()
        goal.request = motion_request
        goal.planning_options.plan_only = bool(plan_only)

        send_future = self._client.send_goal_async(goal)
        if not self._spinner.wait_for_future(send_future, 5.0):
            message = "Timed out sending MoveGroup goal"
            self._node.get_logger().error(message)
            return False, message, None

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            message = "MoveGroup goal was rejected"
            self._node.get_logger().error(message)
            return False, message, None

        # Planning-only goals need the planning budget; executed goals may
        # additionally run the trajectory on the controller.
        timeout_s = self._planning_timeout_s
        if not plan_only:
            timeout_s += max(0.0, self._execution_timeout_s)

        result_future = goal_handle.get_result_async()
        if not self._spinner.wait_for_future(result_future, timeout_s):
            message = f"MoveGroup goal timed out after {timeout_s:.1f}s"
            self._node.get_logger().error(message)
            return False, message, None

        wrapped = result_future.result()
        if wrapped is None:
            return False, "MoveGroup returned no result", None

        error_code = int(wrapped.result.error_code.val)
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED or error_code != MoveItErrorCodes.SUCCESS:
            self._node.get_logger().warning(
                f"MoveGroup failed with raw MoveIt error code: {error_code}"
            )
            return False, _moveit_error_to_message(error_code), None

        action = "planned" if plan_only else "planned and executed"
        return True, f"MoveGroup goal {action} successfully", wrapped.result

    def wait_until_settled(
        self,
        velocity_tolerance: float = 0.01,
        hold_s: float = 0.2,
        timeout_s: float = 3.0,
    ) -> bool:
        """Block until every planning joint stays below a velocity bound.

        Parameters
        ----------
        velocity_tolerance : float, optional
            Maximum absolute joint velocity (rad/s) counted as stopped.
            Defaults to ``0.01``.
        hold_s : float, optional
            How long all joints must stay below the bound. Defaults to
            ``0.2``.
        timeout_s : float, optional
            Give up after this long. Defaults to ``3.0``.

        Returns
        -------
        bool
            ``True`` once the arm has been still for ``hold_s``, ``False``
            on timeout or if no joint state with velocities is available.
        """
        deadline = monotonic() + max(0.0, float(timeout_s))
        still_since: float | None = None
        while monotonic() < deadline:
            with self._joint_state_lock:
                state = self._latest_joint_state
            if state is not None and len(state.velocity) == len(state.name):
                velocities = dict(zip(state.name, state.velocity, strict=False))
                peak = max(abs(velocities[name]) for name in self._joint_names)
                now = monotonic()
                if peak <= velocity_tolerance:
                    if still_since is None:
                        still_since = now
                    if now - still_since >= hold_s:
                        return True
                else:
                    still_since = None
            sleep(0.01)
        return False

    def _on_joint_state(self, msg: JointState) -> None:
        """Cache the latest joint state if it contains all planning joints."""
        for joint_name in self._joint_names:
            if joint_name not in msg.name:
                return
        with self._joint_state_lock:
            self._latest_joint_state = msg

    def _build_start_state(
        self, seed_joint_positions: dict[str, float] | None = None
    ) -> RobotState:
        """Build a complete, zero-velocity start state from latest joint states.

        move_group needs the full robot configuration (all joints) or it
        rejects the request with ``INVALID_ROBOT_STATE``. Pilz PTP/LIN also
        reject any start state with non-zero joint velocity, and the simulator
        leaves small residual velocities after a motion, so every velocity is
        forced to zero. ``seed_joint_positions`` overrides the named joints.
        """
        if seed_joint_positions:
            state = self._build_start_state()
            if state.is_diff:
                state.joint_state.name = list(seed_joint_positions)
                state.joint_state.position = [float(v) for v in seed_joint_positions.values()]
                state.joint_state.velocity = [0.0] * len(seed_joint_positions)
                return state
            names = list(state.joint_state.name)
            positions = list(state.joint_state.position)
            for name, value in seed_joint_positions.items():
                if name in names:
                    positions[names.index(name)] = float(value)
                else:
                    names.append(name)
                    positions.append(float(value))
            state.joint_state.name = names
            state.joint_state.position = positions
            state.joint_state.velocity = [0.0] * len(names)
            return state

        state = RobotState()
        with self._joint_state_lock:
            current = self._latest_joint_state
        if current is None or not current.name:
            self._node.get_logger().warning(
                "No joint states received yet; planning as a diff from the "
                "current planning-scene state."
            )
            state.is_diff = True
            return state

        joint_state = JointState()
        joint_state.header = current.header
        joint_state.name = list(current.name)
        joint_state.position = list(current.position)
        joint_state.velocity = [0.0] * len(current.name)
        joint_state.effort = []
        state.joint_state = joint_state
        state.is_diff = False
        return state

    def _build_pose_constraints(
        self,
        target_position: tuple[float, float, float],
        target_orientation: tuple[float, float, float, float],
        parent_link: str,
        child_link: str,
        tolerance: GoalTolerance = DEFAULT_GOAL_TOLERANCE,
    ) -> Constraints:
        """Build position and orientation goal constraints to reach a pose.

        Parameters
        ----------
        target_position : tuple of float
            Target (x, y, z) of ``child_link`` in ``parent_link``.
        target_orientation : tuple of float
            Target quaternion (x, y, z, w) of ``child_link`` in ``parent_link``.
        parent_link : str
            Frame in which the target pose is expressed.
        child_link : str
            Link that must reach the target pose.
        tolerance : GoalTolerance, optional
            ``position_m`` is the half-width in x and y of the flat box around
            ``target_position`` the link origin must end in. The box height is
            fixed at ``_GOAL_REGION_HEIGHT_M`` so no z offset is accepted.
            ``orientation_rad`` is the allowed error about each axis.

        Returns
        -------
        Constraints
            One position and one orientation constraint.
        """
        # Position: the origin of child_link must end inside a flat box around the target.
        tolerance_region = SolidPrimitive()
        tolerance_region.type = SolidPrimitive.BOX
        tolerance_region.dimensions = [
            2.0 * float(tolerance.position_m),
            2.0 * float(tolerance.position_m),
            _GOAL_REGION_HEIGHT_M,
        ]

        region_pose = Pose()
        region_pose.position.x, region_pose.position.y, region_pose.position.z = (
            float(value) for value in target_position
        )
        region_pose.orientation.w = 1.0

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = parent_link
        position_constraint.link_name = child_link
        # primitives[i] is placed at primitive_poses[i].
        position_constraint.constraint_region.primitives.append(tolerance_region)
        position_constraint.constraint_region.primitive_poses.append(region_pose)
        position_constraint.weight = 1.0

        # Orientation: child_link must match the target quaternion within tolerance per axis.
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = parent_link
        orientation_constraint.link_name = child_link
        (
            orientation_constraint.orientation.x,
            orientation_constraint.orientation.y,
            orientation_constraint.orientation.z,
            orientation_constraint.orientation.w,
        ) = (float(value) for value in target_orientation)
        orientation_constraint.absolute_x_axis_tolerance = tolerance.orientation_rad
        orientation_constraint.absolute_y_axis_tolerance = tolerance.orientation_rad
        orientation_constraint.absolute_z_axis_tolerance = tolerance.orientation_rad
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)
        return constraints

    @staticmethod
    def _build_joint_constraints(
        joint_names: list[str],
        joint_positions: list[float],
        joint_tolerance_rad: float,
    ) -> Constraints:
        """Build joint goal constraints from a target joint configuration."""
        constraints = Constraints()
        for name, position in zip(joint_names, joint_positions, strict=False):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = str(name)
            joint_constraint.position = float(position)
            joint_constraint.tolerance_above = float(joint_tolerance_rad)
            joint_constraint.tolerance_below = float(joint_tolerance_rad)
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        return constraints

    def _publish_workspace_marker(self) -> None:
        """Publish a wireframe box marker of the planning workspace for RViz.

        Draws the same min/max corner box that is sent as
        ``WorkspaceParameters`` on every request, as a ``LINE_LIST`` marker on
        ``/planning_workspace`` (latched). This lets the box be inspected in
        RViz via a Marker display, which the MotionPlanning panel's own "Show
        Workspace" cannot do (that renders RViz's own local workspace, not the
        one the action client sends).
        """
        min_x, min_y, min_z = self._workspace_min
        max_x, max_y, max_z = self._workspace_max

        # The 8 corners of the box.
        corners = [
            (min_x, min_y, min_z),
            (max_x, min_y, min_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (min_x, min_y, max_z),
            (max_x, min_y, max_z),
            (max_x, max_y, max_z),
            (min_x, max_y, max_z),
        ]
        # 12 edges as index pairs into ``corners`` (bottom, top, verticals).
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        marker = Marker()
        marker.header.frame_id = self._workspace_frame
        marker.header.stamp = self._node.get_clock().now().to_msg()
        marker.ns = "planning_workspace"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.005  # line width (m)
        marker.color = ColorRGBA(r=0.1, g=0.8, b=1.0, a=1.0)
        marker.pose.orientation.w = 1.0
        for start, end in edges:
            marker.points.append(
                Point(x=corners[start][0], y=corners[start][1], z=corners[start][2])
            )
            marker.points.append(Point(x=corners[end][0], y=corners[end][1], z=corners[end][2]))

        self._workspace_marker_pub.publish(marker)
        self._node.get_logger().info(
            f"Published planning workspace marker in '{self._workspace_frame}': "
            f"min={self._workspace_min}, max={self._workspace_max}"
        )

    def _build_motion_plan_request(
        self,
        goal_constraints: list[Constraints],
        frame_id: str,
        planner: MotionPlanner,
        max_velocity: float | None,
        max_acceleration: float | None,
        seed_joint_positions: dict[str, float] | None = None,
    ) -> MotionPlanRequest:
        """Assemble a MotionPlanRequest shared by pose and joint goals.

        ``seed_joint_positions`` replaces those joints in the start state, so
        planners that seed IK from the start state search near it.
        """
        motion_request = MotionPlanRequest()

        workspace = WorkspaceParameters()
        workspace.header.frame_id = frame_id
        workspace.header.stamp = self._node.get_clock().now().to_msg()
        workspace.min_corner.x = self._workspace_min[0]
        workspace.min_corner.y = self._workspace_min[1]
        workspace.min_corner.z = self._workspace_min[2]
        workspace.max_corner.x = self._workspace_max[0]
        workspace.max_corner.y = self._workspace_max[1]
        workspace.max_corner.z = self._workspace_max[2]
        motion_request.workspace_parameters = workspace

        motion_request.start_state = self._build_start_state(seed_joint_positions)
        motion_request.goal_constraints = list(goal_constraints)
        motion_request.pipeline_id = planner.pipeline_id
        motion_request.planner_id = planner.planner_id
        motion_request.group_name = self._group_name
        motion_request.num_planning_attempts = 5
        motion_request.allowed_planning_time = max(0.1, self._planning_timeout_s)
        motion_request.max_velocity_scaling_factor = float(
            self._max_velocity if max_velocity is None else max_velocity
        )
        motion_request.max_acceleration_scaling_factor = float(
            self._max_acceleration if max_acceleration is None else max_acceleration
        )
        return motion_request


class ComputeFkClient:
    """Client for MoveIt's ``/compute_fk`` (``moveit_msgs/srv/GetPositionFK``).

    Turns a joint configuration into the Cartesian pose of a requested link
    via forward kinematics. Used to hardcode preplace/place/home targets as
    joint angles while still driving the arm through pose-based moves.
    """

    def __init__(
        self,
        spinner: SpinningNode,
        base_frame: str = "base_link",
        service_name: str = "/compute_fk",
    ) -> None:
        """Create the GetPositionFK service client on the background node.

        Parameters
        ----------
        spinner : SpinningNode
            Background node hosting the service client so calls can block
            without stalling the executor.
        base_frame : str, optional
            Frame the returned pose is expressed in (request header frame).
            Defaults to ``"base_link"``.
        service_name : str, optional
            FK service name. Defaults to ``"/compute_fk"``.
        """
        self._spinner = spinner
        self._node = spinner.node
        self._base_frame = base_frame
        self._service_name = service_name
        self._client = self._node.create_client(GetPositionFK, service_name)

    def wait_for_service(self, timeout_s: float = 5.0) -> bool:
        """Wait for the FK service to be available."""
        return self._client.wait_for_service(timeout_sec=float(timeout_s))

    def compute_pose_from_joints(
        self,
        joint_names: list[str],
        joint_positions: list[float],
        fk_link: str,
        timeout_s: float = 5.0,
    ) -> PoseStamped | None:
        """Compute the pose of ``fk_link`` for the given joint configuration.

        Parameters
        ----------
        joint_names : list of str
            Names of the joints to set in the query robot state.
        joint_positions : list of float
            Target positions, aligned with ``joint_names``.
        fk_link : str
            Link whose pose is requested (e.g. ``end_effector_link``).
        timeout_s : float, optional
            Wait budget for service availability and result. Defaults to
            ``5.0``.

        Returns
        -------
        geometry_msgs.msg.PoseStamped or None
            The link pose in ``base_frame``, or ``None`` on failure.
        """
        if len(joint_names) != len(joint_positions):
            self._node.get_logger().error(
                "compute_fk: joint_names and joint_positions length mismatch "
                f"({len(joint_names)} != {len(joint_positions)})"
            )
            return None
        if not joint_names:
            self._node.get_logger().error("compute_fk: joint goal has no joints")
            return None

        if not self._client.wait_for_service(timeout_sec=float(timeout_s)):
            self._node.get_logger().error(f"FK service '{self._service_name}' is not available")
            return None

        request = GetPositionFK.Request()
        request.header.frame_id = self._base_frame
        request.header.stamp = self._node.get_clock().now().to_msg()
        request.fk_link_names = [str(fk_link)]

        robot_state = RobotState()
        robot_state.joint_state.name = [str(n) for n in joint_names]
        robot_state.joint_state.position = [float(p) for p in joint_positions]
        robot_state.is_diff = False
        request.robot_state = robot_state

        future = self._client.call_async(request)
        if not self._spinner.wait_for_future(future, timeout_s):
            self._node.get_logger().error(f"FK service timed out after {timeout_s:.1f}s")
            return None

        response = future.result()
        if response is None:
            self._node.get_logger().error("FK service returned no response")
            return None
        if int(response.error_code.val) != MoveItErrorCodes.SUCCESS:
            self._node.get_logger().error(
                f"FK failed with MoveIt error code {int(response.error_code.val)}"
            )
            return None
        if not response.pose_stamped:
            self._node.get_logger().error("FK returned no pose for the requested link")
            return None
        return response.pose_stamped[0]
