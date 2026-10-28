"""LEGO assembly task orchestrator node.

Boots by validating every critical dependency (camera stream, joint
states, blueprint lint, segmentation service, MMOR object detection
action, MoveIt planning, gripper action). Only when all checks pass does
it load the blueprint into memory and expose the ``assemble_model``
action. This enforces the "fail fast, don't waste the user's time"
principle: an assembly request never starts against a half-ready system.

Assembly itself is driven by a py_trees behaviour tree (see
``ros/behaviours``). Business logic lives in ``core/``; this node only
wires ROS plumbing, runs health checks, and ticks the tree per goal.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
from pathlib import Path
from threading import Lock
from time import sleep

import py_trees
import py_trees_ros
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from kinova_lego_assembly_interfaces.action import AssembleModel
from kinova_lego_assembly_interfaces.msg import AssemblyStatus
from kinova_lego_assembly_interfaces.srv import ClearPlacementRecord
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.exceptions import InvalidHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import Image, JointState, PointCloud2
from tf2_ros import Buffer, TransformListener

from kinova_lego_assembly.core.blueprint import BlueprintError, BlueprintUtils
from kinova_lego_assembly.core.orchestration import Phase
from kinova_lego_assembly.ros.background import SpinningNode
from kinova_lego_assembly.ros.behaviours import OrchestratorContext, OrchestratorParams, build_tree
from kinova_lego_assembly.ros.clients import (
    ComputeFkClient,
    MoveActionClient,
    ObjectDetectionClient,
    ObjectListReader,
    SegmentImageClient,
)
from kinova_lego_assembly.ros.gripper import GripperController
from kinova_lego_assembly.ros.icp_pose_refiner import (
    IcpPoseRefiner,
    IcpRefinementParams,
)
from kinova_lego_assembly.ros.pick_pose_broadcaster import PickPoseBroadcaster
from kinova_lego_assembly.ros.sensor_data_manager import SensorDataManager
from kinova_lego_assembly.ros.static_transforms import broadcast_static_transforms


def _as_xyz(value: object) -> tuple[float, float, float]:
    """Coerce a length-3 parameter value into an ``(x, y, z)`` float tuple."""
    seq = list(value)  # type: ignore[arg-type]
    if len(seq) != 3:
        raise ValueError(f"expected 3 values, got {len(seq)}: {seq!r}")
    return (float(seq[0]), float(seq[1]), float(seq[2]))


_DEFAULT_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]

_PLACE_POSE_COUNT = 4


class TaskOrchestratorNode(Node):
    """Behaviour-tree driven orchestrator for LEGO model assembly."""

    def __init__(self) -> None:
        """Declare parameters, wire adapters, health-check, and expose the action."""
        super().__init__("task_orchestrator")
        self.log = self.get_logger()
        self._declare_parameters()

        self._params = self._build_params()
        self._blueprint: BlueprintUtils | None = None
        self._healthy = False

        self._callback_group = ReentrantCallbackGroup()
        self._status_pub = self.create_publisher(AssemblyStatus, "~/assembly_status", 10)

        self._spinner = SpinningNode(
            "task_orchestrator_io",
            use_sim_time=bool(self.get_parameter("use_sim_time").value),
        )

        self._tf_buffer = None
        self._icp_pose_refiner = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        if self._params.icp_refinement.enabled:
            self._icp_pose_refiner = IcpPoseRefiner(
                self,
                self._tf_buffer,
                self._params.base_frame,
                self._params.icp_refinement,
            )

        self._move_client = MoveActionClient(
            self._spinner,
            group_name=str(self.get_parameter("orchestrator.moveit_planning_group").value),
            joint_names=[
                str(n) for n in self.get_parameter("orchestrator.moveit_joint_names").value
            ],
            max_velocity=float(
                self.get_parameter("motion_planning.max_velocity_scaling_factor").value
            ),
            max_acceleration=float(
                self.get_parameter("motion_planning.max_acceleration_scaling_factor").value
            ),
            planning_timeout_s=float(self.get_parameter("orchestrator.planning_timeout_s").value),
            execution_timeout_s=float(self.get_parameter("orchestrator.execution_timeout_s").value),
            workspace_center=_as_xyz(self.get_parameter("orchestrator.workspace_center").value),
            workspace_size=_as_xyz(self.get_parameter("orchestrator.workspace_size").value),
            workspace_frame=str(self.get_parameter("orchestrator.base_frame").value),
        )

        self._fk_client = ComputeFkClient(
            self._spinner,
            base_frame=str(self.get_parameter("orchestrator.base_frame").value),
        )

        self._sensor = SensorDataManager(self)
        self._gripper = GripperController(
            self._spinner,
            action_name=str(self.get_parameter("orchestrator.gripper_action").value),
            closed_position=float(
                self.get_parameter("orchestrator.gripper_short_axis_closed_position").value
            ),
            max_effort=float(self.get_parameter("orchestrator.gripper_max_effort").value),
            closed_tolerance_rad=float(
                self.get_parameter("orchestrator.gripper_closed_tolerance_rad").value
            ),
        )
        self._detection_client = ObjectDetectionClient(
            self._spinner, str(self.get_parameter("orchestrator.object_detection_action").value)
        )
        self._object_list_reader = ObjectListReader(
            self._spinner, str(self.get_parameter("orchestrator.object_list_topic").value)
        )
        self._segment_client = SegmentImageClient(
            self._spinner, str(self.get_parameter("orchestrator.segment_service").value)
        )

        if not self._load_blueprint():
            self.log.fatal("Blueprint load/lint failed; assemble_model action will NOT be exposed.")
            return

        self._static_tf_broadcaster = broadcast_static_transforms(self)

        if not self._run_health_check():
            self.log.fatal("Boot health check failed; assemble_model action will NOT be exposed.")
            return

        self._orch_context = self._build_context()

        if not self._validate_tree_definitions():
            self.log.fatal(
                "Behaviour-tree definition check failed; assemble_model action will NOT be exposed."
            )
            return

        self._tick_period = float(self.get_parameter("orchestrator.tree_tick_period_s").value)
        self._tree_lock = Lock()
        self._goal_active = False
        self._tree = self._build_persistent_tree()
        self._board = self.attach_blackboard()
        self._board.placement_record = {}
        # Idle heartbeat: tick the (empty) tree so py-trees-tree-viewer keeps
        # showing it live between goals. Its own callback group lets it run
        # alongside the (reentrant) action callback; the lock serialises ticks
        # against the goal's own ticking and subtree insert/prune.
        self._tree_timer = self.create_timer(
            self._tick_period,
            self._idle_tick,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self._action_server = ActionServer(
            self,
            AssembleModel,
            "assemble_model",
            execute_callback=self._execute_assemble,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._callback_group,
        )
        self._clear_record_srv = self.create_service(
            ClearPlacementRecord,
            "~/clear_placement_record",
            self._on_clear_placement_record,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        if self._tf_buffer is not None:
            self._pick_pose_broadcaster = PickPoseBroadcaster(
                self, self._tf_buffer, str(self._params.base_frame)
            )
        else:
            self.log.warning("tf2 unavailable; pick_pose visualisation disabled")

        self._healthy = True
        self.log.info("Task orchestrator healthy; assemble_model action is available.")

    # ------------------------------------------------------------------ params

    def _declare_parameters(self) -> None:
        """Declare every ROS parameter the orchestrator consumes."""
        self.declare_parameter("orchestrator.blueprint_path", "")
        self.declare_parameter("orchestrator.base_frame", "base_link")
        self.declare_parameter("orchestrator.pose_link", "end_effector_link")
        self.declare_parameter("camera.rgb", "/camera/color/image_raw")
        self.declare_parameter("orchestrator.joint_states_topic", "/joint_states")

        self.declare_parameter("orchestrator.moveit_planning_group", "manipulator")
        self.declare_parameter("orchestrator.moveit_joint_names", _DEFAULT_JOINT_NAMES)
        self.declare_parameter(
            "orchestrator.home_joint_positions", [0.0] * len(_DEFAULT_JOINT_NAMES)
        )
        for i in range(1, _PLACE_POSE_COUNT + 1):
            self.declare_parameter(
                f"orchestrator.place_poses.pose_{i}_joint_positions",
                Parameter.Type.DOUBLE_ARRAY,
            )
        self.declare_parameter("orchestrator.workspace_center", [0.0, 0.0, 0.0])
        self.declare_parameter("orchestrator.workspace_size", [2.0, 2.0, 2.0])

        self.declare_parameter("motion_planning.max_velocity_scaling_factor", 0.1)
        self.declare_parameter("motion_planning.max_acceleration_scaling_factor", 0.1)

        self.declare_parameter("orchestrator.fast_velocity_scaling", 1.0)
        self.declare_parameter("orchestrator.slow_velocity_scaling", 0.25)

        self.declare_parameter("orchestrator.planning_timeout_s", 10.0)
        self.declare_parameter("orchestrator.execution_timeout_s", 30.0)
        self.declare_parameter("orchestrator.detection_timeout_s", 30.0)
        self.declare_parameter("orchestrator.object_list_timeout_s", 10.0)
        self.declare_parameter("orchestrator.segment_timeout_s", 30.0)
        self.declare_parameter("orchestrator.grasp_timeout_s", 15.0)
        self.declare_parameter("orchestrator.image_timeout_s", 5.0)
        self.declare_parameter("orchestrator.gripper_timeout_s", 10.0)
        self.declare_parameter("orchestrator.settle_velocity_tolerance", 0.01)
        self.declare_parameter("orchestrator.settle_hold_s", 0.2)
        self.declare_parameter("orchestrator.settle_timeout_s", 3.0)
        self.declare_parameter("orchestrator.dummy_place_y_offset_m", -0.5)

        self.declare_parameter(
            "orchestrator.gripper_action", "/robotiq_gripper_controller/gripper_cmd"
        )
        self.declare_parameter("orchestrator.gripper_short_axis_closed_position", 0.8)
        self.declare_parameter("orchestrator.pre_snap_offset_m", 1.5e-2)
        self.declare_parameter("orchestrator.snap_pose_offset_m", 5e-3)
        self.declare_parameter("orchestrator.grasp_x_offset_step_m", 0.002)
        self.declare_parameter("orchestrator.grasp_x_offset_steps", 5)
        self.declare_parameter("orchestrator.gripper_max_effort", 65.0)
        self.declare_parameter("orchestrator.gripper_closed_tolerance_rad", 0.05)

        self.declare_parameter("orchestrator.object_detection_action", "/object_detection")
        self.declare_parameter("orchestrator.object_list_topic", "/world_model_node/object_list")
        self.declare_parameter("orchestrator.segment_service", "/LegoSegmentation/segment_image")

        self.declare_parameter("orchestrator.health_check_timeout_s", 15.0)
        self.declare_parameter("orchestrator.tree_tick_period_s", 0.1)

        self.declare_parameter("icp_pose_refinement.enabled", True)
        self.declare_parameter(
            "icp_pose_refinement.point_cloud_topic",
            "/camera/depth/color/points",
        )
        self.declare_parameter(
            "icp_pose_refinement.marker_topic",
            "/icp_pose_refinement/pose_markers",
        )
        self.declare_parameter("icp_pose_refinement.point_cloud_timeout_s", 5.0)
        self.declare_parameter("icp_pose_refinement.transform_timeout_s", 5.0)
        self.declare_parameter(
            "icp_pose_refinement.table_plane_z_base_frame_m",
            -0.025,
        )
        self.declare_parameter(
            "icp_pose_refinement.footprint_padding_from_seed_m",
            0.015,
        )
        self.declare_parameter(
            "icp_pose_refinement.clearance_above_table_plane_m",
            0.006,
        )
        self.declare_parameter(
            "icp_pose_refinement.cluster_neighbor_radius_m",
            0.004,
        )
        self.declare_parameter("icp_pose_refinement.cluster_min_points", 5)
        self.declare_parameter(
            "icp_pose_refinement.coarse_correspondence_max_distance_m",
            0.02,
        )
        self.declare_parameter(
            "icp_pose_refinement.fine_correspondence_max_distance_m",
            0.006,
        )
        self.declare_parameter(
            "icp_pose_refinement.max_iterations_per_stage",
            80,
        )
        self.declare_parameter("icp_pose_refinement.min_fitness", 0.25)
        self.declare_parameter(
            "icp_pose_refinement.max_seed_correction_2x2_m",
            0.015,
        )
        self.declare_parameter("icp_pose_refinement.voxel_size_m", 0.001)
        self.declare_parameter("icp_pose_refinement.model_sample_count", 30000)
        self.declare_parameter("icp_pose_refinement.marker_lifetime_s", 10.0)

    def _build_params(self) -> OrchestratorParams:
        """Assemble the behaviour parameter bundle from declared parameters."""
        get = self.get_parameter
        return OrchestratorParams(
            base_frame=str(get("orchestrator.base_frame").value),
            pose_link=str(get("orchestrator.pose_link").value),
            camera_color_topic=str(get("camera.rgb").value),
            home_joint_names=[str(n) for n in get("orchestrator.moveit_joint_names").value],
            home_joint_positions=[float(p) for p in get("orchestrator.home_joint_positions").value],
            place_joint_positions=self._load_place_joint_positions(),
            fast_velocity_scaling=float(get("orchestrator.fast_velocity_scaling").value),
            slow_velocity_scaling=float(get("orchestrator.slow_velocity_scaling").value),
            planning_timeout_s=float(get("orchestrator.planning_timeout_s").value),
            execution_timeout_s=float(get("orchestrator.execution_timeout_s").value),
            detection_timeout_s=float(get("orchestrator.detection_timeout_s").value),
            object_list_timeout_s=float(get("orchestrator.object_list_timeout_s").value),
            segment_timeout_s=float(get("orchestrator.segment_timeout_s").value),
            grasp_timeout_s=float(get("orchestrator.grasp_timeout_s").value),
            image_timeout_s=float(get("orchestrator.image_timeout_s").value),
            gripper_timeout_s=float(get("orchestrator.gripper_timeout_s").value),
            settle_velocity_tolerance=float(get("orchestrator.settle_velocity_tolerance").value),
            settle_hold_s=float(get("orchestrator.settle_hold_s").value),
            settle_timeout_s=float(get("orchestrator.settle_timeout_s").value),
            pre_snap_offset_m=float(get("orchestrator.pre_snap_offset_m").value),
            snap_pose_offset_m=float(get("orchestrator.snap_pose_offset_m").value),
            grasp_x_offset_step_m=float(get("orchestrator.grasp_x_offset_step_m").value),
            grasp_x_offset_steps=int(get("orchestrator.grasp_x_offset_steps").value),
            dummy_place_y_offset_m=float(get("orchestrator.dummy_place_y_offset_m").value),
            icp_refinement=IcpRefinementParams(
                enabled=bool(get("icp_pose_refinement.enabled").value),
                point_cloud_topic=str(get("icp_pose_refinement.point_cloud_topic").value),
                marker_topic=str(get("icp_pose_refinement.marker_topic").value),
                point_cloud_timeout_s=float(get("icp_pose_refinement.point_cloud_timeout_s").value),
                transform_timeout_s=float(get("icp_pose_refinement.transform_timeout_s").value),
                table_plane_z_base_frame_m=float(
                    get("icp_pose_refinement.table_plane_z_base_frame_m").value
                ),
                footprint_padding_from_seed_m=float(
                    get("icp_pose_refinement.footprint_padding_from_seed_m").value
                ),
                clearance_above_table_plane_m=float(
                    get("icp_pose_refinement.clearance_above_table_plane_m").value
                ),
                cluster_neighbor_radius_m=float(
                    get("icp_pose_refinement.cluster_neighbor_radius_m").value
                ),
                cluster_min_points=int(get("icp_pose_refinement.cluster_min_points").value),
                coarse_correspondence_max_distance_m=float(
                    get("icp_pose_refinement.coarse_correspondence_max_distance_m").value
                ),
                fine_correspondence_max_distance_m=float(
                    get("icp_pose_refinement.fine_correspondence_max_distance_m").value
                ),
                max_iterations_per_stage=int(
                    get("icp_pose_refinement.max_iterations_per_stage").value
                ),
                min_fitness=float(get("icp_pose_refinement.min_fitness").value),
                max_seed_correction_2x2_m=float(
                    get("icp_pose_refinement.max_seed_correction_2x2_m").value
                ),
                voxel_size_m=float(get("icp_pose_refinement.voxel_size_m").value),
                model_sample_count=int(get("icp_pose_refinement.model_sample_count").value),
                marker_lifetime_s=float(get("icp_pose_refinement.marker_lifetime_s").value),
            ),
        )

    def _load_place_joint_positions(self) -> list[list[float]]:
        """Read the place-pose joint configurations in ascending index order."""
        joint_count = len(self.get_parameter("orchestrator.moveit_joint_names").value)
        configs: list[list[float]] = []
        for i in range(1, _PLACE_POSE_COUNT + 1):
            name = f"orchestrator.place_poses.pose_{i}_joint_positions"
            values = [float(p) for p in self.get_parameter(name).value]
            if len(values) != joint_count:
                raise ValueError(f"'{name}' needs {joint_count} values, got {len(values)}")
            configs.append(values)
        return configs

    # -------------------------------------------------------------- boot setup

    def _resolve_blueprint_path(self) -> Path:
        """Resolve the blueprint path from the parameter or the installed share."""
        override = str(self.get_parameter("orchestrator.blueprint_path").value).strip()
        if override:
            return Path(override)
        try:
            share = Path(get_package_share_directory("kinova_lego_assembly"))
            return share / "assets" / "lego_blueprints.yaml"
        except PackageNotFoundError:
            return Path("assets/lego_blueprints.yaml")

    def _load_blueprint(self) -> bool:
        """Load and validate the blueprint (runs the schema lint). Returns success."""
        path = self._resolve_blueprint_path()
        try:
            self._blueprint = BlueprintUtils.from_file(path)
        except BlueprintError as exc:
            self.log.error(f"blueprint validation failed: {exc}")
            return False
        self.log.info(
            f"Blueprint loaded and linted OK from {path} "
            f"({len(self._blueprint.blocks)} blocks, {len(self._blueprint.models)} models)"
        )
        return True

    def _run_health_check(self) -> bool:
        """Verify every critical dependency; log and return whether all passed."""
        timeout = float(self.get_parameter("orchestrator.health_check_timeout_s").value)
        failures: list[str] = []

        joint_topic = str(self.get_parameter("orchestrator.joint_states_topic").value)
        ok, _ = wait_for_message(JointState, self, joint_topic, time_to_wait=timeout)
        if not ok:
            failures.append(f"no joint states on '{joint_topic}'")

        camera_topic = self._params.camera_color_topic
        ok, _ = wait_for_message(Image, self, camera_topic, time_to_wait=timeout)
        if not ok:
            failures.append(f"no camera image on '{camera_topic}'")
        if self._params.icp_refinement.enabled:
            cloud_topic = self._params.icp_refinement.point_cloud_topic
            ok, _ = wait_for_message(
                PointCloud2,
                self,
                cloud_topic,
                qos_profile=qos_profile_sensor_data,
                time_to_wait=timeout,
            )
            if not ok:
                failures.append(f"no point cloud on '{cloud_topic}'")

        if not self._segment_client.wait_for_service(timeout_s=timeout):
            failures.append("segmentation service unavailable")
        if not self._detection_client.wait_for_server(timeout_s=timeout):
            failures.append("object detection action unavailable")
        if self._tf_buffer is None:
            failures.append("grasp generation unavailable (tf buffer missing)")
        if not self._gripper.wait_for_server(timeout_s=timeout):
            failures.append("gripper action unavailable")
        if not self._move_client.wait_for_server(timeout_s=timeout):
            failures.append("MoveGroup action unavailable")

        if failures:
            for failure in failures:
                self.log.error(f"health check: {failure}")
            return False
        self.log.info("Boot health check passed: all critical dependencies are up.")
        return True

    def _build_context(self) -> OrchestratorContext:
        """Assemble the behaviour context shared by every tree behaviour."""
        return OrchestratorContext(
            node=self,
            spinner=self._spinner,
            params=self._params,
            blueprint=self._blueprint,
            move_client=self._move_client,
            fk_client=self._fk_client,
            sensor=self._sensor,
            gripper=self._gripper,
            detection_client=self._detection_client,
            object_list_reader=self._object_list_reader,
            segment_client=self._segment_client,
            tf_buffer=self._tf_buffer,
            icp_pose_refiner=self._icp_pose_refiner,
            status_pub=self._status_pub,
        )

    # ----------------------------------------------------------------- action

    def _validate_tree_definitions(self) -> bool:
        """Build+set up every model's tree at boot to fail fast on wiring bugs.

        The assembly tree is unrolled per recipe, so a malformed model or a
        behaviour-wiring mistake would otherwise only surface when that model
        is first requested. Constructing and setting up each model's tree once
        at startup brings such inconsistencies forward, consistent with the
        node's fail-fast contract. This is a structural check, so it uses the
        core (non-ROS) tree and needs no snapshot-stream plumbing.
        """
        for recipe in self._blueprint.models:
            try:
                tree = py_trees.trees.BehaviourTree(build_tree(self._orch_context, recipe))
                tree.setup(timeout=py_trees.common.Duration.INFINITE)
                tree.shutdown()
            except Exception as exc:
                self.log.fatal(
                    f"Tree definition invalid for model '{recipe.name}' (id={recipe.id}): {exc}"
                )
                return False
        count = len(self._blueprint.models)
        self.log.info(f"Behaviour-tree definitions valid for {count} model(s).")
        return True

    def _build_persistent_tree(self) -> py_trees_ros.trees.BehaviourTree:
        """Create the long-lived ROS tree the viewer stays attached to.

        The root is an empty ``assembly`` sequence; each goal inserts its
        per-recipe subtree here and prunes it on completion. Using one
        persistent ``py_trees_ros`` tree (set up once) keeps the snapshot-stream
        services alive across goals so ``py-trees-tree-viewer`` always sees the
        tree, and avoids re-declaring node parameters every goal.
        """
        root = py_trees.composites.Sequence(name="assembly", memory=True)
        # Build the tree's node ourselves rather than letting py_trees_ros call
        # rclpy.create_node("assembly_tree") internally. That internal path takes
        # no parameter overrides, so the node would default to use_sim_time=false
        # (wall clock) even in sim, mismatching the rest of the stack. Passing an
        # explicit node lets us propagate use_sim_time and pin a distinct name so
        # the snapshot stream lives under /assembly_tree (clean viewer discovery).
        use_sim_time = bool(self.get_parameter("use_sim_time").value)
        tree_node = Node(
            "assembly_tree",
            parameter_overrides=[Parameter("use_sim_time", value=use_sim_time)],
        )
        tree = py_trees_ros.trees.BehaviourTree(root)
        tree.setup(node=tree_node, timeout=py_trees.common.Duration.INFINITE)
        tree.node.set_parameters([Parameter("default_snapshot_stream", value=True)])
        return tree

    def _idle_tick(self) -> None:
        """Tick the tree only while empty; freeze a finished run for inspection.

        Between goals we keep the previous run's subtree grafted on so the
        viewer still shows its final state (including where it failed). We must
        NOT re-tick it, or memory sequences would re-run completed behaviours.
        So tick only when the tree is empty (before the first goal / after an
        explicit clear); otherwise leave the last snapshot frozen.
        """
        if self._goal_active:
            return
        with self._tree_lock:
            if self._tree.root.children:
                return
            self._tree.tick()

    def _on_goal(self, goal_request) -> GoalResponse:
        """Accept every well-formed goal; validity is resolved in execute."""
        self.log.info(f"Received assemble_model goal: target_model={goal_request.target_model!r}")
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle) -> CancelResponse:
        """Accept cancel requests (honoured between behaviour-tree ticks)."""
        return CancelResponse.ACCEPT

    def _execute_assemble(self, goal_handle) -> AssembleModel.Result:
        """Resolve the model, build+tick the tree, and report the outcome."""
        result = AssembleModel.Result()
        target = str(goal_handle.request.target_model)

        try:
            recipe = self._blueprint.resolve_model(target)
        except BlueprintError as exc:
            self.log.error(str(exc))
            result.success = False
            result.message = str(exc)
            result.bricks_completed = 0
            self._publish_status(Phase.FAILED, -1, 0, str(exc))
            goal_handle.abort()
            return result

        total = len(recipe.steps)
        self._orch_context.total_bricks = total
        self.log.info(f"Assembling model '{recipe.name}' (id={recipe.id}) with {total} brick(s).")

        board = self._reset_blackboard(recipe, total)
        # Graft this goal's per-recipe subtree onto the persistent tree so it
        # shows up in the viewer. The previous run's subtree (if any) is only
        # cleared now, at the start of a new goal, so a finished/failed tree
        # stays inspectable in the viewer until the next target is received.
        # The lock serialises graft/tick against the idle heartbeat tick.
        subtree = build_tree(self._orch_context, recipe)
        subtree.setup_with_descendants()

        self._goal_active = True
        succeeded = False
        with self._tree_lock:
            for child in list(self._tree.root.children):
                self._tree.prune_subtree(child.id)
            self._tree.insert_subtree(subtree, self._tree.root.id, 0)

        try:
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = "assembly cancelled by client"
                    result.bricks_completed = self._bricks_completed(board)
                    self._publish_status(Phase.FAILED, -1, total, result.message)
                    return result

                try:
                    with self._tree_lock:
                        self._tree.tick()
                except InvalidHandle:
                    # The node/tree is being torn down (e.g. process shutdown)
                    # while a tick is in flight: the persistent tree's own
                    # snapshot-stream publisher (internal to py_trees_ros) then
                    # raises InvalidHandle. Treat this as a clean abort instead
                    # of letting it crash the executor.
                    self.log.warning("tree tick aborted: node is shutting down")
                    result.success = False
                    result.message = "assembly aborted: node shutting down"
                    result.bricks_completed = self._bricks_completed(board)
                    # Give the goal a terminal state so the client gets an
                    # abort rather than "Goal state not set, assuming aborted".
                    # The action server may already be torn down, hence suppress.
                    with contextlib.suppress(Exception):
                        goal_handle.abort()
                    return result
                self._publish_feedback(goal_handle, board, total)
                if subtree.status != py_trees.common.Status.RUNNING:
                    break
                sleep(self._tick_period)
            succeeded = subtree.status == py_trees.common.Status.SUCCESS
            result.bricks_completed = self._bricks_completed(board)
        finally:
            self._finish_place_slot(board, succeeded)
            self._goal_active = False

        result.message = str(board.message) if board.exists("message") else ""

        if succeeded:
            result.success = True
            if not result.message:
                result.message = f"assembled '{recipe.name}' ({total} bricks)"
            self._publish_status(Phase.COMPLETE, total, total, result.message)
            self.log.info(f"Assembly complete: {result.message}")
            goal_handle.succeed()
        else:
            result.success = False
            if not result.message:
                result.message = "assembly failed"
            self._publish_status(Phase.FAILED, -1, total, result.message)
            self.log.error(f"Assembly failed: {result.message}")
            goal_handle.abort()
        return result

    # --------------------------------------------------------------- helpers

    def _reset_blackboard(self, recipe, total: int) -> py_trees.blackboard.Client:
        """Clear stale per-goal state and seed the blackboard for a fresh run.

        Every ``/assembly`` key except ``placement_record`` is unset, so the
        per-place-index slots survive across goals.
        """
        keep = "/assembly/placement_record"
        for key in list(py_trees.blackboard.Blackboard.storage):
            if key != keep:
                py_trees.blackboard.Blackboard.unset(key)
        board = self._board
        board.recipe = recipe
        board.total_bricks = total
        board.brick_index = -1
        board.phase = Phase.RESOLVING_MODEL.value
        board.message = ""
        return board

    def attach_blackboard(self) -> py_trees.blackboard.Client:
        """Create a node-side blackboard client with the keys used for reporting."""
        board = py_trees.blackboard.Client(name="orchestrator_node", namespace="assembly")
        keys = (
            "recipe",
            "total_bricks",
            "brick_index",
            "placement_record",
            "place_index",
            "phase",
            "message",
        )
        for key in keys:
            board.register_key(key=key, access=py_trees.common.Access.WRITE)
        return board

    def _on_clear_placement_record(self, request, response):
        """Free one place index (1-based) or all of them (-1)."""
        index = int(request.place_index)
        with self._tree_lock:
            if self._goal_active:
                response.success = False
                response.message = "cannot clear placement record while an assembly is running"
            elif index == -1:
                cleared = sorted(self._board.placement_record)
                self._board.placement_record = {}
                response.success = True
                response.message = (
                    f"cleared place indices {cleared}"
                    if cleared
                    else "placement record already empty"
                )
            elif not 1 <= index <= _PLACE_POSE_COUNT:
                response.success = False
                response.message = (
                    f"place_index must be 1..{_PLACE_POSE_COUNT} or -1 (all), got {index}"
                )
            else:
                record = dict(self._board.placement_record)
                removed = record.pop(index, None)
                if removed is None:
                    response.success = True
                    response.message = f"place index {index} was already free"
                else:
                    self._board.placement_record = record
                    response.success = True
                    response.message = f"cleared place index {index}"
                    if not removed.complete:
                        response.message += f" (partial '{removed.model_name}' build discarded)"
        self.log.info(f"clear_placement_record: {response.message}")
        return response

    def _finish_place_slot(self, board: py_trees.blackboard.Client, succeeded: bool) -> None:
        """Settle this goal's place slot once the goal ends.

        On success the slot is marked complete. Otherwise a slot with placed
        bricks stays as a partial build (blocked for other models, resumable
        by the same model) and an empty slot is released.
        """
        if not board.exists("place_index"):
            return
        index = int(board.place_index)
        record = dict(board.placement_record)
        slot = record.get(index)
        if slot is None:
            return
        if succeeded:
            record[index] = replace(slot, complete=True)
        elif slot.records:
            self.log.info(
                f"place index {index} keeps partial '{slot.model_name}' build "
                f"({len(slot.records)} brick(s)); request the same model to resume"
            )
            return
        else:
            del record[index]
            self.log.info(f"place index {index} released (no brick placed)")
        board.placement_record = record

    def _bricks_completed(self, board: py_trees.blackboard.Client) -> int:
        """Return how many bricks of this model are placed at its place index."""
        if not board.exists("place_index"):
            return 0
        slot = board.placement_record.get(int(board.place_index))
        return len(slot.records) if slot else 0

    def _publish_feedback(self, goal_handle, board, total: int) -> None:
        """Publish action feedback from the current blackboard progress."""
        feedback = AssembleModel.Feedback()
        feedback.phase = str(board.phase) if board.exists("phase") else Phase.IDLE.value
        feedback.brick_index = int(board.brick_index) if board.exists("brick_index") else -1
        feedback.total_bricks = int(total)
        feedback.message = str(board.message) if board.exists("message") else ""
        if not rclpy.ok():
            return
        try:
            goal_handle.publish_feedback(feedback)
        except InvalidHandle:
            self.log.warning("feedback publisher unavailable (node shutting down)")

    def _publish_status(self, phase: Phase, brick_index: int, total: int, message: str) -> None:
        """Publish a one-off status message to the ~/assembly_status topic."""
        if not rclpy.ok():
            return
        status = AssemblyStatus()
        status.phase = phase.value
        status.brick_index = int(brick_index)
        status.total_bricks = int(total)
        status.message = message
        try:
            self._status_pub.publish(status)
        except InvalidHandle:
            self.log.warning("status publisher unavailable (node shutting down)")

    def destroy_node(self) -> None:
        """Tear down the persistent tree, background spinners, and MoveIt clients."""
        tree = getattr(self, "_tree", None)
        if tree is not None:
            tree.shutdown()
            tree.node.destroy_node()
        self._spinner.shutdown()
        super().destroy_node()


def _install_shutdown_signal_logging(node: Node) -> None:
    """Log which OS signal (if any) initiated shutdown, for teardown diagnosis.

    The orchestrator has been observed tearing down the instant a goal starts,
    with process-wide shutdown already requested. To tell an external
    SIGINT/SIGTERM apart from an internal fault, chain a logging handler ahead
    of the existing (rclpy-installed) handler for these signals.
    """
    import signal

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous = signal.getsignal(sig)

        def _handler(signum, frame, _previous=previous):
            with contextlib.suppress(Exception):
                node.get_logger().warning(
                    f"received signal {signal.Signals(signum).name}; shutting down"
                )
            if callable(_previous):
                _previous(signum, frame)

        # Not on the main thread, or signal not settable: skip diagnostics.
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _handler)


_MAX_SPIN_RECOVERIES = 50


def _spin_with_recovery(node: Node, executor: MultiThreadedExecutor) -> None:
    """Spin ``executor``, surviving transient ``InvalidHandle`` faults.

    A ROS entity destroyed on one thread while an executor worker is mid-take
    surfaces as ``InvalidHandle`` on the spin thread, because
    ``MultiThreadedExecutor`` re-raises worker exceptions there. That is a
    single-entity fault, not a shutdown request, so letting it out of ``spin()``
    would tear down the whole node -- including the ``assemble_model`` action
    server -- over a race that has already resolved itself. Re-enter spin
    instead, and stop only once the context is genuinely gone.

    Parameters
    ----------
    node : rclpy.node.Node
        Node used for logging the recovery diagnostics.
    executor : rclpy.executors.MultiThreadedExecutor
        The executor to spin.
    """
    recoveries = 0
    while rclpy.ok():
        try:
            executor.spin()
            return
        except InvalidHandle:
            if not rclpy.ok():
                node.get_logger().warning("executor spin interrupted by shutdown; tearing down")
                return
            recoveries += 1
            if recoveries > _MAX_SPIN_RECOVERIES:
                node.get_logger().fatal(
                    f"executor raised InvalidHandle {recoveries} times; giving up on spinning"
                )
                return
            node.get_logger().warning(
                f"executor recovered from InvalidHandle (#{recoveries}); resuming spin"
            )


def main() -> None:
    """Entry point for the ``task_orchestrator_node`` console script."""
    rclpy.init()
    node = TaskOrchestratorNode()
    _install_shutdown_signal_logging(node)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    # The persistent tree runs on its own ROS node; add it so its snapshot-stream
    # services (used by py-trees-tree-viewer) are served by this executor.
    tree = getattr(node, "_tree", None)
    if tree is not None:
        executor.add_node(tree.node)
    try:
        _spin_with_recovery(node, executor)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the executor (and its worker threads) before destroying any node.
        # A goal's execute_callback ticks the tree on an executor worker thread;
        # destroying the node while that thread is mid-tick races entity teardown
        # against use, raising InvalidHandle. Shutting the executor down first
        # joins those workers so teardown happens with no ticks in flight.
        executor.shutdown()
        if tree is not None:
            executor.remove_node(tree.node)
        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
