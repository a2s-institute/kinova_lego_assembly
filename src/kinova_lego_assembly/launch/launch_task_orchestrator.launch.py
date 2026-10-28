"""Single-entry launch for the full LEGO assembly stack.

Startup order is enforced so the task orchestrator (which fail-fast
health-checks every dependency at boot) comes up last:

1. Simulation + MoveIt stack (optional, gated on ``sim_mode:=true``),
   via ``launch_sim_moveit.launch.py`` which sequences MuJoCo sim then
   move_group + RViz internally.
2. Segmentation service node.
3. MMOR object recognition, brought up once segmentation is running.
   MMOR is a managed (lifecycle) node exposed as a ComposableNode, and it
   depends on the segmentation node, so after segmentation starts we
   launch the MMOR container and then drive it configure -> activate.
   Only an active MMOR exposes ``/object_detection`` and publishes the
   object list.
4. Task orchestrator, started in the same burst; its bounded health-check
   waits gate on ``/object_detection`` (i.e. an active MMOR) and the rest
   of the stack, absorbing the remaining startup races.

MMOR is a ComposableNode, not a launch-managed LifecycleNode, so launch
cannot observe its transitions and emitted ChangeState events are not
reliably delivered. Instead we run a ``lifecycle_manager_node`` that calls
MMOR's ``change_state`` service synchronously, so configure blocks until
``on_configure`` (e.g. YOLO weight loading) finishes before activate is
requested. It also drives deactivate -> cleanup on launch shutdown. No
transition-timing guesswork.

Usage:
    ros2 launch kinova_lego_assembly launch_task_orchestrator.launch.py
"""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

from kinova_mujoco_sim.core.scene import layout
from kinova_mujoco_sim.ros import launch_helpers as helpers
from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    """Compose the ordered assembly stack with the orchestrator started last."""
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description=(
            "Use simulation clock across the stack. Defaults to false for real-robot "
            "operation (wall clock); set to true together with sim_mode:=true when "
            "running against the MuJoCo simulation."
        ),
    )
    scene_arg = DeclareLaunchArgument(
        "scene",
        default_value="loose_pile",
        description="Scene passed to kinova_mujoco_sim.",
    )
    show_gui_arg = DeclareLaunchArgument(
        "show_gui",
        default_value="true",
        description="Whether to show the MuJoCo GUI.",
    )
    sim_mode_arg = DeclareLaunchArgument(
        "sim_mode",
        default_value="false",
        description=(
            "If true, include launch_sim_moveit.launch.py to bring up the MuJoCo "
            "simulation and MoveIt stack alongside the orchestrator. If false, "
            "the orchestrator assumes an external sim/MoveIt stack is already running."
        ),
    )
    perception_params_arg = DeclareLaunchArgument(
        "perception_params_file",
        default_value=str(
            Path(get_package_share_directory("kinova_lego_assembly")) / "config" / "params.yml"
        ),
        description=(
            "Path to the shared perception parameter file (camera, segmentation "
            "backend, and MMOR detection topics) loaded by the segmentation node."
        ),
    )
    mmor_bringup_delay_arg = DeclareLaunchArgument(
        "mmor_bringup_delay_s",
        default_value="5.0",
        description=(
            "Delay (s) after segmentation starts before the lifecycle controller "
            "requests MMOR configure+activate, giving the MMOR container time to "
            "spin up and expose its change_state service."
        ),
    )
    # 1. Optional simulation + MoveIt stack (only when sim_mode:=true).
    sim_moveit_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("kinova_lego_assembly"),
                    "launch",
                    "launch_sim_moveit.launch.py",
                ]
            )
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "scene": LaunchConfiguration("scene"),
            "show_gui": LaunchConfiguration("show_gui"),
            "sim_rviz": "false",
            "moveit_rviz": "false",
        }.items(),
        condition=IfCondition(LaunchConfiguration("sim_mode")),
    )

    # Single RViz for the orchestrated task view.
    orchestrator_rviz_config = str(
        Path(get_package_share_directory("kinova_lego_assembly"))
        / "config"
        / "rviz"
        / "orchestrator.rviz"
    )
    # MoveIt config so the RViz MotionPlanning panel gets SRDF, kinematics,
    # and joint limits (otherwise the panel cannot parse robot_description_semantic).
    # This is the SIMULATION robot description (built from the MuJoCo xacro) and
    # is only fed to RViz in sim_mode. On the real robot the bringup publishes
    # /robot_description and /robot_description_semantic, so RViz sources its
    # model from those topics instead (see the real-mode RViz node below).
    # Injecting the sim xacro as a param in real mode makes RViz build a second,
    # sim-flavoured robot on top of the published one (the "multiple robot
    # models" overlap), so we keep these params sim-only.
    kinova_share = get_package_share_directory(helpers.PACKAGE)
    description_xacro = Path(kinova_share) / "description" / "gen3_7dof_mujoco.urdf.xacro"
    home = helpers.home_positions(kinova_share)
    arm_x, arm_y, arm_z = layout.ARM1_ANCHOR
    robot_description = helpers.build_robot_description(
        kinova_share,
        prefix="",
        arm_anchor=layout.ARM1_ANCHOR,
        home=home,
    )
    xacro_mappings = {
        "name": "gen3",
        "prefix": "",
        "vision": "true",
        "arm_x": str(arm_x),
        "arm_y": str(arm_y),
        "arm_z": str(arm_z),
        "init_joint_1": str(home.get("joint_1", 0.0)),
        "init_joint_2": str(home.get("joint_2", 0.0)),
        "init_joint_3": str(home.get("joint_3", 0.0)),
        "init_joint_4": str(home.get("joint_4", 0.0)),
        "init_joint_5": str(home.get("joint_5", 0.0)),
        "init_joint_6": str(home.get("joint_6", 0.0)),
        "init_joint_7": str(home.get("joint_7", 0.0)),
        "init_gripper": "0.0",
    }
    moveit_config = (
        MoveItConfigsBuilder("gen3", package_name="kinova_lego_assembly")
        .robot_description(file_path=str(description_xacro), mappings=xacro_mappings)
        .robot_description_semantic()
        .trajectory_execution()
        .robot_description_kinematics()
        .planning_pipelines()
        .joint_limits()
        .to_moveit_configs()
    )

    # Sim-mode RViz: model + SRDF/kinematics/planning params come from the sim
    # xacro-derived moveit_config (no robot bringup is publishing them).
    orchestrator_rviz_node_sim = Node(
        condition=IfCondition(LaunchConfiguration("sim_mode")),
        package="rviz2",
        executable="rviz2",
        name="orchestrator_rviz",
        output="screen",
        arguments=["-d", orchestrator_rviz_config],
        parameters=[
            robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"),
                    value_type=bool,
                )
            },
        ],
    )

    # Real-mode RViz: the robot bringup already publishes /robot_description and
    # /robot_description_semantic, which both the RobotModel display (topic
    # source) and the MotionPlanning panel pick up. Passing no sim robot
    # description here avoids the duplicate/overlapping robot model.
    orchestrator_rviz_node_real = Node(
        condition=UnlessCondition(LaunchConfiguration("sim_mode")),
        package="rviz2",
        executable="rviz2",
        name="orchestrator_rviz",
        output="screen",
        arguments=["-d", orchestrator_rviz_config],
        parameters=[
            {
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"),
                    value_type=bool,
                )
            },
        ],
    )

    # 2. Segmentation service node (exposes /LegoSegmentation/segment_image).
    segmentation_node = Node(
        package="kinova_lego_assembly",
        executable="lego_segmentation_node",
        name="LegoSegmentation",
        output="screen",
        parameters=[
            LaunchConfiguration("perception_params_file"),
            {
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"),
                    value_type=bool,
                )
            },
        ],
    )

    # 3. MMOR object recognition (provides /object_detection + object_list).
    mmor_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("mir_object_recognition"),
                    "ros",
                    "launch",
                    "multimodal_object_recognition.launch.py",
                ]
            )
        ),
    )

    # 3b. Manage MMOR's lifecycle. On start it drives configure -> activate via
    #     MMOR's change_state service synchronously (configure completes, i.e.
    #     weights load, before activate). It then stays alive and, on launch
    #     shutdown, drives deactivate -> cleanup so MMOR releases resources
    #     cleanly. Delayed slightly so the MMOR container exposes its services.
    mmor_lifecycle_manager = TimerAction(
        period=LaunchConfiguration("mmor_bringup_delay_s"),
        actions=[
            Node(
                package="kinova_lego_assembly",
                executable="lifecycle_manager_node",
                name="mmor_lifecycle_manager",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"),
                            value_type=bool,
                        ),
                        "lc_name": "mmor",
                    }
                ],
            )
        ],
    )

    # 4. Task orchestrator, started only after the segmentation node starts.
    #    Load the sim's home pose so MoveToHome returns the arm to the spawn
    #    configuration (the perception viewpoint) instead of the all-zeros default.
    orchestrator_params = {
        "use_sim_time": ParameterValue(
            LaunchConfiguration("use_sim_time"),
            value_type=bool,
        ),
    }
    static_transforms_file = str(
        Path(get_package_share_directory("kinova_lego_assembly"))
        / "config"
        / "static_transforms.yaml"
    )
    motion_planning_params_file = str(
        Path(get_package_share_directory("kinova_lego_assembly")) / "config" / "params.yml"
    )
    # Publishes the static keep-out cuboids (obstacle arm, table) into MoveIt's
    # planning scene and, from the same params, matching RViz cube markers.
    planning_scene_obstacle_node = Node(
        package="kinova_lego_assembly",
        executable="planning_scene_obstacle_publisher_node",
        name="planning_scene_obstacle_publisher",
        output="screen",
        parameters=[
            motion_planning_params_file,
            {
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"),
                    value_type=bool,
                )
            },
        ],
    )
    # NOTE: do NOT set name= here. launch_ros turns name= into a process-global
    # "-r __node:=task_orchestrator" remap. py_trees_ros creates its own internal
    # tree node via rclpy.create_node("assembly_tree") (no override hook), so the
    # global remap renames THAT node to task_orchestrator too, colliding with this
    # node. The colliding names break py-trees-tree-viewer auto-discovery (it sees
    # duplicate snapshot-stream services under one namespace). The node already
    # names itself "task_orchestrator" in code, so this remap is redundant anyway.
    orchestrator_node = Node(
        package="kinova_lego_assembly",
        executable="task_orchestrator_node",
        output="screen",
        parameters=[
            motion_planning_params_file,
            static_transforms_file,
            orchestrator_params,
        ],
    )

    start_mmor_then_orchestrator = RegisterEventHandler(
        OnProcessStart(
            target_action=segmentation_node,
            on_start=[
                mmor_stack,
                mmor_lifecycle_manager,
                planning_scene_obstacle_node,
                orchestrator_node,
            ],
        )
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            scene_arg,
            show_gui_arg,
            sim_mode_arg,
            perception_params_arg,
            mmor_bringup_delay_arg,
            sim_moveit_stack,
            orchestrator_rviz_node_sim,
            orchestrator_rviz_node_real,
            segmentation_node,
            start_mmor_then_orchestrator,
        ]
    )
