"""Launch MuJoCo simulation together with the MoveIt stack.

Brings up:

1. ``kinova_mujoco_sim/launch_mujoco_ros2_control.launch.py`` (MuJoCo sim +
   ros2_control + robot_state_publisher).
2. ``move_group`` (MoveIt) configured against the same URDF/xacro the sim
   uses, so ``robot_description``, SRDF, kinematics, and controller
   descriptions all agree across sim, MoveIt, and RViz.
3. Optional RViz2 view (``moveit_rviz:=true``) preloaded with the task
   layout and MoveIt semantic/kinematics parameters so the
   MotionPlanning panel can parse SRDF.

MoveIt startup is deferred by ``moveit_start_delay_s`` so the simulation has
time to publish ``/robot_description`` and expose controllers before
``move_group`` initialises.

Usage:
    ros2 launch kinova_lego_assembly launch_sim_moveit.launch.py
"""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
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
    """Launch simulation first, then bring up MoveIt after a short delay."""
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock across the stack.",
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
    sim_rviz_arg = DeclareLaunchArgument(
        "sim_rviz",
        default_value="false",
        description="Enable RViz from the simulation launch file.",
    )
    moveit_rviz_arg = DeclareLaunchArgument(
        "moveit_rviz",
        default_value="true",
        description="Enable the MoveIt RViz view (orchestrator.rviz).",
    )
    prefix_arg = DeclareLaunchArgument(
        "prefix",
        default_value="",
        description="Joint/link prefix forwarded to kinova_mujoco_sim.",
    )
    moveit_start_delay_arg = DeclareLaunchArgument(
        "moveit_start_delay_s",
        default_value="3.0",
        description="Delay before starting the MoveIt stack.",
    )

    # Build robot_description through the same helper MuJoCo uses so MoveIt,
    # RViz, and robot_state_publisher all consume the same URDF.
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
    rviz_config = str(
        Path(get_package_share_directory("kinova_lego_assembly"))
        / "config"
        / "rviz"
        / "orchestrator.rviz"
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
        .planning_pipelines(default_planning_pipeline="pilz_industrial_motion_planner")
        .joint_limits()
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_planning_scene=True,
            publish_geometry_updates=True,
            publish_state_updates=True,
            publish_transforms_updates=True,
        )
        .to_moveit_configs()
    )

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("kinova_mujoco_sim"),
                    "launch",
                    "launch_mujoco_ros2_control.launch.py",
                ]
            )
        ),
        launch_arguments={
            "scene": LaunchConfiguration("scene"),
            "show_gui": LaunchConfiguration("show_gui"),
            "rviz": LaunchConfiguration("sim_rviz"),
            "prefix": LaunchConfiguration("prefix"),
        }.items(),
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            robot_description,
            {
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"),
                    value_type=bool,
                )
            },
            # Must exceed goal_time in kinova_mujoco_sim ros2_controllers.yaml so
            # the controller's settle window ends before MoveIt cancels the goal.
            {
                "trajectory_execution.allowed_execution_duration_scaling": 1.2,
                "trajectory_execution.allowed_goal_duration_margin": 3.0,
            },
        ],
    )

    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration("moveit_rviz")),
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
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

    delayed_moveit_stack = TimerAction(
        period=LaunchConfiguration("moveit_start_delay_s"),
        actions=[move_group_node, rviz_node],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            scene_arg,
            show_gui_arg,
            sim_rviz_arg,
            moveit_rviz_arg,
            prefix_arg,
            moveit_start_delay_arg,
            sim_launch,
            delayed_moveit_stack,
        ]
    )
