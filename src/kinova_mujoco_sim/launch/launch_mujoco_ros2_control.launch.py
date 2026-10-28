"""Launch the Kinova MuJoCo work cell with native ros2_control.

Flow:

1. ``generate_scene_mjcf`` assembles the full work cell (table, work arm +
   gripper, obstacle arm, holder, scene bricks) and dumps it to a single MJCF
   file.
2. Once that file exists, the work arm + gripper ``robot_description`` is built
   from the forked xacros (geometry + ``mujoco_ros2_control/MujocoSystem``) and
   published by ``robot_state_publisher``.
3. The ``mujoco_ros2_control`` node loads the dumped MJCF and the
   ``robot_description`` and starts the controller manager.
4. Once that node is up, the joint state broadcaster, arm joint trajectory
   controller, and gripper controller are spawned.

The work arm anchor and home joint positions come from
``kinova_mujoco_sim.core.scene.layout`` and ``config/home_pose.yaml`` so the URDF
kinematics line up with the pose baked into the MJCF.

The heavy lifting (robot description, camera params, generator interpreter /
environment) lives in ``kinova_mujoco_sim.ros.launch_helpers``; this file only
wires the resulting actions into a node graph.
"""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from kinova_mujoco_sim.ros import launch_helpers as helpers
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context: LaunchContext) -> list:
    """Build the node graph once launch configurations are resolved."""
    from kinova_mujoco_sim.core.scene import layout

    share_dir = get_package_share_directory(helpers.PACKAGE)

    scene = context.perform_substitution(LaunchConfiguration("scene"))
    show_gui = context.perform_substitution(LaunchConfiguration("show_gui"))
    prefix = context.perform_substitution(LaunchConfiguration("prefix"))

    home = helpers.home_positions(share_dir)
    robot_description = helpers.build_robot_description(share_dir, prefix, layout.ARM1_ANCHOR, home)

    controllers_file = str(Path(share_dir) / "config" / "ros2_controllers.yaml")
    cameras_file, camera_remappings = helpers.camera_node_params_file(share_dir)

    # Run the generator with the project venv interpreter (resolved here) since
    # the system interpreter that runs ``ros2`` lacks ``mujoco``/``pytransform3d``.
    # Drop the source-built ``libmujoco`` from LD_LIBRARY_PATH so the pip wheel
    # loads its own ABI-matched library (see helpers.mujoco_free_ld_library_path).
    generate_mjcf = ExecuteProcess(
        cmd=[
            helpers.generator_python(share_dir),
            "-m",
            "kinova_mujoco_sim.core.generate_scene_mjcf",
            "--scene",
            scene,
            "--output",
            helpers.SCENE_MJCF_PATH,
        ],
        name="generate_scene_mjcf",
        output="screen",
        additional_env={"LD_LIBRARY_PATH": helpers.mujoco_free_ld_library_path()},
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[robot_description, {"use_sim_time": True}],
        output="screen",
    )

    mujoco = Node(
        package="mujoco_ros2_control",
        executable="mujoco_ros2_control",
        parameters=[
            robot_description,
            controllers_file,
            cameras_file,
            {"simulation_frequency": 500.0},
            {"realtime_factor": 1.0},
            {"robot_model_path": helpers.SCENE_MJCF_PATH},
            {"show_gui": show_gui.lower() in ("1", "true", "yes")},
        ],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
            *camera_remappings,
        ],
        output="screen",
    )

    def _spawner(controller: str) -> Node:
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                controller,
                "--controller-manager",
                "/controller_manager",
                "--param-file",
                controllers_file,
            ],
            output="screen",
        )

    joint_state_broadcaster = _spawner("joint_state_broadcaster")
    joint_trajectory_controller = _spawner("joint_trajectory_controller")
    robotiq_gripper_controller = _spawner("robotiq_gripper_controller")

    rviz_config = str(Path(share_dir) / "config" / "rviz" / "sim.rviz")
    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration("rviz")),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        output="log",
    )

    start_after_mjcf = RegisterEventHandler(
        OnProcessExit(
            target_action=generate_mjcf,
            on_exit=[
                LogInfo(msg="Scene MJCF generated; starting simulation..."),
                robot_state_publisher,
                mujoco,
                rviz_node,
            ],
        )
    )

    load_controllers = RegisterEventHandler(
        OnProcessStart(
            target_action=mujoco,
            on_start=[
                LogInfo(msg="Simulation up; spawning controllers..."),
                joint_state_broadcaster,
                joint_trajectory_controller,
                robotiq_gripper_controller,
                LogInfo(
                    msg=(
                        "Simulation running. Press Ctrl+C in this terminal to "
                        "stop the simulation and shut down all launched nodes "
                        "(works the same in headless mode). Closing the MuJoCo "
                        "GUI window does not stop the nodes."
                    )
                ),
            ],
        )
    )

    # If the simulation process exits on its own (e.g. it crashes), bring the
    # rest of the launch (robot_state_publisher, rviz, ...) down with it. Normal
    # shutdown is via Ctrl+C, which the launch system propagates to every node.
    shutdown_on_sim_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=mujoco,
            on_exit=[
                LogInfo(msg="Simulation exited; shutting down launch..."),
                EmitEvent(event=Shutdown(reason="MuJoCo simulation exited")),
            ],
        )
    )

    return [generate_mjcf, start_after_mjcf, load_controllers, shutdown_on_sim_exit]


def generate_launch_description() -> LaunchDescription:
    """Declare launch arguments and wire up the deferred setup."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scene",
                default_value="loose_pile",
                description="Registered scene name to populate the work cell.",
            ),
            DeclareLaunchArgument(
                "show_gui",
                default_value="true",
                description="Show the MuJoCo viewer window.",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Start RViz.",
            ),
            DeclareLaunchArgument(
                "prefix",
                default_value="",
                description="Joint/link name prefix (must match the MJCF).",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
