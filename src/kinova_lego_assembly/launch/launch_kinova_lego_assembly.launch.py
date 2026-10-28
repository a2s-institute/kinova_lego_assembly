#!/usr/bin/env python3
"""Launch file for the LegoSegmentation service node."""

from pathlib import Path

from ament_index_python import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter

from launch import LaunchDescription


def generate_launch_description():  # noqa: D103

    remappable_topics = [
        DeclareLaunchArgument("segment_service", default_value="~/segment_image"),
    ]

    args = [
        DeclareLaunchArgument("name", default_value="LegoSegmentation", description="node name"),
        DeclareLaunchArgument("namespace", default_value="", description="node namespace"),
        DeclareLaunchArgument(
            "params",
            default_value=str(
                Path(get_package_share_directory("kinova_lego_assembly")) / "config" / "params.yml"
            ),
            description="path to parameter file",
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="ROS logging level (debug, info, warn, error, fatal)",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="false", description="use simulation clock"
        ),
        *remappable_topics,
    ]

    nodes = [
        Node(
            package="kinova_lego_assembly",
            executable="lego_segmentation_node",
            namespace=LaunchConfiguration("namespace"),
            name=LaunchConfiguration("name"),
            parameters=[LaunchConfiguration("params")],
            arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
            remappings=[
                (la.default_value[0].text, LaunchConfiguration(la.name)) for la in remappable_topics
            ],
            output="screen",
            emulate_tty=True,
        )
    ]

    return LaunchDescription(
        [
            *args,
            SetParameter("use_sim_time", LaunchConfiguration("use_sim_time")),
            *nodes,
        ]
    )
