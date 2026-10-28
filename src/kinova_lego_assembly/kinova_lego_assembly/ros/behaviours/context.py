"""Shared runtime context injected into orchestrator behaviours.

Bundles the tunable parameters and the ROS adapters (motion, gripper,
perception clients, tf) so each behaviour receives everything it needs
via one object instead of reaching back into the node. Behaviours stay
thin; all wiring happens once when the node builds the context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rclpy.node import Node
from rclpy.publisher import Publisher
from tf2_ros import Buffer

from bitbots_lego_assembly.core.blueprint import BlueprintUtils
from bitbots_lego_assembly.ros.background import SpinningNode
from bitbots_lego_assembly.ros.clients import (
    ComputeFkClient,
    MoveActionClient,
    ObjectDetectionClient,
    ObjectListReader,
    SegmentImageClient,
)
from bitbots_lego_assembly.ros.gripper import GripperController
from bitbots_lego_assembly.ros.icp_pose_refiner import (
    IcpPoseRefiner,
    IcpRefinementParams,
)
from bitbots_lego_assembly.ros.sensor_data_manager import SensorDataManager


@dataclass
class OrchestratorParams:
    """Tunable parameters shared across behaviours.

    Motion speed profile: moves to home, pregrasp and preplace (including
    retreats) are fast; all other moves are slow.
    """

    base_frame: str = "base_link"
    pose_link: str = "end_effector_link"
    camera_color_topic: str = "/camera/color/image_raw"

    home_joint_names: list[str] = field(default_factory=list)
    home_joint_positions: list[float] = field(default_factory=list)
    # One joint configuration per holder-board place index, in ascending order.
    place_joint_positions: list[list[float]] = field(default_factory=list)

    fast_velocity_scaling: float = 1.0
    slow_velocity_scaling: float = 0.25

    planning_timeout_s: float = 10.0
    execution_timeout_s: float = 30.0

    detection_timeout_s: float = 30.0
    object_list_timeout_s: float = 10.0
    segment_timeout_s: float = 30.0
    grasp_timeout_s: float = 15.0
    image_timeout_s: float = 5.0
    gripper_timeout_s: float = 10.0
    # Arm counts as settled once every joint stays below this velocity for
    # settle_hold_s; used before capturing images for perception.
    settle_velocity_tolerance: float = 0.01
    settle_hold_s: float = 0.2
    settle_timeout_s: float = 3.0
    # Height above the place pose where the closed gripper
    # starts its snap move
    pre_snap_offset_m: float = 1.2e-2
    # Height above the place pose that the gripper starts snapping attempt
    snap_pose_offset_m: float = 5e-3

    # Grasp search: shift a fully failed candidate group along grasp frame x.
    grasp_x_offset_step_m: float = 0.002
    grasp_x_offset_steps: int = 5

    dummy_place_y_offset_m: float = -0.5
    icp_refinement: IcpRefinementParams = field(default_factory=IcpRefinementParams)


@dataclass
class OrchestratorContext:
    """Everything a behaviour needs at tick time."""

    node: Node
    spinner: SpinningNode
    params: OrchestratorParams
    blueprint: BlueprintUtils
    move_client: MoveActionClient
    fk_client: ComputeFkClient
    sensor: SensorDataManager
    gripper: GripperController
    detection_client: ObjectDetectionClient
    object_list_reader: ObjectListReader
    segment_client: SegmentImageClient
    tf_buffer: Buffer | None = None
    icp_pose_refiner: IcpPoseRefiner | None = None
    status_pub: Publisher | None = None
    total_bricks: int = 0
    place_index: int = 1
