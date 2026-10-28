"""Publish static cuboid obstacles into MoveIt's planning scene.

This sidecar node is intended for real-robot mode where MoveIt may be started
outside this package (for example by vendor bringup). It repeatedly applies a
set of collision boxes via ``ApplyPlanningScene`` so planners always see the
reserved keep-out regions (for example the obstacle arm and the table). The
same ``cuboid_center``/``cuboid_size`` parameters also drive latched RViz
``Marker`` cubes, so the picture in RViz always matches the collision geometry
MoveIt plans against (single source of truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

_DEFAULT_PUBLISH_PERIOD_S = 2.0


@dataclass(frozen=True)
class _Obstacle:
    """A single static cuboid obstacle and its RViz marker colour."""

    object_id: str
    center: tuple[float, ...]
    size: tuple[float, ...]
    color: tuple[float, ...]


def _as_floats(value: object, name: str, count: int) -> tuple[float, ...]:
    """Coerce a fixed-length numeric parameter into a tuple of floats."""
    seq = [float(v) for v in value]  # type: ignore[union-attr]
    if len(seq) != count:
        raise ValueError(f"{name} must have exactly {count} values, got {len(seq)}: {seq!r}")
    return tuple(seq)


class PlanningSceneObstaclePublisher(Node):
    """Keep static cuboid obstacles present in MoveIt's planning scene."""

    def __init__(self) -> None:
        super().__init__("planning_scene_obstacle_publisher")
        self._obstacle_ids = self._declare_parameters()
        self._log = self.get_logger()

        self._enabled = bool(self._param("enabled"))
        self._service_name = str(self._param("apply_service"))
        self._frame_id = str(self._param("frame_id"))
        self._service_wait_timeout_s = float(self._param("service_wait_timeout_s"))
        self._publish_period_s = float(self._param("publish_period_s"))
        self._marker_topic = str(self._param("marker_topic"))
        self._obstacles = self._load_obstacles()

        if not self._enabled:
            self._log.info("planning_scene_obstacle.enabled=false; node will stay idle.")
            return

        self._client = self.create_client(ApplyPlanningScene, self._service_name)
        self._request_in_flight = False
        self._service_warned = False
        self._publish_count = 0

        # Latched markers so RViz shows the exact same boxes MoveIt plans
        # against, driven by the same cuboid_center/cuboid_size parameters. The
        # depth must cover every obstacle: markers are published once and rely
        # on transient-local latching, so a shallower queue would drop all but
        # the last obstacle for any RViz display that joins late.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=max(len(self._obstacles), 1),
        )
        self._marker_pub = self.create_publisher(Marker, self._marker_topic, latched_qos)
        self._publish_markers()

        period = self._publish_period_s
        if period <= 0.0:
            period = _DEFAULT_PUBLISH_PERIOD_S
        self._timer = self.create_timer(period, self._tick)
        obstacle_ids = ", ".join(o.object_id for o in self._obstacles)
        self._log.info(
            "Planning-scene cuboid publisher enabled: "
            f"obstacles=[{obstacle_ids}], frame='{self._frame_id}', "
            f"service='{self._service_name}', marker_topic='{self._marker_topic}', "
            f"period={period:.2f}s"
        )

    def _param(self, name: str) -> Any:
        """Read a ``planning_scene_obstacle.<name>`` parameter value."""
        return self.get_parameter(f"planning_scene_obstacle.{name}").value

    def _load_obstacles(self) -> list[_Obstacle]:
        """Read the per-obstacle geometry/colour parameters into a list."""
        obstacles: list[_Obstacle] = []
        for object_id in self._obstacle_ids:
            center = _as_floats(
                self._param(f"{object_id}.cuboid_center"), f"{object_id}.cuboid_center", 3
            )
            size = _as_floats(
                self._param(f"{object_id}.cuboid_size"), f"{object_id}.cuboid_size", 3
            )
            color = _as_floats(
                self._param(f"{object_id}.marker_color_rgba"), f"{object_id}.marker_color_rgba", 4
            )
            obstacles.append(_Obstacle(object_id=object_id, center=center, size=size, color=color))
        return obstacles

    def _declare_parameters(self) -> list[str]:
        """Declare all parameters and return the configured obstacle ids.

        Only generic fallbacks are declared here; ``params.yml`` is the
        authoritative source for each obstacle's geometry and colour, so the
        real values are not duplicated in code.
        """
        self.declare_parameter("planning_scene_obstacle.enabled", True)
        self.declare_parameter("planning_scene_obstacle.apply_service", "/apply_planning_scene")
        self.declare_parameter("planning_scene_obstacle.frame_id", "base_link")
        self.declare_parameter("planning_scene_obstacle.service_wait_timeout_s", 0.5)
        self.declare_parameter(
            "planning_scene_obstacle.publish_period_s", _DEFAULT_PUBLISH_PERIOD_S
        )
        # RViz visual boxes are driven by the same cuboid parameters, so the
        # picture in RViz always matches the collision geometry MoveIt plans
        # against.
        self.declare_parameter(
            "planning_scene_obstacle.marker_topic", "/planning_scene_obstacle/marker"
        )
        # Each id gets its own <id>.cuboid_center / <id>.cuboid_size /
        # <id>.marker_color_rgba block in params.yml. base_link is attached to
        # the work arm (ARM1); the obstacle arm (ARM2) sits at negative y
        # relative to that frame and the table extends under and around it.
        self.declare_parameter("planning_scene_obstacle.obstacles", ["second_arm_region", "table"])
        obstacle_ids = [
            str(v) for v in self.get_parameter("planning_scene_obstacle.obstacles").value
        ]
        for object_id in obstacle_ids:
            prefix = f"planning_scene_obstacle.{object_id}"
            self.declare_parameter(f"{prefix}.cuboid_center", [0.0, 0.0, 0.0])
            self.declare_parameter(f"{prefix}.cuboid_size", [0.1, 0.1, 0.1])
            self.declare_parameter(f"{prefix}.marker_color_rgba", [0.9, 0.2, 0.2, 0.4])
        return obstacle_ids

    def _tick(self) -> None:
        """Apply or refresh the static cuboid obstacle in the planning scene."""
        if self._request_in_flight:
            return
        if not self._client.wait_for_service(timeout_sec=self._service_wait_timeout_s):
            if not self._service_warned:
                self._log.warn(
                    f"Waiting for MoveIt planning-scene service '{self._service_name}'..."
                )
                self._service_warned = True
            return
        self._service_warned = False

        request = ApplyPlanningScene.Request()
        request.scene = self._build_scene_diff()
        future = self._client.call_async(request)
        self._request_in_flight = True
        future.add_done_callback(self._on_apply_done)

    def _build_scene_diff(self) -> PlanningScene:
        """Build a diff scene containing one static box per obstacle."""
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [
            self._build_collision_object(obstacle) for obstacle in self._obstacles
        ]
        return scene

    @staticmethod
    def _pose_from_center(center: tuple[float, ...]) -> Pose:
        """Build an axis-aligned ``Pose`` at ``center`` (identity orientation)."""
        pose = Pose()
        pose.position.x = float(center[0])
        pose.position.y = float(center[1])
        pose.position.z = float(center[2])
        pose.orientation.w = 1.0
        return pose

    def _build_collision_object(self, obstacle: _Obstacle) -> CollisionObject:
        """Build a single static box collision object for one obstacle."""
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(v) for v in obstacle.size]

        collision_object = CollisionObject()
        collision_object.header.frame_id = self._frame_id
        collision_object.id = obstacle.object_id
        collision_object.primitives = [primitive]
        collision_object.primitive_poses = [self._pose_from_center(obstacle.center)]
        collision_object.operation = CollisionObject.ADD
        return collision_object

    def _publish_markers(self) -> None:
        """Publish one RViz cube marker per obstacle collision box."""
        stamp = self.get_clock().now().to_msg()
        for index, obstacle in enumerate(self._obstacles):
            marker = Marker()
            marker.header.frame_id = self._frame_id
            marker.header.stamp = stamp
            marker.ns = obstacle.object_id
            marker.id = index
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = self._pose_from_center(obstacle.center)
            marker.scale.x = float(obstacle.size[0])
            marker.scale.y = float(obstacle.size[1])
            marker.scale.z = float(obstacle.size[2])
            marker.color = ColorRGBA(
                r=obstacle.color[0],
                g=obstacle.color[1],
                b=obstacle.color[2],
                a=obstacle.color[3],
            )
            self._marker_pub.publish(marker)

    def _on_apply_done(self, future: Future) -> None:
        """Log the result of applying the planning-scene diff."""
        self._request_in_flight = False
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - transport/runtime failures
            self._log.warn(f"Failed to apply planning-scene obstacle: {exc}")
            return

        if response is None or not bool(response.success):
            self._log.warn("MoveIt rejected planning-scene diff for obstacle cuboids.")
            return

        self._publish_count += 1
        if self._publish_count == 1:
            self._log.info(
                f"Obstacle cuboids added to planning scene: "
                f"{', '.join(o.object_id for o in self._obstacles)}."
            )
        elif self._publish_count % 30 == 0:
            self._log.debug(f"Obstacle cuboid refresh count={self._publish_count}")


def main(args: list[str] | None = None) -> None:
    """Run the planning-scene obstacle publisher node."""
    rclpy.init(args=args)
    node = PlanningSceneObstaclePublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
