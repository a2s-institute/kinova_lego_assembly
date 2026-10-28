"""ROS helpers for pulling a single sensor frame on demand.

The task orchestrator only needs one image per capture (for scene
validation), so this uses ``rclpy.wait_for_message`` to grab the next
frame from a camera topic instead of holding a long-lived subscription.
All rclpy usage is confined here to keep the perception logic in
``core/`` ROS-free.
"""

from __future__ import annotations

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import Image, PointCloud2


class SensorDataManager:
    """On-demand accessor for camera frames used by the orchestrator."""

    def __init__(self, node: Node) -> None:
        """Bind the manager to a ROS node for waiting/logging.

        Parameters
        ----------
        node : rclpy.node.Node
            The orchestrator node whose executor drives the message wait
            and whose logger receives diagnostics.
        """
        self._node = node

    def get_color_image_ros(self, topic: str, timeout_s: float = 5.0) -> Image | None:
        """Wait for and return the next colour image on ``topic``.

        Parameters
        ----------
        topic : str
            Camera colour image topic (as configured on the node's
            parameter server).
        timeout_s : float, optional
            Maximum time to wait for a frame. Defaults to ``5.0``.

        Returns
        -------
        sensor_msgs.msg.Image or None
            The received image, or ``None`` if no frame arrived within
            ``timeout_s``.
        """
        success, message = wait_for_message(
            Image,
            self._node,
            topic,
            time_to_wait=float(timeout_s),
        )
        if not success or message is None:
            self._node.get_logger().error(
                f"Timed out after {timeout_s:.1f}s waiting for a colour image on '{topic}'"
            )
            return None
        return message

    def get_point_cloud_ros(
        self,
        topic: str,
        timeout_s: float = 5.0,
    ) -> PointCloud2 | None:
        """Wait for and return the next point cloud on ``topic``.

        Parameters
        ----------
        topic : str
            Point-cloud topic.
        timeout_s : float, optional
            Maximum wait in seconds.

        Returns
        -------
        sensor_msgs.msg.PointCloud2 or None
            Received cloud, or ``None`` after timeout.
        """
        success, message = wait_for_message(
            PointCloud2,
            self._node,
            topic,
            qos_profile=qos_profile_sensor_data,
            time_to_wait=float(timeout_s),
        )
        if not success or message is None:
            self._node.get_logger().error(
                f"Timed out after {timeout_s:.1f}s waiting for a point cloud on '{topic}'"
            )
            return None
        return message
