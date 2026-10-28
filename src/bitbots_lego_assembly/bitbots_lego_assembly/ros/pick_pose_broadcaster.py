"""Background broadcaster for the goal-grasp visualisation frames.

While assembly runs, the orchestrator republishes two debug frames derived
from the current grasp target on the ``assembly`` blackboard:

* ``pick_ee_pose``: the goal ``end_effector_link`` pose sent to the planner,
  mirroring the ``grasp_pose`` blackboard key (parent ``base_link``).
* ``pick_pose``: the corresponding goal ``grip_zone`` pose, chained off
  ``pick_ee_pose`` via the static ``end_effector_link -> grip_zone`` offset.

This is a cross-phase background service (the frames stay alive from grasp
selection until the brick is placed, i.e. until the next pick overwrites
``grasp_pose``), so it lives outside the behaviour tree as its own timer-driven
unit rather than a tree leaf that only ticks while its subtree is active.
"""

from __future__ import annotations

import py_trees
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from tf2_ros import Buffer, TransformBroadcaster, TransformException

_GRASP_POSE_KEY = "/assembly/grasp_pose"
_PICK_EE_FRAME_ID = "pick_ee_pose"
_PICK_GRIP_FRAME_ID = "pick_pose"
_EE_FRAME_ID = "end_effector_link"
_GRIP_ZONE_FRAME_ID = "grip_zone"
_PUBLISH_PERIOD_S = 0.2  # 5 Hz


class PickPoseBroadcaster:
    """Republish the goal ee/grip_zone visualisation frames on a timer.

    Parameters
    ----------
    node : Node
        Owning node; provides the clock, timer, and TF broadcaster.
    tf_buffer : Buffer
        Buffer used to resolve the static ``end_effector_link -> grip_zone``
        offset once and cache it.
    base_frame : str
        Parent frame for ``pick_ee_pose`` (typically ``base_link``).
    """

    def __init__(self, node: Node, tf_buffer: Buffer, base_frame: str) -> None:
        """Wire the broadcaster, blackboard client, and republish timer."""
        self._node = node
        self._tf_buffer = tf_buffer
        self._base_frame = base_frame
        self._ee_to_grip_offset = None

        self._broadcaster = TransformBroadcaster(node)
        self._bb = py_trees.blackboard.Client(name="pick_pose_viz")
        self._bb.register_key(key=_GRASP_POSE_KEY, access=py_trees.common.Access.READ)
        # Own callback group so the periodic broadcast never blocks (or is
        # blocked by) the action or tree-tick callbacks.
        self._timer = node.create_timer(
            _PUBLISH_PERIOD_S,
            self._broadcast,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def _broadcast(self) -> None:
        """Timer callback: broadcast the goal ee/grip_zone visualisation TFs."""
        if not self._bb.exists(_GRASP_POSE_KEY):
            return
        grasp_pose = self._bb.get(_GRASP_POSE_KEY)
        if grasp_pose is None:
            return
        now = self._node.get_clock().now().to_msg()

        ee_tf = TransformStamped()
        ee_tf.header.stamp = now
        ee_tf.header.frame_id = self._base_frame
        ee_tf.child_frame_id = _PICK_EE_FRAME_ID
        ee_tf.transform.translation.x = grasp_pose.pose.position.x
        ee_tf.transform.translation.y = grasp_pose.pose.position.y
        ee_tf.transform.translation.z = grasp_pose.pose.position.z
        ee_tf.transform.rotation = grasp_pose.pose.orientation

        transforms = [ee_tf]

        grip_offset = self._lookup_ee_to_grip_offset()
        if grip_offset is not None:
            grip_tf = TransformStamped()
            grip_tf.header.stamp = now
            grip_tf.header.frame_id = _PICK_EE_FRAME_ID
            grip_tf.child_frame_id = _PICK_GRIP_FRAME_ID
            grip_tf.transform = grip_offset
            transforms.append(grip_tf)

        self._broadcaster.sendTransform(transforms)

    def _lookup_ee_to_grip_offset(self):
        """Return the cached static ``end_effector_link -> grip_zone`` transform."""
        if self._ee_to_grip_offset is not None:
            return self._ee_to_grip_offset
        try:
            tf = self._tf_buffer.lookup_transform(
                _EE_FRAME_ID, _GRIP_ZONE_FRAME_ID, rclpy.time.Time()
            )
        except TransformException:
            return None
        self._ee_to_grip_offset = tf.transform
        return self._ee_to_grip_offset
