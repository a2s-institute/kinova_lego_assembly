"""Boot-time static transform broadcasting for the orchestrator.

The orchestrator publishes a set of fixed frames (e.g. ``grip_zone``, the
grasp reference offset from ``end_effector_link``) once at startup. Keeping
the frames in ROS parameters (see ``config/static_transforms.yaml``) instead
of hard-coding offsets in the node makes the grip geometry a reviewable
config value rather than a magic number buried in code.

Static transforms are published on the latched ``/tf_static`` topic
(``transient_local`` durability), so any subscriber that joins after the
broadcast still receives the frames as long as the broadcasting node stays
alive. The caller must therefore keep the returned broadcaster referenced
for the lifetime of the node.
"""

from __future__ import annotations

from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


def _declare_transform_params(node: Node, name: str) -> None:
    """Declare the per-transform parameters for ``name`` with empty defaults."""
    node.declare_parameter(f"{name}.parent_frame", "")
    node.declare_parameter(f"{name}.child_frame", "")
    node.declare_parameter(f"{name}.translation", [0.0, 0.0, 0.0])
    node.declare_parameter(f"{name}.rotation", [0.0, 0.0, 0.0, 1.0])


def _build_transform(node: Node, name: str) -> TransformStamped:
    """Build one ``TransformStamped`` from the parameters declared for ``name``.

    Parameters
    ----------
    node : rclpy.node.Node
        Node whose parameters and clock are used.
    name : str
        Transform key; parameters are read from ``<name>.*``.

    Returns
    -------
    geometry_msgs.msg.TransformStamped
        The populated static transform, stamped with the node clock.

    Raises
    ------
    ValueError
        If frames are empty or the translation/rotation lengths are wrong.
    """
    parent = str(node.get_parameter(f"{name}.parent_frame").value)
    child = str(node.get_parameter(f"{name}.child_frame").value)
    translation = [float(v) for v in node.get_parameter(f"{name}.translation").value]
    rotation = [float(v) for v in node.get_parameter(f"{name}.rotation").value]

    if not parent or not child:
        raise ValueError(f"static transform '{name}' missing parent/child frame")
    if len(translation) != 3:
        raise ValueError(f"static transform '{name}' translation must be [x, y, z]")
    if len(rotation) != 4:
        raise ValueError(f"static transform '{name}' rotation must be [x, y, z, w]")

    tf = TransformStamped()
    tf.header.stamp = node.get_clock().now().to_msg()
    tf.header.frame_id = parent
    tf.child_frame_id = child
    tf.transform.translation.x = translation[0]
    tf.transform.translation.y = translation[1]
    tf.transform.translation.z = translation[2]
    tf.transform.rotation.x = rotation[0]
    tf.transform.rotation.y = rotation[1]
    tf.transform.rotation.z = rotation[2]
    tf.transform.rotation.w = rotation[3]
    return tf


def broadcast_static_transforms(node: Node) -> StaticTransformBroadcaster:
    """Broadcast every configured static transform once at boot.

    Reads the ``static_transforms`` string-array parameter (the list of
    transform names) and, for each name, the ``<name>.parent_frame``,
    ``<name>.child_frame``, ``<name>.translation`` and ``<name>.rotation``
    parameters, then latches them all on ``/tf_static`` in a single batch.

    Parameters
    ----------
    node : rclpy.node.Node
        Node that owns the broadcaster and provides the parameters/clock.

    Returns
    -------
    tf2_ros.StaticTransformBroadcaster
        The broadcaster. The caller must keep a reference to it (and keep the
        node spinning) so late ``/tf_static`` subscribers still get the frames.
    """
    log = node.get_logger()
    broadcaster = StaticTransformBroadcaster(node)

    # Dynamic typing: an empty-list default cannot have its type inferred, and
    # the real value (a string array) arrives from the param file.
    node.declare_parameter("static_transforms", None, ParameterDescriptor(dynamic_typing=True))
    names = [str(n) for n in (node.get_parameter("static_transforms").value or [])]
    if not names:
        log.warn("No static_transforms configured; nothing to broadcast.")
        return broadcaster

    transforms: list[TransformStamped] = []
    for name in names:
        _declare_transform_params(node, name)
        transforms.append(_build_transform(node, name))

    broadcaster.sendTransform(transforms)
    frames = ", ".join(f"{tf.header.frame_id}->{tf.child_frame_id}" for tf in transforms)
    log.info(f"Broadcast {len(transforms)} static transform(s) on /tf_static: {frames}")
    return broadcaster
