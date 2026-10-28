"""Utilities for computing grasp and pregrasp poses from a PoseStamped estimate."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from math import acos, pi

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from rclpy.duration import Duration
from rclpy.logging import get_logger
from tf2_ros import Buffer, TransformException
from tf_transformations import (
    quaternion_from_euler,
    quaternion_inverse,
    quaternion_matrix,
    quaternion_multiply,
)

_LOG = get_logger("grasp")

_BASE_FRAME_ID = "base_link"
_GRIP_ZONE_FRAME_ID = "grip_zone"
_END_EFFECTOR_FRAME_ID = "end_effector_link"
PREGRASP_OFFSET_Z = 0.10  # meters

# Candidate z-axis rotations (radians) applied after the +pi about-X flip,
# keyed by brick type. A 4x2 brick's long axis must be reoriented so the
# gripper closes along the short axis, giving two options (+/- pi/2). A 2x2
# brick is symmetric, so all four quarter turns are viable candidates.
_Z_ROTATION_CANDIDATES: dict[str, tuple[float, ...]] = {
    "4x2": (pi / 2.0, -pi / 2.0),
    "2x2": (0.0, pi / 2.0, pi, -pi / 2.0),
}


@dataclass(frozen=True)
class GraspPoseCandidate:
    """A ranked grasp/pregrasp pair and the score used to rank it."""

    rank: int
    angular_distance: float
    z_rotation: float
    grasp_pose: PoseStamped
    pregrasp_pose: PoseStamped


def compute_grasp_pregrasp_candidates(
    brick_pose: PoseStamped,
    brick_type: str,
    tf_buffer: Buffer,
) -> tuple[GraspPoseCandidate, ...]:
    """Compute all grasp candidates, ordered from best to worst.

    A local-frame +pi rotation about X is always applied first so the object
    z-axis points up. A z-axis rotation candidate is then chosen so the gripper
    approaches along the brick's short axis (4x2) or the most convenient
    quarter turn (symmetric 2x2). The winning candidate is the one whose goal
    end-effector orientation is angularly closest to the current
    ``end_effector_link`` orientation in ``base_link``.

    All quaternion math uses the ROS2/``geometry_msgs`` ``(x, y, z, w)``
    ordering via :mod:`tf_transformations`.

    Parameters
    ----------
    brick_pose : PoseStamped
        Object pose estimate expressed in ``base_link``.
    brick_type : str
        Brick type, one of ``{"4x2", "2x2"}``, selecting the z-rotation
        candidate set.
    tf_buffer : Buffer
        TF2 buffer used to query the ``grip_zone -> end_effector_link`` offset
        and the current ``base_link -> end_effector_link`` orientation.

    Returns
    -------
    tuple of GraspPoseCandidate
        Candidates sorted by angular distance, with rank ``0`` being the
        candidate closest to the current end-effector orientation.
    """
    candidates = _Z_ROTATION_CANDIDATES.get(brick_type)
    if candidates is None:
        raise RuntimeError(
            f"unknown brick_type {brick_type!r}; expected one of {sorted(_Z_ROTATION_CANDIDATES)}"
        )

    flipped_quat_xyzw = quaternion_multiply(
        _quat_xyzw(brick_pose.pose.orientation),
        quaternion_from_euler(pi, 0.0, 0.0),
    )

    grip_to_ee = _lookup_transform(tf_buffer, _GRIP_ZONE_FRAME_ID, _END_EFFECTOR_FRAME_ID)
    offset_quat_xyzw = _quat_xyzw(grip_to_ee.transform.rotation)
    offset_translation = _xyz(grip_to_ee.transform.translation)

    ee_reference = _lookup_transform(tf_buffer, _BASE_FRAME_ID, _END_EFFECTOR_FRAME_ID)
    ee_quat_xyzw = _quat_xyzw(ee_reference.transform.rotation)

    # Only the goal end-effector orientation drives candidate selection: the
    # grasp position is identical across candidates, so it is left out of the
    # metric and computed once for the winner below.
    evaluated = []
    scored_candidates = []
    for theta in candidates:
        corrected_quat_xyzw = quaternion_multiply(
            flipped_quat_xyzw, quaternion_from_euler(0.0, 0.0, theta)
        )
        grasp_quat_xyzw = quaternion_multiply(corrected_quat_xyzw, offset_quat_xyzw)
        error = _quaternion_angular_distance(grasp_quat_xyzw, ee_quat_xyzw)
        evaluated.append(f"z_rotation_rad={theta:+.6f} angular_distance_rad={error:.6f}")
        scored_candidates.append((error, theta, corrected_quat_xyzw, grasp_quat_xyzw))

    scored_candidates.sort(key=lambda candidate: candidate[0])
    best_error, best_theta, _, _ = scored_candidates[0]
    _LOG.info(
        f"grasp candidate evaluation for brick_type={brick_type!r}: "
        f"[{', '.join(evaluated)}] -> selected "
        f"z_rotation_rad={best_theta:+.6f} angular_distance_rad={best_error:.6f}"
    )

    stamped_now = rclpy.clock.Clock().now().to_msg()
    ranked_candidates = []
    for rank, (error, theta, corrected_quat_xyzw, grasp_quat_xyzw) in enumerate(scored_candidates):
        rotated_offset = _rotate_vector(corrected_quat_xyzw, offset_translation)
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = _BASE_FRAME_ID
        grasp_pose.header.stamp = stamped_now
        grasp_pose.pose.position.x = brick_pose.pose.position.x + rotated_offset[0]
        grasp_pose.pose.position.y = brick_pose.pose.position.y + rotated_offset[1]
        grasp_pose.pose.position.z = brick_pose.pose.position.z + rotated_offset[2]
        grasp_pose.pose.orientation = _to_quaternion(grasp_quat_xyzw)

        pregrasp_pose = PoseStamped()
        pregrasp_pose.header.frame_id = _BASE_FRAME_ID
        pregrasp_pose.header.stamp = stamped_now
        pregrasp_pose.pose.position.x = grasp_pose.pose.position.x
        pregrasp_pose.pose.position.y = grasp_pose.pose.position.y
        pregrasp_pose.pose.position.z = grasp_pose.pose.position.z + PREGRASP_OFFSET_Z
        pregrasp_pose.pose.orientation = grasp_pose.pose.orientation
        ranked_candidates.append(GraspPoseCandidate(rank, error, theta, grasp_pose, pregrasp_pose))

    return tuple(ranked_candidates)


def shift_candidate_x(candidate: GraspPoseCandidate, offset_m: float) -> GraspPoseCandidate:
    """Return a copy of ``candidate`` with grasp and pregrasp moved along grasp x.

    Parameters
    ----------
    candidate : GraspPoseCandidate
        Candidate to shift. Left unchanged.
    offset_m : float
        Translation along the x-axis of the grasp pose orientation, in metres.
        The same translation is applied to both poses.

    Returns
    -------
    GraspPoseCandidate
        Shifted candidate with the same rank, score and orientation.
    """
    grasp_pose = deepcopy(candidate.grasp_pose)
    pregrasp_pose = deepcopy(candidate.pregrasp_pose)
    dx, dy, dz = _rotate_vector(_quat_xyzw(grasp_pose.pose.orientation), (offset_m, 0.0, 0.0))
    for pose in (grasp_pose, pregrasp_pose):
        pose.pose.position.x += dx
        pose.pose.position.y += dy
        pose.pose.position.z += dz
    return replace(candidate, grasp_pose=grasp_pose, pregrasp_pose=pregrasp_pose)


def _quat_xyzw(q: Quaternion) -> tuple[float, float, float, float]:
    """Return a ``(x, y, z, w)`` tuple from a ``Quaternion`` message."""
    return (q.x, q.y, q.z, q.w)


def _xyz(v) -> tuple[float, float, float]:
    """Return an ``(x, y, z)`` tuple from a vector-like message."""
    return (v.x, v.y, v.z)


def _to_quaternion(quat_xyzw) -> Quaternion:
    """Build a ``Quaternion`` message from a ``(x, y, z, w)`` sequence."""
    q = Quaternion()
    q.x, q.y, q.z, q.w = quat_xyzw
    return q


def _rotate_vector(
    quat_xyzw,
    vec_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a vector by an ``(x, y, z, w)`` quaternion via its rotation matrix."""
    rot = quaternion_matrix(quat_xyzw)
    vx, vy, vz = vec_xyz
    rx = rot[0][0] * vx + rot[0][1] * vy + rot[0][2] * vz
    ry = rot[1][0] * vx + rot[1][1] * vy + rot[1][2] * vz
    rz = rot[2][0] * vx + rot[2][1] * vy + rot[2][2] * vz
    return rx, ry, rz


def _quaternion_angular_distance(quat_a_xyzw, quat_b_xyzw) -> float:
    """Return the angular distance (radians) between two ``(x, y, z, w)`` quaternions."""
    relative = quaternion_multiply(quat_a_xyzw, quaternion_inverse(quat_b_xyzw))
    clamped_w = max(-1.0, min(1.0, abs(relative[3])))
    return 2.0 * acos(clamped_w)


def _lookup_transform(tf_buffer: Buffer, target_frame: str, source_frame: str):
    """Look up ``target_frame -> source_frame`` or raise ``RuntimeError``."""
    try:
        return tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=0.2),
        )
    except TransformException as exc:
        raise RuntimeError(f"Failed to lookup TF {target_frame}->{source_frame}: {exc}") from exc
