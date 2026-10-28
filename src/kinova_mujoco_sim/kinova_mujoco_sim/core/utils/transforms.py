"""Rotation / transform helpers.

Thin wrappers over :mod:`pytransform3d.rotations`. pytransform3d already uses
MuJoCo's scalar-first ``(w, x, y, z)`` quaternion convention, so no reordering
is needed; these helpers just take angles in degrees and return plain tuples.
"""

from __future__ import annotations

import numpy as np
from pytransform3d import rotations as pr

Quaternion = tuple[float, float, float, float]

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_quat(axis: str, deg: float) -> np.ndarray:
    """Return the ``(w, x, y, z)`` quaternion for a rotation about one axis."""
    unit = np.zeros(3)
    unit[_AXIS_INDEX[axis]] = 1.0
    return pr.quaternion_from_axis_angle(np.r_[unit, np.deg2rad(deg)])


def euler_quat(seq: str, *angles_deg: float, extrinsic: bool = True) -> Quaternion:
    """Return the MuJoCo ``(w, x, y, z)`` quaternion for an Euler rotation.

    Parameters
    ----------
    seq : str
        Axis sequence, each character one of ``"x"``, ``"y"``, ``"z"`` (e.g.
        ``"z"`` or ``"xyz"``).
    *angles_deg : float
        One angle per axis in ``seq``, in degrees.
    extrinsic : bool, optional
        If True (default), angles are about the fixed world axes; if False,
        about the moving body axes.

    Returns
    -------
    Quaternion
        Scalar-first ``(w, x, y, z)`` quaternion.

    Raises
    ------
    ValueError
        If the number of angles does not match the length of ``seq``.
    """
    if len(seq) != len(angles_deg):
        raise ValueError(f"seq {seq!r} needs {len(seq)} angles, got {len(angles_deg)}")

    quat = np.array([1.0, 0.0, 0.0, 0.0])
    for axis, deg in zip(seq, angles_deg, strict=True):
        step = _axis_quat(axis, deg)
        # Extrinsic rotations pre-multiply (fixed axes); intrinsic post-multiply.
        quat = (
            pr.concatenate_quaternions(step, quat)
            if extrinsic
            else pr.concatenate_quaternions(quat, step)
        )
    return tuple(float(v) for v in quat)


def yaw_quat(deg: float) -> Quaternion:
    """Return the MuJoCo ``(w, x, y, z)`` quaternion for a yaw about +z.

    Parameters
    ----------
    deg : float
        Rotation about the world z-axis, in degrees.

    Returns
    -------
    Quaternion
        Scalar-first ``(w, x, y, z)`` quaternion.
    """
    return euler_quat("z", deg)
