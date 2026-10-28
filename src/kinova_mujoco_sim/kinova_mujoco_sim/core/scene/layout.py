"""Fixed spatial layout for the Kinova MuJoCo work cell.

These constants describe the parts of the scene that stay put across every
scene variant: the table, the two arm anchors, and the lego holder. Scene
modules (see ``scene.scenes``) only add the movable content (bricks) on top
of this fixed stage, so changing or adding a scene never moves the furniture.

All positions are in metres in the world frame. The work arm faces +x, so
from its point of view +x is forward and +y is to its left.
"""

from __future__ import annotations

from typing import Any

# --- Table (a single world-fixed box, no legs) ---
# Defined by full outer dimensions (x: length, y: width, z: thickness); the
# MuJoCo box half-extents are derived.
TABLE_DIMENSIONS: tuple[float, float, float] = (0.8, 0.8, 0.02)
"""Full table size (length_x, width_y, thickness_z) in metres."""

TABLE_TOP_Z: float = 0.1
"""World height of the table's top surface; everything rests on this plane."""

TABLE_HALF_EXTENTS: tuple[float, float, float] = (
    TABLE_DIMENSIONS[0] / 2.0,
    TABLE_DIMENSIONS[1] / 2.0,
    TABLE_DIMENSIONS[2] / 2.0,
)
"""Half sizes (x, y, z) of the table box, derived from ``TABLE_DIMENSIONS``."""

TABLE_RGBA: tuple[float, float, float, float] = (0.694, 0.718, 0.875, 1.0)
"""Table colour #b1b7df."""

# --- Arm anchors (base_link world pose; both face +x) ---
# Both arms are welded near the back (-x) edge, inset by a fixed margin so the
# bases sit on the table. Deriving x from the edge keeps "arms at the edge"
# true for any table length.
ARM_INSET_FROM_BACK_EDGE_X: float = 0.085
"""Distance the arm bases sit in front of the back (-x) table edge, metres."""

ARM_Y_SEPARATION: float = 0.425
"""Centre-to-centre spacing between the two arms along y, metres."""

ARM_MOUNT_PLATE_THICKNESS_Z: float = 0.025
"""Height of the plate each arm base is bolted to, metres.

In the real cell the arm bases are raised off the table by this much, so the
table plane lies 25 mm below ``base_link`` (see
``icp_pose_refinement.table_plane_z_base_frame_m`` in
``bitbots_lego_assembly/config/params.yml``). Reproducing the same riser here
keeps perception and planning geometry identical in sim and on hardware.
"""

ARM_MOUNT_PLATE_RADIUS: float = 0.07
"""Radius of the arm mount plate, slightly wider than the Gen3 base."""

_ARM_X: float = -TABLE_HALF_EXTENTS[0] + ARM_INSET_FROM_BACK_EDGE_X
_ARM_Z: float = TABLE_TOP_Z + ARM_MOUNT_PLATE_THICKNESS_Z

ARM1_ANCHOR: tuple[float, float, float] = (_ARM_X, ARM_Y_SEPARATION / 2.0, _ARM_Z)
"""Work arm base, near the back (-x) edge, on the +y (left) side."""

ARM2_ANCHOR: tuple[float, float, float] = (_ARM_X, -ARM_Y_SEPARATION / 2.0, _ARM_Z)
"""Obstacle arm base, alongside the work arm on the -y (right) side."""

ARM2_QPOS: tuple[float, ...] = (0.0, -0.7485, 3.1414, -2.0566, 0.0, -1.7792, 1.5708)
"""Static joint configuration of the second arm (obstacle arm)."""

# --- Lego holder (welded, front-centre between the two arms) ---
HOLDER_POS_XY: tuple[float, float] = (0.22, 0.175)
"""Holder centre on the table plane, in front of and between the arms."""

HOLDER_YAW_DEG: float = 0.0
"""Holder rotation about the world z-axis, in degrees."""

# ``HOLDER_QUAT`` is derived from ``HOLDER_YAW_DEG`` via ``yaw_quat``, which pulls
# in ``pytransform3d``/``numpy``. Exposing it through module ``__getattr__`` keeps
# importing this module dependency-free (e.g. the launch file only needs the
# plain anchor constants) and defers the heavy import to first access.

WORK_ORIGIN: tuple[float, float, float] = (HOLDER_POS_XY[0], HOLDER_POS_XY[1], TABLE_TOP_Z)
"""Anchor scenes build around: the holder centre on the table top."""

# --- Physics, matched to scripts/launch_duplo_viewer.py ---
CONTACT_SOLREF: tuple[float, float] = (0.018, 1.0)
"""Stiffened contact solref so dropped bricks do not visibly sink on impact."""


def __getattr__(name: str) -> Any:
    """Lazily compute attributes that need optional heavy dependencies (PEP 562)."""
    if name == "HOLDER_QUAT":
        from kinova_mujoco_sim.core.utils.transforms import yaw_quat

        return yaw_quat(HOLDER_YAW_DEG)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
