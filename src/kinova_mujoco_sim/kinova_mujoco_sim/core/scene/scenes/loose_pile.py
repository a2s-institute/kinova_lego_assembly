"""A loose brick pile.

A loose assortment of 2x2 and 4x2 bricks lies flat at assorted yaws in front
of and to the left of the holder, as they would be presented to the work arm
before assembly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco

from kinova_mujoco_sim.core.scene.scenes.base import register
from kinova_mujoco_sim.core.scene.stage import StageContext
from kinova_mujoco_sim.core.utils.transforms import yaw_quat

_LEGO_PKG_ROOT = Path(__file__).resolve().parents[5] / "kinova_lego_assembly"
if str(_LEGO_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGO_PKG_ROOT))

# Brick half-height is a property of the brick mesh, owned by the asset module.
from assets.mujoco.duplo_brick import (  # noqa: E402
    _BRICK_HALF_HEIGHT_M,
    add_duplo_brick,
)

# Loose pile, as (brick_type, colour, dx, dy, yaw_deg) offsets from the pile
# centre. Spread keeps the footprints clear of each other and of the holder.
_PILE_OFFSET_FROM_WORK_ORIGIN = (-0.15, -0.38)
_LOOSE_BRICKS = [
    ("2x2", "yellow", -0.05, 0.06, 12.0),
    ("2x2", "green", 0.03, 0.07, -8.0),
    ("4x2", "green", 0.10, 0.05, 18.0),
    ("2x2", "blue", -0.06, -0.04, -10.0),
    ("2x2", "red", 0.02, -0.05, 6.0),
    ("4x2", "yellow", 0.00, -0.12, 85.0),
    ("4x2", "red", 0.10, -0.03, 12.0),
    ("4x2", "blue", 0.11, -0.13, 28.0),
]


class LoosePileScene:
    """A loose brick pile."""

    name = "loose_pile"

    def populate(self, spec: mujoco.MjSpec, ctx: StageContext) -> None:
        """Add the loose pile to ``spec``."""
        self._add_loose_pile(spec, ctx)

    @staticmethod
    def _add_loose_pile(spec: mujoco.MjSpec, ctx: StageContext) -> None:
        """Lay an assortment of bricks flat in front-left of the holder."""
        wx, wy, _ = ctx.work_origin
        cx = wx + _PILE_OFFSET_FROM_WORK_ORIGIN[0]
        cy = wy + _PILE_OFFSET_FROM_WORK_ORIGIN[1]
        rest_z = ctx.table_top_z + _BRICK_HALF_HEIGHT_M
        for i, (btype, colour, dx, dy, yaw) in enumerate(_LOOSE_BRICKS):
            add_duplo_brick(
                spec,
                brick_type=btype,
                colour=colour,
                name=f"loose_{i:02d}",
                pos=(cx + dx, cy + dy, rest_z),
                quat=yaw_quat(yaw),
            )


register(LoosePileScene())
