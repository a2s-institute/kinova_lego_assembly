"""Holder stack plus a loose brick pile.

* The welded holder (added by the stage) gets a 4x2 brick dropped onto its
  central studs and a 2x2 dropped onto that, with a matching neighbour stack
  one brick-width over, so the snap-fit settles under gravity.
* A loose assortment of 2x2 and 4x2 bricks lies flat at assorted yaws in front
  of and to the left of the holder, as they would be presented to the work arm
  before assembly.

Everything but the (stage-welded) holder is a free body resting on / dropped
onto the table top at ``ctx.table_top_z``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco

from kinova_mujoco_sim.core.scene.scenes.base import register
from kinova_mujoco_sim.core.scene.scenes.loose_pile import LoosePileScene
from kinova_mujoco_sim.core.scene.stage import StageContext

_LEGO_PKG_ROOT = Path(__file__).resolve().parents[5] / "kinova_lego_assembly"
if str(_LEGO_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGO_PKG_ROOT))

# Brick half-height is a property of the brick mesh, owned by the asset module.
from assets.mujoco.duplo_brick import (  # noqa: E402
    _BRICK_HALF_HEIGHT_M,
    add_duplo_brick,
)

_DROP_HEIGHT_M = 0.015


class LoosePileHolderStackScene:
    """Holder stack plus a loose brick pile."""

    name = "loose_pile_holder_stack"

    def populate(self, spec: mujoco.MjSpec, ctx: StageContext) -> None:
        """Add the holder stack and loose pile to ``spec``."""
        self._add_holder_stack(spec, ctx)
        LoosePileScene()._add_loose_pile(spec, ctx)

    @staticmethod
    def _add_holder_stack(spec: mujoco.MjSpec, ctx: StageContext) -> None:
        """Drop a 4x2 then a 2x2 onto the holder studs, plus a neighbour stack."""
        hx, hy, _ = ctx.holder_pos
        first_z = ctx.holder_stud_top_z + _BRICK_HALF_HEIGHT_M + _DROP_HEIGHT_M
        second_z = first_z + 2.0 * _BRICK_HALF_HEIGHT_M + _DROP_HEIGHT_M

        add_duplo_brick(spec, brick_type="4x2", colour="yellow", name="base", pos=(hx, hy, first_z))
        add_duplo_brick(spec, brick_type="2x2", colour="red", name="topper", pos=(hx, hy, second_z))

        # Matching stack one 4x2 footprint (32 mm in y) over, snapped onto the
        # adjacent holder studs so the two stacks sit flush.
        neighbour_y = hy - 0.032
        add_duplo_brick(
            spec,
            brick_type="4x2",
            colour="yellow",
            name="base_below",
            pos=(hx, neighbour_y, first_z),
        )
        add_duplo_brick(
            spec,
            brick_type="2x2",
            colour="red",
            name="topper_below",
            pos=(hx, neighbour_y, second_z),
        )


register(LoosePileHolderStackScene())
