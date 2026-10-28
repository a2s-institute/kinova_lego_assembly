#!/usr/bin/env python3
"""Launch the MuJoCo interactive viewer on a tiny scene of Duplo bricks.

Composes a small worldbody, sharing one ground plane, from the
``assets.mujoco.duplo_brick`` and ``assets.mujoco.lego_holder`` builders:

* The holder scene (at the origin): the holder welded to the world, a 4x2
  brick released above its central studs, and a 2x2 brick released above
  that.
* A brick-on-brick stack (offset in +x, clear of the holder): a 4x2 resting
  flat on the ground with a 2x2 released above it, an isolated check that
  bricks snap onto each other.
* A loose brick pile (offset in -x, clear of the holder): an assortment of
  2x2 and 4x2 bricks lying flat at assorted yaw angles, as they would be
  presented to the robot before assembly.

Only the holder is welded to the world; every brick is a free body, so the
dropped bricks fall, impact, and engage stud-into-cavity while the resting
bricks settle on the table under gravity. Collision geoms live in geom
group 3; press ``3`` in the viewer to toggle their visibility. See
``assets.mujoco.duplo_brick`` for the primitive collision model and ADR-0011.

Controls
--------
Press space to pause / step, drag the mouse to orbit, scroll to zoom,
right-click and drag a body to apply an external force. Press ``3`` to
show or hide the collision pieces.

Run from the repo root::

    python3 scripts/launch_duplo_viewer.py

The script requires a display (the same one MuJoCo Simulate needs) and
will exit gracefully with a helpful message if GLFW cannot find one.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from assets.mujoco.duplo_brick import add_duplo_brick  # noqa: E402
from assets.mujoco.lego_holder import (  # noqa: E402
    REST_HEIGHT_M,
    STUD_TOP_HEIGHT_M,
    add_lego_holder,
)


def build_demo_spec() -> mujoco.MjSpec:
    """Build the demo world (holder stack, brick-on-brick stack, loose pile).

    Returns
    -------
    mujoco.MjSpec
        A spec ready to be compiled with ``spec.compile()``.
    """
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    # Mildly stiffen contacts so a dropped brick does not visibly sink into the
    # part below on impact (default solref ~0.02 s tolerates penetration
    # proportional to impact speed). The primitive cavity has clearance around
    # the studs, so the brick still seats flush across this range (see ADR-0011).
    spec.default.geom.solref = [0.018, 1.0]

    spec.worldbody.add_light(
        name="key",
        pos=(0.2, 0.2, 0.5),
        dir=(-0.2, -0.2, -1.0),
        diffuse=(0.85, 0.85, 0.85),
    )
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.7, 0.5, 0.01),
        rgba=(0.85, 0.85, 0.88, 1.0),
    )

    # Brick height (incl. studs) is 23.75 mm; the aligned mesh is centred
    # on its AABB so each brick spans +/- 11.875 mm around its body origin.
    brick_half_height = 0.011875
    drop_height = 0.015

    # --- Holder scene (origin): welded holder, a 4x2 dropped onto its studs,
    # a 2x2 dropped onto the 4x2. REST_HEIGHT_M rests the plate on the ground;
    # STUD_TOP_HEIGHT_M is where dropped bricks land. ---
    holder_rest_z = REST_HEIGHT_M
    holder_stud_top_z = holder_rest_z + STUD_TOP_HEIGHT_M
    first_brick_pos = (0.0, 0.0, holder_stud_top_z + brick_half_height + drop_height)
    second_brick_pos = (
        0.0,
        0.0,
        first_brick_pos[2] + 2.0 * brick_half_height + drop_height,
    )

    add_lego_holder(spec, name="holder", pos=(0.0, 0.0, holder_rest_z), free=False)
    add_duplo_brick(
        spec,
        brick_type="4x2",
        colour="yellow",
        name="base",
        pos=first_brick_pos,
    )
    add_duplo_brick(
        spec,
        brick_type="2x2",
        colour="red",
        name="topper",
        pos=second_brick_pos,
    )

    # A matching stack one brick-width (-y) below the long side of the first
    # yellow brick, snapped onto the adjacent holder studs so the two bricks
    # sit flush. 4x2 footprint is 32 mm in y, so the centres are 32 mm apart.
    neighbour_y = -0.032
    add_duplo_brick(
        spec,
        brick_type="4x2",
        colour="yellow",
        name="base_below",
        pos=(0.0, neighbour_y, first_brick_pos[2]),
    )
    add_duplo_brick(
        spec,
        brick_type="2x2",
        colour="red",
        name="topper_below",
        pos=(0.0, neighbour_y, second_brick_pos[2]),
    )

    # --- Brick-on-brick stack, offset clear of the holder footprint
    # (~0.19 m): a 4x2 resting flat on the ground with a 2x2 dropped onto it. ---
    stack_x = 0.3
    # Lower brick origin one half-height up puts its underside on z = 0.
    stack_lower_pos = (stack_x, 0.0, brick_half_height)
    stack_lower_stud_top_z = 2.0 * brick_half_height
    # Upper brick nudged 3 mm off centre so the snap visibly self-centres.
    stack_upper_pos = (
        stack_x,
        0.003,
        stack_lower_stud_top_z + brick_half_height + drop_height,
    )
    add_duplo_brick(
        spec,
        brick_type="4x2",
        colour="yellow",
        name="stack_lower",
        pos=stack_lower_pos,
    )
    add_duplo_brick(
        spec,
        brick_type="2x2",
        colour="red",
        name="stack_upper",
        pos=stack_upper_pos,
    )

    # --- Loose brick pile to the left of the holder (-x): an assortment of
    # 2x2 and 4x2 bricks lying flat at assorted yaw angles, as they would be
    # presented to the robot before assembly. Centres are kept clear of the
    # holder footprint (x <= -0.19) and spaced so the footprints do not
    # overlap. Each entry is (brick_type, colour, x, y, yaw_deg). ---
    def _yaw_quat(deg: float) -> tuple[float, float, float, float]:
        half = math.radians(deg) / 2.0
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    loose_bricks = [
        ("2x2", "yellow", -0.50, 0.12, 12.0),
        ("2x2", "green", -0.42, 0.13, -8.0),
        ("4x2", "green", -0.31, 0.13, 18.0),
        ("2x2", "blue", -0.52, -0.02, -10.0),
        ("2x2", "red", -0.44, -0.03, 6.0),
        ("4x2", "yellow", -0.40, -0.15, 85.0),
        ("4x2", "red", -0.30, -0.05, 12.0),
        ("4x2", "blue", -0.29, -0.16, 28.0),
    ]
    for i, (btype, colour, x, y, yaw) in enumerate(loose_bricks):
        add_duplo_brick(
            spec,
            brick_type=btype,
            colour=colour,
            name=f"loose_{i:02d}",
            pos=(x, y, brick_half_height),
            quat=_yaw_quat(yaw),
        )

    return spec


def main() -> int:
    """Compile the demo spec and hand it to ``mujoco.viewer.launch``."""
    spec = build_demo_spec()
    model = spec.compile()
    data = mujoco.MjData(model)

    # Save the compiled spec to XML for later use/inspection
    with open("mjModel.xml", "w") as f:
        f.write(spec.to_xml())

    try:
        import mujoco.viewer as viewer
    except ImportError as exc:
        sys.stderr.write(f"mujoco.viewer is unavailable: {exc}\n")
        return 1

    try:
        viewer.launch(model, data)
    except RuntimeError as exc:
        sys.stderr.write(
            "Failed to launch the MuJoCo viewer. This usually means there is "
            "no display attached (no DISPLAY / Wayland session) or GLFW "
            "cannot initialise. Underlying error:\n"
            f"  {exc}\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
