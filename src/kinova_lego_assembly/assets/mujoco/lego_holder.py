"""MuJoCo ``MjSpec`` builder for the LEGO holder base plate.

The holder is the 24x16-stud Duplo base plate the assembly is built on. It
is added as a single body (a visual geom referencing the full plate STL
plus a primitive collision set); pass ``free=False`` to weld it in place,
which is the usual case for a fixed base plate.

Collision is primitive: a box for the slab plus one cylinder per stud. The
plate is never stacked under another part, so it has no cavity to engage, and
the only contact that matters is bricks seating onto its studs from above
(see issue #58). Duplo bricks use the same primitive approach (see
``assets.mujoco.duplo_brick``).

Geometry (measured from the STL, in millimetres; already at native Duplo
scale like the bricks, so imported with only the mm-to-m conversion):

- Solid base slab 1.6 mm tall; studs 4.4 mm tall, 4.8 mm radius, 16 mm
  apart, on a 24 x 16 grid; total height 6 mm.
- The STL is centred on its bounding box, so the body origin is the plate's
  centre: the underside is 3 mm below it and the stud tops 3 mm above it.
  The visual geom needs no offset; the slab box and stud cylinders are
  placed relative to that centre.

Source
------
``LEGO_Duplo_base_plate_16x24.stl``: LDraw part 6475
(https://library.ldraw.org/parts/11686).

Usage
-----
Python API::

    import mujoco
    from assets.mujoco.lego_holder import add_lego_holder, REST_HEIGHT_M

    spec = mujoco.MjSpec()
    add_lego_holder(spec, name="holder", pos=(0.0, 0.0, REST_HEIGHT_M), free=False)
    model = spec.compile()

CLI::

    # Compile a holder-only world, print geom counts, render a preview.
    python assets/mujoco/lego_holder.py build

    # Drop a brick onto the central studs and assert it comes to rest.
    python assets/mujoco/lego_holder.py smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import mujoco

_LOG = logging.getLogger("lego_holder")

REPO_ROOT = Path(__file__).resolve().parents[2]
_MESH_DIR = REPO_ROOT / "assets" / "meshes" / "lego_duplo_bricks"
_HOLDER_MESH = _MESH_DIR / "LEGO_Duplo_base_plate_16x24.stl"

# === Source mesh measurements, in millimetres (measured directly from
# LEGO_Duplo_base_plate_16x24.stl). The mesh is already at native Duplo
# scale and centred on its bounding box like the brick meshes, so its origin
# is the plate's geometric centre. ===
_MESH_SLAB_HEIGHT_M = 1.6 * 1e-3  # solid base slab is 1.6 mm tall
_MESH_STUD_HEIGHT_M = 4.4 * 1e-3  # studs stand 4.4 mm above the slab
_MESH_STUD_RADIUS_M = 4.8 * 1e-3  # each stud is 4.8 mm in radius
_MESH_STUD_DISTANCE_M = 16 * 1e-3  # 16 mm between neighbouring stud centres
_STUD_COUNT_X = 24
_STUD_COUNT_Y = 16

# The mesh already puts its studs 16 mm apart, matching the Duplo bricks in
# this package, so it needs no rescaling beyond the millimetre-to-metre
# conversion (the same 1e-3 import scale the bricks use).
_MESH_SCALE = 1e-3  # m per STL unit

# --- Holder geometry in the model, metres (equal to the mesh sizes). ---
_SLAB_HEIGHT_M = _MESH_SLAB_HEIGHT_M  # 1.6 mm
_STUD_HEIGHT_M = _MESH_STUD_HEIGHT_M  # 4.4 mm
_STUD_RADIUS_M = _MESH_STUD_RADIUS_M  # 4.8 mm
_STUD_DISTANCE_M = _MESH_STUD_DISTANCE_M  # 16 mm (== brick)
_SLAB_HALF_HEIGHT_M = _SLAB_HEIGHT_M / 2.0
_STUD_HALF_HEIGHT_M = _STUD_HEIGHT_M / 2.0

# Footprint: each stud sits in one stud-distance cell, so the plate spans
# (stud count) x (stud distance) in each direction.
_PLATE_HALF_X_M = _STUD_COUNT_X * _STUD_DISTANCE_M / 2.0  # 0.192 m
_PLATE_HALF_Y_M = _STUD_COUNT_Y * _STUD_DISTANCE_M / 2.0  # 0.128 m

# Vertical layout with the body origin at the plate centre: the slab fills
# the bottom and the studs the top. The underside and the stud tops are each
# half the total height from the origin.
_HALF_HEIGHT_M = (_SLAB_HEIGHT_M + _STUD_HEIGHT_M) / 2.0  # 3 mm
_SLAB_CENTRE_Z_M = -_HALF_HEIGHT_M + _SLAB_HALF_HEIGHT_M  # -2.2 mm
_STUD_CENTRE_Z_M = _HALF_HEIGHT_M - _STUD_HALF_HEIGHT_M  # +0.8 mm

# --- Public placement heights (metres). Both equal half the plate height,
# because the centred mesh puts the underside and the stud tops the same
# distance from the body origin. ---
REST_HEIGHT_M = _HALF_HEIGHT_M
"""Place the holder body at this z so the plate's underside rests flat on a
ground plane at ``z = 0``."""

STUD_TOP_HEIGHT_M = _HALF_HEIGHT_M
"""Height of the stud tops above the body origin; add the holder body's
world z to get the world height an object placed on the studs sits at."""

_HOLDER_RGBA: tuple[float, float, float, float] = (0.25, 0.62, 0.20, 1.0)
"""Bright Duplo green for the holder visual mesh."""

ABS_DENSITY_KG_M3 = 1040.0
"""ABS plastic density; used so the (welded) body still has a sane mass."""

_DEFAULT_FRICTION: tuple[float, float, float] = (0.5, 0.005, 0.0001)
"""Slide / spin / roll friction, matching the engaged Duplo snap-fit value
used for the bricks (see ADR-0011)."""


def _stud_centres_m() -> list[tuple[float, float]]:
    """Return the (x, y) centre of every top stud in the body frame.

    Returns
    -------
    list of (float, float)
        ``_STUD_COUNT_X * _STUD_COUNT_Y`` centres in metres, laid out on a
        regular grid centred on the body origin.
    """
    cx = [(i - (_STUD_COUNT_X - 1) / 2.0) * _STUD_DISTANCE_M for i in range(_STUD_COUNT_X)]
    cy = [(j - (_STUD_COUNT_Y - 1) / 2.0) * _STUD_DISTANCE_M for j in range(_STUD_COUNT_Y)]
    return [(x, y) for x in cx for y in cy]


def _ensure_visual_mesh(spec: mujoco.MjSpec) -> str:
    """Register the holder's visual mesh on ``spec`` once and return its name."""
    name = "lego_holder_visual"
    if not any(m.name == name for m in spec.meshes):
        spec.add_mesh(
            name=name,
            file=str(_HOLDER_MESH),
            scale=(_MESH_SCALE, _MESH_SCALE, _MESH_SCALE),
        )
    return name


def _ensure_material(spec: mujoco.MjSpec) -> str:
    """Register the holder's material on ``spec`` once and return its name."""
    name = "lego_holder_green"
    if not any(mat.name == name for mat in spec.materials):
        spec.add_material(name=name, rgba=_HOLDER_RGBA)
    return name


def add_lego_holder(
    spec: mujoco.MjSpec,
    *,
    name: str = "holder",
    parent: mujoco._specs.MjsBody | None = None,
    pos: Sequence[float] = (0.0, 0.0, 0.0),
    quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    free: bool = True,
    friction: Sequence[float] | None = None,
) -> mujoco._specs.MjsBody:
    """Add the LEGO holder base plate as a body on ``spec``.

    The visual mesh and material are registered on ``spec`` lazily, so
    calling this more than once does not duplicate assets.

    Parameters
    ----------
    spec : mujoco.MjSpec
        The model spec being assembled.
    name : str, optional
        Body name (also used to derive geom names). Defaults to ``"holder"``.
    parent : mujoco._specs.MjsBody, optional
        Parent body. Defaults to ``spec.worldbody``.
    pos : sequence of 3 floats, optional
        Body position in metres relative to the parent.
    quat : sequence of 4 floats, optional
        Body orientation as a quaternion ``(w, x, y, z)``.
    free : bool, optional
        If True (default, matching ``add_duplo_brick``), a free joint is
        added and the slab geom's density gives the body its mass. Pass
        ``free=False`` to weld the holder to its parent (no joint), which is
        the usual case for a fixed base plate.
    friction : sequence of 3 floats, optional
        Override the default ``(slide, spin, roll)`` friction.

    Returns
    -------
    mujoco._specs.MjsBody
        The newly created holder body.
    """
    parent_body = parent if parent is not None else spec.worldbody
    visual_mesh = _ensure_visual_mesh(spec)
    mat_name = _ensure_material(spec)
    fric = list(friction) if friction is not None else list(_DEFAULT_FRICTION)

    body = parent_body.add_body(name=name, pos=list(pos), quat=list(quat))
    if free:
        body.add_freejoint()

    # Visual-only geom: the full plate STL with the green material, kept out
    # of contact (contype/conaffinity = 0). The mesh is centred on its own
    # origin, so it needs no offset. Collision is delegated to the primitive
    # geoms below.
    body.add_geom(
        name=f"{name}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=visual_mesh,
        material=mat_name,
        contype=0,
        conaffinity=0,
        group=0,
    )

    # Slab collision box in the bottom of the plate. Carries the body's mass
    # (so a free holder is well-formed); studs above are massless to avoid
    # double counting.
    body.add_geom(
        name=f"{name}_slab",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(_PLATE_HALF_X_M, _PLATE_HALF_Y_M, _SLAB_HALF_HEIGHT_M),
        pos=(0.0, 0.0, _SLAB_CENTRE_Z_M),
        density=ABS_DENSITY_KG_M3,
        friction=fric,
        contype=1,
        conaffinity=1,
        group=3,
        rgba=(0.4, 0.4, 0.4, 0.4),
    )

    # One cylinder per top stud, in geom group 3 (hidden by default; press
    # '3' in the viewer to toggle).
    for idx, (sx, sy) in enumerate(_stud_centres_m()):
        body.add_geom(
            name=f"{name}_stud_{idx:03d}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(_STUD_RADIUS_M, _STUD_HALF_HEIGHT_M, 0.0),
            pos=(sx, sy, _STUD_CENTRE_Z_M),
            density=0.0,
            friction=fric,
            contype=1,
            conaffinity=1,
            group=3,
            rgba=(0.4, 0.4, 0.4, 0.4),
        )

    return body


def _cmd_build(image_output: Path | None) -> int:
    """Compile a holder-only world, report geom counts, optionally render."""
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.worldbody.add_light(
        name="key",
        pos=(0.3, 0.3, 0.6),
        dir=(-0.3, -0.3, -1.0),
        diffuse=(0.85, 0.85, 0.85),
    )
    add_lego_holder(spec, name="holder", pos=(0.0, 0.0, 0.0), free=False)
    model = spec.compile()

    n_studs = _STUD_COUNT_X * _STUD_COUNT_Y
    _LOG.info(
        "holder compiled: %d geoms (1 visual + 1 slab + %d studs), %d studs expected",
        model.ngeom,
        n_studs,
        n_studs,
    )

    if image_output is not None:
        _render_preview(model, image_output)
    return 0


def _cmd_smoke(steps_seconds: float) -> int:
    """Drop a brick onto the central studs and assert it comes to rest.

    Parameters
    ----------
    steps_seconds : float
        Simulated time to run before checking the resting state.

    Returns
    -------
    int
        ``0`` if the brick settles on the studs (above the slab top, low
        vertical velocity, in contact), ``1`` otherwise.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from assets.mujoco.duplo_brick import add_duplo_brick

    spec = mujoco.MjSpec()
    add_lego_holder(spec, name="holder", pos=(0.0, 0.0, 0.0), free=False)
    # Holder body at z = 0, so its stud tops are STUD_TOP_HEIGHT_M up. Drop a
    # 2x2 brick from ~3 cm above the studs, over the plate centre.
    stud_top_z = STUD_TOP_HEIGHT_M
    brick_half_h = 0.011875
    drop_z = stud_top_z + 0.03 + brick_half_h
    add_duplo_brick(
        spec,
        brick_type="2x2",
        colour="red",
        name="topper",
        pos=(0.0, 0.0, drop_z),
        free=True,
    )
    model = spec.compile()
    data = mujoco.MjData(model)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "topper")
    n_steps = int(steps_seconds / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    z = float(data.xpos[body_id, 2])
    vz = float(data.cvel[body_id, 5])
    ncon = data.ncon
    rested = (z > stud_top_z - brick_half_h) and abs(vz) < 1e-3 and ncon > 0
    _LOG.info(
        "after %.2fs: topper z=%.4f m (stud top %.4f), vz=%.2e m/s, ncon=%d -> %s",
        steps_seconds,
        z,
        stud_top_z,
        vz,
        ncon,
        "RESTING" if rested else "NOT RESTING",
    )
    return 0 if rested else 1


def _render_preview(model: mujoco.MjModel, image_path: Path) -> None:
    """Render an offscreen RGB still of the compiled model and save as PNG."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = (0.0, 0.0, 0.0)
    cam.distance = 0.6
    cam.azimuth = 135.0
    cam.elevation = -35.0

    with mujoco.Renderer(model, height=720, width=1280) as renderer:
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()

    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required for offscreen preview (uv pip install pillow)."
        ) from exc

    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(image_path)
    _LOG.info("wrote preview %s (%d bytes)", image_path, image_path.stat().st_size)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Dispatches to the ``build`` or ``smoke`` subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="LEGO holder base-plate MjSpec builder + smoke test CLI."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="compile a holder-only world and report geoms")
    b.add_argument(
        "--image",
        type=Path,
        default=Path("out") / "holder.png",
        help="path for the rendered preview PNG (default: out/holder.png)",
    )
    b.add_argument("--no-image", action="store_true", help="skip the offscreen render")
    s = sub.add_parser("smoke", help="drop a brick on the studs and assert it rests")
    s.add_argument(
        "--seconds", type=float, default=2.0, help="simulated settle time (default: 2.0)"
    )
    args = parser.parse_args(argv)

    if args.cmd == "build":
        return _cmd_build(None if args.no_image else args.image)
    if args.cmd == "smoke":
        return _cmd_smoke(args.seconds)
    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
