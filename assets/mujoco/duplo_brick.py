"""MuJoCo ``MjSpec`` builder for Duplo bricks.

``add_duplo_brick`` turns an imported Duplo STL into a colour-parameterised
MuJoCo body. Mesh and material assets are registered on the ``MjSpec`` once
per ``brick_type`` / ``colour``, so adding many bricks does not duplicate
assets.

Each brick body carries two geom sets:

* One visual geom referencing the full STL with the colour material
  (``contype=0``), which also carries the body's mass and inertia. Its mass
  is set explicitly to the BrickLink-measured value; the meshes are
  open-bottomed shells with no closed volume, so inertia is integrated over
  the convex hull (``mjMESH_INERTIA_CONVEX``).
* Collision geoms (``contype=1``, ``density=0``, group 3, hidden in the
  viewer) built entirely from primitives: four boxes for the outer walls, a
  box for the top plate, one cylinder per underside tube, and one cylinder
  per top stud. The walls and tubes reproduce the hollow underside so a lower
  brick's studs are captured between the tubes and the walls (a self-centring
  snap-fit); the plate gives the flat top that a stacked brick's rim seats on.

All collision dimensions are measured from the aligned meshes (see the
constants below) and laid out from each brick's stud grid, so no convex
decomposition is needed. See ADR-0011 for the rationale.

References
----------
- BrickLink part #3437 (Duplo 2x2): measured mass 6.75 g.
- BrickLink part #3011 (Duplo 4x2): measured mass 12.06 to 12.20 g; we
  use the midpoint 12.13 g.
- Meshes: LDraw parts 3437 (Duplo 2x2,
  https://library.ldraw.org/parts/6083) and 3011 (Duplo 4x2,
  https://library.ldraw.org/parts/4001), licensed CC BY 4.0.

Usage
-----
Python API::

    import mujoco
    from assets.mujoco.duplo_brick import add_duplo_brick

    spec = mujoco.MjSpec()
    add_duplo_brick(spec, brick_type="2x2", colour="red", name="brick_1")
    add_duplo_brick(spec, brick_type="4x2", colour="blue", name="brick_2",
                    pos=(0.05, 0.0, 0.0))
    model = spec.compile()

CLI::

    # Print volume / density / mass cross-check table.
    python assets/mujoco/duplo_brick.py measure

    # Compile a world with one body per (brick_type, colour), verify
    # the MuJoCo-computed body mass matches BrickLink within 1 percent,
    # and render an offscreen preview to out/duplo_world.png.
    python assets/mujoco/duplo_brick.py build-brick
"""

from __future__ import annotations

import argparse
import colorsys
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco

_LOG = logging.getLogger("duplo_brick")

REPO_ROOT = Path(__file__).resolve().parents[2]
_MESH_DIR = REPO_ROOT / "assets" / "meshes" / "lego_duplo_bricks"

_DEFAULT_FRICTION: tuple[float, float, float] = (0.5, 0.005, 0.0001)
"""MuJoCo geom friction (slide, spin, roll) for engaged Duplo snap-fits.

A rigid-body sim has no elastic interference fit, so the collision pieces
give only a weak normal force and the literature ABS-on-ABS sliding value
(mu = 0.2) is too low to hold a stacked brick against gravity. mu = 0.5
(within the 0.3 to 0.5 static range) keeps it engaged. Spin / roll are
MuJoCo defaults. Override per call via ``friction=`` for bare-sliding
scenarios. See ADR-0011.
"""


# Collision geometry in metres, measured from the aligned LDraw meshes (body
# frame: origin at the mesh AABB centre, studs pointing +z). The 2x2 and 4x2
# share the Duplo vertical profile; only the footprint and the stud / tube
# grid differ, both derived from the stud counts (see BrickSpec). Studs and
# underside tubes are cylinders; the outer walls and top plate are boxes.
_BRICK_HALF_HEIGHT_M = 11.8e-3  # mesh is 23.6 mm tall, centred on its AABB
_SHOULDER_Z_M = 7.4e-3  # flat body top; a stacked brick's rim seats here
_CAVITY_CEILING_Z_M = 5.8e-3  # underside of the top plate (cavity roof)
_RIM_Z_M = -_BRICK_HALF_HEIGHT_M  # underside rim where the cavity opens

_WALL_THICKNESS_M = 1.6e-3  # outer wall thickness
_GRID_PITCH_M = 16e-3  # 16 mm between neighbouring studs / tubes

_STUD_HEIGHT_M = 4.4e-3  # studs stand 4.4 mm above the shoulder
_STUD_RADIUS_M = 4.8e-3  # stud radius (matches the holder studs)
_TUBE_RADIUS_M = 6.4e-3  # underside tube; a stud grips between it and the wall

_STUD_HALF_HEIGHT_M = _STUD_HEIGHT_M / 2.0
_STUD_CENTRE_Z_M = _SHOULDER_Z_M + _STUD_HALF_HEIGHT_M


@dataclass(frozen=True)
class BrickSpec:
    """Per-brick metadata for the Duplo bricks we model.

    Parameters
    ----------
    mesh_file : pathlib.Path
        Absolute path to the aligned STL mesh.
    measured_mass_g : float
        Reference mass in grams; set as the body's explicit mass.
    mass_source : str
        Citation for ``measured_mass_g`` (used by the ``measure`` CLI).
    stud_cols, stud_rows : int
        Number of top studs along x and y. The whole collision model is laid
        out from this grid (``_GRID_PITCH_M`` apart, centred on the body): the
        footprint is ``cols``/``rows`` x pitch, the studs sit on the grid
        nodes, and the underside tubes sit at the interior cell centres.
    """

    mesh_file: Path
    measured_mass_g: float
    mass_source: str
    stud_cols: int
    stud_rows: int


_BRICKS: dict[str, BrickSpec] = {
    "2x2": BrickSpec(
        mesh_file=_MESH_DIR / "LEGO_Duplo_brick_2x2.stl",
        measured_mass_g=6.75,
        mass_source="BrickLink part #3437",
        stud_cols=2,
        stud_rows=2,
    ),
    "4x2": BrickSpec(
        mesh_file=_MESH_DIR / "LEGO_Duplo_brick_4x2.stl",
        measured_mass_g=12.13,
        mass_source="BrickLink part #3011 (12.06-12.20 g, midpoint)",
        stud_cols=4,
        stud_rows=2,
    ),
}


def _stud_centres_m(cols: int, rows: int) -> list[tuple[float, float]]:
    """Return the (x, y) centre of every top stud in the body frame.

    Parameters
    ----------
    cols, rows : int
        Stud counts along x and y.

    Returns
    -------
    list of (float, float)
        ``cols * rows`` centres in metres on a regular ``_GRID_PITCH_M``
        grid centred on the body origin.
    """
    cx = [(i - (cols - 1) / 2.0) * _GRID_PITCH_M for i in range(cols)]
    cy = [(j - (rows - 1) / 2.0) * _GRID_PITCH_M for j in range(rows)]
    return [(x, y) for x in cx for y in cy]


def _tube_centres_m(cols: int, rows: int) -> list[tuple[float, float]]:
    """Return the (x, y) centre of every underside tube in the body frame.

    Duplo tubes sit at the interior nodes of the stud grid (the midpoints
    between adjacent stud columns and rows), giving a ``(cols - 1)`` x
    ``(rows - 1)`` grid. A lower brick's stud drops into the pocket bounded by
    four such tubes (or a tube and the walls), which is what self-centres the
    snap-fit.

    Parameters
    ----------
    cols, rows : int
        Stud counts along x and y.

    Returns
    -------
    list of (float, float)
        ``(cols - 1) * (rows - 1)`` centres in metres; empty for a 1-wide
        brick, which has no interior pocket.
    """
    tx = [(i - (cols - 2) / 2.0) * _GRID_PITCH_M for i in range(cols - 1)]
    ty = [(j - (rows - 2) / 2.0) * _GRID_PITCH_M for j in range(rows - 1)]
    return [(x, y) for x in tx for y in ty]


def _collision_primitives(brick: BrickSpec) -> list[tuple[str, dict]]:
    """Return the brick's collision primitives as (name suffix, geom kwargs).

    Each entry's kwargs hold the ``type``, ``size`` and ``pos`` of one geom in
    the body frame; the caller supplies the contact / appearance attributes
    (``group``, ``rgba``, ``contype``, ...). The set is four wall boxes, one
    plate box, one cylinder per underside tube, and one cylinder per top stud,
    all laid out from the brick's stud grid (see the module constants). Shared
    by ``add_duplo_brick`` and the ``debug-collision`` preview so both build
    the exact same shapes.

    Parameters
    ----------
    brick : BrickSpec
        The brick whose stud grid drives the layout.

    Returns
    -------
    list of (str, dict)
        ``(suffix, geom_kwargs)`` pairs, in wall / plate / tube / stud order.
    """
    box = mujoco.mjtGeom.mjGEOM_BOX
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
    half_x = brick.stud_cols * _GRID_PITCH_M / 2.0
    half_y = brick.stud_rows * _GRID_PITCH_M / 2.0
    t_half = _WALL_THICKNESS_M / 2.0
    wall_z = (_SHOULDER_Z_M + _RIM_Z_M) / 2.0
    wall_half_h = (_SHOULDER_Z_M - _RIM_Z_M) / 2.0

    # Outer walls: a rectangular ring of four boxes from the underside rim up
    # to the shoulder. The +/-x walls are inset by the wall thickness in y so
    # they butt against the +/-y walls instead of overlapping at the corners.
    prims: list[tuple[str, dict]] = [
        (
            "wall_yp",
            dict(type=box, size=(half_x, t_half, wall_half_h), pos=(0.0, half_y - t_half, wall_z)),
        ),
        (
            "wall_yn",
            dict(
                type=box, size=(half_x, t_half, wall_half_h), pos=(0.0, -(half_y - t_half), wall_z)
            ),
        ),
        (
            "wall_xp",
            dict(
                type=box,
                size=(t_half, half_y - _WALL_THICKNESS_M, wall_half_h),
                pos=(half_x - t_half, 0.0, wall_z),
            ),
        ),
        (
            "wall_xn",
            dict(
                type=box,
                size=(t_half, half_y - _WALL_THICKNESS_M, wall_half_h),
                pos=(-(half_x - t_half), 0.0, wall_z),
            ),
        ),
        # Top plate: the solid body roof; its top face is the shoulder a
        # stacked brick's rim seats on.
        (
            "plate",
            dict(
                type=box,
                size=(half_x, half_y, (_SHOULDER_Z_M - _CAVITY_CEILING_Z_M) / 2.0),
                pos=(0.0, 0.0, (_SHOULDER_Z_M + _CAVITY_CEILING_Z_M) / 2.0),
            ),
        ),
    ]

    # Underside tubes: cylinders from the rim up to the cavity ceiling. A lower
    # brick's studs wedge between these and the walls, which self-centres the snap.
    tube_z = (_RIM_Z_M + _CAVITY_CEILING_Z_M) / 2.0
    tube_half_h = (_CAVITY_CEILING_Z_M - _RIM_Z_M) / 2.0
    for idx, (tx, ty) in enumerate(_tube_centres_m(brick.stud_cols, brick.stud_rows)):
        prims.append(
            (
                f"tube_{idx:02d}",
                dict(type=cyl, size=(_TUBE_RADIUS_M, tube_half_h, 0.0), pos=(tx, ty, tube_z)),
            )
        )

    # Top studs: cylinders on the stud grid, the mating partner for a cavity above.
    for idx, (sx, sy) in enumerate(_stud_centres_m(brick.stud_cols, brick.stud_rows)):
        prims.append(
            (
                f"stud_{idx:02d}",
                dict(
                    type=cyl,
                    size=(_STUD_RADIUS_M, _STUD_HALF_HEIGHT_M, 0.0),
                    pos=(sx, sy, _STUD_CENTRE_Z_M),
                ),
            )
        )
    return prims


def _piece_colour(index: int, total: int) -> tuple[float, float, float, float]:
    """Return a distinct opaque RGBA for the ``index``-th of ``total`` pieces.

    Hues are spread evenly around the colour wheel so neighbouring primitives
    are easy to tell apart in the viewer.
    """
    hue = (index / max(total, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return (r, g, b, 1.0)


_COLOURS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.83, 0.18, 0.18, 1.0),
    "green": (0.20, 0.60, 0.30, 1.0),
    "blue": (0.10, 0.35, 0.70, 1.0),
    "yellow": (0.95, 0.80, 0.10, 1.0),
}


def _ensure_visual_mesh(spec: mujoco.MjSpec, brick_type: str) -> str:
    """Register the brick's visual mesh on ``spec`` once and return its name.

    The mesh asset is created with ``inertia=mjMESH_INERTIA_CONVEX`` so the
    compiler integrates the inertia tensor over the mesh's convex hull. The
    manufacturer meshes are open-bottomed shells with no closed-mesh
    volume, for which the exact and legacy integrators are ill-defined; the
    convex hull is always well-defined. The body mass is pinned separately
    by the geom's explicit ``mass`` (see ``add_duplo_brick``), so the hull
    only sets how that fixed mass is distributed.
    """
    name = f"duplo_{brick_type}_visual"
    if not any(m.name == name for m in spec.meshes):
        m = spec.add_mesh(
            name=name,
            file=str(_BRICKS[brick_type].mesh_file),
            scale=(0.001, 0.001, 0.001),
        )
        m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
    return name


def _ensure_material(spec: mujoco.MjSpec, colour: str) -> str:
    """Register the colour's material on ``spec`` once and return its name."""
    name = f"duplo_{colour}"
    if not any(mat.name == name for mat in spec.materials):
        spec.add_material(
            name=name,
            rgba=_COLOURS[colour],
            specular=0.1,
            shininess=0.1,
            reflectance=0.1,
        )
    return name


def add_duplo_brick(
    spec: mujoco.MjSpec,
    *,
    brick_type: str,
    colour: str,
    name: str,
    parent: mujoco._specs.MjsBody | None = None,
    pos: Sequence[float] = (0.0, 0.0, 0.0),
    quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    free: bool = True,
    friction: Sequence[float] | None = None,
) -> mujoco._specs.MjsBody:
    """Add a single Duplo brick body to ``spec``.

    The mesh and material assets are registered on ``spec`` lazily, so
    calling this function multiple times with the same ``brick_type`` /
    ``colour`` does not duplicate assets.

    Parameters
    ----------
    spec : mujoco.MjSpec
        The model spec being assembled.
    brick_type : {"2x2", "4x2"}
        Which Duplo brick to add.
    colour : {"red", "green", "blue", "yellow"}
        Colour of the brick.
    name : str
        Unique body name (also used to derive the geom name).
    parent : mujoco._specs.MjsBody, optional
        Parent body. Defaults to ``spec.worldbody``.
    pos : sequence of 3 floats, optional
        Body position in metres relative to the parent.
    quat : sequence of 4 floats, optional
        Body orientation as a quaternion ``(w, x, y, z)``.
    free : bool, optional
        If True (default), add a free joint so the brick is a free body.
    friction : sequence of 3 floats, optional
        Override the default ``(slide, spin, roll)`` friction.

    Returns
    -------
    mujoco._specs.MjsBody
        The newly created body, for any further customisation (sites,
        sensors, child geoms, ...).

    Raises
    ------
    ValueError
        If ``brick_type`` or ``colour`` is not in the known set.
    """
    if brick_type not in _BRICKS:
        raise ValueError(f"unknown brick_type {brick_type!r}; choose from {sorted(_BRICKS)}")
    if colour not in _COLOURS:
        raise ValueError(f"unknown colour {colour!r}; choose from {sorted(_COLOURS)}")

    parent_body = parent if parent is not None else spec.worldbody
    visual_mesh = _ensure_visual_mesh(spec, brick_type)
    mat_name = _ensure_material(spec, colour)
    fric = list(friction) if friction is not None else list(_DEFAULT_FRICTION)
    brick = _BRICKS[brick_type]
    mass_kg = brick.measured_mass_g / 1000.0

    body = parent_body.add_body(name=name, pos=list(pos), quat=list(quat))
    if free:
        body.add_freejoint()

    # Visual-only geom (contype = 0): carries the colour material and the
    # body's explicit mass; inertia is distributed over its convex hull.
    body.add_geom(
        name=f"{name}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=visual_mesh,
        material=mat_name,
        mass=mass_kg,
        contype=0,
        conaffinity=0,
        group=0,
    )

    # Massless (density = 0) collision primitives, hidden in group 3. The mass
    # rides on the visual geom above, so these only shape contacts. See
    # _collision_primitives for the wall / plate / tube / stud layout.
    for suffix, geom in _collision_primitives(brick):
        body.add_geom(
            name=f"{name}_{suffix}",
            density=0.0,
            friction=fric,
            contype=1,
            conaffinity=1,
            group=3,
            rgba=(0.4, 0.4, 0.4, 0.4),
            **geom,
        )

    return body


def _cmd_measure() -> int:
    header = "| brick    | mesh file                 | mass g | source |"
    sep = (
        "|"
        + "|".join(
            "-" * (len(c) + 2)
            for c in [
                "brick   ",
                "mesh file                ",
                "mass g",
                "source",
            ]
        )
        + "|"
    )
    print(header)
    print(sep)
    for brick_type, brick in _BRICKS.items():
        print(
            f"| {brick_type:8s} | {brick.mesh_file.name:25s} |"
            f" {brick.measured_mass_g:6.2f} | {brick.mass_source} |"
        )
    return 0


def _cmd_build_brick(output: Path, image_output: Path | None) -> int:
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.worldbody.add_light(
        name="key",
        pos=(0.3, 0.3, 0.5),
        dir=(-0.3, -0.3, -1.0),
        diffuse=(0.8, 0.8, 0.8),
    )
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.5, 0.5, 0.01),
        rgba=(0.85, 0.85, 0.88, 1.0),
    )

    grid_step = 0.05
    for i, brick_type in enumerate(_BRICKS):
        for j, colour in enumerate(_COLOURS):
            add_duplo_brick(
                spec,
                brick_type=brick_type,
                colour=colour,
                name=f"{brick_type}_{colour}",
                pos=(i * grid_step, j * grid_step, 0.0125),
                free=True,
            )
    model = spec.compile()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(spec.to_xml())
    _LOG.info("wrote %s (%d bytes)", output, output.stat().st_size)

    failures: list[str] = []
    for brick_type, brick in _BRICKS.items():
        if brick.measured_mass_g is None:
            continue
        for colour in _COLOURS:
            body_name = f"{brick_type}_{colour}"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            mass_g = float(model.body_mass[body_id]) * 1000.0
            diff_pct = abs(mass_g - brick.measured_mass_g) / brick.measured_mass_g * 100.0
            status = "ok" if diff_pct < 1.0 else "FAIL"
            _LOG.info(
                "%s: model=%.3f g  ref=%.3f g  diff=%.3f %% (%s)",
                body_name,
                mass_g,
                brick.measured_mass_g,
                diff_pct,
                status,
            )
            if diff_pct >= 1.0:
                failures.append(body_name)

    if image_output is not None:
        _render_preview(model, image_output)

    if failures:
        _LOG.error("body_mass deviates by >=1%% from reference for: %s", failures)
        return 1
    return 0


def _render_preview(
    model: mujoco.MjModel,
    image_path: Path,
    *,
    lookat: Sequence[float] = (0.05, 0.075, 0.012),
    distance: float = 0.35,
    azimuth: float = 135.0,
    elevation: float = -28.0,
) -> None:
    """Render an offscreen RGB still of the compiled model and save as PNG."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

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


def _cmd_debug_collision(brick_type: str, image_output: Path | None) -> int:
    """Visualise one brick's collision primitives, each in a distinct colour.

    Builds only the collision primitives (no visual mesh) so every wall,
    plate, tube and stud is individually visible, each in its own colour. With
    ``image_output`` set it renders an offscreen still; otherwise it launches
    the interactive viewer (which needs a display).
    """
    if brick_type not in _BRICKS:
        raise SystemExit(f"unknown brick_type {brick_type!r}; choose from {sorted(_BRICKS)}")
    brick = _BRICKS[brick_type]
    prims = _collision_primitives(brick)

    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.worldbody.add_light(
        name="key",
        pos=(0.05, 0.05, 0.2),
        dir=(-0.05, -0.05, -1.0),
        diffuse=(0.85, 0.85, 0.85),
    )
    body = spec.worldbody.add_body(name=brick_type)
    for i, (suffix, geom) in enumerate(prims):
        # Display-only geoms (no contact); each primitive gets its own colour.
        body.add_geom(
            name=f"{brick_type}_{suffix}",
            contype=0,
            conaffinity=0,
            group=0,
            rgba=_piece_colour(i, len(prims)),
            **geom,
        )
    model = spec.compile()
    _LOG.info("%s: %d collision primitives", brick_type, len(prims))

    if image_output is not None:
        _render_preview(
            model,
            image_output,
            lookat=(0.0, 0.0, 0.004),
            distance=0.11,
            azimuth=130.0,
            elevation=-20.0,
        )
        return 0

    data = mujoco.MjData(model)
    try:
        import mujoco.viewer as viewer
    except ImportError as exc:
        _LOG.error("MuJoCo viewer unavailable (%s); pass --image to render offscreen", exc)
        return 1
    _LOG.info("previewing in the viewer; close the window to exit")
    try:
        viewer.launch(model, data)
    except RuntimeError as exc:
        _LOG.error(
            "could not launch the viewer (no display?): %s; pass --image to render offscreen", exc
        )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Dispatches to the ``measure`` or ``build-world`` subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Duplo brick MjSpec builder + measurement / smoke-test CLI."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure", help="print volume / density / mass cross-check table")
    bb = sub.add_parser(
        "build-brick",
        help="compile a worldbody with one brick per (type, colour) and check masses",
    )
    bb.add_argument(
        "--output",
        type=Path,
        default=Path("out") / "duplo_world.xml",
        help="path for the compiled-then-serialised XML (default: out/duplo_world.xml)",
    )
    bb.add_argument(
        "--image",
        type=Path,
        default=Path("out") / "duplo_world.png",
        help="path for the rendered preview PNG (default: out/duplo_world.png)",
    )
    bb.add_argument(
        "--no-image",
        action="store_true",
        help="skip the offscreen render entirely",
    )
    dc = sub.add_parser(
        "debug-collision",
        help="visualise one brick's collision primitives, each in its own colour",
    )
    dc.add_argument(
        "--brick",
        choices=sorted(_BRICKS),
        default="4x2",
        help="which brick's collision primitives to show (default: 4x2)",
    )
    dc.add_argument(
        "--image",
        type=Path,
        default=None,
        help="render an offscreen PNG to this path instead of launching the "
        "interactive viewer (use when there is no display)",
    )
    args = parser.parse_args(argv)

    if args.cmd == "measure":
        return _cmd_measure()
    if args.cmd == "build-brick":
        image = None if args.no_image else args.image
        return _cmd_build_brick(args.output, image)
    if args.cmd == "debug-collision":
        return _cmd_debug_collision(args.brick, args.image)
    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
