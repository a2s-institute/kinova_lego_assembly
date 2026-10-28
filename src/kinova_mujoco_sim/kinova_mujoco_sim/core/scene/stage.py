"""Fixed "stage" for the Kinova MuJoCo work cell.

The stage is everything that stays put across scenes: a world-fixed legless
table, the work arm (Kinova Gen3 + Robotiq 2F-85), a second Gen3 reused as a
static obstacle, and the welded lego holder. Scene modules add the movable
content (bricks) on top of the spec the stage returns.

The obstacle arm is the same ``gen3.xml`` loaded a second time and attached
with an ``obs_`` prefix, so no second model file is needed. Its joints,
actuators and cameras come along but nothing wires them to ros2_control. The
MJCF generator writes an initial keyframe that sets the obstacle joints (and
their stiff position actuators' ``ctrl``) to ``layout.ARM2_QPOS``, so the arm
holds that static configuration as a passive obstacle.

The lego brick / holder builders live in the ``kinova_lego_assembly`` source
package under ``assets/mujoco``. They are imported through a ``sys.path``
shim; replacing this with proper packaging is tracked as deferred work.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco

from kinova_mujoco_sim.core.scene import layout

# Make ``assets.mujoco`` importable from the sibling ``kinova_lego_assembly``
# source package. From this file, parents[4] is the repo's ``src/`` dir.
_LEGO_PKG_ROOT = Path(__file__).resolve().parents[4] / "kinova_lego_assembly"
if str(_LEGO_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGO_PKG_ROOT))

from assets.mujoco.lego_holder import (  # noqa: E402
    REST_HEIGHT_M,
    STUD_TOP_HEIGHT_M,
    add_lego_holder,
)

_ARM_XML = ("mujoco", "arm", "gen3.xml")
_GRIPPER_XML = ("mujoco", "gripper", "2f85.xml")

OBSTACLE_PREFIX = "obs_"
"""Name prefix applied to every element of the obstacle arm."""


@dataclass(frozen=True)
class StageContext:
    """Fixed anchors a scene needs to place its movable content.

    Parameters
    ----------
    table_top_z : float
        World height of the table top; bricks rest on this plane.
    work_origin : tuple of float
        World ``(x, y, z)`` the scene builds around (the holder centre on the
        table top).
    holder_pos : tuple of float
        World position of the holder body origin.
    holder_stud_top_z : float
        World height of the holder stud tops; bricks dropped onto the holder
        land here.
    obstacle_joint_names : tuple of str
        Names of the obstacle arm's seven joints (prefixed); the MJCF generator
        pins them to a static configuration via the initial keyframe.
    """

    table_top_z: float
    work_origin: tuple[float, float, float]
    holder_pos: tuple[float, float, float]
    holder_stud_top_z: float
    obstacle_joint_names: tuple[str, ...]


def build_stage(
    asset_root: Path,
    offscreen_size: tuple[int, int] = (640, 480),
) -> tuple[mujoco.MjSpec, StageContext]:
    """Assemble the fixed stage and return it with its anchors.

    Parameters
    ----------
    asset_root : pathlib.Path
        Path to the ``models`` directory (contains the ``mujoco`` sub-tree).
    offscreen_size : tuple[int, int], optional
        ``(width, height)`` for MuJoCo's shared offscreen framebuffer
        (``visual/global`` ``offwidth``/``offheight``). Must be at least as
        large as the largest camera resolution used at runtime: MuJoCo
        allocates one offscreen FBO at this size for all cameras, and requests
        beyond it silently clamp. Oversizing wastes VRAM and can trip
        ``GL_FRAMEBUFFER_UNSUPPORTED`` on constrained GL stacks, so callers
        derive it from ``config/camera.yaml``.

    Returns
    -------
    tuple[mujoco.MjSpec, StageContext]
        The uncompiled spec and the anchors a scene builds around.
    """
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = int(offscreen_size[0])
    spec.visual.global_.offheight = int(offscreen_size[1])
    # The arm's actuators are stiff (kp up to 2000); the explicit Euler default
    # diverges with them. gen3.xml ships implicitfast, but that option is lost
    # when the arm is attached into this fresh parent spec, so set it here.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.default.geom.solref = list(layout.CONTACT_SOLREF)

    # Lighting: ambient headlight and a directional sun light.
    spec.visual.headlight.ambient = (0.3, 0.3, 0.3)
    spec.visual.headlight.diffuse = (0.0, 0.0, 0.0)
    spec.visual.headlight.specular = (0.0, 0.0, 0.0)

    sun = spec.worldbody.add_light(
        name="sun",
        pos=(2.0, -2.0, 4.0),
        dir=(-0.4, 0.4, -1.0),
        diffuse=(0.8, 0.78, 0.7),
        specular=(0.3, 0.3, 0.3),
        castshadow=True,
    )
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    _add_table(spec)
    _add_arm_mount_plates(spec)
    _add_work_arm(spec, asset_root)
    obstacle_joint_names = _add_obstacle_arm(spec, asset_root)

    holder_z = layout.TABLE_TOP_Z + REST_HEIGHT_M
    holder_pos = (layout.HOLDER_POS_XY[0], layout.HOLDER_POS_XY[1], holder_z)
    add_lego_holder(spec, name="holder", pos=holder_pos, quat=layout.HOLDER_QUAT, free=False)

    ctx = StageContext(
        table_top_z=layout.TABLE_TOP_Z,
        work_origin=layout.WORK_ORIGIN,
        holder_pos=holder_pos,
        holder_stud_top_z=holder_z + STUD_TOP_HEIGHT_M,
        obstacle_joint_names=obstacle_joint_names,
    )
    return spec, ctx


def _add_table(spec: mujoco.MjSpec) -> None:
    """Add the world-fixed, legless table box."""
    half_x, half_y, half_z = layout.TABLE_HALF_EXTENTS
    spec.worldbody.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, layout.TABLE_TOP_Z - half_z),
        size=(half_x, half_y, half_z),
        rgba=layout.TABLE_RGBA,
        contype=1,
        conaffinity=1,
        condim=3,
    )


def _add_arm_mount_plates(spec: mujoco.MjSpec) -> None:
    """Add the riser plate each arm base is bolted to, as in the real cell.

    The plates fill the gap between the table top and the arm anchors (see
    ``layout.ARM_MOUNT_PLATE_THICKNESS_Z``).
    """
    half_z = layout.ARM_MOUNT_PLATE_THICKNESS_Z / 2.0
    anchors = {"arm1": layout.ARM1_ANCHOR, "arm2": layout.ARM2_ANCHOR}
    for name, (arm_x, arm_y, _) in anchors.items():
        spec.worldbody.add_geom(
            name=f"{name}_mount_plate",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=(arm_x, arm_y, layout.TABLE_TOP_Z + half_z),
            size=(layout.ARM_MOUNT_PLATE_RADIUS, half_z, 0.0),
            rgba=layout.TABLE_RGBA,
            contype=1,
            conaffinity=1,
            condim=3,
        )


def _load_arm_with_gripper(asset_root: Path) -> mujoco.MjSpec:
    """Load a Gen3 spec with the 2F-85 gripper attached at the pinch site.

    The gripper is attached with the ``g_`` prefix so its bodies and default
    classes are namespaced (avoiding collisions with the arm's ``visual`` /
    ``collision`` classes and with scene body names like ``base``). The active
    driver joint is therefore ``g_robotiq_85_left_knuckle_joint`` and the
    actuator ``g_fingers_actuator`` in the in-memory model; the MJCF generator
    canonicalises those two names (dropping the ``g_``) so the ros2_control
    ``MujocoSystem`` plugin binds them to the kortex/robotiq
    ``robot_description``.
    """
    arm_spec = mujoco.MjSpec.from_file(str(asset_root.joinpath(*_ARM_XML)))
    gripper_spec = mujoco.MjSpec.from_file(str(asset_root.joinpath(*_GRIPPER_XML)))
    attach_frame = arm_spec.attach(gripper_spec, prefix="g_", site=arm_spec.site("pinch_site"))
    attach_frame.pos = [0.0, 0.0, 0.0]
    attach_frame.quat = [1.0, 0.0, 0.0, 0.0]
    return arm_spec


def _add_work_arm(spec: mujoco.MjSpec, asset_root: Path) -> None:
    """Attach the work arm (+gripper) at the work anchor, unprefixed.

    Arm joint / actuator names stay unprefixed (``joint_1`` ... ``joint_7``);
    the gripper keeps its ``g_`` prefix (see :func:`_load_arm_with_gripper`).
    """
    arm_spec = _load_arm_with_gripper(asset_root)
    frame = spec.worldbody.add_frame(pos=list(layout.ARM1_ANCHOR))
    spec.attach(arm_spec, prefix="", frame=frame)


def _add_obstacle_arm(spec: mujoco.MjSpec, asset_root: Path) -> tuple[str, ...]:
    """Attach the obstacle arm (bare Gen3) with the ``obs_`` prefix.

    Returns the prefixed names of its seven joints; the MJCF generator pins
    them to a static configuration via the initial keyframe.
    """
    obstacle_spec = mujoco.MjSpec.from_file(str(asset_root.joinpath(*_ARM_XML)))
    frame = spec.worldbody.add_frame(pos=list(layout.ARM2_ANCHOR))
    spec.attach(obstacle_spec, prefix=OBSTACLE_PREFIX, frame=frame)
    return tuple(f"{OBSTACLE_PREFIX}joint_{i}" for i in range(1, 8))
