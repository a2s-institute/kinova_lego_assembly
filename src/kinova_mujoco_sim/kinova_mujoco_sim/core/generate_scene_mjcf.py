"""Generate a combined MJCF file for the Kinova MuJoCo work cell.

This script assembles the work cell with the existing Python ``MjSpec`` builders
(table, work arm + gripper, obstacle arm, welded holder, and the selected
scene's movable bricks), compiles it, and writes the result to disk as a single
MJCF XML file. The DFKI ``mujoco_ros2_control`` node then loads that file via
its ``robot_model_path`` parameter.

It is a one-shot command-line tool: it builds, writes, validates, and exits, so
a launch file can sequence the real simulator to start only after the file
exists. It depends only on ``mujoco`` (not ``rclpy``) so it can run under the
``mujoco`` virtual environment interpreter.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import yaml
from ament_index_python.packages import get_package_share_directory

from kinova_mujoco_sim.core.scene import layout, scenes
from kinova_mujoco_sim.core.scene.builder import DEFAULT_SCENE
from kinova_mujoco_sim.core.scene.stage import OBSTACLE_PREFIX, build_stage

_DEFAULT_OUTPUT = "/tmp/kinova_mujoco_sim/scene.xml"

_WORK_ARM_JOINTS = tuple(f"joint_{i}" for i in range(1, 8))


def _share_dir() -> Path:
    """Return the installed ``kinova_mujoco_sim`` package share directory.

    Returns
    -------
    pathlib.Path
        Path to the package's ``share`` directory (models, config, ...).
    """
    return Path(get_package_share_directory("kinova_mujoco_sim"))


def _camera_offscreen_size() -> tuple[int, int]:
    """Compute the shared offscreen FBO size needed by ``config/camera.yaml``.

    MuJoCo allocates a single offscreen framebuffer per model (sized by
    ``visual/global`` ``offwidth``/``offheight``) that every camera renders
    into; render requests larger than it clamp silently. Take the max width
    and height across all cameras so every stream fits without oversizing the
    FBO (which wastes VRAM and can trip ``GL_FRAMEBUFFER_UNSUPPORTED`` on
    constrained GL stacks).

    Returns
    -------
    tuple[int, int]
        ``(width, height)`` in pixels.
    """
    camera_file = _share_dir() / "config" / "camera.yaml"
    cam = yaml.safe_load(camera_file.read_text())["/**"]["ros__parameters"]["camera"]
    widths, heights = [], []
    for stream in cam.values():
        resolution = stream.get("resolution")
        if not resolution:
            continue
        widths.append(int(resolution["width"]))
        heights.append(int(resolution["height"]))
    return max(widths), max(heights)


def _home_positions() -> dict[str, float]:
    """Read the work arm home joint positions from ``config/home_pose.yaml``.

    Returns
    -------
    dict[str, float]
        Mapping of work arm joint name to home position. Only joints that have
        a matching position entry are included.
    """
    home_pose_file = _share_dir() / "config" / "home_pose.yaml"
    data = yaml.safe_load(home_pose_file.read_text())
    home_pose = data["/**"]["ros__parameters"]["home_pose"]
    names = home_pose["joint_names"]
    positions = home_pose["positions"]
    return {name: float(pos) for name, pos in zip(names, positions, strict=False)}


# The gripper is attached with a ``g_`` prefix for namespacing; these two names
# are canonicalised in the dumped MJCF so they match the kortex/robotiq
# ``robot_description`` that ros2_control binds against. Both prefixed names are
# unique substrings, so a plain string replace updates the definitions and every
# reference (actuator ``joint=``, tendon, equality) consistently.
_GRIPPER_NAME_CANONICALISATIONS = {
    "g_robotiq_85_left_knuckle_joint": "robotiq_85_left_knuckle_joint",
    "g_fingers_actuator": "fingers_actuator",
}


def _canonicalise_gripper_names(xml: str) -> str:
    """Drop the ``g_`` prefix from the ros2_control-bound gripper names.

    Parameters
    ----------
    xml : str
        MJCF serialised from the assembled spec.

    Returns
    -------
    str
        MJCF with the gripper active joint and actuator named canonically.
    """
    for prefixed, canonical in _GRIPPER_NAME_CANONICALISATIONS.items():
        xml = xml.replace(prefixed, canonical)
    return xml


def _strip_cameras_by_prefix(xml: str, prefix: str) -> str:
    """Remove every ``<camera>`` whose name starts with ``prefix``.

    The obstacle arm reuses the work arm model, so it carries the same wrist
    cameras (under the obstacle prefix). They are pure-physics scenery with no
    use here, and ``mujoco_ros2_control`` spawns a render thread per ``<camera>``
    element, so dropping them frees those resources.

    Parameters
    ----------
    xml : str
        MJCF serialised from the assembled spec.
    prefix : str
        Camera-name prefix to remove (e.g. the obstacle arm prefix).

    Returns
    -------
    str
        MJCF with the matching cameras removed.
    """
    root = ET.fromstring(xml)
    parent_map = {child: parent for parent in root.iter() for child in parent}
    for camera in list(root.iter("camera")):
        if camera.get("name", "").startswith(prefix):
            parent_map[camera].remove(camera)
    return ET.tostring(root, encoding="unicode")


def _name_anonymous_nested_defaults(xml: str) -> str:
    """Give every anonymous nested ``<default>`` a unique class name.

    ``MjSpec.attach`` nests each attached spec's (anonymous) main default under
    the parent's main default. ``MjSpec.to_xml`` then emits these nested
    ``<default>`` blocks without a ``class`` attribute, which ``mj_loadXML``
    rejects ("empty class name"), since only the single top-level main default
    may be anonymous. The named child default classes (e.g. ``visual``,
    ``large_actuator``) resolve globally by name regardless of nesting, and no
    element references these anonymous blocks via ``childclass``, so assigning
    them generated names makes the file loadable without changing semantics.

    Parameters
    ----------
    xml : str
        MJCF serialised by :meth:`MjSpec.to_xml`.

    Returns
    -------
    str
        MJCF with all nested ``<default>`` blocks carrying a ``class`` name.
    """
    root = ET.fromstring(xml)
    main_default = root.find("default")
    if main_default is None:
        return xml

    counter = 0
    for parent in main_default.iter("default"):
        for child in parent.findall("default"):
            if "class" not in child.attrib:
                child.set("class", f"_attached_default_{counter}")
                counter += 1

    if counter == 0:
        return xml
    return ET.tostring(root, encoding="unicode")


def _absolutize_mesh_paths(xml: str, mesh_root: Path) -> str:
    """Rewrite ``<mesh file="...">`` references to absolute paths.

    ``MjSpec.to_xml`` emits mesh files by basename and drops the per-spec
    ``meshdir``, so the dumped MJCF cannot resolve meshes relative to its own
    location. Mesh basenames are unique across the arm and gripper, so map each
    basename to its absolute path under ``mesh_root`` and write that in, making
    the file self-contained regardless of where it is written.

    Parameters
    ----------
    xml : str
        MJCF whose ``<asset><mesh>`` files use basenames.
    mesh_root : pathlib.Path
        Directory tree that contains the mesh files.

    Returns
    -------
    str
        MJCF with absolute mesh file paths.
    """
    by_name = {p.name: p for p in mesh_root.rglob("*.stl")}
    root = ET.fromstring(xml)
    for mesh in root.iter("mesh"):
        ref = mesh.get("file")
        if ref is None:
            continue
        resolved = by_name.get(Path(ref).name)
        if resolved is not None:
            mesh.set("file", str(resolved))
    return ET.tostring(root, encoding="unicode")


def _build_scene_spec(asset_root: Path, scene_name: str) -> tuple[mujoco.MjSpec, mujoco.MjModel]:
    """Build the stage, populate a scene, compile, and return spec and model.

    Parameters
    ----------
    asset_root : pathlib.Path
        Path to the ``models`` directory (contains the ``mujoco`` sub-tree).
    scene_name : str
        Name of the registered scene to populate.

    Returns
    -------
    tuple[mujoco.MjSpec, mujoco.MjModel]
        The compiled spec (ready to serialise with :meth:`MjSpec.to_xml`) and
        the compiled model (used to compute the initial keyframe).
    """
    spec, ctx = build_stage(asset_root, offscreen_size=_camera_offscreen_size())
    scenes.get(scene_name).populate(spec, ctx)
    model = spec.compile()
    return spec, model


def _seed_joint(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, angle: float) -> None:
    """Set a joint's ``qpos`` and its position actuator's ``ctrl`` to ``angle``.

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled combined model.
    data : mujoco.MjData
        The data buffer whose ``qpos``/``ctrl`` are seeded.
    joint_name : str
        Name of the joint (and of its position actuator, which shares the name
        for both arms in this model).
    angle : float
        Target joint angle, radians.
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id != -1:
        data.qpos[model.jnt_qposadr[joint_id]] = angle
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
    if actuator_id != -1:
        data.ctrl[actuator_id] = angle


def _initial_keyframe_xml(model: mujoco.MjModel, home: dict[str, float]) -> str:
    """Build a single ``<keyframe>`` for the deterministic initial pose.

    The DFKI ``mujoco_ros2_control`` plugin resets the simulation to keyframe 0
    on start (and on reset) whenever the model has any keyframes, and the
    ros2_control controllers hold whatever joint state they read at activation.
    To make both the unmanaged obstacle arm and the managed work arm start in
    the intended pose, this keyframe seeds:

    * the obstacle arm joints to ``layout.ARM2_QPOS`` (held by its own stiff
      position actuators, since nothing else commands them), and
    * the work arm joints to their ``home`` positions (so the joint trajectory
      controller activates holding the home pose instead of zero).

    For every seeded joint the matching position actuator's ``ctrl`` is set too.
    All other joints stay at their compiled default (``qpos0``). Replacing the
    arm models' bundled ``home``/``retract`` keyframes with this single
    deterministic one avoids an ambiguous reset.

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled combined model.
    home : dict[str, float]
        Work arm joint name to home position mapping.

    Returns
    -------
    str
        A ``<keyframe>`` element with one ``<key name="init" .../>``.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    for index, angle in enumerate(layout.ARM2_QPOS, start=1):
        _seed_joint(model, data, f"{OBSTACLE_PREFIX}joint_{index}", angle)

    for joint_name in _WORK_ARM_JOINTS:
        if joint_name in home:
            _seed_joint(model, data, joint_name, home[joint_name])

    qpos = " ".join(f"{value:.10g}" for value in data.qpos)
    ctrl = " ".join(f"{value:.10g}" for value in data.ctrl)
    return f'<keyframe><key name="init" qpos="{qpos}" ctrl="{ctrl}"/></keyframe>'


def _set_initial_keyframe(xml: str, model: mujoco.MjModel, home: dict[str, float]) -> str:
    """Replace any bundled keyframes with the single deterministic init key.

    Parameters
    ----------
    xml : str
        MJCF serialised from the assembled spec.
    model : mujoco.MjModel
        The compiled combined model used to compute the keyframe.
    home : dict[str, float]
        Work arm joint name to home position mapping.

    Returns
    -------
    str
        MJCF whose only keyframe is the computed init key.
    """
    root = ET.fromstring(xml)
    for existing in root.findall("keyframe"):
        root.remove(existing)
    # nkey is recomputed from the keyframe element, so drop any stale count.
    size = root.find("size")
    if size is not None and "nkey" in size.attrib:
        del size.attrib["nkey"]
    root.append(ET.fromstring(_initial_keyframe_xml(model, home)))
    return ET.tostring(root, encoding="unicode")


def generate_scene_mjcf(scene_name: str, output: Path) -> None:
    """Build, serialise, write, and validate the combined MJCF file.

    Parameters
    ----------
    scene_name : str
        Name of the registered scene to populate.
    output : pathlib.Path
        Destination path for the combined MJCF file.
    """
    asset_root = _share_dir() / "models"
    print(f"Building scene {scene_name!r} from assets at {asset_root}")

    spec, model = _build_scene_spec(asset_root, scene_name)
    xml = _name_anonymous_nested_defaults(spec.to_xml())
    xml = _absolutize_mesh_paths(xml, asset_root / "meshes")
    xml = _canonicalise_gripper_names(xml)
    xml = _strip_cameras_by_prefix(xml, OBSTACLE_PREFIX)
    xml = _set_initial_keyframe(xml, model, _home_positions())

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(xml)
    print(f"Wrote combined MJCF to {output}")

    # Fail fast if the dumped file (and its mesh references) cannot be
    # reloaded; the C++ node would otherwise fail later with less context.
    mujoco.MjModel.from_xml_path(str(output))
    print("Validated: combined MJCF reloads successfully")


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, generate the scene MJCF once, then exit.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scene",
        default=DEFAULT_SCENE,
        help="Registered scene name to populate (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help="Destination MJCF path (default: %(default)s).",
    )
    # Tolerate ROS-style remapping/args if launched via ros2 run.
    args, _ = parser.parse_known_args(argv)

    try:
        generate_scene_mjcf(args.scene, Path(args.output))
    except Exception as exc:
        print(f"generate_scene_mjcf failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
