"""Launch-time helpers for the ``kinova_mujoco_sim`` ros2_control bringup.

These build the inputs the launch file feeds to its nodes: the work arm
``robot_description`` (with kortex-only artifacts stripped), the per-camera ros2
parameter file plus topic remappings, the home pose, and the interpreter /
environment the MJCF generator subprocess needs. They are kept out of the launch
file so it stays a thin node-graph definition. Everything here is launch/ROS
specific (ament share dirs, xacro/URDF, ros2 parameters and remappings); it does
not need the ``mujoco`` / ``pytransform3d`` wheels the generator uses.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import xacro
import yaml

PACKAGE = "kinova_mujoco_sim"

# Where the MJCF generator dumps the assembled scene, and where the camera node
# params file is written alongside it.
SCENE_MJCF_PATH = "/tmp/kinova_mujoco_sim/scene.xml"

# MJCF camera names (from models/mujoco/arm/gen3.xml) bound to the color/depth
# blocks of config/camera.yaml.
RGB_CAMERA_NAME = "wrist"
DEPTH_CAMERA_NAME = "wrist_depth"

ARM_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
)

MUJOCO_SYSTEM_PLUGIN = "mujoco_ros2_control/MujocoSystem"

# Position bound applied to the arm's continuous joints, see
# :func:`bound_continuous_joints`.
CONTINUOUS_JOINT_LIMIT = math.pi


def mujoco_free_ld_library_path() -> str:
    """Return ``LD_LIBRARY_PATH`` with directories that ship ``libmujoco`` dropped.

    The ``mujoco_ros2_control`` C++ node links its own source-built
    ``libmujoco.so.3.9.0`` and puts it on ``LD_LIBRARY_PATH``. The pip ``mujoco``
    wheel used by the generator has the identical SONAME but a different binary;
    because the wheel's extensions use ``DT_RUNPATH`` (searched after
    ``LD_LIBRARY_PATH``), the loader would otherwise bind them to the foreign
    library, corrupting the ABI. Removing the offending directories lets the
    wheel resolve its own bundled lib via its ``$ORIGIN`` runpath while leaving
    every other search path untouched.

    Returns
    -------
    str
        The filtered ``LD_LIBRARY_PATH`` value.
    """
    kept = []
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not entry:
            continue
        try:
            has_mujoco = any(p.name.startswith("libmujoco") for p in Path(entry).iterdir())
        except OSError:
            has_mujoco = False
        if not has_mujoco:
            kept.append(entry)
    return os.pathsep.join(kept)


def generator_python(share_dir: str) -> str:
    """Resolve the interpreter used to run the MJCF generator subprocess.

    The generator needs ``mujoco`` and ``pytransform3d``, which live in the
    project virtual environment rather than the system interpreter that runs
    ``ros2``. Resolving the venv here lets the user simply source the install
    space and run ``ros2 launch`` without prefixing the command with the venv
    python. Resolution order: ``KINOVA_MUJOCO_SIM_PYTHON`` override, an active
    ``VIRTUAL_ENV``, the workspace ``venv`` next to ``install/``, then the
    system ``python3`` as a last resort.

    Parameters
    ----------
    share_dir : str
        The package share directory, used to locate the workspace root.

    Returns
    -------
    str
        Path to (or name of) the Python interpreter for the generator.
    """
    override = os.environ.get("KINOVA_MUJOCO_SIM_PYTHON")
    if override:
        return override

    active_venv = os.environ.get("VIRTUAL_ENV")
    if active_venv:
        candidate = Path(active_venv) / "bin" / "python3"
        if candidate.exists():
            return str(candidate)

    # share_dir is <ws>/install/<pkg>/share/<pkg>; the workspace root is 3 up.
    workspace_root = Path(share_dir).resolve().parents[3]
    candidate = workspace_root / "venv" / "bin" / "python3"
    if candidate.exists():
        return str(candidate)

    return "python3"


def home_positions(share_dir: str) -> dict[str, float]:
    """Read the home pose joint positions from ``config/home_pose.yaml``.

    Parameters
    ----------
    share_dir : str
        The package share directory.

    Returns
    -------
    dict[str, float]
        Mapping of joint name to home position for the seven arm joints.
    """
    home_pose_file = Path(share_dir) / "config" / "home_pose.yaml"
    with home_pose_file.open() as handle:
        data = yaml.safe_load(handle)
    home_pose = data["/**"]["ros__parameters"]["home_pose"]
    names = home_pose["joint_names"]
    positions = home_pose["positions"]
    return {name: float(pos) for name, pos in zip(names, positions, strict=False)}


def camera_node_params_file(share_dir: str) -> tuple[str, list[tuple[str, str]]]:
    """Translate ``config/camera.yaml`` into per-camera-node ros2 parameters.

    ``mujoco_ros2_control`` creates one camera node per MJCF ``<camera>`` element,
    named after the camera. The work arm carries a color camera (``wrist``) and a
    depth camera (``wrist_depth``); ``camera.yaml`` is the single human-facing
    source of truth for both, in the legacy nested ``camera.rgb`` / ``camera.depth``
    layout. This maps the ``rgb`` block to the ``wrist`` node and the ``depth``
    block to the ``wrist_depth`` node (topic namespace, resolution, frame id,
    encoding, and explicit intrinsics) and writes a node-keyed params file the
    ``mujoco_ros2_control`` process loads.

    Parameters
    ----------
    share_dir : str
        The package share directory.

    Returns
    -------
    tuple[str, list[tuple[str, str]]]
        Path to the generated node-keyed camera parameters file, and the topic
        remappings that map the camera node's fixed leaf topics onto the full
        topic / info_topic names from ``camera.yaml``.
    """
    camera_file = Path(share_dir) / "config" / "camera.yaml"
    with camera_file.open() as handle:
        cam = yaml.safe_load(handle)["/**"]["ros__parameters"]["camera"]

    def _namespace(topic: str) -> str:
        return topic.strip("/").split("/")[0]

    rgb, depth = cam["rgb"], cam["depth"]
    nodes = {
        RGB_CAMERA_NAME: {
            "ros__parameters": {
                "topic_namespace": _namespace(rgb["topic"]),
                "width": int(rgb["resolution"]["width"]),
                "height": int(rgb["resolution"]["height"]),
                "color_image": True,
                "depth_image": False,
                # A fixed MuJoCo camera can render color and depth in one pass.
                # Publish the registered cloud here so it shares the RGB image's
                # pixel grid, intrinsics, frame and timestamp, matching the real
                # camera's /camera/depth/color/points contract.
                "point_cloud": bool(rgb.get("point_cloud", False)),
                "color_frame_id": rgb["frame_id"],
                "depth_frame_id": rgb["frame_id"],
                "color_encoding": str(rgb["encoding"]),
                "camera_info_k": [float(v) for v in rgb["camera_info"]["k"]],
                "camera_info_p": [float(v) for v in rgb["camera_info"]["p"]],
                "camera_info_r": [float(v) for v in rgb["camera_info"]["r"]],
            }
        },
        DEPTH_CAMERA_NAME: {
            "ros__parameters": {
                "topic_namespace": _namespace(depth["topic"]),
                "width": int(depth["resolution"]["width"]),
                "height": int(depth["resolution"]["height"]),
                "color_image": False,
                "depth_image": True,
                # This instance represents native raw depth only. The RGB
                # instance above owns the registered cloud.
                "point_cloud": bool(depth.get("point_cloud", False)),
                "depth_frame_id": depth["frame_id"],
                "depth_in_millimeters": depth["encoding"] == "16UC1",
                "camera_info_k": [float(v) for v in depth["camera_info"]["k"]],
                "camera_info_p": [float(v) for v in depth["camera_info"]["p"]],
                "camera_info_r": [float(v) for v in depth["camera_info"]["r"]],
            }
        },
    }

    # The camera node publishes on fixed leaf names under its topic_namespace
    # (color/image_raw, color/camera_info, depth/image_rect_raw,
    # depth/camera_info, depth/points). Remap those onto the full
    # topic / info_topic / points_topic from camera.yaml so the yaml is the
    # real source of truth for topic names.
    ns_rgb, ns_depth = _namespace(rgb["topic"]), _namespace(depth["topic"])
    remappings = [
        (f"/{ns_rgb}/color/image_raw", rgb["topic"]),
        (f"/{ns_rgb}/color/camera_info", rgb["info_topic"]),
        (f"/{ns_depth}/depth/image_rect_raw", depth["topic"]),
        (f"/{ns_depth}/depth/camera_info", depth["info_topic"]),
    ]
    if "points_topic" in depth:
        remappings.append((f"/{ns_depth}/depth/points", depth["points_topic"]))

    out_dir = Path(SCENE_MJCF_PATH).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "camera_nodes.yaml"
    with out_path.open("w") as handle:
        yaml.safe_dump(nodes, handle, default_flow_style=False)
    return str(out_path), remappings


def strip_kortex_artifacts(urdf_xml: str, prefix: str) -> str:
    """Remove kortex-only ros2_control and camera frames from the work arm URDF.

    ``kortex_description``'s ``load_arm`` always emits its own ``<ros2_control>``
    block (kortex driver / mock / gazebo, never MuJoCo) and, with ``vision=true``,
    a ``camera_link`` plus ``camera``/``color``/``depth_module`` joints mounting
    the wrist camera frames on ``end_effector_link``. This drops both so only the
    MujocoSystem control blocks survive and the ``camera_color_frame`` /
    ``camera_depth_frame`` links (kept) are re-parented to ``bracelet_link`` by
    the joints declared in the top-level xacro.

    Parameters
    ----------
    urdf_xml : str
        The xacro-processed work arm URDF.
    prefix : str
        Joint/link name prefix used when building the description.

    Returns
    -------
    str
        The URDF with the kortex-only ros2_control block and camera mounts removed.
    """
    root = ET.fromstring(urdf_xml)
    drop_joint_names = {
        f"{prefix}camera_module",
        f"{prefix}color_module",
        f"{prefix}depth_module",
    }
    to_remove = []
    for child in list(root):
        if child.tag == "ros2_control":
            plugin = child.find("./hardware/plugin")
            if plugin is None or (plugin.text or "").strip() != MUJOCO_SYSTEM_PLUGIN:
                to_remove.append(child)
        elif (child.tag == "joint" and child.get("name") in drop_joint_names) or (
            child.tag == "link" and child.get("name") == f"{prefix}camera_link"
        ):
            to_remove.append(child)
    for child in to_remove:
        root.remove(child)
    return ET.tostring(root, encoding="unicode")


def bound_continuous_joints(
    urdf_xml: str, prefix: str, limit: float = CONTINUOUS_JOINT_LIMIT
) -> str:
    """Turn the arm's continuous joints into revolute joints bounded to ``+-limit``.

    ``kortex_description`` declares joints 1, 3, 5 and 7 ``continuous``, which is
    true of the hardware but makes MoveIt unwind its solutions: a goal of -2.78
    rad is executed as -2.78 + 2*pi = 3.50 rad to take the short way round, and
    the joint then reports 3.50. Every later IK query is seeded with that state,
    and ``kdl_kinematics_plugin`` bounds continuous joints to [-pi, pi], so it
    fails to converge from a seed that far outside (measured: fine up to about
    pi + 0.05, ``NO_IK_SOLUTION`` from pi + 0.1). The result is that a single
    wrapped motion poisons every subsequent Cartesian move.

    Bounding the joints keeps planning, execution, and the reported state inside
    [-pi, pi], so commanded positions, joint states, and goal values all agree
    and the controller's goal tolerances stay meaningful. The cost is that the
    arm can no longer take a shortcut through +-pi, so poses must be reachable
    without winding past the bound.

    Parameters
    ----------
    urdf_xml : str
        The work arm URDF.
    prefix : str
        Joint/link name prefix used when building the description.
    limit : float, optional
        Symmetric position bound in radians, by default ``pi``.

    Returns
    -------
    str
        The URDF with the arm's continuous joints bounded.
    """
    root = ET.fromstring(urdf_xml)
    arm_joints = {f"{prefix}{name}" for name in ARM_JOINT_NAMES}
    for joint in root.findall("joint"):
        if joint.get("type") != "continuous" or joint.get("name") not in arm_joints:
            continue
        joint.set("type", "revolute")
        limit_element = joint.find("limit")
        if limit_element is None:
            limit_element = ET.SubElement(joint, "limit")
        limit_element.set("lower", repr(-limit))
        limit_element.set("upper", repr(limit))
    return ET.tostring(root, encoding="unicode")


def build_robot_description(
    share_dir: str,
    prefix: str,
    arm_anchor: tuple[float, float, float],
    home: dict[str, float],
    init_gripper: str = "0.0",
) -> dict[str, str]:
    """Process the work arm xacro into a ``robot_description`` parameter dict.

    Runs ``description/gen3_7dof_mujoco.urdf.xacro`` through xacro with the arm
    anchor and home joint positions, strips the kortex-only artifacts (see
    :func:`strip_kortex_artifacts`) so only the MujocoSystem interfaces remain,
    and bounds the continuous arm joints (see :func:`bound_continuous_joints`).

    Parameters
    ----------
    share_dir : str
        The package share directory.
    prefix : str
        Joint/link name prefix (must match the MJCF).
    arm_anchor : tuple of float
        World ``(x, y, z)`` where the work arm base is welded.
    home : dict[str, float]
        Mapping of arm joint name to its initial position.
    init_gripper : str, optional
        Initial gripper command, by default ``"0.0"``.

    Returns
    -------
    dict[str, str]
        ``{"robot_description": <urdf xml>}`` ready to pass as a node parameter.
    """
    arm_x, arm_y, arm_z = arm_anchor
    description_xacro = Path(share_dir) / "description" / "gen3_7dof_mujoco.urdf.xacro"
    mappings = {
        "name": "gen3",
        "prefix": prefix,
        "vision": "true",
        "arm_x": str(arm_x),
        "arm_y": str(arm_y),
        "arm_z": str(arm_z),
        "init_gripper": init_gripper,
    }
    mappings.update({f"init_{name}": str(home.get(name, 0.0)) for name in ARM_JOINT_NAMES})
    arm_urdf = xacro.process_file(str(description_xacro), mappings=mappings).toprettyxml(
        indent="  "
    )
    stripped = strip_kortex_artifacts(arm_urdf, prefix)
    return {"robot_description": bound_continuous_joints(stripped, prefix)}
