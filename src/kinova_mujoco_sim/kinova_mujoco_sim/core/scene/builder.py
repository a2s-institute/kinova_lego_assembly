"""Scene assembly for the Kinova Gen3 + Robotiq 2F-85 MuJoCo work cell.

Composes a compiled MjModel/MjData pair from two layers:

* a fixed stage (table, work arm, obstacle arm, welded holder) built by
  :func:`kinova_mujoco_sim.core.scene.stage.build_stage`, and
* a pluggable scene that adds the movable bricks, selected by name from
  :mod:`kinova_mujoco_sim.core.scene.scenes`.

Keeping scene construction here (rather than in the node) makes it easy to
reuse the scene in offline scripts and unit tests.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from kinova_mujoco_sim.core.scene import scenes
from kinova_mujoco_sim.core.scene.stage import StageContext, build_stage

DEFAULT_SCENE = "loose_pile"


def build_scene(
    asset_root: Path, scene_name: str = DEFAULT_SCENE
) -> tuple[mujoco.MjModel, mujoco.MjData, StageContext]:
    """Build the fixed stage, populate a scene, and compile the model.

    Parameters
    ----------
    asset_root : pathlib.Path
        Path to the ``models`` directory (contains the ``mujoco`` sub-tree).
    scene_name : str, optional
        Name of the registered scene to populate. Defaults to
        :data:`DEFAULT_SCENE`.

    Returns
    -------
    tuple[mujoco.MjModel, mujoco.MjData, StageContext]
        ``(model, data)`` ready for simulation plus the stage anchors (which
        carry, among other things, the obstacle joint names).
    """
    spec, ctx = build_stage(asset_root)
    scenes.get(scene_name).populate(spec, ctx)
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data, ctx
