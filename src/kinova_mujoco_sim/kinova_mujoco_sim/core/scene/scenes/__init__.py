"""Registry of pluggable scenes.

Built-in scenes register themselves on import. Add a new scene by writing a
module here that defines a :class:`~kinova_mujoco_sim.core.scene.scenes.base.Scene`
and calling :func:`register` with an instance.
"""

from __future__ import annotations

import importlib
import pkgutil

from kinova_mujoco_sim.core.scene.scenes.base import Scene, available, get, register

# automatically import new scenes
for _, _name, _ in pkgutil.iter_modules(__path__):
    if _name != "base":
        importlib.import_module(f"{__name__}.{_name}")

__all__ = ["Scene", "available", "get", "register"]
