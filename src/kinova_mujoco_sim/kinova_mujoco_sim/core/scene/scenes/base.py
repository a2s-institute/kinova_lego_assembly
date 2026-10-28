"""Scene plugin interface and registry.

A scene adds the movable content (free brick bodies) on top of the fixed
stage built by :func:`kinova_mujoco_sim.core.scene.stage.build_stage`. Scenes
receive the stage's :class:`~kinova_mujoco_sim.core.scene.stage.StageContext` so
they can place content relative to the (fixed) work origin and holder without
hard-coding world coordinates.

The registry lives here (rather than in the package ``__init__``) so scene
modules can import :func:`register` without a circular import.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import mujoco

from kinova_mujoco_sim.core.scene.stage import StageContext


@runtime_checkable
class Scene(Protocol):
    """A pluggable scene that populates a stage spec with movable content."""

    name: str

    def populate(self, spec: mujoco.MjSpec, ctx: StageContext) -> None:
        """Add this scene's bodies to ``spec`` using the stage anchors ``ctx``."""
        ...


_REGISTRY: dict[str, Scene] = {}


def register(scene: Scene) -> Scene:
    """Register ``scene`` under its ``name`` and return it.

    Raises
    ------
    ValueError
        If a scene with the same name is already registered.
    """
    if scene.name in _REGISTRY:
        raise ValueError(f"scene {scene.name!r} is already registered")
    _REGISTRY[scene.name] = scene
    return scene


def get(name: str) -> Scene:
    """Return the registered scene called ``name``.

    Raises
    ------
    KeyError
        If no scene is registered under ``name``.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown scene {name!r}; choose from {available()}") from exc


def available() -> list[str]:
    """Return the sorted names of all registered scenes."""
    return sorted(_REGISTRY)
