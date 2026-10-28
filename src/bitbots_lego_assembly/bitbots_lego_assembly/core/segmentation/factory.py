"""Backend dispatch for segmentation models.

Centralises everything that varies per backend (default knob values and
how to build the concrete :class:`SegmentationModel`) so callers, ROS
or otherwise, can stay generic. Add a new backend by registering a
:class:`BackendSpec` factory under its method name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bitbots_lego_assembly.core.segmentation.base import SegmentationModel
from bitbots_lego_assembly.core.segmentation.yolo_segmentor import YOLOSegmentor


@dataclass(frozen=True)
class BackendSpec:
    """Static description of one segmentation backend.

    Parameters
    ----------
    knobs : dict
        Tunable knob name -> default value. Each entry corresponds to a
        ROS parameter declared under ``<method>.<knob>``.
    build : callable
        Concrete factory ``(version, **overrides) -> SegmentationModel``.
    """

    knobs: dict[str, Any]
    build: Callable[..., SegmentationModel]


def _yolo_spec() -> BackendSpec:
    return BackendSpec(
        knobs=dict(YOLOSegmentor.DEFAULTS),
        build=lambda version=None, **overrides: YOLOSegmentor(version=version, **overrides),
    )


_REGISTRY: dict[str, Callable[[], BackendSpec]] = {
    "yolo": _yolo_spec,
}


def supported_methods() -> list[str]:
    """List every registered backend method name."""
    return sorted(_REGISTRY)


def get_spec(method: str) -> BackendSpec:
    """Return the :class:`BackendSpec` for ``method``.

    Raises
    ------
    ValueError
        If ``method`` is not registered.
    """
    if method not in _REGISTRY:
        raise ValueError(f"Unknown segmentation method: {method!r}. Known: {supported_methods()}")
    return _REGISTRY[method]()


def build_segmentor(
    method: str,
    version: str | int | None = None,
    **overrides: Any,
) -> SegmentationModel:
    """Build a segmentor for ``method`` at ``version`` with optional overrides.

    Parameters
    ----------
    method : str
        Backend name registered in :func:`supported_methods`.
    version : str or int, optional
        Model version token (``"v3"``, ``"3"`` or ``3``); ``None`` selects
        the latest available version on disk.
    **overrides
        Knob overrides forwarded to the backend constructor. Values that
        are ``None`` are ignored by the backend (see :meth:`SegmentationModel.update_cfg`).

    Returns
    -------
    SegmentationModel
        A ready-to-use segmentation backend instance.
    """
    spec = get_spec(method)
    return spec.build(version=version, **overrides)
