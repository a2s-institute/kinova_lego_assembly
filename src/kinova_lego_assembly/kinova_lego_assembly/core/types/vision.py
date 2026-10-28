"""Vision-related datatypes (detection results, segmentation outputs).

Designed to be ROS-independent so they can be exercised with pure ``pytest``.
The schema mirrors what Ultralytics' ``Results`` exposes for a segment task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LegoObject:
    """One segmented LEGO instance with per-detection metadata.

    Parameters
    ----------
    obj_id : int
        Sequential per-inference identifier ``0..N-1`` assigned by the
        decoder. Stable for the lifetime of one inference call only.
    class_id : int
        Class index predicted by the model (matches ``class_names``).
    class_name : str
        Human-readable label resolved from the model's class table.
    score : float
        Confidence in ``[0, 1]`` (from ``boxes.conf``).
    bbox_xyxy : tuple of float
        Bounding box ``(x1, y1, x2, y2)`` in absolute pixel coordinates
        of the original image (from ``boxes.xyxy``).
    bbox_xyxyn : tuple of float
        Bounding box ``(x1, y1, x2, y2)`` normalized to ``[0, 1]``
        relative to the original image size (from ``boxes.xyxyn``).
    mask_polygon_xy : numpy.ndarray, optional
        Polygon contour of the mask in absolute pixels with shape
        ``(M, 2)`` and dtype ``float32`` (from ``masks.xy``). ``None`` if
        the model does not produce masks.
    mask_polygon_xyn : numpy.ndarray, optional
        Same polygon normalized to ``[0, 1]`` (from ``masks.xyn``).
        ``None`` if the model does not produce masks.
    """

    obj_id: int
    class_id: int
    class_name: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_xyxyn: tuple[float, float, float, float]
    mask_polygon_xy: np.ndarray | None
    mask_polygon_xyn: np.ndarray | None


@dataclass(frozen=True)
class SegmentationResult:
    """Decoded inference output for a single image.

    Parameters
    ----------
    image_shape : tuple of int
        Original image shape as ``(H, W)``.
    detections : list of LegoObject
        Per-instance detections, ordered by their ``obj_id``.
    class_names : dict
        Full ``{class_id: class_name}`` table known to the model.
    """

    image_shape: tuple[int, int]
    detections: list[LegoObject]
    class_names: dict[int, str]
