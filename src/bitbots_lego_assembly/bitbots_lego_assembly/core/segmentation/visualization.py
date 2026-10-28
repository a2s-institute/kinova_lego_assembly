"""ROS-independent rendering of segmentation results onto an RGB image.

Consumes duck-typed detection objects exposing:

* ``class_id`` (int)
* ``class_name`` (str)
* ``score`` (float)
* ``bbox_xyxy`` (iterable of 4 floats)
* ``mask_polygon_xy`` (optional; iterable of points, each either a
  ``(x, y)`` pair or an object with ``.x`` / ``.y`` attributes)

This shape is satisfied by both :class:`LegoObject` and the ROS
``DetectedObject`` message, so the same helper can be reused from either
the core inference path or a ROS overlay publisher.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors


def _polygon_to_ndarray(polygon) -> np.ndarray | None:
    """Normalise a polygon (msgs of Point32 or (x, y) pairs) to an int32 array."""
    if polygon is None:
        return None
    pts: list[tuple[int, int]] = []
    for p in polygon:
        if hasattr(p, "x") and hasattr(p, "y"):
            pts.append((int(p.x), int(p.y)))
        else:
            pts.append((int(p[0]), int(p[1])))
    if not pts:
        return None
    return np.array(pts, dtype=np.int32)


def render_overlay(image_rgb: np.ndarray, detections: Iterable) -> np.ndarray:
    """Draw boxes, labels and mask polygons for ``detections`` on a copy of ``image_rgb``."""
    out = image_rgb.copy()
    annotator = Annotator(out, line_width=2)
    for det in detections:
        color = colors(int(det.class_id), True)
        label = f"{det.class_name} {float(det.score):.2f}"
        annotator.box_label(list(det.bbox_xyxy), label, color=color)
        pts = _polygon_to_ndarray(getattr(det, "mask_polygon_xy", None))
        if pts is not None:
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.4, out, 0.6, 0, dst=out)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
    return annotator.result()
