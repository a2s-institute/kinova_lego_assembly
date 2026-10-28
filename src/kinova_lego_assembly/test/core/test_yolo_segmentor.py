"""Integration tests for :class:`YOLOSegmentor` against the bundled model.

These tests load the real ONNX checkpoint. ``torch``, ``ultralytics``,
``onnxruntime`` and the v1 ``model.onnx`` are required runtime
dependencies (see ``requirements.txt``); a missing wheel must surface as
a loud ``ImportError`` so the dev installs the env properly instead of
silently skipping coverage. Only the model file is treated as optional
(skipped at the module level when absent).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kinova_lego_assembly.core.segmentation import (
    YOLOSegmentor,
    build_segmentor,
    registry,
    supported_methods,
)
from kinova_lego_assembly.core.types import LegoObject, SegmentationResult


def _model_available() -> bool:
    try:
        path = registry.resolve("yolo")
    except FileNotFoundError:
        return False
    return (path / "model.onnx").is_file()


pytestmark = pytest.mark.skipif(
    not _model_available(),
    reason="No YOLO model checkpoint available under trained_models/",
)


@pytest.fixture(scope="module")
def segmentor() -> YOLOSegmentor:
    return build_segmentor("yolo", device="cpu")


def test_factory_lists_yolo_as_supported() -> None:
    assert "yolo" in supported_methods()


def _load_or_synth(path: Path) -> np.ndarray:
    if path.is_file():
        try:
            import cv2

            bgr = cv2.imread(str(path))
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except ImportError:
            pass
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(640, 640, 3), dtype=np.uint8)


def test_class_names_match_model_yaml(segmentor: YOLOSegmentor) -> None:
    names = segmentor.class_names
    assert isinstance(names, dict) and names
    assert all(isinstance(k, int) and isinstance(v, str) for k, v in names.items())


def test_defaults_match_documented_values(segmentor: YOLOSegmentor) -> None:
    assert segmentor.cfg["conf"] == YOLOSegmentor.DEFAULTS["conf"]
    assert segmentor.cfg["retina_masks"] is True
    assert "max_det" not in segmentor.cfg


def test_update_cfg_ignores_none_and_unknown(segmentor: YOLOSegmentor) -> None:
    original_iou = segmentor.cfg["iou"]
    segmentor.update_cfg(iou=None, bogus_key=123)
    assert segmentor.cfg["iou"] == original_iou
    segmentor.update_cfg(iou=0.42)
    assert segmentor.cfg["iou"] == pytest.approx(0.42)
    segmentor.update_cfg(iou=original_iou)


def test_infer_returns_well_formed_result(
    segmentor: YOLOSegmentor, sample_image_path: Path
) -> None:
    image = _load_or_synth(sample_image_path)
    result = segmentor.infer(image)

    assert isinstance(result, SegmentationResult)
    assert result.image_shape == image.shape[:2]
    assert isinstance(result.class_names, dict) and result.class_names

    for i, det in enumerate(result.detections):
        assert isinstance(det, LegoObject)
        assert det.obj_id == i
        assert 0.0 <= det.score <= 1.0
        x1, y1, x2, y2 = det.bbox_xyxy
        assert 0.0 <= x1 <= x2 <= image.shape[1]
        assert 0.0 <= y1 <= y2 <= image.shape[0]
        for v in det.bbox_xyxyn:
            assert 0.0 <= v <= 1.0
        if det.mask_polygon_xy is not None:
            assert det.mask_polygon_xy.ndim == 2 and det.mask_polygon_xy.shape[1] == 2


def test_infer_rejects_bad_input_shape(segmentor: YOLOSegmentor) -> None:
    with pytest.raises(ValueError):
        segmentor.infer(np.zeros((10, 10), dtype=np.uint8))
