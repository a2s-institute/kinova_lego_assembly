"""Ultralytics YOLO segmentation backend.

Wraps an Ultralytics ``YOLO`` model in a ROS-free class that returns the
package's own :class:`SegmentationResult`. The implementation aims to be
fire-and-forget: each :meth:`YOLOSegmentor.infer` call runs under
``torch.inference_mode`` and releases CUDA cache afterwards, so repeated
service requests do not leak GPU memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from kinova_lego_assembly.core.segmentation import registry
from kinova_lego_assembly.core.segmentation.base import SegmentationModel
from kinova_lego_assembly.core.types import LegoObject, SegmentationResult


class YOLOSegmentor(SegmentationModel):
    """YOLO-based instance segmentation backend.

    The constructor only accepts knobs we explicitly own. ``max_det`` is
    intentionally omitted so Ultralytics' shipped default (currently 300)
    applies; expose it here if a future use case needs control.

    Parameters
    ----------
    version : str or int, optional
        Model version. ``None`` selects the highest available ``v<N>``.
    device : str, optional
        Torch device string. ``None`` selects ``"cuda:0"`` when CUDA is
        available, otherwise ``"cpu"``.
    **overrides
        Any of the keys in :attr:`DEFAULTS`. Values that are ``None`` are
        ignored so callers can pass through ``get_parameter().value``
        results unchanged.
    """

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "conf": 0.85,
        "iou": 0.7,
        "imgsz": 640,
        "retina_masks": True,
        "agnostic_nms": True,
    }

    def __init__(
        self,
        version: str | int | None = None,
        device: str | None = None,
        **overrides: Any,
    ) -> None:
        self._dir: Path = registry.resolve("yolo", version)
        weights = self._dir / "model.onnx"
        if not weights.is_file():
            raise FileNotFoundError(f"Missing weights file: {weights}")

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cfg: dict[str, Any] = dict(self.DEFAULTS)
        self.update_cfg(**overrides)

        # Pass task="segment" explicitly: ONNX exports may not carry task
        # metadata, and Ultralytics otherwise defaults to detect.
        self._model = YOLO(str(weights), task="segment")
        # .to()/.eval() only apply to PyTorch backends; ONNX (AutoBackend)
        # wraps an onnxruntime session and selects device per predict call.
        if weights.suffix == ".pt":
            self._model.to(self._device)
            if hasattr(self._model, "model") and hasattr(self._model.model, "eval"):
                self._model.model.eval()

        self._class_names = self._load_class_names()

    def _load_class_names(self) -> dict[int, str]:
        """Load the ``{class_id: class_name}`` table.

        Prefers ``model.yaml`` next to the weights when present (single
        source of truth for the deployed class table); falls back to the
        names embedded in the checkpoint.
        """
        yml = self._dir / "model.yaml"
        if yml.is_file():
            payload = yaml.safe_load(yml.read_text()) or {}
            names = payload.get("names")
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            if isinstance(names, list):
                return {i: str(n) for i, n in enumerate(names)}
        return {int(k): str(v) for k, v in dict(self._model.names).items()}

    @property
    def class_names(self) -> dict[int, str]:
        """Return a copy of the ``{class_id: class_name}`` table."""
        return dict(self._class_names)

    @property
    def device(self) -> str:
        """Return the active torch device string (e.g. ``"cuda:0"``)."""
        return self._device

    @property
    def cfg(self) -> dict[str, Any]:
        """Return a copy of the current inference-knob configuration."""
        return dict(self._cfg)

    def update_cfg(self, **overrides: Any) -> None:
        """Update inference knobs in-place.

        Unknown keys and ``None`` values are silently ignored. Callers
        (e.g. the ROS ``on_set_parameters`` callback) can therefore
        forward parameter changes verbatim.
        """
        for key, value in overrides.items():
            if value is None or key not in self.DEFAULTS:
                continue
            self._cfg[key] = value

    def infer(self, image: np.ndarray) -> SegmentationResult:
        """Run YOLO inference on a single RGB image.

        Parameters
        ----------
        image : numpy.ndarray
            ``HxWx3`` RGB array with dtype ``uint8``.

        Returns
        -------
        SegmentationResult
            Decoded detections and the class table.

        Raises
        ------
        ValueError
            If ``image`` is not a 3-channel array.
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected HxWx3 RGB image array")

        with torch.inference_mode():
            results = self._model.predict(
                source=image[..., ::-1],  # if src image is np.ndarray, channels must be BGR!
                device=self._device,
                verbose=False,
                stream=False,
                **self._cfg,
            )
        try:
            return self._decode(results[0])
        finally:
            del results
            if self._device.startswith("cuda"):
                torch.cuda.empty_cache()

    def _decode(self, result: Any) -> SegmentationResult:
        """Convert a single Ultralytics ``Results`` into :class:`SegmentationResult`."""
        h, w = result.orig_shape
        names = self._class_names

        if result.boxes is None or len(result.boxes) == 0:
            return SegmentationResult(
                image_shape=(int(h), int(w)),
                detections=[],
                class_names=dict(names),
            )

        xyxy = result.boxes.xyxy.cpu().numpy()
        xyxyn = result.boxes.xyxyn.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        polys_xy = list(result.masks.xy) if result.masks is not None else [None] * len(cls)
        polys_xyn = list(result.masks.xyn) if result.masks is not None else [None] * len(cls)

        detections: list[LegoObject] = []
        for i, _ in enumerate(cls):
            class_id = int(cls[i])
            detections.append(
                LegoObject(
                    obj_id=i,
                    class_id=class_id,
                    class_name=names.get(class_id, str(class_id)),
                    score=float(conf[i]),
                    bbox_xyxy=tuple(float(v) for v in xyxy[i]),
                    bbox_xyxyn=tuple(float(v) for v in xyxyn[i]),
                    mask_polygon_xy=(
                        np.asarray(polys_xy[i], dtype=np.float32)
                        if polys_xy[i] is not None
                        else None
                    ),
                    mask_polygon_xyn=(
                        np.asarray(polys_xyn[i], dtype=np.float32)
                        if polys_xyn[i] is not None
                        else None
                    ),
                )
            )

        return SegmentationResult(
            image_shape=(int(h), int(w)),
            detections=detections,
            class_names=dict(names),
        )
