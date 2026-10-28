"""Abstract base class for segmentation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from bitbots_lego_assembly.core.types import SegmentationResult


class SegmentationModel(ABC):
    """Interface every segmentation backend must implement."""

    @property
    @abstractmethod
    def class_names(self) -> dict[int, str]:
        """Return the ``{class_id: class_name}`` table known to the model."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the compute device the model runs on (e.g. ``"cuda:0"``, ``"cpu"``)."""

    @abstractmethod
    def infer(self, image: np.ndarray) -> SegmentationResult:
        """Run inference on a single image and return decoded detections.

        Parameters
        ----------
        image : numpy.ndarray
            Input image as an ``HxWx3`` array in RGB order, dtype ``uint8``.

        Returns
        -------
        SegmentationResult
            Decoded detections plus class table and timing breakdown.
        """

    def update_cfg(self, **overrides: Any) -> None:  # noqa: B027
        """Hot-reconfigure backend-specific inference knobs.

        Default implementation is intentionally a no-op so backends
        without tunable knobs (e.g. classical CV pipelines) need no
        override; subclasses with knobs override it to accept named
        keyword arguments. Unknown keys and ``None`` values should be
        ignored so callers can forward parameter dicts verbatim.
        """
