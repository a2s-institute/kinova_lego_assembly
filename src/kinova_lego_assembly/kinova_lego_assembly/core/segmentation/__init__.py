"""Segmentation inference backends (ROS-independent)."""

from kinova_lego_assembly.core.segmentation.base import SegmentationModel
from kinova_lego_assembly.core.segmentation.factory import (
    BackendSpec,
    build_segmentor,
    get_spec,
    supported_methods,
)
from kinova_lego_assembly.core.segmentation.registry import list_versions, resolve
from kinova_lego_assembly.core.segmentation.visualization import render_overlay
from kinova_lego_assembly.core.segmentation.yolo_segmentor import YOLOSegmentor

__all__ = [
    "BackendSpec",
    "SegmentationModel",
    "YOLOSegmentor",
    "build_segmentor",
    "get_spec",
    "list_versions",
    "render_overlay",
    "resolve",
    "supported_methods",
]
