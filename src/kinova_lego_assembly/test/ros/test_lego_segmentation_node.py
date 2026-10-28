"""Smoke test for the ``LegoSegmentation`` service node.

Spins the node in-process, calls ``~/segment_image`` with a small RGB
frame and verifies the response is well formed. Skips only when the ROS
runtime (``rclpy`` / ``cv_bridge``) is not importable, since those come
from a sourced colcon workspace rather than ``pip``; pip-managed deps
(torch, ultralytics, onnxruntime) are required and must fail loudly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# `rclpy` and `cv_bridge` ship via the ROS apt stack and only resolve
# once ``install/setup.bash`` has been sourced. Without that source
# step they are legitimately absent (e.g. running pytest from a plain
# pip venv), so `importorskip` turns the missing module into a clean
# SKIP rather than collection failure. Pip-managed perception deps are
# allowed to raise ImportError as usual to flag a broken venv setup.
rclpy = pytest.importorskip("rclpy")
pytest.importorskip("cv_bridge")

from bitbots_lego_assembly.core.segmentation import registry
from bitbots_lego_assembly.ros.nodes.lego_segmentation import LegoSegmentationNode
from bitbots_lego_assembly_interfaces.srv import SegmentImage
from cv_bridge import CvBridge


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


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def _make_request_image(path: Path, bridge: CvBridge):
    arr: np.ndarray
    if path.is_file():
        try:
            import cv2

            bgr = cv2.imread(str(path))
            if bgr is not None:
                arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                arr = np.random.default_rng(0).integers(0, 255, (640, 640, 3), dtype=np.uint8)
        except ImportError:
            arr = np.random.default_rng(0).integers(0, 255, (640, 640, 3), dtype=np.uint8)
    else:
        arr = np.random.default_rng(0).integers(0, 255, (640, 640, 3), dtype=np.uint8)
    return bridge.cv2_to_imgmsg(arr, encoding="rgb8")


def test_segment_service_returns_class_table(ros_context, sample_image_path: Path) -> None:
    node = LegoSegmentationNode()
    try:
        client = node.create_client(SegmentImage, "~/segment_image")
        assert client.wait_for_service(timeout_sec=5.0), "Service did not come up"

        req = SegmentImage.Request()
        req.image = _make_request_image(sample_image_path, CvBridge())

        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
        assert future.done(), "Service call timed out"
        resp = future.result()

        assert resp is not None
        assert isinstance(resp.class_names, list) and resp.class_names
        for det in resp.detections:
            assert 0.0 <= det.score <= 1.0
            assert det.obj_id >= 0
            assert len(det.bbox_xyxy) == 4
    finally:
        node.destroy_node()
