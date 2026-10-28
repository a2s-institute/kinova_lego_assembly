"""ROS2 service node: thin wrapper around the core segmentation factory.

Reads ROS parameters, hands them to
:func:`bitbots_lego_assembly.core.segmentation.build_segmentor`, and
exposes the resulting backend via ``~/segment_image``. All
backend-specific knowledge (knob names, defaults, build logic) lives in
the pure-Python factory so this node stays generic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from bitbots_lego_assembly_interfaces.msg import DetectedObject
from bitbots_lego_assembly_interfaces.srv import SegmentImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32
from mir_interfaces.msg import Object, ObjectList
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_parameters, qos_profile_sensor_data
from sensor_msgs.msg import Image

from bitbots_lego_assembly.core.segmentation import (
    SegmentationModel,
    build_segmentor,
    get_spec,
    supported_methods,
)
from bitbots_lego_assembly.core.segmentation.visualization import render_overlay
from bitbots_lego_assembly.core.types import LegoObject


def _ensure_models_root_env() -> None:
    """Point the registry at the installed share directory when no override is set.

    Keeps the core registry ROS-agnostic while still resolving model
    paths correctly once the package is installed via colcon.
    """
    if os.environ.get("BITBOTS_LEGO_MODELS_ROOT"):
        return
    try:
        share = Path(get_package_share_directory("bitbots_lego_assembly"))
    except PackageNotFoundError:
        return
    candidate = share / "trained_models"
    if candidate.is_dir():
        os.environ["BITBOTS_LEGO_MODELS_ROOT"] = str(candidate)


class LegoSegmentationNode(Node):
    """Service-only node that runs an instance-segmentation backend on incoming images."""

    def __init__(self) -> None:
        super().__init__("LegoSegmentation")
        self._log = self.get_logger()

        _ensure_models_root_env()
        self._declare_parameters()
        self._bridge = CvBridge()

        self._method = str(self.get_parameter("segmentation.model_method").value)
        version = self.get_parameter("segmentation.model_version").value or None
        try:
            spec = get_spec(self._method)
        except ValueError as exc:
            self._log.fatal(str(exc))
            raise SystemExit(1) from exc

        self._knob_keys: tuple[str, ...] = tuple(spec.knobs)
        self._knob_prefix = f"segmentation.{self._method}."
        overrides = {
            k: self.get_parameter(f"{self._knob_prefix}{k}").value for k in self._knob_keys
        }
        self._segmentor: SegmentationModel = build_segmentor(
            self._method, version=version, **overrides
        )
        self._log.info(
            f"Loaded segmentation backend '{self._method}' "
            f"(version={version or 'latest'}, device={self._segmentor.device}, "
            f"classes={len(self._segmentor.class_names)})"
        )
        self._warmup_segmentor()

        self.add_on_set_parameters_callback(self._on_param_change)
        self._seg_srv = self.create_service(SegmentImage, "~/segment_image", self._on_segment)
        _input_image_topic = str(self.get_parameter("detection.mmor.input_image_topic").value)
        _output_object_list = str(
            self.get_parameter("detection.mmor.output_object_list_topic").value
        )
        self._image_sub = self.create_subscription(
            Image,
            _input_image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self._object_list_pub = self.create_publisher(
            ObjectList,
            _output_object_list,
            qos_profile_parameters,
        )
        _viz_topic = str(self.get_parameter("visualization.output_topic").value)
        self._viz_pub = self.create_publisher(Image, _viz_topic, 1)
        self._log.info(f"Segmentation service ready: {self._seg_srv.srv_name}")
        self._log.info(
            "MMOR bridge ready: subscribed to "
            f"'{self._image_sub.topic_name}', publishing '{self._object_list_pub.topic_name}'"
        )

    def _warmup_segmentor(self) -> None:
        """Run one throwaway inference so the first real request is not slow."""
        imgsz = int(getattr(self._segmentor, "cfg", {}).get("imgsz") or 640)
        try:
            self._segmentor.infer(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))
        except Exception as exc:
            self._log.warn(f"Segmentation warm-up failed, first request may be slow: {exc}")
        else:
            self._log.info(f"Segmentation backend warmed up at imgsz={imgsz}")

    def _declare_parameters(self) -> None:
        """Declare every ROS parameter the node consumes.

        Knob parameters for every known backend are declared upfront so
        ``segmentation.model_method`` can switch between them without
        redeploying.
        """
        self.declare_parameter("segmentation.model_method", "yolo")
        self.declare_parameter("segmentation.model_version", "")
        self.declare_parameter("detection.mmor.input_image_topic", "/input_image_topic")
        self.declare_parameter("detection.mmor.output_object_list_topic", "/output_object_list")
        self.declare_parameter("visualization.output_topic", "/visualize/segmentation")

        for method in supported_methods():
            for knob, default in get_spec(method).knobs.items():
                self.declare_parameter(f"segmentation.{method}.{knob}", default)

    def _on_param_change(self, params: list[Any]) -> SetParametersResult:
        """Forward live ``segmentation.<method>.*`` parameter changes to the segmentor."""
        updates: dict[str, Any] = {}
        for p in params:
            if not p.name.startswith(self._knob_prefix):
                continue
            key = p.name[len(self._knob_prefix) :]
            if key in self._knob_keys:
                updates[key] = p.value
        if updates:
            self._segmentor.update_cfg(**updates)
            self._log.info(f"Reconfigured '{self._method}' segmentation knobs: {updates}")
        return SetParametersResult(successful=True)

    def _on_segment(
        self, req: SegmentImage.Request, resp: SegmentImage.Response
    ) -> SegmentImage.Response:
        """Decode the request image, run inference, pack the response."""
        self._log.info(
            f"segment_image: request received ({req.image.width}x{req.image.height} "
            f"encoding={req.image.encoding!r}); cfg={getattr(self._segmentor, 'cfg', '?')}"
        )

        try:
            img = self._bridge.imgmsg_to_cv2(req.image, desired_encoding="rgb8")
        except Exception as exc:
            self._log.error(f"Failed to convert request image: {exc}")
            resp.detections = []
            resp.class_names = []
            return resp

        result = self._segmentor.infer(img)
        resp.detections = [_lego_object_to_msg(d) for d in result.detections]
        resp.class_names = [result.class_names[i] for i in sorted(result.class_names)]
        self._publish_overlay(img, result.detections, req.image.header)
        self._log.info(f"segment_image: returned {len(result.detections)} detection(s)")
        return resp

    def _on_image(self, image_msg: Image) -> None:
        """Run segmentation for each incoming RGB frame and publish MMOR-compatible output."""
        self._log.info(
            f"on_image: frame received ({image_msg.width}x{image_msg.height} "
            f"encoding={image_msg.encoding!r})"
        )
        try:
            img = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        except Exception as exc:
            self._log.error(f"Failed to convert subscribed image: {exc}")
            return

        result = self._segmentor.infer(img)
        msg = ObjectList()
        msg.objects = [self._lego_object_to_mir_object(det) for det in result.detections]
        self._object_list_pub.publish(msg)
        self._publish_overlay(img, result.detections, image_msg.header)

    def _publish_overlay(self, image_rgb, detections, header) -> None:
        """Render + publish an annotated overlay when at least one subscriber is present."""
        if self._viz_pub.get_subscription_count() == 0:
            return
        try:
            overlay = render_overlay(image_rgb, detections)
            out_msg = self._bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
            out_msg.header = header
            self._viz_pub.publish(out_msg)
        except Exception as exc:
            self._log.warn(f"Failed to publish segmentation overlay: {exc}")

    def _lego_object_to_mir_object(self, d: LegoObject) -> Object:
        """Convert one detection to mir_interfaces/Object with ROI from bbox_xyxy."""
        x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)

        obj = Object()
        obj.name = d.class_name
        obj.probability = float(d.score)
        obj.roi.x_offset = int(x1)
        obj.roi.y_offset = int(y1)
        obj.roi.width = int(x2 - x1)
        obj.roi.height = int(y2 - y1)

        return obj


def _lego_object_to_msg(d: LegoObject) -> DetectedObject:
    """Build a ROS ``DetectedObject`` from a core :class:`LegoObject`."""
    msg = DetectedObject()
    msg.obj_id = int(d.obj_id)
    msg.class_id = int(d.class_id)
    msg.class_name = d.class_name
    msg.score = float(d.score)
    msg.bbox_xyxy = [float(v) for v in d.bbox_xyxy]
    msg.bbox_xyxyn = [float(v) for v in d.bbox_xyxyn]
    if d.mask_polygon_xy is not None:
        msg.mask_polygon_xy = [
            Point32(x=float(p[0]), y=float(p[1]), z=0.0) for p in d.mask_polygon_xy
        ]
    else:
        msg.mask_polygon_xy = []
    return msg


def main() -> None:
    """Entry point for the ``lego_segmentation_node`` console script."""
    rclpy.init()
    node = LegoSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
