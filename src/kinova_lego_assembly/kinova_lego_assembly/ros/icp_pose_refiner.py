"""ROS adapter for tabletop brick pose refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer
from tf_transformations import quaternion_from_matrix, quaternion_matrix
from visualization_msgs.msg import Marker, MarkerArray

from kinova_lego_assembly.core.icp import (
    BRICK_HALF_HEIGHT_M,
    BrickModel,
    crop_observed_brick,
    load_brick_model,
    refine_brick_pose,
)


@dataclass(frozen=True)
class IcpRefinementParams:
    """ROS and algorithm settings for brick pose refinement."""

    enabled: bool = True
    point_cloud_topic: str = "/camera/depth/color/points"
    marker_topic: str = "/icp_pose_refinement/pose_markers"
    point_cloud_timeout_s: float = 5.0
    transform_timeout_s: float = 5.0
    table_plane_z_base_frame_m: float = -0.025
    footprint_padding_from_seed_m: float = 0.015
    clearance_above_table_plane_m: float = 0.006
    cluster_neighbor_radius_m: float = 0.004
    cluster_min_points: int = 5
    coarse_correspondence_max_distance_m: float = 0.02
    fine_correspondence_max_distance_m: float = 0.006
    max_iterations_per_stage: int = 80
    min_fitness: float = 0.25
    max_seed_correction_2x2_m: float = 0.015
    voxel_size_m: float = 0.001
    model_sample_count: int = 30_000
    marker_lifetime_s: float = 10.0


@dataclass(frozen=True)
class RefinedPose:
    """Accepted refinement associated with its input candidate index."""

    candidate_index: int
    pose: PoseStamped
    fitness: float
    inlier_rmse_m: float
    seed_correction_m: float


class IcpPoseRefiner:
    """Transform point clouds and seeds around the ROS-independent ICP core."""

    def __init__(
        self,
        node: Node,
        tf_buffer: Buffer,
        base_frame: str,
        params: IcpRefinementParams,
    ) -> None:
        """Load brick models and create the diagnostic marker publisher."""
        self._node = node
        self._tf_buffer = tf_buffer
        self._base_frame = str(base_frame)
        self.params = params
        positive_parameters = {
            "point_cloud_timeout_s": params.point_cloud_timeout_s,
            "transform_timeout_s": params.transform_timeout_s,
            "footprint_padding_from_seed_m": (params.footprint_padding_from_seed_m),
            "cluster_neighbor_radius_m": params.cluster_neighbor_radius_m,
            "coarse_correspondence_max_distance_m": (params.coarse_correspondence_max_distance_m),
            "fine_correspondence_max_distance_m": (params.fine_correspondence_max_distance_m),
            "max_seed_correction_2x2_m": params.max_seed_correction_2x2_m,
            "voxel_size_m": params.voxel_size_m,
        }
        self._MODEL_SPECS = {
            "2x2": (0.032, 0.032),
            "4x2": (0.064, 0.032),
        }
        invalid = [name for name, value in positive_parameters.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"ICP parameters must be positive: {', '.join(invalid)}")
        if params.clearance_above_table_plane_m < 0.0:
            raise ValueError("clearance_above_table_plane_m must be non-negative")
        if not 0.0 <= params.min_fitness <= 1.0:
            raise ValueError("min_fitness must be within [0, 1]")
        if params.cluster_min_points < 1 or params.max_iterations_per_stage < 1:
            raise ValueError("cluster_min_points and max_iterations_per_stage must be positive")
        if params.model_sample_count < 1 or params.marker_lifetime_s < 0.0:
            raise ValueError(
                "model_sample_count must be positive and marker_lifetime_s must be non-negative"
            )
        mesh_directory = (
            Path(get_package_share_directory("kinova_lego_assembly"))
            / "assets/meshes/lego_duplo_bricks"
        )
        self._models: dict[str, BrickModel] = {}
        for brick_type, (size_x_m, size_y_m) in self._MODEL_SPECS.items():
            mesh_path = mesh_directory / f"LEGO_Duplo_brick_{brick_type}.stl"
            self._models[brick_type] = load_brick_model(
                mesh_path,
                size_x_m,
                size_y_m,
                params.model_sample_count,
            )

        marker_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._marker_publisher = node.create_publisher(
            MarkerArray,
            params.marker_topic,
            marker_qos,
        )

    def refine_candidates(
        self,
        cloud: PointCloud2,
        seed_poses: list[PoseStamped],
        brick_type: str,
    ) -> list[RefinedPose]:
        """Refine candidate poses against one point-cloud snapshot."""
        model = self._models.get(brick_type)
        if model is None:
            self._node.get_logger().error(f"ICP does not support brick type '{brick_type}'")
            return []

        records = point_cloud2.read_points(
            cloud,
            field_names=["x", "y", "z"],
            skip_nans=True,
        )
        scene = np.stack(
            (records["x"], records["y"], records["z"]),
            axis=-1,
        ).astype(np.float64)
        cloud_to_base = self._lookup_matrix(
            cloud.header.frame_id,
            cloud.header.stamp,
        )
        scene_homogeneous = np.column_stack((scene, np.ones(scene.shape[0])))
        scene_base = (cloud_to_base @ scene_homogeneous.T).T[:, :3]

        accepted: list[RefinedPose] = []
        seed_matrices: list[np.ndarray] = []
        for candidate_index, seed_pose in enumerate(seed_poses):
            try:
                seed_model_to_base = self._pose_matrix_in_base(seed_pose)
            except Exception as exc:
                self._node.get_logger().warning(
                    f"ICP skipped candidate {candidate_index}: seed transform failed: {exc}"
                )
                continue
            seed_matrices.append(seed_model_to_base)
            observed = crop_observed_brick(
                scene_base,
                seed_model_to_base,
                model,
                self.params.table_plane_z_base_frame_m,
                self.params.footprint_padding_from_seed_m,
                self.params.clearance_above_table_plane_m,
                self.params.cluster_neighbor_radius_m,
                self.params.cluster_min_points,
            )
            if observed.shape[0] < 20:
                self._node.get_logger().warning(
                    f"ICP skipped candidate {candidate_index}: "
                    f"ROI has {observed.shape[0]} clustered points"
                )
                continue

            result = refine_brick_pose(
                observed,
                seed_model_to_base,
                model,
                self.params.table_plane_z_base_frame_m,
                self.params.coarse_correspondence_max_distance_m,
                self.params.fine_correspondence_max_distance_m,
                self.params.max_iterations_per_stage,
                self.params.voxel_size_m,
            )
            seed_correction_m = float(
                np.linalg.norm(result.model_to_base[:2, 3] - seed_model_to_base[:2, 3])
            )
            correction_scale = max(model.size_x_m, model.size_y_m) / 0.032
            max_seed_correction_m = self.params.max_seed_correction_2x2_m * correction_scale
            if (
                result.fitness < self.params.min_fitness
                or seed_correction_m > max_seed_correction_m
            ):
                self._node.get_logger().warning(
                    f"ICP rejected candidate {candidate_index}: "
                    f"fitness={result.fitness:.3f}, "
                    f"rmse={result.inlier_rmse_m:.4f} m, "
                    f"seed correction={seed_correction_m:.4f} m "
                    f"(limit={max_seed_correction_m:.4f} m)"
                )
                continue
            accepted.append(
                RefinedPose(
                    candidate_index=candidate_index,
                    pose=self._matrix_to_pose(result.model_to_base, cloud),
                    fitness=result.fitness,
                    inlier_rmse_m=result.inlier_rmse_m,
                    seed_correction_m=seed_correction_m,
                )
            )

        self._publish_markers(seed_matrices, accepted, model, cloud)
        return accepted

    def _lookup_matrix(self, source_frame: str, stamp) -> np.ndarray:
        """Return source-to-base homogeneous transform at the message stamp."""
        if source_frame == self._base_frame:
            return np.eye(4, dtype=np.float64)
        transform = self._tf_buffer.lookup_transform(
            self._base_frame,
            source_frame,
            Time.from_msg(stamp),
            timeout=Duration(nanoseconds=int(self.params.transform_timeout_s * 1e9)),
        ).transform
        matrix = quaternion_matrix(
            [
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            ]
        )
        matrix[:3, 3] = (
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        )
        return matrix

    def _pose_matrix_in_base(self, pose: PoseStamped) -> np.ndarray:
        """Convert one stamped pose to a model-to-base homogeneous matrix."""
        model_to_parent = quaternion_matrix(
            [
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ]
        )
        model_to_parent[:3, 3] = (
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )
        return self._lookup_matrix(pose.header.frame_id, pose.header.stamp) @ model_to_parent

    def _matrix_to_pose(
        self,
        model_to_base: np.ndarray,
        cloud: PointCloud2,
    ) -> PoseStamped:
        """Convert a homogeneous model pose to a base-frame ROS pose."""
        quaternion = quaternion_from_matrix(model_to_base)
        pose = PoseStamped()
        pose.header.frame_id = self._base_frame
        pose.header.stamp = cloud.header.stamp
        pose.pose.position.x = float(model_to_base[0, 3])
        pose.pose.position.y = float(model_to_base[1, 3])
        pose.pose.position.z = float(model_to_base[2, 3])
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        return pose

    def _publish_markers(
        self,
        seed_matrices: list[np.ndarray],
        accepted: list[RefinedPose],
        model: BrickModel,
        cloud: PointCloud2,
    ) -> None:
        """Publish red seed cubes and green accepted-refinement cubes."""
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers = [clear]
        groups = (
            ("mmor_seed", seed_matrices, (1.0, 0.0, 0.0)),
            (
                "icp_refined",
                [self._pose_matrix_in_base(result.pose) for result in accepted],
                (0.0, 1.0, 0.0),
            ),
        )
        for namespace, matrices, color in groups:
            for marker_id, matrix in enumerate(matrices):
                quaternion = quaternion_from_matrix(matrix)
                marker = Marker()
                marker.header.frame_id = self._base_frame
                marker.header.stamp = cloud.header.stamp
                marker.ns = namespace
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = float(matrix[0, 3])
                marker.pose.position.y = float(matrix[1, 3])
                marker.pose.position.z = float(matrix[2, 3])
                marker.pose.orientation.x = quaternion[0]
                marker.pose.orientation.y = quaternion[1]
                marker.pose.orientation.z = quaternion[2]
                marker.pose.orientation.w = quaternion[3]
                marker.scale.x = model.size_x_m
                marker.scale.y = model.size_y_m
                marker.scale.z = 2.0 * BRICK_HALF_HEIGHT_M
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 0.45
                if self.params.marker_lifetime_s > 0.0:
                    marker.lifetime = Duration(
                        nanoseconds=int(self.params.marker_lifetime_s * 1e9)
                    ).to_msg()
                markers.append(marker)
        message = MarkerArray()
        message.markers = markers
        self._marker_publisher.publish(message)
