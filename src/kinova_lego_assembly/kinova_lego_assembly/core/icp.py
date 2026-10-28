"""ROS-independent constrained ICP for tabletop Duplo bricks."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from transforms3d.euler import euler2mat

BRICK_BODY_HEIGHT_M = 0.0192
BRICK_STUD_HEIGHT_M = 0.0044
BRICK_HALF_HEIGHT_M = (BRICK_BODY_HEIGHT_M + BRICK_STUD_HEIGHT_M) / 2.0
MESH_MM_TO_M = 0.001


@dataclass(frozen=True)
class BrickModel:
    """Sampled observable surface and dimensions for one brick type."""

    size_x_m: float
    size_y_m: float
    points: np.ndarray
    tree: cKDTree


@dataclass(frozen=True)
class IcpResult:
    """Result of fitting one brick model to observed base-frame points."""

    model_to_base: np.ndarray
    fitness: float
    inlier_rmse_m: float


def load_brick_model(
    mesh_path: Path,
    size_x_m: float,
    size_y_m: float,
    sample_count: int,
) -> BrickModel:
    """Load a centered STL and sample surfaces visible from above.

    Parameters
    ----------
    mesh_path : pathlib.Path
        Brick STL path. Mesh units must be millimetres.
    size_x_m, size_y_m : float
        Brick footprint dimensions along model X and Y in metres.
    sample_count : int
        Number of points sampled before hidden surfaces are discarded.

    Returns
    -------
    BrickModel
        Filtered surface sample and nearest-neighbour index.
    """
    if not mesh_path.is_file():
        raise FileNotFoundError(f"brick mesh not found: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"brick mesh has no triangles: {mesh_path}")
    mesh.scale(MESH_MM_TO_M, center=(0.0, 0.0, 0.0))
    sampled = mesh.sample_points_uniformly(number_of_points=int(sample_count))
    points = np.asarray(sampled.points)

    plate_top_m = -BRICK_HALF_HEIGHT_M + BRICK_BODY_HEIGHT_M
    surface_margin_m = 0.0005
    observable = (
        (np.abs(points[:, 0]) > size_x_m / 2.0 - surface_margin_m)
        | (np.abs(points[:, 1]) > size_y_m / 2.0 - surface_margin_m)
        | (points[:, 2] > plate_top_m - surface_margin_m)
    ) & (points[:, 2] > -BRICK_HALF_HEIGHT_M + surface_margin_m)
    points = points[observable]
    return BrickModel(
        size_x_m=float(size_x_m),
        size_y_m=float(size_y_m),
        points=points,
        tree=cKDTree(points),
    )


def crop_observed_brick(
    scene_base_xyz: np.ndarray,
    seed_model_to_base: np.ndarray,
    model: BrickModel,
    table_plane_z_base_frame_m: float,
    footprint_padding_from_seed_m: float,
    clearance_above_table_plane_m: float,
    cluster_neighbor_radius_m: float,
    cluster_min_points: int,
) -> np.ndarray:
    """Crop and cluster scene points around one oriented seed footprint."""
    points = np.asarray(scene_base_xyz, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    local = (np.linalg.inv(seed_model_to_base) @ homogeneous.T).T[:, :3]
    in_footprint = (np.abs(local[:, 0]) <= model.size_x_m / 2.0 + footprint_padding_from_seed_m) & (
        np.abs(local[:, 1]) <= model.size_y_m / 2.0 + footprint_padding_from_seed_m
    )
    total_brick_height_m = 2.0 * BRICK_HALF_HEIGHT_M
    above_table = points[:, 2] >= (table_plane_z_base_frame_m + clearance_above_table_plane_m)
    below_crop_ceiling = points[:, 2] <= (table_plane_z_base_frame_m + total_brick_height_m + 0.006)
    cropped = points[in_footprint & above_table & below_crop_ceiling]
    if cropped.shape[0] == 0:
        return cropped

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cropped)
    labels = np.asarray(
        cloud.cluster_dbscan(
            eps=float(cluster_neighbor_radius_m),
            min_points=max(1, int(cluster_min_points)),
            print_progress=False,
        )
    )
    valid_labels = labels[labels >= 0]
    if valid_labels.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    cluster_ids, counts = np.unique(valid_labels, return_counts=True)
    largest_cluster = cluster_ids[int(np.argmax(counts))]
    return cropped[labels == largest_cluster]


def refine_brick_pose(
    observed_base_xyz: np.ndarray,
    seed_model_to_base: np.ndarray,
    model: BrickModel,
    table_plane_z_base_frame_m: float,
    coarse_correspondence_max_distance_m: float,
    fine_correspondence_max_distance_m: float,
    max_iterations_per_stage: int,
    voxel_size_m: float,
) -> IcpResult:
    """Fit x, y, and yaw while fixing upright pose and table height."""
    observed = np.asarray(observed_base_xyz, dtype=np.float64).reshape(-1, 3)
    observed = observed[np.isfinite(observed).all(axis=1)]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(observed)
    observed = np.asarray(cloud.voxel_down_sample(voxel_size=float(voxel_size_m)).points)
    homogeneous = np.column_stack((observed, np.ones(observed.shape[0])))

    model_center_z_base_frame_m = table_plane_z_base_frame_m + BRICK_HALF_HEIGHT_M
    x = float(seed_model_to_base[0, 3])
    y = float(seed_model_to_base[1, 3])
    yaw = atan2(seed_model_to_base[1, 0], seed_model_to_base[0, 0])

    for correspondence_max_distance_m in (
        float(coarse_correspondence_max_distance_m),
        float(fine_correspondence_max_distance_m),
    ):
        for _ in range(max(1, int(max_iterations_per_stage))):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = euler2mat(0.0, 0.0, yaw)
            pose[:3, 3] = (x, y, model_center_z_base_frame_m)
            in_model = (np.linalg.inv(pose) @ homogeneous.T).T[:, :3]
            distances, indices = model.tree.query(in_model, k=1)
            keep = distances <= correspondence_max_distance_m
            if int(keep.sum()) < 10:
                break

            model_correspondences = model.points[indices[keep]]
            model_homogeneous = np.column_stack(
                (
                    model_correspondences,
                    np.ones(model_correspondences.shape[0]),
                )
            )
            source_xy = (pose @ model_homogeneous.T).T[:, :2]
            target_xy = observed[keep][:, :2]
            source_mean = source_xy.mean(axis=0)
            target_mean = target_xy.mean(axis=0)
            source_centered = source_xy - source_mean
            target_centered = target_xy - target_mean
            delta_yaw = atan2(
                float(
                    np.sum(
                        source_centered[:, 0] * target_centered[:, 1]
                        - source_centered[:, 1] * target_centered[:, 0]
                    )
                ),
                float(
                    np.sum(
                        source_centered[:, 0] * target_centered[:, 0]
                        + source_centered[:, 1] * target_centered[:, 1]
                    )
                ),
            )
            rotation = np.array(
                [
                    [cos(delta_yaw), -sin(delta_yaw)],
                    [sin(delta_yaw), cos(delta_yaw)],
                ]
            )
            shift = target_mean - rotation @ source_mean
            moved_xy = rotation @ np.array([x, y]) + shift
            translation_step_m = float(np.hypot(moved_xy[0] - x, moved_xy[1] - y))
            x, y = float(moved_xy[0]), float(moved_xy[1])
            yaw += delta_yaw
            if translation_step_m < 1e-5 and abs(delta_yaw) < 1e-4:
                break

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = euler2mat(0.0, 0.0, yaw)
    pose[:3, 3] = (x, y, model_center_z_base_frame_m)
    in_model = (np.linalg.inv(pose) @ homogeneous.T).T[:, :3]
    distances, _ = model.tree.query(in_model, k=1)
    inliers = distances <= float(fine_correspondence_max_distance_m)
    return IcpResult(
        model_to_base=pose,
        fitness=float(inliers.mean()) if distances.size else 0.0,
        inlier_rmse_m=float(np.sqrt(np.mean(distances[inliers] ** 2)))
        if bool(inliers.any())
        else float("inf"),
    )
