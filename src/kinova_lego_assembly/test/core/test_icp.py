"""Tests for ROS-independent constrained brick ICP."""

from __future__ import annotations

import numpy as np
import pytest
from bitbots_lego_assembly.core.icp import (
    BRICK_HALF_HEIGHT_M,
    BrickModel,
    crop_observed_brick,
    refine_brick_pose,
)
from scipy.spatial import cKDTree
from transforms3d.euler import euler2mat


def _box_surface(size_x_m: float, size_y_m: float) -> BrickModel:
    """Create a dense synthetic top and wall surface."""
    x = np.linspace(-size_x_m / 2.0, size_x_m / 2.0, 33)
    y = np.linspace(-size_y_m / 2.0, size_y_m / 2.0, 17)
    z = np.linspace(-BRICK_HALF_HEIGHT_M, BRICK_HALF_HEIGHT_M, 13)
    top_x, top_y = np.meshgrid(x, y)
    surfaces = [
        np.column_stack(
            (
                top_x.ravel(),
                top_y.ravel(),
                np.full(top_x.size, BRICK_HALF_HEIGHT_M),
            )
        )
    ]
    for wall_x_m in (-size_x_m / 2.0, size_x_m / 2.0):
        wall_y, wall_z = np.meshgrid(y, z)
        surfaces.append(
            np.column_stack(
                (
                    np.full(wall_y.size, wall_x_m),
                    wall_y.ravel(),
                    wall_z.ravel(),
                )
            )
        )
    for wall_y_m in (-size_y_m / 2.0, size_y_m / 2.0):
        wall_x, wall_z = np.meshgrid(x, z)
        surfaces.append(
            np.column_stack(
                (
                    wall_x.ravel(),
                    np.full(wall_x.size, wall_y_m),
                    wall_z.ravel(),
                )
            )
        )
    points = np.vstack(surfaces)
    return BrickModel(size_x_m, size_y_m, points, cKDTree(points))


@pytest.mark.parametrize("size_x_m", [0.032, 0.064])
def test_refine_brick_pose_recovers_planar_offset(size_x_m: float) -> None:
    model = _box_surface(size_x_m, 0.032)
    table_plane_z_base_frame_m = -0.025
    expected = np.eye(4)
    expected[:3, :3] = euler2mat(0.0, 0.0, 0.18)
    expected[:3, 3] = (
        0.52,
        -0.24,
        table_plane_z_base_frame_m + BRICK_HALF_HEIGHT_M,
    )
    model_homogeneous = np.column_stack((model.points, np.ones(model.points.shape[0])))
    observed = (expected @ model_homogeneous.T).T[:, :3]

    seed = expected.copy()
    seed[:3, :3] = euler2mat(0.0, 0.0, 0.10)
    seed[0, 3] -= 0.004
    seed[1, 3] += 0.003
    result = refine_brick_pose(
        observed,
        seed,
        model,
        table_plane_z_base_frame_m,
        coarse_correspondence_max_distance_m=0.02,
        fine_correspondence_max_distance_m=0.004,
        max_iterations_per_stage=80,
        voxel_size_m=0.001,
    )

    assert result.fitness > 0.95
    assert result.inlier_rmse_m < 0.002
    assert result.model_to_base[:2, 3] == pytest.approx(
        expected[:2, 3],
        abs=0.002,
    )


def test_crop_observed_brick_uses_oriented_4x2_footprint() -> None:
    model = _box_surface(0.064, 0.032)
    table_plane_z_base_frame_m = -0.025
    seed = np.eye(4)
    seed[:3, :3] = euler2mat(0.0, 0.0, np.pi / 2.0)
    seed[:3, 3] = (
        0.5,
        -0.2,
        table_plane_z_base_frame_m + BRICK_HALF_HEIGHT_M,
    )
    inside = np.array(
        [
            [0.5, -0.225, -0.010],
            [0.5, -0.215, -0.010],
            [0.5, -0.205, -0.010],
        ]
    )
    distractor = np.array([[0.54, -0.2, -0.010]])
    table = np.array([[0.5, -0.2, table_plane_z_base_frame_m]])
    observed = crop_observed_brick(
        np.vstack((inside, distractor, table)),
        seed,
        model,
        table_plane_z_base_frame_m,
        footprint_padding_from_seed_m=0.001,
        clearance_above_table_plane_m=0.003,
        cluster_neighbor_radius_m=0.012,
        cluster_min_points=1,
    )

    assert observed.shape == (3, 3)
    assert not np.any(np.all(np.isclose(observed, distractor[0]), axis=1))
