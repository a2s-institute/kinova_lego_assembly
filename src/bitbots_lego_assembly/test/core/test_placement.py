"""Tests for the placement stacking geometry and record type."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from bitbots_lego_assembly.core.blueprint import ModelStep
from bitbots_lego_assembly.core.placement import PlacementError
from bitbots_lego_assembly.ros.placement import (
    BRICK_HALF_WIDTH_Z,
    PREPLACE_OFFSET_Z,
    PRIOR_POSE_OFFSET_Z,
    FkPlacementProvider,
    PlacementRecord,
)
from geometry_msgs.msg import PoseStamped


def _pose(x: float, y: float, z: float) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    pose.pose.orientation.w = 1.0
    return pose


def _step() -> ModelStep:
    return ModelStep(index=0, brick_id=8, brick_name="4x2_yellow")


def _context() -> SimpleNamespace:
    """Minimal context: the stacking branch only reads ``params`` and never FK."""
    return SimpleNamespace(params=SimpleNamespace())


def test_stacked_place_pose_offsets_prior_in_z() -> None:
    prior = _pose(0.3, -0.5, 0.4)
    provider = FkPlacementProvider(context=_context())

    preplace_pose, place_pose = provider.compute_poses(_step(), prior_place_pose=prior)

    expected_place_z = 0.4 + BRICK_HALF_WIDTH_Z + PRIOR_POSE_OFFSET_Z
    assert place_pose.pose.position.z == expected_place_z
    assert place_pose.pose.position.x == prior.pose.position.x
    assert place_pose.pose.position.y == prior.pose.position.y
    assert preplace_pose.pose.position.z == expected_place_z + PREPLACE_OFFSET_Z


def test_stacked_place_pose_does_not_mutate_prior() -> None:
    prior = _pose(0.3, -0.5, 0.4)
    prior_copy = copy.deepcopy(prior)
    provider = FkPlacementProvider(context=_context())

    provider.compute_poses(_step(), prior_place_pose=prior)

    assert prior.pose.position.z == prior_copy.pose.position.z


def test_first_place_pose_selects_place_index_joint_configuration() -> None:
    captured: dict[str, list[float]] = {}

    class FakeFkClient:
        def compute_pose_from_joints(self, *, joint_names, joint_positions, fk_link):
            captured["joint_positions"] = joint_positions
            return _pose(0.3, -0.5, 0.4)

    place_configs = [[float(i)] for i in range(4)]
    context = SimpleNamespace(
        params=SimpleNamespace(
            home_joint_names=["joint_1"], pose_link="tool", place_joint_positions=place_configs
        ),
        fk_client=FakeFkClient(),
    )
    provider = FkPlacementProvider(context=context)

    provider.compute_poses(_step(), place_index=4)

    assert captured["joint_positions"] == place_configs[3]


def test_first_place_pose_rejects_unconfigured_place_index() -> None:
    context = SimpleNamespace(
        params=SimpleNamespace(
            home_joint_names=["joint_1"], pose_link="tool", place_joint_positions=[[0.0]]
        ),
        fk_client=None,
    )
    provider = FkPlacementProvider(context=context)

    with pytest.raises(PlacementError):
        provider.compute_poses(_step(), place_index=2)


def test_placement_record() -> None:
    pose = _pose(0.0, -0.5, 0.4)
    record = PlacementRecord(step_index=2, brick_id=8, brick_name="4x2_yellow", place_pose=pose)
    assert record.step_index == 2
    assert record.brick_name == "4x2_yellow"
    assert record.place_pose is pose
