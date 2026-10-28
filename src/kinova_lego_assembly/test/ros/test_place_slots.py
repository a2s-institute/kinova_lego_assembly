"""Tick tests for place-index selection, resume, and placement recording."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

py_trees = pytest.importorskip("py_trees")
behaviours = pytest.importorskip(
    "kinova_lego_assembly.ros.behaviours.behaviours",
    reason="requires built ROS interfaces (kinova_lego_assembly_interfaces)",
)
context_mod = pytest.importorskip("kinova_lego_assembly.ros.behaviours.context")
placement = pytest.importorskip("kinova_lego_assembly.ros.placement")

from geometry_msgs.msg import PoseStamped

from kinova_lego_assembly.core.blueprint import ModelRecipe, ModelStep

PlaceSlot = placement.PlaceSlot
PlacementRecord = placement.PlacementRecord


class _Logger:
    def info(self, *_a, **_k) -> None: ...


def _step(index: int) -> ModelStep:
    return ModelStep(index=index, brick_id=1, brick_name="2x2_red")


def _recipe(name: str, bricks: int = 2) -> ModelRecipe:
    return ModelRecipe(id=1, name=name, steps=[_step(i) for i in range(bricks)])


def _record(step_index: int) -> PlacementRecord:
    return PlacementRecord(
        step_index=step_index, brick_id=1, brick_name="2x2_red", place_pose=PoseStamped()
    )


@pytest.fixture
def board():
    py_trees.blackboard.Blackboard.clear()
    client = py_trees.blackboard.Client(name="test", namespace="assembly")
    for key in ("placement_record", "place_index", "recipe", "place_pose"):
        client.register_key(key=key, access=py_trees.common.Access.WRITE)
    client.placement_record = {}
    return client


@pytest.fixture
def context():
    return SimpleNamespace(
        params=context_mod.OrchestratorParams(place_joint_positions=[[0.0]] * 4),
        node=SimpleNamespace(get_logger=lambda: _Logger()),
        status_pub=None,
        total_bricks=0,
        place_index=1,
    )


def _tick(behaviour):
    behaviour.tick_once()
    return behaviour.status


def test_claims_first_free_index(board, context) -> None:
    board.recipe = _recipe("battery")
    board.placement_record = {1: PlaceSlot("e_stop", complete=True)}

    assert _tick(behaviours.SelectPlaceIndex(context)) == py_trees.common.Status.SUCCESS
    assert board.place_index == context.place_index == 2
    assert board.placement_record[2] == PlaceSlot("battery")


def test_partial_build_blocks_other_models(board, context) -> None:
    board.recipe = _recipe("e_stop")
    board.placement_record = {1: PlaceSlot("battery", records=(_record(0),))}

    _tick(behaviours.SelectPlaceIndex(context))

    assert board.place_index == 2


def test_same_model_resumes_partial_build(board, context) -> None:
    board.recipe = _recipe("battery")
    partial = PlaceSlot("battery", records=(_record(0),))
    board.placement_record = {1: PlaceSlot("battery", complete=True), 3: partial}

    _tick(behaviours.SelectPlaceIndex(context))

    assert board.place_index == 3
    assert board.placement_record[3] is partial


def test_fails_when_no_index_available(board, context) -> None:
    board.recipe = _recipe("battery")
    board.placement_record = {i: PlaceSlot("e_stop", complete=True) for i in range(1, 5)}

    assert _tick(behaviours.SelectPlaceIndex(context)) == py_trees.common.Status.FAILURE


def test_brick_already_placed_guard(board, context) -> None:
    board.placement_record = {1: PlaceSlot("battery", records=(_record(0),))}

    assert _tick(behaviours.BrickAlreadyPlaced(context, _step(0))) == (
        py_trees.common.Status.SUCCESS
    )
    assert _tick(behaviours.BrickAlreadyPlaced(context, _step(1))) == (
        py_trees.common.Status.FAILURE
    )


def test_pre_snap_uses_current_place_pose_before_recording(board, context) -> None:
    board.placement_record = {1: PlaceSlot("battery")}
    place_pose = PoseStamped()
    place_pose.pose.position.z = 0.2
    board.place_pose = place_pose
    board.register_key(key="pre_snap_pose", access=py_trees.common.Access.READ)

    behaviour = behaviours.ComputePreSnapPose(context, _step(0))

    assert _tick(behaviour) == py_trees.common.Status.SUCCESS
    assert board.pre_snap_pose.pose.position.z == pytest.approx(
        0.2 + context.params.pre_snap_offset_m
    )


def test_record_placement_appends_to_slot(board, context) -> None:
    board.placement_record = {1: PlaceSlot("battery", records=(_record(0),))}
    board.place_pose = PoseStamped()

    assert _tick(behaviours.RecordPlacement(context, _step(1))) == py_trees.common.Status.SUCCESS
    assert board.placement_record[1].placed_steps() == {0, 1}
