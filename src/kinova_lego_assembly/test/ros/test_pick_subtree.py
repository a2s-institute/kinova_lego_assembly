"""Standalone tick test for the pick subtree.

Demonstrates the modularity goal: one pick subtree can be built and
ticked in isolation with stubbed ROS adapters, no running nodes. Skipped
when the ROS message packages (built via colcon) are not importable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

py_trees = pytest.importorskip("py_trees")
behaviours = pytest.importorskip(
    "kinova_lego_assembly.ros.behaviours.behaviours",
    reason="requires built ROS interfaces (kinova_lego_assembly_interfaces)",
)
tree_mod = pytest.importorskip("kinova_lego_assembly.ros.behaviours.tree")
context_mod = pytest.importorskip("kinova_lego_assembly.ros.behaviours.context")

from geometry_msgs.msg import PoseStamped
from mir_interfaces.msg import Object, ObjectList

from kinova_lego_assembly.core.blueprint import ModelStep


class _Logger:
    def info(self, *_a, **_k) -> None: ...
    def warning(self, *_a, **_k) -> None: ...
    def error(self, *_a, **_k) -> None: ...
    def debug(self, *_a, **_k) -> None: ...


def _pose(frame: str = "base_link") -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = frame
    pose.pose.orientation.w = 1.0
    return pose


def _make_context(icp_enabled: bool = False):
    params = context_mod.OrchestratorParams(base_frame="base_link")
    params.icp_refinement = context_mod.IcpRefinementParams(enabled=icp_enabled)

    obj = Object()
    obj.name = "2x2_red"
    obj.probability = 0.9
    obj.pose = _pose()
    object_list = ObjectList()
    object_list.objects = [obj]

    return context_mod.OrchestratorContext(
        node=SimpleNamespace(get_logger=lambda: _Logger()),
        spinner=None,
        params=params,
        blueprint=None,
        move_client=SimpleNamespace(
            move_to_pose=lambda **_k: (True, "ok"),
            can_plan_to_pose=lambda **_k: (True, "ok"),
            wait_until_settled=lambda **_k: True,
        ),
        fk_client=None,
        sensor=SimpleNamespace(get_point_cloud_ros=lambda *_a, **_k: object()),
        gripper=SimpleNamespace(
            open=lambda timeout_s=0.0: True,
            close=lambda timeout_s=0.0: True,
            move_to=lambda _p, timeout_s=0.0: True,
        ),
        detection_client=SimpleNamespace(detect=lambda timeout_s=0.0: True),
        object_list_reader=SimpleNamespace(get_fresh=lambda _t, timeout_s=0.0: object_list),
        segment_client=None,
        tf_buffer=SimpleNamespace(),
        icp_pose_refiner=SimpleNamespace(
            refine_candidates=lambda _cloud, poses, _brick_type: [
                SimpleNamespace(
                    candidate_index=0,
                    pose=poses[0],
                    fitness=0.95,
                    inlier_rmse_m=0.001,
                )
            ]
        ),
        status_pub=None,
    )


def test_pick_subtree_succeeds(monkeypatch) -> None:
    # Grasp poses are generated in-process from tf; stub the geometry so the
    # subtree can tick without a live tf tree.
    def _candidate(rank: int, x: float):
        grasp_pose = _pose()
        grasp_pose.pose.position.x = x
        pregrasp_pose = _pose()
        pregrasp_pose.pose.position.x = x
        return SimpleNamespace(
            rank=rank,
            z_rotation=0.0,
            angular_distance=float(rank),
            grasp_pose=grasp_pose,
            pregrasp_pose=pregrasp_pose,
        )

    candidates = (_candidate(0, 1.0), _candidate(1, 2.0))
    monkeypatch.setattr(
        behaviours,
        "compute_grasp_pregrasp_candidates",
        lambda *_a, **_k: candidates,
    )

    py_trees.blackboard.Blackboard.clear()
    context = _make_context()
    plan_checks = []

    def _check_plan(**kwargs):
        plan_checks.append(kwargs)
        return (len(plan_checks) > 1, "candidate plan result")

    context.move_client.can_plan_to_pose = _check_plan
    step = ModelStep(index=0, brick_id=1, brick_name="2x2_red")

    perceive = tree_mod.build_perceive_brick_pile_subtree(context, step)
    for _ in range(10):
        perceive.tick_once()
        if perceive.status != py_trees.common.Status.RUNNING:
            break
    assert perceive.status == py_trees.common.Status.SUCCESS

    subtree = tree_mod.build_pick_subtree(context, step)
    for _ in range(10):
        subtree.tick_once()
        if subtree.status != py_trees.common.Status.RUNNING:
            break

    assert subtree.status == py_trees.common.Status.SUCCESS
    assert [check["target_pose"].pose.position.x for check in plan_checks] == [1.0, 2.0]
    assert all(
        check["max_velocity"] == context.params.fast_velocity_scaling for check in plan_checks
    )


def test_pick_subtree_refines_before_generating_grasp(monkeypatch) -> None:
    calls: list[str] = []

    def _compute_grasp(*_args, **_kwargs):
        calls.append("grasp")
        return (
            SimpleNamespace(
                rank=0,
                z_rotation=0.0,
                angular_distance=0.0,
                grasp_pose=_pose(),
                pregrasp_pose=_pose(),
            ),
        )

    monkeypatch.setattr(
        behaviours,
        "compute_grasp_pregrasp_candidates",
        _compute_grasp,
    )
    context = _make_context(icp_enabled=True)
    original_refine = context.icp_pose_refiner.refine_candidates

    def _refine(*args):
        calls.append("icp")
        return original_refine(*args)

    context.icp_pose_refiner.refine_candidates = _refine
    py_trees.blackboard.Blackboard.clear()
    step = ModelStep(
        index=0,
        brick_id=1,
        brick_name="2x2_red",
    )

    perceive = tree_mod.build_perceive_brick_pile_subtree(context, step)
    for _ in range(10):
        perceive.tick_once()
        if perceive.status != py_trees.common.Status.RUNNING:
            break
    assert perceive.status == py_trees.common.Status.SUCCESS

    subtree = tree_mod.build_pick_subtree(context, step)
    for _ in range(10):
        subtree.tick_once()
        if subtree.status != py_trees.common.Status.RUNNING:
            break

    assert subtree.status == py_trees.common.Status.SUCCESS
    assert calls == ["icp", "grasp"]


def _run_grasp_search(monkeypatch, plannable_from_check: int | None):
    grasp_mod = pytest.importorskip("kinova_lego_assembly.ros.grasp")
    candidates = tuple(
        grasp_mod.GraspPoseCandidate(rank, float(rank), 0.0, _pose(), _pose()) for rank in (0, 1)
    )
    monkeypatch.setattr(
        behaviours,
        "compute_grasp_pregrasp_candidates",
        lambda *_a, **_k: candidates,
    )
    py_trees.blackboard.Blackboard.clear()
    context = _make_context()
    plan_checks = []

    def _check_plan(**kwargs):
        plan_checks.append(kwargs["target_pose"].pose.position.x)
        ok = plannable_from_check is not None and len(plan_checks) >= plannable_from_check
        return ok, "candidate plan result"

    context.move_client.can_plan_to_pose = _check_plan
    step = ModelStep(index=0, brick_id=1, brick_name="2x2_red")

    perceive = tree_mod.build_perceive_brick_pile_subtree(context, step)
    for _ in range(10):
        perceive.tick_once()
        if perceive.status != py_trees.common.Status.RUNNING:
            break
    assert perceive.status == py_trees.common.Status.SUCCESS

    grasp = behaviours.GenerateGrasp(context, step, planner=tree_mod.PILZ_PTP)
    grasp.setup_with_descendants()
    grasp.tick_once()
    return grasp.status, plan_checks


def test_grasp_search_shifts_group_in_x_after_full_failure(monkeypatch) -> None:
    status, plan_checks = _run_grasp_search(monkeypatch, plannable_from_check=3)

    assert status == py_trees.common.Status.SUCCESS
    assert plan_checks == pytest.approx([0.0, 0.0, 0.002])


def test_grasp_search_tries_ik_seed_before_x_shift(monkeypatch) -> None:
    grasp_mod = pytest.importorskip("kinova_lego_assembly.ros.grasp")
    candidates = (grasp_mod.GraspPoseCandidate(0, 0.0, 0.0, _pose(), _pose()),)
    monkeypatch.setattr(
        behaviours,
        "compute_grasp_pregrasp_candidates",
        lambda *_a, **_k: candidates,
    )
    py_trees.blackboard.Blackboard.clear()
    context = _make_context()
    context.params.home_joint_names = ["joint_1", "joint_2"]
    calls = []
    seed = {"joint_1": 0.4, "joint_2": 0.9}

    def _check_plan(**kwargs):
        calls.append(("plain", kwargs["target_pose"].pose.position.x))
        return False, "no plan"

    def _solve(**kwargs):
        calls.append(("seeded", kwargs["seed_joint_positions"]))
        return {"joint_1": 0.5, "joint_2": 1.0, "finger": 0.0}, "ok"

    def _joint_goal(**kwargs):
        calls.append(("joint_check", kwargs["plan_only"]))
        return True, "ok"

    context.move_client.can_plan_to_pose = _check_plan
    context.move_client.solve_pose_from_seed = _solve
    context.move_client.move_to_joint_goal = _joint_goal
    step = ModelStep(index=0, brick_id=1, brick_name="2x2_red")

    perceive = tree_mod.build_perceive_brick_pile_subtree(context, step)
    for _ in range(10):
        perceive.tick_once()
        if perceive.status != py_trees.common.Status.RUNNING:
            break

    grasp = behaviours.GenerateGrasp(
        context, step, planner=tree_mod.PILZ_PTP, seed_joint_positions=seed
    )
    grasp.setup_with_descendants()
    grasp.tick_once()

    assert grasp.status == py_trees.common.Status.SUCCESS
    assert calls == [("plain", 0.0), ("seeded", seed), ("joint_check", True)]
    assert grasp.bb.pregrasp_joint_positions == {"joint_1": 0.5, "joint_2": 1.0}


def test_grasp_search_stops_after_max_shifts(monkeypatch) -> None:
    status, plan_checks = _run_grasp_search(monkeypatch, plannable_from_check=None)

    assert status == py_trees.common.Status.FAILURE
    expected = [shift * 0.002 for shift in range(6) for _ in range(2)]
    assert plan_checks == pytest.approx(expected)


def test_shift_candidate_x_follows_grasp_frame() -> None:
    grasp_mod = pytest.importorskip("kinova_lego_assembly.ros.grasp")
    grasp_pose = _pose()
    grasp_pose.pose.orientation.z = 2.0**-0.5
    grasp_pose.pose.orientation.w = 2.0**-0.5
    pregrasp_pose = _pose()
    pregrasp_pose.pose.position.z = 0.1
    candidate = grasp_mod.GraspPoseCandidate(0, 0.0, 0.0, grasp_pose, pregrasp_pose)

    shifted = grasp_mod.shift_candidate_x(candidate, 0.002)

    for pose, z in ((shifted.grasp_pose, 0.0), (shifted.pregrasp_pose, 0.1)):
        position = pose.pose.position
        assert (position.x, position.y, position.z) == pytest.approx((0.0, 0.002, z))
    assert shifted.grasp_pose.pose.orientation == grasp_pose.pose.orientation
    assert candidate.grasp_pose.pose.position.y == 0.0
