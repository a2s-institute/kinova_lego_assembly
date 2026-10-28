"""Behaviour-tree factories for the LEGO assembly orchestrator.

The tree is built per goal (the recipe is known then), so the per-brick
work is unrolled into one labelled subtree per step. This keeps every
phase individually visible in the py_trees viewer and lets tests tick a
single pick subtree in isolation. All sequences use ``memory=True`` so a
completed child is not re-ticked while a later child runs.
"""

from __future__ import annotations

import py_trees

from bitbots_lego_assembly.core.blueprint import ModelRecipe, ModelStep
from bitbots_lego_assembly.core.orchestration import Phase
from bitbots_lego_assembly.ros.behaviours.behaviours import (
    BrickAlreadyPlaced,
    CaptureImage,
    ComputePlacePoses,
    ComputePreSnapPose,
    DetectObjects,
    GenerateGrasp,
    MovePoseFromBlackboard,
    MoveToJointGoal,
    RecordPlacement,
    RefineDetectedObjectPoses,
    SelectPlaceIndex,
    SetGripper,
    ValidateScene,
    WaitForArmSettled,
)
from bitbots_lego_assembly.ros.behaviours.context import OrchestratorContext
from bitbots_lego_assembly.ros.clients import GoalTolerance, MotionPlanner

PILZ_PTP = MotionPlanner("pilz_industrial_motion_planner", "PTP")
PILZ_LIN = MotionPlanner("pilz_industrial_motion_planner", "LIN")
OMPL_RRT_CONNECT = MotionPlanner("ompl", "RRTConnectkConfigDefault")
PRECISE_TOLERANCE = GoalTolerance(position_m=0.001, orientation_rad=0.001)
PREGRASP_IK_SEED = {
    "joint_1": 0.44346116247304734,
    "joint_2": 0.9128618468586028,
    "joint_3": -2.828527414806618,
    "joint_4": -1.055631782368879,
    "joint_5": -0.3030831783853394,
    "joint_6": -1.1456162671771297,
    "joint_7": 2.3170580063930855,
}


def build_home_subtree(context: OrchestratorContext, label: str) -> py_trees.behaviour.Behaviour:
    """Build a subtree that drives the arm to the home joint configuration."""
    params = context.params
    home = py_trees.composites.Sequence(name=label, memory=True)
    home.add_children(
        [
            MoveToJointGoal(
                context,
                name=f"{label}_move",
                joint_names=params.home_joint_names,
                joint_positions=params.home_joint_positions,
                velocity_scaling=params.fast_velocity_scaling,
                planner=OMPL_RRT_CONNECT,
                description="moving to home joint configuration (OMPL RRTConnect)",
                phase=Phase.MOVING_HOME,
            ),
        ]
    )
    return home


def build_perceive_brick_pile_subtree(
    context: OrchestratorContext, step: ModelStep
) -> py_trees.behaviour.Behaviour:
    """Build the perceive subtree for one recipe step.

    Sequence: detect the pile (retry up to 3 failures) -> ICP-refine poses
    of objects matching this brick. Writes ``brick_pose_candidates`` for the
    subsequent pick subtree.
    """
    perceive = py_trees.composites.Sequence(
        name=f"perceive_brick_pile_brick{step.index}_{step.brick_name}",
        memory=True,
    )
    perceive.add_children(
        [
            WaitForArmSettled(context, Phase.PICKING, brick_index=step.index),
            py_trees.decorators.Retry(
                name=f"DetectObjects_brick{step.index}_retry",
                child=DetectObjects(context, step),
                num_failures=3,
            ),
            py_trees.decorators.Retry(
                name=f"RefineDetectedObjectPoses_brick{step.index}_retry",
                child=RefineDetectedObjectPoses(context, step),
                num_failures=3,
            ),
        ]
    )
    return perceive


def build_pick_subtree(
    context: OrchestratorContext, step: ModelStep
) -> py_trees.behaviour.Behaviour:
    """Build the pick subtree for one recipe step.

    Sequence: grasp generation -> pregrasp -> approach grasp -> close gripper -> retreat to pregrasp
    Requires ``brick_pose_candidates`` from
    :func:`build_perceive_brick_pile_subtree`.
    """
    params = context.params
    label = f"pick_brick{step.index}_{step.brick_name}"
    pick = py_trees.composites.Sequence(name=label, memory=True)
    pick.add_children(
        [
            GenerateGrasp(
                context,
                step,
                planner=PILZ_PTP,
                seed_joint_positions=PREGRASP_IK_SEED,
                tolerance=PRECISE_TOLERANCE,
            ),
            MovePoseFromBlackboard(
                context,
                name=f"MoveToPregrasp_brick{step.index}",
                pose_key="pregrasp_pose",
                joint_goal_key="pregrasp_joint_positions",
                velocity_scaling=params.fast_velocity_scaling,
                planner=PILZ_PTP,
                step=step,
                description="moving to pregrasp",
                tolerance=PRECISE_TOLERANCE,
            ),
            MovePoseFromBlackboard(
                context,
                name=f"ApproachGrasp_brick{step.index}",
                pose_key="grasp_pose",
                velocity_scaling=params.slow_velocity_scaling,
                planner=PILZ_LIN,
                step=step,
                description="approaching grasp",
                tolerance=PRECISE_TOLERANCE,
            ),
            SetGripper(context, mode="close", brick_index=step.index, phase=Phase.PICKING),
            MovePoseFromBlackboard(
                context,
                name=f"RetreatToPregrasp_brick{step.index}",
                pose_key="pregrasp_pose",
                velocity_scaling=params.fast_velocity_scaling,
                planner=PILZ_LIN,
                step=step,
                description="retreating to pregrasp",
            ),
        ]
    )
    return pick


def build_place_subtree(
    context: OrchestratorContext, step: ModelStep
) -> py_trees.behaviour.Behaviour:
    """Build the place subtree for one recipe step.

    Sequence: compute preplace/place poses (FK from hardcoded joint
    configs) -> move to preplace -> move to place -> open
    gripper -> snap the brick into the stack (see
    :func:`build_snap_subtree`). If this is not the last brick,
    return to home so the next pick starts from the home viewpoint.
    """
    params = context.params
    label = f"place_brick{step.index}_{step.brick_name}"
    place = py_trees.composites.Sequence(name=label, memory=True)
    place.add_children(
        [
            ComputePlacePoses(context, step),
            MovePoseFromBlackboard(
                context,
                name=f"MoveToPreplace_brick{step.index}",
                pose_key="preplace_pose",
                velocity_scaling=params.fast_velocity_scaling,
                planner=PILZ_LIN,
                step=step,
                phase=Phase.PLACING,
                tolerance=PRECISE_TOLERANCE,
                description="moving to preplace",
            ),
            MovePoseFromBlackboard(
                context,
                name=f"MoveToPlace_brick{step.index}",
                pose_key="place_pose",
                velocity_scaling=params.slow_velocity_scaling,
                planner=PILZ_LIN,
                step=step,
                phase=Phase.PLACING,
                tolerance=PRECISE_TOLERANCE,
                description="moving to place",
            ),
            SetGripper(
                context,
                mode="open",
                position=0.0,
                brick_index=step.index,
                phase=Phase.PLACING,
            ),
        ]
    )
    return place


def build_snap_subtree(
    context: OrchestratorContext, step: ModelStep
) -> py_trees.behaviour.Behaviour:
    """Build the snapping subtree that snaps a just-placed brick into the stack.

    Everything from the pre-snap approach to the final retreat lives under
    this one ``Phase.SNAPPING`` subtree: compute the pre-snap pose -> move there
    (LIN) -> close the gripper to the snap-close position -> snap onto
    the place pose while closed -> reopen -> retreat to preplace (LIN) ->
    record the brick as placed in ``placement_record``.
    Requires ``preplace_pose``/``place_pose`` from :func:`build_place_subtree`.
    """
    params = context.params
    label = f"snap_brick{step.index}_{step.brick_name}"
    snap = py_trees.composites.Sequence(name=label, memory=True)
    snap.add_children(
        [
            ComputePreSnapPose(context, step),
            MovePoseFromBlackboard(
                context,
                name=f"MoveToPreSnap_brick{step.index}",
                pose_key="pre_snap_pose",
                velocity_scaling=params.slow_velocity_scaling,
                planner=PILZ_LIN,
                step=step,
                phase=Phase.SNAPPING,
                tolerance=PRECISE_TOLERANCE,
                description="moving to the pre-snap pose to start the snap (LIN)",
            ),
            SetGripper(
                context,
                mode="close",
                position=1.0,
                brick_index=step.index,
                phase=Phase.SNAPPING,
            ),
            MovePoseFromBlackboard(
                context,
                name=f"SnapBrickIntoStack_brick{step.index}",
                pose_key="snap_pose",
                velocity_scaling=params.slow_velocity_scaling,
                planner=PILZ_LIN,
                step=step,
                phase=Phase.SNAPPING,
                tolerance=PRECISE_TOLERANCE,
                description="snapping the brick onto the stack while the gripperis closed (LIN)",
            ),
            MovePoseFromBlackboard(
                context,
                name=f"RetreatToPreplace_brick{step.index}",
                pose_key="preplace_pose",
                velocity_scaling=params.fast_velocity_scaling,
                planner=PILZ_LIN,
                step=step,
                phase=Phase.SNAPPING,
                tolerance=PRECISE_TOLERANCE,
                description="retreating to preplace (LIN)",
            ),
            SetGripper(
                context,
                mode="open",
                position=0.0,
                brick_index=step.index,
                phase=Phase.SNAPPING,
            ),
            RecordPlacement(context, step),
        ]
    )
    return snap


def build_brick_subtree(
    context: OrchestratorContext, step: ModelStep
) -> py_trees.behaviour.Behaviour:
    """Build the full perceive-then-pick-then-place subtree for one recipe step.

    The work is guarded by :class:`BrickAlreadyPlaced`, so a resumed assembly
    skips bricks an earlier, partially successful goal already placed.
    """
    label = f"brick{step.index}_{step.brick_name}"
    work = py_trees.composites.Sequence(name=f"build_{label}", memory=True)
    work.add_children(
        [
            build_perceive_brick_pile_subtree(context, step),
            build_pick_subtree(context, step),
            build_place_subtree(context, step),
            build_snap_subtree(context, step),
            build_home_subtree(context, f"return_home_{label}"),
        ]
    )
    brick = py_trees.composites.Selector(name=label, memory=True)
    brick.add_children([BrickAlreadyPlaced(context, step), work])
    return brick


def build_tree(context: OrchestratorContext, recipe: ModelRecipe) -> py_trees.behaviour.Behaviour:
    """Build the full assembly tree for a resolved recipe.

    Sequence: resume or claim a place index -> compute+move home (FK) -> open
    gripper -> validate scene -> per-brick pick+place subtrees (in recipe
    order).
    """
    root = py_trees.composites.Sequence(name=f"assemble_{recipe.name}", memory=True)

    validate = py_trees.composites.Sequence(name="validate_scene", memory=True)
    validate.add_children(
        [
            WaitForArmSettled(context, Phase.VALIDATING_SCENE),
            py_trees.decorators.Retry(
                name="CaptureImage_retry",
                child=CaptureImage(context),
                num_failures=3,
            ),
            ValidateScene(context),
        ]
    )

    root.add_children(
        [
            SelectPlaceIndex(context),
            SetGripper(context, mode="open", position=0.0),
            build_home_subtree(context, "move_home_start"),
            validate,
        ]
    )
    for step in recipe.steps:
        root.add_child(build_brick_subtree(context, step))
    return root
