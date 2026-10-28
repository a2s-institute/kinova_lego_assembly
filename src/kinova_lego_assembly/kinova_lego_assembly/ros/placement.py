"""ROS placement providers implementing the core placement protocol.

The core layer defines :class:`PlacementProvider` (interface) and
:class:`PlacementRecord` (value type). The concrete provider here needs
rclpy (forward kinematics), so it lives in ``ros/``.

:class:`FkPlacementProvider` is a stand-in until stud-aligned placement
lands. It takes the configured joint configuration of the selected
holder-board place index (``params.place_joint_positions``), turns it
into a ``base_link`` end-effector pose via the ``/compute_fk`` service,
and hands the resulting poses back to the tree so the arm can be driven
through the standard pose-based move behaviours. It computes poses only;
it does not drive the arm or gripper (that stays in the behaviour tree),
mirroring how ``ros/grasp.py`` produces grasp poses.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from geometry_msgs.msg import PoseStamped

from kinova_lego_assembly.core.blueprint import ModelStep
from kinova_lego_assembly.core.placement import PlacementError


@dataclass(frozen=True)
class PlacementRecord:
    """Record of one placed brick, kept in ROS-native form on the blackboard.

    Unlike the ROS-independent ``core`` record, this stores the place pose as
    a :class:`geometry_msgs.msg.PoseStamped` so the frame id and the native
    (x, y, z, w) quaternion are preserved without conversion.

    Parameters
    ----------
    step_index : int
        Zero-based recipe step index that was placed.
    brick_id : int
        Catalog id of the placed brick.
    brick_name : str
        Synthesized ``"<type>_<colour>"`` name of the placed brick.
    place_pose : geometry_msgs.msg.PoseStamped
        End-effector place pose the brick was released at, in ``base_frame``.
    """

    step_index: int
    brick_id: int
    brick_name: str
    place_pose: PoseStamped


@dataclass(frozen=True)
class PlaceSlot:
    """State of one holder-board place index in the ``placement_record``.

    Parameters
    ----------
    model_name : str
        Name of the model being built at this place index.
    records : tuple of PlacementRecord, optional
        Bricks placed and snapped so far, in placement order.
    complete : bool, optional
        ``True`` once an assembly goal for this model succeeded here.
    """

    model_name: str
    records: tuple[PlacementRecord, ...] = ()
    complete: bool = False

    def placed_steps(self) -> set[int]:
        """Return the recipe step indices already placed at this slot."""
        return {record.step_index for record in self.records}


PREPLACE_OFFSET_Z = 0.15

# Duplo brick half-width (12 mm). Subsequent bricks stack on top of the
# previously placed brick by this z offset in the base frame.
BRICK_HALF_WIDTH_Z = 12e-3
PRIOR_POSE_OFFSET_Z = 8e-3


class FkPlacementProvider:
    """Compute preplace/place end-effector poses from hardcoded joint angles."""

    def __init__(self, context) -> None:
        """Bind the provider to the shared orchestrator context."""
        self._context = context

    def compute_poses(
        self,
        step: ModelStep,
        prior_place_pose: PoseStamped | None = None,
        place_index: int = 1,
    ) -> tuple[PoseStamped, PoseStamped]:
        """Return ``(preplace_pose, place_pose)`` in the base frame.

        The first brick's place pose comes from forward kinematics on the
        configured joint positions of the selected place index. Every subsequent
        brick stacks on top of the previously placed brick by
        :data:`BRICK_HALF_WIDTH_Z` in z. The prior pose is supplied by the
        caller (read from the placement records on the blackboard) so this
        method stays pure geometry with no hidden state.

        Parameters
        ----------
        step : ModelStep
            The recipe step being placed (used for error messages only).
        prior_place_pose : geometry_msgs.msg.PoseStamped or None, optional
            Place pose of the previously placed brick. ``None`` (the default)
            selects the FK path for the first brick; otherwise the new place
            pose is this pose offset by :data:`BRICK_HALF_WIDTH_Z` in z.
        place_index : int, optional
            One-based holder-board place index used for the first brick.
            Defaults to ``1``.

        Returns
        -------
        tuple of geometry_msgs.msg.PoseStamped
            The preplace and place end-effector poses, in ``base_frame``.

        Raises
        ------
        PlacementError
            If ``place_index`` is out of range or forward kinematics fails
            for the first brick.
        """
        params = self._context.params

        if prior_place_pose is None:
            joint_names = list(params.home_joint_names)
            place_configs = params.place_joint_positions
            if not 1 <= place_index <= len(place_configs):
                raise PlacementError(
                    f"place index {place_index} is not configured "
                    f"({len(place_configs)} place poses available)"
                )
            place_pose = self._context.fk_client.compute_pose_from_joints(
                joint_names=joint_names,
                joint_positions=list(place_configs[place_index - 1]),
                fk_link=params.pose_link,
            )
            if place_pose is None:
                raise PlacementError(f"FK failed for place pose of brick '{step.brick_name}'")
        else:
            place_pose = copy.deepcopy(prior_place_pose)
            place_pose.pose.position.z += BRICK_HALF_WIDTH_Z + PRIOR_POSE_OFFSET_Z

        preplace_pose = copy.deepcopy(place_pose)
        preplace_pose.pose.position.z += PREPLACE_OFFSET_Z

        return preplace_pose, place_pose
