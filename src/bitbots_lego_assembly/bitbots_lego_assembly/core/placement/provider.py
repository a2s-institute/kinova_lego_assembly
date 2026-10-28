"""Placement provider interface.

Only the :class:`PlacementProvider` protocol, the :class:`PlacementRecord`
value type, and :class:`PlacementError` live here. The concrete provider
that computes poses and commands the arm and gripper is a ROS adapter
under ``ros/`` because it needs rclpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from geometry_msgs.msg import PoseStamped

from bitbots_lego_assembly.core.blueprint.utils import ModelStep


class PlacementError(RuntimeError):
    """Raised by a :class:`PlacementProvider` when a placement fails."""


@dataclass(frozen=True)
class PlacementRecord:
    """Record of one (real or dummy) placement, kept for progress tracking.

    Parameters
    ----------
    step_index : int
        Zero-based recipe step index that was placed.
    brick_id : int
        Catalog id of the placed brick.
    brick_name : str
        Synthesized ``"<type>_<colour>"`` name of the placed brick.
    place_pose : PoseStamped
        Pose the brick was (nominally) released at, in ``base_link``.
        For the dummy provider this is a fixed offset from the retreat
        pose, not a stud-aligned target.
    """

    step_index: int
    brick_id: int
    brick_name: str
    place_pose: PoseStamped


@runtime_checkable
class PlacementProvider(Protocol):
    """Contract for placing one picked brick and reporting the result.

    Implementations may command the arm and gripper (ROS adapters) or be
    pure test doubles. They must return a :class:`PlacementRecord` on
    success and raise on failure.
    """

    def place(self, step: ModelStep, retreat_pose: PoseStamped) -> PlacementRecord:
        """Place the brick for ``step`` and return a record of where it went.

        Parameters
        ----------
        step : ModelStep
            The recipe step being placed.
        retreat_pose : PoseStamped
            The end-effector retreat pose (in ``base_link``) reached at
            the end of the pick, used as the geometric anchor for the
            placement move.

        Returns
        -------
        PlacementRecord
            Where the brick was placed.

        Raises
        ------
        PlacementError
            If the placement could not be completed.
        """
        ...
