"""Gripper control adapter over the ``control_msgs/GripperCommand`` action.

The ros2_control stack exposes ``robotiq_gripper_controller`` (a
``position_controllers/GripperActionController``) as a
``control_msgs/action/GripperCommand`` action. Position is the
``robotiq_85_left_knuckle_joint`` angle in radians: ~0 open, ~0.8 closed.
``max_effort`` sets grip force.

Known limitation (by design, see the orchestrator ADR/plan): grasp
success is inferred *only* from the gripper reaching its expected closed
position band. We do NOT verify that a brick is physically held (no force
or vision confirmation), so a close onto empty air still reports the
gripper "closed" and would be treated as a completed close.
"""

from __future__ import annotations

from control_msgs.action import GripperCommand
from rclpy.action import ActionClient

from bitbots_lego_assembly.ros.background import SpinningNode


class GripperController:
    """Open/close the gripper via a GripperCommand action client."""

    def __init__(
        self,
        spinner: SpinningNode,
        action_name: str = "/robotiq_gripper_controller/gripper_cmd",
        open_position: float = 0.0,
        closed_position: float = 1.0,
        max_effort: float = 65.0,
        closed_tolerance_rad: float = 0.05,
    ) -> None:
        """Configure the gripper adapter.

        Parameters
        ----------
        spinner : SpinningNode
            Background node hosting the action client.
        action_name : str, optional
            GripperCommand action name.
        open_position : float, optional
            Joint position (rad) treated as fully open. Defaults to ``0.0``.
        closed_position : float, optional
            Joint position (rad) commanded for a closed grip. Defaults to
            ``1.0``.
        max_effort : float, optional
            Grip force for close commands. Defaults to ``65.0``.
        closed_tolerance_rad : float, optional
            Band around ``closed_position`` within which the gripper is
            considered "closed" for grasp-success inference. Defaults to
            ``0.05``.
        """
        self._spinner = spinner
        self._node = spinner.node
        self._open_position = float(open_position)
        self._closed_position = float(closed_position)
        self._max_effort = float(max_effort)
        self._closed_tolerance_rad = float(closed_tolerance_rad)
        self._client = ActionClient(self._node, GripperCommand, action_name)
        self._action_name = action_name

    def wait_for_server(self, timeout_s: float = 5.0) -> bool:
        """Wait for the gripper action server to be available."""
        return self._client.wait_for_server(timeout_sec=float(timeout_s))

    def open(self, timeout_s: float = 10.0) -> bool:
        """Command the gripper fully open and wait for completion."""
        result = self._command(self._open_position, self._max_effort, timeout_s)
        return result is not None

    def close(self, timeout_s: float = 10.0) -> bool:
        """Command a closed grip and report whether the closed band was reached.

        Returns
        -------
        bool
            ``True`` if the command completed and the final position is
            within the configured closed band. Note this is a position
            check only, not an object-presence check.
        """
        result = self._command(self._closed_position, self._max_effort, timeout_s)
        if result is None:
            return False
        reached = self.is_at_closed_band(float(result.position))
        self._node.get_logger().info(
            f"Gripper close finished: position={float(result.position):.4f} "
            f"stalled={bool(result.stalled)} reached_goal={bool(result.reached_goal)} "
            f"within_closed_band={reached}"
        )
        return reached

    def move_to(self, position: float, timeout_s: float = 10.0) -> bool:
        """Command an explicit joint position and report completion.

        Unlike :meth:`open`/:meth:`close`, this drives the gripper to an
        arbitrary ``position`` (e.g. a full clamp to snap a stack together)
        and does not check the configured closed band. Success reflects only
        that the action completed, not object presence.

        Parameters
        ----------
        position : float
            Target joint position (rad).
        timeout_s : float, optional
            Action timeout. Defaults to ``10.0``.

        Returns
        -------
        bool
            ``True`` if the command completed, ``False`` otherwise.
        """
        result = self._command(float(position), self._max_effort, timeout_s)
        if result is None:
            return False
        self._node.get_logger().info(
            f"Gripper move_to finished: target={float(position):.4f} "
            f"position={float(result.position):.4f} stalled={bool(result.stalled)} "
            f"reached_goal={bool(result.reached_goal)}"
        )
        return True

    def is_at_closed_band(self, position: float) -> bool:
        """Return whether ``position`` is within the closed-position band."""
        return abs(float(position) - self._closed_position) <= self._closed_tolerance_rad

    def _command(
        self,
        position: float,
        max_effort: float,
        timeout_s: float,
    ) -> GripperCommand.Result | None:
        """Send one GripperCommand goal and block until it resolves.

        Returns
        -------
        control_msgs.action.GripperCommand.Result or None
            The action result, or ``None`` on server unavailability,
            rejection, or timeout.
        """
        if not self._client.wait_for_server(timeout_sec=3.0):
            self._node.get_logger().error(
                f"Gripper action server '{self._action_name}' is not available"
            )
            return None

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)

        send_future = self._client.send_goal_async(goal)
        if not self._spinner.wait_for_future(send_future, 5.0):
            self._node.get_logger().error("Timed out sending gripper goal")
            return None

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._node.get_logger().error("Gripper goal was rejected")
            return None

        result_future = goal_handle.get_result_async()
        if not self._spinner.wait_for_future(result_future, timeout_s):
            self._node.get_logger().error(f"Gripper action timed out after {timeout_s:.1f}s")
            return None

        wrapped = result_future.result()
        if wrapped is None:
            return None
        return wrapped.result
