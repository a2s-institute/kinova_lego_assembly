"""Manage a target managed (lifecycle) node for the assembly stack.

This mirrors the non-interactive part of the ``lifecycle_controller`` node
(its ``execute`` service calls) without the keyboard-driven loop, which
cannot work inside a launch file. On startup it drives the target node
configure -> activate; it then stays alive so that on launch shutdown
(Ctrl-C / SIGINT) it drives the target deactivate -> cleanup, letting the
managed node release its resources (e.g. GPU / model weights) cleanly
instead of being hard-killed mid-activation.

Every transition is requested via the target's ``change_state`` service
synchronously, so each one completes (e.g. MMOR's ``on_configure`` loading
YOLO weights) before the next is requested.

Parameters
----------
lc_name : str
    Base name of the managed node (services resolve to ``<lc_name>/change_state``
    and ``<lc_name>/get_state``).
service_timeout_s : float
    How long to wait for services to appear and for each bringup transition
    to complete.
teardown_timeout_s : float
    Shorter per-transition timeout used during shutdown so teardown does not
    stall the whole launch when the target is already gone.
"""

from __future__ import annotations

import sys

import rclpy
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState, GetState
from rclpy.node import Node


class LifecycleManager(Node):
    """Bring a managed node up on start and tear it down on shutdown."""

    def __init__(self) -> None:
        super().__init__("lifecycle_manager")
        self.declare_parameter("lc_name", "")
        self.declare_parameter("service_timeout_s", 30.0)
        self.declare_parameter("teardown_timeout_s", 5.0)

        self._lc_name = self.get_parameter("lc_name").value
        self._timeout_s = float(self.get_parameter("service_timeout_s").value)
        self._teardown_timeout_s = float(self.get_parameter("teardown_timeout_s").value)

        self.bringup_ok = False
        self._active = False

        if not self._lc_name:
            self.get_logger().error(
                "Parameter 'lc_name' is empty; nothing to manage "
                "(e.g. --ros-args -p lc_name:=mmor)."
            )
            return

        self._change_client = self.create_client(ChangeState, f"{self._lc_name}/change_state")
        self._get_client = self.create_client(GetState, f"{self._lc_name}/get_state")

        self.bringup_ok = self._bringup()

    def _bringup(self) -> bool:
        """Run configure then activate, returning True only if both succeed."""
        transitions = (
            ("configure", Transition.TRANSITION_CONFIGURE),
            ("activate", Transition.TRANSITION_ACTIVATE),
        )
        for label, transition_id in transitions:
            self.get_logger().info(f"Requesting '{label}' on '{self._lc_name}'...")
            if not self._change_state(transition_id, self._timeout_s):
                self.get_logger().error(
                    f"Transition '{label}' on '{self._lc_name}' failed; aborting."
                )
                return False
            self._log_state()
        self._active = True
        self.get_logger().info(f"'{self._lc_name}' is active.")
        return True

    def teardown(self) -> None:
        """Drive the target deactivate -> cleanup; best-effort on shutdown."""
        if not self._active:
            return
        self._active = False
        transitions = (
            ("deactivate", Transition.TRANSITION_DEACTIVATE),
            ("cleanup", Transition.TRANSITION_CLEANUP),
        )
        for label, transition_id in transitions:
            self.get_logger().info(f"Requesting '{label}' on '{self._lc_name}'...")
            if not self._change_state(transition_id, self._teardown_timeout_s):
                self.get_logger().warning(
                    f"Transition '{label}' on '{self._lc_name}' did not complete; "
                    "the node may already be shutting down."
                )
                return
        self.get_logger().info(f"'{self._lc_name}' cleaned up.")

    def _change_state(self, transition_id: int, timeout_s: float) -> bool:
        """Call change_state synchronously; return the transition success flag."""
        if not self._change_client.wait_for_service(timeout_sec=timeout_s):
            self.get_logger().error(
                f"Service '{self._change_client.srv_name}' unavailable after {timeout_s:.1f}s."
            )
            return False
        request = ChangeState.Request()
        request.transition.id = transition_id
        future = self._change_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            self.get_logger().error("Timed out waiting for change_state response.")
            return False
        return bool(future.result().success)

    def _log_state(self) -> None:
        """Best-effort query and log of the target's current lifecycle state."""
        if not self._get_client.wait_for_service(timeout_sec=self._timeout_s):
            return
        future = self._get_client.call_async(GetState.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._timeout_s)
        if future.done() and future.result() is not None:
            self.get_logger().info(
                f"'{self._lc_name}' state: {future.result().current_state.label}"
            )


def main(args: list[str] | None = None) -> None:
    """Bring the target up, spin until shutdown, then tear it down."""
    rclpy.init(args=args)
    node = LifecycleManager()
    if not node.bringup_ok:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(1)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.teardown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
