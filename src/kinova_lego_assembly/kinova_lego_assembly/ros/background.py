"""A ROS node spun on its own background thread for blocking client calls.

The task orchestrator ticks a behaviour tree; behaviour ``update()`` calls
that need to talk to services/actions must not block the executor that is
spinning the tree. This wraps a dedicated node in its own single-threaded
executor thread so client futures complete in the background while a
behaviour busy-waits on them via :meth:`wait_for_future`.
"""

from __future__ import annotations

from threading import Thread
from time import monotonic, sleep

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter


class SpinningNode:
    """A dedicated node spun in a background daemon thread."""

    def __init__(self, name: str, use_sim_time: bool = False) -> None:
        """Create and start spinning a background node.

        Parameters
        ----------
        name : str
            Node name for the background node.
        use_sim_time : bool, optional
            Whether the background node should follow simulation time.
            Defaults to ``False``.
        """
        self._node = Node(
            name,
            parameter_overrides=[Parameter("use_sim_time", value=bool(use_sim_time))],
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._shutdown = False
        self._thread = Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    @property
    def node(self) -> Node:
        """Return the wrapped background node."""
        return self._node

    @property
    def active(self) -> bool:
        """Whether this spinner can still complete futures.

        Returns
        -------
        bool
            ``False`` once the spinner has been shut down or the ROS context
            has gone away, in which case a pending future can never complete
            and a failed :meth:`wait_for_future` is an abort, not a timeout.
        """
        return not self._shutdown and rclpy.ok()

    def wait_for_future(self, future, timeout_s: float) -> bool:
        """Busy-wait for a future to complete within a deadline.

        The future is completed by the background spin thread, not by this
        call. The wait aborts early if the background executor is shut down
        or the ROS context is no longer valid, so a torn-down spinner does
        not force callers to block for the full ``timeout_s`` on a future
        that can never complete.

        Parameters
        ----------
        future : rclpy.task.Future
            The future to wait on.
        timeout_s : float
            Maximum time to wait (a floor of 0.1 s is applied).

        Returns
        -------
        bool
            ``True`` if the future completed before the deadline.
        """
        deadline = monotonic() + max(0.1, float(timeout_s))
        while not future.done() and monotonic() < deadline:
            if self._shutdown or not rclpy.ok():
                break
            sleep(0.01)
        return future.done()

    def shutdown(self) -> None:
        """Stop the background executor and destroy the node."""
        self._shutdown = True
        self._executor.shutdown()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._node.destroy_node()
