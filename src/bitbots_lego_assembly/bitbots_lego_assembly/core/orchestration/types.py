"""Shared orchestration value types (ROS-independent).

These describe the phases the assembly passes through. The behaviour tree
(ROS side) maps its ticks onto these phases so action feedback, the
``/task_orchestrator/assembly_status`` topic, and logs all speak the same vocabulary.
"""

from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    """High-level phases reported as assembly progresses.

    Inherits from ``str`` so the value serializes directly into the
    ``phase`` string field of the action feedback and status message.
    """

    IDLE = "idle"
    RESOLVING_MODEL = "resolving_model"
    MOVING_HOME = "moving_home"
    VALIDATING_SCENE = "validating_scene"
    PICKING = "picking"
    PLACING = "placing"
    SNAPPING = "snapping"
    COMPLETE = "complete"
    FAILED = "failed"
