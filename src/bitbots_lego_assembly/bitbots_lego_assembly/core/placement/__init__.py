"""Placement provider interface (ROS-independent)."""

from bitbots_lego_assembly.core.placement.provider import (
    PlacementError,
    PlacementProvider,
    PlacementRecord,
)

__all__ = [
    "PlacementError",
    "PlacementProvider",
    "PlacementRecord",
]
