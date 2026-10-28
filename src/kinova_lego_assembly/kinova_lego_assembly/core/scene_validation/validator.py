"""Scene sufficiency check for LEGO assembly.

Pure business logic (no ROS): given the bricks a segmentation backend
detected in the workspace and the recipe for the target model, decide
whether enough bricks of each ``"<type>_<colour>"`` variant are present
to build the model. The orchestrator uses this to fail fast before it
starts picking, honouring the "don't waste the user's time" principle.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from kinova_lego_assembly.core.blueprint.utils import BlueprintUtils, ModelRecipe


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of comparing detected bricks against a recipe's demand.

    Parameters
    ----------
    enough : bool
        ``True`` when every required brick variant is present in at
        least the required quantity.
    demand : dict
        Required count per ``"<type>_<colour>"`` brick name.
    available : dict
        Detected count per brick name (restricted to names the recipe
        needs; unrelated detections are ignored).
    shortfall : dict
        For each brick name that is under-supplied, how many more are
        needed (``demand - available``, only positive entries). Empty
        when ``enough`` is ``True``.
    """

    enough: bool
    demand: dict[str, int] = field(default_factory=dict)
    available: dict[str, int] = field(default_factory=dict)
    shortfall: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a one-line human-readable summary for logs and messages."""
        if self.enough:
            return f"scene sufficient: demand={self.demand} available={self.available}"
        return (
            f"scene insufficient: shortfall={self.shortfall} "
            f"(demand={self.demand}, available={self.available})"
        )


def count_detected(names: Iterable[str]) -> dict[str, int]:
    """Tally detected brick names into a count map.

    Parameters
    ----------
    names : iterable of str
        The ``"<type>_<colour>"`` names of detected bricks (one entry per
        detected instance).

    Returns
    -------
    dict
        Mapping of brick name to the number of detections.
    """
    return dict(Counter(name for name in names))


def verify(detected_counts: dict[str, int], recipe: ModelRecipe) -> ValidationReport:
    """Check whether detected bricks satisfy a recipe's brick demand.

    Parameters
    ----------
    detected_counts : dict
        Count of detected bricks per ``"<type>_<colour>"`` name.
    recipe : ModelRecipe
        The target model recipe whose demand must be met.

    Returns
    -------
    ValidationReport
        Whether the scene is sufficient plus the per-variant breakdown.
    """
    demand = BlueprintUtils.demand_counts(recipe)
    available = {name: int(detected_counts.get(name, 0)) for name in demand}
    shortfall = {
        name: required - available[name]
        for name, required in demand.items()
        if available[name] < required
    }
    return ValidationReport(
        enough=not shortfall,
        demand=demand,
        available=available,
        shortfall=shortfall,
    )
