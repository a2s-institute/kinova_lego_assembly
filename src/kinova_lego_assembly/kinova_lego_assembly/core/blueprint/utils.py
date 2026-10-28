"""In-memory representation of the LEGO blueprint catalog and models.

Pure business logic (no ROS). Loads ``assets/lego_blueprints.yaml``,
validates it against the schema in ``docs/adr/0009-blueprint-schema.md``
via :mod:`kinova_lego_assembly.core.blueprint.lint`, and exposes lookup
helpers the orchestrator uses at runtime.

Each catalog block gets a synthesized ``name`` of the form
``"<type>_<colour>"`` (e.g. ``"2x2_red"``, ``"4x2_blue"``) so detected
objects reported by perception can be matched by a single string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kinova_lego_assembly.core.blueprint.lint import LintError, lint_blueprints


class BlueprintError(ValueError):
    """Raised when a blueprint file is missing, unparsable, or invalid."""

    def __init__(self, message: str, errors: list[LintError] | None = None) -> None:
        """Store a summary message plus the underlying schema violations."""
        super().__init__(message)
        self.errors: list[LintError] = list(errors or [])


@dataclass(frozen=True)
class BrickSpec:
    """One catalog brick variant with its synthesized match name.

    Parameters
    ----------
    id : int
        Rule-book brick id (primary key).
    type : str
        Brick footprint, one of ``{"2x2", "4x2"}``.
    colour : str
        Brick colour, one of ``{"red", "green", "blue", "yellow"}``.
    description : str
        Backend-neutral natural-language label from the catalog.
    name : str
        Synthesized ``"<type>_<colour>"`` identifier used to match
        perception detections.
    """

    id: int
    type: str
    colour: str
    description: str
    name: str


@dataclass(frozen=True)
class ModelStep:
    """One ordered placement step in a model recipe.

    Parameters
    ----------
    index : int
        Zero-based position of this step in the bottom-up sequence.
    brick_id : int
        Catalog id of the brick placed at this step.
    brick_name : str
        Synthesized ``"<type>_<colour>"`` name of the brick.
    """

    index: int
    brick_id: int
    brick_name: str


@dataclass(frozen=True)
class ModelRecipe:
    """A resolved model with its ordered, brick-named build steps.

    Parameters
    ----------
    id : int
        Rule-book model id.
    name : str
        Snake_case model name.
    steps : list of ModelStep
        Ordered bottom-up placement steps.
    """

    id: int
    name: str
    steps: list[ModelStep] = field(default_factory=list)


class BlueprintUtils:
    """Validated, in-memory view of the blueprint catalog and models."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Build lookup tables from an already-validated blueprint document.

        Prefer :meth:`from_file`; this constructor assumes ``data`` has
        already passed :func:`lint_blueprints`.
        """
        self._blocks_by_id: dict[int, BrickSpec] = {}
        self._blocks_by_name: dict[str, BrickSpec] = {}
        for block in data["blocks"]:
            spec = BrickSpec(
                id=int(block["id"]),
                type=str(block["type"]),
                colour=str(block["colour"]),
                description=str(block["description"]),
                name=f"{block['type']}_{block['colour']}",
            )
            self._blocks_by_id[spec.id] = spec
            self._blocks_by_name[spec.name] = spec

        self._models_by_id: dict[int, ModelRecipe] = {}
        self._models_by_name: dict[str, ModelRecipe] = {}
        for model in data["models"]:
            steps = [
                ModelStep(
                    index=idx,
                    brick_id=int(step["id"]),
                    brick_name=self._blocks_by_id[int(step["id"])].name,
                )
                for idx, step in enumerate(model["steps"])
            ]
            recipe = ModelRecipe(id=int(model["id"]), name=str(model["name"]), steps=steps)
            self._models_by_id[recipe.id] = recipe
            self._models_by_name[recipe.name] = recipe

    @classmethod
    def from_file(cls, path: str | Path) -> BlueprintUtils:
        """Load, validate, and build a :class:`BlueprintUtils` from a YAML file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the ``lego_blueprints.yaml`` file.

        Returns
        -------
        BlueprintUtils
            The validated in-memory blueprint.

        Raises
        ------
        BlueprintError
            If the file is missing, cannot be parsed, or fails schema
            validation. ``BlueprintError.errors`` carries the individual
            :class:`LintError` violations when validation failed.
        """
        file_path = Path(path)
        try:
            raw = file_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise BlueprintError(f"blueprint file not found: {file_path}") from exc

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise BlueprintError(f"could not parse blueprint file {file_path}: {exc}") from exc

        errors = lint_blueprints(data)
        if errors:
            summary = "; ".join(str(err) for err in errors)
            raise BlueprintError(
                f"blueprint file {file_path} failed schema validation "
                f"({len(errors)} violation(s)): {summary}",
                errors=errors,
            )
        return cls(data)

    @property
    def blocks(self) -> list[BrickSpec]:
        """Return all catalog bricks ordered by id."""
        return [self._blocks_by_id[k] for k in sorted(self._blocks_by_id)]

    @property
    def models(self) -> list[ModelRecipe]:
        """Return all models ordered by id."""
        return [self._models_by_id[k] for k in sorted(self._models_by_id)]

    def resolve_model(self, target: str) -> ModelRecipe:
        """Resolve a target model given either its id (string) or name.

        Parameters
        ----------
        target : str
            The model rule-book id as a string (e.g. ``"81"``) or the
            snake_case model name (e.g. ``"e_stop"``).

        Returns
        -------
        ModelRecipe
            The matching recipe.

        Raises
        ------
        BlueprintError
            If no model matches ``target``.
        """
        key = target.strip()
        if not key:
            raise BlueprintError("empty target model identifier")

        if key.isdigit():
            recipe = self._models_by_id.get(int(key))
            if recipe is not None:
                return recipe

        recipe = self._models_by_name.get(key)
        if recipe is not None:
            return recipe

        known_ids = ", ".join(str(m.id) for m in self.models)
        known_names = ", ".join(m.name for m in self.models)
        raise BlueprintError(
            f"unknown target model {target!r}; known ids: [{known_ids}], "
            f"known names: [{known_names}]"
        )

    @staticmethod
    def demand_counts(recipe: ModelRecipe) -> dict[str, int]:
        """Count how many bricks of each ``"<type>_<colour>"`` a recipe needs.

        Parameters
        ----------
        recipe : ModelRecipe
            The model whose brick demand is being tallied.

        Returns
        -------
        dict
            Mapping of brick name to the number required by the recipe.
        """
        counts: dict[str, int] = {}
        for step in recipe.steps:
            counts[step.brick_name] = counts.get(step.brick_name, 0) + 1
        return counts
