"""Schema validator for ``assets/lego_blueprints.yaml``.

The schema, field semantics, conventions, and the full list of invariants
this module enforces are documented in
``docs/adr/0009-blueprint-schema.md`` (see the "Invariants enforced by
scripts/lint_blueprints.py" subsection). This module is the executable
form of that spec; if the two disagree, the ADR is canonical and the
module is the bug.

This is pure business logic with no ROS dependency. The command-line
entry point lives in ``scripts/lint_blueprints.py``, which imports these
functions.
"""

from __future__ import annotations

from typing import Any, NamedTuple

VALID_TYPES: frozenset[str] = frozenset({"2x2", "4x2"})
VALID_COLOURS: frozenset[str] = frozenset({"red", "green", "blue", "yellow"})


class LintError(NamedTuple):
    """A single schema violation.

    Parameters
    ----------
    path : str
        JSON-path-like locator describing where the violation occurred,
        e.g. ``blocks[3].colour`` or ``models[1].steps[2].id``.
    message : str
        Human-readable description of the violation.
    """

    path: str
    message: str

    def __str__(self) -> str:
        """Render the error as ``<path>: <message>`` for human-friendly output."""
        return f"{self.path}: {self.message}"


def _require_keys(obj: Any, required: tuple[str, ...], path: str) -> list[LintError]:
    """Check that ``obj`` is a dict and contains every key in ``required``.

    Parameters
    ----------
    obj : Any
        Candidate dict.
    required : tuple of str
        Keys that must be present.
    path : str
        Locator used to prefix any returned errors.

    Returns
    -------
    list of LintError
        Empty if ``obj`` is a dict with all required keys, otherwise one
        error per missing key (or one error if ``obj`` is not a dict).
    """
    if not isinstance(obj, dict):
        return [LintError(path, f"expected a mapping, got {type(obj).__name__}")]
    return [
        LintError(path, f"missing required field {key!r}") for key in required if key not in obj
    ]


def lint_block(block: Any, path: str) -> list[LintError]:
    """Validate one entry in ``blocks:``.

    Parameters
    ----------
    block : Any
        The candidate block entry.
    path : str
        Locator used to prefix returned errors (e.g. ``blocks[2]``).

    Returns
    -------
    list of LintError
        All schema violations found in this block.
    """
    errors = _require_keys(block, ("id", "type", "colour", "description"), path)
    if errors and not isinstance(block, dict):
        return errors

    block_id = block.get("id")
    if not isinstance(block_id, int) or isinstance(block_id, bool) or block_id <= 0:
        errors.append(LintError(f"{path}.id", f"expected a positive int, got {block_id!r}"))

    block_type = block.get("type")
    if block_type not in VALID_TYPES:
        errors.append(
            LintError(
                f"{path}.type",
                f"{block_type!r} is not in {sorted(VALID_TYPES)}",
            )
        )

    colour = block.get("colour")
    if colour not in VALID_COLOURS:
        errors.append(
            LintError(
                f"{path}.colour",
                f"{colour!r} is not in {sorted(VALID_COLOURS)}",
            )
        )

    description = block.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append(
            LintError(
                f"{path}.description",
                f"expected a non-empty string, got {description!r}",
            )
        )

    return errors


def lint_step(step: Any, path: str, valid_ids: frozenset[int]) -> list[LintError]:
    """Validate one entry in a model's ``steps:`` list.

    Parameters
    ----------
    step : Any
        The candidate step entry.
    path : str
        Locator used to prefix returned errors
        (e.g. ``models[1].steps[2]``).
    valid_ids : frozenset of int
        IDs that resolve against the catalog.

    Returns
    -------
    list of LintError
        All schema violations found in this step.
    """
    errors = _require_keys(step, ("id",), path)
    if errors and not isinstance(step, dict):
        return errors

    step_id = step.get("id")
    if not isinstance(step_id, int) or isinstance(step_id, bool):
        errors.append(LintError(f"{path}.id", f"expected an int, got {step_id!r}"))
    elif step_id not in valid_ids:
        errors.append(
            LintError(
                f"{path}.id",
                f"{step_id} is not a known block id (catalog has {sorted(valid_ids)})",
            )
        )

    return errors


def lint_model(model: Any, path: str, valid_ids: frozenset[int]) -> list[LintError]:
    """Validate one entry in ``models:``.

    Parameters
    ----------
    model : Any
        The candidate model entry.
    path : str
        Locator used to prefix returned errors (e.g. ``models[1]``).
    valid_ids : frozenset of int
        IDs that resolve against the catalog.

    Returns
    -------
    list of LintError
        All schema violations found in this model.
    """
    errors = _require_keys(model, ("id", "name", "steps"), path)
    if errors and not isinstance(model, dict):
        return errors

    model_id = model.get("id")
    if not isinstance(model_id, int) or isinstance(model_id, bool) or model_id <= 0:
        errors.append(LintError(f"{path}.id", f"expected a positive int, got {model_id!r}"))

    name = model.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append(
            LintError(
                f"{path}.name",
                f"expected a non-empty string, got {name!r}",
            )
        )

    steps = model.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append(
            LintError(
                f"{path}.steps",
                f"expected a non-empty list, got {steps!r}",
            )
        )
        return errors

    for idx, step in enumerate(steps):
        errors.extend(lint_step(step, f"{path}.steps[{idx}]", valid_ids))

    return errors


def lint_blueprints(data: Any) -> list[LintError]:
    """Validate the full top-level blueprints document.

    Parameters
    ----------
    data : Any
        The parsed YAML document.

    Returns
    -------
    list of LintError
        All schema violations found across catalog and models. Empty on
        success.
    """
    errors: list[LintError] = []

    if not isinstance(data, dict):
        return [LintError("<root>", f"expected a mapping, got {type(data).__name__}")]

    blocks = data.get("blocks")
    models = data.get("models")

    if blocks is None:
        errors.append(LintError("<root>", "missing required field 'blocks'"))
    if models is None:
        errors.append(LintError("<root>", "missing required field 'models'"))

    if not isinstance(blocks, list) or not blocks:
        errors.append(
            LintError("blocks", f"expected a non-empty list, got {type(blocks).__name__}")
        )
        return errors

    seen_ids: set[int] = set()
    for idx, block in enumerate(blocks):
        block_errors = lint_block(block, f"blocks[{idx}]")
        errors.extend(block_errors)
        if isinstance(block, dict):
            block_id = block.get("id")
            if isinstance(block_id, int) and not isinstance(block_id, bool):
                if block_id in seen_ids:
                    errors.append(
                        LintError(f"blocks[{idx}].id", f"duplicate id {block_id} (must be unique)")
                    )
                seen_ids.add(block_id)

    if not isinstance(models, list):
        errors.append(LintError("models", f"expected a list, got {type(models).__name__}"))
        return errors

    seen_names: set[str] = set()
    seen_model_ids: set[int] = set()
    valid_ids = frozenset(seen_ids)
    for idx, model in enumerate(models):
        model_errors = lint_model(model, f"models[{idx}]", valid_ids)
        errors.extend(model_errors)
        if isinstance(model, dict):
            name = model.get("name")
            if isinstance(name, str):
                if name in seen_names:
                    errors.append(
                        LintError(
                            f"models[{idx}].name",
                            f"duplicate model name {name!r} (must be unique)",
                        )
                    )
                seen_names.add(name)

            model_id = model.get("id")
            if isinstance(model_id, int) and not isinstance(model_id, bool):
                if model_id in seen_model_ids:
                    errors.append(
                        LintError(
                            f"models[{idx}].id",
                            f"duplicate model id {model_id} (must be unique)",
                        )
                    )
                seen_model_ids.add(model_id)

    return errors
