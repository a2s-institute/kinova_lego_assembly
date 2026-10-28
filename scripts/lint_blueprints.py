#!/usr/bin/env python3
"""Command-line schema validator for ``src/kinova_lego_assembly/assets/lego_blueprints.yaml``.

The validation logic lives in
``kinova_lego_assembly.core.blueprint.lint`` (pure business logic, no
ROS). This script is a thin CLI wrapper so the same rules run in CI and
inside the task orchestrator's boot health check.

Exits 0 if every invariant holds, 1 otherwise. All violations are printed
in a single pass so authors get the full picture from one run.

Usage examples
--------------

Validate the canonical file from the repo root::

    python3 scripts/lint_blueprints.py

Validate an alternative file (e.g. when authoring a new schema variant)::

    python3 scripts/lint_blueprints.py path/to/other_blueprints.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make the in-tree Python package importable when running from a source
# checkout (before the package is installed via colcon).
_PACKAGE_SRC = Path(__file__).resolve().parent.parent / "src" / "kinova_lego_assembly"
if str(_PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_SRC))

from kinova_lego_assembly.core.blueprint.lint import lint_blueprints  # noqa: E402

DEFAULT_BLUEPRINTS_PATH: Path = _PACKAGE_SRC / "assets" / "lego_blueprints.yaml"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (excluding the program name). Defaults to
        ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code. ``0`` on success, ``1`` if any schema
        violation was found or the file could not be loaded.
    """
    parser = argparse.ArgumentParser(
        description="Validate the structure of a lego_blueprints.yaml file."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_BLUEPRINTS_PATH,
        help=("Path to the blueprints YAML file (default: %(default)s)."),
    )
    args = parser.parse_args(argv)

    try:
        with args.path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError:
        print(f"FAIL: file not found: {args.path}", file=sys.stderr)
        return 1
    except yaml.YAMLError as exc:
        print(f"FAIL: could not parse {args.path}: {exc}", file=sys.stderr)
        return 1

    errors = lint_blueprints(data)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(
            f"FAIL: {len(errors)} schema violation(s) in {args.path}",
            file=sys.stderr,
        )
        return 1

    n_blocks = len(data["blocks"])
    n_models = len(data["models"])
    print(f"OK: {n_blocks} block(s), {n_models} model(s) validated in {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
