#!/usr/bin/env python3
"""Align STL meshes so the bounding box is centred and the long edge is on +X.

For every STL file matched by the supplied glob pattern(s) the script will:

1. Load the mesh with PyMeshLab.
2. Compute the axis-aligned bounding box (AABB) of the input.
3. Apply at most a 90 degree yaw around the world Z-axis: if the AABB extent
   along Y is greater than along X, rotate by 90 degrees about Z so that the
   longer horizontal edge ends up on X. The mesh's "up" (Z) direction is
   preserved exactly. The object is never tilted around X or Y.
4. Translate so the resulting AABB is centred on the origin.
5. Bake the transform into the geometry and write the result back to disk.

Examples
--------
Align every brick under ``assets/`` and write the results next to the inputs
with an ``_aligned.stl`` suffix::

    python3 align_stl_axes.py "assets/**/*.stl"

Overwrite the originals in place::

    python3 align_stl_axes.py --in-place "assets/bricks/*.stl"

Write aligned copies to a separate folder::

    python3 align_stl_axes.py "assets/*.stl" --output-dir out/
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

try:
    import pymeshlab as pml
except ImportError as exc:  # pragma: no cover - explicit hint for users
    sys.stderr.write(
        "pymeshlab is required. Install it into your venv with "
        "`uv pip install pymeshlab` (or `pip install pymeshlab`).\n"
    )
    raise SystemExit(1) from exc


_LOG = logging.getLogger("align_stl_axes")


def _expand_patterns(patterns: Sequence[str]) -> list[Path]:
    """Expand shell-style glob patterns to a sorted, de-duplicated path list.

    Parameters
    ----------
    patterns : sequence of str
        Glob patterns (``**`` is supported via ``recursive=True``). Patterns
        without any wildcards are treated as literal paths.

    Returns
    -------
    list of pathlib.Path
        Existing ``.stl`` files matched by the patterns.
    """
    seen: set[Path] = set()
    matched: list[Path] = []
    for pattern in patterns:
        hits = glob.glob(pattern, recursive=True)
        if not hits and not any(ch in pattern for ch in "*?["):
            hits = [pattern]
        for hit in hits:
            path = Path(hit).resolve()
            if path.suffix.lower() != ".stl":
                _LOG.debug("Skipping non-STL match %s", path)
                continue
            if not path.is_file():
                _LOG.warning("Skipping non-file match %s", path)
                continue
            if path in seen:
                continue
            seen.add(path)
            matched.append(path)
    return sorted(matched)


def _yaw_rotation(vertices: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation that puts the longer horizontal AABB edge on X.

    The rotation is either the identity or a 90 degree turn around the world
    Z-axis. Z is therefore preserved exactly; the mesh is never tilted around
    X or Y. The decision is made on the AABB extents of the input mesh: if the
    Y extent is strictly greater than the X extent, a +90 degree yaw is
    applied (so old +Y maps to new +X), otherwise the identity is returned.

    Parameters
    ----------
    vertices : numpy.ndarray
        Array of shape ``(N, 3)`` holding the mesh vertex coordinates.

    Returns
    -------
    numpy.ndarray
        A ``(3, 3)`` rotation matrix ``R`` such that ``new = R @ old``.
    """
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    extent = maxs - mins
    if extent[1] > extent[0]:
        return np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    return np.eye(3, dtype=np.float64)


def _build_transform(vertices: np.ndarray) -> np.ndarray:
    """Compose the yaw rotation and AABB-centring translation into a 4x4.

    The yaw (see ``_yaw_rotation``) is applied first around the origin, then
    a translation is chosen so the AABB of the rotated mesh is centred at
    ``(0, 0, 0)``.

    Parameters
    ----------
    vertices : numpy.ndarray
        Array of shape ``(N, 3)`` with the original vertex coordinates.

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` homogeneous transform ready to be fed into PyMeshLab's
        ``set_matrix`` filter.
    """
    rotation = _yaw_rotation(vertices)
    rotated = vertices @ rotation.T
    aabb_center = 0.5 * (rotated.min(axis=0) + rotated.max(axis=0))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -aabb_center
    return transform


def _resolve_output_path(
    src: Path,
    *,
    in_place: bool,
    output_dir: Path | None,
    suffix: str,
) -> Path:
    """Compute where the aligned mesh should be written for ``src``.

    Parameters
    ----------
    src : pathlib.Path
        Original input STL path.
    in_place : bool
        If True, overwrite ``src``.
    output_dir : pathlib.Path or None
        If given, write the result into this directory (it is created on
        demand) and ``suffix`` is ignored.
    suffix : str
        Suffix appended before the ``.stl`` extension when writing next to the
        input.

    Returns
    -------
    pathlib.Path
        The destination path.
    """
    if in_place:
        return src
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / src.name
    return src.with_name(f"{src.stem}{suffix}.stl")


def align_mesh(src: Path, dst: Path) -> None:
    """Align a single STL file and write the result to ``dst``.

    Parameters
    ----------
    src : pathlib.Path
        Path to the input STL file.
    dst : pathlib.Path
        Path where the aligned STL should be written.
    """
    ms = pml.MeshSet()
    ms.load_new_mesh(str(src))
    mesh = ms.current_mesh()
    vertices = np.asarray(mesh.vertex_matrix(), dtype=np.float64)
    if vertices.shape[0] < 3:
        raise ValueError(f"{src} has fewer than 3 vertices; cannot align")

    transform = _build_transform(vertices)

    ms.apply_filter(
        "set_matrix",
        transformmatrix=transform.astype(np.float64),
        compose=False,
        freeze=True,
        alllayers=False,
    )
    ms.save_current_mesh(str(dst), binary=True)


def _process(
    paths: Iterable[Path],
    *,
    in_place: bool,
    output_dir: Path | None,
    suffix: str,
) -> int:
    """Iterate over input paths and align each one. Returns the failure count."""
    failures = 0
    for src in paths:
        dst = _resolve_output_path(src, in_place=in_place, output_dir=output_dir, suffix=suffix)
        try:
            align_mesh(src, dst)
        except Exception as exc:
            failures += 1
            _LOG.error("Failed to align %s: %s", src, exc)
            continue
        rel = dst if dst.is_absolute() else dst.resolve()
        _LOG.info("Aligned %s -> %s", src, rel)
    return failures


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Centre STL meshes on the origin and yaw them (at most 90 "
            "degrees around Z) so the longer horizontal AABB edge lies on X. "
            "Z (up) is preserved exactly; the mesh is never tilted around X "
            "or Y."
        )
    )
    parser.add_argument(
        "patterns",
        nargs="+",
        help=(
            "One or more STL paths or glob patterns (quote them so the shell "
            "does not pre-expand, e.g. 'assets/**/*.stl')."
        ),
    )
    out = parser.add_mutually_exclusive_group()
    out.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each input file with its aligned version.",
    )
    out.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write aligned meshes (keeping their original filenames) here.",
    )
    parser.add_argument(
        "--suffix",
        default="_aligned",
        help=(
            "Suffix appended before the .stl extension when writing next to "
            "the input (ignored with --in-place or --output-dir). Default: "
            "'_aligned'."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: align every STL matched by the input glob patterns."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    paths = _expand_patterns(args.patterns)
    if not paths:
        _LOG.error("No STL files matched: %s", ", ".join(args.patterns))
        return 2

    _LOG.info("Aligning %d STL file(s)", len(paths))
    failures = _process(
        paths,
        in_place=args.in_place,
        output_dir=args.output_dir,
        suffix=args.suffix,
    )
    if failures:
        _LOG.error("%d file(s) failed to align", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
