"""On-disk model registry resolving ``method`` + ``version`` to a directory.

Layout convention::

    trained_models/<task>/<method>/v<N>/model.onnx
    trained_models/<task>/<method>/v<N>/model.yaml   # optional

Versions are picked by the highest integer suffix in ``v<N>`` when no
explicit version is requested. The root can be overridden through the
``BITBOTS_LEGO_MODELS_ROOT`` environment variable (handy for tests).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_VERSION_RE = re.compile(r"^v(\d+)$")
_DEFAULT_TASK = "instance_segmentation"


def models_root() -> Path:
    """Return the directory containing all trained model versions.

    The lookup order is:

    1. ``$BITBOTS_LEGO_MODELS_ROOT`` if set (ROS launch and tests use this).
    2. The first ancestor of this file whose name is
       ``bitbots_lego_assembly`` and which contains a ``trained_models``
       directory (matches the source-tree layout in dev installs).

    Returns
    -------
    pathlib.Path
        The resolved root, or a non-existent placeholder when no candidate
        is found (callers handle missing directories explicitly).
    """
    env = os.environ.get("BITBOTS_LEGO_MODELS_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / "trained_models"
        if candidate.is_dir():
            return candidate.resolve()

    return (Path(__file__).resolve().parents[2] / "trained_models").resolve()


def _normalize_version(version: str | int) -> str:
    """Coerce ``"3"``, ``3`` or ``"v3"`` to ``"v3"``."""
    token = str(version).strip().lower()
    if token.startswith("v"):
        token = token[1:]
    if not token.isdigit():
        raise ValueError(f"Invalid version token: {version!r}")
    return f"v{int(token)}"


def list_versions(method: str = "yolo", task: str = _DEFAULT_TASK) -> list[str]:
    """List all available ``v<N>`` directories for ``method``/``task``.

    Returns
    -------
    list of str
        Version names sorted ascending by numeric suffix.
    """
    base = models_root() / task / method
    if not base.is_dir():
        return []
    versions = [p.name for p in base.iterdir() if p.is_dir() and _VERSION_RE.match(p.name)]
    return sorted(versions, key=lambda v: int(_VERSION_RE.match(v).group(1)))


def resolve(
    method: str = "yolo",
    version: str | int | None = None,
    task: str = _DEFAULT_TASK,
) -> Path:
    """Resolve a model version directory.

    Parameters
    ----------
    method : str
        Backend name (e.g. ``"yolo"``).
    version : str or int, optional
        Either ``"v3"``, ``"3"`` or ``3``. When ``None`` the highest
        available ``v<N>`` is chosen.
    task : str
        Top-level task folder.

    Returns
    -------
    pathlib.Path
        The resolved ``v<N>`` directory.

    Raises
    ------
    FileNotFoundError
        If the method directory does not exist, no versions are available,
        or the requested version is missing.
    """
    base = models_root() / task / method
    if not base.is_dir():
        raise FileNotFoundError(f"Model method directory not found: {base}")

    if version is None:
        versions = list_versions(method, task)
        if not versions:
            raise FileNotFoundError(f"No model versions found under {base}")
        chosen = versions[-1]
    else:
        chosen = _normalize_version(version)

    target = base / chosen
    if not target.is_dir():
        raise FileNotFoundError(f"Model version not found: {target}")
    return target
