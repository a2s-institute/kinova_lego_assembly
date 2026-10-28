"""Shared pytest fixtures for both ``test/core`` and ``test/ros``.

Test assets live under ``test/assets`` so that pure-Python and ROS test
suites can pull from the same source. The module also installs a
``pytest_configure`` hook that warns on drift between
``config/params.yml`` and each backend's ``DEFAULTS`` (see ADR-0014).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

ASSETS_DIR = (Path(__file__).resolve().parent / "assets").resolve()
PARAMS_YML = (Path(__file__).resolve().parents[1] / "config" / "params.yml").resolve()


@pytest.fixture(scope="session")
def assets_dir() -> Path:
    """Absolute path to the shared test asset directory."""
    return ASSETS_DIR


@pytest.fixture(scope="session")
def sample_image_path(assets_dir: Path) -> Path:
    """Path to a small RGB image with at least one LEGO brick visible."""
    return assets_dir / "sample_lego_kinova_rgb.png"


def pytest_configure(config: pytest.Config) -> None:
    """Warn on drift between ``config/params.yml`` and backend ``DEFAULTS``.

    The ROS overlay (``config/params.yml``) and the pure-Python defaults
    (``<method>Segmentor.DEFAULTS``) are two separate sources of truth
    by design, see ADR-0014. This hook compares them at test-collection
    time and emits a ``UserWarning`` for every missing key, extra key,
    or value mismatch so silent drift is visible the moment a developer
    runs the suite. It never fails the run; the warning shows up in
    pytest's terminal summary.
    """
    del config

    try:
        import yaml
        from bitbots_lego_assembly.core.segmentation import factory
    except ImportError:
        return

    if not PARAMS_YML.is_file():
        return

    try:
        raw = yaml.safe_load(PARAMS_YML.read_text()) or {}
    except yaml.YAMLError as exc:
        warnings.warn(f"params.yml drift-check skipped: {exc}", stacklevel=2)
        return

    ros_params: dict = {}
    for value in raw.values():
        if isinstance(value, dict) and "ros__parameters" in value:
            ros_params = value["ros__parameters"] or {}
            break
    seg_params: dict = ros_params.get("segmentation") or {}

    for method in factory.supported_methods():
        defaults = factory.get_spec(method).knobs
        yml_block = seg_params.get(method) or {}
        for key, default_value in defaults.items():
            if key not in yml_block:
                warnings.warn(
                    f"params.yml is missing 'segmentation.{method}.{key}'; ROS overlay "
                    f"falls back to DEFAULTS={default_value!r} (see ADR-0014).",
                    stacklevel=2,
                )
            elif yml_block[key] != default_value:
                warnings.warn(
                    f"params.yml 'segmentation.{method}.{key}={yml_block[key]!r}' differs "
                    f"from DEFAULTS={default_value!r} (see ADR-0014).",
                    stacklevel=2,
                )
        for key in yml_block:
            if key not in defaults:
                warnings.warn(
                    f"params.yml has 'segmentation.{method}.{key}={yml_block[key]!r}' "
                    f"that is not in DEFAULTS (see ADR-0014).",
                    stacklevel=2,
                )
