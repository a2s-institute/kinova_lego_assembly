"""Tests for the model directory registry."""

from __future__ import annotations

from pathlib import Path

import pytest
from bitbots_lego_assembly.core.segmentation import registry


@pytest.fixture
def fake_models_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a fake on-disk model layout with v1, v2 and v10."""
    base = tmp_path / "trained_models" / "instance_segmentation" / "yolo"
    for v in ("v1", "v2", "v10"):
        (base / v).mkdir(parents=True)
        (base / v / "model.onnx").write_bytes(b"\x00")
    monkeypatch.setenv("BITBOTS_LEGO_MODELS_ROOT", str(tmp_path / "trained_models"))
    return tmp_path / "trained_models"


def test_list_versions_sorted_numerically(fake_models_root: Path) -> None:
    assert registry.list_versions("yolo") == ["v1", "v2", "v10"]


def test_resolve_latest_picks_highest_numeric_suffix(fake_models_root: Path) -> None:
    assert registry.resolve("yolo").name == "v10"


@pytest.mark.parametrize("token", ["v2", "2", 2])
def test_resolve_accepts_multiple_version_formats(fake_models_root: Path, token: object) -> None:
    assert registry.resolve("yolo", version=token).name == "v2"


def test_resolve_missing_method_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BITBOTS_LEGO_MODELS_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        registry.resolve("yolo")


def test_resolve_missing_version_raises(fake_models_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        registry.resolve("yolo", version="v42")


def test_env_override_uses_provided_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BITBOTS_LEGO_MODELS_ROOT", str(tmp_path))
    assert registry.models_root() == tmp_path.resolve()


def test_default_root_falls_back_to_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BITBOTS_LEGO_MODELS_ROOT", raising=False)
    assert registry.models_root().name == "trained_models"
