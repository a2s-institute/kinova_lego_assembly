"""Tests for the pure-Python blueprint utilities."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from bitbots_lego_assembly.core.blueprint import BlueprintError, BlueprintUtils

_VALID = textwrap.dedent(
    """
    blocks:
      - { id: 1, type: "2x2", colour: red, description: "red 2x2 lego brick" }
      - { id: 8, type: "4x2", colour: yellow, description: "yellow 4x2 lego brick" }
    models:
      - id: 81
        name: e_stop
        steps:
          - { id: 8 }
          - { id: 1 }
    """
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "bp.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_synthesized_names(tmp_path: Path) -> None:
    bp = BlueprintUtils.from_file(_write(tmp_path, _VALID))
    names = {block.id: block.name for block in bp.blocks}
    assert names[1] == "2x2_red"
    assert names[8] == "4x2_yellow"


def test_resolve_by_id_and_name(tmp_path: Path) -> None:
    bp = BlueprintUtils.from_file(_write(tmp_path, _VALID))
    by_id = bp.resolve_model("81")
    by_name = bp.resolve_model("e_stop")
    assert by_id.name == by_name.name == "e_stop"
    assert [s.brick_name for s in by_id.steps] == ["4x2_yellow", "2x2_red"]


def test_demand_counts(tmp_path: Path) -> None:
    bp = BlueprintUtils.from_file(_write(tmp_path, _VALID))
    demand = BlueprintUtils.demand_counts(bp.resolve_model("e_stop"))
    assert demand == {"4x2_yellow": 1, "2x2_red": 1}


def test_unknown_model_raises(tmp_path: Path) -> None:
    bp = BlueprintUtils.from_file(_write(tmp_path, _VALID))
    with pytest.raises(BlueprintError):
        bp.resolve_model("does_not_exist")


def test_invalid_blueprint_raises(tmp_path: Path) -> None:
    bad = _VALID.replace("- { id: 1 }", "- { id: 99 }")
    with pytest.raises(BlueprintError) as exc:
        BlueprintUtils.from_file(_write(tmp_path, bad))
    assert exc.value.errors


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(BlueprintError):
        BlueprintUtils.from_file(tmp_path / "nope.yaml")
