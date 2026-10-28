"""Tests for the pure-Python scene validator."""

from __future__ import annotations

from bitbots_lego_assembly.core.blueprint import ModelRecipe, ModelStep
from bitbots_lego_assembly.core.scene_validation import count_detected, verify


def _recipe() -> ModelRecipe:
    return ModelRecipe(
        id=81,
        name="e_stop",
        steps=[
            ModelStep(0, 8, "4x2_yellow"),
            ModelStep(1, 1, "2x2_red"),
        ],
    )


def test_count_detected() -> None:
    assert count_detected(["2x2_red", "2x2_red", "4x2_yellow"]) == {
        "2x2_red": 2,
        "4x2_yellow": 1,
    }


def test_enough() -> None:
    report = verify({"4x2_yellow": 1, "2x2_red": 3}, _recipe())
    assert report.enough
    assert report.shortfall == {}


def test_shortfall() -> None:
    report = verify({"4x2_yellow": 1}, _recipe())
    assert not report.enough
    assert report.shortfall == {"2x2_red": 1}


def test_unrelated_detections_ignored() -> None:
    report = verify({"4x2_yellow": 1, "2x2_red": 1, "2x2_blue": 5}, _recipe())
    assert report.enough
    assert "2x2_blue" not in report.available
