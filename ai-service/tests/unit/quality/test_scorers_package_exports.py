"""Focused tests for app.quality.scorers package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_scorers_surface() -> None:
    module = importlib.import_module("app.quality.scorers")

    expected = [
        "BaseQualityScorer",
        "ScorerConfig",
        "ScorerStats",
        "GameQualityScorer",
        "GameScorerConfig",
        "GameScorerWeights",
        "get_game_quality_scorer",
        "reset_game_quality_scorer",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
