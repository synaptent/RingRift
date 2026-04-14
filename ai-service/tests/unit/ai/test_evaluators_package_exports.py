"""Focused tests for app.ai.evaluators package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_evaluators_surface() -> None:
    module = importlib.import_module("app.ai.evaluators")

    expected = [
        "EndgameEvaluator",
        "EndgameScore",
        "EndgameWeights",
        "MaterialEvaluator",
        "MaterialScore",
        "MaterialWeights",
        "MobilityEvaluator",
        "MobilityScore",
        "MobilityWeights",
        "PositionalEvaluator",
        "PositionalScore",
        "PositionalWeights",
        "StrategicEvaluator",
        "StrategicScore",
        "StrategicWeights",
        "TacticalEvaluator",
        "TacticalScore",
        "TacticalWeights",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
