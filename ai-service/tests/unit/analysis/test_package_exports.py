"""Focused tests for app.analysis package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_analysis_surface() -> None:
    module = importlib.import_module("app.analysis")

    expected = [
        "GameBalanceAnalyzer",
        "BalanceReport",
        "BalanceIssue",
        "WinRateStats",
        "GameLengthStats",
        "CrossConfigAnalysis",
        "analyze_game_balance",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
