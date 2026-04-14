"""Focused tests for app.rules package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_rules_surface() -> None:
    module = importlib.import_module("app.rules")

    expected = [
        "RulesEngine",
        "get_rules_engine",
        "MutableGameState",
        "create_game_state",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
