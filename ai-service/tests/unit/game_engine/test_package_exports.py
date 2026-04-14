"""Focused tests for app.game_engine package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_game_engine_surface() -> None:
    module = importlib.import_module("app.game_engine")

    expected = [
        "STRICT_NO_MOVE_INVARIANT",
        "GameEngine",
        "PhaseRequirement",
        "PhaseRequirementType",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
