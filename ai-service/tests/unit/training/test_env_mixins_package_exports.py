"""Focused tests for app.training.env_mixins package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_env_mixins_surface() -> None:
    module = importlib.import_module("app.training.env_mixins")

    expected = [
        "BookkeepingMoveHandlerMixin",
        "MoveGenerationMixin",
        "RewardCalculatorMixin",
        "TerminationHandlerMixin",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
