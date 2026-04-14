"""Focused tests for app.routes package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_routes_surface() -> None:
    module = importlib.import_module("app.routes")

    expected = [
        "include_all_routes",
        "replay_router",
        "cluster_router",
        "training_router",
        "human_games_router",
        "online_learning_router",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
