"""Focused tests for app.mcts package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_mcts_surface() -> None:
    module = importlib.import_module("app.mcts")

    expected = [
        "ImprovedMCTS",
        "MCTSConfig",
        "MCTSNode",
        "MCTSWithPonder",
        "NeuralNetworkInterface",
        "ParallelMCTS",
        "TranspositionTable",
        "GameState",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
