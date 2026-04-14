"""Focused tests for app.notation package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_notation_surface() -> None:
    module = importlib.import_module("app.notation")

    expected = [
        "CODE_TO_MOVE_TYPE",
        "MOVE_TYPE_TO_CODE",
        "algebraic_to_move",
        "algebraic_to_position",
        "game_to_pgn",
        "move_to_algebraic",
        "moves_to_notation_list",
        "parse_pgn",
        "position_to_algebraic",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
