"""Focused tests for app.db package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_key_public_exports() -> None:
    module = importlib.import_module("app.db")

    expected = [
        "GameReplayDB",
        "GameWriter",
        "ParityValidationError",
        "RecordSource",
        "get_or_create_db",
        "record_completed_game",
        "record_completed_game_with_parity_check",
        "validate_game_parity",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)


def test_historical_root_imports_remain_lazy_compatibility_aliases() -> None:
    module = importlib.import_module("app.db")

    assert "check_database_integrity" not in module.__all__
    assert "check_database_integrity" not in dir(module)
    assert callable(module.check_database_integrity)
