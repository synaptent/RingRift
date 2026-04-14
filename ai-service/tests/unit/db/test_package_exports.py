"""Focused tests for app.db package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_key_public_exports() -> None:
    module = importlib.import_module("app.db")

    expected = {
        "DEFAULT_INTEGRITY_CHECK_TIMEOUT",
        "GameRecorder",
        "GameReplayDB",
        "GameWriter",
        "ParityMode",
        "ParityValidationError",
        "RecordSource",
        "RecordingConfig",
        "UnifiedGameRecorder",
        "check_database_integrity",
        "get_database_stats",
        "get_parity_mode",
        "get_unified_db",
        "record_game_unified",
        "validate_game_parity",
    }

    assert expected.issubset(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)


def test_package_dir_covers_declared_public_surface() -> None:
    module = importlib.import_module("app.db")

    assert len(module.__all__) == len(set(module.__all__))
    assert set(module.__all__).issubset(set(dir(module)))
