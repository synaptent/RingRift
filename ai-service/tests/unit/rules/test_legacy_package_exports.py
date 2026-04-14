"""Focused tests for app.rules.legacy package exports."""

from __future__ import annotations

import importlib
import warnings


def test_package_dir_lists_declared_legacy_rules_surface() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("app.rules.legacy")

    warning_text = [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, PendingDeprecationWarning)
    ]
    assert any("backward compatibility" in text.lower() for text in warning_text)

    expected = [
        "LEGACY_TO_CANONICAL_MOVE_TYPE",
        "convert_legacy_move_type",
        "is_legacy_move_type",
        "replay_with_legacy_fallback",
        "requires_legacy_replay",
        "normalize_legacy_phase",
        "normalize_legacy_state",
        "normalize_legacy_status",
        "auto_advance_phase",
        "get_auto_advance_stats",
        "reset_auto_advance_stats",
        "auto_inject_before_move",
        "auto_inject_no_action_moves",
        "get_phase_injection_stats",
        "reset_phase_injection_stats",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
