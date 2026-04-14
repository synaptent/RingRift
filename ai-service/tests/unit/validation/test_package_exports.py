"""Focused tests for app.validation package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_validation_surface() -> None:
    module = importlib.import_module("app.validation")

    expected = [
        "ValidationError",
        "ValidationResult",
        "Validator",
        "each_item",
        "has_keys",
        "has_length",
        "in_range",
        "is_instance",
        "is_non_negative",
        "is_not_empty",
        "is_positive",
        "is_type",
        "is_valid_board_type",
        "is_valid_config_key",
        "is_valid_elo",
        "is_valid_model_path",
        "matches_pattern",
        "max_length",
        "validate",
        "validate_all",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
