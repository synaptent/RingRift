"""Focused tests for app.quality.validators package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_validators_surface() -> None:
    module = importlib.import_module("app.quality.validators")

    expected = [
        "BaseValidator",
        "ValidatorConfig",
        "DatabaseValidator",
        "DatabaseValidatorConfig",
        "NpzValidator",
        "NpzValidatorConfig",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
