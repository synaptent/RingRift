"""Focused tests for app.ai.harness package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_harness_surface() -> None:
    module = importlib.import_module("app.ai.harness")

    expected = [
        "AIHarness",
        "HarnessType",
        "ModelType",
        "EvaluationMetadata",
        "HarnessCompatibility",
        "create_harness",
        "get_harness_compatibility",
        "get_compatible_harnesses",
        "get_all_harness_types",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
