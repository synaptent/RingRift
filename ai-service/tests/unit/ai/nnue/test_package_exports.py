"""Focused tests for app.ai.nnue_registry package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_nnue_registry_surface() -> None:
    module = importlib.import_module("app.ai.nnue_registry")

    expected = [
        "CANONICAL_CONFIGS",
        "NNUEModelInfo",
        "NNUERegistryStats",
        "get_nnue_canonical_path",
        "get_nnue_config_key",
        "get_nnue_model_info",
        "get_all_nnue_paths",
        "get_existing_nnue_models",
        "get_missing_nnue_models",
        "get_nnue_registry_stats",
        "get_nnue_output_path",
        "promote_nnue_model",
        "print_nnue_registry_status",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
