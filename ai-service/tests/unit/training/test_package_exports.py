"""Focused tests for app.training package exports."""

from __future__ import annotations

import importlib
import warnings


def test_package_dir_lists_key_public_exports() -> None:
    module = importlib.import_module("app.training")

    expected = {
        "DataAugmentor",
        "PromotionController",
        "TemperatureScheduler",
        "SelfplayConfig",
        "RingRiftDataset",
        "StreamingDataLoader",
        "UnifiedTrainingOrchestrator",
        "get_model_architecture",
        "get_model_store",
        "HAS_TRAIN_CONFIG",
        "HAS_TRAIN_VALIDATION",
        "HAS_HIGH_TIER_CONFIG",
    }

    assert expected.issubset(set(module.__all__))
    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)


def test_package_dir_covers_declared_public_surface() -> None:
    module = importlib.import_module("app.training")

    assert len(module.__all__) == len(set(module.__all__))
    assert set(module.__all__).issubset(set(dir(module)))


def test_all_declared_training_exports_resolve() -> None:
    module = importlib.import_module("app.training")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        for name in module.__all__:
            assert hasattr(module, name)
            assert name in dir(module)


def test_stale_training_facade_exports_are_removed() -> None:
    module = importlib.import_module("app.training")

    stale_exports = {
        "HAS_CONFIG_RESOLVER",
        "AugmentorConfig",
        "ResolvedTrainingParams",
        "get_board_size",
        "get_effective_architecture",
        "resolve_training_params",
        "validate_model_id_for_board",
    }

    assert stale_exports.isdisjoint(set(module.__all__))
    assert stale_exports.isdisjoint(set(dir(module)))


def test_legacy_training_orchestrator_exports_remain_discoverable() -> None:
    module = importlib.import_module("app.training")

    for name in ("TrainingOrchestrator", "TrainingOrchestratorConfig", "get_training_orchestrator"):
        assert name in module.__all__
        assert name in dir(module)
        assert hasattr(module, name)
