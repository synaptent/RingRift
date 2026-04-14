"""Focused tests for app.training package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_key_public_exports() -> None:
    module = importlib.import_module("app.training")

    expected = {
        "PromotionController",
        "TemperatureScheduler",
        "SelfplayConfig",
        "RingRiftDataset",
        "StreamingDataLoader",
        "UnifiedTrainingOrchestrator",
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


def test_legacy_training_orchestrator_exports_remain_discoverable() -> None:
    module = importlib.import_module("app.training")

    for name in ("TrainingOrchestrator", "TrainingOrchestratorConfig", "get_training_orchestrator"):
        assert name in module.__all__
        assert name in dir(module)
        assert hasattr(module, name)
