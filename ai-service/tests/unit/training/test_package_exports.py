"""Focused tests for app.training package exports."""

from __future__ import annotations

import importlib
import warnings

EXPECTED_PUBLIC_TRAINING_EXPORTS = [
    "UnifiedTrainingOrchestrator",
    "OrchestratorConfig",
    "TrainingOrchestrator",
    "TrainingOrchestratorConfig",
    "TrainingOrchestratorState",
    "get_training_orchestrator",
    "UnifiedModelStore",
    "ModelInfo",
    "ModelStoreStage",
    "ModelStoreType",
    "get_model_store",
    "get_production_model",
    "register_model",
    "promote_model",
    "PromotionController",
    "PromotionCriteria",
    "PromotionDecision",
    "PromotionType",
    "get_promotion_controller",
    "RegressionConfig",
    "RegressionDetector",
    "RegressionSeverity",
    "get_regression_detector",
]


def test_package_dir_lists_declared_public_training_surface() -> None:
    module = importlib.import_module("app.training")

    assert module.__all__ == EXPECTED_PUBLIC_TRAINING_EXPORTS
    assert len(module.__all__) == 23
    assert len(module.__all__) < 25
    assert len(module.__all__) == len(set(module.__all__))

    for name in EXPECTED_PUBLIC_TRAINING_EXPORTS:
        assert hasattr(module, name)
        assert name in dir(module)


def test_package_dir_covers_declared_public_surface() -> None:
    module = importlib.import_module("app.training")

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
        "DataAugmentor",
        "DistributedMetrics",
        "HAS_HIGH_TIER_CONFIG",
        "HAS_TRAIN_CONFIG",
        "HAS_TRAIN_VALIDATION",
        "get_model_architecture",
        "HAS_CONFIG_RESOLVER",
        "AugmentorConfig",
        "ResolvedTrainingParams",
        "get_board_size",
        "get_effective_architecture",
        "resolve_training_params",
        "validate_model_id_for_board",
        "SelfplayConfig",
        "EngineMode",
        "OutputFormat",
        "create_argument_parser",
        "get_default_config",
        "get_production_config",
        "parse_selfplay_args",
        "RingRiftDataset",
        "WeightedRingRiftDataset",
        "StreamingDataLoader",
        "WeightedStreamingDataLoader",
        "HotDataBuffer",
        "create_hot_buffer",
        "DistributedConfig",
        "setup_distributed",
        "cleanup_distributed",
        "is_main_process",
        "EBMOOnlineAI",
        "EBMOOnlineConfig",
        "EBMOOnlineLearner",
        "OnlineLearningConfig",
        "create_online_learner",
        "TemperatureScheduler",
        "TemperatureConfig",
        "create_temperature_scheduler",
        "wilson_score_interval",
    }

    assert stale_exports.isdisjoint(set(module.__all__))
    assert stale_exports.isdisjoint(set(dir(module)))


def test_legacy_training_orchestrator_exports_remain_discoverable() -> None:
    module = importlib.import_module("app.training")

    for name in ("TrainingOrchestrator", "TrainingOrchestratorConfig", "get_training_orchestrator"):
        assert name in module.__all__
        assert name in dir(module)
        assert hasattr(module, name)
