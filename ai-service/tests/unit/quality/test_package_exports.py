"""Focused tests for app.quality package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_quality_surface() -> None:
    module = importlib.import_module("app.quality")

    expected = [
        "HIGH_QUALITY_THRESHOLD",
        "MIN_QUALITY_FOR_PRIORITY_SYNC",
        "MIN_QUALITY_FOR_TRAINING",
        "QualityThresholds",
        "get_quality_thresholds",
        "is_high_quality",
        "is_priority_sync_worthy",
        "is_training_worthy",
        "BatchQualityScorer",
        "QualityLevel",
        "QualityResult",
        "QualityScorer",
        "ValidationResult",
        "BaseQualityScorer",
        "ScorerConfig",
        "ScorerStats",
        "GameQuality",
        "QualityCategory",
        "UnifiedQualityScorer",
        "compute_game_quality",
        "compute_game_quality_from_params",
        "compute_sample_weight",
        "compute_sync_priority",
        "get_quality_category",
        "get_quality_scorer",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
