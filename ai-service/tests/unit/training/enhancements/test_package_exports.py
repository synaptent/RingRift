"""Focused tests for app.training.enhancements package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_training_enhancements_surface() -> None:
    module = importlib.import_module("app.training.enhancements")

    expected = [
        "TrainingConfig",
        "GradientAccumulator",
        "AdaptiveGradientClipper",
        "CheckpointAverager",
        "average_checkpoints",
        "AdaptiveLRScheduler",
        "WarmRestartsScheduler",
        "SeedManager",
        "set_reproducible_seed",
        "CalibrationAutomation",
        "EWCRegularizer",
        "ModelEnsemble",
        "EvaluationFeedbackHandler",
        "create_evaluation_feedback_handler",
        "GameQualityScore",
        "DataQualityScorer",
        "QualityWeightedSampler",
        "compute_per_sample_loss",
        "PerSampleLossRecord",
        "PerSampleLossTracker",
        "HardExample",
        "HardExampleMiner",
        "FacadeConfig",
        "EpochStatistics",
        "TrainingEnhancementsFacade",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
