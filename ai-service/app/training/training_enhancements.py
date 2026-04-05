"""
Training Enhancements for RingRift AI.

This module provides advanced training optimizations:
1. Checkpoint averaging for improved final model
2. Gradient accumulation for larger effective batch sizes
3. Data quality scoring for sample prioritization (with freshness weighting)
4. Adaptive learning rate based on Elo progress
5. Early stopping with patience
6. EWC (Elastic Weight Consolidation) for continual learning
7. Model ensemble support for self-play
8. Value head calibration automation
9. Training anomaly detection (NaN/Inf, loss spikes, gradient explosions)
10. Configurable validation intervals (step/epoch-based, adaptive)

Usage:
    from app.training.training_enhancements import (
        TrainingConfig,
        CheckpointAverager,
        GradientAccumulator,
        DataQualityScorer,
        HardExampleMiner,
        AdaptiveLRScheduler,
        WarmRestartsScheduler,
        AdaptiveGradientClipper,
        EWCRegularizer,
        ModelEnsemble,
        EnhancedEarlyStopping,
        TrainingAnomalyDetector,
        ValidationIntervalManager,
        SeedManager,
        create_training_enhancements,
    )
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import torch.nn as nn
import torch.optim as optim

try:
    from app.quality.unified_quality import QualityWeights, UnifiedQualityScorer
    HAS_UNIFIED_QUALITY = True
except ImportError:
    HAS_UNIFIED_QUALITY = False
    QualityWeights = None
    UnifiedQualityScorer = None

# Import from modularized subpackage (December 2025)
from app.training.enhancements import (
    AdaptiveGradientClipper,
    AdaptiveLRScheduler,
    CalibrationAutomation,
    CheckpointAverager,
    DataQualityScorer,
    EWCRegularizer,
    EvaluationFeedbackHandler,
    GameQualityScore,
    GradientAccumulator,
    HardExample,
    HardExampleMiner,
    ModelEnsemble,
    PerSampleLossRecord,
    PerSampleLossTracker,
    QualityWeightedSampler,
    SeedManager,
    TrainingConfig,
    WarmRestartsScheduler,
    average_checkpoints,
    compute_per_sample_loss,
    create_evaluation_feedback_handler,
    set_reproducible_seed,
)

# Anomaly detection extracted to dedicated module (December 2025)
from app.training.anomaly_detection import (
    AnomalyEvent,
    TrainingAnomalyDetector,
    TrainingLossAnomalyHandler,
    wire_training_loss_anomaly_handler,
)

# Validation scheduling extracted to dedicated module (December 2025)
from app.training.validation_scheduling import (
    EnhancedEarlyStopping,
    ValidationIntervalManager,
    ValidationResult,
)
from app.training.validation_scheduling import EarlyStopping  # Backwards compatible alias

logger = logging.getLogger(__name__)

# Distillation integration (December 2025)
# Re-export distillation classes for unified training enhancements API
try:
    from app.training.distillation import (
        DistillationConfig,
        DistillationTrainer,
        EnsembleTeacher,
        SoftTargetLoss,
        create_distillation_trainer,
        distill_checkpoint_ensemble,
    )
    HAS_DISTILLATION = True
except ImportError:
    HAS_DISTILLATION = False
    DistillationConfig = None
    DistillationTrainer = None
    EnsembleTeacher = None
    SoftTargetLoss = None
    create_distillation_trainer = None
    distill_checkpoint_ensemble = None

__all__ = [
    # Gradient management
    "AdaptiveGradientClipper",
    # Learning rate schedulers
    "AdaptiveLRScheduler",
    # Anomaly detection (extracted to anomaly_detection.py December 2025)
    "AnomalyEvent",
    # Core utilities
    "CheckpointAverager",
    "DataQualityScorer",
    # Regularization (extracted to enhancements/ewc_regularization.py December 2025)
    "EWCRegularizer",
    "EarlyStopping",  # Backwards compatible alias
    # Training control
    "EnhancedEarlyStopping",
    # Evaluation feedback (extracted to enhancements/evaluation_feedback.py December 2025)
    "EvaluationFeedbackHandler",
    "create_evaluation_feedback_handler",
    "GradientAccumulator",
    "HardExampleMiner",
    # Ensemble (extracted to enhancements/model_ensemble.py December 2025)
    "ModelEnsemble",
    "PerSampleLossTracker",
    # Reproducibility
    "SeedManager",
    "TrainingAnomalyDetector",
    "TrainingLossAnomalyHandler",
    # Configuration
    "TrainingConfig",
    # Validation scheduling (extracted to validation_scheduling.py December 2025)
    "ValidationIntervalManager",
    "ValidationResult",
    "WarmRestartsScheduler",
    # Per-sample loss tracking (2025-12)
    "compute_per_sample_loss",
    # Distillation (2025-12)
    "DistillationConfig",
    "DistillationTrainer",
    "EnsembleTeacher",
    "SoftTargetLoss",
    "create_distillation_trainer",
    "distill_checkpoint_ensemble",
    # Factory function
    "create_training_enhancements",
    # Anomaly handler wiring (December 2025)
    "wire_training_loss_anomaly_handler",
]


# =============================================================================
# 0. Consolidated Training Configuration (Phase 7)
# =============================================================================

# MOVED: TrainingConfig is now imported from app.training.enhancements.training_config
# The class definition has been extracted to the enhancements subpackage.
# Import statement at top of file maintains backward compatibility.

# =============================================================================
# 1. Checkpoint Averaging
# =============================================================================


# MOVED: CheckpointAverager and average_checkpoints() are now imported from
# app.training.enhancements.checkpoint_averaging
# The class definitions have been extracted to the enhancements subpackage.

# =============================================================================
# 2. Gradient Accumulation
# =============================================================================


# MOVED: GradientAccumulator is now imported from
# app.training.enhancements.gradient_management
# The class definition has been extracted to the enhancements subpackage.

# =============================================================================
# 2b. Adaptive Gradient Clipping
# =============================================================================


# MOVED: AdaptiveGradientClipper is now imported from
# app.training.enhancements.gradient_management
# The class definition has been extracted to the enhancements subpackage.


# =============================================================================
# 3. Data Quality Scoring (MOVED December 2025)
# =============================================================================
# GameQualityScore, DataQualityScorer, QualityWeightedSampler moved to
# app.training.enhancements.data_quality_scoring
# Import at top of file maintains backward compatibility.


# =============================================================================
# 3b. Per-Sample Loss Tracking (MOVED December 2025)
# =============================================================================
# compute_per_sample_loss, PerSampleLossRecord, PerSampleLossTracker moved to
# app.training.enhancements.per_sample_loss
# Import at top of file maintains backward compatibility.


# =============================================================================
# 3c. Hard Example Mining (MOVED December 2025)
# =============================================================================
# HardExample, HardExampleMiner moved to
# app.training.enhancements.hard_example_mining
# Import at top of file maintains backward compatibility.


# =============================================================================
# 4. Adaptive Learning Rate (MOVED to enhancements/learning_rate_scheduling.py)
# =============================================================================
# AdaptiveLRScheduler - moved to app.training.enhancements.learning_rate_scheduling
# WarmRestartsScheduler - moved to app.training.enhancements.learning_rate_scheduling


# =============================================================================
# 4b. Training Anomaly Detection (MOVED December 2025)
# =============================================================================
# AnomalyEvent, TrainingAnomalyDetector moved to app.training.anomaly_detection
# Import at top of file maintains backward compatibility.


# =============================================================================
# 4c. Configurable Validation Intervals (MOVED December 2025)
# =============================================================================
# ValidationResult, ValidationIntervalManager moved to app.training.validation_scheduling
# Import at top of file maintains backward compatibility.


# =============================================================================
# 5. Enhanced Early Stopping (MOVED December 2025)
# =============================================================================
# EnhancedEarlyStopping, EarlyStopping moved to app.training.validation_scheduling
# Import at top of file maintains backward compatibility.


# =============================================================================
# 6. EWC (Elastic Weight Consolidation) for Continual Learning
# =============================================================================

# MOVED: EWCRegularizer is now imported from
# app.training.enhancements.ewc_regularization
# The class definition has been extracted to the enhancements subpackage.


# =============================================================================
# 7. Model Ensemble for Self-Play
# =============================================================================

# MOVED: ModelEnsemble is now imported from
# app.training.enhancements.model_ensemble
# The class definition has been extracted to the enhancements subpackage.


# Note: Value Head Calibration Automation moved to app.training.enhancements.calibration


# =============================================================================
# 8. Calibration & Seed Management (MOVED to enhancements/ subpackage)
# =============================================================================
# CalibrationAutomation - moved to app.training.enhancements.calibration
# SeedManager - moved to app.training.enhancements.seed_management
# set_reproducible_seed() - moved to app.training.enhancements.seed_management


# =============================================================================
# Convenience Functions
# =============================================================================


def create_training_enhancements(
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: dict[str, Any] | None = None,
    validation_fn: Callable[[nn.Module], tuple[float, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    """
    Create a suite of training enhancements with default configuration.

    Args:
        model: Model to enhance training for
        optimizer: Optimizer to use
        config: Optional configuration overrides
        validation_fn: Optional validation function for ValidationIntervalManager

    Returns:
        Dictionary of enhancement objects
    """
    config = config or {}
    if HAS_UNIFIED_QUALITY and QualityWeights is not None and UnifiedQualityScorer is not None:
        quality_weights = QualityWeights.from_config()
        # Keep the two legacy knobs wired for callers that still pass them via
        # create_training_enhancements(), even though scoring now comes from the
        # unified quality stack.
        quality_weights.recency_half_life_hours = config.get(
            'freshness_decay_hours',
            quality_weights.recency_half_life_hours,
        )
        quality_weights.recency_weight = config.get(
            'freshness_weight',
            quality_weights.recency_weight,
        )
        quality_scorer: Any = UnifiedQualityScorer(weights=quality_weights)
    else:
        quality_scorer = DataQualityScorer(
            freshness_decay_hours=config.get('freshness_decay_hours', 24.0),
            freshness_weight=config.get('freshness_weight', 0.2),
        )

    enhancements = {
        'checkpoint_averager': CheckpointAverager(
            num_checkpoints=config.get('avg_checkpoints', 5),
        ),
        'gradient_accumulator': GradientAccumulator(
            accumulation_steps=config.get('accumulation_steps', 1),
            max_grad_norm=config.get('max_grad_norm', 1.0),
        ),
        'quality_scorer': quality_scorer,
        'adaptive_lr': AdaptiveLRScheduler(
            optimizer=optimizer,
            base_lr=config.get('base_lr', 0.001),
        ),
        'early_stopping': EnhancedEarlyStopping(
            patience=config.get('patience', 10),
        ),
        'ewc': EWCRegularizer(
            model=model,
            lambda_ewc=config.get('lambda_ewc', 1000.0),
        ),
        'calibration': CalibrationAutomation(
            deviation_threshold=config.get('calibration_threshold', 0.05),
        ),
        'anomaly_detector': TrainingAnomalyDetector(
            loss_spike_threshold=config.get('loss_spike_threshold', 3.0),
            gradient_norm_threshold=config.get('gradient_norm_threshold', 100.0),
            halt_on_nan=config.get('halt_on_nan', True),
        ),
        'validation_manager': ValidationIntervalManager(
            validation_fn=validation_fn,
            interval_steps=config.get('validation_interval_steps', 1000),
            interval_epochs=config.get('validation_interval_epochs', None),
            subset_size=config.get('validation_subset_size', 1.0),
            adaptive_interval=config.get('adaptive_validation_interval', False),
        ),
        'hard_example_miner': HardExampleMiner(
            buffer_size=config.get('hard_example_buffer_size', 10000),
            hard_fraction=config.get('hard_example_fraction', 0.3),
            loss_threshold_percentile=config.get('hard_example_percentile', 80.0),
            min_samples_before_mining=config.get('min_samples_before_mining', 1000),
        ),
        'seed_manager': SeedManager(
            seed=config.get('seed'),
            deterministic=config.get('deterministic', False),
            benchmark=config.get('benchmark', True),
        ),
    }

    # Optionally add warm restarts scheduler
    if config.get('lr_scheduler') == 'warm_restarts':
        enhancements['warm_restarts_scheduler'] = WarmRestartsScheduler(
            optimizer=optimizer,
            T_0=config.get('warm_restart_t0', 10),
            T_mult=config.get('warm_restart_t_mult', 2),
            eta_min=config.get('warm_restart_eta_min', 1e-6),
            warmup_steps=config.get('warmup_steps', 0),
        )

    return enhancements


# =============================================================================
# Evaluation Feedback Handler (Phase 2 - December 2025)
# =============================================================================

# MOVED: EvaluationFeedbackHandler and create_evaluation_feedback_handler are now imported from
# app.training.enhancements.evaluation_feedback
# The class and factory function have been extracted to the enhancements subpackage.


# =============================================================================
# TRAINING_LOSS_ANOMALY → QUALITY_CHECK Handler (MOVED December 2025)
# =============================================================================
# TrainingLossAnomalyHandler, wire_training_loss_anomaly_handler moved to
# app.training.anomaly_detection
# Import at top of file maintains backward compatibility.
