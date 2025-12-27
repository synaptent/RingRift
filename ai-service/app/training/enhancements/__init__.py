"""
Training Enhancements Subpackage for RingRift AI.

This subpackage provides modularized training enhancement components:
- Training configuration
- Gradient management (accumulation and adaptive clipping)
- Checkpoint averaging
- Data quality scoring
- Hard example mining
- Learning rate schedulers
- Early stopping
- EWC regularization
- Model ensembles
- Anomaly detection
- Validation management
- Seed management

Extracted from training_enhancements.py (December 2025 modularization).
"""

from __future__ import annotations

# Phase 1 exports: Training config, gradient management, checkpoint averaging
from app.training.enhancements.checkpoint_averaging import (
    CheckpointAverager,
    average_checkpoints,
)
from app.training.enhancements.gradient_management import (
    AdaptiveGradientClipper,
    GradientAccumulator,
)
from app.training.enhancements.training_config import TrainingConfig

# Phase 2 exports: Learning rate scheduling, seed management, calibration
from app.training.enhancements.calibration import CalibrationAutomation
from app.training.enhancements.learning_rate_scheduling import (
    AdaptiveLRScheduler,
    WarmRestartsScheduler,
)
from app.training.enhancements.seed_management import (
    SeedManager,
    set_reproducible_seed,
)

# Phase 3 exports: EWC, Model Ensemble, Evaluation Feedback (December 2025)
from app.training.enhancements.evaluation_feedback import (
    EvaluationFeedbackHandler,
    create_evaluation_feedback_handler,
)
from app.training.enhancements.ewc_regularization import EWCRegularizer
from app.training.enhancements.model_ensemble import ModelEnsemble

# Phase 4 exports: Data quality, per-sample loss, hard example mining (December 2025)
from app.training.enhancements.data_quality_scoring import (
    DataQualityScorer,
    GameQualityScore,
    QualityWeightedSampler,
)
from app.training.enhancements.hard_example_mining import (
    HardExample,
    HardExampleMiner,
)
from app.training.enhancements.per_sample_loss import (
    PerSampleLossRecord,
    PerSampleLossTracker,
    compute_per_sample_loss,
)

# Phase 5 exports: Unified Training Facade (December 2025)
from app.training.enhancements.training_facade import (
    EpochStatistics,
    FacadeConfig,
    TrainingEnhancementsFacade,
)

__all__ = [
    # Training configuration
    "TrainingConfig",
    # Gradient management
    "GradientAccumulator",
    "AdaptiveGradientClipper",
    # Checkpoint averaging
    "CheckpointAverager",
    "average_checkpoints",
    # Learning rate scheduling
    "AdaptiveLRScheduler",
    "WarmRestartsScheduler",
    # Seed management
    "SeedManager",
    "set_reproducible_seed",
    # Calibration
    "CalibrationAutomation",
    # EWC Regularization (Phase 3 - December 2025)
    "EWCRegularizer",
    # Model Ensemble (Phase 3 - December 2025)
    "ModelEnsemble",
    # Evaluation Feedback (Phase 3 - December 2025)
    "EvaluationFeedbackHandler",
    "create_evaluation_feedback_handler",
    # Data quality scoring (Phase 4 - December 2025)
    "GameQualityScore",
    "DataQualityScorer",
    "QualityWeightedSampler",
    # Per-sample loss tracking (Phase 4 - December 2025)
    "compute_per_sample_loss",
    "PerSampleLossRecord",
    "PerSampleLossTracker",
    # Hard example mining (Phase 4 - December 2025)
    "HardExample",
    "HardExampleMiner",
    # Unified Training Facade (Phase 5 - December 2025)
    "FacadeConfig",
    "EpochStatistics",
    "TrainingEnhancementsFacade",
]
