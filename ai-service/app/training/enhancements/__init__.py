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
]
