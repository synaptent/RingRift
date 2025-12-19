"""Quality Thresholds - Single Source of Truth.

This module provides the canonical quality thresholds used across the codebase.
All code that needs quality thresholds should import from here rather than
defining hardcoded values.

Usage:
    from app.quality.thresholds import (
        MIN_QUALITY_FOR_TRAINING,
        MIN_QUALITY_FOR_PRIORITY_SYNC,
        HIGH_QUALITY_THRESHOLD,
        get_quality_thresholds,
    )

    if quality_score >= HIGH_QUALITY_THRESHOLD:
        print("High quality game!")

Thresholds are loaded from QualityConfig when available, with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Try to load from unified config
_config_thresholds = None
try:
    from app.config.unified_config import get_config
    _unified_config = get_config()
    if _unified_config and hasattr(_unified_config, 'quality'):
        _config_thresholds = _unified_config.quality
except ImportError:
    pass
except Exception:
    pass


# Default threshold values (used if config unavailable)
_DEFAULT_MIN_QUALITY_FOR_TRAINING = 0.3
_DEFAULT_MIN_QUALITY_FOR_PRIORITY_SYNC = 0.5
_DEFAULT_HIGH_QUALITY_THRESHOLD = 0.7


def _get_threshold(attr: str, default: float) -> float:
    """Get threshold from config or use default."""
    if _config_thresholds is not None:
        return getattr(_config_thresholds, attr, default)
    return default


# Canonical threshold constants
MIN_QUALITY_FOR_TRAINING: float = _get_threshold(
    "min_quality_for_training", _DEFAULT_MIN_QUALITY_FOR_TRAINING
)
MIN_QUALITY_FOR_PRIORITY_SYNC: float = _get_threshold(
    "min_quality_for_priority_sync", _DEFAULT_MIN_QUALITY_FOR_PRIORITY_SYNC
)
HIGH_QUALITY_THRESHOLD: float = _get_threshold(
    "high_quality_threshold", _DEFAULT_HIGH_QUALITY_THRESHOLD
)


@dataclass(frozen=True)
class QualityThresholds:
    """Container for all quality thresholds.

    Use get_quality_thresholds() to get an instance with current values.
    """
    min_quality_for_training: float = _DEFAULT_MIN_QUALITY_FOR_TRAINING
    min_quality_for_priority_sync: float = _DEFAULT_MIN_QUALITY_FOR_PRIORITY_SYNC
    high_quality_threshold: float = _DEFAULT_HIGH_QUALITY_THRESHOLD

    def is_training_worthy(self, score: float) -> bool:
        """Check if score meets minimum quality for training."""
        return score >= self.min_quality_for_training

    def is_priority_sync_worthy(self, score: float) -> bool:
        """Check if score meets minimum quality for priority sync."""
        return score >= self.min_quality_for_priority_sync

    def is_high_quality(self, score: float) -> bool:
        """Check if score meets high quality threshold."""
        return score >= self.high_quality_threshold


def get_quality_thresholds() -> QualityThresholds:
    """Get quality thresholds (loads from config if available)."""
    return QualityThresholds(
        min_quality_for_training=MIN_QUALITY_FOR_TRAINING,
        min_quality_for_priority_sync=MIN_QUALITY_FOR_PRIORITY_SYNC,
        high_quality_threshold=HIGH_QUALITY_THRESHOLD,
    )


# Convenience functions
def is_training_worthy(score: float) -> bool:
    """Check if score meets minimum quality for training."""
    return score >= MIN_QUALITY_FOR_TRAINING


def is_priority_sync_worthy(score: float) -> bool:
    """Check if score meets minimum quality for priority sync."""
    return score >= MIN_QUALITY_FOR_PRIORITY_SYNC


def is_high_quality(score: float) -> bool:
    """Check if score meets high quality threshold."""
    return score >= HIGH_QUALITY_THRESHOLD
