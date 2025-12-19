"""Quality Thresholds - Convenience Module.

This module re-exports quality thresholds from app.config.thresholds
(the single source of truth) and provides convenience functions.

Usage:
    from app.quality.thresholds import (
        MIN_QUALITY_FOR_TRAINING,
        MIN_QUALITY_FOR_PRIORITY_SYNC,
        HIGH_QUALITY_THRESHOLD,
        get_quality_thresholds,
        is_high_quality,
    )

    if quality_score >= HIGH_QUALITY_THRESHOLD:
        print("High quality game!")

For direct constant access, prefer importing from app.config.thresholds.
This module adds helper functions and the QualityThresholds dataclass.

December 2025: Consolidated to use app.config.thresholds as canonical source.
"""

from __future__ import annotations

from dataclasses import dataclass

# Import from canonical source (app/config/thresholds.py)
try:
    from app.config.thresholds import (
        MIN_QUALITY_FOR_TRAINING,
        MIN_QUALITY_FOR_PRIORITY_SYNC,
        HIGH_QUALITY_THRESHOLD,
    )
except ImportError:
    # Fallback defaults if central config not available
    MIN_QUALITY_FOR_TRAINING = 0.3
    MIN_QUALITY_FOR_PRIORITY_SYNC = 0.5
    HIGH_QUALITY_THRESHOLD = 0.7


@dataclass(frozen=True)
class QualityThresholds:
    """Container for all quality thresholds.

    Use get_quality_thresholds() to get an instance with current values.
    """
    min_quality_for_training: float = MIN_QUALITY_FOR_TRAINING
    min_quality_for_priority_sync: float = MIN_QUALITY_FOR_PRIORITY_SYNC
    high_quality_threshold: float = HIGH_QUALITY_THRESHOLD

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
