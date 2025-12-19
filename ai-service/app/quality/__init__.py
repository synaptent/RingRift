"""Unified Quality Scoring Module.

This module provides a SINGLE SOURCE OF TRUTH for all game quality
scoring and sample weighting operations across the RingRift system.

Consolidates logic from:
- app/training/game_quality_scorer.py
- app/distributed/quality_extractor.py
- app/training/streaming_pipeline.py
- app/training/elo_weighting.py
- app/distributed/unified_manifest.py

Usage:
    from app.quality import (
        UnifiedQualityScorer,
        compute_game_quality,
        compute_sample_weight,
        compute_sync_priority,
    )

    # Get singleton scorer
    scorer = UnifiedQualityScorer.get_instance()

    # Compute quality for a game
    quality = scorer.compute_game_quality(game_data)

    # Compute sample weight for training
    weight = scorer.compute_sample_weight(quality, recency_hours=2.0)
"""

from app.quality.unified_quality import (
    UnifiedQualityScorer,
    GameQuality,
    QualityCategory,
    compute_game_quality,
    compute_game_quality_from_params,
    compute_sample_weight,
    compute_sync_priority,
    get_quality_scorer,
    get_quality_category,
)

from app.quality.thresholds import (
    MIN_QUALITY_FOR_TRAINING,
    MIN_QUALITY_FOR_PRIORITY_SYNC,
    HIGH_QUALITY_THRESHOLD,
    QualityThresholds,
    get_quality_thresholds,
    is_training_worthy,
    is_priority_sync_worthy,
    is_high_quality,
)

__all__ = [
    # Quality scorer
    "UnifiedQualityScorer",
    "GameQuality",
    "QualityCategory",
    "compute_game_quality",
    "compute_game_quality_from_params",
    "compute_sample_weight",
    "compute_sync_priority",
    "get_quality_scorer",
    "get_quality_category",
    # Thresholds
    "MIN_QUALITY_FOR_TRAINING",
    "MIN_QUALITY_FOR_PRIORITY_SYNC",
    "HIGH_QUALITY_THRESHOLD",
    "QualityThresholds",
    "get_quality_thresholds",
    "is_training_worthy",
    "is_priority_sync_worthy",
    "is_high_quality",
]
