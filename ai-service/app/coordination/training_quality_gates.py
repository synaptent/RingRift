"""Training quality gate functions.

Jan 4, 2026 - Sprint 17.9: Extracted from training_trigger_daemon.py as part of
daemon decomposition following selfplay_scheduler → priority_calculator.py pattern.

This module contains pure quality gate functions with no I/O dependencies:
- compute_decayed_quality_score() - Exponential decay for stale quality scores
- intensity_from_quality() - Map quality scores to training intensity levels
- check_quality_gate_conditions() - Pure logic for quality gate decisions

Usage:
    from app.coordination.training_quality_gates import (
        compute_decayed_quality_score,
        intensity_from_quality,
        check_quality_gate_conditions,
        QualityGateResult,
        QUALITY_INTENSITY_THRESHOLDS,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from app.config.coordination_defaults import QualityGateDefaults
from app.config.thresholds import (
    apply_quality_confidence_weighting,
    get_quality_confidence,
)

logger = logging.getLogger(__name__)

# Quality score to intensity mapping thresholds
# Maps quality_score ranges to training intensity levels
QUALITY_INTENSITY_THRESHOLDS = {
    "hot_path": 0.90,      # >= 0.90 → hot_path (most aggressive training)
    "accelerated": 0.80,   # >= 0.80 → accelerated
    "normal": 0.65,        # >= 0.65 → normal
    # Below 0.65 depends on per-config threshold
    # >= min_threshold → reduced
    # < min_threshold → paused
}

# Quality decay constants
DEFAULT_DECAY_HALF_LIFE_HOURS = 1.0  # Quality confidence halves every hour
DEFAULT_DECAY_FLOOR = 0.5  # Never decay below this quality level

# Quality gate relaxation thresholds
MINIMUM_QUALITY_FLOOR = 0.40  # Absolute minimum (prevents garbage data training)
DATA_STARVED_THRESHOLD = 5000  # Configs with <5K games are bootstrapping
TRAINING_STALL_HOURS = 24.0  # Emergency override after 24h stall


@dataclass
class QualityGateResult:
    """Result of quality gate check.

    Contains decision and supporting information for logging/debugging.
    """

    passed: bool
    reason: str
    quality_score: float | None = None
    threshold: float | None = None
    is_relaxed: bool = False
    relaxed_reason: str = ""


def compute_quality_confidence(games_assessed: int) -> float:
    """Compute confidence factor based on number of games assessed.

    Delegates to centralized thresholds.py for consistent confidence tier
    definitions across the codebase.

    Args:
        games_assessed: Number of games used in quality assessment

    Returns:
        Confidence factor between 0.5 and 1.0
    """
    return get_quality_confidence(games_assessed)


def apply_confidence_weighting(quality_score: float, games_assessed: int) -> float:
    """Apply confidence weighting to quality score.

    Delegates to centralized thresholds.py for consistent confidence weighting.

    Args:
        quality_score: Raw quality score (0.0-1.0)
        games_assessed: Number of games used in assessment

    Returns:
        Confidence-adjusted quality score
    """
    return apply_quality_confidence_weighting(quality_score, games_assessed)


def compute_decayed_quality_score(
    last_quality_score: float,
    last_quality_update: float,
    current_time: float,
    decay_enabled: bool = True,
    half_life_hours: float = DEFAULT_DECAY_HALF_LIFE_HOURS,
    decay_floor: float = DEFAULT_DECAY_FLOOR,
) -> float:
    """Apply confidence decay to stored quality score.

    Quality scores decay over time to prevent stale assessments from blocking
    training. If no new quality data arrives, the effective quality score
    drops toward the decay floor, potentially unblocking training.

    Uses exponential decay: effective_quality = floor + (score - floor) * 0.5^(age/half_life)

    Args:
        last_quality_score: The stored quality score (0.0-1.0)
        last_quality_update: Timestamp when quality was last updated
        current_time: Current timestamp
        decay_enabled: Whether decay is enabled
        half_life_hours: Time in hours for quality confidence to halve
        decay_floor: Minimum quality score (never decays below this)

    Returns:
        Decayed quality score (never below decay_floor)
    """
    if not decay_enabled:
        return last_quality_score

    if last_quality_update <= 0:
        # No quality data yet - use default
        return last_quality_score

    age_hours = (current_time - last_quality_update) / 3600.0

    if age_hours <= 0:
        return last_quality_score

    # Exponential decay toward floor
    decay_factor = 0.5 ** (age_hours / half_life_hours)

    # Decay from current score toward floor
    decayed = decay_floor + (last_quality_score - decay_floor) * decay_factor
    return max(decay_floor, decayed)


def intensity_from_quality(
    quality_score: float,
    config_key: str | None = None,
) -> str:
    """Map quality scores to training intensity.

    Higher quality data allows more aggressive training (hot_path),
    while lower quality data requires more conservative training (reduced/paused).

    Args:
        quality_score: The data quality score (0.0 to 1.0)
        config_key: Optional config key (e.g., "hex8_4p") for per-config thresholds

    Returns:
        Training intensity level: "hot_path", "accelerated", "normal", "reduced", or "paused"
    """
    # Get config-specific minimum threshold (4p configs need higher quality)
    min_threshold = QualityGateDefaults.get_quality_threshold(config_key or "")

    if quality_score >= QUALITY_INTENSITY_THRESHOLDS["hot_path"]:
        return "hot_path"
    if quality_score >= QUALITY_INTENSITY_THRESHOLDS["accelerated"]:
        return "accelerated"
    if quality_score >= QUALITY_INTENSITY_THRESHOLDS["normal"]:
        return "normal"
    if quality_score >= min_threshold:
        return "reduced"
    return "paused"


def check_quality_gate_conditions(
    quality_score: float | None,
    config_key: str,
    game_count: int | None = None,
    hours_since_training: float | None = None,
    decayed_quality: float | None = None,
) -> QualityGateResult:
    """Check if quality gate conditions allow training.

    This is a pure function containing the quality gate decision logic.
    It does not perform I/O - callers must provide the quality score and
    other required data.

    Args:
        quality_score: Current quality score (None if unavailable)
        config_key: Configuration key (e.g., "hex8_2p")
        game_count: Number of games for this config (for bootstrap mode)
        hours_since_training: Hours since last training (for stall detection)
        decayed_quality: Decayed quality score to use if quality_score is None

    Returns:
        QualityGateResult with decision and reasoning
    """
    # Get config-specific quality threshold
    quality_threshold = QualityGateDefaults.get_quality_threshold(config_key)

    # No quality data available
    if quality_score is None:
        if decayed_quality is not None and decayed_quality > 0:
            # Use decayed quality from stored state
            quality_score = decayed_quality
            logger.debug(
                f"[QualityGate] {config_key}: using decayed quality {quality_score:.2f}"
            )
        else:
            # No quality data at all - allow training with warning
            return QualityGateResult(
                passed=True,
                reason="no quality data (proceeding anyway)",
                quality_score=None,
                threshold=quality_threshold,
            )

    # Quality meets threshold
    if quality_score >= quality_threshold:
        return QualityGateResult(
            passed=True,
            reason=f"quality ok ({quality_score:.2f})",
            quality_score=quality_score,
            threshold=quality_threshold,
        )

    # Quality below threshold - check for relaxation conditions
    allow_degraded = False
    relaxed_reason = ""

    # Check minimum floor
    if quality_score >= MINIMUM_QUALITY_FLOOR:
        # Check if data-starved (bootstrap mode)
        if game_count is not None and game_count < DATA_STARVED_THRESHOLD:
            allow_degraded = True
            relaxed_reason = f"bootstrap mode ({game_count} < {DATA_STARVED_THRESHOLD} games)"

        # Check if training is stalled (emergency override)
        if (
            not allow_degraded
            and hours_since_training is not None
            and hours_since_training > TRAINING_STALL_HOURS
        ):
            allow_degraded = True
            relaxed_reason = (
                f"training stalled ({hours_since_training:.1f}h > {TRAINING_STALL_HOURS}h)"
            )

    if allow_degraded:
        return QualityGateResult(
            passed=True,
            reason=f"quality degraded but allowed ({quality_score:.2f}, {relaxed_reason})",
            quality_score=quality_score,
            threshold=quality_threshold,
            is_relaxed=True,
            relaxed_reason=relaxed_reason,
        )

    # Quality too low - block training
    return QualityGateResult(
        passed=False,
        reason=f"quality too low ({quality_score:.2f} < {quality_threshold})",
        quality_score=quality_score,
        threshold=quality_threshold,
    )


def get_quality_from_state(
    last_quality_score: float,
    last_quality_update: float,
    last_training_time: float,
    current_time: float,
    decay_config: dict | None = None,
) -> tuple[float | None, float | None]:
    """Get quality score with optional decay from training state.

    Helper function to compute quality and hours-since-training from state fields.

    Args:
        last_quality_score: Stored quality score
        last_quality_update: Timestamp of last quality update
        last_training_time: Timestamp of last training
        current_time: Current timestamp
        decay_config: Optional decay configuration dict with:
            - enabled: bool
            - half_life_hours: float
            - floor: float

    Returns:
        Tuple of (decayed_quality, hours_since_training)
    """
    # Compute decayed quality
    decayed_quality = None
    if last_quality_update > 0:
        decay_enabled = decay_config.get("enabled", True) if decay_config else True
        half_life = (
            decay_config.get("half_life_hours", DEFAULT_DECAY_HALF_LIFE_HOURS)
            if decay_config
            else DEFAULT_DECAY_HALF_LIFE_HOURS
        )
        floor = (
            decay_config.get("floor", DEFAULT_DECAY_FLOOR)
            if decay_config
            else DEFAULT_DECAY_FLOOR
        )
        decayed_quality = compute_decayed_quality_score(
            last_quality_score,
            last_quality_update,
            current_time,
            decay_enabled=decay_enabled,
            half_life_hours=half_life,
            decay_floor=floor,
        )

    # Compute hours since training
    hours_since_training = None
    if last_training_time > 0:
        hours_since_training = (current_time - last_training_time) / 3600.0

    return decayed_quality, hours_since_training


__all__ = [
    # Constants
    "QUALITY_INTENSITY_THRESHOLDS",
    "DEFAULT_DECAY_HALF_LIFE_HOURS",
    "DEFAULT_DECAY_FLOOR",
    "MINIMUM_QUALITY_FLOOR",
    "DATA_STARVED_THRESHOLD",
    "TRAINING_STALL_HOURS",
    # Dataclasses
    "QualityGateResult",
    # Functions
    "compute_quality_confidence",
    "apply_confidence_weighting",
    "compute_decayed_quality_score",
    "intensity_from_quality",
    "check_quality_gate_conditions",
    "get_quality_from_state",
]
