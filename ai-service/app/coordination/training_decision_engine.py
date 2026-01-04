"""Training decision engine with pure decision functions.

Jan 4, 2026 - Sprint 17.9: Extracted from training_trigger_daemon.py as part of
daemon decomposition following selfplay_scheduler → priority_calculator.py pattern.

This module contains pure decision logic functions with no side effects:
- compute_velocity_adjusted_cooldown() - Cooldown modulation based on Elo velocity
- get_training_params_for_intensity() - Training parameters for intensity level
- apply_velocity_amplification() - Velocity-based parameter adjustment
- compute_dynamic_sample_threshold() - Bootstrap-aware sample thresholds
- check_confidence_early_trigger() - Statistical confidence validation

These functions accept dependencies as parameters (callback-based injection) to
maintain testability and avoid hard-coded imports.

Usage:
    from app.coordination.training_decision_engine import (
        compute_velocity_adjusted_cooldown,
        get_training_params_for_intensity,
        apply_velocity_amplification,
        compute_dynamic_sample_threshold,
        check_confidence_early_trigger,
    )

    # Pure function calls with explicit dependencies
    cooldown = compute_velocity_adjusted_cooldown(
        base_cooldown_hours=0.083,
        velocity=1.5,
        velocity_trend="accelerating",
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Default training parameters by intensity level
INTENSITY_DEFAULTS: dict[str, tuple[int, int, float]] = {
    # hot_path: Fast iteration with larger batches, higher LR
    "hot_path": (30, 1024, 1.5),
    # accelerated: More aggressive training
    "accelerated": (40, 768, 1.2),
    # normal: Default parameters (overridden by config)
    "normal": (50, 512, 1.0),
    # reduced: Slower, more careful training for struggling configs
    "reduced": (60, 256, 0.8),
    # paused: Should not reach here, but use minimal params
    "paused": (10, 128, 0.5),
}

# Velocity-based cooldown multipliers
VELOCITY_TREND_COOLDOWN_MULTIPLIERS: dict[str, float] = {
    "accelerating": 0.5,    # 50% cooldown - train faster
    "stable": 1.0,          # Normal cooldown
    "decelerating": 1.5,    # 150% cooldown - train slower
    # January 3, 2026: Fixed plateauing multiplier - should train MORE aggressively
    # to break the plateau, not slower. This was backwards before.
    "plateauing": 0.6,      # 60% cooldown - train faster to break plateau
}

# Player count threshold multipliers
PLAYER_COUNT_THRESHOLD_MULTIPLIERS: dict[int, float] = {
    2: 1.0,    # 2p: Full threshold (5000 samples)
    3: 0.6,    # 3p: 60% threshold (3000 samples)
    4: 0.4,    # 4p: 40% threshold (2000 samples)
}

# Bootstrap tier thresholds (game count → sample threshold)
BOOTSTRAP_TIERS: list[tuple[int, int]] = [
    (500, 500),    # <500 games → 500 samples
    (1000, 1000),  # <1000 games → 1000 samples
    (2000, 2000),  # <2000 games → 2000 samples
    (5000, 3000),  # <5000 games → 3000 samples
]


@dataclass
class TrainingParams:
    """Training parameters result."""

    epochs: int
    batch_size: int
    lr_multiplier: float

    def to_tuple(self) -> tuple[int, int, float]:
        """Convert to tuple for backward compatibility."""
        return (self.epochs, self.batch_size, self.lr_multiplier)


def get_training_params_for_intensity(
    intensity: str,
    default_epochs: int = 50,
    default_batch_size: int = 512,
) -> tuple[int, int, float]:
    """Map training intensity to (epochs, batch_size, lr_multiplier).

    December 2025: Training intensity controls parameter selection based on
    quality score assessment by FeedbackLoopController:
      - hot_path (quality >= 0.90): Fast iteration, high LR
      - accelerated (quality >= 0.80): Increased training, moderate LR boost
      - normal (quality >= 0.65): Default parameters
      - reduced (quality >= 0.50): More epochs at lower LR for struggling configs
      - paused: Skip training entirely (handled in trigger daemon)

    Args:
        intensity: Training intensity level
        default_epochs: Default epoch count for "normal" intensity
        default_batch_size: Default batch size for "normal" intensity

    Returns:
        Tuple of (epochs, batch_size, learning_rate_multiplier)

    Example:
        >>> get_training_params_for_intensity("hot_path")
        (30, 1024, 1.5)
        >>> get_training_params_for_intensity("normal", default_epochs=100)
        (100, 512, 1.0)
    """
    # Override normal intensity with provided defaults
    intensity_params = INTENSITY_DEFAULTS.copy()
    intensity_params["normal"] = (default_epochs, default_batch_size, 1.0)

    params = intensity_params.get(intensity)
    if params is None:
        logger.warning(
            f"[TrainingDecisionEngine] Unknown intensity '{intensity}', using 'normal'"
        )
        params = intensity_params["normal"]

    return params


def apply_velocity_amplification(
    base_params: tuple[int, int, float],
    elo_velocity: float,
    velocity_trend: str,
) -> tuple[int, int, float]:
    """Apply Elo velocity-based amplification to training parameters.

    January 3, 2026: Sprint 12 P1 improvement - Wire Elo velocity to training
    intensity for dynamic parameter adjustment based on improvement rate.

    Velocity thresholds:
      - velocity > 2.0: Fast improvement → increase epochs/batch for momentum
      - velocity > 1.0: Good progress → slight boost
      - velocity < 0.5: Slow improvement → reduce LR for more careful updates
      - velocity < 0.0: Regression → even lower LR, more epochs

    The velocity_trend ("accelerating", "stable", "decelerating", "plateauing")
    provides secondary signal for fine-tuning.

    Args:
        base_params: (epochs, batch_size, lr_multiplier) from intensity mapping
        elo_velocity: Elo gain per hour (can be negative during regression)
        velocity_trend: Trend indicator from Elo tracking

    Returns:
        Adjusted (epochs, batch_size, lr_multiplier)

    Example:
        >>> apply_velocity_amplification((50, 512, 1.0), 2.5, "accelerating")
        (75, 512, 1.365)  # 1.5x epochs, 1.3x LR + 1.05x trend boost
    """
    epochs, batch_size, lr_mult = base_params

    # Fast improvement: Capitalize on momentum with more aggressive training
    if elo_velocity > 2.0:
        # High velocity: 1.5x epochs, bump batch size to 512+, 1.3x LR
        epochs = int(epochs * 1.5)
        batch_size = max(batch_size, 512)
        lr_mult = lr_mult * 1.3
        logger.debug(
            f"[TrainingDecisionEngine] Velocity amplification (fast): "
            f"velocity={elo_velocity:.2f} → epochs={epochs}, batch={batch_size}, lr_mult={lr_mult:.2f}"
        )

    elif elo_velocity > 1.0:
        # Good velocity: 1.2x epochs, slight batch boost
        epochs = int(epochs * 1.2)
        batch_size = max(batch_size, 384)
        lr_mult = lr_mult * 1.1
        logger.debug(
            f"[TrainingDecisionEngine] Velocity amplification (good): "
            f"velocity={elo_velocity:.2f} → epochs={epochs}, batch={batch_size}, lr_mult={lr_mult:.2f}"
        )

    elif elo_velocity < 0.0:
        # Negative velocity (regression): Very careful training
        # More epochs but lower LR to avoid overcorrection
        epochs = int(epochs * 1.3)
        lr_mult = lr_mult * 0.6
        logger.debug(
            f"[TrainingDecisionEngine] Velocity amplification (regression): "
            f"velocity={elo_velocity:.2f} → epochs={epochs}, lr_mult={lr_mult:.2f}"
        )

    elif elo_velocity < 0.5:
        # Slow improvement: Reduce LR for more careful updates
        lr_mult = lr_mult * 0.8
        logger.debug(
            f"[TrainingDecisionEngine] Velocity amplification (slow): "
            f"velocity={elo_velocity:.2f} → lr_mult={lr_mult:.2f}"
        )

    # Secondary adjustment based on trend
    if velocity_trend == "accelerating" and elo_velocity > 0.5:
        # Accelerating improvement: slight LR boost to maintain momentum
        lr_mult = lr_mult * 1.05
    elif velocity_trend == "plateauing":
        # Plateauing: increase epochs to break through
        epochs = int(epochs * 1.15)
    elif velocity_trend == "decelerating" and elo_velocity > 0.5:
        # Slowing down but still positive: be slightly more conservative
        lr_mult = lr_mult * 0.95

    # Clamp to reasonable bounds
    epochs = max(10, min(epochs, 150))  # 10-150 epochs
    batch_size = max(128, min(batch_size, 2048))  # 128-2048 batch
    lr_mult = max(0.3, min(lr_mult, 2.5))  # 0.3x-2.5x LR multiplier

    return epochs, batch_size, lr_mult


def compute_velocity_adjusted_cooldown(
    base_cooldown_hours: float,
    velocity: float,
    velocity_trend: str,
) -> float:
    """Compute training cooldown adjusted for Elo velocity.

    December 29, 2025: Implements velocity-based cooldown modulation.
    Configs with positive velocity get shorter cooldowns to capitalize on momentum.
    Configs with negative velocity get longer cooldowns to avoid wasteful training.

    Args:
        base_cooldown_hours: Base cooldown between training runs in hours
        velocity: Current Elo velocity (Elo/hour rate of change)
        velocity_trend: Trend string ("accelerating", "stable", "decelerating", "plateauing")

    Returns:
        Adjusted cooldown in seconds

    Example:
        >>> compute_velocity_adjusted_cooldown(0.5, 25.0, "accelerating")
        630.0  # 0.5h * 0.5 (trend) * 0.7 (velocity) * 3600
    """
    base_cooldown_seconds = base_cooldown_hours * 3600

    multiplier = VELOCITY_TREND_COOLDOWN_MULTIPLIERS.get(velocity_trend, 1.0)

    # Additional adjustment based on actual velocity value
    if velocity > 20.0:
        # Very rapid improvement - train even faster
        multiplier *= 0.7
    elif velocity < -10.0:
        # Significant regression - slow down more
        multiplier *= 1.3

    return base_cooldown_seconds * multiplier


def compute_dynamic_sample_threshold(
    config_key: str,
    num_players: int,
    base_threshold: int = 5000,
    game_count: int | None = None,
    dynamic_threshold_getter: Callable[[str], int] | None = None,
) -> int:
    """Compute dynamically adjusted sample threshold for training.

    Phase 5 (Dec 2025): Uses ImprovementOptimizer to adjust thresholds
    based on training success patterns:
    - On promotion streak: Lower threshold → faster iteration
    - Struggling/regression: Higher threshold → more conservative

    Dec 30, 2025: Added player-count-based threshold reduction.
    3p/4p configs get lower thresholds since they generate fewer games.

    Jan 3, 2026: Added game-count-based graduated thresholds for bootstrap.
    Configs with limited training data get lower thresholds to enable faster
    iteration during the bootstrap phase.

    Args:
        config_key: Configuration identifier (e.g., "hex8_2p")
        num_players: Number of players (2, 3, or 4)
        base_threshold: Base minimum sample count (default: 5000)
        game_count: Current game count for bootstrap tier detection (optional)
        dynamic_threshold_getter: Callback to get dynamic threshold from
            ImprovementOptimizer (optional)

    Returns:
        Minimum sample count required to trigger training

    Example:
        >>> compute_dynamic_sample_threshold("hex8_4p", 4, game_count=300)
        500  # Bootstrap tier 1 for <500 games
        >>> compute_dynamic_sample_threshold("hex8_2p", 2, base_threshold=5000)
        5000  # Full threshold for 2p config
    """
    # Jan 3, 2026: Game-count-based graduated thresholds for bootstrap configs
    if game_count is not None:
        for max_games, threshold in BOOTSTRAP_TIERS:
            if game_count < max_games:
                logger.debug(
                    f"[TrainingDecisionEngine] {config_key}: bootstrap tier "
                    f"({game_count} games → {threshold} samples)"
                )
                return threshold

    # Dec 30, 2025: Player-count-based threshold multipliers
    multiplier = PLAYER_COUNT_THRESHOLD_MULTIPLIERS.get(num_players, 1.0)

    # Try to get dynamic threshold from optimizer
    if dynamic_threshold_getter is not None:
        try:
            dynamic_threshold = dynamic_threshold_getter(config_key)
            adjusted_threshold = int(dynamic_threshold * multiplier)
            if adjusted_threshold != base_threshold:
                logger.debug(
                    f"[TrainingDecisionEngine] Dynamic threshold for {config_key}: "
                    f"{adjusted_threshold} (base: {base_threshold}, "
                    f"dynamic: {dynamic_threshold}, multiplier: {multiplier})"
                )
            return adjusted_threshold
        except Exception as e:
            logger.debug(f"[TrainingDecisionEngine] Error getting dynamic threshold: {e}")

    # Fallback: Apply player count multiplier to static base threshold
    adjusted_threshold = int(base_threshold * multiplier)
    if multiplier != 1.0:
        logger.debug(
            f"[TrainingDecisionEngine] Player-adjusted threshold for {config_key}: "
            f"{adjusted_threshold} ({num_players}p config, multiplier: {multiplier})"
        )
    return adjusted_threshold


def check_confidence_early_trigger(
    config_key: str,
    sample_count: int,
    min_samples: int = 1000,
    target_ci_width: float = 0.05,
    confidence_enabled: bool = True,
    variance_getter: Callable[[str], float] | None = None,
) -> tuple[bool, str]:
    """Check if confidence-based early trigger conditions are met.

    Dec 29, 2025: Implements confidence-based training thresholds.
    Allows training to start earlier than min_samples_threshold when
    statistical confidence in training data is high enough.

    The confidence is estimated using the formula for 95% CI width:
        CI_width = 2 * 1.96 * sqrt(variance / n)

    For win rate estimates with variance ~0.25 (binary outcome):
        CI_width ≈ 0.98 / sqrt(n)

    To achieve target_ci_width of 0.05 (±2.5%):
        n = (0.98 / 0.05)^2 ≈ 384 samples

    Args:
        config_key: Configuration identifier
        sample_count: Current number of training samples
        min_samples: Safety floor - never trigger below this (default: 1000)
        target_ci_width: Target 95% CI width (default: 0.05 = ±2.5%)
        confidence_enabled: Whether confidence early trigger is enabled
        variance_getter: Optional callback to get actual variance from quality monitor

    Returns:
        Tuple of (should_trigger, reason)

    Example:
        >>> check_confidence_early_trigger("hex8_2p", 500)
        (True, "confidence threshold met (CI=0.0438)")
        >>> check_confidence_early_trigger("hex8_2p", 100)
        (False, "below safety floor (100 < 1000)")
    """
    if not confidence_enabled:
        return False, "confidence early trigger disabled"

    # Safety floor: never trigger with fewer than min_samples
    if sample_count < min_samples:
        return False, f"below safety floor ({sample_count} < {min_samples})"

    # Estimate confidence interval width
    # For binary outcomes (win/loss), variance is p*(1-p) ≤ 0.25
    # Using 0.25 as conservative estimate
    variance = 0.25
    z_score = 1.96  # 95% confidence

    # Try to get actual variance from quality monitor via callback
    if variance_getter is not None:
        try:
            actual_variance = variance_getter(config_key)
            if actual_variance is not None:
                variance = min(0.25, actual_variance)  # Cap at 0.25
        except Exception:
            pass  # Use default variance if callback fails

    # Calculate CI width: 2 * z * sqrt(variance / n)
    ci_width = 2 * z_score * math.sqrt(variance / sample_count)

    # Log CI width for debugging
    logger.debug(
        f"[TrainingDecisionEngine] {config_key} CI validation: "
        f"CI_width={ci_width:.4f}, variance={variance:.4f}, samples={sample_count}, "
        f"target_CI={target_ci_width:.4f}"
    )

    # Check if confidence is high enough
    if ci_width <= target_ci_width:
        logger.info(
            f"[TrainingDecisionEngine] Confidence early trigger for {config_key}: "
            f"CI_width={ci_width:.4f} ≤ target={target_ci_width:.4f}, "
            f"samples={sample_count}, variance={variance:.4f}"
        )
        return True, f"confidence threshold met (CI={ci_width:.4f})"

    # Calculate samples needed to meet threshold
    samples_needed = int((2 * z_score / target_ci_width) ** 2 * variance)
    logger.debug(
        f"[TrainingDecisionEngine] {config_key} confidence not met: "
        f"CI_width={ci_width:.4f} > target={target_ci_width:.4f}, "
        f"samples_needed≈{samples_needed}"
    )
    return False, f"confidence not met (CI={ci_width:.4f} > target={target_ci_width:.4f})"


def compute_adaptive_max_data_age(
    base_max_age_hours: float,
    velocity_trend: str,
    last_training_time: float,
    current_time: float,
) -> float:
    """Compute adaptive max data age based on velocity trend.

    January 3, 2026: Stalled/plateauing configs should be more lenient on
    data freshness to allow training with whatever data is available and
    break the plateau. Accelerating configs should be stricter to maintain
    training quality.

    Args:
        base_max_age_hours: Base max data age from config
        velocity_trend: Current Elo velocity trend
        last_training_time: Timestamp of last training
        current_time: Current timestamp

    Returns:
        Adaptive max data age in hours

    Example:
        >>> compute_adaptive_max_data_age(4.0, "plateauing", 0, 100000)
        24.0  # 4h * 3.0 (trend) * 2.0 (>48h since training)
    """
    # Trend-based multipliers for data freshness
    # Higher multiplier = more lenient (accepts older data)
    freshness_multipliers = {
        "accelerating": 0.5,    # Stricter - need fresh data for quality
        "stable": 1.0,          # Normal threshold
        "decelerating": 1.5,    # Slightly lenient
        "plateauing": 3.0,      # Very lenient - accept older data to break stall
    }

    multiplier = freshness_multipliers.get(velocity_trend, 1.0)

    # Also consider time since last successful training
    # If config hasn't been trained in a long time, be more lenient
    time_since_training = current_time - last_training_time
    if time_since_training > 172800:  # >48h since last training
        multiplier *= 2.0
    elif time_since_training > 86400:  # >24h since last training
        multiplier *= 1.5

    return base_max_age_hours * multiplier


__all__ = [
    # Core functions
    "get_training_params_for_intensity",
    "apply_velocity_amplification",
    "compute_velocity_adjusted_cooldown",
    "compute_dynamic_sample_threshold",
    "check_confidence_early_trigger",
    "compute_adaptive_max_data_age",
    # Constants
    "INTENSITY_DEFAULTS",
    "VELOCITY_TREND_COOLDOWN_MULTIPLIERS",
    "PLAYER_COUNT_THRESHOLD_MULTIPLIERS",
    "BOOTSTRAP_TIERS",
    # Types
    "TrainingParams",
]
