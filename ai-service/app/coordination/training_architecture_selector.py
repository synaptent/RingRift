"""Architecture Selection for Training Trigger Daemon.

This module contains the architecture selection logic extracted from
TrainingTriggerDaemon to reduce file size and improve maintainability.

January 2026: Extracted from training_trigger_daemon.py as part of
modularization effort.

Usage:
    from app.coordination.training_architecture_selector import (
        get_training_params_for_intensity,
        select_architecture_for_training,
        apply_velocity_amplification,
    )

    # Get training params based on intensity
    epochs, batch_size, lr_mult = get_training_params_for_intensity("normal")

    # Select architecture based on Elo performance
    arch = select_architecture_for_training("hex8", 2)

    # Apply velocity-based amplification
    epochs, batch_size, lr_mult = apply_velocity_amplification(
        base_params=(30, 512, 1.0),
        elo_velocity=1.5,
        velocity_trend="accelerating",
    )
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.coordination.training_trigger_types import TrainingTriggerConfig

logger = logging.getLogger(__name__)


# Default training parameters
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 512


def get_training_params_for_intensity(
    intensity: str,
    default_epochs: int = DEFAULT_EPOCHS,
    default_batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[int, int, float]:
    """Map training intensity to (epochs, batch_size, lr_multiplier).

    The FeedbackLoopController sets intensity based on quality score:
      - hot_path (quality >= 0.90): Fast iteration, high LR
      - accelerated (quality >= 0.80): Increased training, moderate LR boost
      - normal (quality >= 0.65): Default parameters
      - reduced (quality >= 0.50): More epochs at lower LR for struggling configs
      - paused: Skip training entirely (handled in _maybe_trigger_training)

    Args:
        intensity: Training intensity level ("hot_path", "accelerated", "normal",
                   "reduced", "paused")
        default_epochs: Default epoch count for "normal" intensity
        default_batch_size: Default batch size for "normal" intensity

    Returns:
        Tuple of (epochs, batch_size, learning_rate_multiplier)
    """
    intensity_params = {
        # hot_path: Fast iteration with larger batches, higher LR
        "hot_path": (30, 1024, 1.5),
        # accelerated: More aggressive training
        "accelerated": (40, 768, 1.2),
        # normal: Default parameters
        "normal": (default_epochs, default_batch_size, 1.0),
        # reduced: Slower, more careful training for struggling configs
        "reduced": (60, 256, 0.8),
        # paused: Should not reach here, but use minimal params
        "paused": (10, 128, 0.5),
    }

    params = intensity_params.get(intensity)
    if params is None:
        logger.warning(
            f"[ArchitectureSelector] Unknown intensity '{intensity}', using 'normal'"
        )
        params = intensity_params["normal"]

    return params


def select_architecture_for_training(
    board_type: str,
    num_players: int,
    temperature: float = 0.5,
    default_arch: str = "v5",
) -> str:
    """Select architecture version for training based on Elo performance.

    Uses ArchitectureTracker's compute_allocation_weights() to select
    architectures biased toward better Elo performance.

    Args:
        board_type: Board type (hex8, square8, etc.)
        num_players: Number of players (2, 3, or 4)
        temperature: Balance between exploration and exploitation (0-1)
        default_arch: Fallback architecture if tracker unavailable

    Returns:
        Architecture version string (e.g., "v5", "v2", "v5-heavy").
        Falls back to default_arch if tracker unavailable or no weights exist.
    """
    try:
        from app.training.architecture_tracker import get_allocation_weights
    except ImportError:
        logger.debug(
            f"[ArchitectureSelector] ArchitectureTracker not available, "
            f"using default: {default_arch}"
        )
        return default_arch

    try:
        weights = get_allocation_weights(
            board_type=board_type,
            num_players=num_players,
            temperature=temperature,
        )

        if not weights:
            logger.debug(
                f"[ArchitectureSelector] No architecture weights for "
                f"{board_type}_{num_players}p, using default: {default_arch}"
            )
            return default_arch

        # Weighted random selection based on Elo performance
        architectures = list(weights.keys())
        arch_weights = list(weights.values())
        selected_arch = random.choices(architectures, weights=arch_weights, k=1)[0]

        logger.info(
            f"[ArchitectureSelector] Architecture selection for "
            f"{board_type}_{num_players}p: {selected_arch} (weights: {weights})"
        )
        return selected_arch

    except (KeyError, ValueError, TypeError) as e:
        logger.debug(
            f"[ArchitectureSelector] Error selecting architecture for "
            f"{board_type}_{num_players}p: {e}, using default: {default_arch}"
        )
        return default_arch


def apply_velocity_amplification(
    base_params: tuple[int, int, float],
    elo_velocity: float,
    velocity_trend: str,
) -> tuple[int, int, float]:
    """Apply Elo velocity-based amplification to training parameters.

    Velocity thresholds:
      - velocity > 2.0: Fast improvement -> increase epochs/batch for momentum
      - velocity > 1.0: Good progress -> slight boost
      - velocity < 0.5: Slow improvement -> reduce LR for more careful updates
      - velocity < 0.0: Regression -> even lower LR, more epochs

    The velocity_trend ("accelerating", "stable", "decelerating", "plateauing")
    provides secondary signal for fine-tuning.

    Args:
        base_params: (epochs, batch_size, lr_multiplier) from intensity mapping
        elo_velocity: Elo gain per hour (can be negative during regression)
        velocity_trend: Trend indicator from Elo tracking

    Returns:
        Adjusted (epochs, batch_size, lr_multiplier)
    """
    epochs, batch_size, lr_mult = base_params

    # Fast improvement: Capitalize on momentum with more aggressive training
    if elo_velocity > 2.0:
        # High velocity: 1.5x epochs, bump batch size to 512+, 1.3x LR
        epochs = int(epochs * 1.5)
        batch_size = max(batch_size, 512)
        lr_mult = lr_mult * 1.3
        logger.debug(
            f"[ArchitectureSelector] Velocity amplification (fast): "
            f"velocity={elo_velocity:.2f} -> epochs={epochs}, batch={batch_size}, lr_mult={lr_mult:.2f}"
        )

    elif elo_velocity > 1.0:
        # Good velocity: 1.2x epochs, slight batch boost
        epochs = int(epochs * 1.2)
        batch_size = max(batch_size, 384)
        lr_mult = lr_mult * 1.1
        logger.debug(
            f"[ArchitectureSelector] Velocity amplification (good): "
            f"velocity={elo_velocity:.2f} -> epochs={epochs}, batch={batch_size}, lr_mult={lr_mult:.2f}"
        )

    elif elo_velocity < 0.0:
        # Negative velocity (regression): Very careful training
        # More epochs but lower LR to avoid overcorrection
        epochs = int(epochs * 1.3)
        lr_mult = lr_mult * 0.6
        logger.debug(
            f"[ArchitectureSelector] Velocity amplification (regression): "
            f"velocity={elo_velocity:.2f} -> epochs={epochs}, lr_mult={lr_mult:.2f}"
        )

    elif elo_velocity < 0.5:
        # Slow improvement: Reduce LR for more careful updates
        lr_mult = lr_mult * 0.8
        logger.debug(
            f"[ArchitectureSelector] Velocity amplification (slow): "
            f"velocity={elo_velocity:.2f} -> lr_mult={lr_mult:.2f}"
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
