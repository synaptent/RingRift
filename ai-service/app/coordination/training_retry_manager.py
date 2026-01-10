"""Retry Management for Training Trigger Daemon.

This module contains the retry queue management logic extracted from
TrainingTriggerDaemon to reduce file size and improve maintainability.

January 2026: Extracted from training_trigger_daemon.py as part of
modularization effort.

Usage:
    from app.coordination.training_retry_manager import (
        RetryQueueConfig,
        RetryQueueManager,
        get_velocity_adjusted_cooldown,
        get_adaptive_max_data_age,
    )

    # Create retry manager
    config = RetryQueueConfig(max_attempts=3, base_delay=300.0)
    manager = RetryQueueManager(config)

    # Queue a retry
    queued = manager.queue_training_retry(
        "hex8_2p", "hex8", 2, "GPU OOM error"
    )

    # Process pending retries
    ready_items = manager.get_ready_retries()

    # Get velocity-adjusted parameters
    cooldown = get_velocity_adjusted_cooldown(state.elo_velocity_trend, state.elo_velocity, 1.0)
    max_age = get_adaptive_max_data_age(config_key, "plateauing", time_since_training=90000)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.coordination.training_trigger_types import ConfigTrainingState

logger = logging.getLogger(__name__)


# Type alias for retry queue items
# (config_key, board_type, num_players, attempts, next_retry_time, error)
RetryQueueItem = tuple[str, str, int, int, float, str]


@dataclass
class RetryQueueConfig:
    """Configuration for training retry queue management.

    Attributes:
        max_attempts: Maximum retry attempts before giving up
        base_delay: Base delay in seconds between retries
        max_delay: Maximum delay in seconds (for exponential backoff cap)
        jitter: Jitter factor (0.0-1.0) to randomize delays
        max_queue_size: Maximum number of items in retry queue
    """

    max_attempts: int = 3
    base_delay: float = 300.0  # 5 minutes
    max_delay: float = 1200.0  # 20 minutes
    jitter: float = 0.1
    max_queue_size: int = 100

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff.

        Args:
            attempt: Current attempt number (1-based)

        Returns:
            Delay in seconds with jitter applied
        """
        import random

        # Exponential backoff: base * 2^(attempt-1)
        delay = self.base_delay * (2 ** (attempt - 1))
        delay = min(delay, self.max_delay)

        # Apply jitter
        if self.jitter > 0:
            jitter_amount = delay * self.jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(delay, 0.0)


@dataclass
class RetryStats:
    """Statistics for retry queue operations."""

    retries_queued: int = 0
    retries_succeeded: int = 0
    retries_exhausted: int = 0


class RetryQueueManager:
    """Manages the training retry queue with exponential backoff.

    Handles queuing failed training jobs and tracking when they're
    ready for retry based on configurable delays.
    """

    def __init__(self, config: RetryQueueConfig | None = None):
        """Initialize the retry queue manager.

        Args:
            config: Retry configuration. Uses defaults if not provided.
        """
        self.config = config or RetryQueueConfig()
        self._queue: deque[RetryQueueItem] = deque(maxlen=self.config.max_queue_size)
        self._stats = RetryStats()

    @property
    def stats(self) -> dict[str, int]:
        """Get retry statistics as a dictionary."""
        return {
            "retries_queued": self._stats.retries_queued,
            "retries_succeeded": self._stats.retries_succeeded,
            "retries_exhausted": self._stats.retries_exhausted,
        }

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    def record_success(self) -> None:
        """Record a successful retry."""
        self._stats.retries_succeeded += 1

    def queue_training_retry(
        self,
        config_key: str,
        board_type: str,
        num_players: int,
        error: str,
        current_attempts: int = 0,
    ) -> bool:
        """Queue failed training for retry with exponential backoff.

        December 29, 2025 (Phase 3): Implements automatic retry for transient failures.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            board_type: Board type for the training
            num_players: Number of players
            error: Failure reason (for logging)
            current_attempts: Number of attempts already made

        Returns:
            True if queued for retry, False if max attempts exceeded.
        """
        # Check existing retries for this config
        existing_attempts = 0
        for item in self._queue:
            if item[0] == config_key:
                existing_attempts = max(existing_attempts, item[3])

        attempts = max(current_attempts, existing_attempts) + 1

        if attempts > self.config.max_attempts:
            self._stats.retries_exhausted += 1
            logger.error(
                f"[RetryQueueManager] Max retries ({self.config.max_attempts}) exceeded "
                f"for {config_key}: {error[:100]}"
            )
            return False

        delay = self.config.get_delay(attempts)
        next_retry = time.time() + delay

        self._queue.append(
            (config_key, board_type, num_players, attempts, next_retry, error[:200])
        )
        self._stats.retries_queued += 1

        logger.info(
            f"[RetryQueueManager] Queued training retry #{attempts} for {config_key} "
            f"in {delay / 60:.0f}min (reason: {error[:50]}...)"
        )
        return True

    def get_ready_retries(self) -> list[tuple[str, str, int, int, str]]:
        """Get retry items that are ready for processing.

        Returns items whose delay has elapsed and removes them from the queue.

        Returns:
            List of (config_key, board_type, num_players, attempts, error) tuples
            for items ready for retry.
        """
        if not self._queue:
            return []

        now = time.time()
        ready: list[tuple[str, str, int, int, str]] = []
        remaining: list[RetryQueueItem] = []

        while self._queue:
            item = self._queue.popleft()
            config_key, board_type, num_players, attempts, next_retry_time, error = item

            if next_retry_time <= now:
                ready.append((config_key, board_type, num_players, attempts, error))
            else:
                remaining.append(item)

        # Put back items not yet ready
        for item in remaining:
            self._queue.append(item)

        return ready

    def requeue_with_short_delay(
        self,
        config_key: str,
        board_type: str,
        num_players: int,
        attempts: int,
        error: str,
        delay_seconds: float = 60.0,
    ) -> None:
        """Re-queue an item with a short delay (e.g., when training is in progress).

        Args:
            config_key: Configuration key
            board_type: Board type
            num_players: Number of players
            attempts: Current attempt count (preserved)
            error: Error message
            delay_seconds: Delay before next retry attempt
        """
        next_retry = time.time() + delay_seconds
        self._queue.append(
            (config_key, board_type, num_players, attempts, next_retry, error)
        )

    def get_pending_for_config(self, config_key: str) -> list[RetryQueueItem]:
        """Get all pending retries for a specific config.

        Args:
            config_key: Configuration key to look up

        Returns:
            List of pending retry items for the config
        """
        return [item for item in self._queue if item[0] == config_key]


def get_velocity_adjusted_cooldown(
    elo_velocity_trend: str,
    elo_velocity: float,
    base_cooldown_hours: float,
) -> float:
    """Get training cooldown adjusted for Elo velocity.

    December 29, 2025: Implements velocity-based cooldown modulation.
    Configs with positive velocity get shorter cooldowns to capitalize on momentum.
    Configs with negative velocity get longer cooldowns to avoid wasteful training.

    Args:
        elo_velocity_trend: Velocity trend ("accelerating", "stable", "decelerating", "plateauing")
        elo_velocity: Elo velocity value (Elo/hour)
        base_cooldown_hours: Base cooldown in hours

    Returns:
        Adjusted cooldown in seconds
    """
    base_cooldown = base_cooldown_hours * 3600

    # Velocity-based multipliers
    velocity_multipliers = {
        "accelerating": 0.5,  # 50% cooldown - train faster
        "stable": 1.0,  # Normal cooldown
        "decelerating": 1.5,  # 150% cooldown - train slower
        # January 3, 2026: Fixed plateauing multiplier - should train MORE aggressively
        # to break the plateau, not slower. This was backwards before.
        "plateauing": 0.6,  # 60% cooldown - train faster to break plateau
    }

    multiplier = velocity_multipliers.get(elo_velocity_trend, 1.0)

    # Additional adjustment based on actual velocity value
    if elo_velocity > 20.0:
        # Very rapid improvement - train even faster
        multiplier *= 0.7
    elif elo_velocity < -10.0:
        # Significant regression - slow down more
        multiplier *= 1.3

    return base_cooldown * multiplier


def get_adaptive_max_data_age(
    config_key: str,
    elo_velocity_trend: str,
    base_max_age_hours: float,
    time_since_training: float = 0.0,
    game_count: int | None = None,
) -> float:
    """Get adaptive max data age based on velocity trend (January 3, 2026).

    Stalled/plateauing configs should be more lenient on data freshness
    to allow training with whatever data is available and break the plateau.
    Accelerating configs should be stricter to maintain training quality.

    January 5, 2026 (Session 17.27): Added game count-based freshness for
    starved configs (< 500 games). These configs use 168h (1 week) threshold
    to ensure training can proceed with any available data.

    Args:
        config_key: Configuration key for logging
        elo_velocity_trend: Velocity trend indicator
        base_max_age_hours: Base max data age in hours
        time_since_training: Time since last training in seconds
        game_count: Optional game count for starved config detection

    Returns:
        Adaptive max data age in hours
    """
    # January 5, 2026: Game count-based freshness for starved configs
    if game_count is not None and game_count < 500:
        # Starved config: use 168h (1 week) threshold
        logger.debug(
            f"[RetryManager] {config_key}: starved config "
            f"({game_count} games < 500), using 168h freshness threshold"
        )
        return 168.0  # 1 week for starved configs

    # Trend-based multipliers for data freshness
    # Higher multiplier = more lenient (accepts older data)
    freshness_multipliers = {
        "accelerating": 0.5,  # Stricter - need fresh data for quality
        "stable": 1.0,  # Normal threshold
        "decelerating": 1.5,  # Slightly lenient
        "plateauing": 3.0,  # Very lenient - accept older data to break stall
    }

    multiplier = freshness_multipliers.get(elo_velocity_trend, 1.0)

    # Also consider time since last successful training
    # If config hasn't been trained in a long time, be more lenient
    if time_since_training > 86400:  # >24h since last training
        multiplier *= 1.5
    elif time_since_training > 172800:  # >48h since last training
        multiplier *= 2.0

    adaptive_age = base_max_age_hours * multiplier
    logger.debug(
        f"[RetryManager] {config_key}: adaptive_max_data_age="
        f"{adaptive_age:.1f}h (base={base_max_age_hours:.1f}h, "
        f"trend={elo_velocity_trend}, mult={multiplier:.2f})"
    )

    return adaptive_age


def get_velocity_adjusted_cooldown_from_state(
    state: ConfigTrainingState,
    base_cooldown_hours: float,
) -> float:
    """Convenience function to get cooldown from ConfigTrainingState.

    Args:
        state: Training state for the config
        base_cooldown_hours: Base cooldown in hours

    Returns:
        Adjusted cooldown in seconds
    """
    return get_velocity_adjusted_cooldown(
        state.elo_velocity_trend,
        state.elo_velocity,
        base_cooldown_hours,
    )


def get_adaptive_max_data_age_from_state(
    state: ConfigTrainingState,
    base_max_age_hours: float,
) -> float:
    """Convenience function to get adaptive max data age from ConfigTrainingState.

    Args:
        state: Training state for the config
        base_max_age_hours: Base max data age in hours

    Returns:
        Adaptive max data age in hours
    """
    import time

    time_since_training = time.time() - state.last_training_time

    # Try to get game count if available
    game_count = None
    try:
        from app.coordination.event_utils import parse_config_key
        from app.utils.game_discovery import count_games_for_config

        parsed = parse_config_key(state.config_key)
        if parsed:
            game_count = count_games_for_config(parsed.board_type, parsed.num_players)
    except (ImportError, ValueError, OSError):
        pass

    return get_adaptive_max_data_age(
        config_key=state.config_key,
        elo_velocity_trend=state.elo_velocity_trend,
        base_max_age_hours=base_max_age_hours,
        time_since_training=time_since_training,
        game_count=game_count,
    )
