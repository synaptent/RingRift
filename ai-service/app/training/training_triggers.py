"""Training Triggers - Adapter Layer for Unified Signal Computation.

This module provides the stable API for training decisions.
Internally delegates to UnifiedSignalComputer for actual computation.

The 3 core signals are:
1. Data Freshness: New games available since last training
2. Model Staleness: Time since last training for config
3. Performance Regression: Elo/win rate below acceptable threshold

Usage:
    from app.training.training_triggers import TrainingTriggers, should_train

    triggers = TrainingTriggers(config)

    # Check if training should run
    decision = triggers.should_train("square8_2p", current_state)
    if decision.should_train:
        print(f"Training triggered by: {decision.reason}")

See: app.config.thresholds for canonical threshold constants.
See: app.training.unified_signals for the centralized signal computation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .unified_signals import (
    TrainingUrgency,
    get_signal_computer,
)

# Import event system for quality-aware triggering
try:
    from app.distributed.data_events import (
        DataEvent,
        DataEventType,
        get_event_bus,
    )
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False

logger = logging.getLogger(__name__)

# Import canonical thresholds
try:
    from app.config.thresholds import (
        INITIAL_ELO_RATING,
        MIN_WIN_RATE_PROMOTE,
        TRAINING_BOOTSTRAP_GAMES,
        TRAINING_MIN_INTERVAL_SECONDS,
        TRAINING_STALENESS_HOURS,
        TRAINING_TRIGGER_GAMES,
    )
    DEFAULT_FRESHNESS_THRESHOLD = TRAINING_TRIGGER_GAMES
    DEFAULT_STALENESS_HOURS = TRAINING_STALENESS_HOURS
    DEFAULT_MIN_WIN_RATE = MIN_WIN_RATE_PROMOTE
    DEFAULT_MIN_INTERVAL_MINUTES = TRAINING_MIN_INTERVAL_SECONDS / 60
    DEFAULT_BOOTSTRAP_THRESHOLD = TRAINING_BOOTSTRAP_GAMES
except ImportError:
    INITIAL_ELO_RATING = 1500.0
    DEFAULT_FRESHNESS_THRESHOLD = 500
    DEFAULT_STALENESS_HOURS = 6
    DEFAULT_MIN_WIN_RATE = 0.45
    DEFAULT_MIN_INTERVAL_MINUTES = 20
    DEFAULT_BOOTSTRAP_THRESHOLD = 50


@dataclass
class TriggerConfig:
    """Configuration for training triggers.

    Note: Defaults sourced from app.config.thresholds.
    """
    # Data freshness
    freshness_threshold: int = DEFAULT_FRESHNESS_THRESHOLD
    freshness_weight: float = 1.0

    # Model staleness
    staleness_hours: float = DEFAULT_STALENESS_HOURS
    staleness_weight: float = 0.8

    # Performance regression
    min_win_rate: float = DEFAULT_MIN_WIN_RATE
    regression_weight: float = 1.5  # Higher weight for regression

    # Global constraints
    min_interval_minutes: float = DEFAULT_MIN_INTERVAL_MINUTES
    max_concurrent_training: int = 3

    # Bootstrap (new configs with no models)
    bootstrap_threshold: int = DEFAULT_BOOTSTRAP_THRESHOLD


@dataclass
class TriggerDecision:
    """Result of training trigger evaluation."""
    should_train: bool
    reason: str
    signal_scores: dict[str, float] = field(default_factory=dict)
    config_key: str = ""
    priority: float = 0.0  # Higher = more urgent

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_train": self.should_train,
            "reason": self.reason,
            "signal_scores": self.signal_scores,
            "config_key": self.config_key,
            "priority": self.priority,
        }


@dataclass
class ConfigState:
    """State for a single board/player configuration.

    Note: Uses INITIAL_ELO_RATING from app.config.thresholds as default.
    """
    config_key: str
    games_since_training: int = 0
    last_training_time: float = 0
    last_training_games: int = 0
    model_count: int = 0
    current_elo: float = INITIAL_ELO_RATING
    win_rate: float = 0.5
    win_rate_trend: float = 0.0


@dataclass
class QualityState:
    """Quality state for a configuration.

    Tracks quality information received via events to influence training priority.
    """
    config_key: str
    high_quality_available: bool = False
    high_quality_count: int = 0
    avg_quality_score: float = 0.5
    low_quality_warning: bool = False
    last_quality_update: float = 0.0
    # Priority boost when high quality data is available
    quality_priority_boost: float = 0.0


class TrainingTriggers:
    """Simplified training trigger system with 3 core signals.

    Delegates actual computation to UnifiedSignalComputer for consistency
    across all training decision systems.

    Quality-Aware Features (December 2025):
    - Subscribes to HIGH_QUALITY_DATA_AVAILABLE events
    - Boosts training priority when quality data is ready
    - Tracks quality distribution changes
    """

    def __init__(self, config: TriggerConfig | None = None):
        self.config = config or TriggerConfig()
        self._config_states: dict[str, ConfigState] = {}
        self._last_training_times: dict[str, float] = {}
        # Delegate to unified signal computer
        self._signal_computer = get_signal_computer()
        # Quality state tracking (December 2025)
        self._quality_states: dict[str, QualityState] = {}
        self._event_subscribed = False
        self._quality_callbacks: list[Callable[[str, QualityState], None]] = []

    def get_config_state(self, config_key: str) -> ConfigState:
        """Get or create state for a config."""
        if config_key not in self._config_states:
            self._config_states[config_key] = ConfigState(config_key=config_key)
        return self._config_states[config_key]

    def update_config_state(
        self,
        config_key: str,
        games_count: int | None = None,
        elo: float | None = None,
        win_rate: float | None = None,
        model_count: int | None = None,
    ) -> None:
        """Update state for a config.

        Updates both local state and the unified signal computer.
        """
        state = self.get_config_state(config_key)

        if games_count is not None:
            state.games_since_training = games_count - state.last_training_games

        if elo is not None:
            state.current_elo = elo

        if win_rate is not None:
            old_win_rate = state.win_rate
            state.win_rate = win_rate
            # Simple trend: difference from last update
            state.win_rate_trend = win_rate - old_win_rate

        if model_count is not None:
            state.model_count = model_count

        # Sync with unified signal computer
        self._signal_computer.update_config_state(
            config_key=config_key,
            model_count=model_count,
            current_elo=elo,
            win_rate=win_rate,
        )

    def record_training_complete(
        self,
        config_key: str,
        games_at_training: int,
        new_elo: float | None = None,
    ) -> None:
        """Record that training completed for a config.

        Updates both local state and the unified signal computer.
        """
        state = self.get_config_state(config_key)
        state.last_training_time = time.time()
        state.last_training_games = games_at_training
        state.games_since_training = 0
        self._last_training_times[config_key] = time.time()

        # Sync with unified signal computer
        self._signal_computer.record_training_started(games_at_training, config_key)
        self._signal_computer.record_training_completed(new_elo, config_key)

    def should_train(self, config_key: str, state: ConfigState | None = None) -> TriggerDecision:
        """Evaluate whether training should run for a config.

        Uses UnifiedSignalComputer for consistent signal computation.
        The 3 core signals are:
        1. Data Freshness: Are there enough new games?
        2. Model Staleness: Has it been too long since training?
        3. Performance Regression: Is the model underperforming?

        Quality-aware features (December 2025):
        - Priority is boosted when HIGH_QUALITY_DATA_AVAILABLE is received
        - Priority is reduced when LOW_QUALITY_DATA_WARNING is received

        Returns a TriggerDecision with the result and reasoning.
        """
        if state is None:
            state = self.get_config_state(config_key)

        # Compute current games count from state
        current_games = state.last_training_games + state.games_since_training

        # Delegate to unified signal computer
        signals = self._signal_computer.compute_signals(
            current_games=current_games,
            current_elo=state.current_elo,
            config_key=config_key,
            win_rate=state.win_rate,
            model_count=state.model_count,
        )

        # Build signal scores for backward compatibility
        signal_scores: dict[str, float] = {
            "freshness": signals.games_threshold_ratio,
            "staleness": signals.staleness_ratio,
            "regression": 1.0 if signals.win_rate_regression or signals.elo_regression_detected else 0.0,
        }

        # Add bootstrap indicator
        if signals.is_bootstrap:
            signal_scores["bootstrap"] = 1.0

        # Apply quality-aware priority adjustment (December 2025)
        quality_state = self.get_quality_state(config_key)
        adjusted_priority = signals.priority + quality_state.quality_priority_boost

        # Add quality signal score
        signal_scores["quality"] = quality_state.avg_quality_score
        if quality_state.high_quality_available:
            signal_scores["high_quality_boost"] = quality_state.quality_priority_boost

        return TriggerDecision(
            should_train=signals.should_train,
            reason=signals.reason,
            signal_scores=signal_scores,
            config_key=config_key,
            priority=adjusted_priority,
        )

    def get_training_queue(self) -> list[TriggerDecision]:
        """Get all configs that should train, sorted by priority."""
        decisions = []

        for config_key in self._config_states:
            decision = self.should_train(config_key)
            if decision.should_train:
                decisions.append(decision)

        # Sort by priority (highest first)
        decisions.sort(key=lambda d: d.priority, reverse=True)
        return decisions

    def get_next_training_config(self) -> TriggerDecision | None:
        """Get the highest priority config that should train."""
        queue = self.get_training_queue()
        return queue[0] if queue else None

    def get_urgency(self, config_key: str) -> TrainingUrgency:
        """Get current training urgency level for a config.

        Returns:
            TrainingUrgency enum value
        """
        state = self.get_config_state(config_key)
        current_games = state.last_training_games + state.games_since_training

        signals = self._signal_computer.compute_signals(
            current_games=current_games,
            current_elo=state.current_elo,
            config_key=config_key,
            win_rate=state.win_rate,
            model_count=state.model_count,
        )
        return signals.urgency

    def get_detailed_status(self, config_key: str) -> dict[str, Any]:
        """Get detailed status for logging/debugging.

        Returns a dictionary with all signal details.
        """
        state = self.get_config_state(config_key)
        current_games = state.last_training_games + state.games_since_training

        signals = self._signal_computer.compute_signals(
            current_games=current_games,
            current_elo=state.current_elo,
            config_key=config_key,
            win_rate=state.win_rate,
            model_count=state.model_count,
        )

        # Include quality state
        quality_state = self.get_quality_state(config_key)

        return {
            "should_train": signals.should_train,
            "urgency": signals.urgency.value,
            "reason": signals.reason,
            "games_ratio": signals.games_threshold_ratio,
            "games_since_training": signals.games_since_last_training,
            "games_threshold": signals.games_threshold,
            "elo_trend": signals.elo_trend,
            "time_threshold_met": signals.time_threshold_met,
            "staleness_hours": signals.staleness_hours,
            "win_rate": signals.win_rate,
            "win_rate_regression": signals.win_rate_regression,
            "elo_regression_detected": signals.elo_regression_detected,
            "priority": signals.priority,
            "is_bootstrap": signals.is_bootstrap,
            # Quality-aware fields (December 2025)
            "high_quality_available": quality_state.high_quality_available,
            "high_quality_count": quality_state.high_quality_count,
            "avg_quality_score": quality_state.avg_quality_score,
            "quality_priority_boost": quality_state.quality_priority_boost,
        }

    # =========================================================================
    # Quality-Aware Triggering (December 2025)
    # =========================================================================

    def get_quality_state(self, config_key: str) -> QualityState:
        """Get or create quality state for a config."""
        if config_key not in self._quality_states:
            self._quality_states[config_key] = QualityState(config_key=config_key)
        return self._quality_states[config_key]

    def subscribe_to_quality_events(self) -> bool:
        """Subscribe to quality events from the data event bus.

        Returns True if successfully subscribed.
        """
        if not HAS_EVENT_BUS:
            logger.warning("Event bus not available, quality events disabled")
            return False

        if self._event_subscribed:
            return True

        bus = get_event_bus()

        # Subscribe to quality events
        bus.subscribe(
            DataEventType.HIGH_QUALITY_DATA_AVAILABLE,
            self._handle_high_quality_event,
        )
        bus.subscribe(
            DataEventType.LOW_QUALITY_DATA_WARNING,
            self._handle_low_quality_event,
        )
        bus.subscribe(
            DataEventType.QUALITY_DISTRIBUTION_CHANGED,
            self._handle_quality_distribution_event,
        )

        self._event_subscribed = True
        logger.info("TrainingTriggers subscribed to quality events")
        return True

    def unsubscribe_from_quality_events(self) -> None:
        """Unsubscribe from quality events."""
        if not HAS_EVENT_BUS or not self._event_subscribed:
            return

        bus = get_event_bus()
        bus.unsubscribe(DataEventType.HIGH_QUALITY_DATA_AVAILABLE, self._handle_high_quality_event)
        bus.unsubscribe(DataEventType.LOW_QUALITY_DATA_WARNING, self._handle_low_quality_event)
        bus.unsubscribe(DataEventType.QUALITY_DISTRIBUTION_CHANGED, self._handle_quality_distribution_event)
        self._event_subscribed = False

    def _handle_high_quality_event(self, event: DataEvent) -> None:
        """Handle HIGH_QUALITY_DATA_AVAILABLE event.

        Boosts training priority when high-quality data becomes available.
        """
        config_key = event.payload.get("config", "")
        if not config_key:
            return

        quality_state = self.get_quality_state(config_key)
        quality_state.high_quality_available = True
        quality_state.high_quality_count = event.payload.get("high_quality_count", 0)
        quality_state.avg_quality_score = event.payload.get("avg_quality", 0.7)
        quality_state.last_quality_update = time.time()
        quality_state.low_quality_warning = False

        # Calculate priority boost based on quality and count
        # More high-quality data = higher boost (capped at 0.3)
        count_factor = min(quality_state.high_quality_count / 1000.0, 1.0)
        quality_factor = quality_state.avg_quality_score
        quality_state.quality_priority_boost = 0.3 * count_factor * quality_factor

        logger.info(
            f"Quality event: {config_key} has {quality_state.high_quality_count} "
            f"high-quality games (boost: {quality_state.quality_priority_boost:.3f})"
        )

        # Also update the signal computer's quality score
        self._signal_computer.update_data_quality(config_key, quality_state.avg_quality_score)

        # Notify any registered callbacks
        for callback in self._quality_callbacks:
            try:
                callback(config_key, quality_state)
            except Exception as e:
                logger.warning(f"Quality callback error: {e}")

    def _handle_low_quality_event(self, event: DataEvent) -> None:
        """Handle LOW_QUALITY_DATA_WARNING event.

        Reduces training priority when data quality is poor.
        """
        config_key = event.payload.get("config", "")
        if not config_key:
            return

        quality_state = self.get_quality_state(config_key)
        quality_state.low_quality_warning = True
        quality_state.high_quality_available = False
        quality_state.avg_quality_score = event.payload.get("avg_quality", 0.3)
        quality_state.last_quality_update = time.time()
        # Negative boost when quality is low
        quality_state.quality_priority_boost = -0.1

        logger.warning(
            f"Low quality warning: {config_key} avg_quality={quality_state.avg_quality_score:.3f}"
        )

        self._signal_computer.update_data_quality(config_key, quality_state.avg_quality_score)

    def _handle_quality_distribution_event(self, event: DataEvent) -> None:
        """Handle QUALITY_DISTRIBUTION_CHANGED event.

        Updates quality state when distribution shifts significantly.
        """
        config_key = event.payload.get("config", "")
        if not config_key:
            return

        quality_state = self.get_quality_state(config_key)
        quality_state.avg_quality_score = event.payload.get("avg_quality", 0.5)
        quality_state.high_quality_count = event.payload.get("high_quality_count", 0)
        quality_state.last_quality_update = time.time()

        # Recalculate boost based on new distribution
        if quality_state.avg_quality_score >= 0.6:
            count_factor = min(quality_state.high_quality_count / 1000.0, 1.0)
            quality_state.quality_priority_boost = 0.2 * count_factor * quality_state.avg_quality_score
            quality_state.high_quality_available = True
            quality_state.low_quality_warning = False
        elif quality_state.avg_quality_score < 0.4:
            quality_state.quality_priority_boost = -0.1
            quality_state.high_quality_available = False
            quality_state.low_quality_warning = True
        else:
            quality_state.quality_priority_boost = 0.0
            quality_state.high_quality_available = False
            quality_state.low_quality_warning = False

        self._signal_computer.update_data_quality(config_key, quality_state.avg_quality_score)

    def add_quality_callback(self, callback: Callable[[str, QualityState], None]) -> None:
        """Add a callback to be notified when quality state changes.

        Args:
            callback: Function(config_key, quality_state) called on quality updates
        """
        self._quality_callbacks.append(callback)

    def remove_quality_callback(self, callback: Callable[[str, QualityState], None]) -> bool:
        """Remove a quality callback.

        Returns True if callback was found and removed.
        """
        if callback in self._quality_callbacks:
            self._quality_callbacks.remove(callback)
            return True
        return False

    def get_quality_adjusted_priority(self, config_key: str, base_priority: float) -> float:
        """Get priority adjusted for data quality.

        Args:
            config_key: Configuration identifier
            base_priority: Base priority from signal computation

        Returns:
            Adjusted priority including quality boost
        """
        quality_state = self.get_quality_state(config_key)
        return base_priority + quality_state.quality_priority_boost


# Convenience singleton
_default_triggers: TrainingTriggers | None = None


def get_training_triggers(
    config: TriggerConfig | None = None,
    subscribe_quality_events: bool = True,
) -> TrainingTriggers:
    """Get the default training triggers instance.

    Args:
        config: Optional trigger configuration
        subscribe_quality_events: If True (default), auto-subscribe to quality events
            for priority boosting when high-quality data becomes available.

    Returns:
        TrainingTriggers singleton instance
    """
    global _default_triggers
    if _default_triggers is None:
        _default_triggers = TrainingTriggers(config)
        # Auto-subscribe to quality events (December 2025)
        if subscribe_quality_events and HAS_EVENT_BUS:
            _default_triggers.subscribe_to_quality_events()
    return _default_triggers


def reset_training_triggers() -> None:
    """Reset the singleton for testing purposes."""
    global _default_triggers
    if _default_triggers is not None:
        _default_triggers.unsubscribe_from_quality_events()
    _default_triggers = None


def should_train(config_key: str, games_since_training: int, **kwargs) -> TriggerDecision:
    """Quick check if training should run for a config.

    Args:
        config_key: Config identifier (e.g., "square8_2p")
        games_since_training: Number of new games since last training
        **kwargs: Additional state fields (elo, win_rate, model_count, etc.)

    Returns:
        TriggerDecision with result and reasoning
    """
    triggers = get_training_triggers()

    # Update state with provided info
    triggers.update_config_state(
        config_key,
        games_count=games_since_training + triggers.get_config_state(config_key).last_training_games,
        **kwargs,
    )

    return triggers.should_train(config_key)
