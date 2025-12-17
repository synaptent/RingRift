"""Simplified Training Triggers for Unified Loop.

Consolidates training decision logic into 3 core signals:
1. Data Freshness: New games available since last training
2. Model Staleness: Time since last training for config
3. Performance Regression: Elo/win rate below acceptable threshold

This replaces the complex 8+ signal system with a cleaner, more predictable
approach that's easier to understand and tune.

Usage:
    from app.training.training_triggers import TrainingTriggers, should_train

    triggers = TrainingTriggers(config)

    # Check if training should run
    decision = triggers.should_train("square8_2p", current_state)
    if decision.should_train:
        print(f"Training triggered by: {decision.reason}")
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Constants
AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRESHNESS_THRESHOLD = 500  # New games needed
DEFAULT_STALENESS_HOURS = 6  # Hours before config is "stale"
DEFAULT_MIN_WIN_RATE = 0.45  # Below this triggers urgent training
DEFAULT_MIN_INTERVAL_MINUTES = 20  # Minimum time between training runs


@dataclass
class TriggerConfig:
    """Configuration for training triggers."""
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
    bootstrap_threshold: int = 50  # Very low threshold for configs with 0 models


@dataclass
class TriggerDecision:
    """Result of training trigger evaluation."""
    should_train: bool
    reason: str
    signal_scores: Dict[str, float] = field(default_factory=dict)
    config_key: str = ""
    priority: float = 0.0  # Higher = more urgent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_train": self.should_train,
            "reason": self.reason,
            "signal_scores": self.signal_scores,
            "config_key": self.config_key,
            "priority": self.priority,
        }


@dataclass
class ConfigState:
    """State for a single board/player configuration."""
    config_key: str
    games_since_training: int = 0
    last_training_time: float = 0
    last_training_games: int = 0
    model_count: int = 0
    current_elo: float = 1500.0
    win_rate: float = 0.5
    win_rate_trend: float = 0.0


class TrainingTriggers:
    """Simplified training trigger system with 3 core signals."""

    def __init__(self, config: Optional[TriggerConfig] = None):
        self.config = config or TriggerConfig()
        self._config_states: Dict[str, ConfigState] = {}
        self._last_training_times: Dict[str, float] = {}

    def get_config_state(self, config_key: str) -> ConfigState:
        """Get or create state for a config."""
        if config_key not in self._config_states:
            self._config_states[config_key] = ConfigState(config_key=config_key)
        return self._config_states[config_key]

    def update_config_state(
        self,
        config_key: str,
        games_count: Optional[int] = None,
        elo: Optional[float] = None,
        win_rate: Optional[float] = None,
        model_count: Optional[int] = None,
    ) -> None:
        """Update state for a config."""
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

    def record_training_complete(self, config_key: str, games_at_training: int) -> None:
        """Record that training completed for a config."""
        state = self.get_config_state(config_key)
        state.last_training_time = time.time()
        state.last_training_games = games_at_training
        state.games_since_training = 0
        self._last_training_times[config_key] = time.time()

    def should_train(self, config_key: str, state: Optional[ConfigState] = None) -> TriggerDecision:
        """Evaluate whether training should run for a config.

        Uses 3 core signals:
        1. Data Freshness: Are there enough new games?
        2. Model Staleness: Has it been too long since training?
        3. Performance Regression: Is the model underperforming?

        Returns a TriggerDecision with the result and reasoning.
        """
        if state is None:
            state = self.get_config_state(config_key)

        now = time.time()
        cfg = self.config
        signal_scores: Dict[str, float] = {}

        # Check minimum interval constraint
        last_training = self._last_training_times.get(config_key, 0)
        minutes_since_training = (now - last_training) / 60
        if minutes_since_training < cfg.min_interval_minutes:
            return TriggerDecision(
                should_train=False,
                reason=f"Too soon since last training ({minutes_since_training:.0f}m < {cfg.min_interval_minutes}m)",
                signal_scores={},
                config_key=config_key,
                priority=0,
            )

        # Bootstrap check: new configs with no models get priority
        if state.model_count == 0 and state.games_since_training >= cfg.bootstrap_threshold:
            return TriggerDecision(
                should_train=True,
                reason=f"Bootstrap: config has 0 models and {state.games_since_training} games",
                signal_scores={"bootstrap": 1.0},
                config_key=config_key,
                priority=10.0,  # Highest priority for bootstrap
            )

        # Signal 1: Data Freshness
        freshness_score = min(1.0, state.games_since_training / cfg.freshness_threshold)
        signal_scores["freshness"] = freshness_score

        # Signal 2: Model Staleness
        hours_since_training = (now - state.last_training_time) / 3600
        staleness_score = min(1.0, hours_since_training / cfg.staleness_hours)
        signal_scores["staleness"] = staleness_score

        # Signal 3: Performance Regression
        regression_score = 0.0
        if state.win_rate < cfg.min_win_rate:
            # Strong regression signal
            regression_score = 1.0 - (state.win_rate / cfg.min_win_rate)
        elif state.win_rate_trend < -0.05:
            # Declining performance (not yet critical)
            regression_score = min(0.5, abs(state.win_rate_trend) * 5)
        signal_scores["regression"] = regression_score

        # Compute weighted priority score
        priority = (
            freshness_score * cfg.freshness_weight +
            staleness_score * cfg.staleness_weight +
            regression_score * cfg.regression_weight
        )

        # Determine if we should train
        # Training triggers if:
        # 1. Enough new data (freshness >= 1.0), OR
        # 2. Model is stale AND has some new data, OR
        # 3. Performance regression detected

        should_train = False
        reason = ""

        if freshness_score >= 1.0:
            should_train = True
            reason = f"Data freshness: {state.games_since_training} new games"
        elif staleness_score >= 1.0 and freshness_score >= 0.3:
            should_train = True
            reason = f"Model staleness: {hours_since_training:.1f}h since training"
        elif regression_score >= 0.5:
            should_train = True
            reason = f"Performance regression: win_rate={state.win_rate:.1%}"

        return TriggerDecision(
            should_train=should_train,
            reason=reason,
            signal_scores=signal_scores,
            config_key=config_key,
            priority=priority,
        )

    def get_training_queue(self) -> List[TriggerDecision]:
        """Get all configs that should train, sorted by priority."""
        decisions = []

        for config_key in self._config_states:
            decision = self.should_train(config_key)
            if decision.should_train:
                decisions.append(decision)

        # Sort by priority (highest first)
        decisions.sort(key=lambda d: d.priority, reverse=True)
        return decisions

    def get_next_training_config(self) -> Optional[TriggerDecision]:
        """Get the highest priority config that should train."""
        queue = self.get_training_queue()
        return queue[0] if queue else None


# Convenience singleton
_default_triggers: Optional[TrainingTriggers] = None


def get_training_triggers(config: Optional[TriggerConfig] = None) -> TrainingTriggers:
    """Get the default training triggers instance."""
    global _default_triggers
    if _default_triggers is None:
        _default_triggers = TrainingTriggers(config)
    return _default_triggers


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
