"""Canonical feedback state definitions for training pipeline coordination.

This module provides the single source of truth for FeedbackState dataclasses.
Consolidates 5 duplicate definitions from:
- app/coordination/feedback_loop_controller.py
- app/coordination/unified_feedback.py
- app/coordination/feedback_signals.py
- app/integration/pipeline_feedback.py
- scripts/unified_loop/config.py

Created: December 28, 2025
Migration: Use these classes instead of module-local FeedbackState definitions.

Classes:
    CanonicalFeedbackState: Base class with core metrics (22 fields)
    SignalFeedbackState: Extended for unified orchestrator (+5 fields)
    MonitoringFeedbackState: Extended for monitoring/decisions (+9 fields, +4 methods)

Usage:
    from app.coordination.feedback_state import (
        CanonicalFeedbackState,
        SignalFeedbackState,
        MonitoringFeedbackState,
    )

    # For simple per-config tracking
    state = CanonicalFeedbackState(config_key="hex8_2p")

    # For unified feedback orchestration
    state = SignalFeedbackState(config_key="hex8_2p")
    state.training_intensity = "hot_path"

    # For monitoring and urgency computation
    state = MonitoringFeedbackState(config_key="hex8_2p")
    state.update_elo(1650.0)
    urgency = state.compute_urgency()
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CanonicalFeedbackState:
    """Base feedback state - core metrics tracked across all feedback loops.

    This is the canonical definition. All other FeedbackState classes should
    either inherit from this or import it directly.

    Attributes:
        config_key: Configuration identifier (e.g., "hex8_2p", "square8_4p")

        Quality Metrics (inputs to feedback computation):
            quality_score: Selfplay data quality (0-1)
            training_accuracy: Policy prediction accuracy (0-1)
            win_rate: Latest evaluation win rate (0-1)

        Elo Rating (primary performance metric):
            elo_current: Current Elo rating
            elo_velocity: Rating change per hour
            elo_history: Recent (timestamp, elo) tuples for trend analysis

        Status Tracking:
            consecutive_successes: Promotion success streak
            consecutive_failures: Promotion failure streak

        Data Quality:
            parity_failure_rate: Rolling average of parity check failures (0-1)
            data_quality_score: Composite data quality metric (0-1)

        Curriculum:
            curriculum_weight: Training priority multiplier (0.5-2.0)
            curriculum_last_update: Last curriculum adjustment timestamp

        Timing:
            last_selfplay_time: Last selfplay completion timestamp
            last_training_time: Last training completion timestamp
            last_evaluation_time: Last evaluation completion timestamp
            last_promotion_time: Last model promotion timestamp
    """

    config_key: str

    # Quality metrics (input to feedback computation)
    quality_score: float = 0.0
    training_accuracy: float = 0.0
    win_rate: float = 0.5

    # Elo rating (primary performance metric)
    elo_current: float = 1500.0
    elo_velocity: float = 0.0
    elo_history: deque = field(default_factory=lambda: deque(maxlen=20))

    # Status tracking (counters for consecutive events)
    consecutive_successes: int = 0
    consecutive_failures: int = 0

    # Data quality
    parity_failure_rate: float = 0.0
    data_quality_score: float = 1.0

    # Curriculum (training priority)
    curriculum_weight: float = 1.0
    curriculum_last_update: float = 0.0

    # Timing (all event timestamps)
    last_selfplay_time: float = 0.0
    last_training_time: float = 0.0
    last_evaluation_time: float = 0.0
    last_promotion_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "config_key": self.config_key,
            "quality_score": self.quality_score,
            "training_accuracy": self.training_accuracy,
            "win_rate": self.win_rate,
            "elo_current": self.elo_current,
            "elo_velocity": self.elo_velocity,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "parity_failure_rate": self.parity_failure_rate,
            "data_quality_score": self.data_quality_score,
            "curriculum_weight": self.curriculum_weight,
            "last_selfplay_time": self.last_selfplay_time,
            "last_training_time": self.last_training_time,
            "last_evaluation_time": self.last_evaluation_time,
            "last_promotion_time": self.last_promotion_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CanonicalFeedbackState:
        """Deserialize from dict."""
        return cls(
            config_key=data["config_key"],
            quality_score=data.get("quality_score", 0.0),
            training_accuracy=data.get("training_accuracy", 0.0),
            win_rate=data.get("win_rate", 0.5),
            elo_current=data.get("elo_current", 1500.0),
            elo_velocity=data.get("elo_velocity", 0.0),
            consecutive_successes=data.get("consecutive_successes", 0),
            consecutive_failures=data.get("consecutive_failures", 0),
            parity_failure_rate=data.get("parity_failure_rate", 0.0),
            data_quality_score=data.get("data_quality_score", 1.0),
            curriculum_weight=data.get("curriculum_weight", 1.0),
        )


@dataclass
class SignalFeedbackState(CanonicalFeedbackState):
    """Extended feedback state for unified orchestrator - includes feedback signals.

    Used by: app/coordination/unified_feedback.py (UnifiedFeedbackOrchestrator)

    Adds computed feedback signals and signal-specific state for dynamic
    training parameter adjustment.

    Additional Attributes:
        training_intensity: Current training intensity level
            - "paused": No training
            - "reduced": Reduced epochs/batch
            - "normal": Standard training
            - "accelerated": Increased frequency
            - "hot_path": Maximum urgency

        exploration_boost: Exploration multiplier (1.0 = normal, >1.0 = more exploration)
        data_freshness_hours: Hours since last data update (inf = unknown)
        consecutive_anomalies: Training loss anomaly count
        quality_penalties_applied: Curriculum quality penalty count
        last_adjustment_time: Last signal adjustment timestamp (for cooldowns)
    """

    # Feedback signals (computed from metrics)
    training_intensity: str = "normal"
    exploration_boost: float = 1.0
    data_freshness_hours: float = float("inf")

    # Signal-specific state
    consecutive_anomalies: int = 0
    quality_penalties_applied: int = 0
    last_adjustment_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict including signal fields."""
        base = super().to_dict()
        base.update(
            {
                "training_intensity": self.training_intensity,
                "exploration_boost": self.exploration_boost,
                "data_freshness_hours": self.data_freshness_hours,
                "consecutive_anomalies": self.consecutive_anomalies,
                "quality_penalties_applied": self.quality_penalties_applied,
                "last_adjustment_time": self.last_adjustment_time,
            }
        )
        return base


@dataclass
class MonitoringFeedbackState(CanonicalFeedbackState):
    """Extended feedback state for monitoring and decision-making.

    Used by: scripts/unified_loop/config.py (ConfigState.feedback field)

    Adds extended tracking for Elo trends, win rate patterns, and urgency
    computation for training prioritization decisions.

    Additional Attributes:
        Elo Tracking:
            elo_trend: Change from last evaluation (+ = improving)
            elo_peak: Historical peak rating
            elo_plateau_count: Consecutive evals without significant gain

        Win Rate Tracking:
            win_rate_trend: Change from last evaluation
            consecutive_high_win_rate: Streak above 70%
            consecutive_low_win_rate: Streak below 50%

        Quality Metrics:
            parity_checks_total: Total parity checks performed

        Urgency:
            urgency_score: Composite urgency (0-1, higher = more urgent)
            last_urgency_update: Last urgency computation timestamp

    Methods:
        update_parity(): Update rolling parity failure rate
        update_elo(): Update Elo with trend and plateau detection
        update_win_rate(): Update win rate with trend tracking
        compute_urgency(): Compute composite urgency score
        compute_data_quality(): Compute data quality from factors
        is_data_quality_acceptable(): Check if quality meets threshold
    """

    # Extended Elo tracking
    elo_trend: float = 0.0
    elo_peak: float = 1500.0
    elo_plateau_count: int = 0

    # Win rate tracking
    win_rate_trend: float = 0.0
    consecutive_high_win_rate: int = 0
    consecutive_low_win_rate: int = 0

    # Extended quality metrics
    parity_checks_total: int = 0

    # Computed urgency for training prioritization
    urgency_score: float = 0.0
    last_urgency_update: float = 0.0

    # Promotion tracking (Dec 29, 2025 - migrated from feedback_loop_controller)
    last_promotion_success: bool | None = None

    # Engine bandit tracking (Dec 29, 2025 - migrated from feedback_loop_controller)
    last_selfplay_engine: str = "gumbel-mcts"
    last_selfplay_games: int = 0
    elo_before_training: float = 1500.0

    # Curriculum tier tracking (Dec 29, 2025 - migrated from feedback_loop_controller)
    curriculum_tier: int = 0
    curriculum_last_advanced: float = 0.0

    # Work queue metrics (Dec 29, 2025 - migrated from feedback_loop_controller)
    work_completed_count: int = 0
    last_work_completion_time: float = 0.0

    # Feedback signals (Dec 29, 2025 - migrated from feedback_loop_controller)
    training_intensity: str = "normal"  # normal, accelerated, hot_path, reduced
    exploration_boost: float = 1.0  # 1.0 = normal, >1.0 = more exploration
    search_budget: int = 400  # Gumbel MCTS budget

    def update_parity(self, passed: bool, alpha: float = 0.1) -> None:
        """Update rolling parity failure rate.

        Args:
            passed: Whether the parity check passed
            alpha: Exponential moving average weight (0-1)
        """
        result = 0.0 if passed else 1.0
        self.parity_failure_rate = alpha * result + (1 - alpha) * self.parity_failure_rate
        self.parity_checks_total += 1

    def update_elo(self, new_elo: float, plateau_threshold: float = 15.0) -> None:
        """Update Elo with trend and plateau detection.

        Args:
            new_elo: New Elo rating after evaluation
            plateau_threshold: Minimum Elo change to reset plateau count
        """
        old_elo = self.elo_current
        self.elo_trend = new_elo - old_elo
        self.elo_current = new_elo
        self.elo_peak = max(self.elo_peak, new_elo)

        # Track in history
        self.elo_history.append((time.time(), new_elo))

        # Plateau detection
        if abs(self.elo_trend) < plateau_threshold:
            self.elo_plateau_count += 1
        else:
            self.elo_plateau_count = 0

    def update_win_rate(self, new_win_rate: float) -> None:
        """Update win rate with trend tracking.

        Args:
            new_win_rate: New win rate from evaluation (0-1)
        """
        old_win_rate = self.win_rate
        self.win_rate_trend = new_win_rate - old_win_rate
        self.win_rate = new_win_rate

        # Track consecutive high/low streaks
        if new_win_rate > 0.7:
            self.consecutive_high_win_rate += 1
            self.consecutive_low_win_rate = 0
        elif new_win_rate < 0.5:
            self.consecutive_low_win_rate += 1
            self.consecutive_high_win_rate = 0
        else:
            self.consecutive_high_win_rate = 0
            self.consecutive_low_win_rate = 0

    def compute_urgency(self) -> float:
        """Compute composite urgency score (0-1, higher = more urgent).

        Urgency factors:
        - Low win rate (up to 0.2)
        - Declining win rate trend (up to 0.2)
        - Elo plateau (up to 0.2)
        - High curriculum weight (up to 0.2)
        - Poor data quality reduces urgency (multiplier)

        Returns:
            Urgency score between 0 and 1
        """
        urgency = 0.0

        # Factor 1: Low win rate (up to 0.2)
        if self.win_rate < 0.5:
            urgency += (0.5 - self.win_rate) * 0.4

        # Factor 2: Declining win rate (up to 0.2)
        if self.win_rate_trend < 0:
            urgency += min(0.2, abs(self.win_rate_trend) * 2)

        # Factor 3: Elo plateau (up to 0.2)
        urgency += min(0.2, self.elo_plateau_count * 0.04)

        # Factor 4: High curriculum weight (up to 0.2)
        if self.curriculum_weight > 1.0:
            urgency += min(0.2, (self.curriculum_weight - 1.0) * 0.2)

        # Factor 5: Reduce urgency if data quality is poor
        if self.parity_failure_rate > 0.1:
            urgency *= 0.5

        self.urgency_score = min(1.0, urgency)
        self.last_urgency_update = time.time()
        return self.urgency_score

    def compute_data_quality(
        self,
        sample_diversity: float = 1.0,
        avg_game_length: float = 50.0,
        min_game_length: float = 10.0,
        max_game_length: float = 200.0,
    ) -> float:
        """Compute composite data quality score (0-1).

        Args:
            sample_diversity: Diversity metric (0-1)
            avg_game_length: Average game length in moves
            min_game_length: Minimum acceptable game length
            max_game_length: Maximum expected game length

        Returns:
            Composite data quality score between 0 and 1
        """
        quality = 0.0

        # Factor 1: Parity pass rate (40%)
        parity_score = 1.0 - self.parity_failure_rate
        quality += parity_score * 0.4

        # Factor 2: Sample diversity (30%)
        quality += max(0, min(1.0, sample_diversity)) * 0.3

        # Factor 3: Game length normalization (30%)
        if avg_game_length < min_game_length:
            length_score = avg_game_length / min_game_length
        elif avg_game_length > max_game_length:
            length_score = max(0.5, max_game_length / avg_game_length)
        else:
            length_score = 1.0
        quality += length_score * 0.3

        self.data_quality_score = min(1.0, max(0.0, quality))
        return self.data_quality_score

    def is_data_quality_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if data quality meets minimum threshold.

        Args:
            threshold: Minimum acceptable quality score (0-1)

        Returns:
            True if data quality >= threshold
        """
        return self.data_quality_score >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict including monitoring fields."""
        base = super().to_dict()
        base.update(
            {
                "elo_trend": self.elo_trend,
                "elo_peak": self.elo_peak,
                "elo_plateau_count": self.elo_plateau_count,
                "win_rate_trend": self.win_rate_trend,
                "consecutive_high_win_rate": self.consecutive_high_win_rate,
                "consecutive_low_win_rate": self.consecutive_low_win_rate,
                "parity_checks_total": self.parity_checks_total,
                "urgency_score": self.urgency_score,
                "last_urgency_update": self.last_urgency_update,
                # Dec 29, 2025 - migrated from feedback_loop_controller
                "last_promotion_success": self.last_promotion_success,
                "last_selfplay_engine": self.last_selfplay_engine,
                "last_selfplay_games": self.last_selfplay_games,
                "elo_before_training": self.elo_before_training,
                "curriculum_tier": self.curriculum_tier,
                "curriculum_last_advanced": self.curriculum_last_advanced,
                "work_completed_count": self.work_completed_count,
                "last_work_completion_time": self.last_work_completion_time,
                "training_intensity": self.training_intensity,
                "exploration_boost": self.exploration_boost,
                "search_budget": self.search_budget,
            }
        )
        return base


# Backward compatibility aliases
FeedbackState = CanonicalFeedbackState

__all__ = [
    "CanonicalFeedbackState",
    "SignalFeedbackState",
    "MonitoringFeedbackState",
    "FeedbackState",  # Alias for backward compat
]
