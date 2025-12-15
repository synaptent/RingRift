"""Positive Feedback Accelerator for RingRift AI Self-Improvement.

This module optimizes the AI training loop by:
1. Tracking Elo momentum and accelerating training when improvement is detected
2. Implementing adaptive training triggers based on feedback signals
3. Creating a "hot path" for fast iteration when models are improving
4. Dynamically adjusting curriculum weights based on per-config Elo performance

The goal is to maximize positive feedback in the self-improvement cycle:
- When a model improves, generate more data with it quickly
- Train more frequently when signals are positive
- Reduce barriers to promotion for clearly improving models
- Focus resources on configs/models showing momentum

Usage:
    from app.training.feedback_accelerator import (
        FeedbackAccelerator,
        get_feedback_accelerator,
        should_trigger_training,
        get_training_intensity,
    )

    accelerator = get_feedback_accelerator()

    # Check if training should be triggered
    if accelerator.should_trigger_training(config_key):
        intensity = accelerator.get_training_intensity(config_key)
        # Train with adjusted intensity (epochs, learning rate)

    # Record positive feedback
    accelerator.record_elo_update(config_key, new_elo=1650, games_played=100)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_DB_PATH = AI_SERVICE_ROOT / "data" / "feedback" / "accelerator_state.db"

# Elo momentum thresholds
ELO_MOMENTUM_LOOKBACK = 5  # Number of recent updates to consider
ELO_STRONG_IMPROVEMENT = 30.0  # Elo points gained = strong improvement
ELO_MODERATE_IMPROVEMENT = 15.0  # Elo points gained = moderate improvement
ELO_PLATEAU_THRESHOLD = 5.0  # Less than this = plateau

# Training trigger thresholds
MIN_GAMES_FOR_TRAINING = int(os.environ.get("RINGRIFT_MIN_GAMES_TRAINING", "500"))
ACCELERATED_MIN_GAMES = int(os.environ.get("RINGRIFT_ACCEL_MIN_GAMES", "250"))
HOT_PATH_MIN_GAMES = int(os.environ.get("RINGRIFT_HOT_MIN_GAMES", "100"))

# Intensity multipliers
MAX_INTENSITY_MULTIPLIER = 2.0
MIN_INTENSITY_MULTIPLIER = 0.5


class MomentumState(str, Enum):
    """Model improvement momentum states."""
    ACCELERATING = "accelerating"  # Strong improvement, accelerate training
    IMPROVING = "improving"  # Moderate improvement, normal+ training
    STABLE = "stable"  # No significant change
    PLATEAU = "plateau"  # Stuck, may need intervention
    REGRESSING = "regressing"  # Getting worse, slow down and analyze


class TrainingIntensity(str, Enum):
    """Training intensity levels."""
    HOT_PATH = "hot_path"  # Maximum intensity, fastest iteration
    ACCELERATED = "accelerated"  # Above normal intensity
    NORMAL = "normal"  # Standard training
    REDUCED = "reduced"  # Below normal, conserving resources
    PAUSED = "paused"  # No training (analyzing issues)


@dataclass
class EloSnapshot:
    """A point-in-time Elo rating snapshot."""
    elo: float
    games_played: int
    timestamp: float
    model_id: Optional[str] = None


@dataclass
class ConfigMomentum:
    """Momentum tracking for a single config (board_type + num_players)."""
    config_key: str
    current_elo: float = 1500.0
    elo_history: List[EloSnapshot] = field(default_factory=list)
    momentum_state: MomentumState = MomentumState.STABLE
    intensity: TrainingIntensity = TrainingIntensity.NORMAL
    games_since_training: int = 0
    last_training_time: float = 0.0
    consecutive_improvements: int = 0
    consecutive_plateaus: int = 0
    last_promotion_elo: float = 1500.0
    total_promotions: int = 0

    def add_snapshot(self, elo: float, games: int, model_id: Optional[str] = None) -> None:
        """Add an Elo snapshot and update momentum."""
        snapshot = EloSnapshot(
            elo=elo,
            games_played=games,
            timestamp=time.time(),
            model_id=model_id,
        )
        self.elo_history.append(snapshot)

        # Keep bounded history
        if len(self.elo_history) > 100:
            self.elo_history = self.elo_history[-100:]

        self.current_elo = elo
        self._update_momentum()

    def _update_momentum(self) -> None:
        """Update momentum state based on recent Elo history."""
        if len(self.elo_history) < 2:
            self.momentum_state = MomentumState.STABLE
            return

        # Calculate recent trend
        recent = self.elo_history[-ELO_MOMENTUM_LOOKBACK:]
        if len(recent) < 2:
            return

        elo_change = recent[-1].elo - recent[0].elo
        avg_change = elo_change / max(1, len(recent) - 1)

        # Determine momentum state
        if elo_change >= ELO_STRONG_IMPROVEMENT:
            self.momentum_state = MomentumState.ACCELERATING
            self.consecutive_improvements += 1
            self.consecutive_plateaus = 0
        elif elo_change >= ELO_MODERATE_IMPROVEMENT:
            self.momentum_state = MomentumState.IMPROVING
            self.consecutive_improvements += 1
            self.consecutive_plateaus = 0
        elif elo_change < -ELO_MODERATE_IMPROVEMENT:
            self.momentum_state = MomentumState.REGRESSING
            self.consecutive_improvements = 0
            self.consecutive_plateaus = 0
        elif abs(elo_change) < ELO_PLATEAU_THRESHOLD:
            self.momentum_state = MomentumState.PLATEAU
            self.consecutive_improvements = 0
            self.consecutive_plateaus += 1
        else:
            self.momentum_state = MomentumState.STABLE
            self.consecutive_plateaus = 0

        # Update intensity based on momentum
        self._update_intensity()

    def _update_intensity(self) -> None:
        """Update training intensity based on momentum state."""
        if self.momentum_state == MomentumState.ACCELERATING:
            if self.consecutive_improvements >= 3:
                self.intensity = TrainingIntensity.HOT_PATH
            else:
                self.intensity = TrainingIntensity.ACCELERATED
        elif self.momentum_state == MomentumState.IMPROVING:
            self.intensity = TrainingIntensity.ACCELERATED
        elif self.momentum_state == MomentumState.STABLE:
            self.intensity = TrainingIntensity.NORMAL
        elif self.momentum_state == MomentumState.PLATEAU:
            if self.consecutive_plateaus >= 3:
                self.intensity = TrainingIntensity.REDUCED
            else:
                self.intensity = TrainingIntensity.NORMAL
        elif self.momentum_state == MomentumState.REGRESSING:
            self.intensity = TrainingIntensity.REDUCED

    def get_elo_trend(self, lookback: int = 5) -> float:
        """Get recent Elo trend (change per update)."""
        if len(self.elo_history) < 2:
            return 0.0

        recent = self.elo_history[-lookback:]
        if len(recent) < 2:
            return 0.0

        return (recent[-1].elo - recent[0].elo) / max(1, len(recent) - 1)

    def get_improvement_rate(self) -> float:
        """Get improvement rate as Elo points per hour."""
        if len(self.elo_history) < 2:
            return 0.0

        first = self.elo_history[0]
        last = self.elo_history[-1]
        time_diff = last.timestamp - first.timestamp

        if time_diff < 60:  # Less than a minute
            return 0.0

        hours = time_diff / 3600
        elo_diff = last.elo - first.elo

        return elo_diff / hours

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_key": self.config_key,
            "current_elo": self.current_elo,
            "momentum_state": self.momentum_state.value,
            "intensity": self.intensity.value,
            "games_since_training": self.games_since_training,
            "consecutive_improvements": self.consecutive_improvements,
            "consecutive_plateaus": self.consecutive_plateaus,
            "elo_trend": self.get_elo_trend(),
            "improvement_rate_per_hour": self.get_improvement_rate(),
            "total_promotions": self.total_promotions,
        }


@dataclass
class TrainingDecision:
    """Decision about whether/how to train."""
    should_train: bool
    intensity: TrainingIntensity
    epochs_multiplier: float
    learning_rate_multiplier: float
    min_games_threshold: int
    reason: str
    config_key: str
    momentum: MomentumState


class FeedbackAccelerator:
    """Accelerates positive feedback in the AI self-improvement loop.

    This singleton tracks Elo momentum across configurations and adjusts
    training parameters to maximize improvement velocity while maintaining
    stability.
    """

    _instance: Optional["FeedbackAccelerator"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "FeedbackAccelerator":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._db_path = FEEDBACK_DB_PATH
        self._db_lock = threading.RLock()

        # Per-config momentum tracking
        self._configs: Dict[str, ConfigMomentum] = {}

        # Callbacks for training triggers
        self._training_callbacks: List[Callable[[str, TrainingDecision], None]] = []

        # Initialize database
        self._init_db()
        self._load_state()

        logger.info("FeedbackAccelerator initialized")

    def _init_db(self) -> None:
        """Initialize the feedback database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS config_momentum (
                    config_key TEXT PRIMARY KEY,
                    current_elo REAL DEFAULT 1500,
                    momentum_state TEXT DEFAULT 'stable',
                    intensity TEXT DEFAULT 'normal',
                    games_since_training INTEGER DEFAULT 0,
                    last_training_time REAL DEFAULT 0,
                    consecutive_improvements INTEGER DEFAULT 0,
                    consecutive_plateaus INTEGER DEFAULT 0,
                    last_promotion_elo REAL DEFAULT 1500,
                    total_promotions INTEGER DEFAULT 0,
                    updated_at REAL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS elo_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT NOT NULL,
                    elo REAL NOT NULL,
                    games_played INTEGER NOT NULL,
                    model_id TEXT,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (config_key) REFERENCES config_momentum(config_key)
                );

                CREATE INDEX IF NOT EXISTS idx_elo_history_config
                    ON elo_history(config_key, timestamp);

                CREATE TABLE IF NOT EXISTS training_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT NOT NULL,
                    should_train INTEGER NOT NULL,
                    intensity TEXT NOT NULL,
                    epochs_multiplier REAL NOT NULL,
                    lr_multiplier REAL NOT NULL,
                    reason TEXT NOT NULL,
                    momentum_state TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_training_decisions_config
                    ON training_decisions(config_key, timestamp);
            """)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_state(self) -> None:
        """Load state from database."""
        with self._db_lock:
            with self._get_connection() as conn:
                # Load config momentum
                rows = conn.execute("SELECT * FROM config_momentum").fetchall()
                for row in rows:
                    config_key = row["config_key"]
                    momentum = ConfigMomentum(
                        config_key=config_key,
                        current_elo=row["current_elo"],
                        momentum_state=MomentumState(row["momentum_state"]),
                        intensity=TrainingIntensity(row["intensity"]),
                        games_since_training=row["games_since_training"],
                        last_training_time=row["last_training_time"],
                        consecutive_improvements=row["consecutive_improvements"],
                        consecutive_plateaus=row["consecutive_plateaus"],
                        last_promotion_elo=row["last_promotion_elo"],
                        total_promotions=row["total_promotions"],
                    )

                    # Load recent Elo history
                    history_rows = conn.execute("""
                        SELECT elo, games_played, model_id, timestamp
                        FROM elo_history
                        WHERE config_key = ?
                        ORDER BY timestamp DESC
                        LIMIT 100
                    """, (config_key,)).fetchall()

                    for h in reversed(history_rows):
                        momentum.elo_history.append(EloSnapshot(
                            elo=h["elo"],
                            games_played=h["games_played"],
                            model_id=h["model_id"],
                            timestamp=h["timestamp"],
                        ))

                    self._configs[config_key] = momentum

    def _save_config(self, config_key: str) -> None:
        """Save config momentum to database."""
        momentum = self._configs.get(config_key)
        if not momentum:
            return

        with self._db_lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO config_momentum (
                        config_key, current_elo, momentum_state, intensity,
                        games_since_training, last_training_time,
                        consecutive_improvements, consecutive_plateaus,
                        last_promotion_elo, total_promotions, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    config_key, momentum.current_elo, momentum.momentum_state.value,
                    momentum.intensity.value, momentum.games_since_training,
                    momentum.last_training_time, momentum.consecutive_improvements,
                    momentum.consecutive_plateaus, momentum.last_promotion_elo,
                    momentum.total_promotions, time.time(),
                ))
                conn.commit()

    # =========================================================================
    # Elo Update Tracking
    # =========================================================================

    def record_elo_update(
        self,
        config_key: str,
        new_elo: float,
        games_played: int,
        model_id: Optional[str] = None,
    ) -> ConfigMomentum:
        """Record an Elo update and update momentum tracking.

        Args:
            config_key: Configuration key (e.g., "square8_2p")
            new_elo: New Elo rating
            games_played: Total games played
            model_id: Optional model identifier

        Returns:
            Updated ConfigMomentum
        """
        if config_key not in self._configs:
            self._configs[config_key] = ConfigMomentum(config_key=config_key)

        momentum = self._configs[config_key]
        old_elo = momentum.current_elo

        # Add snapshot and update momentum
        momentum.add_snapshot(new_elo, games_played, model_id)

        # Save to database
        with self._db_lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO elo_history (config_key, elo, games_played, model_id, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (config_key, new_elo, games_played, model_id, time.time()))
                conn.commit()

        self._save_config(config_key)

        # Log significant changes
        elo_change = new_elo - old_elo
        if abs(elo_change) >= ELO_MODERATE_IMPROVEMENT:
            direction = "+" if elo_change > 0 else ""
            logger.info(
                f"[FeedbackAccelerator] {config_key}: Elo {direction}{elo_change:.1f} "
                f"({old_elo:.0f} -> {new_elo:.0f}), momentum={momentum.momentum_state.value}"
            )

        return momentum

    def record_games_generated(self, config_key: str, games: int) -> None:
        """Record games generated for a config."""
        if config_key not in self._configs:
            self._configs[config_key] = ConfigMomentum(config_key=config_key)

        self._configs[config_key].games_since_training += games
        self._save_config(config_key)

    def record_training_complete(
        self,
        config_key: str,
        success: bool = True,
        new_elo: Optional[float] = None,
    ) -> None:
        """Record that training completed for a config."""
        if config_key not in self._configs:
            return

        momentum = self._configs[config_key]
        momentum.last_training_time = time.time()
        momentum.games_since_training = 0

        if new_elo is not None:
            momentum.add_snapshot(new_elo, momentum.elo_history[-1].games_played if momentum.elo_history else 0)

        self._save_config(config_key)

    def record_promotion(
        self,
        config_key: str,
        new_elo: float,
        model_id: Optional[str] = None,
    ) -> None:
        """Record a successful model promotion."""
        if config_key not in self._configs:
            self._configs[config_key] = ConfigMomentum(config_key=config_key)

        momentum = self._configs[config_key]
        momentum.last_promotion_elo = new_elo
        momentum.total_promotions += 1

        logger.info(
            f"[FeedbackAccelerator] {config_key}: Promotion #{momentum.total_promotions} "
            f"at Elo {new_elo:.0f}"
        )

        self._save_config(config_key)

    # =========================================================================
    # Training Decisions
    # =========================================================================

    def should_trigger_training(self, config_key: str) -> bool:
        """Check if training should be triggered for a config.

        Returns True if enough games have been generated based on current
        momentum and intensity settings.
        """
        decision = self.get_training_decision(config_key)
        return decision.should_train

    def get_training_decision(self, config_key: str) -> TrainingDecision:
        """Get a detailed training decision for a config.

        Returns TrainingDecision with should_train, intensity, and multipliers.
        """
        if config_key not in self._configs:
            # No data yet, use defaults
            return TrainingDecision(
                should_train=False,
                intensity=TrainingIntensity.NORMAL,
                epochs_multiplier=1.0,
                learning_rate_multiplier=1.0,
                min_games_threshold=MIN_GAMES_FOR_TRAINING,
                reason="No momentum data yet",
                config_key=config_key,
                momentum=MomentumState.STABLE,
            )

        momentum = self._configs[config_key]
        games = momentum.games_since_training

        # Determine threshold based on intensity
        if momentum.intensity == TrainingIntensity.HOT_PATH:
            threshold = HOT_PATH_MIN_GAMES
            epochs_mult = 1.5
            lr_mult = 1.2
        elif momentum.intensity == TrainingIntensity.ACCELERATED:
            threshold = ACCELERATED_MIN_GAMES
            epochs_mult = 1.2
            lr_mult = 1.1
        elif momentum.intensity == TrainingIntensity.REDUCED:
            threshold = MIN_GAMES_FOR_TRAINING * 1.5
            epochs_mult = 0.8
            lr_mult = 0.9
        elif momentum.intensity == TrainingIntensity.PAUSED:
            return TrainingDecision(
                should_train=False,
                intensity=momentum.intensity,
                epochs_multiplier=0.0,
                learning_rate_multiplier=0.0,
                min_games_threshold=999999,
                reason="Training paused due to issues",
                config_key=config_key,
                momentum=momentum.momentum_state,
            )
        else:
            threshold = MIN_GAMES_FOR_TRAINING
            epochs_mult = 1.0
            lr_mult = 1.0

        should_train = games >= threshold
        reason = (
            f"Games {games}/{threshold}, momentum={momentum.momentum_state.value}, "
            f"intensity={momentum.intensity.value}"
        )

        decision = TrainingDecision(
            should_train=should_train,
            intensity=momentum.intensity,
            epochs_multiplier=epochs_mult,
            learning_rate_multiplier=lr_mult,
            min_games_threshold=int(threshold),
            reason=reason,
            config_key=config_key,
            momentum=momentum.momentum_state,
        )

        # Record decision
        self._record_decision(decision)

        return decision

    def _record_decision(self, decision: TrainingDecision) -> None:
        """Record a training decision for analysis."""
        with self._db_lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO training_decisions (
                        config_key, should_train, intensity, epochs_multiplier,
                        lr_multiplier, reason, momentum_state, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision.config_key, int(decision.should_train),
                    decision.intensity.value, decision.epochs_multiplier,
                    decision.learning_rate_multiplier, decision.reason,
                    decision.momentum.value, time.time(),
                ))
                conn.commit()

    def get_training_intensity(self, config_key: str) -> Dict[str, float]:
        """Get training intensity parameters for a config.

        Returns dict with epochs_multiplier, learning_rate_multiplier, etc.
        """
        decision = self.get_training_decision(config_key)
        return {
            "epochs_multiplier": decision.epochs_multiplier,
            "learning_rate_multiplier": decision.learning_rate_multiplier,
            "min_games_threshold": decision.min_games_threshold,
            "intensity": decision.intensity.value,
        }

    # =========================================================================
    # Curriculum Weight Recommendations
    # =========================================================================

    def get_curriculum_weights(self) -> Dict[str, float]:
        """Get recommended curriculum weights based on momentum.

        Returns dict mapping config_key to weight multiplier.
        Configs with better momentum get higher weights.
        """
        weights = {}

        for config_key, momentum in self._configs.items():
            # Base weight
            weight = 1.0

            # Boost for improving configs (more training data helps momentum)
            if momentum.momentum_state == MomentumState.ACCELERATING:
                weight = 1.5
            elif momentum.momentum_state == MomentumState.IMPROVING:
                weight = 1.3
            elif momentum.momentum_state == MomentumState.PLATEAU:
                # Slight boost to break plateau
                weight = 1.2
            elif momentum.momentum_state == MomentumState.REGRESSING:
                # Significant boost to reverse regression
                weight = 1.4

            # Adjust for consecutive patterns
            if momentum.consecutive_improvements >= 3:
                weight *= 1.2  # Reward sustained improvement
            elif momentum.consecutive_plateaus >= 3:
                weight *= 1.3  # Increase focus on stuck configs

            weights[config_key] = min(MAX_INTENSITY_MULTIPLIER, weight)

        return weights

    def get_selfplay_rate_recommendation(self, config_key: str) -> float:
        """Get recommended selfplay rate multiplier for a config.

        Improving models should generate more data to capitalize on momentum.
        """
        if config_key not in self._configs:
            return 1.0

        momentum = self._configs[config_key]

        if momentum.momentum_state == MomentumState.ACCELERATING:
            return 1.5  # Generate 50% more data
        elif momentum.momentum_state == MomentumState.IMPROVING:
            return 1.3
        elif momentum.momentum_state == MomentumState.PLATEAU:
            return 1.1  # Slight increase to try breaking plateau
        elif momentum.momentum_state == MomentumState.REGRESSING:
            return 0.8  # Reduce until we understand the issue
        else:
            return 1.0

    # =========================================================================
    # Status and Reporting
    # =========================================================================

    def get_config_momentum(self, config_key: str) -> Optional[ConfigMomentum]:
        """Get momentum data for a config."""
        return self._configs.get(config_key)

    def get_all_momentum(self) -> Dict[str, Dict[str, Any]]:
        """Get momentum data for all configs."""
        return {k: v.to_dict() for k, v in self._configs.items()}

    def get_improving_configs(self) -> List[str]:
        """Get list of configs that are improving."""
        return [
            k for k, v in self._configs.items()
            if v.momentum_state in (MomentumState.ACCELERATING, MomentumState.IMPROVING)
        ]

    def get_plateau_configs(self) -> List[str]:
        """Get list of configs that are in plateau."""
        return [
            k for k, v in self._configs.items()
            if v.momentum_state == MomentumState.PLATEAU
        ]

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of feedback accelerator status."""
        improving = self.get_improving_configs()
        plateau = self.get_plateau_configs()

        # Calculate overall improvement rate
        total_improvement_rate = sum(
            v.get_improvement_rate() for v in self._configs.values()
        )

        return {
            "total_configs": len(self._configs),
            "improving_configs": len(improving),
            "plateau_configs": len(plateau),
            "improving": improving,
            "plateau": plateau,
            "total_improvement_rate_per_hour": total_improvement_rate,
            "configs": self.get_all_momentum(),
        }

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics suitable for Prometheus exposition."""
        metrics = {
            "ringrift_feedback_total_configs": len(self._configs),
            "ringrift_feedback_improving_configs": len(self.get_improving_configs()),
            "ringrift_feedback_plateau_configs": len(self.get_plateau_configs()),
        }

        # Per-config metrics
        for config_key, momentum in self._configs.items():
            safe_key = config_key.replace("-", "_").replace(".", "_")
            metrics[f"ringrift_elo_{safe_key}"] = momentum.current_elo
            metrics[f"ringrift_elo_trend_{safe_key}"] = momentum.get_elo_trend()
            metrics[f"ringrift_improvement_rate_{safe_key}"] = momentum.get_improvement_rate()

        return metrics

    # =========================================================================
    # Training Callbacks
    # =========================================================================

    def register_training_callback(
        self,
        callback: Callable[[str, TrainingDecision], None],
    ) -> None:
        """Register a callback to be called when training should be triggered."""
        self._training_callbacks.append(callback)

    def check_and_trigger_training(self) -> List[TrainingDecision]:
        """Check all configs and trigger training callbacks where appropriate.

        Returns list of training decisions that were triggered.
        """
        triggered = []

        for config_key in list(self._configs.keys()):
            decision = self.get_training_decision(config_key)
            if decision.should_train:
                triggered.append(decision)
                for callback in self._training_callbacks:
                    try:
                        callback(config_key, decision)
                    except Exception as e:
                        logger.error(f"Training callback failed for {config_key}: {e}")

        return triggered


# =============================================================================
# Module-level convenience functions
# =============================================================================

_accelerator: Optional[FeedbackAccelerator] = None


def get_feedback_accelerator() -> FeedbackAccelerator:
    """Get the singleton feedback accelerator."""
    global _accelerator
    if _accelerator is None:
        _accelerator = FeedbackAccelerator()
    return _accelerator


def should_trigger_training(config_key: str) -> bool:
    """Check if training should be triggered for a config."""
    return get_feedback_accelerator().should_trigger_training(config_key)


def get_training_intensity(config_key: str) -> Dict[str, float]:
    """Get training intensity parameters for a config."""
    return get_feedback_accelerator().get_training_intensity(config_key)


def record_elo_update(
    config_key: str,
    new_elo: float,
    games_played: int,
    model_id: Optional[str] = None,
) -> None:
    """Record an Elo update."""
    get_feedback_accelerator().record_elo_update(config_key, new_elo, games_played, model_id)


def record_games_generated(config_key: str, games: int) -> None:
    """Record games generated for a config."""
    get_feedback_accelerator().record_games_generated(config_key, games)


def record_training_complete(config_key: str, success: bool = True) -> None:
    """Record that training completed for a config."""
    get_feedback_accelerator().record_training_complete(config_key, success)


def record_promotion(config_key: str, new_elo: float, model_id: Optional[str] = None) -> None:
    """Record a successful model promotion."""
    get_feedback_accelerator().record_promotion(config_key, new_elo, model_id)


def get_curriculum_weights() -> Dict[str, float]:
    """Get recommended curriculum weights based on momentum."""
    return get_feedback_accelerator().get_curriculum_weights()


def get_selfplay_rate_recommendation(config_key: str) -> float:
    """Get recommended selfplay rate multiplier for a config."""
    return get_feedback_accelerator().get_selfplay_rate_recommendation(config_key)
