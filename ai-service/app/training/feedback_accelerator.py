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

import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Import canonical Elo and timeout constants
try:
    from app.config.thresholds import INITIAL_ELO_RATING, SQLITE_TIMEOUT
except ImportError:
    INITIAL_ELO_RATING = 1500.0
    SQLITE_TIMEOUT = 30

# Import unified signals for cross-system consistency
from app.utils.paths import AI_SERVICE_ROOT

from .unified_signals import (
    TrainingSignals,
    TrainingUrgency,
    get_signal_computer,
)
from .improvement_optimizer import (
    get_improvement_optimizer,
    get_dynamic_threshold,
    get_selfplay_priority_boost,
    should_fast_track_training as optimizer_should_fast_track,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================
FEEDBACK_DB_PATH = AI_SERVICE_ROOT / "data" / "feedback" / "accelerator_state.db"

# Elo momentum thresholds
ELO_MOMENTUM_LOOKBACK = 5  # Number of recent updates to consider
ELO_STRONG_IMPROVEMENT = 25.0  # OPTIMIZED: Lower bar for strong improvement (was 30)
ELO_MODERATE_IMPROVEMENT = 12.0  # OPTIMIZED: Lower bar for moderate improvement (was 15)
ELO_PLATEAU_THRESHOLD = 5.0  # Less than this = plateau

# Training trigger thresholds
MIN_GAMES_FOR_TRAINING = int(os.environ.get("RINGRIFT_MIN_GAMES_TRAINING", "300"))  # OPTIMIZED: 300 (was 500)
ACCELERATED_MIN_GAMES = int(os.environ.get("RINGRIFT_ACCEL_MIN_GAMES", "150"))  # OPTIMIZED: 150 (was 250)
HOT_PATH_MIN_GAMES = int(os.environ.get("RINGRIFT_HOT_MIN_GAMES", "75"))  # OPTIMIZED: 75 games for maximum speed

# Intensity multipliers
MAX_INTENSITY_MULTIPLIER = 2.5  # OPTIMIZED: Higher ceiling for accelerating models (was 2.0)
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
    model_id: str | None = None


@dataclass
class ConfigMomentum:
    """Momentum tracking for a single config (board_type + num_players).

    Note: Uses INITIAL_ELO_RATING from app.config.thresholds as default.
    """
    config_key: str
    current_elo: float = INITIAL_ELO_RATING
    elo_history: list[EloSnapshot] = field(default_factory=list)
    momentum_state: MomentumState = MomentumState.STABLE
    intensity: TrainingIntensity = TrainingIntensity.NORMAL
    games_since_training: int = 0
    last_training_time: float = 0.0
    consecutive_improvements: int = 0
    consecutive_plateaus: int = 0
    last_promotion_elo: float = INITIAL_ELO_RATING
    total_promotions: int = 0

    def add_snapshot(self, elo: float, games: int, model_id: str | None = None) -> None:
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
        # Note: elo_change_per_update can be computed as elo_change / max(1, len(recent) - 1)
        # but we use total elo_change for momentum thresholds which is more stable

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
        """Update training intensity based on momentum state.

        OPTIMIZED: Faster hot-path activation (2 consecutive instead of 3)
        to maximize positive feedback during improvement phases.
        """
        if self.momentum_state == MomentumState.ACCELERATING:
            # OPTIMIZED: Trigger hot-path after just 2 consecutive improvements (was 3)
            if self.consecutive_improvements >= 2:
                self.intensity = TrainingIntensity.HOT_PATH
            else:
                self.intensity = TrainingIntensity.ACCELERATED
        elif self.momentum_state == MomentumState.IMPROVING:
            # OPTIMIZED: Jump to accelerated immediately on any improvement
            self.intensity = TrainingIntensity.ACCELERATED
        elif self.momentum_state == MomentumState.STABLE:
            self.intensity = TrainingIntensity.NORMAL
        elif self.momentum_state == MomentumState.PLATEAU:
            # OPTIMIZED: Stay at normal longer before reducing (was 3)
            if self.consecutive_plateaus >= 4:
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

    def to_dict(self) -> dict[str, Any]:
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

    _instance: FeedbackAccelerator | None = None
    _lock = threading.RLock()

    def __new__(cls) -> FeedbackAccelerator:
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
        self._configs: dict[str, ConfigMomentum] = {}

        # Track last selfplay rate recommendations for change detection (Phase 19.3)
        self._last_rates: dict[str, float] = {}

        # Callbacks for training triggers
        self._training_callbacks: list[Callable[[str, TrainingDecision], None]] = []

        # Unified signal computer for cross-system consistency
        self._signal_computer = get_signal_computer()

        # Initialize database
        self._init_db()
        self._load_state()

        # Auto-wire evaluation events for feedback loop (December 2025)
        self._auto_wire_events()

        logger.info("FeedbackAccelerator initialized")

    def _auto_wire_events(self) -> None:
        """Auto-wire event subscriptions for the feedback loop.

        Called during initialization to ensure evaluation results
        automatically update momentum tracking.
        """
        try:
            wire_evaluation_to_feedback()
        except Exception as e:
            # Non-fatal - events may not be available in all contexts
            logger.debug(f"[FeedbackAccelerator] Event auto-wiring deferred: {e}")

        # Phase 5: Subscribe to HYPERPARAMETER_UPDATED for closed-loop feedback
        try:
            wire_hyperparameter_feedback()
        except Exception as e:
            logger.debug(f"[FeedbackAccelerator] Hyperparameter event wiring deferred: {e}")

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
        conn = sqlite3.connect(str(self._db_path), timeout=SQLITE_TIMEOUT)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_state(self) -> None:
        """Load state from database."""
        with self._db_lock, self._get_connection() as conn:
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

        with self._db_lock, self._get_connection() as conn:
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
        model_id: str | None = None,
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

        # Sync with unified signal computer for cross-system consistency
        self._signal_computer.update_config_state(
            config_key=config_key,
            current_elo=new_elo,
        )

        # Save to database
        with self._db_lock, self._get_connection() as conn:
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

        # Check for Elo plateau and trigger response (December 2025)
        self._check_elo_plateau(config_key, momentum)

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
        new_elo: float | None = None,
        games_at_training: int | None = None,
    ) -> None:
        """Record that training completed for a config.

        Syncs state with unified signal computer for cross-system consistency.
        """
        if config_key not in self._configs:
            return

        momentum = self._configs[config_key]
        momentum.last_training_time = time.time()
        momentum.games_since_training = 0

        if new_elo is not None:
            momentum.add_snapshot(new_elo, momentum.elo_history[-1].games_played if momentum.elo_history else 0)

        # Sync with unified signal computer
        if games_at_training is not None:
            self._signal_computer.record_training_started(games_at_training, config_key)
        self._signal_computer.record_training_completed(new_elo, config_key)

        self._save_config(config_key)

    def record_promotion(
        self,
        config_key: str,
        new_elo: float,
        model_id: str | None = None,
    ) -> None:
        """Record a successful model promotion.

        Also notifies ImprovementOptimizer to adjust dynamic thresholds
        based on promotion success (+5-15 Elo potential from accelerated cycles).
        """
        if config_key not in self._configs:
            self._configs[config_key] = ConfigMomentum(config_key=config_key)

        momentum = self._configs[config_key]
        old_elo = momentum.last_promotion_elo
        elo_gain = new_elo - old_elo
        momentum.last_promotion_elo = new_elo
        momentum.total_promotions += 1

        # Notify improvement optimizer for dynamic threshold adjustment
        optimizer = get_improvement_optimizer()
        optimizer.record_promotion_success(config_key, elo_gain, model_id or "")

        logger.info(
            f"[FeedbackAccelerator] {config_key}: Promotion #{momentum.total_promotions} "
            f"at Elo {new_elo:.0f} (+{elo_gain:.0f})"
        )

        self._save_config(config_key)

    def signal_training_needed(
        self,
        config_key: str | None = None,
        urgency: str = "normal",
        reason: str = "",
    ) -> None:
        """Signal that training is urgently needed for a config.

        This is called by CurriculumFeedback when promotion fails, indicating
        the need for more aggressive training to break plateaus.

        Args:
            config_key: Configuration key (e.g., "square8_2p"). If None, applies globally.
            urgency: One of "low", "normal", "high", "critical"
            reason: Human-readable reason for the urgency
        """
        urgency_to_intensity = {
            "critical": TrainingIntensity.HOT_PATH,
            "high": TrainingIntensity.ACCELERATED,
            "normal": TrainingIntensity.NORMAL,
            "low": TrainingIntensity.REDUCED,
        }

        new_intensity = urgency_to_intensity.get(urgency, TrainingIntensity.NORMAL)

        if config_key:
            if config_key not in self._configs:
                self._configs[config_key] = ConfigMomentum(config_key=config_key)

            momentum = self._configs[config_key]
            old_intensity = momentum.intensity

            # Only upgrade intensity, never downgrade via signal
            if self._intensity_priority(new_intensity) > self._intensity_priority(old_intensity):
                momentum.intensity = new_intensity
                self._save_config(config_key)

                logger.info(
                    f"[FeedbackAccelerator] Training urgency signaled for {config_key}: "
                    f"{old_intensity.value} → {new_intensity.value} ({reason})"
                )
        else:
            # Apply to all configs
            for key in list(self._configs.keys()):
                self.signal_training_needed(key, urgency, reason)

    @staticmethod
    def _intensity_priority(intensity: TrainingIntensity) -> int:
        """Get numeric priority for intensity comparison."""
        priority_map = {
            TrainingIntensity.PAUSED: 0,
            TrainingIntensity.REDUCED: 1,
            TrainingIntensity.NORMAL: 2,
            TrainingIntensity.ACCELERATED: 3,
            TrainingIntensity.HOT_PATH: 4,
        }
        return priority_map.get(intensity, 2)

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
        Integrates with ImprovementOptimizer for dynamic threshold adjustment.
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

        # Get dynamic threshold from improvement optimizer (+5-15 Elo potential)
        optimizer_threshold = get_dynamic_threshold(config_key)
        fast_track = optimizer_should_fast_track(config_key)

        # Determine threshold based on intensity, adjusted by optimizer
        # OPTIMIZED: Higher multipliers for accelerating models to maximize learning velocity
        if momentum.intensity == TrainingIntensity.HOT_PATH:
            threshold = min(HOT_PATH_MIN_GAMES, optimizer_threshold)  # Use lower of two
            epochs_mult = 2.0  # OPTIMIZED: Double epochs for hot-path (was 1.5)
            lr_mult = 1.3  # OPTIMIZED: Higher LR for faster convergence (was 1.2)
        elif momentum.intensity == TrainingIntensity.ACCELERATED or fast_track:
            threshold = min(ACCELERATED_MIN_GAMES, optimizer_threshold)
            epochs_mult = 1.5  # OPTIMIZED: More epochs (was 1.2)
            lr_mult = 1.2  # OPTIMIZED: Higher LR (was 1.1)
        elif momentum.intensity == TrainingIntensity.REDUCED:
            threshold = max(int(MIN_GAMES_FOR_TRAINING * 1.5), optimizer_threshold)  # 450 games
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
            threshold = optimizer_threshold  # Use optimizer's dynamic threshold
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
        with self._db_lock, self._get_connection() as conn:
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

    def get_training_intensity(self, config_key: str) -> dict[str, float]:
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

    def set_intensity(self, config_key: str, intensity: TrainingIntensity) -> None:
        """Set training intensity for a configuration.

        Called by FeedbackLoopController when Elo momentum changes require
        an intensity adjustment. This is part of the Phase 22 bug fix.

        Args:
            config_key: Configuration key (e.g., "square8_2p")
            intensity: New training intensity level
        """
        if config_key not in self._configs:
            self._configs[config_key] = ConfigMomentum(config_key=config_key)

        old_intensity = self._configs[config_key].intensity
        self._configs[config_key].intensity = intensity

        if old_intensity != intensity:
            logger.info(
                f"[FeedbackAccelerator] Intensity for {config_key}: "
                f"{old_intensity.value} → {intensity.value}"
            )

        self._save_config(config_key)

    def get_selfplay_multiplier(self, config_key: str) -> float:
        """Get selfplay games multiplier based on Elo momentum (December 2025).

        This implements the Elo momentum → Selfplay rate coupling:
        - ACCELERATING: 1.5x games (capitalize on positive momentum)
        - IMPROVING: 1.25x games (boost for continued improvement)
        - STABLE: 1.0x games (normal rate)
        - PLATEAU: 1.1x games (slight boost to try to break plateau)
        - REGRESSING: 0.75x games (reduce noise, focus on quality)

        Args:
            config_key: Config identifier (e.g., "square8_2p")

        Returns:
            Multiplier for selfplay games (0.5 - 1.5)
        """
        momentum = self._configs.get(config_key)
        if not momentum:
            return 1.0

        # Map momentum state to selfplay multiplier
        multiplier_map = {
            MomentumState.ACCELERATING: 1.5,  # Max boost during strong improvement
            MomentumState.IMPROVING: 1.25,    # Good boost during improvement
            MomentumState.STABLE: 1.0,        # Normal rate when stable
            MomentumState.PLATEAU: 1.1,       # Slight boost to break plateau
            MomentumState.REGRESSING: 0.75,   # Reduce rate during regression
        }

        base_multiplier = multiplier_map.get(momentum.momentum_state, 1.0)

        # Additional boost for consecutive improvements (compound positive feedback)
        if momentum.consecutive_improvements >= 3:
            base_multiplier = min(base_multiplier * 1.1, 1.5)

        # Limit during consecutive plateaus
        if momentum.consecutive_plateaus >= 3:
            base_multiplier = max(base_multiplier * 0.9, 0.5)

        logger.debug(
            f"[FeedbackAccelerator] Selfplay multiplier for {config_key}: "
            f"{base_multiplier:.2f} (momentum={momentum.momentum_state.value})"
        )

        # Phase 19.3: Check for significant rate change and emit event
        self._check_rate_change(config_key, base_multiplier)

        return base_multiplier

    def _check_rate_change(self, config_key: str, new_rate: float) -> None:
        """Check if selfplay rate recommendation changed significantly.

        Phase 19.3: Emit SELFPLAY_RATE_CHANGED event when rate changes by >20%.
        This enables monitoring dashboards and other systems to track rate changes.

        Args:
            config_key: Configuration key (e.g., "square8_2p")
            new_rate: New selfplay rate multiplier
        """
        old_rate = self._last_rates.get(config_key, 1.0)

        # Check for significant change (>20%)
        if abs(new_rate - old_rate) > 0.2:
            self._emit_rate_change_event(config_key, old_rate, new_rate)
            self._last_rates[config_key] = new_rate
        elif config_key not in self._last_rates:
            # Initialize tracking for new config
            self._last_rates[config_key] = new_rate

    def _emit_rate_change_event(
        self,
        config_key: str,
        old_rate: float,
        new_rate: float,
    ) -> None:
        """Emit SELFPLAY_RATE_CHANGED event for monitoring.

        Args:
            config_key: Configuration key
            old_rate: Previous rate multiplier
            new_rate: New rate multiplier
        """
        momentum = self._configs.get(config_key)
        momentum_state = momentum.momentum_state.value if momentum else "unknown"

        direction = "increased" if new_rate > old_rate else "decreased"
        change_pct = ((new_rate - old_rate) / old_rate * 100) if old_rate > 0 else 0

        logger.info(
            f"[FeedbackAccelerator] Selfplay rate {direction} for {config_key}: "
            f"{old_rate:.2f}x → {new_rate:.2f}x ({change_pct:+.0f}%), "
            f"momentum={momentum_state}"
        )

        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()
            # Phase 22.2 fix: Use publish_sync instead of emit (which doesn't exist)
            router.publish_sync(
                DataEventType.SELFPLAY_RATE_CHANGED.value
                if hasattr(DataEventType, 'SELFPLAY_RATE_CHANGED')
                else "selfplay_rate_changed",
                {
                    "config": config_key,
                    "old_rate": old_rate,
                    "new_rate": new_rate,
                    "change_percent": change_pct,
                    "direction": direction,
                    "momentum_state": momentum_state,
                    "timestamp": time.time(),
                },
                source="feedback_accelerator",
            )
        except Exception as e:
            # Non-fatal - event system may not be available
            logger.debug(f"[FeedbackAccelerator] Could not publish rate change event: {e}")

    # =========================================================================
    # Curriculum Weight Recommendations
    # =========================================================================

    def get_curriculum_weights(self) -> dict[str, float]:
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
        OPTIMIZED: Higher multipliers to maximize data generation during improvement.
        Integrates ImprovementOptimizer priority boost for +5-15 Elo potential.
        """
        # Get priority boost from improvement optimizer (based on promotion streaks, etc.)
        optimizer_boost = get_selfplay_priority_boost(config_key)

        if config_key not in self._configs:
            return 1.0 + optimizer_boost

        momentum = self._configs[config_key]
        base_multiplier = 1.0

        if momentum.momentum_state == MomentumState.ACCELERATING:
            # OPTIMIZED: Generate 2x more data during strong improvement (was 1.5)
            if momentum.consecutive_improvements >= 3:
                base_multiplier = 2.5  # Super hot-path: 2.5x data generation
            else:
                base_multiplier = 2.0
        elif momentum.momentum_state == MomentumState.IMPROVING:
            base_multiplier = 1.5  # OPTIMIZED: was 1.3
        elif momentum.momentum_state == MomentumState.PLATEAU:
            base_multiplier = 1.2  # OPTIMIZED: Increase to try breaking plateau (was 1.1)
        elif momentum.momentum_state == MomentumState.REGRESSING:
            base_multiplier = 0.7  # OPTIMIZED: Reduce more to avoid bad data (was 0.8)

        # Apply optimizer boost (can add up to +0.15 for promotion streaks)
        return min(3.0, base_multiplier + optimizer_boost)  # Cap at 3x

    def get_aggregate_selfplay_recommendation(self) -> dict[str, Any]:
        """Get aggregate selfplay rate recommendation across all configs.

        Returns a recommendation based on the overall momentum state of the system,
        useful for cluster-wide selfplay rate adjustments.
        """
        if not self._configs:
            return {
                'recommended_multiplier': 1.0,
                'reason': 'no configs tracked',
                'aggregate_momentum': 'unknown',
            }

        # Count configs in each state
        accelerating = 0
        improving = 0
        stable = 0
        plateau = 0
        regressing = 0

        for config in self._configs.values():
            if config.momentum_state == MomentumState.ACCELERATING:
                accelerating += 1
            elif config.momentum_state == MomentumState.IMPROVING:
                improving += 1
            elif config.momentum_state == MomentumState.PLATEAU:
                plateau += 1
            elif config.momentum_state == MomentumState.REGRESSING:
                regressing += 1
            else:
                stable += 1

        total = len(self._configs)

        # Determine aggregate state based on majority
        if accelerating >= total * 0.5:
            aggregate = 'accelerating'
            multiplier = 2.0
            reason = f'{accelerating}/{total} configs accelerating'
        elif accelerating + improving >= total * 0.5:
            aggregate = 'improving'
            multiplier = 1.5
            reason = f'{accelerating + improving}/{total} configs improving'
        elif regressing >= total * 0.3:
            aggregate = 'regressing'
            multiplier = 0.8
            reason = f'{regressing}/{total} configs regressing'
        elif plateau >= total * 0.5:
            aggregate = 'plateau'
            multiplier = 1.1
            reason = f'{plateau}/{total} configs in plateau'
        else:
            aggregate = 'stable'
            multiplier = 1.0
            reason = 'mixed momentum states'

        return {
            'recommended_multiplier': multiplier,
            'reason': reason,
            'aggregate_momentum': aggregate,
            'states': {
                'accelerating': accelerating,
                'improving': improving,
                'stable': stable,
                'plateau': plateau,
                'regressing': regressing,
            }
        }

    # =========================================================================
    # Status and Reporting
    # =========================================================================

    def get_config_momentum(self, config_key: str) -> ConfigMomentum | None:
        """Get momentum data for a config."""
        return self._configs.get(config_key)

    def get_all_momentum(self) -> dict[str, dict[str, Any]]:
        """Get momentum data for all configs."""
        return {k: v.to_dict() for k, v in self._configs.items()}

    def get_improving_configs(self) -> list[str]:
        """Get list of configs that are improving."""
        return [
            k for k, v in self._configs.items()
            if v.momentum_state in (MomentumState.ACCELERATING, MomentumState.IMPROVING)
        ]

    def get_plateau_configs(self) -> list[str]:
        """Get list of configs that are in plateau."""
        return [
            k for k, v in self._configs.items()
            if v.momentum_state == MomentumState.PLATEAU
        ]

    def get_unified_urgency(self, config_key: str) -> TrainingUrgency:
        """Get unified training urgency for a config.

        Maps FeedbackAccelerator's intensity to unified TrainingUrgency.
        This provides cross-system compatibility.
        """
        if config_key not in self._configs:
            return TrainingUrgency.NONE

        momentum = self._configs[config_key]

        # Map intensity to urgency
        intensity_to_urgency = {
            TrainingIntensity.HOT_PATH: TrainingUrgency.CRITICAL,
            TrainingIntensity.ACCELERATED: TrainingUrgency.HIGH,
            TrainingIntensity.NORMAL: TrainingUrgency.NORMAL,
            TrainingIntensity.REDUCED: TrainingUrgency.LOW,
            TrainingIntensity.PAUSED: TrainingUrgency.NONE,
        }

        return intensity_to_urgency.get(momentum.intensity, TrainingUrgency.NORMAL)

    def get_unified_signals(self, config_key: str) -> TrainingSignals | None:
        """Get unified training signals for a config.

        Combines FeedbackAccelerator state with unified signal computer
        for cross-system consistency.
        """
        if config_key not in self._configs:
            return None

        momentum = self._configs[config_key]

        # Get base signals from unified computer
        current_games = momentum.elo_history[-1].games_played if momentum.elo_history else 0
        signals = self._signal_computer.compute_signals(
            current_games=current_games,
            current_elo=momentum.current_elo,
            config_key=config_key,
        )

        return signals

    def get_status_summary(self) -> dict[str, Any]:
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

    def get_metrics_dict(self) -> dict[str, Any]:
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
    # Elo Plateau Detection (December 2025)
    # =========================================================================

    def _check_elo_plateau(self, config_key: str, momentum: ConfigMomentum) -> None:
        """Check for Elo plateau and trigger response if detected.

        This integrates the detect_elo_plateau() function from adaptive_controller
        with the FeedbackAccelerator to automatically respond to training stalls.

        When a plateau is detected:
        1. Signal training needed with high urgency
        2. Emit PLATEAU_DETECTED event via on_plateau_detected()

        Args:
            config_key: Configuration key (e.g., "square8_2p")
            momentum: ConfigMomentum with Elo history
        """
        # Need at least 5 data points to detect plateau
        if len(momentum.elo_history) < 5:
            return

        try:
            from app.training.adaptive_controller import detect_elo_plateau, on_plateau_detected

            # Extract Elo values from history
            elo_values = [snapshot.elo for snapshot in momentum.elo_history]

            # Check for plateau
            is_plateau, details = detect_elo_plateau(
                elo_history=elo_values,
                window_size=10,
                threshold_elo_per_game=0.5,
            )

            if is_plateau and details.get("confidence", 0) > 0.5:
                logger.info(
                    f"[FeedbackAccelerator] Elo plateau detected for {config_key}: "
                    f"slope={details.get('slope', 0):.2f}, confidence={details.get('confidence', 0):.2f}"
                )

                # Trigger async response in background
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(on_plateau_detected(config_key, details))
                except RuntimeError:
                    # No event loop running - run synchronously
                    asyncio.run(on_plateau_detected(config_key, details))

        except ImportError:
            pass  # adaptive_controller not available
        except Exception as e:
            logger.debug(f"[FeedbackAccelerator] Plateau detection failed for {config_key}: {e}")

    # =========================================================================
    # Training Callbacks
    # =========================================================================

    def register_training_callback(
        self,
        callback: Callable[[str, TrainingDecision], None],
    ) -> None:
        """Register a callback to be called when training should be triggered."""
        self._training_callbacks.append(callback)

    def check_and_trigger_training(self) -> list[TrainingDecision]:
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

_accelerator: FeedbackAccelerator | None = None


def get_feedback_accelerator() -> FeedbackAccelerator:
    """Get the singleton feedback accelerator."""
    global _accelerator
    if _accelerator is None:
        _accelerator = FeedbackAccelerator()
    return _accelerator


def should_trigger_training(config_key: str) -> bool:
    """Check if training should be triggered for a config."""
    return get_feedback_accelerator().should_trigger_training(config_key)


def get_training_intensity(config_key: str) -> dict[str, float]:
    """Get training intensity parameters for a config."""
    return get_feedback_accelerator().get_training_intensity(config_key)


def get_selfplay_multiplier(config_key: str) -> float:
    """Get selfplay games multiplier based on Elo momentum (December 2025).

    This implements the Elo momentum → Selfplay rate coupling.
    Call this when determining how many selfplay games to generate.

    Args:
        config_key: Config identifier (e.g., "square8_2p")

    Returns:
        Multiplier for selfplay games (0.5 - 1.5)
    """
    return get_feedback_accelerator().get_selfplay_multiplier(config_key)


def record_elo_update(
    config_key: str,
    new_elo: float,
    games_played: int,
    model_id: str | None = None,
) -> None:
    """Record an Elo update."""
    get_feedback_accelerator().record_elo_update(config_key, new_elo, games_played, model_id)


def record_games_generated(config_key: str, games: int) -> None:
    """Record games generated for a config."""
    get_feedback_accelerator().record_games_generated(config_key, games)


def record_training_complete(
    config_key: str,
    success: bool = True,
    loss_improved: bool = False,
    games_used: int = 0,
    new_elo: float | None = None,
) -> None:
    """Record that training completed for a config.

    Args:
        config_key: Configuration identifier (e.g., 'square8_2p')
        success: Whether training completed successfully
        loss_improved: Whether the loss improved during training
        games_used: Number of games used for training
        new_elo: Optional new Elo rating after evaluation
    """
    # Map loss_improved to success if success wasn't explicitly set
    effective_success = success and (loss_improved if loss_improved is not None else True)
    get_feedback_accelerator().record_training_complete(
        config_key,
        success=effective_success,
        new_elo=new_elo,
        games_at_training=games_used if games_used > 0 else None,
    )


def record_promotion(config_key: str, new_elo: float, model_id: str | None = None) -> None:
    """Record a successful model promotion."""
    get_feedback_accelerator().record_promotion(config_key, new_elo, model_id)


def get_curriculum_weights() -> dict[str, float]:
    """Get recommended curriculum weights based on momentum."""
    return get_feedback_accelerator().get_curriculum_weights()


def get_selfplay_rate_recommendation(config_key: str) -> float:
    """Get recommended selfplay rate multiplier for a config."""
    return get_feedback_accelerator().get_selfplay_rate_recommendation(config_key)


def get_aggregate_selfplay_recommendation() -> dict[str, Any]:
    """Get aggregate selfplay rate recommendation across all configs.

    Returns:
        Dict with 'recommended_multiplier', 'reason', 'aggregate_momentum', 'states'
    """
    return get_feedback_accelerator().get_aggregate_selfplay_recommendation()


# =============================================================================
# Event Bus Integration (December 2025)
# =============================================================================

_evaluation_watcher_active = False


def _on_evaluation_completed(event) -> None:
    """Handle EVALUATION_COMPLETED event to update momentum tracking.

    This connects evaluation results to selfplay rate recommendations.
    """
    payload = event.payload if hasattr(event, 'payload') else {}

    config = payload.get("config", "")
    new_elo = payload.get("new_elo") or payload.get("elo")
    games_played = payload.get("games_played", 0)
    model_id = payload.get("model_id")

    if not config or new_elo is None:
        return

    try:
        record_elo_update(config, float(new_elo), games_played, model_id)
        logger.debug(f"[FeedbackAccelerator] Updated momentum for {config}: elo={new_elo}")
    except Exception as e:
        logger.warning(f"[FeedbackAccelerator] Failed to update momentum from evaluation: {e}")


def wire_evaluation_to_feedback() -> bool:
    """Subscribe to EVALUATION_COMPLETED events to update momentum tracking.

    This enables the feedback loop:
    EVALUATION_COMPLETED → momentum update → selfplay rate recommendation

    Returns:
        True if successfully subscribed
    """
    global _evaluation_watcher_active

    if _evaluation_watcher_active:
        return True

    try:
        from app.coordination.event_router import get_router
        from app.coordination.event_router import DataEventType

        router = get_router()
        router.subscribe(DataEventType.EVALUATION_COMPLETED.value, _on_evaluation_completed)
        _evaluation_watcher_active = True
        logger.info("[FeedbackAccelerator] Subscribed to EVALUATION_COMPLETED events")
        return True
    except Exception as e:
        logger.warning(f"[FeedbackAccelerator] Failed to subscribe to evaluation events: {e}")
        return False


def unwire_evaluation_from_feedback() -> None:
    """Unsubscribe from EVALUATION_COMPLETED events."""
    global _evaluation_watcher_active

    if not _evaluation_watcher_active:
        return

    try:
        from app.coordination.event_router import get_router
        from app.coordination.event_router import DataEventType

        router = get_router()
        router.unsubscribe(DataEventType.EVALUATION_COMPLETED.value, _on_evaluation_completed)
        _evaluation_watcher_active = False
        logger.info("[FeedbackAccelerator] Unsubscribed from EVALUATION_COMPLETED events")
    except Exception:
        pass


# =============================================================================
# Phase 5: HYPERPARAMETER_UPDATED event wiring (December 2025)
# =============================================================================

_hyperparameter_watcher_active = False


def _on_hyperparameter_updated(event) -> None:
    """Handle HYPERPARAMETER_UPDATED events from GauntletFeedbackController.

    This closes the feedback loop by applying gauntlet-derived adjustments
    to the FeedbackAccelerator's training recommendations.

    Args:
        event: Event containing hyperparameter adjustment recommendations
    """
    try:
        payload = event.payload if hasattr(event, "payload") else {}

        config_key = payload.get("config_key", "")
        if not config_key:
            return

        # Extract adjustment parameters
        lr_multiplier = float(payload.get("learning_rate_multiplier", 1.0))
        batch_multiplier = float(payload.get("batch_size_multiplier", 1.0))
        epochs_multiplier = float(payload.get("epochs_multiplier", 1.0))
        reason = payload.get("reason", "gauntlet_feedback")

        # Get the feedback accelerator singleton
        accelerator = get_feedback_accelerator()

        # Apply adjustments by recording them in the training decisions table
        with accelerator._db_lock, accelerator._get_connection() as conn:
            import time
            conn.execute(
                """
                INSERT INTO training_decisions
                (config_key, should_train, intensity, epochs_multiplier, lr_multiplier, reason, momentum_state, timestamp)
                VALUES (?, 1, ?, ?, ?, ?, 'feedback_applied', ?)
                """,
                (
                    config_key,
                    "adjusted" if lr_multiplier != 1.0 else "normal",
                    epochs_multiplier,
                    lr_multiplier,
                    reason,
                    time.time(),
                ),
            )

        logger.info(
            f"[FeedbackAccelerator] Applied hyperparameter update for {config_key}: "
            f"LR×{lr_multiplier:.2f}, epochs×{epochs_multiplier:.2f}"
        )

    except Exception as e:
        logger.warning(f"[FeedbackAccelerator] Failed to apply hyperparameter update: {e}")


def wire_hyperparameter_feedback() -> bool:
    """Subscribe to HYPERPARAMETER_UPDATED events for closed-loop feedback.

    This enables gauntlet evaluation results to influence training parameters:
    EVALUATION_COMPLETED → GauntletFeedbackController → HYPERPARAMETER_UPDATED → here

    Returns:
        True if successfully subscribed
    """
    global _hyperparameter_watcher_active

    if _hyperparameter_watcher_active:
        return True

    try:
        from app.coordination.event_router import get_router
        from app.distributed.data_events import DataEventType

        router = get_router()
        router.subscribe(DataEventType.HYPERPARAMETER_UPDATED.value, _on_hyperparameter_updated)
        _hyperparameter_watcher_active = True
        logger.info("[FeedbackAccelerator] Subscribed to HYPERPARAMETER_UPDATED events (Phase 5)")
        return True
    except Exception as e:
        logger.warning(f"[FeedbackAccelerator] Failed to subscribe to hyperparameter events: {e}")
        return False


def unwire_hyperparameter_feedback() -> None:
    """Unsubscribe from HYPERPARAMETER_UPDATED events."""
    global _hyperparameter_watcher_active

    if not _hyperparameter_watcher_active:
        return

    try:
        from app.coordination.event_router import get_router
        from app.distributed.data_events import DataEventType

        router = get_router()
        router.unsubscribe(DataEventType.HYPERPARAMETER_UPDATED.value, _on_hyperparameter_updated)
        _hyperparameter_watcher_active = False
        logger.info("[FeedbackAccelerator] Unsubscribed from HYPERPARAMETER_UPDATED events")
    except Exception:
        pass
