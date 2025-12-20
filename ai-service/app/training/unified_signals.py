"""
Unified Training Signal Computation

Single source of truth for all training-related metrics and decisions.
Other systems should subscribe to these signals rather than computing their own.

This module consolidates signal computation from:
- TrainingTriggers (training_triggers.py)
- FeedbackAccelerator (feedback_accelerator.py)
- ModelLifecycleManager (model_lifecycle.py)
- PromotionController (promotion_controller.py)
- OptimizedPipeline (optimized_pipeline.py)

Usage:
    from app.training.unified_signals import get_signal_computer, TrainingUrgency

    # Get computed signals
    computer = get_signal_computer()
    signals = computer.compute_signals(current_games=10000, current_elo=1650.0)

    if signals.should_train:
        print(f"Training triggered: {signals.reason}")
        print(f"Urgency: {signals.urgency.value}")

    # Record training events
    computer.record_training_started(games_count=10000)
    # ... training runs ...
    computer.record_training_completed(new_elo=1680.0)
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# Import canonical thresholds
try:
    from app.config.thresholds import (
        INITIAL_ELO_RATING,
        MIN_WIN_RATE_PROMOTE,
        TRAINING_MIN_INTERVAL_SECONDS,
        TRAINING_STALENESS_HOURS,
        TRAINING_TRIGGER_GAMES,
    )
except ImportError:
    INITIAL_ELO_RATING = 1500.0
    TRAINING_TRIGGER_GAMES = 500
    TRAINING_STALENESS_HOURS = 6
    MIN_WIN_RATE_PROMOTE = 0.45
    TRAINING_MIN_INTERVAL_SECONDS = 1200

# Import quality config for quality-aware training decisions
try:
    from app.config.unified_config import QualityConfig, get_config
    HAS_QUALITY_CONFIG = True
except ImportError:
    HAS_QUALITY_CONFIG = False
    QualityConfig = None

# Import centralized quality thresholds
try:
    from app.quality.thresholds import (
        HIGH_QUALITY_THRESHOLD,
        MIN_QUALITY_FOR_TRAINING,
    )
except ImportError:
    MIN_QUALITY_FOR_TRAINING = 0.3
    HIGH_QUALITY_THRESHOLD = 0.7

# Import data manifest for quality scores
try:
    from pathlib import Path

    from app.distributed.unified_manifest import DataManifest
    HAS_DATA_MANIFEST = True
except ImportError:
    HAS_DATA_MANIFEST = False
    DataManifest = None


class TrainingUrgency(Enum):
    """Training urgency levels.

    Used by all training decision systems to communicate priority.
    Maps to FeedbackAccelerator intensity levels:
    - CRITICAL -> HOT_PATH (2.0x intensity)
    - HIGH -> ACCELERATED (1.5x intensity)
    - NORMAL -> NORMAL (1.0x intensity)
    - LOW -> REDUCED (0.75x intensity)
    - NONE -> PAUSED (0.5x intensity)
    """
    CRITICAL = "critical"      # Train immediately (regression detected)
    HIGH = "high"              # Train soon (threshold exceeded significantly)
    NORMAL = "normal"          # Train when convenient (threshold met)
    LOW = "low"                # Can wait (threshold approaching)
    NONE = "none"              # No training needed


@dataclass
class TrainingSignals:
    """Computed training signals - read-only snapshot.

    This dataclass contains all computed training metrics from a single
    point in time. It's immutable after creation and safe to pass around.
    """
    # Core metrics
    games_since_last_training: int = 0
    time_since_last_training: timedelta = field(default_factory=lambda: timedelta(0))
    current_elo: float = INITIAL_ELO_RATING
    elo_trend: float = 0.0  # positive = improving, negative = regressing (Elo/hour)

    # Thresholds (from unified config)
    games_threshold: int = TRAINING_TRIGGER_GAMES
    min_interval: timedelta = field(
        default_factory=lambda: timedelta(seconds=TRAINING_MIN_INTERVAL_SECONDS)
    )

    # Derived signals
    games_threshold_ratio: float = 0.0  # games / threshold (>1 = exceeded)
    time_threshold_met: bool = False
    data_quality_score: float = 1.0  # 0-1, lower = more issues
    staleness_hours: float = 0.0
    staleness_ratio: float = 0.0  # time / staleness_threshold

    # Win rate tracking (from TrainingTriggers)
    win_rate: float = 0.5
    win_rate_trend: float = 0.0
    win_rate_regression: bool = False

    # Regression indicators
    elo_regression_detected: bool = False
    elo_drop_magnitude: float = 0.0
    consecutive_losses: int = 0

    # Resource availability
    gpu_available: bool = True
    memory_pressure: float = 0.0  # 0-1

    # Config-specific (for multi-config tracking)
    config_key: str = ""
    model_count: int = 0
    is_bootstrap: bool = False  # New config with no models

    # Final recommendation
    urgency: TrainingUrgency = TrainingUrgency.NONE
    should_train: bool = False
    reason: str = ""
    priority: float = 0.0  # Higher = more urgent

    # Timestamp
    computed_at: datetime = field(default_factory=datetime.now)

    @property
    def summary(self) -> str:
        """Human-readable summary of signals."""
        return (f"Urgency={self.urgency.value}, "
                f"games={self.games_since_last_training}/{self.games_threshold}, "
                f"elo_trend={self.elo_trend:+.1f}/hr, "
                f"quality={self.data_quality_score:.2f}, "
                f"priority={self.priority:.2f}, "
                f"reason={self.reason}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "games_since_last_training": self.games_since_last_training,
            "time_since_last_training_seconds": self.time_since_last_training.total_seconds(),
            "current_elo": self.current_elo,
            "elo_trend": self.elo_trend,
            "games_threshold": self.games_threshold,
            "games_threshold_ratio": self.games_threshold_ratio,
            "time_threshold_met": self.time_threshold_met,
            "staleness_hours": self.staleness_hours,
            "data_quality_score": self.data_quality_score,
            "win_rate": self.win_rate,
            "win_rate_regression": self.win_rate_regression,
            "elo_regression_detected": self.elo_regression_detected,
            "elo_drop_magnitude": self.elo_drop_magnitude,
            "config_key": self.config_key,
            "is_bootstrap": self.is_bootstrap,
            "urgency": self.urgency.value,
            "should_train": self.should_train,
            "reason": self.reason,
            "priority": self.priority,
            "computed_at": self.computed_at.isoformat(),
        }


@dataclass
class ConfigTrainingState:
    """Per-config training state tracking."""
    config_key: str
    last_training_time: datetime | None = None
    last_training_games: int = 0
    model_count: int = 0
    current_elo: float = INITIAL_ELO_RATING
    win_rate: float = 0.5
    elo_history: list[tuple[datetime, float]] = field(default_factory=list)
    win_rate_history: list[tuple[datetime, float]] = field(default_factory=list)


class UnifiedSignalComputer:
    """
    Central computation engine for training signals.

    All training decision systems should use this instead of
    computing their own metrics. This ensures consistent decisions
    across the system.

    Thread-safe with internal locking and short-term caching.
    """

    def __init__(self, config=None):
        """Initialize the signal computer.

        Args:
            config: Optional config object. If not provided, loads from
                   app.config.unified_config.get_config()
        """
        self._config = config
        self._lock = threading.RLock()
        self._last_computation: TrainingSignals | None = None
        self._computation_cache_ttl = timedelta(seconds=5)
        self._last_computation_time: datetime | None = None

        # Per-config state tracking
        self._config_states: dict[str, ConfigTrainingState] = {}

        # Global state (for backward compatibility with single-config usage)
        self._last_training_time: datetime | None = None
        self._last_training_games: int = 0
        self._elo_history: list[tuple[datetime, float]] = []
        self._win_rate_history: list[tuple[datetime, float]] = []

        # Configuration thresholds (loaded lazily)
        self._games_threshold: int | None = None
        self._min_interval_seconds: int | None = None
        self._staleness_hours: float | None = None
        self._min_win_rate: float | None = None

        # Quality-aware training (December 2025)
        self._manifest: DataManifest | None = None
        self._quality_config: QualityConfig | None = None

    def _get_manifest(self):
        """Lazily load data manifest for quality scores."""
        if self._manifest is None and HAS_DATA_MANIFEST:
            try:
                # Get manifest path from config or use default
                config = self._get_config()
                if config and hasattr(config, 'data_manifest_db'):
                    manifest_path = Path(__file__).parent.parent.parent / config.data_manifest_db
                else:
                    manifest_path = Path(__file__).parent.parent.parent / "data" / "data_manifest.db"

                if manifest_path.exists():
                    self._manifest = DataManifest(manifest_path)
                    logger.debug(f"Loaded data manifest from {manifest_path}")
            except Exception as e:
                logger.debug(f"Could not load data manifest: {e}")
        return self._manifest

    def _get_quality_config(self):
        """Get quality config from unified config."""
        if self._quality_config is None and HAS_QUALITY_CONFIG:
            try:
                config = self._get_config()
                if config and hasattr(config, 'quality'):
                    self._quality_config = config.quality
            except Exception as e:
                logger.debug(f"Could not load quality config: {e}")
        return self._quality_config

    def get_data_quality_score(
        self,
        config_key: str = "",
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> float:
        """Get average quality score, preferring event-driven cache.

        Priority order:
        1. Cached quality from quality events (if fresh)
        2. Quality from data manifest
        3. Default 1.0

        Args:
            config_key: Optional config key (e.g., "square8_2p")
            board_type: Optional board type filter
            num_players: Optional player count filter

        Returns:
            Average quality score (0-1), or 1.0 if unavailable
        """
        # First check event-driven cache (fresher, more responsive)
        if config_key:
            cached = self.get_cached_quality(config_key)
            if cached is not None:
                return cached

        # Fall back to manifest
        manifest = self._get_manifest()
        if not manifest:
            return 1.0  # Default to 1.0 (no quality filtering)

        try:
            # Parse config_key if provided
            if config_key and not board_type:
                parts = config_key.replace("_", " ").replace("p", "").split()
                if len(parts) >= 1:
                    board_type = parts[0]
                if len(parts) >= 2 and parts[1].isdigit():
                    num_players = int(parts[1])

            distribution = manifest.get_quality_distribution(
                board_type=board_type,
                num_players=num_players,
            )
            avg_quality = distribution.get("avg_quality_score", 1.0)
            return avg_quality if avg_quality > 0 else 1.0
        except Exception as e:
            logger.debug(f"Could not get quality distribution: {e}")
            return 1.0

    def _get_config(self):
        """Lazily load config."""
        if self._config is None:
            try:
                from app.config.unified_config import get_config
                self._config = get_config()
            except ImportError:
                logger.warning("Could not import unified_config, using defaults")
                self._config = None
        return self._config

    @property
    def games_threshold(self) -> int:
        """Get games threshold from config or default."""
        if self._games_threshold is None:
            config = self._get_config()
            if config and hasattr(config, 'training'):
                self._games_threshold = getattr(
                    config.training, 'trigger_threshold_games', TRAINING_TRIGGER_GAMES
                )
            else:
                self._games_threshold = TRAINING_TRIGGER_GAMES
        return self._games_threshold

    @property
    def min_interval_seconds(self) -> int:
        """Get minimum interval from config or default."""
        if self._min_interval_seconds is None:
            config = self._get_config()
            if config and hasattr(config, 'training'):
                self._min_interval_seconds = getattr(
                    config.training, 'min_interval_seconds', TRAINING_MIN_INTERVAL_SECONDS
                )
            else:
                self._min_interval_seconds = TRAINING_MIN_INTERVAL_SECONDS
        return self._min_interval_seconds

    @property
    def staleness_hours(self) -> float:
        """Get staleness threshold from config or default."""
        if self._staleness_hours is None:
            config = self._get_config()
            if config and hasattr(config, 'training'):
                self._staleness_hours = getattr(
                    config.training, 'staleness_hours', TRAINING_STALENESS_HOURS
                )
            else:
                self._staleness_hours = TRAINING_STALENESS_HOURS
        return self._staleness_hours

    @property
    def min_win_rate(self) -> float:
        """Get minimum win rate from config or default."""
        if self._min_win_rate is None:
            config = self._get_config()
            if config and hasattr(config, 'promotion'):
                self._min_win_rate = getattr(
                    config.promotion, 'min_win_rate', MIN_WIN_RATE_PROMOTE
                )
            else:
                self._min_win_rate = MIN_WIN_RATE_PROMOTE
        return self._min_win_rate

    def _get_config_state(self, config_key: str) -> ConfigTrainingState:
        """Get or create per-config state."""
        if config_key not in self._config_states:
            self._config_states[config_key] = ConfigTrainingState(config_key=config_key)
        return self._config_states[config_key]

    def compute_signals(
        self,
        current_games: int,
        current_elo: float,
        config_key: str = "",
        win_rate: float | None = None,
        model_count: int | None = None,
        force_recompute: bool = False,
    ) -> TrainingSignals:
        """
        Compute current training signals.

        Args:
            current_games: Total games in training corpus
            current_elo: Current model Elo rating
            config_key: Optional config identifier for multi-config tracking
            win_rate: Optional current win rate
            model_count: Optional number of models for this config
            force_recompute: Bypass cache

        Returns:
            TrainingSignals snapshot with all computed metrics
        """
        with self._lock:
            now = datetime.now()

            # Use per-config state if config_key provided
            if config_key:
                state = self._get_config_state(config_key)
                last_training_time = state.last_training_time
                last_training_games = state.last_training_games
                elo_history = state.elo_history
                config_model_count = state.model_count if model_count is None else model_count
            else:
                last_training_time = self._last_training_time
                last_training_games = self._last_training_games
                elo_history = self._elo_history
                config_model_count = model_count or 0

            # Check cache for global (non-config-specific) calls
            if (not config_key and not force_recompute and
                self._last_computation_time and
                now - self._last_computation_time < self._computation_cache_ttl):
                return self._last_computation

            signals = TrainingSignals()
            signals.config_key = config_key
            signals.computed_at = now

            # Core metrics
            signals.games_since_last_training = current_games - last_training_games
            signals.games_threshold = self.games_threshold
            signals.games_threshold_ratio = (
                signals.games_since_last_training / signals.games_threshold
                if signals.games_threshold > 0 else 0
            )

            # Time since training
            if last_training_time:
                signals.time_since_last_training = now - last_training_time
                signals.min_interval = timedelta(seconds=self.min_interval_seconds)
                signals.time_threshold_met = (
                    signals.time_since_last_training >= signals.min_interval
                )
                signals.staleness_hours = signals.time_since_last_training.total_seconds() / 3600
                signals.staleness_ratio = signals.staleness_hours / self.staleness_hours
            else:
                signals.time_threshold_met = True  # First training
                signals.staleness_hours = float('inf')
                signals.staleness_ratio = 2.0  # Treat as stale

            # Model count and bootstrap detection
            signals.model_count = config_model_count
            signals.is_bootstrap = (
                config_model_count == 0 and
                signals.games_since_last_training >= 50  # Bootstrap threshold
            )

            # Elo tracking
            signals.current_elo = current_elo
            elo_history.append((now, current_elo))

            # Keep last hour, max 100 points
            cutoff = now - timedelta(hours=1)
            elo_history[:] = [
                (t, e) for t, e in elo_history
                if t > cutoff
            ][-100:]

            if len(elo_history) >= 2:
                signals.elo_trend = self._compute_elo_trend(elo_history)
                signals.elo_regression_detected = signals.elo_trend < -10
                signals.elo_drop_magnitude = abs(min(0, signals.elo_trend))

            # Win rate tracking
            if win_rate is not None:
                signals.win_rate = win_rate
                win_history = (
                    state.win_rate_history if config_key else self._win_rate_history
                )
                win_history.append((now, win_rate))
                win_history[:] = [
                    (t, w) for t, w in win_history
                    if t > cutoff
                ][-50:]

                if len(win_history) >= 2:
                    signals.win_rate_trend = win_history[-1][1] - win_history[0][1]

                signals.win_rate_regression = win_rate < self.min_win_rate

            # Get data quality score from manifest
            signals.data_quality_score = self.get_data_quality_score(
                config_key=config_key
            )

            # Compute urgency and final recommendation
            signals.urgency, signals.reason = self._compute_urgency(signals)
            signals.priority = self._compute_priority(signals)
            signals.should_train = (
                signals.urgency in (
                    TrainingUrgency.CRITICAL,
                    TrainingUrgency.HIGH,
                    TrainingUrgency.NORMAL
                ) and signals.time_threshold_met
            )

            # Cache for global calls
            if not config_key:
                self._last_computation = signals
                self._last_computation_time = now

            logger.debug(f"Computed signals: {signals.summary}")
            return signals

    def _compute_elo_trend(self, elo_history: list[tuple[datetime, float]]) -> float:
        """Compute Elo trend via linear regression.

        Returns Elo change per hour (positive = improving).
        """
        if len(elo_history) < 2:
            return 0.0

        # Convert to relative seconds and Elo values
        base_time = elo_history[0][0]
        times = [(t - base_time).total_seconds() for t, _ in elo_history]
        elos = [e for _, e in elo_history]

        n = len(times)
        sum_t = sum(times)
        sum_e = sum(elos)
        sum_te = sum(t * e for t, e in zip(times, elos, strict=False))
        sum_t2 = sum(t * t for t in times)

        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-10:
            return 0.0

        slope = (n * sum_te - sum_t * sum_e) / denom
        # Convert to Elo per hour
        return slope * 3600

    def _compute_urgency(self, signals: TrainingSignals) -> tuple[TrainingUrgency, str]:
        """Determine training urgency and reason.

        Priority order:
        1. Bootstrap (new config with no models)
        2. Critical regression
        3. High (significantly over threshold)
        4. Normal (threshold met)
        5. Low (approaching threshold)
        6. None

        Quality modifiers:
        - Low quality data (< min_quality_for_training) defers urgency
        - High quality data (> high_quality_threshold) accelerates urgency
        """
        # Get quality thresholds from centralized module
        min_quality = MIN_QUALITY_FOR_TRAINING

        # Check if data quality is too low - defer training
        if signals.data_quality_score < min_quality and not signals.elo_regression_detected:
            return (
                TrainingUrgency.LOW,
                f"Data quality too low ({signals.data_quality_score:.2f} < {min_quality}), deferring"
            )

        # Bootstrap takes highest priority
        if signals.is_bootstrap:
            return (
                TrainingUrgency.HIGH,
                f"Bootstrap: config has {signals.model_count} models and "
                f"{signals.games_since_last_training} games"
            )

        # Critical: Severe regression detected
        if signals.elo_regression_detected and signals.elo_drop_magnitude > 30:
            return (
                TrainingUrgency.CRITICAL,
                f"Elo regression: {signals.elo_drop_magnitude:.0f} drop/hr"
            )

        # Critical: Win rate below minimum
        if signals.win_rate_regression:
            return (
                TrainingUrgency.CRITICAL,
                f"Win rate regression: {signals.win_rate:.1%} < {self.min_win_rate:.1%}"
            )

        # High: Significantly over games threshold
        if signals.games_threshold_ratio >= 1.5:
            return (
                TrainingUrgency.HIGH,
                f"Games {signals.games_threshold_ratio:.1f}x threshold"
            )

        # High: Very stale model with some data
        if signals.staleness_ratio >= 1.5 and signals.games_threshold_ratio >= 0.3:
            return (
                TrainingUrgency.HIGH,
                f"Model stale ({signals.staleness_hours:.1f}h) with "
                f"{signals.games_since_last_training} games"
            )

        # Normal: Games threshold met
        if signals.games_threshold_ratio >= 1.0:
            return (
                TrainingUrgency.NORMAL,
                f"Games threshold met ({signals.games_since_last_training})"
            )

        # Normal: Stale with reasonable data
        if signals.staleness_ratio >= 1.0 and signals.games_threshold_ratio >= 0.5:
            return (
                TrainingUrgency.NORMAL,
                f"Model stale ({signals.staleness_hours:.1f}h)"
            )

        # Low: Approaching threshold
        if signals.games_threshold_ratio >= 0.8:
            return (
                TrainingUrgency.LOW,
                f"Approaching threshold ({signals.games_threshold_ratio:.0%})"
            )

        # Low: Mild regression trend
        if signals.elo_regression_detected:
            return (
                TrainingUrgency.LOW,
                f"Mild Elo decline: {signals.elo_trend:.1f}/hr"
            )

        return TrainingUrgency.NONE, "Below threshold"

    def _compute_priority(self, signals: TrainingSignals) -> float:
        """Compute numeric priority for queue ordering.

        Higher values = more urgent.
        """
        base_priority = {
            TrainingUrgency.CRITICAL: 100.0,
            TrainingUrgency.HIGH: 50.0,
            TrainingUrgency.NORMAL: 20.0,
            TrainingUrgency.LOW: 5.0,
            TrainingUrgency.NONE: 0.0,
        }.get(signals.urgency, 0.0)

        # Add modifiers
        priority = base_priority

        # Bootstrap bonus
        if signals.is_bootstrap:
            priority += 25.0

        # Games over threshold bonus
        if signals.games_threshold_ratio > 1.0:
            priority += (signals.games_threshold_ratio - 1.0) * 10.0

        # Regression bonus
        if signals.elo_regression_detected:
            priority += signals.elo_drop_magnitude * 0.5

        if signals.win_rate_regression:
            priority += (self.min_win_rate - signals.win_rate) * 50.0

        # Quality modifiers - use centralized thresholds
        high_quality = HIGH_QUALITY_THRESHOLD
        min_quality = MIN_QUALITY_FOR_TRAINING

        # High quality data gets priority boost
        if signals.data_quality_score >= high_quality:
            quality_boost = (signals.data_quality_score - high_quality) * 30.0
            priority += quality_boost

        # Low quality data gets priority penalty (unless critical)
        elif signals.data_quality_score < min_quality:
            if signals.urgency not in (TrainingUrgency.CRITICAL,):
                quality_penalty = (min_quality - signals.data_quality_score) * 20.0
                priority -= quality_penalty

        return max(0.0, priority)

    def record_training_started(self, games_count: int, config_key: str = "") -> None:
        """Record that training has started.

        Args:
            games_count: Current total games at training start
            config_key: Optional config identifier
        """
        with self._lock:
            now = datetime.now()
            if config_key:
                state = self._get_config_state(config_key)
                state.last_training_time = now
                state.last_training_games = games_count
            else:
                self._last_training_time = now
                self._last_training_games = games_count

            logger.info(
                f"Training started: config={config_key or 'global'}, "
                f"games={games_count}"
            )

    def record_training_completed(
        self,
        new_elo: float | None = None,
        config_key: str = "",
        model_count: int | None = None,
    ) -> None:
        """Record that training has completed.

        Args:
            new_elo: Optional new Elo rating after training
            config_key: Optional config identifier
            model_count: Optional updated model count
        """
        with self._lock:
            now = datetime.now()

            if config_key:
                state = self._get_config_state(config_key)
                if new_elo is not None:
                    state.elo_history.append((now, new_elo))
                    state.current_elo = new_elo
                if model_count is not None:
                    state.model_count = model_count
            else:
                if new_elo is not None:
                    self._elo_history.append((now, new_elo))

            logger.info(
                f"Training completed: config={config_key or 'global'}, "
                f"new_elo={new_elo}"
            )

    def update_config_state(
        self,
        config_key: str,
        model_count: int | None = None,
        current_elo: float | None = None,
        win_rate: float | None = None,
    ) -> None:
        """Update state for a specific config.

        Args:
            config_key: Config identifier
            model_count: Optional model count update
            current_elo: Optional Elo update
            win_rate: Optional win rate update
        """
        with self._lock:
            state = self._get_config_state(config_key)
            now = datetime.now()

            if model_count is not None:
                state.model_count = model_count
            if current_elo is not None:
                state.current_elo = current_elo
                state.elo_history.append((now, current_elo))
            if win_rate is not None:
                state.win_rate = win_rate
                state.win_rate_history.append((now, win_rate))

    def update_data_quality(self, config_key: str, quality_score: float) -> None:
        """Update data quality score for a config from quality events.

        This method is called by TrainingTriggers when it receives quality
        events (HIGH_QUALITY_DATA_AVAILABLE, LOW_QUALITY_DATA_WARNING, etc.).
        The score is cached for use in subsequent compute_signals() calls.

        Args:
            config_key: Config identifier
            quality_score: Quality score from 0.0 to 1.0
        """
        with self._lock:
            # Store in per-config quality cache
            if not hasattr(self, '_quality_cache'):
                self._quality_cache: dict[str, tuple[float, datetime]] = {}

            self._quality_cache[config_key] = (quality_score, datetime.now())
            logger.debug(
                f"Updated quality score for {config_key}: {quality_score:.3f}"
            )

    def get_cached_quality(self, config_key: str, max_age_seconds: float = 300.0) -> float | None:
        """Get cached quality score if available and fresh.

        Args:
            config_key: Config identifier
            max_age_seconds: Maximum age for cache validity (default 5 min)

        Returns:
            Cached quality score, or None if not available or stale
        """
        if not hasattr(self, '_quality_cache'):
            return None

        if config_key not in self._quality_cache:
            return None

        score, cached_at = self._quality_cache[config_key]
        age = (datetime.now() - cached_at).total_seconds()

        if age > max_age_seconds:
            return None

        return score

    def get_all_config_signals(
        self,
        games_by_config: dict[str, int],
        elo_by_config: dict[str, float],
    ) -> list[TrainingSignals]:
        """Compute signals for all configs.

        Args:
            games_by_config: Map of config_key -> current games
            elo_by_config: Map of config_key -> current Elo

        Returns:
            List of TrainingSignals sorted by priority (highest first)
        """
        signals_list = []
        for config_key in games_by_config:
            signals = self.compute_signals(
                current_games=games_by_config[config_key],
                current_elo=elo_by_config.get(config_key, INITIAL_ELO_RATING),
                config_key=config_key,
            )
            signals_list.append(signals)

        # Sort by priority (highest first)
        signals_list.sort(key=lambda s: s.priority, reverse=True)
        return signals_list

    def get_training_queue(
        self,
        games_by_config: dict[str, int],
        elo_by_config: dict[str, float],
    ) -> list[TrainingSignals]:
        """Get configs that should train, sorted by priority.

        Args:
            games_by_config: Map of config_key -> current games
            elo_by_config: Map of config_key -> current Elo

        Returns:
            List of TrainingSignals for configs that should train
        """
        all_signals = self.get_all_config_signals(games_by_config, elo_by_config)
        return [s for s in all_signals if s.should_train]


# Singleton instance
_signal_computer: UnifiedSignalComputer | None = None
_signal_computer_lock = threading.Lock()


def get_signal_computer() -> UnifiedSignalComputer:
    """Get the singleton signal computer instance.

    Thread-safe lazy initialization.
    """
    global _signal_computer
    if _signal_computer is None:
        with _signal_computer_lock:
            if _signal_computer is None:
                _signal_computer = UnifiedSignalComputer()
    return _signal_computer


def reset_signal_computer() -> None:
    """Reset the singleton for testing purposes."""
    global _signal_computer
    with _signal_computer_lock:
        _signal_computer = None


# Convenience functions for backward compatibility with TrainingTriggers API
def should_train(
    config_key: str,
    current_games: int,
    current_elo: float = INITIAL_ELO_RATING,
    **kwargs,
) -> bool:
    """Quick check if training should run for a config.

    Args:
        config_key: Config identifier (e.g., "square8_2p")
        current_games: Total games in corpus
        current_elo: Current model Elo
        **kwargs: Additional state (win_rate, model_count)

    Returns:
        True if training should be triggered
    """
    computer = get_signal_computer()
    signals = computer.compute_signals(
        current_games=current_games,
        current_elo=current_elo,
        config_key=config_key,
        win_rate=kwargs.get('win_rate'),
        model_count=kwargs.get('model_count'),
    )
    return signals.should_train


def get_urgency(
    config_key: str,
    current_games: int,
    current_elo: float = INITIAL_ELO_RATING,
) -> TrainingUrgency:
    """Get current training urgency for a config.

    Args:
        config_key: Config identifier
        current_games: Total games in corpus
        current_elo: Current model Elo

    Returns:
        TrainingUrgency enum value
    """
    computer = get_signal_computer()
    signals = computer.compute_signals(
        current_games=current_games,
        current_elo=current_elo,
        config_key=config_key,
    )
    return signals.urgency


def get_training_intensity(
    config_key: str,
    current_games: int,
    current_elo: float = INITIAL_ELO_RATING,
) -> float:
    """Get training intensity multiplier (0.5 - 2.0).

    Maps urgency to FeedbackAccelerator intensity levels.

    Args:
        config_key: Config identifier
        current_games: Total games in corpus
        current_elo: Current model Elo

    Returns:
        Intensity multiplier (0.5 to 2.0)
    """
    computer = get_signal_computer()
    signals = computer.compute_signals(
        current_games=current_games,
        current_elo=current_elo,
        config_key=config_key,
    )

    # Map urgency to intensity
    base_intensity = {
        TrainingUrgency.CRITICAL: 2.0,
        TrainingUrgency.HIGH: 1.5,
        TrainingUrgency.NORMAL: 1.0,
        TrainingUrgency.LOW: 0.75,
        TrainingUrgency.NONE: 0.5,
    }.get(signals.urgency, 1.0)

    # Adjust for Elo trend
    if signals.elo_trend > 20:
        base_intensity *= 0.9  # Improving, can ease off
    elif signals.elo_trend < -20:
        base_intensity *= 1.2  # Regressing, push harder

    return max(0.5, min(2.0, base_intensity))


# ==============================================================================
# Prometheus Metrics Export
# ==============================================================================

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class SignalMetricsExporter:
    """Export training signals to Prometheus.

    Usage:
        exporter = SignalMetricsExporter()

        # In your training loop:
        signals = get_signal_computer().compute_signals(...)
        exporter.update(signals)
    """

    def __init__(self, namespace: str = "ringrift_training"):
        """Initialize Prometheus metrics.

        Args:
            namespace: Prefix for all metric names
        """
        self._initialized = False

        if not PROMETHEUS_AVAILABLE:
            logger.warning(
                "prometheus_client not installed, metrics will not be exported. "
                "Install with: pip install prometheus-client"
            )
            return

        # Gauge metrics (current values)
        self.games_since_training = Gauge(
            f"{namespace}_games_since_training",
            "Number of games since last training",
            ["config_key"]
        )

        self.current_elo = Gauge(
            f"{namespace}_current_elo",
            "Current model Elo rating",
            ["config_key"]
        )

        self.elo_trend = Gauge(
            f"{namespace}_elo_trend",
            "Elo change per hour (positive = improving)",
            ["config_key"]
        )

        self.training_urgency = Gauge(
            f"{namespace}_urgency",
            "Training urgency level (0=NONE, 1=LOW, 2=NORMAL, 3=HIGH, 4=CRITICAL)",
            ["config_key"]
        )

        self.training_priority = Gauge(
            f"{namespace}_priority",
            "Numeric training priority (higher = more urgent)",
            ["config_key"]
        )

        self.win_rate = Gauge(
            f"{namespace}_win_rate",
            "Current model win rate",
            ["config_key"]
        )

        self.model_count = Gauge(
            f"{namespace}_model_count",
            "Number of models for this config",
            ["config_key"]
        )

        self.staleness_hours = Gauge(
            f"{namespace}_staleness_hours",
            "Hours since last training",
            ["config_key"]
        )

        self.data_quality = Gauge(
            f"{namespace}_data_quality",
            "Average data quality score (0-1, higher = better)",
            ["config_key"]
        )

        # Counter metrics (cumulative)
        self.training_triggers = Counter(
            f"{namespace}_triggers_total",
            "Number of times training was triggered",
            ["config_key", "urgency"]
        )

        self.regression_events = Counter(
            f"{namespace}_regression_events_total",
            "Number of regression events detected",
            ["config_key", "type"]
        )

        # Histogram for signal computation time
        self.computation_time = Histogram(
            f"{namespace}_signal_computation_seconds",
            "Time to compute training signals",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )

        self._initialized = True
        logger.info(f"SignalMetricsExporter initialized with namespace '{namespace}'")

    def update(self, signals: TrainingSignals) -> None:
        """Update Prometheus metrics from signals.

        Args:
            signals: Computed TrainingSignals to export
        """
        if not self._initialized:
            return

        config_key = signals.config_key or "default"

        # Update gauges
        self.games_since_training.labels(config_key=config_key).set(
            signals.games_since_last_training
        )
        self.current_elo.labels(config_key=config_key).set(signals.current_elo)
        self.elo_trend.labels(config_key=config_key).set(signals.elo_trend)
        self.training_priority.labels(config_key=config_key).set(signals.priority)
        self.win_rate.labels(config_key=config_key).set(signals.win_rate)
        self.model_count.labels(config_key=config_key).set(signals.model_count)
        self.staleness_hours.labels(config_key=config_key).set(signals.staleness_hours)
        self.data_quality.labels(config_key=config_key).set(signals.data_quality_score)

        # Map urgency to numeric value for graphing
        urgency_value = {
            TrainingUrgency.NONE: 0,
            TrainingUrgency.LOW: 1,
            TrainingUrgency.NORMAL: 2,
            TrainingUrgency.HIGH: 3,
            TrainingUrgency.CRITICAL: 4,
        }.get(signals.urgency, 0)
        self.training_urgency.labels(config_key=config_key).set(urgency_value)

        # Track regression events
        if signals.elo_regression_detected:
            self.regression_events.labels(
                config_key=config_key,
                type="elo"
            ).inc()

        if signals.win_rate_regression:
            self.regression_events.labels(
                config_key=config_key,
                type="win_rate"
            ).inc()

    def record_training_triggered(
        self,
        config_key: str,
        urgency: TrainingUrgency
    ) -> None:
        """Record that training was triggered.

        Args:
            config_key: Config identifier
            urgency: Urgency level that triggered training
        """
        if not self._initialized:
            return

        self.training_triggers.labels(
            config_key=config_key or "default",
            urgency=urgency.value
        ).inc()

    def time_computation(self):
        """Context manager to time signal computation.

        Usage:
            with exporter.time_computation():
                signals = computer.compute_signals(...)
        """
        if not self._initialized:
            from contextlib import nullcontext
            return nullcontext()

        return self.computation_time.time()


# Singleton exporter instance
_metrics_exporter: SignalMetricsExporter | None = None
_metrics_exporter_lock = threading.Lock()


def get_metrics_exporter() -> SignalMetricsExporter:
    """Get the singleton metrics exporter instance."""
    global _metrics_exporter
    if _metrics_exporter is None:
        with _metrics_exporter_lock:
            if _metrics_exporter is None:
                _metrics_exporter = SignalMetricsExporter()
    return _metrics_exporter


def export_signals_to_prometheus(signals: TrainingSignals) -> None:
    """Convenience function to export signals to Prometheus.

    Args:
        signals: Computed TrainingSignals to export
    """
    get_metrics_exporter().update(signals)
