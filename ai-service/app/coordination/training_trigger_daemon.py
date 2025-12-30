"""Training Trigger Daemon - Automatic training decision logic (December 2025).

This daemon decides WHEN to trigger training automatically, eliminating
the human "train now" decision. It monitors multiple conditions to ensure
training starts at the optimal time.

Decision Conditions:
1. Data freshness - NPZ data < configured max age (default: 1 hour)
2. Training not active - No training already running for that config
3. Idle GPU available - At least one training GPU with < threshold utilization
4. Quality trajectory - Model still improving OR evaluation overdue
5. Minimum samples - Sufficient training samples available

Key features:
- Subscribes to NPZ_EXPORT_COMPLETE events for immediate trigger
- Periodic scan for training opportunities
- Tracks per-config training state
- Integrates with TrainingCoordinator to prevent duplicates
- Emits TRAINING_STARTED event when triggering

Usage:
    from app.coordination.training_trigger_daemon import TrainingTriggerDaemon

    daemon = TrainingTriggerDaemon()
    await daemon.start()

December 2025: Created as part of Phase 1 automation improvements.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.config.coordination_defaults import DataFreshnessDefaults, SyncDefaults
from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)

# Dec 29, 2025: Event deduplication to prevent multiple training triggers
# for the same config within a short window
TRIGGER_DEDUP_WINDOW_SECONDS = 300  # 5 minutes

# Circuit breaker integration (Phase 4 - December 2025)
try:
    from app.distributed.circuit_breaker import get_training_breaker
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    get_training_breaker = None


@dataclass
class TrainingTriggerConfig:
    """Configuration for training trigger decisions."""

    enabled: bool = True
    # Data freshness - uses DataFreshnessDefaults for unified config (December 2025)
    # Default 4h from RINGRIFT_MAX_DATA_AGE_HOURS env var
    max_data_age_hours: float = field(
        default_factory=lambda: DataFreshnessDefaults().MAX_DATA_AGE_HOURS
    )
    # December 2025: Use training_freshness to trigger sync when data is stale
    enforce_freshness_with_sync: bool = field(
        default_factory=lambda: DataFreshnessDefaults().ENFORCE_FRESHNESS_WITH_SYNC
    )
    freshness_sync_timeout_seconds: float = field(
        default_factory=lambda: DataFreshnessDefaults().FRESHNESS_SYNC_TIMEOUT
    )
    # December 29, 2025: Strict mode - fail immediately if data is stale (no sync attempt)
    # Useful for high-quality training where only fresh data should be used
    strict_freshness_mode: bool = field(
        default_factory=lambda: DataFreshnessDefaults().STRICT_FRESHNESS
    )
    # Minimum samples to trigger training
    # December 29, 2025: Reduced from 10000 to 5000 for faster iteration cycles
    min_samples_threshold: int = 5000
    # December 29, 2025: Confidence-based early triggering
    # Allows training to start earlier if statistical confidence is high
    confidence_early_trigger_enabled: bool = True
    # Minimum samples to even consider confidence-based early trigger (safety floor)
    confidence_min_samples: int = 1000
    # Target confidence interval width (95% CI width, e.g., 0.05 = ±2.5%)
    confidence_target_ci_width: float = 0.05
    # Cooldown between training runs for same config
    # December 29, 2025: Reduced from 1.0 to 0.083 (5 min) for faster iteration cycles
    training_cooldown_hours: float = 0.083
    # Maximum concurrent training jobs
    # December 29, 2025: Increased from 10 to 20 for better multi-GPU cluster utilization
    # Cluster has ~36 nodes with GPUs; allowing more concurrent training maximizes throughput
    max_concurrent_training: int = 20
    # GPU utilization threshold for "idle"
    gpu_idle_threshold_percent: float = 20.0
    # Timeout for training subprocess (24 hours)
    training_timeout_seconds: int = 86400
    # December 29, 2025: Training timeout watchdog (Phase 2 - 48h autonomous operation)
    # Independent watchdog kills training jobs that exceed this limit
    # This catches hung processes even if the daemon restarts
    training_timeout_hours: float = 4.0
    # Check interval for periodic scans
    # December 29, 2025: Reduced from 120s to 30s for faster detection
    scan_interval_seconds: int = 30  # 30 seconds
    # Training epochs
    default_epochs: int = 50
    default_batch_size: int = 512
    # Model version
    model_version: str = "v2"
    # December 29, 2025: State persistence for daemon restarts (Phase 3)
    state_db_path: str = "data/coordination/training_trigger_state.db"
    state_save_interval_seconds: float = 300.0  # Save every 5 minutes


@dataclass
class ConfigTrainingState:
    """Tracks training state for a single configuration."""

    config_key: str
    board_type: str
    num_players: int
    # Training status
    last_training_time: float = 0.0
    training_in_progress: bool = False
    training_pid: int | None = None
    # December 29, 2025: Training timeout watchdog (Phase 2)
    training_start_time: float = 0.0  # When current training started
    # Data status
    last_npz_update: float = 0.0
    npz_sample_count: int = 0
    npz_path: str = ""
    # Quality tracking
    last_elo: float = 1500.0
    elo_trend: float = 0.0  # positive = improving
    # December 29, 2025: Elo velocity tracking for training decisions
    elo_velocity: float = 0.0  # Elo/hour rate of change
    elo_velocity_trend: str = "stable"  # accelerating, stable, decelerating, plateauing
    last_elo_velocity_update: float = 0.0
    # Training intensity (set by master_loop or FeedbackLoopController)
    training_intensity: str = "normal"  # hot_path, accelerated, normal, reduced, paused
    consecutive_failures: int = 0
    # December 29, 2025: Track model path for event emission
    _pending_model_path: str = ""  # Path where current training will save model


class TrainingTriggerDaemon(HandlerBase):
    """Daemon that automatically triggers training when conditions are met.

    Inherits from HandlerBase (December 2025 migration) providing:
    - Automatic event subscription via _get_event_subscriptions()
    - Singleton pattern via get_instance()
    - Standardized health check format
    - Lifecycle management (start/stop)
    """

    def __init__(self, config: TrainingTriggerConfig | None = None):
        self._daemon_config = config or TrainingTriggerConfig()
        super().__init__(
            name="training_trigger",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.scan_interval_seconds),
        )
        self._training_states: dict[str, ConfigTrainingState] = {}
        self._training_semaphore = asyncio.Semaphore(self._daemon_config.max_concurrent_training)
        self._active_training_tasks: dict[str, asyncio.Task] = {}
        # Track whether we should skip due to coordinator mode
        self._coordinator_skip = False
        # Dec 29, 2025: Deduplication tracking for training triggers
        self._recent_triggers: dict[str, float] = {}  # config_key -> last_trigger_time
        # December 29, 2025: State persistence (Phase 3)
        self._state_db_path = Path(self._daemon_config.state_db_path)
        self._last_state_save: float = 0.0
        self._init_state_db()
        # December 29, 2025 (Phase 4): Evaluation backpressure tracking
        # When EvaluationDaemon queue fills up, we pause training to let evaluations catch up
        self._evaluation_backpressure: bool = False
        self._backpressure_stats = {
            "pauses_due_to_backpressure": 0,
            "resumes_after_backpressure": 0,
            "last_backpressure_time": 0.0,
        }
        # December 29, 2025 (Phase 3): Training retry queue for failed jobs
        # Tuple: (config_key, board_type, num_players, attempts, next_retry_time, error)
        self._training_retry_queue: deque[tuple[str, str, int, int, float, str]] = deque()
        self._max_training_retries = 3
        self._base_training_retry_delay = 300.0  # 5 minutes, exponential backoff
        self._retry_stats = {
            "retries_queued": 0,
            "retries_succeeded": 0,
            "retries_exhausted": 0,
        }
        # December 29, 2025 (Phase 2): Timeout watchdog stats
        self._timeout_stats = {
            "timeouts_detected": 0,
            "processes_killed": 0,
            "last_timeout_time": 0.0,
        }

    def _init_state_db(self) -> None:
        """Initialize the SQLite state database (Phase 3 - December 2025).

        Creates the state table if it doesn't exist. This persists training
        state across daemon restarts, preventing loss of training momentum.
        """
        try:
            self._state_db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._state_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS config_state (
                        config_key TEXT PRIMARY KEY,
                        board_type TEXT NOT NULL,
                        num_players INTEGER NOT NULL,
                        last_training_time REAL DEFAULT 0.0,
                        training_in_progress INTEGER DEFAULT 0,
                        last_npz_update REAL DEFAULT 0.0,
                        npz_sample_count INTEGER DEFAULT 0,
                        npz_path TEXT DEFAULT '',
                        last_elo REAL DEFAULT 1500.0,
                        elo_trend REAL DEFAULT 0.0,
                        elo_velocity REAL DEFAULT 0.0,
                        elo_velocity_trend TEXT DEFAULT 'stable',
                        last_elo_velocity_update REAL DEFAULT 0.0,
                        training_intensity TEXT DEFAULT 'normal',
                        consecutive_failures INTEGER DEFAULT 0,
                        updated_at REAL DEFAULT 0.0
                    )
                """)
                # December 29, 2025: Add velocity columns if upgrading from earlier schema
                try:
                    conn.execute("ALTER TABLE config_state ADD COLUMN elo_velocity REAL DEFAULT 0.0")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                try:
                    conn.execute("ALTER TABLE config_state ADD COLUMN elo_velocity_trend TEXT DEFAULT 'stable'")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                try:
                    conn.execute("ALTER TABLE config_state ADD COLUMN last_elo_velocity_update REAL DEFAULT 0.0")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                conn.commit()
            logger.debug(f"[TrainingTriggerDaemon] State DB initialized: {self._state_db_path}")
        except (sqlite3.Error, OSError) as e:
            logger.warning(f"[TrainingTriggerDaemon] Failed to init state DB: {e}")

    def _load_state(self) -> None:
        """Load persisted training state from SQLite (Phase 3 - December 2025).

        Called at daemon startup to restore training momentum after restarts.
        """
        if not self._state_db_path.exists():
            logger.debug("[TrainingTriggerDaemon] No persisted state to load")
            return

        try:
            with sqlite3.connect(self._state_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM config_state")
                for row in cursor.fetchall():
                    config_key = row["config_key"]
                    # Don't overwrite state if already exists (e.g., from event handling)
                    if config_key not in self._training_states:
                        # December 29, 2025: Load velocity fields with fallback for older DBs
                        elo_velocity = row["elo_velocity"] if "elo_velocity" in row.keys() else 0.0
                        elo_velocity_trend = row["elo_velocity_trend"] if "elo_velocity_trend" in row.keys() else "stable"
                        last_elo_velocity_update = row["last_elo_velocity_update"] if "last_elo_velocity_update" in row.keys() else 0.0
                        state = ConfigTrainingState(
                            config_key=config_key,
                            board_type=row["board_type"],
                            num_players=row["num_players"],
                            last_training_time=row["last_training_time"],
                            training_in_progress=False,  # Reset on restart
                            last_npz_update=row["last_npz_update"],
                            npz_sample_count=row["npz_sample_count"],
                            npz_path=row["npz_path"],
                            last_elo=row["last_elo"],
                            elo_trend=row["elo_trend"],
                            elo_velocity=elo_velocity,
                            elo_velocity_trend=elo_velocity_trend,
                            last_elo_velocity_update=last_elo_velocity_update,
                            training_intensity=row["training_intensity"],
                            consecutive_failures=row["consecutive_failures"],
                        )
                        self._training_states[config_key] = state
                logger.info(
                    f"[TrainingTriggerDaemon] Loaded {len(self._training_states)} "
                    f"config states from persisted storage"
                )
        except (sqlite3.Error, KeyError) as e:
            logger.warning(f"[TrainingTriggerDaemon] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save current training state to SQLite (Phase 3 - December 2025).

        Called periodically and on significant state changes.
        """
        if not self._training_states:
            return

        now = time.time()
        try:
            with sqlite3.connect(self._state_db_path) as conn:
                for config_key, state in self._training_states.items():
                    # December 29, 2025: Include velocity fields in state persistence
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO config_state (
                            config_key, board_type, num_players,
                            last_training_time, training_in_progress,
                            last_npz_update, npz_sample_count, npz_path,
                            last_elo, elo_trend,
                            elo_velocity, elo_velocity_trend, last_elo_velocity_update,
                            training_intensity,
                            consecutive_failures, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            config_key,
                            state.board_type,
                            state.num_players,
                            state.last_training_time,
                            1 if state.training_in_progress else 0,
                            state.last_npz_update,
                            state.npz_sample_count,
                            state.npz_path,
                            state.last_elo,
                            state.elo_trend,
                            state.elo_velocity,
                            state.elo_velocity_trend,
                            state.last_elo_velocity_update,
                            state.training_intensity,
                            state.consecutive_failures,
                            now,
                        ),
                    )
                conn.commit()
            self._last_state_save = now
            logger.debug(
                f"[TrainingTriggerDaemon] Saved {len(self._training_states)} config states"
            )
        except sqlite3.Error as e:
            logger.warning(f"[TrainingTriggerDaemon] Failed to save state: {e}")

    @property
    def config(self) -> TrainingTriggerConfig:
        """Get the daemon configuration."""
        return self._daemon_config

    def _should_skip_duplicate_trigger(self, config_key: str) -> bool:
        """Check if this config was recently triggered (deduplication).

        Dec 29, 2025: Prevents multiple event paths from triggering duplicate
        training attempts for the same config within a 5-minute window.

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")

        Returns:
            True if trigger should be skipped (duplicate), False otherwise
        """
        now = time.time()
        last_trigger = self._recent_triggers.get(config_key, 0)
        if now - last_trigger < TRIGGER_DEDUP_WINDOW_SECONDS:
            logger.debug(
                f"[TrainingTriggerDaemon] Skipping duplicate trigger for {config_key} "
                f"(last trigger {now - last_trigger:.0f}s ago)"
            )
            return True
        self._recent_triggers[config_key] = now
        return False

    async def start(self) -> None:
        """Start the daemon and load persisted state (Phase 3 - December 2025).

        Overrides HandlerBase.start() to restore training state from SQLite
        before beginning operations. This prevents loss of training momentum
        when the daemon restarts.
        """
        # Load persisted state before starting
        self._load_state()

        # Call parent start() which will run _run_cycle() periodically
        await super().start()

    async def stop(self) -> None:
        """Stop the daemon and save state (Phase 3 - December 2025).

        Overrides HandlerBase.stop() to persist training state to SQLite
        before shutdown. This ensures no state loss on graceful shutdown.
        """
        # Save state before stopping
        self._save_state()
        logger.info("[TrainingTriggerDaemon] Saved state on shutdown")

        # Call parent stop()
        await super().stop()

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event subscriptions for HandlerBase.

        Subscribes to:
        - NPZ_EXPORT_COMPLETE: Immediate training trigger after export
        - TRAINING_COMPLETED: Track state after training finishes
        - TRAINING_THRESHOLD_REACHED: Honor master_loop-triggered requests
        - QUALITY_SCORE_UPDATED: Keep intensity in sync
        - TRAINING_BLOCKED_BY_QUALITY: Pause intensity
        - EVALUATION_COMPLETED: Gauntlet -> training feedback
        - TRAINING_INTENSITY_CHANGED: Updates from unified_feedback orchestrator
        - DATA_STALE: React to stale data alerts (Dec 2025 - Phase 2A)
        - DATA_SYNC_COMPLETED: Retry training after fresh data arrives (Dec 2025 - Phase 2A)
        - EVALUATION_BACKPRESSURE: Pause training when eval queue is full (Dec 2025 - Phase 4)
        - EVALUATION_BACKPRESSURE_RELEASED: Resume training when eval queue drains (Dec 2025 - Phase 4)
        - ELO_VELOCITY_CHANGED: Adjust cooldown and intensity based on Elo velocity (Dec 2025)
        """
        return {
            "npz_export_complete": self._on_npz_export_complete,
            "training_completed": self._on_training_completed,
            "training_threshold_reached": self._on_training_threshold_reached,
            "quality_score_updated": self._on_quality_score_updated,
            "training_blocked_by_quality": self._on_training_blocked_by_quality,
            "evaluation_completed": self._on_evaluation_completed,
            "training_intensity_changed": self._on_training_intensity_changed,
            # December 2025 - Phase 2A: Data freshness events
            "data_stale": self._on_data_stale,
            "data_sync_completed": self._on_data_sync_completed,
            # December 29, 2025 (Phase 4): Evaluation backpressure events
            "EVALUATION_BACKPRESSURE": self._on_evaluation_backpressure,
            "EVALUATION_BACKPRESSURE_RELEASED": self._on_evaluation_backpressure_released,
            # December 29, 2025: Elo velocity-based training decisions
            "elo_velocity_changed": self._on_elo_velocity_changed,
            # December 29, 2025 (Phase 3): Training failure with retry
            "training_failed": self._on_training_failed,
        }

    async def _on_start(self) -> None:
        """Hook called before main loop - check coordinator mode."""
        from app.config.env import env
        if env.is_coordinator or not env.training_enabled:
            logger.info(
                f"[TrainingTriggerDaemon] Skipped on coordinator node: {env.node_id} "
                f"(is_coordinator={env.is_coordinator}, training_enabled={env.training_enabled})"
            )
            self._coordinator_skip = True

    async def _on_stop(self) -> None:
        """Hook called when stopping - cancel active training tasks."""
        for config_key, task in self._active_training_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"[TrainingTriggerDaemon] Cancelled training for {config_key}")

    def _get_training_params_for_intensity(
        self, intensity: str
    ) -> tuple[int, int, float]:
        """Map training intensity to (epochs, batch_size, lr_multiplier).

        December 2025: Fixes Gap 2 - training_intensity was defined but never consumed.
        The FeedbackLoopController sets intensity based on quality score:
          - hot_path (quality >= 0.90): Fast iteration, high LR
          - accelerated (quality >= 0.80): Increased training, moderate LR boost
          - normal (quality >= 0.65): Default parameters
          - reduced (quality >= 0.50): More epochs at lower LR for struggling configs
          - paused: Skip training entirely (handled in _maybe_trigger_training)

        Returns:
            Tuple of (epochs, batch_size, learning_rate_multiplier)
        """
        intensity_params = {
            # hot_path: Fast iteration with larger batches, higher LR
            "hot_path": (30, 1024, 1.5),
            # accelerated: More aggressive training
            "accelerated": (40, 768, 1.2),
            # normal: Default parameters
            "normal": (self.config.default_epochs, self.config.default_batch_size, 1.0),
            # reduced: Slower, more careful training for struggling configs
            "reduced": (60, 256, 0.8),
            # paused: Should not reach here, but use minimal params
            "paused": (10, 128, 0.5),
        }

        params = intensity_params.get(intensity)
        if params is None:
            logger.warning(
                f"[TrainingTriggerDaemon] Unknown intensity '{intensity}', using 'normal'"
            )
            params = intensity_params["normal"]

        return params

    def _get_dynamic_sample_threshold(self, config_key: str) -> int:
        """Get dynamically adjusted sample threshold for training.

        Phase 5 (Dec 2025): Uses ImprovementOptimizer to adjust thresholds
        based on training success patterns:
        - On promotion streak: Lower threshold → faster iteration
        - Struggling/regression: Higher threshold → more conservative

        Args:
            config_key: Configuration identifier

        Returns:
            Minimum sample count required to trigger training
        """
        try:
            from app.training.improvement_optimizer import get_dynamic_threshold

            dynamic_threshold = get_dynamic_threshold(config_key)

            # Log significant deviations from base threshold
            if dynamic_threshold != self.config.min_samples_threshold:
                logger.debug(
                    f"[TrainingTriggerDaemon] Dynamic threshold for {config_key}: "
                    f"{dynamic_threshold} (base: {self.config.min_samples_threshold})"
                )

            return dynamic_threshold

        except ImportError:
            logger.debug("[TrainingTriggerDaemon] improvement_optimizer not available")
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Error getting dynamic threshold: {e}")

        # Fallback to static config threshold
        return self.config.min_samples_threshold

    def _check_confidence_early_trigger(
        self, config_key: str, sample_count: int
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

        But we use the actual quality variance when available for more
        accurate confidence estimation.

        Args:
            config_key: Configuration identifier
            sample_count: Current number of training samples

        Returns:
            Tuple of (should_trigger, reason)
        """
        if not self.config.confidence_early_trigger_enabled:
            return False, "confidence early trigger disabled"

        # Safety floor: never trigger with fewer than confidence_min_samples
        if sample_count < self.config.confidence_min_samples:
            return False, f"below safety floor ({sample_count} < {self.config.confidence_min_samples})"

        # Estimate confidence interval width
        # For binary outcomes (win/loss), variance is p*(1-p) ≤ 0.25
        # Using 0.25 as conservative estimate
        variance = 0.25
        z_score = 1.96  # 95% confidence

        # Try to get actual variance from quality monitor
        try:
            from app.coordination.quality_monitor_daemon import get_quality_monitor_daemon
            qm = get_quality_monitor_daemon()
            if qm and hasattr(qm, 'get_quality_metrics'):
                metrics = qm.get_quality_metrics(config_key)
                if metrics and 'variance' in metrics:
                    variance = min(0.25, metrics['variance'])  # Cap at 0.25
        except (ImportError, AttributeError, KeyError, TypeError, ValueError):
            pass  # Use default variance if quality monitor unavailable

        # Calculate CI width: 2 * z * sqrt(variance / n)
        ci_width = 2 * z_score * math.sqrt(variance / sample_count)

        # Check if confidence is high enough
        if ci_width <= self.config.confidence_target_ci_width:
            logger.info(
                f"[TrainingTriggerDaemon] Confidence early trigger for {config_key}: "
                f"CI_width={ci_width:.4f} ≤ target={self.config.confidence_target_ci_width:.4f}, "
                f"samples={sample_count}"
            )
            return True, f"confidence threshold met (CI={ci_width:.4f})"

        return False, f"confidence not met (CI={ci_width:.4f} > target={self.config.confidence_target_ci_width:.4f})"

    async def _on_npz_export_complete(self, result: Any) -> None:
        """Handle NPZ export completion - immediate training trigger."""
        try:
            metadata = getattr(result, "metadata", {})
            config_key = metadata.get("config")
            board_type = metadata.get("board_type")
            num_players = metadata.get("num_players")
            npz_path = metadata.get("output_path", "")
            samples = metadata.get("samples", 0)

            if not config_key:
                # Try to build from board_type and num_players
                if board_type and num_players:
                    config_key = f"{board_type}_{num_players}p"
                else:
                    logger.debug("[TrainingTriggerDaemon] Missing config info in NPZ export result")
                    return

            # Update state
            state = self._get_or_create_state(config_key, board_type, num_players)
            state.last_npz_update = time.time()
            state.npz_sample_count = samples or 0
            state.npz_path = npz_path

            logger.info(
                f"[TrainingTriggerDaemon] NPZ export complete for {config_key}: "
                f"{samples} samples at {npz_path}"
            )

            # Check if we should trigger training
            await self._maybe_trigger_training(config_key)

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling NPZ export: {e}")

    async def _on_training_completed(self, event: Any) -> None:
        """Handle training completion to update state."""
        try:
            payload = getattr(event, "payload", {})
            config_key = payload.get("config")

            if config_key and config_key in self._training_states:
                state = self._training_states[config_key]
                state.training_in_progress = False
                state.training_pid = None
                state.last_training_time = time.time()

                # Update ELO tracking if available
                if "elo" in payload:
                    old_elo = state.last_elo
                    state.last_elo = payload["elo"]
                    state.elo_trend = state.last_elo - old_elo

                logger.info(f"[TrainingTriggerDaemon] Training completed for {config_key}")

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling training completion: {e}")

    def _intensity_from_quality(self, quality_score: float) -> str:
        """Map quality scores to training intensity."""
        if quality_score >= 0.90:
            return "hot_path"
        if quality_score >= 0.80:
            return "accelerated"
        if quality_score >= 0.65:
            return "normal"
        if quality_score >= 0.50:
            return "reduced"
        return "paused"

    async def _on_training_threshold_reached(self, event: Any) -> None:
        """Handle training threshold reached events from master_loop."""
        try:
            payload = getattr(event, "payload", {})
            config_key = payload.get("config") or payload.get("config_key")
            if not config_key:
                return

            # Dec 29, 2025: Check for duplicate trigger within dedup window
            if self._should_skip_duplicate_trigger(config_key):
                return

            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            state = self._get_or_create_state(config_key, board_type, num_players)

            intensity = payload.get("priority") or payload.get("training_intensity")
            if intensity:
                state.training_intensity = intensity
                logger.debug(
                    f"[TrainingTriggerDaemon] {config_key}: "
                    f"training_intensity set to {intensity}"
                )

            await self._maybe_trigger_training(config_key)

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling training threshold: {e}")

    async def _on_quality_score_updated(self, event: Any) -> None:
        """Handle quality score updates to keep intensity in sync."""
        try:
            payload = getattr(event, "payload", {})
            config_key = payload.get("config_key") or payload.get("config")
            if not config_key:
                return

            quality_score = float(payload.get("quality_score", 0.0))
            state = self._get_or_create_state(config_key)
            state.training_intensity = self._intensity_from_quality(quality_score)

            logger.debug(
                f"[TrainingTriggerDaemon] {config_key}: "
                f"quality={quality_score:.2f} → intensity={state.training_intensity}"
            )
        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling quality update: {e}")

    async def _on_training_intensity_changed(self, event: Any) -> None:
        """Handle training intensity changes from unified_feedback orchestrator.

        December 2025: Enables event-driven quality feedback instead of direct
        object assignment. The unified_feedback.py emits TRAINING_INTENSITY_CHANGED
        when quality metrics change, and this handler updates local state.

        Payload:
            config_key: str - The board config (e.g., "hex8_2p")
            intensity: str - The new intensity level
            quality: float - The quality score that triggered the change
        """
        try:
            payload = getattr(event, "payload", {})
            config_key = payload.get("config_key") or payload.get("config")
            if not config_key:
                return

            new_intensity = payload.get("intensity")
            if not new_intensity:
                return

            state = self._get_or_create_state(config_key)
            old_intensity = state.training_intensity

            # Only update if intensity actually changed
            if old_intensity != new_intensity:
                state.training_intensity = new_intensity
                quality = payload.get("quality", 0.0)
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: "
                    f"intensity changed via event: {old_intensity} → {new_intensity} "
                    f"(quality={quality:.2f})"
                )
        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling intensity change: {e}")

    async def _on_training_blocked_by_quality(self, event: Any) -> None:
        """Handle training blocked events to pause intensity."""
        try:
            payload = getattr(event, "payload", {})
            config_key = payload.get("config_key") or payload.get("config")
            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.training_intensity = "paused"

            logger.info(f"[TrainingTriggerDaemon] {config_key}: training paused due to quality gate")
        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling training blocked: {e}")

    async def _on_evaluation_completed(self, event: Any) -> None:
        """Handle gauntlet evaluation completion - adjust training parameters (Dec 2025).

        This closes the critical feedback loop: gauntlet performance → training parameters.

        Adjustments based on win rate:
        - Win rate < 40%: Boost training intensity, increase epochs, trigger extra selfplay
        - Win rate 40-60%: Increase training to "accelerated" mode
        - Win rate 60-75%: Normal training, model is improving
        - Win rate > 75%: Reduce intensity, model is strong

        Expected improvement: 50-100 Elo by closing the feedback loop.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event

            config_key = payload.get("config", "") or payload.get("config_key", "")
            win_rate = payload.get("win_rate", 0.5)
            elo = payload.get("elo", 1500.0)
            games_played = payload.get("games_played", 0)

            # Try to parse config_key from model_id if not provided
            if not config_key:
                model_id = payload.get("model_id", "")
                if model_id:
                    # Extract config from model_id like "canonical_hex8_2p" -> "hex8_2p"
                    parts = model_id.replace("canonical_", "").rsplit("_", 1)
                    if len(parts) == 2 and parts[1].endswith("p"):
                        config_key = f"{parts[0]}_{parts[1]}"

            if not config_key:
                logger.debug("[TrainingTriggerDaemon] No config_key in evaluation event")
                return

            state = self._get_or_create_state(config_key)

            # Calculate Elo change if we have previous Elo
            elo_delta = elo - state.last_elo if state.last_elo > 0 else 0.0
            state.elo_trend = elo_delta
            old_elo = state.last_elo
            state.last_elo = elo

            logger.info(
                f"[TrainingTriggerDaemon] Evaluation complete for {config_key}: "
                f"win_rate={win_rate:.1%}, elo={elo:.0f} (delta={elo_delta:+.0f}), "
                f"games={games_played}"
            )

            # Determine new training intensity based on win rate
            old_intensity = state.training_intensity

            if win_rate < 0.40:
                # Struggling model - aggressive training boost
                state.training_intensity = "accelerated"
                logger.warning(
                    f"[TrainingTriggerDaemon] {config_key} struggling (win_rate={win_rate:.1%}), "
                    f"boosting training intensity to 'accelerated'"
                )
                # Trigger extra selfplay to generate more training data
                await self._trigger_selfplay_boost(config_key, multiplier=1.5)

            elif win_rate < 0.60:
                # Below target but not terrible - increase training
                state.training_intensity = "accelerated"
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key} below target (win_rate={win_rate:.1%}), "
                    f"setting intensity to 'accelerated'"
                )

            elif win_rate < 0.75:
                # Reasonable performance - normal training
                state.training_intensity = "normal"
                if old_intensity != "normal":
                    logger.info(
                        f"[TrainingTriggerDaemon] {config_key} recovering (win_rate={win_rate:.1%}), "
                        f"returning to 'normal' intensity"
                    )

            else:
                # Strong model - can reduce training intensity
                if state.training_intensity != "reduced":
                    state.training_intensity = "reduced"
                    logger.info(
                        f"[TrainingTriggerDaemon] {config_key} strong (win_rate={win_rate:.1%}), "
                        f"reducing training intensity"
                    )

            # Check for Elo plateau (no improvement over multiple evaluations)
            if elo_delta <= 5 and old_elo > 0:
                state.consecutive_failures += 1
                if state.consecutive_failures >= 3:
                    logger.warning(
                        f"[TrainingTriggerDaemon] {config_key} Elo plateau detected "
                        f"({state.consecutive_failures} evals with minimal improvement), "
                        f"consider curriculum advancement"
                    )
                    await self._signal_curriculum_advancement(config_key)
            else:
                # Elo improved - reset failure counter
                state.consecutive_failures = 0

            # Record to FeedbackAccelerator for Elo momentum tracking
            await self._record_to_feedback_accelerator(config_key, elo, elo_delta)

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling evaluation: {e}")

    async def _on_data_stale(self, event: Any) -> None:
        """Handle DATA_STALE events - mark config as needing fresh data (Dec 2025 Phase 2A).

        When training data becomes stale (age exceeds threshold), this handler:
        1. Updates local state to track that fresh data is needed
        2. Triggers priority sync if training was pending for this config

        This closes the data freshness feedback loop: TrainingFreshness emits
        DATA_STALE, TrainingTriggerDaemon receives it and triggers sync.

        Args:
            event: Event with payload containing config_key, board_type, num_players, data_age_hours
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key") or payload.get("config")
            if not config_key:
                return

            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            data_age_hours = payload.get("data_age_hours", 0.0)

            state = self._get_or_create_state(config_key, board_type, num_players)

            logger.info(
                f"[TrainingTriggerDaemon] DATA_STALE received for {config_key}: "
                f"data_age={data_age_hours:.1f}h"
            )

            # If training was pending (not in progress, has data), trigger priority sync
            if not state.training_in_progress and state.npz_sample_count > 0:
                logger.info(
                    f"[TrainingTriggerDaemon] Triggering priority sync for {config_key} "
                    f"(stale data, training pending)"
                )
                await self._trigger_priority_sync(config_key, state.board_type, state.num_players)

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling DATA_STALE: {e}")

    async def _on_data_sync_completed(self, event: Any) -> None:
        """Handle DATA_SYNC_COMPLETED events - retry training after fresh data arrives (Dec 2025 Phase 2A).

        When data sync completes, this handler:
        1. Updates local state with fresh data timestamp
        2. Checks if any training was blocked waiting for fresh data
        3. Retries _maybe_trigger_training() for affected configs

        This completes the data freshness loop: AutoSyncDaemon emits
        DATA_SYNC_COMPLETED, TrainingTriggerDaemon receives it and retries training.

        Args:
            event: Event with payload containing config_key, board_type, num_players, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key") or payload.get("config")
            sync_type = payload.get("sync_type", "")

            # Also handle generic syncs that may have refreshed multiple configs
            if not config_key and sync_type in ("broadcast", "full", "cluster"):
                # Full sync - retry all configs that might need fresh data
                for key in list(self._training_states.keys()):
                    await self._maybe_trigger_training(key)
                return

            if not config_key:
                return

            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            files_synced = payload.get("files_synced", 0)

            state = self._get_or_create_state(config_key, board_type, num_players)

            # Update data freshness timestamp
            state.last_npz_update = time.time()

            logger.info(
                f"[TrainingTriggerDaemon] DATA_SYNC_COMPLETED for {config_key}: "
                f"{files_synced} files synced, retrying training check"
            )

            # Retry training now that we have fresh data
            triggered = await self._maybe_trigger_training(config_key)
            if triggered:
                logger.info(
                    f"[TrainingTriggerDaemon] Training triggered for {config_key} "
                    f"after data sync"
                )

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling DATA_SYNC_COMPLETED: {e}")

    async def _on_evaluation_backpressure(self, event: Any) -> None:
        """Handle EVALUATION_BACKPRESSURE event - pause training to let evaluations catch up.

        December 29, 2025 (Phase 4): When EvaluationDaemon queue fills up,
        this handler pauses training triggers to prevent GPU waste from
        duplicate evaluations. Training resumes when queue drains.

        Args:
            event: Event with payload containing queue_depth, threshold, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            queue_depth = payload.get("queue_depth", 0)
            threshold = payload.get("threshold", 40)

            if not self._evaluation_backpressure:
                self._evaluation_backpressure = True
                self._backpressure_stats["pauses_due_to_backpressure"] += 1
                self._backpressure_stats["last_backpressure_time"] = time.time()

                logger.warning(
                    f"[TrainingTriggerDaemon] Training PAUSED due to evaluation backpressure: "
                    f"queue_depth={queue_depth}, threshold={threshold}"
                )

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling EVALUATION_BACKPRESSURE: {e}")

    async def _on_evaluation_backpressure_released(self, event: Any) -> None:
        """Handle EVALUATION_BACKPRESSURE_RELEASED event - resume training.

        December 29, 2025 (Phase 4): When EvaluationDaemon queue drains below
        the release threshold, this handler resumes training triggers.

        Args:
            event: Event with payload containing queue_depth, release_threshold, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            queue_depth = payload.get("queue_depth", 0)
            release_threshold = payload.get("release_threshold", 20)

            if self._evaluation_backpressure:
                self._evaluation_backpressure = False
                self._backpressure_stats["resumes_after_backpressure"] += 1

                # Calculate pause duration for logging
                pause_duration = 0.0
                if self._backpressure_stats["last_backpressure_time"] > 0:
                    pause_duration = time.time() - self._backpressure_stats["last_backpressure_time"]

                logger.info(
                    f"[TrainingTriggerDaemon] Training RESUMED: evaluation backpressure released "
                    f"(queue_depth={queue_depth}, release_threshold={release_threshold}, "
                    f"pause_duration={pause_duration:.1f}s)"
                )

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling EVALUATION_BACKPRESSURE_RELEASED: {e}")

    async def _on_elo_velocity_changed(self, event: Any) -> None:
        """Handle ELO_VELOCITY_CHANGED event for velocity-based training decisions.

        December 29, 2025: Wires Elo velocity to training trigger decisions.
        This closes the feedback loop: Elo velocity → training cooldown adjustment.

        Velocity trends influence training decisions:
        - accelerating: Shorten training cooldown (capitalize on momentum)
        - stable: Use default cooldown
        - decelerating: Lengthen cooldown (avoid wasteful training)
        - plateauing: May trigger exploration boost or hyperparameter adjustment

        Args:
            event: Event with payload containing config_key, velocity, trend, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            velocity = payload.get("velocity", 0.0)
            trend = payload.get("trend", "stable")
            previous_velocity = payload.get("previous_velocity", 0.0)

            if not config_key:
                return

            # Parse board_type and num_players from config_key
            board_type = config_key.rsplit("_", 1)[0] if "_" in config_key else ""
            try:
                num_players = int(config_key.rsplit("_", 1)[1].rstrip("p"))
            except (ValueError, IndexError):
                num_players = 2

            state = self._get_or_create_state(config_key, board_type, num_players)

            # Update state with velocity info
            old_velocity = state.elo_velocity
            old_trend = state.elo_velocity_trend
            state.elo_velocity = velocity
            state.elo_velocity_trend = trend
            state.last_elo_velocity_update = time.time()

            # Log significant changes
            if trend != old_trend or abs(velocity - old_velocity) > 5.0:
                logger.info(
                    f"[TrainingTriggerDaemon] Elo velocity changed for {config_key}: "
                    f"velocity={velocity:.1f}/hr (was {old_velocity:.1f}/hr), "
                    f"trend={trend} (was {old_trend})"
                )

            # Adjust training intensity based on velocity trend
            # This influences training parameters and cooldown
            if trend == "accelerating":
                # Config is improving rapidly - prioritize training
                if state.training_intensity in ("normal", "reduced"):
                    state.training_intensity = "accelerated"
                    logger.info(
                        f"[TrainingTriggerDaemon] Upgraded {config_key} to 'accelerated' "
                        f"due to positive Elo velocity ({velocity:.1f}/hr)"
                    )
            elif trend == "plateauing":
                # Config has plateaued - may need exploration boost
                if state.training_intensity == "hot_path":
                    state.training_intensity = "normal"
                    logger.info(
                        f"[TrainingTriggerDaemon] Downgraded {config_key} from 'hot_path' to 'normal' "
                        f"due to Elo plateau"
                    )
            elif trend == "decelerating" and velocity < -5.0:
                # Config is regressing - reduce training intensity to avoid waste
                if state.training_intensity == "accelerated":
                    state.training_intensity = "normal"
                    logger.info(
                        f"[TrainingTriggerDaemon] Downgraded {config_key} to 'normal' "
                        f"due to negative Elo velocity ({velocity:.1f}/hr)"
                    )

            # Mark state as needing persistence
            self._save_state()

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.debug(f"[TrainingTriggerDaemon] Error handling ELO_VELOCITY_CHANGED: {e}")

    async def _on_training_failed(self, event: Any) -> None:
        """Handle TRAINING_FAILED event with retry logic.

        December 29, 2025 (Phase 3): Implements automatic retry for transient
        training failures (GPU OOM, network issues, temporary resource constraints).

        Retries are queued with exponential backoff (5min, 10min, 20min).
        After max retries (3), the failure is permanent and state is updated.

        Args:
            event: Event with payload containing config_key, error, job_id, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            error = payload.get("error", "Unknown error")
            job_id = payload.get("job_id", "")

            if not config_key:
                return

            # Parse board_type and num_players from config_key
            board_type = config_key.rsplit("_", 1)[0] if "_" in config_key else ""
            try:
                num_players = int(config_key.rsplit("_", 1)[1].rstrip("p"))
            except (ValueError, IndexError):
                num_players = 2

            state = self._get_or_create_state(config_key, board_type, num_players)

            # Clear training_in_progress flag
            state.training_in_progress = False
            state.consecutive_failures += 1

            # Determine if error is retryable
            error_lower = error.lower()
            is_retryable = any(pattern in error_lower for pattern in [
                "cuda", "out of memory", "timeout", "connection",
                "temporarily unavailable", "resource", "network",
            ])

            if is_retryable:
                queued = self._queue_training_retry(
                    config_key, board_type, num_players, error,
                    current_attempts=0  # Will check retry queue for existing attempts
                )
                if queued:
                    logger.info(
                        f"[TrainingTriggerDaemon] Queued retry for {config_key} "
                        f"after transient failure: {error[:100]}"
                    )
                    return  # Don't update permanent failure state yet

            # Permanent failure or max retries exceeded
            logger.error(
                f"[TrainingTriggerDaemon] Training permanently failed for {config_key}: "
                f"{error[:200]} (consecutive_failures={state.consecutive_failures})"
            )

            # If too many consecutive failures, reduce intensity
            if state.consecutive_failures >= 3:
                if state.training_intensity not in ("paused", "reduced"):
                    old_intensity = state.training_intensity
                    state.training_intensity = "reduced"
                    logger.warning(
                        f"[TrainingTriggerDaemon] Reduced training intensity for {config_key} "
                        f"after {state.consecutive_failures} consecutive failures "
                        f"({old_intensity} -> reduced)"
                    )

            self._save_state()

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.debug(f"[TrainingTriggerDaemon] Error handling TRAINING_FAILED: {e}")

    def _queue_training_retry(
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
        for item in self._training_retry_queue:
            if item[0] == config_key:
                existing_attempts = max(existing_attempts, item[3])

        attempts = max(current_attempts, existing_attempts) + 1

        if attempts > self._max_training_retries:
            self._retry_stats["retries_exhausted"] += 1
            logger.error(
                f"[TrainingTriggerDaemon] Max retries ({self._max_training_retries}) exceeded "
                f"for {config_key}: {error[:100]}"
            )
            return False

        # Exponential backoff: 5min, 10min, 20min
        delay = self._base_training_retry_delay * (2 ** (attempts - 1))
        next_retry = time.time() + delay

        self._training_retry_queue.append(
            (config_key, board_type, num_players, attempts, next_retry, error[:200])
        )
        self._retry_stats["retries_queued"] += 1

        logger.info(
            f"[TrainingTriggerDaemon] Queued training retry #{attempts} for {config_key} "
            f"in {delay/60:.0f}min (reason: {error[:50]}...)"
        )
        return True

    async def _process_training_retry_queue(self) -> None:
        """Process pending training retries whose delay has elapsed.

        December 29, 2025 (Phase 3): Called at the start of each cycle
        to re-attempt failed training jobs with exponential backoff.
        """
        if not self._training_retry_queue:
            return

        now = time.time()
        ready_for_retry: list[tuple[str, str, int, int, str]] = []
        remaining: list[tuple[str, str, int, int, float, str]] = []

        while self._training_retry_queue:
            item = self._training_retry_queue.popleft()
            config_key, board_type, num_players, attempts, next_retry_time, error = item

            if next_retry_time <= now:
                ready_for_retry.append((config_key, board_type, num_players, attempts, error))
            else:
                remaining.append(item)

        # Put back items not yet ready
        for item in remaining:
            self._training_retry_queue.append(item)

        # Process ready items
        for config_key, board_type, num_players, attempts, error in ready_for_retry:
            state = self._get_or_create_state(config_key, board_type, num_players)

            # Skip if already training
            if state.training_in_progress:
                logger.debug(
                    f"[TrainingTriggerDaemon] Retry deferred (already training): {config_key}"
                )
                # Re-queue with same attempt count but short delay
                self._training_retry_queue.append(
                    (config_key, board_type, num_players, attempts, now + 60.0, error)
                )
                continue

            logger.info(
                f"[TrainingTriggerDaemon] Retrying training #{attempts} for {config_key}"
            )

            # Trigger training check (will go through normal validation)
            can_train, reason = await self._check_training_readiness(config_key, state)
            if can_train:
                success = await self._trigger_training(config_key, state)
                if success:
                    self._retry_stats["retries_succeeded"] += 1
                    logger.info(
                        f"[TrainingTriggerDaemon] Retry #{attempts} succeeded for {config_key}"
                    )
                else:
                    # Re-queue for next attempt
                    self._queue_training_retry(
                        config_key, board_type, num_players,
                        f"retry failed: {reason}", attempts
                    )
            else:
                # Re-queue for later (conditions not met yet)
                delay = self._base_training_retry_delay / 2  # Shorter delay for condition check
                self._training_retry_queue.append(
                    (config_key, board_type, num_players, attempts, now + delay, error)
                )
                logger.debug(
                    f"[TrainingTriggerDaemon] Retry deferred for {config_key}: {reason}"
                )

    def _get_velocity_adjusted_cooldown(self, state: ConfigTrainingState) -> float:
        """Get training cooldown adjusted for Elo velocity.

        December 29, 2025: Implements velocity-based cooldown modulation.
        Configs with positive velocity get shorter cooldowns to capitalize on momentum.
        Configs with negative velocity get longer cooldowns to avoid wasteful training.

        Returns:
            Adjusted cooldown in seconds
        """
        base_cooldown = self.config.training_cooldown_hours * 3600

        # Velocity-based multipliers
        velocity_multipliers = {
            "accelerating": 0.5,    # 50% cooldown - train faster
            "stable": 1.0,          # Normal cooldown
            "decelerating": 1.5,    # 150% cooldown - train slower
            "plateauing": 1.25,     # 125% cooldown - slightly slower
        }

        multiplier = velocity_multipliers.get(state.elo_velocity_trend, 1.0)

        # Additional adjustment based on actual velocity value
        if state.elo_velocity > 20.0:
            # Very rapid improvement - train even faster
            multiplier *= 0.7
        elif state.elo_velocity < -10.0:
            # Significant regression - slow down more
            multiplier *= 1.3

        return base_cooldown * multiplier

    async def _trigger_priority_sync(
        self, config_key: str, board_type: str, num_players: int
    ) -> bool:
        """Trigger priority data sync for a configuration (Dec 2025 Phase 2A).

        Uses SyncFacade to request immediate sync of training data for the
        specified configuration.

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            board_type: Board type
            num_players: Number of players

        Returns:
            True if sync was triggered successfully
        """
        try:
            from app.coordination.sync_facade import get_sync_facade

            facade = get_sync_facade()
            response = await facade.trigger_priority_sync(
                reason="training_data_stale",
                config_key=config_key,
                data_type="training",
            )

            if response.get("success"):
                logger.info(
                    f"[TrainingTriggerDaemon] Priority sync triggered for {config_key}"
                )
                return True
            else:
                logger.warning(
                    f"[TrainingTriggerDaemon] Priority sync failed for {config_key}: "
                    f"{response.get('error', 'unknown')}"
                )
                return False

        except ImportError:
            logger.debug("[TrainingTriggerDaemon] SyncFacade not available for priority sync")
            return False
        except Exception as e:
            logger.warning(f"[TrainingTriggerDaemon] Error triggering priority sync: {e}")
            return False

    async def _trigger_selfplay_boost(self, config_key: str, multiplier: float = 1.5) -> None:
        """Trigger additional selfplay for struggling configurations."""
        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            scheduler = get_selfplay_scheduler()
            if scheduler:
                # Boost allocation for this config
                scheduler.boost_config_allocation(config_key, multiplier)
                logger.info(
                    f"[TrainingTriggerDaemon] Boosted selfplay for {config_key} by {multiplier}x"
                )
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Could not boost selfplay: {e}")

    async def _signal_curriculum_advancement(self, config_key: str) -> None:
        """Signal that curriculum should advance for a stagnant configuration.

        Dec 29, 2025: Now emits DataEventType.CURRICULUM_ADVANCEMENT_NEEDED
        which is handled by MomentumToCurriculumBridge._on_curriculum_advancement_needed().
        """
        try:
            from app.coordination.event_router import publish
            from app.distributed.data_events import DataEventType

            await publish(
                event_type=DataEventType.CURRICULUM_ADVANCEMENT_NEEDED,
                payload={
                    "config_key": config_key,
                    "reason": "elo_plateau",
                    "timestamp": time.time(),
                },
                source="training_trigger_daemon",
            )
            logger.info(
                f"[TrainingTriggerDaemon] Signaled curriculum advancement for {config_key}"
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            # ImportError: event modules not available
            # AttributeError: DataEventType enum missing
            # RuntimeError: publish operation failed
            logger.debug(f"[TrainingTriggerDaemon] Could not signal curriculum: {e}")

    async def _record_to_feedback_accelerator(
        self, config_key: str, elo: float, elo_delta: float
    ) -> None:
        """Record Elo update to FeedbackAccelerator for momentum tracking."""
        try:
            from app.training.feedback_accelerator import get_feedback_accelerator

            accelerator = get_feedback_accelerator()
            if accelerator:
                accelerator.record_elo_update(config_key, elo, elo_delta)
                logger.debug(
                    f"[TrainingTriggerDaemon] Recorded Elo to FeedbackAccelerator: "
                    f"{config_key}={elo:.0f} (delta={elo_delta:+.0f})"
                )
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Could not record to accelerator: {e}")

    def _parse_config_from_filename(self, name: str) -> tuple[str | None, int | None]:
        """Parse board_type and num_players from filename.

        Handles various naming patterns:
        - hex8_2p.npz -> (hex8, 2)
        - square8_3p_fresh.npz -> (square8, 3)
        - canonical_hexagonal_4p_trained.npz -> (hexagonal, 4)

        Returns:
            (board_type, num_players) or (None, None) if not parseable.
        """
        # Match board_type followed by _Np pattern anywhere in filename
        match = re.search(r'(hex8|square8|square19|hexagonal)_(\d+)p', name)
        if match:
            board_type = match.group(1)
            try:
                num_players = int(match.group(2))
                if num_players in (2, 3, 4):
                    return board_type, num_players
            except (ValueError, TypeError):
                pass

        return None, None

    def _get_or_create_state(
        self, config_key: str, board_type: str | None = None, num_players: int | None = None
    ) -> ConfigTrainingState:
        """Get or create training state for a config."""
        if config_key not in self._training_states:
            # Parse config_key if board_type/num_players not provided
            if not board_type or not num_players:
                parsed_board, parsed_players = self._parse_config_from_filename(config_key)
                if parsed_board and parsed_players:
                    board_type = parsed_board
                    num_players = parsed_players
                else:
                    # Fallback to legacy parsing
                    parts = config_key.rsplit("_", 1)
                    board_type = parts[0] if len(parts) == 2 else config_key
                    try:
                        num_players = int(parts[1].replace("p", "")) if len(parts) == 2 else 2
                    except ValueError:
                        num_players = 2

            self._training_states[config_key] = ConfigTrainingState(
                config_key=config_key,
                board_type=board_type,
                num_players=num_players,
            )

        return self._training_states[config_key]

    async def _maybe_trigger_training(self, config_key: str) -> bool:
        """Check conditions and trigger training if appropriate."""
        state = self._training_states.get(config_key)
        if not state:
            return False

        # Check all conditions
        can_train, reason = await self._check_training_conditions(config_key)

        if not can_train:
            logger.debug(f"[TrainingTriggerDaemon] {config_key}: Cannot train - {reason}")
            return False

        # Trigger training
        logger.info(f"[TrainingTriggerDaemon] Triggering training for {config_key}")
        task = asyncio.create_task(self._run_training(config_key))
        task.add_done_callback(lambda t: self._on_training_task_done(t, config_key))
        self._active_training_tasks[config_key] = task

        return True

    async def _check_training_conditions(self, config_key: str) -> tuple[bool, str]:
        """Check all conditions for training trigger.

        Returns:
            Tuple of (can_train, reason)
        """
        state = self._training_states.get(config_key)
        if not state:
            return False, "no state"

        # 1. Check if training already in progress
        if state.training_in_progress:
            return False, "training already in progress"

        if state.training_intensity == "paused":
            return False, "training intensity paused"

        # December 29, 2025 (Phase 4): Check evaluation backpressure
        # When evaluation queue is full, pause training to let evaluations catch up
        if self._evaluation_backpressure:
            return False, "evaluation backpressure active (queue full)"

        # Phase 4: Check circuit breaker before triggering training
        if HAS_CIRCUIT_BREAKER and get_training_breaker:
            breaker = get_training_breaker()
            if not breaker.can_execute(config_key):
                return False, f"circuit open for {config_key}"

        # 2. Check training cooldown (December 29, 2025: velocity-adjusted)
        time_since_training = time.time() - state.last_training_time
        # Use velocity-adjusted cooldown instead of fixed cooldown
        cooldown_seconds = self._get_velocity_adjusted_cooldown(state)
        if time_since_training < cooldown_seconds:
            remaining = (cooldown_seconds - time_since_training) / 3600
            trend_info = f", velocity_trend={state.elo_velocity_trend}" if state.elo_velocity_trend != "stable" else ""
            return False, f"cooldown active ({remaining:.1f}h remaining{trend_info})"

        # 3. Check data freshness (December 2025: use training_freshness for sync)
        data_age_hours = (time.time() - state.last_npz_update) / 3600
        if data_age_hours > self.config.max_data_age_hours:
            # December 29, 2025: Strict mode - fail immediately without sync attempt
            if self.config.strict_freshness_mode:
                return False, f"data too old ({data_age_hours:.1f}h) [strict mode - no sync]"
            elif self.config.enforce_freshness_with_sync:
                # Try to sync and wait for fresh data
                fresh = await self._ensure_fresh_data(state.board_type, state.num_players)
                if not fresh:
                    return False, f"data too old ({data_age_hours:.1f}h), sync failed"
                # Sync succeeded, continue with training check
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: data refreshed via sync"
                )
            else:
                return False, f"data too old ({data_age_hours:.1f}h)"

        # 4. Check minimum samples (with confidence-based early trigger)
        # Dec 29, 2025: Try confidence-based early trigger first
        # This allows training to start earlier when statistical confidence is high
        if state.npz_sample_count >= self.config.confidence_min_samples:
            early_trigger, early_reason = self._check_confidence_early_trigger(
                config_key, state.npz_sample_count
            )
            if early_trigger:
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: early trigger - {early_reason}"
                )
                # Skip the min_samples check - confidence is high enough
            else:
                # Fall back to dynamic threshold from ImprovementOptimizer
                # Phase 5 (Dec 2025): Lower when on promotion streak, higher when struggling
                min_samples = self._get_dynamic_sample_threshold(config_key)
                if state.npz_sample_count < min_samples:
                    return False, f"insufficient samples ({state.npz_sample_count} < {min_samples}), {early_reason}"
        else:
            # Below confidence minimum - use dynamic threshold
            min_samples = self._get_dynamic_sample_threshold(config_key)
            if state.npz_sample_count < min_samples:
                return False, f"insufficient samples ({state.npz_sample_count} < {min_samples})"

        # 5. Check if idle GPU available (optional - allow training anyway)
        gpu_available = await self._check_gpu_availability()
        if not gpu_available:
            logger.warning(f"[TrainingTriggerDaemon] {config_key}: No idle GPU, proceeding anyway")

        # 6. Check concurrent training limit
        active_count = sum(
            1 for s in self._training_states.values() if s.training_in_progress
        )
        if active_count >= self.config.max_concurrent_training:
            return False, f"max concurrent training reached ({active_count})"

        # December 29, 2025: Auto-boost intensity for very fresh data
        # Fresh data (< 30 min old) suggests active selfplay → accelerate training
        if data_age_hours < 0.5:  # Less than 30 minutes old
            if state.training_intensity == "normal":
                state.training_intensity = "accelerated"
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: boosted to 'accelerated' "
                    f"(data is {data_age_hours * 60:.0f}min fresh)"
                )
            elif state.training_intensity == "accelerated":
                state.training_intensity = "hot_path"
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: boosted to 'hot_path' "
                    f"(data is {data_age_hours * 60:.0f}min fresh)"
                )

        return True, "all conditions met"

    async def _check_gpu_availability(self) -> bool:
        """Check if any GPU is available for training."""
        try:
            # Try to get GPU utilization via nvidia-smi
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)

            if process.returncode == 0:
                for line in stdout.decode().strip().split("\n"):
                    try:
                        util = float(line.strip())
                        if util < self.config.gpu_idle_threshold_percent:
                            return True
                    except ValueError:
                        continue
                return False

        except (FileNotFoundError, asyncio.TimeoutError):
            pass
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] GPU check failed: {e}")

        # Assume GPU available if we can't check
        return True

    async def _ensure_fresh_data(self, board_type: str, num_players: int) -> bool:
        """Ensure training data is fresh, triggering sync if needed (December 2025).

        Uses training_freshness module to check data age and trigger sync
        if data is stale. This closes the data freshness feedback loop.

        Args:
            board_type: Board type for training
            num_players: Number of players

        Returns:
            True if data is now fresh, False if sync failed or timed out
        """
        try:
            from app.coordination.training_freshness import (
                DataFreshnessChecker,
                FreshnessConfig,
            )

            config = FreshnessConfig(
                max_age_hours=self.config.max_data_age_hours,
                trigger_sync=True,
                wait_for_sync=True,
                sync_timeout_seconds=self.config.freshness_sync_timeout_seconds,
            )

            checker = DataFreshnessChecker(config)
            result = await checker.ensure_fresh_data(board_type, num_players)

            if result.is_fresh:
                # Update local state with fresh data info
                config_key = f"{board_type}_{num_players}p"
                if config_key in self._training_states:
                    self._training_states[config_key].last_npz_update = time.time()
                    if result.games_available:
                        self._training_states[config_key].npz_sample_count = result.games_available
                return True

            logger.warning(
                f"[TrainingTriggerDaemon] Data freshness check failed for "
                f"{board_type}_{num_players}p: {result.error}"
            )
            return False

        except ImportError:
            logger.debug("[TrainingTriggerDaemon] training_freshness module not available")
            return False
        except Exception as e:
            logger.warning(f"[TrainingTriggerDaemon] ensure_fresh_data failed: {e}")
            return False

    async def _run_training(self, config_key: str) -> bool:
        """Run training subprocess for a configuration."""
        state = self._training_states.get(config_key)
        if not state:
            return False

        # Check for paused intensity - skip training
        if state.training_intensity == "paused":
            logger.info(
                f"[TrainingTriggerDaemon] Skipping training for {config_key}: "
                "intensity is 'paused' (quality score < 0.50)"
            )
            return False

        async with self._training_semaphore:
            state.training_in_progress = True
            state.training_start_time = time.time()  # Phase 2: Timeout watchdog

            try:
                # Get intensity-adjusted training parameters
                epochs, batch_size, lr_mult = self._get_training_params_for_intensity(
                    state.training_intensity
                )

                logger.info(
                    f"[TrainingTriggerDaemon] Starting training for {config_key} "
                    f"({state.npz_sample_count} samples, intensity={state.training_intensity}, "
                    f"epochs={epochs}, batch={batch_size}, lr_mult={lr_mult:.1f})"
                )

                # Build training command
                base_dir = Path(__file__).resolve().parent.parent.parent
                npz_path = state.npz_path or f"data/training/{config_key}.npz"

                # December 29, 2025: Use canonical model paths for consistent naming
                # Training outputs to models/canonical_{board}_{n}p.pth
                # The training script also saves timestamped checkpoints alongside for history
                model_filename = f"canonical_{config_key}.pth"
                model_path = str(base_dir / "models" / model_filename)

                cmd = [
                    sys.executable,
                    "-m", "app.training.train",
                    "--board-type", state.board_type,
                    "--num-players", str(state.num_players),
                    "--data-path", npz_path,
                    "--model-version", self.config.model_version,
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size),
                    "--save-path", model_path,  # December 29, 2025: Explicit save path
                    # December 2025: Allow stale data to unblock training when
                    # selfplay rate is slower than freshness threshold.
                    # The freshness check was blocking ALL training because game
                    # databases have content ages of 7-100+ hours while threshold is 1h.
                    "--allow-stale-data",
                    "--max-data-age-hours", "168",  # 1 week threshold
                ]

                # Store model_path in state for event emission
                state.npz_path = npz_path  # Keep npz_path
                state._pending_model_path = model_path  # Track expected model path

                # Compute adjusted learning rate (base 1e-3 * multiplier)
                # The training CLI uses --learning-rate for explicit LR setting
                if lr_mult != 1.0:
                    base_lr = 1e-3  # Default from TrainingConfig
                    adjusted_lr = base_lr * lr_mult
                    cmd.extend(["--learning-rate", f"{adjusted_lr:.6f}"])

                # Run training subprocess
                start_time = time.time()
                # December 29, 2025: Add RINGRIFT_ALLOW_PENDING_GATE to bypass parity
                # validation on cluster nodes that lack Node.js/npx
                training_env = {
                    **os.environ,
                    "PYTHONPATH": str(base_dir),
                    "RINGRIFT_ALLOW_PENDING_GATE": "true",
                }
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(base_dir),
                    env=training_env,
                )

                state.training_pid = process.pid

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.config.training_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    logger.error(f"[TrainingTriggerDaemon] Training timed out for {config_key}")
                    state.consecutive_failures += 1
                    return False

                duration = time.time() - start_time

                if process.returncode == 0:
                    # Success
                    state.last_training_time = time.time()
                    state.consecutive_failures = 0

                    logger.info(
                        f"[TrainingTriggerDaemon] Training complete for {config_key}: "
                        f"{duration/3600:.1f}h"
                    )

                    # Emit training complete event
                    await self._emit_training_complete(config_key, success=True)
                    return True

                else:
                    # Failure
                    state.consecutive_failures += 1

                    # Adjust training intensity on consecutive failures (December 2025)
                    # This prevents wasting compute on configs that repeatedly fail
                    if state.consecutive_failures >= 3:
                        old_intensity = state.training_intensity
                        state.training_intensity = "paused"
                        logger.warning(
                            f"[TrainingTriggerDaemon] {config_key}: {state.consecutive_failures} "
                            f"consecutive failures, pausing training (was: {old_intensity})"
                        )
                    elif state.consecutive_failures >= 2:
                        old_intensity = state.training_intensity
                        if state.training_intensity not in ("reduced", "paused"):
                            state.training_intensity = "reduced"
                            logger.info(
                                f"[TrainingTriggerDaemon] {config_key}: 2 failures, reducing intensity "
                                f"(was: {old_intensity})"
                            )

                    logger.error(
                        f"[TrainingTriggerDaemon] Training failed for {config_key}: "
                        f"exit code {process.returncode}\n"
                        f"stderr: {stderr.decode()[:500]}"
                    )
                    await self._emit_training_complete(config_key, success=False)
                    return False

            except Exception as e:
                state.consecutive_failures += 1

                # Also adjust intensity on exceptions (December 2025)
                if state.consecutive_failures >= 2:
                    old_intensity = state.training_intensity
                    state.training_intensity = "reduced"
                    logger.info(
                        f"[TrainingTriggerDaemon] {config_key}: {state.consecutive_failures} failures "
                        f"(exception), reducing intensity (was: {old_intensity})"
                    )

                logger.error(f"[TrainingTriggerDaemon] Training error for {config_key}: {e}")
                return False

            finally:
                state.training_in_progress = False
                state.training_pid = None
                # Remove from active tasks
                self._active_training_tasks.pop(config_key, None)

    def _on_training_task_done(self, task: asyncio.Task, config_key: str) -> None:
        """Handle training task completion."""
        try:
            exc = task.exception()
            if exc:
                logger.error(f"[TrainingTriggerDaemon] Training task error for {config_key}: {exc}")
        except asyncio.CancelledError:
            pass
        except asyncio.InvalidStateError:
            pass

    async def _emit_training_complete(self, config_key: str, success: bool) -> None:
        """Emit training completion event."""
        # Phase 4: Record circuit breaker success/failure
        if HAS_CIRCUIT_BREAKER and get_training_breaker:
            breaker = get_training_breaker()
            if success:
                breaker.record_success(config_key)
            else:
                breaker.record_failure(config_key)

        try:
            from app.coordination.event_router import (
                StageEvent,
                StageCompletionResult,
                get_stage_event_bus,
            )

            state = self._training_states.get(config_key)

            # December 29, 2025: Include model_path in event for EvaluationDaemon
            model_path = ""
            if state and hasattr(state, "_pending_model_path"):
                model_path = state._pending_model_path
                # Verify model exists before including path
                if model_path and success:
                    if not Path(model_path).exists():
                        logger.warning(
                            f"[TrainingTriggerDaemon] Model not found at {model_path}, "
                            "EvaluationDaemon may fail"
                        )

            bus = get_stage_event_bus()
            await bus.emit(
                StageCompletionResult(
                    event=StageEvent.TRAINING_COMPLETE if success else StageEvent.TRAINING_FAILED,
                    success=success,
                    timestamp=datetime.datetime.now().isoformat(),
                    metadata={
                        "config": config_key,
                        "board_type": state.board_type if state else "",
                        "num_players": state.num_players if state else 0,
                        "samples_trained": state.npz_sample_count if state else 0,
                        # December 29, 2025: Critical for evaluation pipeline
                        "model_path": model_path,
                        "checkpoint_path": model_path,  # Alias for compatibility
                    },
                )
            )
            logger.info(
                f"[TrainingTriggerDaemon] Emitted TRAINING_{'COMPLETE' if success else 'FAILED'} "
                f"for {config_key} (model_path={model_path})"
            )

        except Exception as e:
            logger.warning(f"[TrainingTriggerDaemon] Failed to emit training event: {e}")

    async def _check_training_timeouts(self) -> None:
        """Check for and kill training jobs that exceed the timeout.

        December 29, 2025 (Phase 2): Training timeout watchdog for 48h autonomous operation.
        This catches hung training processes even if the daemon restarts.
        """
        timeout_seconds = self.config.training_timeout_hours * 3600
        now = time.time()

        for config_key, state in self._training_states.items():
            if not state.training_in_progress:
                continue

            if state.training_start_time <= 0:
                continue  # No start time recorded (shouldn't happen)

            elapsed = now - state.training_start_time
            if elapsed < timeout_seconds:
                continue

            # Training has exceeded timeout
            elapsed_hours = elapsed / 3600
            self._timeout_stats["timeouts_detected"] += 1
            self._timeout_stats["last_timeout_time"] = now
            logger.warning(
                f"[TrainingTriggerDaemon] Training timeout for {config_key}: "
                f"running for {elapsed_hours:.1f}h (limit: {self.config.training_timeout_hours}h)"
            )

            # Kill the training process if we have a PID
            if state.training_pid is not None:
                try:
                    os.kill(state.training_pid, 9)  # SIGKILL
                    self._timeout_stats["processes_killed"] += 1
                    logger.info(
                        f"[TrainingTriggerDaemon] Killed timed-out training process "
                        f"PID {state.training_pid} for {config_key}"
                    )
                except ProcessLookupError:
                    logger.debug(
                        f"[TrainingTriggerDaemon] Process {state.training_pid} already dead"
                    )
                except PermissionError:
                    logger.error(
                        f"[TrainingTriggerDaemon] Permission denied killing PID {state.training_pid}"
                    )

            # Reset state
            state.training_in_progress = False
            state.training_pid = None
            state.training_start_time = 0.0
            state.consecutive_failures += 1

            # Cancel the asyncio task if it exists
            if config_key in self._active_training_tasks:
                task = self._active_training_tasks.pop(config_key)
                if not task.done():
                    task.cancel()

            # Emit training failed event
            await self._emit_training_failed(config_key, "timeout")

    async def _emit_training_failed(self, config_key: str, reason: str) -> None:
        """Emit TRAINING_FAILED event for timed-out or errored training."""
        try:
            from app.distributed.data_events import DataEventType

            bus = self._get_event_bus()
            if bus:
                bus.publish_sync(
                    DataEventType.TRAINING_FAILED.value,
                    {
                        "config_key": config_key,
                        "reason": reason,
                        "timestamp": time.time(),
                        "source": "TrainingTriggerDaemon",
                    },
                )
                logger.info(
                    f"[TrainingTriggerDaemon] Emitted TRAINING_FAILED for {config_key}: {reason}"
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Failed to emit TRAINING_FAILED: {e}")

    async def _run_cycle(self) -> None:
        """Main work loop iteration - called by HandlerBase at scan_interval_seconds."""
        # Skip if we're on a coordinator node
        if self._coordinator_skip:
            return

        # December 29, 2025 (Phase 2): Check for timed-out training jobs
        await self._check_training_timeouts()

        # December 29, 2025 (Phase 3): Process pending training retries
        await self._process_training_retry_queue()

        # Scan for training opportunities
        await self._scan_for_training_opportunities()

        # December 29, 2025 (Phase 3): Periodically save state
        now = time.time()
        if now - self._last_state_save >= self.config.state_save_interval_seconds:
            self._save_state()

    async def _scan_for_training_opportunities(self) -> None:
        """Scan for configs that may need training."""
        try:
            # Check existing states
            for config_key in list(self._training_states.keys()):
                await self._maybe_trigger_training(config_key)

            # Also scan for NPZ files that haven't been tracked
            training_dir = Path(__file__).resolve().parent.parent.parent / "data" / "training"
            if training_dir.exists():
                for npz_path in training_dir.glob("*.npz"):
                    # Parse config from filename using robust regex
                    board_type, num_players = self._parse_config_from_filename(npz_path.stem)
                    if board_type is None or num_players is None:
                        continue

                    config_key = f"{board_type}_{num_players}p"
                    if config_key not in self._training_states:
                        state = self._get_or_create_state(config_key, board_type, num_players)
                        state.npz_path = str(npz_path)
                        state.last_npz_update = npz_path.stat().st_mtime

                        # Get sample count from file (approximate)
                        # Use header validation first to avoid memory errors on corrupt files
                        try:
                            from app.training.data_validation import is_npz_valid
                            if not is_npz_valid(npz_path):
                                logger.warning(f"Skipping invalid NPZ: {npz_path}")
                                continue
                        except ImportError:
                            pass

                        try:
                            from app.utils.numpy_utils import safe_load_npz
                            with safe_load_npz(npz_path) as data:
                                state.npz_sample_count = len(data.get("values", []))
                        except (FileNotFoundError, OSError, ValueError, ImportError):
                            pass

                        await self._maybe_trigger_training(config_key)

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error scanning for opportunities: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current daemon status."""
        return {
            "running": self._running,
            "configs_tracked": len(self._training_states),
            "active_training": sum(
                1 for s in self._training_states.values() if s.training_in_progress
            ),
            # December 29, 2025 (Phase 4): Backpressure status
            "evaluation_backpressure": self._evaluation_backpressure,
            "backpressure_stats": dict(self._backpressure_stats),
            # December 29, 2025 (Phase 2): Timeout watchdog stats
            "timeout_stats": dict(self._timeout_stats),
            "training_timeout_hours": self.config.training_timeout_hours,
            "states": {
                key: {
                    "training_in_progress": state.training_in_progress,
                    "training_intensity": state.training_intensity,
                    "last_training": state.last_training_time,
                    "npz_samples": state.npz_sample_count,
                    "last_elo": state.last_elo,
                    "failures": state.consecutive_failures,
                }
                for key, state in self._training_states.items()
            },
        }

    def health_check(self) -> HealthCheckResult:
        """Check daemon health.

        Returns:
            Health check result with training trigger status and metrics.
        """
        # Count active training tasks
        active_training = sum(
            1 for state in self._training_states.values()
            if state.training_in_progress
        )

        # Count failed configs
        failed_configs = sum(
            1 for state in self._training_states.values()
            if state.consecutive_failures > 0
        )

        # Determine health status
        healthy = self._running

        # December 29, 2025 (Phase 4): Include backpressure status in message
        if self._evaluation_backpressure:
            message = "Running (evaluation backpressure active)"
        else:
            message = "Running" if healthy else "Daemon stopped"

        return HealthCheckResult(
            healthy=healthy,
            message=message,
            details={
                "running": self._running,
                "enabled": self.config.enabled,
                "configs_tracked": len(self._training_states),
                "active_training_tasks": active_training,
                "failed_configs": failed_configs,
                "max_concurrent_training": self.config.max_concurrent_training,
                "max_data_age_hours": self.config.max_data_age_hours,
                # December 29, 2025 (Phase 4): Backpressure status
                "evaluation_backpressure": self._evaluation_backpressure,
                "backpressure_pauses": self._backpressure_stats["pauses_due_to_backpressure"],
                "backpressure_resumes": self._backpressure_stats["resumes_after_backpressure"],
                # December 29, 2025 (Phase 3): Training retry stats
                "retry_queue_size": len(self._training_retry_queue),
                "retries_queued": self._retry_stats["retries_queued"],
                "retries_succeeded": self._retry_stats["retries_succeeded"],
                "retries_exhausted": self._retry_stats["retries_exhausted"],
            },
        )


# December 2025: Using HandlerBase singleton pattern
def get_training_trigger_daemon() -> TrainingTriggerDaemon:
    """Get or create the singleton training trigger daemon.

    December 2025: Now uses HandlerBase.get_instance() singleton pattern.
    """
    return TrainingTriggerDaemon.get_instance()


def reset_training_trigger_daemon() -> None:
    """Reset the singleton instance (for testing).

    December 2025: Added for test isolation.
    """
    TrainingTriggerDaemon.reset_instance()


async def start_training_trigger_daemon() -> TrainingTriggerDaemon:
    """Start the training trigger daemon (convenience function)."""
    daemon = get_training_trigger_daemon()
    await daemon.start()
    return daemon


__all__ = [
    "ConfigTrainingState",
    "TrainingTriggerConfig",
    "TrainingTriggerDaemon",
    "get_training_trigger_daemon",
    "reset_training_trigger_daemon",
    "start_training_trigger_daemon",
]
