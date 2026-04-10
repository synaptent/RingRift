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
import contextlib
import datetime
import logging
import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from app.config.coordination_defaults import (
    QualityGateDefaults,
    SyncDefaults,
)
from app.config.env import env
from app.config.ports import get_local_p2p_status_url
from app.coordination.event_handler_utils import extract_config_from_path, extract_config_key
from app.coordination.event_utils import make_config_key, parse_config_key
from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.training_executor_actions import TrainingExecutorActionsMixin
from app.coordination.training_trigger_types import (
    TRIGGER_DEDUP_WINDOW_SECONDS,
    TrainingTriggerConfig,
    ConfigTrainingState,
    TrainingDecision,
    ArchitectureSpec,
    MultiArchitectureConfig,
)
# Jan 4, 2026 - Sprint 17.9: Quality gate functions moved to training_quality_gates.py
from app.coordination.training_quality_gates import (
    compute_quality_confidence,
    apply_confidence_weighting,
    compute_decayed_quality_score,
    intensity_from_quality,
    check_quality_gate_conditions,
    get_quality_from_state,
    QualityGateResult,
    MINIMUM_QUALITY_FLOOR,
    DATA_STARVED_THRESHOLD,
    TRAINING_STALL_HOURS,
)
# Jan 4, 2026 - Sprint 17.9: Execution functions moved to training_execution.py
from app.coordination.training_execution import (
    TrainingExecutor,
    TrainingExecutionConfig,
    TrainingResult,
    graceful_kill_process as _graceful_kill_process_impl,
    emit_training_complete as _emit_training_complete_impl,
    emit_training_failed as _emit_training_failed_impl,
)
# Jan 9, 2026: Architecture selection functions moved to training_architecture_selector.py
from app.coordination.training_architecture_selector import (
    get_training_params_for_intensity,
    select_architecture_for_training,
    apply_velocity_amplification,
)
# Jan 9, 2026: Data availability functions moved to training_data_availability.py
from app.coordination.training_data_availability import (
    DataAvailabilityChecker,
    DataAvailabilityConfig,
    check_gpu_availability,
    check_cluster_availability,
    scan_local_npz_files,
    parse_config_from_filename,
)
# Jan 9, 2026: Retry management utilities moved to training_retry_manager.py
from app.coordination.training_retry_manager import (
    get_velocity_adjusted_cooldown,
    get_adaptive_max_data_age,
    RetryQueueConfig,
)
# Feb 2026: Pure decision functions extracted to training_decision_engine.py
from app.coordination.training_decision_engine import (
    compute_velocity_adjusted_cooldown,
    compute_dynamic_sample_threshold,
    check_confidence_early_trigger as check_confidence_early_trigger_fn,
    compute_adaptive_max_data_age,
)
from app.utils.retry import RetryConfig

logger = logging.getLogger(__name__)

# Circuit breaker integration (Phase 4 - December 2025)
try:
    from app.distributed.circuit_breaker import get_training_breaker
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    get_training_breaker = None

# Distributed lock integration (January 2026 - Sprint 3)
# Prevents duplicate training jobs across cluster nodes
try:
    from app.coordination.p2p_integration import (
        with_training_lock,
        is_p2p_available,
    )
    HAS_DISTRIBUTED_LOCK = True
except ImportError:
    HAS_DISTRIBUTED_LOCK = False
    with_training_lock = None  # type: ignore
    is_p2p_available = None  # type: ignore


# Jan 4, 2026 - Sprint 17.9: Type definitions moved to training_trigger_types.py
# The following are imported from that module for backward compatibility:
# - TRIGGER_DEDUP_WINDOW_SECONDS
# - TrainingTriggerConfig
# - ConfigTrainingState
# - TrainingDecision
# - ArchitectureSpec
# - MultiArchitectureConfig


class TrainingTriggerDaemon(TrainingExecutorActionsMixin, HandlerBase):
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
        # Sprint 16.1 (Jan 3, 2026): Track pending quality rechecks to avoid duplicates
        self._pending_quality_rechecks: dict[str, asyncio.Task] = {}
        # Track whether we should skip due to coordinator mode (DEPRECATED - use _dispatch_to_queue)
        self._coordinator_skip = False
        # Dec 30, 2025: When True, dispatch training to work queue instead of running locally
        # This allows coordinator nodes to trigger training on remote GPU nodes
        self._dispatch_to_queue = False
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
        # January 8, 2026: Added maxlen=100 to prevent unbounded queue growth
        self._training_retry_queue: deque[tuple[str, str, int, int, float, str]] = deque(maxlen=100)
        # December 30, 2025: Use centralized RetryConfig for consistent retry behavior
        self._retry_config = RetryConfig(
            max_attempts=3,
            base_delay=300.0,  # 5 minutes
            max_delay=1200.0,  # 20 minutes
            jitter=0.1,  # Add slight jitter to avoid thundering herd
        )
        # January 2026: Lazy-loaded UnifiedGameAggregator for cluster-wide game counts
        self._game_aggregator = None
        # January 5, 2026 (Phase 7.9): Quality assessment cache to reduce SQLite lookups
        # Expected improvement: +2-4 Elo from reduced quality check latency
        self._quality_cache: dict[str, tuple[float, float]] = {}  # config -> (score, timestamp)
        # Session 17.46 (Jan 6, 2026): Extended from 10s to 60s for +2-4 Elo improvement.
        # 10s caused repeated SQLite queries for quality assessment.
        # 60s cache is sufficient since quality changes slowly (game-level updates).
        self._quality_cache_ttl = 60.0  # 60 second cache TTL
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
        # January 3, 2026: NPZ discovery event-driven cache
        # Caches NPZ metadata from events to skip redundant disk scans during _run_cycle
        # Key: config_key, Value: (mtime, sample_count, path)
        self._npz_cache: dict[str, tuple[float, int, str]] = {}
        # December 30, 2025: Multi-architecture training support
        # Tracks training per (config_key, architecture) tuple
        self._architecture_config = MultiArchitectureConfig.load()
        # Track last training time per (config_key, architecture)
        self._architecture_training_times: dict[tuple[str, str], float] = {}
        # Track active training per architecture
        self._active_architecture_training: dict[tuple[str, str], bool] = {}
        # Jan 2, 2026: Local-only mode for training without cluster connectivity
        # When enabled, skips cluster GPU checks and uses only local NPZ files
        self._local_only_mode: bool = self._daemon_config.local_only_mode
        self._cluster_available: bool = True  # Assume available until checked
        # Feb 2026: Periodic Elo sync from unified_elo.db to prevent stale last_elo values.
        # Without this, configs that don't receive EVALUATION_COMPLETED events stay at
        # default 1500 Elo, causing incorrect simulation budgets and training intensity.
        self._last_elo_db_sync: float = 0.0
        self._elo_db_sync_interval: float = 300.0  # Sync every 5 minutes
        # Mar 2026: Track last failure time per config for auto-recovery.
        # When consecutive_failures causes "paused"/"reduced" intensity, this allows
        # automatic recovery after a cooldown period without requiring manual restart.
        self._last_failure_time: dict[str, float] = {}
        self._failure_recovery_cooldown: float = 1800.0  # 30 min before auto-recovery

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

                        # January 2026: Validate board_type loaded from SQLite
                        board_type = row["board_type"]
                        if board_type and not isinstance(board_type, str):
                            logger.warning(
                                f"[TrainingTriggerDaemon] Invalid board_type in persisted state for {config_key}"
                            )
                            continue  # Skip this corrupted entry

                        state = ConfigTrainingState(
                            config_key=config_key,
                            board_type=board_type,
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
                        # Mar 2026: Seed _last_failure_time for configs with failures
                        # so auto-recovery works correctly after daemon restart
                        if state.consecutive_failures > 0:
                            updated_at = row["updated_at"] if "updated_at" in row.keys() else time.time()
                            self._last_failure_time[config_key] = updated_at or time.time()
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
        # December 30, 2025: Wrap blocking SQLite I/O with asyncio.to_thread()
        await asyncio.to_thread(self._load_state)

        # Call parent start() which will run _run_cycle() periodically
        await super().start()

    async def stop(self) -> None:
        """Stop the daemon and save state (Phase 3 - December 2025).

        Overrides HandlerBase.stop() to persist training state to SQLite
        before shutdown. This ensures no state loss on graceful shutdown.
        """
        # Save state before stopping
        # December 30, 2025: Wrap blocking SQLite I/O with asyncio.to_thread()
        await asyncio.to_thread(self._save_state)
        logger.info("[TrainingTriggerDaemon] Saved state on shutdown")

        # Call parent stop()
        await super().stop()

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event subscriptions for HandlerBase.

        Subscribes to:
        - NPZ_EXPORT_COMPLETE: Immediate training trigger after export
        - NPZ_COMBINATION_COMPLETE: Training trigger after quality-weighted combination (Dec 2025)
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
            # December 30, 2025: Trigger training after quality-weighted NPZ combination
            "npz_combination_complete": self._on_npz_combination_complete,
            # December 30, 2025: Handle regression events to reduce training intensity
            "regression_detected": self._on_regression_detected,
        }

    async def _on_start(self) -> None:
        """Hook called before main loop - check coordinator mode.

        December 30, 2025: Modified to support work queue dispatch.
        On coordinator nodes or nodes without GPU, we still run the daemon
        for decision-making, but dispatch training jobs to the work queue
        instead of running locally.
        """
        if env.is_coordinator or not env.training_enabled:
            logger.info(
                f"[TrainingTriggerDaemon] Running in dispatch mode on {env.node_id} "
                f"(is_coordinator={env.is_coordinator}, training_enabled={env.training_enabled}). "
                f"Training jobs will be dispatched to cluster work queue."
            )
            self._dispatch_to_queue = True
            # Note: We no longer set _coordinator_skip = True
            # The daemon will still run cycles and process events

    async def _on_stop(self) -> None:
        """Hook called when stopping - cancel active training tasks."""
        for config_key, task in self._active_training_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"[TrainingTriggerDaemon] Cancelled training for {config_key}")

    # _get_training_params_for_intensity: Feb 2026 - Moved to training_architecture_selector.py
    # Use: get_training_params_for_intensity(intensity, default_epochs, default_batch_size)

    # _select_architecture_for_training: Feb 2026 - Moved to training_architecture_selector.py
    # Use: select_architecture_for_training(board_type, num_players)

    # _apply_velocity_amplification: Feb 2026 - Moved to training_architecture_selector.py
    # Use: apply_velocity_amplification(base_params, elo_velocity, velocity_trend)

    def _get_game_aggregator(self):
        """Get or create the UnifiedGameAggregator instance.

        January 2026: Provides lazy-loaded access to cluster-wide game counts.
        """
        if self._game_aggregator is None:
            try:
                from app.utils.unified_game_aggregator import get_unified_game_aggregator
                self._game_aggregator = get_unified_game_aggregator()
            except ImportError:
                logger.debug("[TrainingTriggerDaemon] UnifiedGameAggregator not available")
        return self._game_aggregator

    async def _log_aggregated_game_counts(
        self, config_key: str, board_type: str, num_players: int
    ) -> None:
        """Log cluster-wide game counts for visibility.

        January 2026: Shows game availability from all sources (local, cluster, S3, OWC).
        Useful for debugging training eligibility and understanding data distribution.
        """
        if not self._daemon_config.log_aggregated_game_counts:
            return

        aggregator = self._get_game_aggregator()
        if aggregator is None:
            return

        try:
            counts = await aggregator.get_total_games(
                board_type, num_players,
                include_remote=True,
                include_s3=True,
                include_owc=True,
            )
            logger.info(
                f"[TrainingTriggerDaemon] {config_key} cluster-wide games: "
                f"total={counts.total_games:,}, sources={counts.sources}"
            )
            if counts.errors:
                logger.debug(
                    f"[TrainingTriggerDaemon] {config_key} aggregation errors: {counts.errors}"
                )
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Failed to get aggregated counts: {e}")

    # _get_dynamic_sample_threshold: Feb 2026 - Moved to training_decision_engine.py
    # Use: compute_dynamic_sample_threshold(config_key, num_players, base_threshold)

    # _check_confidence_early_trigger: Feb 2026 - Moved to training_decision_engine.py
    # Use: check_confidence_early_trigger_fn(config_key, sample_count, ...)

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
                    config_key = make_config_key(board_type, num_players)
                else:
                    logger.debug("[TrainingTriggerDaemon] Missing config info in NPZ export result")
                    return

            # Update state
            state = self._get_or_create_state(config_key, board_type, num_players)
            cache_mtime = time.time()
            if npz_path:
                with contextlib.suppress(OSError):
                    cache_mtime = Path(npz_path).stat().st_mtime
            state.last_npz_update = cache_mtime
            state.npz_sample_count = samples or 0
            state.npz_path = npz_path

            # January 3, 2026: Update NPZ cache to skip redundant disk scans
            self._npz_cache[config_key] = (cache_mtime, samples or 0, npz_path)

            logger.info(
                f"[TrainingTriggerDaemon] NPZ export complete for {config_key}: "
                f"{samples} samples at {npz_path}"
            )

            # Check if we should trigger training
            await self._maybe_trigger_training(config_key)

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling NPZ export: {e}")

    async def _on_npz_combination_complete(self, result: Any) -> None:
        """Handle NPZ combination completion - trigger training on quality-weighted data.

        December 30, 2025: Added to ensure training uses quality-weighted combined NPZ
        when combination is enabled. This closes the export→combine→train flow.
        """
        try:
            metadata = getattr(result, "metadata", {})
            config_key = metadata.get("config") or metadata.get("config_key")
            board_type = metadata.get("board_type")
            num_players = metadata.get("num_players")
            output_path = metadata.get("output_path", "")
            samples = metadata.get("total_samples", 0)
            quality_weighted = metadata.get("quality_weighted", True)

            if not config_key:
                # Try to build from board_type and num_players
                if board_type and num_players:
                    config_key = make_config_key(board_type, num_players)
                else:
                    logger.debug(
                        "[TrainingTriggerDaemon] Missing config info in NPZ combination result"
                    )
                    return

            # Update state with combined NPZ
            state = self._get_or_create_state(config_key, board_type, num_players)
            cache_mtime = time.time()
            if output_path:
                with contextlib.suppress(OSError):
                    cache_mtime = Path(output_path).stat().st_mtime
            state.last_npz_update = cache_mtime
            state.npz_sample_count = samples or 0
            state.npz_path = output_path

            # January 3, 2026: Update NPZ cache to skip redundant disk scans
            self._npz_cache[config_key] = (cache_mtime, samples or 0, output_path)

            logger.info(
                f"[TrainingTriggerDaemon] NPZ combination complete for {config_key}: "
                f"{samples} samples at {output_path} (quality_weighted={quality_weighted})"
            )

            # Trigger training on combined data
            await self._maybe_trigger_training(config_key)

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling NPZ combination: {e}")

    async def _on_training_completed(self, event: Any) -> None:
        """Handle training completion to update state."""
        try:
            payload = getattr(event, "payload", {})
            config_key = extract_config_key(payload)

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

    # Feb 2026: Quality wrapper methods (_compute_quality_confidence,
    # _apply_confidence_weighting, _get_decayed_quality_score, _intensity_from_quality)
    # removed - call sites now use imported functions from training_quality_gates.py directly.

    async def _on_training_threshold_reached(self, event: Any) -> None:
        """Handle training threshold reached events from master_loop."""
        try:
            payload = getattr(event, "payload", {})
            config_key = extract_config_key(payload)
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
        """Handle quality score updates to keep intensity in sync.

        January 3, 2026: Now stores quality score and timestamp for confidence decay.
        When quality scores become stale (no updates), they decay toward a floor value,
        potentially unblocking training that was blocked by high quality gates.

        Sprint 12 Session 8: Added confidence weighting based on games_assessed.
        Quality scores from small samples are weighted toward neutral (0.5).

        Session 17.25: Added immediate training trigger when quality transitions from
        "paused" to a non-paused state. This reduces latency from 10-30s cycle time
        to immediate response (+2-5 Elo improvement).
        """
        try:
            payload = getattr(event, "payload", {})
            config_key = extract_config_key(payload)
            if not config_key:
                return

            raw_quality_score = float(payload.get("quality_score", 0.0))
            games_assessed = int(payload.get("games_assessed", 0))
            state = self._get_or_create_state(config_key)

            # Session 17.25: Track old intensity to detect transitions
            old_intensity = state.training_intensity

            # Sprint 12 Session 8: Apply confidence weighting based on sample size
            # Small samples are biased toward neutral (0.5) to avoid overconfident decisions
            if games_assessed > 0:
                adjusted_quality = apply_confidence_weighting(
                    raw_quality_score, games_assessed
                )
            else:
                adjusted_quality = raw_quality_score

            # Store raw quality score and metadata for decay calculation
            state.last_quality_score = adjusted_quality
            state.last_quality_update = time.time()
            state.games_assessed = games_assessed

            new_intensity = intensity_from_quality(
                adjusted_quality, config_key
            )
            state.training_intensity = new_intensity

            confidence = compute_quality_confidence(games_assessed)
            logger.debug(
                f"[TrainingTriggerDaemon] {config_key}: "
                f"raw_quality={raw_quality_score:.2f}, games={games_assessed}, "
                f"confidence={confidence:.0%}, adjusted={adjusted_quality:.2f} → "
                f"intensity={new_intensity}"
            )

            # Session 17.25: Immediate training trigger when quality unblocks
            # If intensity transitions from "paused" to anything else, training may
            # now be possible. Trigger immediate check instead of waiting for cycle.
            if old_intensity == "paused" and new_intensity != "paused":
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: quality gate cleared "
                    f"(paused → {new_intensity}), triggering immediate training check"
                )
                # Reset quality block count on successful unblock
                if hasattr(self, "_quality_block_counts"):
                    self._quality_block_counts.pop(config_key, None)
                # Cancel any pending quality recheck tasks
                if config_key in self._pending_quality_rechecks:
                    old_task = self._pending_quality_rechecks.pop(config_key)
                    if not old_task.done():
                        old_task.cancel()
                # Trigger immediate training check
                await self._maybe_trigger_training(config_key)

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
            config_key = extract_config_key(payload)
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
        """Handle training blocked events to pause intensity.

        January 2026 Sprint 10: Enhanced logging for quality gate blocks.
        Logs the specific quality score, threshold, and reason to help
        diagnose why training was blocked. Expected +10-15 Elo from
        better quality monitoring and faster remediation.
        """
        try:
            payload = getattr(event, "payload", {})
            config_key = extract_config_key(payload)
            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            old_intensity = state.training_intensity
            state.training_intensity = "paused"

            # Sprint 10: Extract and log quality gate details
            quality_score = payload.get("quality_score", 0.0)
            threshold = payload.get("threshold", 0.7)
            reason = payload.get("reason", "unknown")
            quality_history = payload.get("quality_history", [])

            # Log detailed quality gate block information
            logger.info(
                f"[TrainingTriggerDaemon] {config_key}: training BLOCKED by quality gate "
                f"(score={quality_score:.3f} < threshold={threshold:.2f}, reason={reason}). "
                f"Intensity: {old_intensity} → paused"
            )

            # Log quality history if available (helps diagnose trends)
            if quality_history:
                history_str = ", ".join(f"{q:.2f}" for q in quality_history[-5:])
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: recent quality history: [{history_str}]"
                )

            # Sprint 10: Track quality block stats for monitoring
            if not hasattr(self, "_quality_block_counts"):
                self._quality_block_counts: dict[str, int] = {}
            self._quality_block_counts[config_key] = self._quality_block_counts.get(config_key, 0) + 1

            # Warn if repeated quality blocks (indicates systemic issue)
            block_count = self._quality_block_counts[config_key]
            if block_count >= 3:
                logger.warning(
                    f"[TrainingTriggerDaemon] {config_key}: repeated quality blocks ({block_count}x). "
                    f"Consider: 1) increasing Gumbel budget, 2) checking selfplay for issues, "
                    f"3) verifying training data pipeline"
                )

            # Sprint 16.1 (Jan 3, 2026): Schedule automatic recheck instead of waiting for full cycle
            # This allows faster recovery when quality improves
            self._schedule_quality_recheck(config_key, delay_seconds=300)

        except Exception as e:
            logger.error(f"[TrainingTriggerDaemon] Error handling training blocked: {e}")

    def _schedule_quality_recheck(
        self, config_key: str, delay_seconds: float = 300, max_rechecks: int = 6
    ) -> None:
        """Schedule an automatic quality recheck after a delay.

        Sprint 16.1 (Jan 3, 2026): When training is blocked by quality gate, schedule
        an automatic recheck instead of waiting for the next full cycle. This reduces
        recovery time from potentially 30+ minutes to 5 minutes.

        Args:
            config_key: The config to recheck (e.g., "hex8_2p")
            delay_seconds: How long to wait before rechecking (default: 5 minutes)
            max_rechecks: Maximum recheck attempts before giving up (default: 6 = 30 min)
        """
        # Cancel existing recheck for this config (avoid duplicates)
        if config_key in self._pending_quality_rechecks:
            old_task = self._pending_quality_rechecks.pop(config_key)
            if not old_task.done():
                old_task.cancel()

        # Check recheck count to avoid infinite loops
        if not hasattr(self, "_quality_recheck_counts"):
            self._quality_recheck_counts: dict[str, int] = {}
        current_count = self._quality_recheck_counts.get(config_key, 0)
        if current_count >= max_rechecks:
            logger.info(
                f"[TrainingTriggerDaemon] {config_key}: max quality rechecks ({max_rechecks}) "
                f"reached, waiting for external quality update"
            )
            self._quality_recheck_counts[config_key] = 0  # Reset for next block
            return

        # Schedule the recheck task with safe error handling (Sprint 17.4)
        task = self._safe_create_task(
            self._run_quality_recheck(config_key, delay_seconds, max_rechecks),
            context=f"quality_recheck:{config_key}",
        )
        self._pending_quality_rechecks[config_key] = task

        # Track recheck count
        self._quality_recheck_counts[config_key] = current_count + 1

        logger.debug(
            f"[TrainingTriggerDaemon] {config_key}: scheduled quality recheck "
            f"in {delay_seconds}s (attempt {current_count + 1}/{max_rechecks})"
        )

    async def _run_quality_recheck(
        self, config_key: str, delay_seconds: float, max_rechecks: int
    ) -> None:
        """Execute a delayed quality recheck.

        Sprint 16.1 (Jan 3, 2026): After waiting, check if quality has improved.
        If so, update intensity and potentially trigger training. If not,
        schedule another recheck.
        """
        try:
            # Wait for the specified delay
            await asyncio.sleep(delay_seconds)

            # Check if we're still blocked (state may have changed)
            state = self._training_states.get(config_key)
            if not state or state.training_intensity != "paused":
                logger.debug(
                    f"[TrainingTriggerDaemon] {config_key}: quality recheck skipped, "
                    f"intensity is now {state.training_intensity if state else 'unknown'}"
                )
                # Clear recheck count since we're no longer blocked
                if hasattr(self, "_quality_recheck_counts"):
                    self._quality_recheck_counts.pop(config_key, None)
                return

            # Recheck quality gate
            quality_ok, reason = await self._check_quality_gate(config_key)

            if quality_ok:
                # Quality improved - update intensity and log success
                decayed_quality = compute_decayed_quality_score(
                    last_quality_score=state.last_quality_score,
                    last_quality_update=state.last_quality_update,
                    current_time=time.time(),
                    decay_enabled=self.config.quality_decay_enabled,
                    half_life_hours=self.config.quality_decay_half_life_hours,
                    decay_floor=self.config.quality_decay_floor,
                )
                new_intensity = intensity_from_quality(decayed_quality, config_key)
                state.training_intensity = new_intensity

                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: quality recheck PASSED, "
                    f"quality={decayed_quality:.3f}, intensity={new_intensity}"
                )

                # Clear recheck count since we're no longer blocked
                if hasattr(self, "_quality_recheck_counts"):
                    self._quality_recheck_counts.pop(config_key, None)

                # Optionally trigger training check in the next cycle
                # (the regular cycle will pick this up, no need for immediate trigger)
            else:
                # Still blocked - schedule another recheck
                logger.debug(
                    f"[TrainingTriggerDaemon] {config_key}: quality recheck still blocked, "
                    f"reason={reason}, scheduling another recheck"
                )
                self._schedule_quality_recheck(config_key, delay_seconds, max_rechecks)

        except asyncio.CancelledError:
            logger.debug(
                f"[TrainingTriggerDaemon] {config_key}: quality recheck cancelled"
            )
        except Exception as e:
            logger.error(
                f"[TrainingTriggerDaemon] {config_key}: quality recheck error: {e}"
            )
        finally:
            # Remove from pending dict
            self._pending_quality_rechecks.pop(config_key, None)

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
            # December 30, 2025: Use extract_evaluation_data for consistency
            from app.coordination.event_utils import extract_evaluation_data

            # December 30, 2025: Use consolidated extraction from HandlerBase
            payload = self._get_payload(event)
            data = extract_evaluation_data(payload)

            config_key = data.config_key
            win_rate = data.win_rate
            elo = data.elo
            games_played = data.games_played

            # December 30, 2025: If multi-harness, use best harness for decisions
            if data.is_multi_harness and data.harness_results and data.best_harness:
                best_result = data.harness_results.get(data.best_harness, {})
                if isinstance(best_result, dict):
                    elo = best_result.get("elo", elo)
                    win_rate = best_result.get("win_rate", win_rate)
                    games_played = best_result.get("games_played", games_played)

            if not config_key:
                logger.debug("[TrainingTriggerDaemon] No config_key in evaluation event")
                return

            # Mar 2026: Skip zero-game evaluations — they produce meaningless
            # win_rate=0, elo=1000 defaults that falsely trigger "struggling" intensity
            # and inflate consecutive_failures via Elo plateau detection.
            if games_played <= 0:
                logger.debug(
                    f"[TrainingTriggerDaemon] Ignoring zero-game evaluation for {config_key}"
                )
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
                self._last_failure_time[config_key] = time.time()
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
                self._last_failure_time.pop(config_key, None)

            # Record to FeedbackAccelerator for Elo momentum tracking
            await self._record_to_feedback_accelerator(config_key, elo, elo_delta)

            # January 2026 Sprint 10: Immediately attempt training after evaluation
            # This reduces evaluation→training latency by triggering training
            # as soon as evaluation completes instead of waiting for the next cycle.
            logger.info(
                f"[TrainingTriggerDaemon] {config_key}: Checking immediate training "
                f"after evaluation (intensity={state.training_intensity})"
            )
            triggered = await self._maybe_trigger_training(config_key)
            if triggered:
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: Immediate training triggered "
                    f"after evaluation completion"
                )

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
            # December 30, 2025: Use consolidated extraction from HandlerBase
            payload = self._get_payload(event)
            config_key = extract_config_key(payload)
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
            # December 30, 2025: Use consolidated extraction from HandlerBase
            payload = self._get_payload(event)
            config_key = extract_config_key(payload)
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
            # December 30, 2025: Use consolidated extraction from HandlerBase
            payload = self._get_payload(event)
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
            # December 30, 2025: Use consolidated extraction from HandlerBase
            payload = self._get_payload(event)
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
            # December 30, 2025: Use consolidated extraction from HandlerBase
            payload = self._get_payload(event)
            config_key = extract_config_key(payload)
            velocity = payload.get("velocity", 0.0)
            trend = payload.get("trend", "stable")
            previous_velocity = payload.get("previous_velocity", 0.0)

            if not config_key:
                return

            # Parse board_type and num_players from config_key
            parsed = parse_config_key(config_key)
            if not parsed:
                logger.debug(f"[TrainingTriggerDaemon] Invalid config_key: {config_key}")
                return
            board_type = parsed.board_type
            num_players = parsed.num_players

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
            # January 2026 Sprint 17.4: Wrap blocking SQLite I/O with asyncio.to_thread()
            await asyncio.to_thread(self._save_state)

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
            # December 30, 2025: Use consolidated extraction from HandlerBase
            payload = self._get_payload(event)
            config_key = extract_config_key(payload)
            error = payload.get("error", "Unknown error")
            job_id = payload.get("job_id", "")

            if not config_key:
                return

            # Parse board_type and num_players from config_key
            parsed = parse_config_key(config_key)
            if not parsed:
                logger.debug(f"[TrainingTriggerDaemon] Invalid config_key: {config_key}")
                return
            board_type = parsed.board_type
            num_players = parsed.num_players

            state = self._get_or_create_state(config_key, board_type, num_players)

            # Clear training_in_progress flag
            state.training_in_progress = False
            state.consecutive_failures += 1
            self._last_failure_time[config_key] = time.time()

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

            # January 2026 Sprint 17.4: Wrap blocking SQLite I/O with asyncio.to_thread()
            await asyncio.to_thread(self._save_state)

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.debug(f"[TrainingTriggerDaemon] Error handling TRAINING_FAILED: {e}")

    async def _on_regression_detected(self, event: Any) -> None:
        """Handle REGRESSION_DETECTED event to reduce training intensity.

        December 30, 2025: Critical fix - regression events were not being
        handled by TrainingTriggerDaemon, allowing training to continue
        even when models regressed. This slowed down recovery from regressions.

        Actions:
        1. Reduce training intensity for the affected config
        2. Extend cooldown period to allow more data collection
        3. Track regression in state for debugging

        Severity levels:
        - "critical" or "severe": Pause training immediately
        - "moderate": Reduce to "reduced" intensity
        - "minor": Reduce to "normal" if currently accelerated/hot_path

        Args:
            event: Event with payload containing config_key, severity, elo_change
        """
        try:
            payload = self._get_payload(event)
            config_key = extract_config_key(payload)
            if not config_key:
                return

            severity = payload.get("severity", "moderate")
            elo_change = payload.get("elo_change", 0.0)
            reason = payload.get("reason", "")

            # Parse config key
            parsed = parse_config_key(config_key)
            if not parsed:
                logger.debug(f"[TrainingTriggerDaemon] Invalid config_key: {config_key}")
                return

            state = self._get_or_create_state(
                config_key, parsed.board_type, parsed.num_players
            )
            old_intensity = state.training_intensity

            # Determine new intensity based on severity
            if severity in ("critical", "severe"):
                new_intensity = "paused"
            elif severity == "moderate":
                new_intensity = "reduced"
            elif old_intensity in ("hot_path", "accelerated"):
                new_intensity = "normal"
            else:
                new_intensity = old_intensity  # No change for minor regressions

            # Apply intensity change
            if new_intensity != old_intensity:
                state.training_intensity = new_intensity
                # Extend cooldown to allow more data collection
                state.training_cooldown_until = time.time() + 600.0  # 10 min cooldown

                logger.warning(
                    f"[TrainingTriggerDaemon] REGRESSION_DETECTED: {config_key} "
                    f"severity={severity}, elo_change={elo_change:.1f}, "
                    f"intensity: {old_intensity} → {new_intensity}"
                )

                # Track regression event
                state.consecutive_failures += 1
                self._last_failure_time[config_key] = time.time()
                # January 2026 Sprint 17.4: Wrap blocking SQLite I/O with asyncio.to_thread()
                await asyncio.to_thread(self._save_state)
            else:
                logger.info(
                    f"[TrainingTriggerDaemon] REGRESSION_DETECTED: {config_key} "
                    f"severity={severity} (no intensity change needed)"
                )

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.debug(f"[TrainingTriggerDaemon] Error handling REGRESSION_DETECTED: {e}")

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

        if attempts > self._retry_config.max_attempts:
            self._retry_stats["retries_exhausted"] += 1
            logger.error(
                f"[TrainingTriggerDaemon] Max retries ({self._retry_config.max_attempts}) exceeded "
                f"for {config_key}: {error[:100]}"
            )
            return False

        # December 30, 2025: Use RetryConfig for consistent delay calculation
        delay = self._retry_config.get_delay(attempts)
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


    # Feb 2026: _get_velocity_adjusted_cooldown and _get_adaptive_max_data_age
    # removed - now using compute_velocity_adjusted_cooldown() and
    # compute_adaptive_max_data_age() from training_decision_engine.py

































    async def _run_cycle(self) -> None:
        """Main work loop iteration - called by HandlerBase at scan_interval_seconds.

        December 30, 2025: Removed _coordinator_skip check. The daemon now runs
        on all nodes, including coordinators. On coordinator nodes, training jobs
        are dispatched to the work queue via _dispatch_to_queue mode.

        January 2, 2026: Added auto_detect_local_mode to enable local-only training
        when cluster is unreachable.
        """
        # Jan 2, 2026: Auto-detect local-only mode if enabled
        if self._daemon_config.auto_detect_local_mode and not self._daemon_config.local_only_mode:
            was_available = self._cluster_available
            self._cluster_available = await self._check_cluster_availability()

            if was_available and not self._cluster_available:
                self._local_only_mode = True
                logger.warning(
                    "[TrainingTriggerDaemon] Cluster unavailable, switching to local-only mode"
                )
            elif not was_available and self._cluster_available:
                self._local_only_mode = False
                logger.info(
                    "[TrainingTriggerDaemon] Cluster recovered, switching to normal mode"
                )

        # Mar 2026: Auto-recover from consecutive failure pauses
        await self._check_failure_recovery()

        # December 29, 2025 (Phase 2): Check for timed-out training jobs
        await self._check_training_timeouts()

        # January 2, 2026: Check for backpressure recovery timeout
        # If backpressure has been active too long, auto-release to prevent indefinite pause
        await self._check_backpressure_recovery_timeout()

        # December 29, 2025 (Phase 3): Process pending training retries
        await self._process_training_retry_queue()

        # Feb 2026: Periodically sync Elo from unified_elo.db to prevent stale values
        await self._sync_elo_from_unified_db()

        # Scan for training opportunities
        await self._scan_for_training_opportunities()

        # January 10, 2026: Log periodic diagnostic summary
        # This helps diagnose why training is or isn't triggering
        await self._log_training_diagnostic_summary()

        # December 29, 2025 (Phase 3): Periodically save state
        now = time.time()
        if now - self._last_state_save >= self.config.state_save_interval_seconds:
            # January 2026 Sprint 17.4: Wrap blocking SQLite I/O with asyncio.to_thread()
            await asyncio.to_thread(self._save_state)




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
        healthy = self.is_running

        # December 29, 2025 (Phase 4): Include backpressure status in message
        # Jan 2, 2026: Include local-only mode in message
        if self._local_only_mode:
            message = "Running (local-only mode)"
        elif self._evaluation_backpressure:
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
                # Jan 2, 2026: Local-only mode status
                "local_only_mode": self._local_only_mode,
                "cluster_available": self._cluster_available,
                "auto_detect_local_mode": self._daemon_config.auto_detect_local_mode,
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
    # Jan 4, 2026 - Sprint 17.9: Re-exports for backward compatibility
    # Prefer importing directly from training_execution.py
    "TrainingExecutor",
    "TrainingExecutionConfig",
    "TrainingResult",
]
