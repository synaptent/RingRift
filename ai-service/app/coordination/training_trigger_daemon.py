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
            state.last_npz_update = time.time()
            state.npz_sample_count = samples or 0
            state.npz_path = npz_path

            # January 3, 2026: Update NPZ cache to skip redundant disk scans
            self._npz_cache[config_key] = (time.time(), samples or 0, npz_path)

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
            state.last_npz_update = time.time()
            state.npz_sample_count = samples or 0
            state.npz_path = output_path

            # January 3, 2026: Update NPZ cache to skip redundant disk scans
            self._npz_cache[config_key] = (time.time(), samples or 0, output_path)

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
                # December 30, 2025: Use RetryConfig base_delay for consistency
                delay = self._retry_config.base_delay / 2  # Shorter delay for condition check
                self._training_retry_queue.append(
                    (config_key, board_type, num_players, attempts, now + delay, error)
                )
                logger.debug(
                    f"[TrainingTriggerDaemon] Retry deferred for {config_key}: {reason}"
                )

    # Feb 2026: _get_velocity_adjusted_cooldown and _get_adaptive_max_data_age
    # removed - now using compute_velocity_adjusted_cooldown() and
    # compute_adaptive_max_data_age() from training_decision_engine.py

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
            # Dec 30, 2025: Add timeout protection to prevent hanging indefinitely
            # on network issues during 48h autonomous operation
            response = await asyncio.wait_for(
                facade.trigger_priority_sync(
                    reason="training_data_stale",
                    config_key=config_key,
                    data_type="training",
                ),
                timeout=300.0,  # 5 minute timeout for sync operation
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

        except asyncio.TimeoutError:
            logger.warning(
                f"[TrainingTriggerDaemon] Priority sync timed out for {config_key} after 5min"
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

        December 30, 2025: Migrated to use consolidated extraction utilities.

        Handles various naming patterns:
        - hex8_2p.npz -> (hex8, 2)
        - square8_3p_fresh.npz -> (square8, 3)
        - canonical_hexagonal_4p_trained.npz -> (hexagonal, 4)

        Returns:
            (board_type, num_players) or (None, None) if not parseable.
        """
        # Use consolidated utilities for config extraction
        config_key = extract_config_from_path(name)
        if config_key:
            parsed = parse_config_key(config_key)
            if parsed:
                return parsed.board_type, parsed.num_players
        return None, None

    def _get_or_create_state(
        self, config_key: str, board_type: str | None = None, num_players: int | None = None
    ) -> ConfigTrainingState:
        """Get or create training state for a config."""
        # January 2026: Defensive validation - ensure board_type is a string
        # This protects against event payloads containing tuples instead of strings
        if board_type is not None and not isinstance(board_type, str):
            logger.warning(
                f"[TrainingTriggerDaemon] Invalid board_type type for {config_key}: "
                f"expected str, got {type(board_type).__name__}={board_type}"
            )
            # Try to extract string if it's a tuple (board_type, num_players)
            if isinstance(board_type, tuple) and len(board_type) >= 1:
                board_type = str(board_type[0]) if board_type[0] else None
            else:
                board_type = None

        if config_key not in self._training_states:
            # Parse config_key if board_type/num_players not provided
            if not board_type or not num_players:
                parsed_board, parsed_players = self._parse_config_from_filename(config_key)
                if parsed_board and parsed_players:
                    board_type = parsed_board
                    num_players = parsed_players
                else:
                    # Use canonical parse_config_key utility
                    parsed = parse_config_key(config_key)
                    if parsed:
                        board_type = parsed.board_type
                        num_players = parsed.num_players
                    else:
                        board_type = config_key
                        num_players = 2

            self._training_states[config_key] = ConfigTrainingState(
                config_key=config_key,
                board_type=board_type,
                num_players=num_players,
            )

        return self._training_states[config_key]

    async def _maybe_trigger_training(self, config_key: str) -> bool:
        """Check conditions and trigger training for all applicable architectures.

        December 30, 2025: Updated to support multi-architecture training.
        Iterates over architectures configured for this config and triggers
        training for each one that hasn't trained recently.
        """
        state = self._training_states.get(config_key)
        if not state:
            return False

        # Check base conditions (applies to all architectures)
        can_train, reason = await self._check_training_conditions(config_key)

        if not can_train:
            logger.debug(f"[TrainingTriggerDaemon] {config_key}: Cannot train - {reason}")
            return False

        # December 30, 2025: Iterate over architectures for this config
        # January 4, 2026: Sort by priority (highest first) for multi-architecture training
        architectures = sorted(
            self._architecture_config.get_architectures_for_config(config_key),
            key=lambda a: a.priority,
            reverse=True,  # Highest priority first (v5: 35%, v4: 20%, etc.)
        )
        if not architectures:
            # Fallback to default v5 if no architectures configured
            architectures = [ArchitectureSpec(
                name="v5", enabled=True, configs=["*"], priority=1.0
            )]

        triggered_any = False
        for arch in architectures:
            # Check architecture-specific cooldown
            arch_key = (config_key, arch.name)
            last_train_time = self._architecture_training_times.get(arch_key, 0.0)
            time_since_training = time.time() - last_train_time
            cooldown_seconds = self._architecture_config.min_hours_between_runs * 3600

            if time_since_training < cooldown_seconds:
                remaining_hours = (cooldown_seconds - time_since_training) / 3600
                logger.debug(
                    f"[TrainingTriggerDaemon] {config_key}/{arch.name}: "
                    f"Architecture cooldown ({remaining_hours:.1f}h remaining)"
                )
                continue

            # Check if already training this architecture
            if self._active_architecture_training.get(arch_key, False):
                logger.debug(
                    f"[TrainingTriggerDaemon] {config_key}/{arch.name}: "
                    f"Already training this architecture"
                )
                continue

            # Trigger training for this architecture with safe error handling (Sprint 17.4)
            logger.info(
                f"[TrainingTriggerDaemon] Triggering training for {config_key} "
                f"with architecture {arch.name}"
            )
            task = self._safe_create_task(
                self._run_training(config_key, arch),
                context=f"run_training:{config_key}:{arch.name}",
            )
            task.add_done_callback(
                lambda t, ck=config_key, a=arch.name: self._on_training_task_done(t, ck, a)
            )
            # Track with architecture suffix
            task_key = f"{config_key}:{arch.name}"
            self._active_training_tasks[task_key] = task
            self._active_architecture_training[arch_key] = True
            self._architecture_training_times[arch_key] = time.time()
            triggered_any = True

        return triggered_any

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
        cooldown_seconds = compute_velocity_adjusted_cooldown(
            self.config.training_cooldown_hours, state.elo_velocity, state.elo_velocity_trend,
        )
        if time_since_training < cooldown_seconds:
            remaining = (cooldown_seconds - time_since_training) / 3600
            trend_info = f", velocity_trend={state.elo_velocity_trend}" if state.elo_velocity_trend != "stable" else ""
            return False, f"cooldown active ({remaining:.1f}h remaining{trend_info})"

        # 3. Check data freshness (December 2025: use training_freshness for sync)
        # January 2026 (Phase 4.1): Auto-sync on stale data instead of blocking
        # January 3, 2026: Adaptive data freshness based on velocity trend
        # - Plateauing configs get 3x threshold (more lenient) to break stalls
        # - Accelerating configs get 0.5x threshold (stricter) to maintain quality
        data_age_hours = (time.time() - state.last_npz_update) / 3600
        # Feb 2026: Compute game_count for starved config detection in adaptive age
        _game_count_for_age = None
        try:
            from app.utils.game_discovery import count_games_for_config as _cgfc
            parsed_ck = parse_config_key(config_key)
            if parsed_ck:
                _game_count_for_age = _cgfc(parsed_ck.board_type, parsed_ck.num_players)
        except (ImportError, ValueError, OSError):
            pass
        adaptive_max_age = compute_adaptive_max_data_age(
            self.config.max_data_age_hours, state.elo_velocity_trend,
            state.last_training_time, time.time(), game_count=_game_count_for_age,
        )
        if data_age_hours > adaptive_max_age:
            # December 29, 2025: Strict mode - fail immediately without sync attempt
            if self.config.strict_freshness_mode:
                return False, f"data too old ({data_age_hours:.1f}h) [strict mode - no sync]"

            # January 2026 (Phase 4.1): Check if data is "very stale" (>2x adaptive threshold)
            # Very stale data → proceed with warning (don't block indefinitely)
            very_stale_threshold = adaptive_max_age * 2
            if data_age_hours > very_stale_threshold:
                # Data is very old - proceed anyway with warning to prevent indefinite blocks
                logger.warning(
                    f"[TrainingTriggerDaemon] {config_key}: proceeding with very stale data "
                    f"(age={data_age_hours:.1f}h > {very_stale_threshold:.1f}h threshold). "
                    f"Triggering background sync."
                )
                # Trigger background sync with safe error handling (Sprint 17.4)
                self._safe_create_task(
                    self._trigger_priority_sync(config_key, state.board_type, state.num_players),
                    context=f"priority_sync_very_stale:{config_key}",
                )
                # Continue with training (data will be fresher next time)
            elif self.config.enforce_freshness_with_sync:
                # Moderately stale - try to sync and wait for fresh data
                fresh = await self._ensure_fresh_data(state.board_type, state.num_players)
                if not fresh:
                    return False, f"data stale ({data_age_hours:.1f}h), sync triggered but not ready"
                # Sync succeeded, continue with training check
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: data refreshed via sync"
                )
            else:
                # Not enforcing with sync - just trigger background sync and block
                # Use safe task creation for error handling (Sprint 17.4)
                self._safe_create_task(
                    self._trigger_priority_sync(config_key, state.board_type, state.num_players),
                    context=f"priority_sync_stale:{config_key}",
                )
                return False, f"data stale ({data_age_hours:.1f}h), sync triggered"

        # January 2026: Log cluster-wide game counts for visibility
        # This helps understand data distribution across all sources
        await self._log_aggregated_game_counts(config_key, state.board_type, state.num_players)

        # 3.5 January 2026 Sprint 10: Check data quality before training
        # This ensures training only proceeds with high-quality data.
        # Expected improvement: +15-20 Elo from tighter quality feedback.
        quality_ok, quality_reason = await self._check_quality_gate(config_key)
        if not quality_ok:
            return False, quality_reason

        # 3.6 January 6, 2026 (Session 17.41): Graduated minimum game requirement
        # Use graduated thresholds based on player count to enable training earlier
        # for 4p configs while still requiring sufficient data quality.
        # 2p: 50, 3p: 70, 4p: 100 (synced with PromotionGameDefaults)
        try:
            from app.config.coordination_defaults import PromotionGameDefaults
            from app.utils.game_discovery import count_games_for_config

            game_count = count_games_for_config(state.board_type, state.num_players)
            min_games = PromotionGameDefaults.get_min_games(state.num_players)

            if game_count < min_games:
                return False, (
                    f"insufficient games for {state.num_players}p config "
                    f"({game_count} < {min_games} graduated minimum)"
                )
        except Exception as e:
            logger.warning(
                f"[TrainingTriggerDaemon] {config_key}: could not check game count: {e}"
            )

        # 4. Check minimum samples (with confidence-based early trigger)
        # Dec 29, 2025: Try confidence-based early trigger first
        # This allows training to start earlier when statistical confidence is high
        if state.npz_sample_count >= self.config.confidence_min_samples:
            early_trigger, early_reason = check_confidence_early_trigger_fn(
                config_key, state.npz_sample_count,
                min_samples=self.config.confidence_min_samples,
                target_ci_width=self.config.confidence_target_ci_width,
                confidence_enabled=self.config.confidence_early_trigger_enabled,
            )
            if early_trigger:
                logger.info(
                    f"[TrainingTriggerDaemon] {config_key}: early trigger - {early_reason}"
                )
                # Skip the min_samples check - confidence is high enough
            else:
                # Fall back to dynamic threshold from ImprovementOptimizer
                # Phase 5 (Dec 2025): Lower when on promotion streak, higher when struggling
                min_samples = compute_dynamic_sample_threshold(
                    config_key, state.num_players or 2,
                    base_threshold=self.config.min_samples_threshold,
                )
                if state.npz_sample_count < min_samples:
                    return False, f"insufficient samples ({state.npz_sample_count} < {min_samples}), {early_reason}"
        else:
            # Below confidence minimum - use dynamic threshold
            min_samples = compute_dynamic_sample_threshold(
                config_key, state.num_players or 2,
                base_threshold=self.config.min_samples_threshold,
            )
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

    async def get_training_decision(self, config_key: str) -> TrainingDecision:
        """Get detailed training decision for a config (December 30, 2025 - RPC API).

        This method exposes the full training decision logic for external callers,
        including the P2P orchestrator's /training/trigger-decision endpoint.

        Args:
            config_key: Config key like "hex8_2p"

        Returns:
            TrainingDecision with full condition details
        """
        state = self._training_states.get(config_key)
        if not state:
            return TrainingDecision(
                config_key=config_key,
                can_trigger=False,
                reason="config not tracked",
            )

        # Calculate all condition values
        time_since_training = time.time() - state.last_training_time
        cooldown_seconds = compute_velocity_adjusted_cooldown(
            self.config.training_cooldown_hours, state.elo_velocity, state.elo_velocity_trend,
        )
        cooldown_remaining = max(0, cooldown_seconds - time_since_training) / 3600

        data_age_hours = (time.time() - state.last_npz_update) / 3600

        min_samples = compute_dynamic_sample_threshold(
            config_key, state.num_players or 2,
            base_threshold=self.config.min_samples_threshold,
        )

        active_count = sum(
            1 for s in self._training_states.values() if s.training_in_progress
        )

        # Check circuit breaker
        circuit_breaker_open = False
        if HAS_CIRCUIT_BREAKER and get_training_breaker:
            breaker = get_training_breaker()
            circuit_breaker_open = not breaker.can_execute(config_key)

        # GPU availability (quick check, don't block)
        # Jan 2026: Reduced timeout from 5s to 2s for faster training trigger decisions
        gpu_available = True
        try:
            gpu_available = await asyncio.wait_for(
                self._check_gpu_availability(), timeout=2.0
            )
        except asyncio.TimeoutError:
            pass

        # Get the actual decision
        can_trigger, reason = await self._check_training_conditions(config_key)

        return TrainingDecision(
            config_key=config_key,
            can_trigger=can_trigger,
            reason=reason,
            training_in_progress=state.training_in_progress,
            intensity_paused=state.training_intensity == "paused",
            evaluation_backpressure=self._evaluation_backpressure,
            circuit_breaker_open=circuit_breaker_open,
            cooldown_remaining_hours=cooldown_remaining,
            data_age_hours=data_age_hours,
            max_data_age_hours=self.config.max_data_age_hours,
            sample_count=state.npz_sample_count,
            sample_threshold=min_samples,
            gpu_available=gpu_available,
            concurrent_training_count=active_count,
            max_concurrent_training=self.config.max_concurrent_training,
            npz_path=state.npz_path,
            current_elo=state.last_elo,
            elo_velocity=state.elo_velocity,
            elo_velocity_trend=state.elo_velocity_trend,
        )

    def get_tracked_configs(self) -> list[str]:
        """Get list of all tracked config keys (December 30, 2025 - RPC API)."""
        return list(self._training_states.keys())

    def _get_cached_quality(self, config_key: str) -> float | None:
        """Get cached quality score if fresh.

        January 5, 2026 (Phase 7.9): Quality assessment cache to reduce SQLite lookups.
        Returns cached score if within TTL, None otherwise.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            Cached quality score if fresh, None if stale or not cached
        """
        if config_key in self._quality_cache:
            score, timestamp = self._quality_cache[config_key]
            if time.time() - timestamp < self._quality_cache_ttl:
                return score
        return None

    def _update_quality_cache(self, config_key: str, quality: float) -> None:
        """Update quality cache with fresh score.

        January 5, 2026 (Phase 7.9): Cache quality results for 10 seconds.
        """
        self._quality_cache[config_key] = (quality, time.time())

    async def _check_quality_gate(self, config_key: str) -> tuple[bool, str]:
        """Check if data quality meets minimum threshold for training.

        January 2026 Sprint 10: Tighter quality feedback before training.
        Blocks training if data quality is below threshold, ensuring we only
        train on high-quality data.

        January 3, 2026: Added quality confidence decay. When quality data is stale
        (no recent updates), the effective quality score decays toward a floor value.
        This prevents stale high-quality assessments from blocking training indefinitely.

        Expected improvement: +15-20 Elo from better training data quality.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            Tuple of (quality_ok, reason):
            - (True, "quality ok (X.XX)") if quality >= threshold
            - (False, "quality too low (X.XX < threshold)") if quality < threshold
        """
        # January 3, 2026 (Sprint 10): Use board-specific quality thresholds
        # Larger/more complex boards need higher quality data for effective training
        # See QualityGateDefaults.QUALITY_GATES for per-config thresholds
        try:
            from app.config.coordination_defaults import QualityGateDefaults
            quality_threshold = QualityGateDefaults.get_quality_threshold(config_key)
        except ImportError:
            quality_threshold = 0.50  # Fallback to default

        try:
            # January 5, 2026 (Phase 7.9): Check cache first to reduce SQLite lookups
            cached = self._get_cached_quality(config_key)
            if cached is not None:
                quality = cached
                logger.debug(
                    f"[TrainingTriggerDaemon] {config_key}: using cached quality {quality:.2f}"
                )
            else:
                # Cache miss - fetch from quality_monitor
                from app.coordination.quality_monitor_daemon import get_quality_monitor

                quality_monitor = get_quality_monitor()
                quality = quality_monitor.get_quality_for_config(config_key)

                if quality is None:
                    # Jan 3, 2026: No fresh quality data - try decayed stored quality
                    state = self._training_states.get(config_key)
                    if state and state.last_quality_update > 0:
                        decayed_quality = compute_decayed_quality_score(
                            last_quality_score=state.last_quality_score,
                            last_quality_update=state.last_quality_update,
                            current_time=time.time(),
                            decay_enabled=self.config.quality_decay_enabled,
                            half_life_hours=self.config.quality_decay_half_life_hours,
                            decay_floor=self.config.quality_decay_floor,
                        )
                        logger.debug(
                            f"[TrainingTriggerDaemon] {config_key}: using decayed quality "
                            f"{decayed_quality:.2f} (original: {state.last_quality_score:.2f})"
                        )
                        quality = decayed_quality
                    else:
                        # No quality data available at all - allow training with warning
                        logger.debug(
                            f"[TrainingTriggerDaemon] {config_key}: no quality data available, "
                            f"proceeding with training"
                        )
                        return True, "no quality data (proceeding anyway)"

                # Update cache with fresh quality score
                if quality is not None:
                    self._update_quality_cache(config_key, quality)

            if quality < quality_threshold:
                # January 3, 2026: Relax quality gate for data-starved or stalled configs
                # This prevents configs with limited data from being permanently blocked
                # while maintaining quality floor to prevent garbage data training
                # Jan 4, 2026 - Sprint 17.9: Constants now imported from training_quality_gates.py

                allow_degraded = False
                degraded_reason = ""

                # Check if quality meets minimum floor
                if quality >= MINIMUM_QUALITY_FLOOR:
                    # Get game count for this config
                    try:
                        from app.utils.game_discovery import count_games_for_config
                        from app.coordination.event_utils import parse_config_key

                        parsed = parse_config_key(config_key)
                        if parsed:
                            game_count = count_games_for_config(
                                parsed.board_type, parsed.num_players
                            )
                            if game_count < DATA_STARVED_THRESHOLD:
                                allow_degraded = True
                                degraded_reason = (
                                    f"bootstrap mode ({game_count} < {DATA_STARVED_THRESHOLD} games)"
                                )
                    except (ImportError, ValueError, OSError) as e:
                        logger.debug(f"[TrainingTriggerDaemon] Game count check failed: {e}")

                    # Check if training is stalled (emergency override)
                    if not allow_degraded:
                        state = self._training_states.get(config_key)
                        if state and state.last_training_time > 0:
                            hours_since = (time.time() - state.last_training_time) / 3600
                            if hours_since > TRAINING_STALL_HOURS:
                                allow_degraded = True
                                degraded_reason = f"training stalled ({hours_since:.1f}h > {TRAINING_STALL_HOURS}h)"

                if allow_degraded:
                    logger.info(
                        f"[TrainingTriggerDaemon] {config_key}: quality gate RELAXED - "
                        f"allowing degraded quality ({quality:.2f}) for {degraded_reason}"
                    )
                    return True, f"quality degraded but allowed ({quality:.2f}, {degraded_reason})"

                # Quality too low - block training and emit event
                logger.warning(
                    f"[TrainingTriggerDaemon] {config_key}: quality gate FAILED "
                    f"(quality={quality:.2f} < threshold={quality_threshold})"
                )

                # Emit TRAINING_BLOCKED_BY_QUALITY event for feedback loop
                from app.coordination.event_emission_helpers import safe_emit_event

                safe_emit_event(
                    "TRAINING_BLOCKED_BY_QUALITY",
                    {
                        "config_key": config_key,
                        "quality_score": quality,
                        "threshold": quality_threshold,
                        "reason": "pre_training_quality_gate",
                        "source": "training_trigger_daemon",
                    },
                    context="TrainingTriggerDaemon",
                )

                return False, f"quality too low ({quality:.2f} < {quality_threshold})"

            # Quality is acceptable
            logger.debug(
                f"[TrainingTriggerDaemon] {config_key}: quality gate passed "
                f"(quality={quality:.2f} >= {quality_threshold})"
            )
            return True, f"quality ok ({quality:.2f})"

        except ImportError:
            # QualityMonitorDaemon not available - allow training
            logger.debug(
                f"[TrainingTriggerDaemon] {config_key}: quality monitor not available, "
                f"proceeding with training"
            )
            return True, "quality monitor not available"
        except Exception as e:
            # Unexpected error - allow training but log
            logger.debug(f"[TrainingTriggerDaemon] Quality check error: {e}")
            return True, f"quality check error: {e}"

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

    async def _check_cluster_availability(self) -> bool:
        """Check if cluster is available with fast timeout (Jan 2, 2026).

        Used by auto_detect_local_mode to determine if we should fall back
        to local-only mode when cluster is unreachable.

        Returns:
            True if cluster is reachable, False otherwise
        """
        timeout = self._daemon_config.cluster_availability_timeout_seconds

        try:
            # Check P2P status endpoint
            import aiohttp

            p2p_url = get_local_p2p_status_url()
            async with aiohttp.ClientSession() as session:
                async with session.get(p2p_url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        alive_peers = data.get("alive_peers", 0)
                        if alive_peers > 0:
                            return True
                        # No peers alive - cluster not functional
                        logger.debug(
                            "[TrainingTriggerDaemon] Cluster check: no alive peers"
                        )
                        return False

        except ImportError:
            logger.debug("[TrainingTriggerDaemon] aiohttp not available for cluster check")
        except asyncio.TimeoutError:
            logger.debug(
                f"[TrainingTriggerDaemon] Cluster check timed out after {timeout}s"
            )
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Cluster check failed: {e}")

        return False

    def _scan_local_npz_files(self) -> list[tuple[str, str, int, Path]]:
        """Scan local NPZ files for training in local-only mode (Jan 2, 2026).

        Returns list of (config_key, board_type, num_players, npz_path) tuples
        for all valid NPZ files found locally.
        """
        results: list[tuple[str, str, int, Path]] = []

        training_dir = Path(__file__).resolve().parent.parent.parent / "data" / "training"
        if not training_dir.exists():
            return results

        for npz_path in training_dir.glob("*.npz"):
            board_type, num_players = self._parse_config_from_filename(npz_path.stem)
            if board_type is None or num_players is None:
                continue

            config_key = make_config_key(board_type, num_players)
            results.append((config_key, board_type, num_players, npz_path))

        return results

    async def _ensure_fresh_data(self, board_type: str, num_players: int) -> bool:
        """Ensure training data is fresh, triggering sync if needed (December 2025).

        Uses training_freshness module to check data age and trigger sync
        if data is stale. This closes the data freshness feedback loop.

        Jan 2, 2026: In local-only mode, skips sync and just checks if local data exists.

        Args:
            board_type: Board type for training
            num_players: Number of players

        Returns:
            True if data is now fresh, False if sync failed or timed out
        """
        # Jan 2, 2026: In local-only mode, just check if local NPZ exists
        if self._local_only_mode:
            config_key = make_config_key(board_type, num_players)
            local_npz = Path(f"data/training/{config_key}.npz")
            if local_npz.exists():
                logger.debug(
                    f"[TrainingTriggerDaemon] Local-only mode: using existing NPZ for {config_key}"
                )
                return True
            logger.debug(
                f"[TrainingTriggerDaemon] Local-only mode: no NPZ for {config_key}"
            )
            return False

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
                config_key = make_config_key(board_type, num_players)
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

    async def _check_all_data_sources(
        self, config_key: str, min_samples_needed: int
    ) -> tuple[int, str | None]:
        """Check all sources for available training data (January 2026).

        Queries local NPZ files, TrainingDataManifest (S3/OWC), and ClusterManifest
        to find total available samples across all data sources.

        Jan 2, 2026: In local-only mode, skips remote data sources (S3, OWC, Cluster).

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            min_samples_needed: Minimum samples required for training

        Returns:
            Tuple of (total_samples_available, best_remote_path_if_any)
        """
        total_samples = 0
        best_remote_path: str | None = None

        # 1. Check local NPZ files
        try:
            local_npz = Path(f"data/training/{config_key}.npz")
            if local_npz.exists():
                import numpy as np
                data = np.load(local_npz)
                local_count = len(data.get("features", data.get("states", [])))
                total_samples += local_count
                logger.debug(
                    f"[TrainingTriggerDaemon] Local NPZ for {config_key}: {local_count} samples"
                )
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Local NPZ check failed: {e}")

        # Jan 2, 2026: Skip remote sources in local-only mode
        if self._local_only_mode:
            logger.debug(
                f"[TrainingTriggerDaemon] Local-only mode: skipping remote data sources for {config_key}"
            )
            return total_samples, best_remote_path

        # 2. Check TrainingDataManifest for S3/OWC data
        try:
            from app.coordination.training_data_manifest import (
                get_training_manifest,
                DataSource,
            )

            manifest = get_training_manifest()

            # Check S3
            s3_data = manifest.get_data_for_config(config_key, source=DataSource.S3)
            if s3_data and s3_data.sample_count > 0:
                logger.debug(
                    f"[TrainingTriggerDaemon] S3 has {s3_data.sample_count} samples for {config_key}"
                )
                if s3_data.sample_count > total_samples:
                    total_samples = s3_data.sample_count
                    best_remote_path = s3_data.path

            # Check OWC
            owc_data = manifest.get_data_for_config(config_key, source=DataSource.OWC)
            if owc_data and owc_data.sample_count > 0:
                logger.debug(
                    f"[TrainingTriggerDaemon] OWC has {owc_data.sample_count} samples for {config_key}"
                )
                if owc_data.sample_count > total_samples:
                    total_samples = owc_data.sample_count
                    best_remote_path = owc_data.path

        except ImportError:
            logger.debug("[TrainingTriggerDaemon] TrainingDataManifest not available")
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Manifest check failed: {e}")

        # 3. Check ClusterManifest for games on other nodes (estimate samples)
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            cluster_manifest = get_cluster_manifest()
            remote_games = cluster_manifest.get_game_count(config_key)
            if remote_games > 0:
                # Estimate ~50 samples per game (typical move count)
                estimated_samples = remote_games * 50
                logger.debug(
                    f"[TrainingTriggerDaemon] Cluster has ~{remote_games} games "
                    f"(~{estimated_samples} samples) for {config_key}"
                )
                # Don't override total_samples, just log for awareness
                # The games need to be synced and exported first

        except ImportError:
            logger.debug("[TrainingTriggerDaemon] ClusterManifest not available")
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Cluster check failed: {e}")

        return total_samples, best_remote_path

    async def _fetch_remote_data_if_needed(
        self, config_key: str, local_count: int, min_samples_needed: int
    ) -> bool:
        """Fetch remote data if local is insufficient (January 2026).

        When local training data is below threshold, attempts to download
        from S3 or OWC to enable training.

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            local_count: Current local sample count
            min_samples_needed: Minimum samples required for training

        Returns:
            True if data was fetched and is now available locally
        """
        if local_count >= min_samples_needed:
            return True  # Already have enough locally

        try:
            from app.coordination.training_data_manifest import (
                get_training_manifest,
                DataSource,
            )

            manifest = get_training_manifest()

            # Find best remote source
            best_source = None
            best_count = local_count

            for source in [DataSource.S3, DataSource.OWC]:
                data = manifest.get_data_for_config(config_key, source=source)
                if data and data.sample_count > best_count:
                    best_source = data
                    best_count = data.sample_count

            if best_source and best_count >= min_samples_needed:
                logger.info(
                    f"[TrainingTriggerDaemon] Fetching {config_key} from "
                    f"{best_source.source.value} ({best_count} samples)"
                )

                # Download to local training directory
                local_path = await manifest.download_to_local(best_source)
                if local_path and local_path.exists():
                    logger.info(
                        f"[TrainingTriggerDaemon] Downloaded {config_key} to {local_path}"
                    )
                    return True
                else:
                    logger.warning(
                        f"[TrainingTriggerDaemon] Download failed for {config_key}"
                    )
                    return False

            logger.debug(
                f"[TrainingTriggerDaemon] No remote source with enough data for {config_key}"
            )
            return False

        except ImportError:
            logger.debug("[TrainingTriggerDaemon] TrainingDataManifest not available")
            return False
        except Exception as e:
            logger.warning(f"[TrainingTriggerDaemon] Remote fetch failed: {e}")
            return False

    async def _dispatch_training_to_queue(
        self,
        config_key: str,
        state: ConfigTrainingState,
        arch: ArchitectureSpec | None = None,
    ) -> bool:
        """Dispatch training job to work queue for remote execution.

        December 30, 2025: Added to support coordinator-based training dispatch.
        When the daemon runs on a coordinator node (no GPU), it dispatches
        training jobs to the centralized work queue. GPU nodes in the cluster
        will claim and execute these jobs.

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            state: Current training state for this config
            arch: Optional architecture specification

        Returns:
            True if job was successfully queued
        """
        try:
            from app.coordination.work_distributor import get_work_distributor

            distributor = get_work_distributor()

            # Get intensity-adjusted training parameters
            epochs, batch_size, lr_mult = get_training_params_for_intensity(
                state.training_intensity,
                default_epochs=self.config.default_epochs,
                default_batch_size=self.config.default_batch_size,
            )

            # Apply architecture-specific overrides if provided
            # Session 17.22: Use tracker-informed selection when arch is not explicitly specified
            if arch is not None:
                arch_name = arch.name
                if arch.epochs is not None:
                    epochs = arch.epochs
                if arch.batch_size is not None:
                    batch_size = arch.batch_size
            else:
                # No explicit arch - use ArchitectureTracker for performance-based selection
                arch_name = select_architecture_for_training(
                    board_type=state.board_type,
                    num_players=state.num_players,
                )

            # Compute priority based on config characteristics
            priority = 50
            # Higher priority for underrepresented configs
            if state.board_type in ("square19", "hexagonal"):
                priority = min(100, priority + 20)
            if state.num_players in (3, 4):
                priority = min(100, priority + 15)
            # Boost priority for accelerating configs (positive Elo velocity)
            if state.elo_velocity > 10.0:
                priority = min(100, priority + 10)

            # Build config for work queue submission
            from app.coordination.work_distributor import DistributedWorkConfig
            work_config = DistributedWorkConfig(
                require_gpu=True,  # Training requires GPU
                require_high_memory=state.board_type in ("square19", "hexagonal"),
                priority=priority,
            )

            # Submit to work queue
            work_id = await distributor.submit_training(
                board=state.board_type,
                num_players=state.num_players,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=1e-3 * lr_mult,
                config=work_config,
                model_version=arch_name,
            )

            if work_id:
                logger.info(
                    f"[TrainingTriggerDaemon] Dispatched training to queue: {config_key} "
                    f"(work_id={work_id}, arch={arch_name}, epochs={epochs}, batch={batch_size})"
                )
                # Update state to track that training was dispatched
                state.training_in_progress = True
                state.training_start_time = time.time()
                # Store work_id for tracking (use _pending prefix to avoid conflicts)
                if not hasattr(state, "_pending_work_id"):
                    state._pending_work_id = None
                state._pending_work_id = work_id
                return True
            else:
                logger.warning(
                    f"[TrainingTriggerDaemon] Failed to dispatch training for {config_key}: "
                    "work queue returned None"
                )
                return False

        except ImportError as e:
            logger.warning(
                f"[TrainingTriggerDaemon] Cannot dispatch to work queue (module not available): {e}"
            )
            return False
        except Exception as e:
            logger.error(
                f"[TrainingTriggerDaemon] Failed to dispatch training for {config_key}: {e}"
            )
            return False

    async def _run_training(
        self,
        config_key: str,
        arch: ArchitectureSpec | None = None,
    ) -> bool:
        """Run training subprocess for a configuration.

        December 30, 2025: Added arch parameter for multi-architecture training.
        January 2026 (Sprint 3): Added distributed lock to prevent duplicate jobs.
        """
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

        # January 2026 (Sprint 3): Acquire distributed lock to prevent duplicate training
        # This ensures only one node trains a given config at a time across the cluster.
        # Lock timeout is 30 minutes (1800s) to cover long training runs.
        if HAS_DISTRIBUTED_LOCK and with_training_lock:
            try:
                # Check if P2P is available before attempting lock
                if is_p2p_available and await is_p2p_available():
                    arch_suffix = f":{arch.name}" if arch else ""
                    lock_name = f"{config_key}{arch_suffix}"
                    async with with_training_lock(lock_name, timeout_seconds=1800.0) as lock_result:
                        if not lock_result.acquired:
                            logger.info(
                                f"[TrainingTriggerDaemon] Training lock for {config_key} "
                                f"not acquired (held by another node), skipping"
                            )
                            return False
                        logger.debug(
                            f"[TrainingTriggerDaemon] Acquired training lock for {config_key}"
                        )
                        # Run training within lock context
                        return await self._run_training_inner(config_key, state, arch)
            except Exception as e:
                # If lock acquisition fails, log and proceed without lock
                # This ensures training still works when P2P is unavailable
                logger.warning(
                    f"[TrainingTriggerDaemon] Distributed lock error for {config_key}: {e}, "
                    "proceeding without cluster lock"
                )

        # Fallback: run without distributed lock (P2P unavailable or error)
        return await self._run_training_inner(config_key, state, arch)

    async def _run_training_inner(
        self,
        config_key: str,
        state: "TrainingConfigState",
        arch: ArchitectureSpec | None = None,
    ) -> bool:
        """Inner training logic (called by _run_training with lock held).

        January 2026 (Sprint 3): Extracted to enable lock wrapper.
        """
        # December 30, 2025: Dispatch to work queue on coordinator nodes
        # This allows the coordinator to trigger training on remote GPU nodes
        if self._dispatch_to_queue:
            return await self._dispatch_training_to_queue(config_key, state, arch)

        # December 30, 2025: Default to v5 if no architecture specified
        if arch is None:
            arch = ArchitectureSpec(
                name="v5", enabled=True, configs=["*"], priority=1.0
            )

        async with self._training_semaphore:
            state.training_in_progress = True
            state.training_start_time = time.time()  # Phase 2: Timeout watchdog

            try:
                # Get intensity-adjusted training parameters
                epochs, batch_size, lr_mult = get_training_params_for_intensity(
                    state.training_intensity,
                    default_epochs=self.config.default_epochs,
                    default_batch_size=self.config.default_batch_size,
                )

                # January 3, 2026 (Sprint 12): Apply Elo velocity-based amplification
                # High velocity configs get more aggressive training to capitalize on momentum
                # Low velocity configs get more conservative LR to avoid overshooting
                epochs, batch_size, lr_mult = apply_velocity_amplification(
                    (epochs, batch_size, lr_mult),
                    state.elo_velocity,
                    state.elo_velocity_trend,
                )

                # December 30, 2025: Apply architecture-specific overrides
                if arch.epochs is not None:
                    epochs = arch.epochs
                if arch.batch_size is not None:
                    batch_size = arch.batch_size

                logger.info(
                    f"[TrainingTriggerDaemon] Starting training for {config_key} "
                    f"with architecture {arch.name} "
                    f"({state.npz_sample_count} samples, intensity={state.training_intensity}, "
                    f"velocity={state.elo_velocity:.2f}, trend={state.elo_velocity_trend}, "
                    f"epochs={epochs}, batch={batch_size}, lr_mult={lr_mult:.2f})"
                )

                # Build training command
                base_dir = Path(__file__).resolve().parent.parent.parent
                npz_path = state.npz_path or f"data/training/{config_key}.npz"

                # December 30, 2025: Include architecture in model filename
                # Format: canonical_{config}_{arch}.pth (e.g., canonical_hex8_2p_v4.pth)
                model_filename = f"canonical_{config_key}_{arch.name}.pth"
                model_path = str(base_dir / "models" / model_filename)

                cmd = [
                    sys.executable,
                    "-m", "app.training.train",
                    "--board-type", state.board_type,
                    "--num-players", str(state.num_players),
                    "--data-path", npz_path,
                    "--model-version", arch.name,  # December 30, 2025: Use architecture name
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
                # December 30, 2025: Remove from active tasks using architecture-aware key
                # Note: The _on_training_task_done callback also cleans up,
                # but this ensures cleanup on exceptions before callback fires
                task_key = f"{config_key}:{arch.name}"
                self._active_training_tasks.pop(task_key, None)
                arch_key = (config_key, arch.name)
                self._active_architecture_training.pop(arch_key, None)

    def _on_training_task_done(
        self, task: asyncio.Task, config_key: str, arch_name: str | None = None
    ) -> None:
        """Handle training task completion.

        December 30, 2025: Added arch_name parameter for multi-architecture tracking.
        """
        try:
            exc = task.exception()
            if exc:
                logger.error(
                    f"[TrainingTriggerDaemon] Training task error for "
                    f"{config_key}/{arch_name or 'v5'}: {exc}"
                )
        except asyncio.CancelledError:
            pass
        except asyncio.InvalidStateError:
            pass

        # December 30, 2025: Clear architecture-specific tracking
        if arch_name:
            arch_key = (config_key, arch_name)
            self._active_architecture_training.pop(arch_key, None)
            # Remove task with architecture suffix
            task_key = f"{config_key}:{arch_name}"
            self._active_training_tasks.pop(task_key, None)

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
            # January 2, 2026: Use graceful shutdown - SIGTERM first to allow checkpoint save,
            # then SIGKILL after grace period if still running
            if state.training_pid is not None:
                await self._graceful_kill_process(
                    state.training_pid,
                    config_key,
                    grace_seconds=self.config.graceful_kill_timeout_seconds,
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

    async def _graceful_kill_process(
        self, pid: int, config_key: str, grace_seconds: float = 30.0
    ) -> None:
        """Gracefully kill a training process - SIGTERM first, then SIGKILL.

        January 2, 2026: Added to prevent model checkpoint corruption during timeout.
        Sends SIGTERM first to allow the training process to save checkpoints,
        waits up to grace_seconds, then sends SIGKILL if still running.

        Args:
            pid: Process ID to kill
            config_key: Config key for logging
            grace_seconds: Time to wait between SIGTERM and SIGKILL
        """
        try:
            # Jan 3, 2026: Emit TRAINING_TIMEOUT_REACHED before killing to allow
            # other systems (curriculum, feedback loop) to react
            from app.coordination.event_emission_helpers import safe_emit_event

            safe_emit_event(
                "TRAINING_TIMEOUT_REACHED",
                {
                    "config_key": config_key,
                    "pid": pid,
                    "timeout_hours": self.config.training_timeout_hours,
                    "grace_seconds": grace_seconds,
                    "timestamp": time.time(),
                },
                context="TrainingTriggerDaemon",
                log_after=f"Emitted TRAINING_TIMEOUT_REACHED for {config_key}",
            )

            # First, send SIGTERM for graceful shutdown
            os.kill(pid, signal.SIGTERM)
            logger.info(
                f"[TrainingTriggerDaemon] Sent SIGTERM to training process "
                f"PID {pid} for {config_key}, waiting {grace_seconds}s for graceful exit"
            )

            # Wait for process to exit gracefully
            start_wait = time.time()
            while time.time() - start_wait < grace_seconds:
                try:
                    # Check if process still exists (os.kill with signal 0 checks existence)
                    os.kill(pid, 0)
                    await asyncio.sleep(1.0)  # Check every second
                except ProcessLookupError:
                    # Process has exited gracefully
                    logger.info(
                        f"[TrainingTriggerDaemon] Training process PID {pid} "
                        f"exited gracefully after SIGTERM for {config_key}"
                    )
                    self._timeout_stats["processes_killed"] += 1
                    return

            # Process still running after grace period - send SIGKILL
            try:
                os.kill(pid, signal.SIGKILL)
                self._timeout_stats["processes_killed"] += 1
                logger.warning(
                    f"[TrainingTriggerDaemon] Sent SIGKILL to training process "
                    f"PID {pid} for {config_key} (did not exit after {grace_seconds}s SIGTERM)"
                )
            except ProcessLookupError:
                # Process exited between our check and SIGKILL - that's fine
                logger.info(
                    f"[TrainingTriggerDaemon] Training process PID {pid} "
                    f"exited just before SIGKILL for {config_key}"
                )

        except ProcessLookupError:
            logger.debug(
                f"[TrainingTriggerDaemon] Process {pid} already dead for {config_key}"
            )
        except PermissionError:
            logger.error(
                f"[TrainingTriggerDaemon] Permission denied killing PID {pid} for {config_key}"
            )
        except OSError as e:
            logger.error(
                f"[TrainingTriggerDaemon] OS error killing PID {pid} for {config_key}: {e}"
            )

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

    async def _log_training_diagnostic_summary(self) -> None:
        """Log periodic diagnostic summary of training trigger status.

        January 10, 2026: Added to help diagnose why training is or isn't triggering.
        Logs every 5 minutes (not every cycle) to avoid log spam.
        Shows:
        - NPZ files found and sample counts
        - Quality scores per config
        - Why training was NOT triggered (which condition failed)
        - Active training jobs

        This is critical for debugging the self-improvement loop.
        """
        # Rate-limit to every 5 minutes to avoid log spam
        now = time.time()
        if not hasattr(self, "_last_diagnostic_log"):
            self._last_diagnostic_log = 0.0
        if now - self._last_diagnostic_log < 300.0:  # 5 minutes
            return
        self._last_diagnostic_log = now

        if not self._training_states:
            logger.info(
                "[TrainingTriggerDaemon] DIAGNOSTIC: No training states tracked yet. "
                "Waiting for NPZ_EXPORT_COMPLETE events or scan to discover configs."
            )
            return

        # Build diagnostic summary
        lines = ["[TrainingTriggerDaemon] DIAGNOSTIC SUMMARY:"]
        lines.append(f"  Tracked configs: {len(self._training_states)}")
        lines.append(f"  Evaluation backpressure: {'ACTIVE (paused)' if self._evaluation_backpressure else 'OK'}")
        lines.append(f"  Local-only mode: {'YES' if self._local_only_mode else 'NO'}")

        active_training = []
        blocked_configs = []
        ready_configs = []

        for config_key, state in sorted(self._training_states.items()):
            # Check if actively training
            if state.training_in_progress:
                active_training.append(
                    f"    - {config_key}: IN PROGRESS (started {self._format_age(state.training_start_time)})"
                )
                continue

            # Check why training is blocked
            can_train, reason = await self._check_training_conditions(config_key)

            if can_train:
                ready_configs.append(
                    f"    - {config_key}: READY ({state.npz_sample_count:,} samples, "
                    f"Elo={state.last_elo:.0f}, intensity={state.training_intensity})"
                )
            else:
                blocked_configs.append(
                    f"    - {config_key}: BLOCKED - {reason} "
                    f"(samples={state.npz_sample_count:,}, Elo={state.last_elo:.0f})"
                )

        # Log summary sections
        if active_training:
            lines.append(f"  Active training ({len(active_training)}):")
            lines.extend(active_training)

        if ready_configs:
            lines.append(f"  Ready to train ({len(ready_configs)}):")
            lines.extend(ready_configs)

        if blocked_configs:
            lines.append(f"  Blocked ({len(blocked_configs)}):")
            lines.extend(blocked_configs)

        # Log all at once to keep summary together
        logger.info("\n".join(lines))

        # If everything is blocked, log a warning with suggestions
        if blocked_configs and not active_training and not ready_configs:
            # Count common blockers
            blockers = {}
            for line in blocked_configs:
                if "cooldown" in line.lower():
                    blockers["cooldown"] = blockers.get("cooldown", 0) + 1
                elif "insufficient samples" in line.lower():
                    blockers["insufficient_samples"] = blockers.get("insufficient_samples", 0) + 1
                elif "quality" in line.lower():
                    blockers["quality"] = blockers.get("quality", 0) + 1
                elif "paused" in line.lower():
                    blockers["paused"] = blockers.get("paused", 0) + 1

            if blockers:
                top_blocker = max(blockers.items(), key=lambda x: x[1])
                logger.warning(
                    f"[TrainingTriggerDaemon] All {len(blocked_configs)} configs are blocked! "
                    f"Top blocker: {top_blocker[0]} ({top_blocker[1]} configs). "
                    f"Check: 1) NPZ export daemon running? 2) Quality scores? 3) Cooldown settings?"
                )

    def _format_age(self, timestamp: float) -> str:
        """Format a timestamp as human-readable age string."""
        if timestamp <= 0:
            return "unknown"
        age_seconds = time.time() - timestamp
        if age_seconds < 60:
            return f"{age_seconds:.0f}s ago"
        elif age_seconds < 3600:
            return f"{age_seconds/60:.0f}m ago"
        else:
            return f"{age_seconds/3600:.1f}h ago"

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

    async def _sync_elo_from_unified_db(self) -> None:
        """Periodically sync Elo ratings from unified_elo.db to training trigger state.

        Feb 2026: Fixes stale last_elo values that caused incorrect simulation budgets.
        The training_trigger_state only updates last_elo when EVALUATION_COMPLETED events
        fire. If evaluations don't run for a config, last_elo stays at default 1500,
        causing the budget calculator to use bootstrap-tier budgets even for strong models.
        """
        now = time.time()
        if now - self._last_elo_db_sync < self._elo_db_sync_interval:
            return

        self._last_elo_db_sync = now

        def _do_sync() -> int:
            """Blocking sync (runs in thread)."""
            import sqlite3

            db_path = Path("data/unified_elo.db")
            if not db_path.exists():
                return 0

            updated = 0
            try:
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT board_type || '_' || num_players || 'p' as config_key, "
                    "MAX(rating) as best_rating "
                    "FROM elo_ratings GROUP BY board_type, num_players"
                ).fetchall()
                conn.close()

                for row in rows:
                    config_key = row["config_key"]
                    best_elo = row["best_rating"]
                    if config_key in self._training_states:
                        state = self._training_states[config_key]
                        if best_elo > state.last_elo + 5.0:
                            state.last_elo = best_elo
                            updated += 1
            except (sqlite3.Error, OSError) as e:
                logger.debug(f"[TrainingTriggerDaemon] Elo DB sync failed: {e}")

            return updated

        try:
            updated = await asyncio.to_thread(_do_sync)
            if updated > 0:
                logger.info(
                    f"[TrainingTriggerDaemon] Synced Elo from unified_elo.db: "
                    f"{updated} configs updated"
                )
        except Exception as e:
            logger.debug(f"[TrainingTriggerDaemon] Elo sync error: {e}")

    async def _check_backpressure_recovery_timeout(self) -> None:
        """Check if evaluation backpressure has exceeded max duration and auto-release.

        January 2, 2026: Added to prevent indefinite training pause if
        EVALUATION_BACKPRESSURE_RELEASED event is lost or never received.
        """
        if not self._evaluation_backpressure:
            return

        last_backpressure = self._backpressure_stats.get("last_backpressure_time", 0)
        if last_backpressure <= 0:
            return

        now = time.time()
        duration = now - last_backpressure

        if duration >= self.config.backpressure_max_duration_seconds:
            self._evaluation_backpressure = False
            self._backpressure_stats["resumes_after_backpressure"] += 1
            self._backpressure_stats["auto_recovery_count"] = (
                self._backpressure_stats.get("auto_recovery_count", 0) + 1
            )

            duration_minutes = duration / 60
            max_minutes = self.config.backpressure_max_duration_seconds / 60
            logger.warning(
                f"[TrainingTriggerDaemon] Auto-released evaluation backpressure after "
                f"{duration_minutes:.1f}m (max: {max_minutes:.0f}m). "
                f"Training RESUMED - possible lost BACKPRESSURE_RELEASED event."
            )

            # Emit an event for visibility
            try:
                from app.distributed.data_events import DataEventType

                bus = self._get_event_bus()
                if bus:
                    bus.publish_sync(
                        "training_backpressure_auto_released",
                        {
                            "duration_seconds": duration,
                            "max_duration_seconds": self.config.backpressure_max_duration_seconds,
                            "auto_recovery_count": self._backpressure_stats["auto_recovery_count"],
                            "timestamp": now,
                            "source": "TrainingTriggerDaemon",
                        },
                    )
            except Exception:
                pass  # Event emission is optional

    async def _scan_for_training_opportunities(self) -> None:
        """Scan for configs that may need training."""
        try:
            # Check existing states
            for config_key in list(self._training_states.keys()):
                await self._maybe_trigger_training(config_key)

            # Also scan for NPZ files that haven't been tracked
            # January 3, 2026: Skip files already known via event-driven cache
            training_dir = Path(__file__).resolve().parent.parent.parent / "data" / "training"
            if training_dir.exists():
                for npz_path in training_dir.glob("*.npz"):
                    # Parse config from filename using robust regex
                    board_type, num_players = self._parse_config_from_filename(npz_path.stem)
                    if board_type is None or num_players is None:
                        continue

                    config_key = make_config_key(board_type, num_players)

                    # January 3, 2026: Skip if already in cache with same or newer mtime
                    # This avoids redundant disk I/O when events already informed us
                    if config_key in self._npz_cache:
                        cached_mtime, cached_samples, cached_path = self._npz_cache[config_key]
                        current_mtime = npz_path.stat().st_mtime
                        if current_mtime <= cached_mtime:
                            # File hasn't changed since last event, skip disk read
                            continue

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
                                # Update cache with discovered file
                                self._npz_cache[config_key] = (
                                    state.last_npz_update,
                                    state.npz_sample_count,
                                    str(npz_path),
                                )
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
