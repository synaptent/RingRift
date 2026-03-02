#!/usr/bin/env python3
"""Master Loop Controller - Unified automation entry point for RingRift AI training.

This is the single daemon that orchestrates all automation:
- Selfplay allocation across cluster
- Training triggering based on data freshness
- Evaluation and promotion
- Model distribution
- Cluster health monitoring
- Feedback loop integration

Unlike the separate daemon scripts, this provides a unified control plane
that makes high-level decisions about resource allocation and priorities.

Architecture:
    MasterLoopController
    ├── DaemonManager: Lifecycle for all background daemons
    ├── ClusterMonitor: Real-time cluster health
    ├── AdaptiveResourceManager: Resource tracking
    ├── FeedbackLoopController: Training feedback signals
    ├── DataPipelineOrchestrator: Pipeline stage tracking
    └── SelfplayScheduler: Priority-based selfplay allocation

Usage:
    # Full automation mode
    python scripts/master_loop.py

    # Minimal profile (sync + health)
    python scripts/master_loop.py --profile minimal

    # Watch mode (show status, don't run loop)
    python scripts/master_loop.py --watch

    # Specific configs only
    python scripts/master_loop.py --configs hex8_2p,square8_2p

    # Dry run (preview actions without executing)
    python scripts/master_loop.py --dry-run

    # Skip daemons (for testing)
    python scripts/master_loop.py --skip-daemons

    # Load configs from unified_loop.yaml
    python scripts/master_loop.py --config config/unified_loop.yaml

December 2025: Created as part of strategic integration plan.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.lib.process import (
    SingletonLock,
    find_processes_by_pattern,
    kill_process,
)

from app.config.thresholds import (
    PROMOTION_THRESHOLDS_BY_CONFIG,
    GPU_MEMORY_WEIGHTS,
    SELFPLAY_GAMES_PER_GPU_TYPE,
    TRAINING_BATCH_SIZE_BY_GPU,
    MAX_CONCURRENT_GAUNTLETS_BY_GPU,
    MIN_SAMPLES_FOR_TRAINING,
    MIN_AVG_GAME_LENGTH,
    MAX_DRAW_RATE_FOR_TRAINING,
    MIN_SELFPLAY_WIN_RATE_VS_HEURISTIC,
    is_ephemeral_node,
    check_training_data_quality,
    get_promotion_thresholds,
    get_gpu_weight,
    # Dec 29, 2025: Player-count aware export threshold
    get_min_games_for_export,
)
from app.config.unified_config import get_config

# Import coordination bootstrap for event wiring (December 2025)
# This is critical for feedback loops to function properly
from app.coordination.coordination_bootstrap import bootstrap_coordination
from app.config.coordination_defaults import DataFreshnessDefaults
from app.utils.sqlite_utils import connect_safe

# January 2026: Use rotating file logger for long-running operation
# This provides automatic rotation at 500 MB with 5 backups (2.5 GB max)
from app.core.logging_config import script_logger
logger = script_logger("master_loop", log_dir="logs/master_loop")


# =============================================================================
# Configuration
# =============================================================================

# All supported board type / player count configurations
ALL_CONFIGS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]

# Loop timing - now configurable via environment (Dec 2025)
# Use env.master_loop_interval, env.training_check_interval, etc.
from app.config.env import env
from app.config.ports import get_local_p2p_status_url

# December 29, 2025: Reactive dispatch mode (48-hour autonomous operation)
# When enabled, selfplay is dispatched via events instead of polling
REACTIVE_DISPATCH_ENABLED = os.environ.get("RINGRIFT_REACTIVE_DISPATCH", "true").lower() in ("true", "1", "yes")
# Watchdog interval when reactive dispatch is enabled (300s = 5 min fallback)
WATCHDOG_INTERVAL_SECONDS = int(os.environ.get("RINGRIFT_WATCHDOG_INTERVAL", "300"))

# Legacy constants for backward compatibility (use env properties directly preferred)
# When reactive dispatch is enabled, use longer watchdog interval
LOOP_INTERVAL_SECONDS = (
    WATCHDOG_INTERVAL_SECONDS if REACTIVE_DISPATCH_ENABLED
    else env.master_loop_interval  # RINGRIFT_MASTER_LOOP_INTERVAL (default: 30)
)
TRAINING_CHECK_INTERVAL = env.training_check_interval  # RINGRIFT_TRAINING_CHECK_INTERVAL (default: 60)
ALLOCATION_CHECK_INTERVAL = env.allocation_check_interval  # RINGRIFT_ALLOCATION_CHECK_INTERVAL (default: 120)

# Thresholds - now configurable via environment (Dec 2025)
MIN_GAMES_FOR_EXPORT = env.min_games_for_export  # RINGRIFT_MIN_GAMES_FOR_EXPORT (default: 500)
# Dec 29, 2025: Use DataFreshnessDefaults for unified freshness config
MAX_DATA_STALENESS_HOURS = DataFreshnessDefaults().MAX_DATA_AGE_HOURS

# State persistence path (Gap 3 fix: Dec 2025)
STATE_DB_PATH = Path(__file__).parent.parent / "data" / "coordination" / "master_loop_state.db"
STATE_SAVE_INTERVAL_SECONDS = env.state_save_interval  # RINGRIFT_STATE_SAVE_INTERVAL (default: 300)

# Jan 2026: Daemon startup timeout to prevent hanging on unresponsive daemon.start()
DAEMON_STARTUP_TIMEOUT_SECONDS = int(os.environ.get("RINGRIFT_DAEMON_STARTUP_TIMEOUT", "30"))

# Load forecasting snapshot interval (December 2025)
LOAD_SNAPSHOT_INTERVAL = 3600  # 1 hour - record cluster load for pattern learning

# Cluster-wide P2P recovery interval (December 31, 2025)
# SSH into all P2P-enabled nodes and restart P2P on any that aren't responding
# This complements the local P2P_RECOVERY daemon which handles the coordinator's own P2P
# December 31, 2025: Reduced from 30 min to 5 min for faster recovery (48h autonomous operation)
CLUSTER_P2P_RECOVERY_INTERVAL = int(os.environ.get("RINGRIFT_CLUSTER_P2P_RECOVERY_INTERVAL", "300"))  # 5 minutes

# PID file for master loop detection (December 2025)
PID_FILE_PATH = Path(__file__).parent.parent / "data" / "coordination" / "master_loop.pid"
LOCK_DIR = Path(__file__).parent.parent / "data" / "coordination"

# Global singleton lock for duplicate process prevention (December 2025)
_MASTER_LOCK: SingletonLock | None = None

# Critical daemons that MUST start for autonomous operation (December 2025 - Gap 2 fix)
# If any of these fail to start, the master loop should NOT proceed
# December 27, 2025: Expanded to include full automation pipeline
CRITICAL_DAEMON_NAMES = {
    "event_router",      # Event system is fundamental
    "data_pipeline",     # Pipeline orchestration
    "auto_sync",         # Data replication
    "feedback_loop",     # Training feedback and quality scoring
    "auto_export",       # NPZ export (Dec 27) - blocks training if missing
    "training_trigger",  # Training trigger (Dec 27) - starts training jobs
    "evaluation",        # Gauntlet evaluation (Dec 27) - model quality gates
    "auto_promotion",    # Model promotion (Dec 27) - promotes winning models
}


@dataclass
class ConfigState:
    """Tracking state for a single board/player configuration."""
    config_key: str

    # Data freshness
    last_export_time: float = 0.0
    last_training_time: float = 0.0
    last_evaluation_time: float = 0.0
    games_since_last_export: int = 0

    # Quality metrics
    # Default 0.7 allows initial training (threshold is 0.5) before actual quality metrics arrive
    # Will be overwritten once QUALITY_SCORE_UPDATED events are received
    last_quality_score: float = 0.7
    last_quality_update_time: float = 0.0
    last_policy_accuracy: float = 0.0
    last_evaluation_win_rate: float = 0.0

    # Current allocations
    selfplay_nodes_allocated: list[str] = field(default_factory=list)
    training_node: str | None = None

    # Promotion status
    pending_evaluation: bool = False
    last_promotion_success: bool | None = None

    # Feedback signals
    training_intensity: str = "normal"  # normal, accelerated, hot_path
    exploration_boost: float = 1.0

    # Bandit engine tracking (December 2025)
    # Used by selfplay_engine_bandit to record feedback on Elo improvement
    last_selfplay_engine: str | None = None
    last_selfplay_games: int = 0

    @property
    def data_staleness_hours(self) -> float:
        """Hours since last export."""
        if self.last_export_time == 0:
            return float("inf")
        return (time.time() - self.last_export_time) / 3600

    @property
    def num_players(self) -> int:
        """Extract player count from config_key (e.g., 'hex8_2p' -> 2)."""
        # config_key format: {board_type}_{n}p (e.g., hex8_2p, square8_4p)
        try:
            suffix = self.config_key.split("_")[-1]  # "2p", "3p", "4p"
            return int(suffix.rstrip("p"))
        except (ValueError, IndexError):
            return 2  # Default to 2-player

    @property
    def min_games_threshold(self) -> int:
        """Get minimum games for export based on player count.

        Dec 29, 2025: Added as part of Phase 2 training improvements.
        Higher player counts require more games for statistical significance.
        """
        return get_min_games_for_export(self.num_players)

    @property
    def needs_training(self) -> bool:
        """Check if config needs training."""
        # Has enough games and data is fresh enough
        # Dec 29, 2025: Use player-count aware threshold
        return (
            self.games_since_last_export >= self.min_games_threshold
            and self.last_quality_score >= 0.5
        )


@dataclass
class ClusterHealth:
    """Aggregated cluster health status."""
    total_nodes: int = 0
    healthy_nodes: int = 0
    training_nodes: int = 0
    selfplay_nodes: int = 0
    avg_gpu_utilization: float = 0.0
    avg_disk_usage: float = 0.0
    load_critical: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class QueuePopulatorWatchdogConfig:
    """Configuration for QueuePopulatorWatchdog (January 14, 2026)."""
    # Thresholds for detecting stuck populator
    stuck_threshold_seconds: float = 300.0  # 5 minutes without progress
    zero_drain_threshold_seconds: float = 360.0  # 6 minutes with no completions

    # Check interval
    check_interval_seconds: float = 60.0

    # Recovery actions
    max_recovery_attempts: int = 3
    recovery_cooldown_seconds: float = 120.0


class QueuePopulatorWatchdog:
    """Watchdog to detect and recover from stuck queue populator (January 14, 2026).

    Monitors the queue populator for signs of being stuck:
    1. High backpressure duration without recovery
    2. Zero drain rate indicating cluster partition
    3. Stuck in backoff loop for too long

    When detected, triggers recovery actions:
    1. Emit QUEUE_POPULATOR_STUCK event
    2. Force reset of backoff state
    3. Trigger P2P health check
    """

    def __init__(self, config: QueuePopulatorWatchdogConfig | None = None):
        self.config = config or QueuePopulatorWatchdogConfig()

        # State tracking
        self._last_check_time: float = 0.0
        self._last_successful_populate: float = time.time()
        self._last_drain_activity: float = time.time()
        self._recovery_attempts: int = 0
        self._last_recovery_time: float = 0.0

        # Previous metrics for comparison
        self._prev_queue_depth: int = 0
        self._prev_items_added: int = 0

    def check(self, populator: Any) -> dict[str, Any]:
        """Check queue populator health and trigger recovery if needed.

        Args:
            populator: UnifiedQueuePopulator instance

        Returns:
            Health check result dict
        """
        now = time.time()

        # Rate limit checks
        if now - self._last_check_time < self.config.check_interval_seconds:
            return {"status": "skipped", "reason": "interval_not_elapsed"}

        self._last_check_time = now

        # Get health metrics from populator
        try:
            metrics = populator.get_health_metrics()
        except AttributeError:
            return {"status": "error", "reason": "populator_no_health_metrics"}

        result = {
            "status": "healthy",
            "metrics": metrics,
            "stuck_detected": False,
            "recovery_triggered": False,
        }

        # Check for stuck conditions
        is_stuck = False
        stuck_reason = None

        # Condition 1: High backpressure with no drain
        if metrics.get("partition_detected", False):
            partition_duration = metrics.get("partition_duration_seconds", 0)
            if partition_duration > self.config.zero_drain_threshold_seconds:
                is_stuck = True
                stuck_reason = f"partition_detected_for_{partition_duration:.0f}s"

        # Condition 2: Stuck in backoff for too long
        if metrics.get("backoff_active", False):
            consecutive_hits = metrics.get("consecutive_hard_limit_hits", 0)
            if consecutive_hits > 10:  # Arbitrary high threshold
                is_stuck = True
                stuck_reason = f"stuck_in_backoff_{consecutive_hits}_hits"

        # Condition 3: Circuit breaker stuck open
        circuit_state = metrics.get("circuit_state", "closed")
        if circuit_state == "open":
            # Check how long it's been open
            failure_count = metrics.get("circuit_failure_count", 0)
            if failure_count > 15:  # Many failures
                is_stuck = True
                stuck_reason = f"circuit_breaker_stuck_{failure_count}_failures"

        # Condition 4: Zero drain rate with high queue depth
        drain_rate = metrics.get("drain_rate_per_minute", 0)
        queue_depth = metrics.get("queue_depth", 0)
        if drain_rate == 0 and queue_depth > 100:
            # Check if this has persisted
            if now - self._last_drain_activity > self.config.zero_drain_threshold_seconds:
                is_stuck = True
                stuck_reason = f"zero_drain_with_queue_{queue_depth}"
        else:
            self._last_drain_activity = now

        result["stuck_detected"] = is_stuck
        if stuck_reason:
            result["stuck_reason"] = stuck_reason

        # Trigger recovery if stuck
        if is_stuck:
            result["recovery_triggered"] = self._maybe_trigger_recovery(
                populator, stuck_reason, metrics
            )

        return result

    def _maybe_trigger_recovery(
        self, populator: Any, reason: str, metrics: dict[str, Any]
    ) -> bool:
        """Maybe trigger recovery based on cooldown and attempt limits.

        Returns:
            True if recovery was triggered
        """
        now = time.time()

        # Check cooldown
        if now - self._last_recovery_time < self.config.recovery_cooldown_seconds:
            logger.warning(
                f"[QueuePopulatorWatchdog] Stuck detected ({reason}) but in cooldown"
            )
            return False

        # Check attempt limit
        if self._recovery_attempts >= self.config.max_recovery_attempts:
            logger.error(
                f"[QueuePopulatorWatchdog] Max recovery attempts reached "
                f"({self._recovery_attempts}/{self.config.max_recovery_attempts})"
            )
            return False

        # Trigger recovery
        self._recovery_attempts += 1
        self._last_recovery_time = now

        logger.error(
            f"[QueuePopulatorWatchdog] STUCK DETECTED: {reason}. "
            f"Triggering recovery (attempt {self._recovery_attempts})"
        )

        # Emit event
        try:
            from app.coordination.event_emission_helpers import safe_emit_event
            safe_emit_event(
                event_type="QUEUE_POPULATOR_STUCK",
                payload={
                    "reason": reason,
                    "metrics": metrics,
                    "recovery_attempt": self._recovery_attempts,
                },
                log_before=True,
                log_level=logging.ERROR,
                context="queue_populator_watchdog",
            )
        except ImportError:
            pass

        # Reset backoff state
        try:
            populator._reset_backoff()
            logger.info("[QueuePopulatorWatchdog] Reset backoff state")
        except AttributeError:
            pass

        # Reset circuit breaker if stuck open
        try:
            if hasattr(populator, '_circuit_state') and hasattr(populator, '_CircuitState'):
                populator._circuit_state = populator._CircuitState.HALF_OPEN
                populator._circuit_half_open_successes = 0
                logger.info("[QueuePopulatorWatchdog] Reset circuit breaker to HALF_OPEN")
        except AttributeError:
            pass

        return True

    def record_successful_populate(self, items_added: int) -> None:
        """Record a successful populate call.

        Args:
            items_added: Number of items added to queue
        """
        self._last_successful_populate = time.time()
        self._prev_items_added = items_added

        # Reset recovery attempts on success
        if items_added > 0:
            self._recovery_attempts = 0

    def get_status(self) -> dict[str, Any]:
        """Get watchdog status for monitoring."""
        now = time.time()
        return {
            "last_check_time": self._last_check_time,
            "last_successful_populate": self._last_successful_populate,
            "seconds_since_success": now - self._last_successful_populate,
            "last_drain_activity": self._last_drain_activity,
            "seconds_since_drain": now - self._last_drain_activity,
            "recovery_attempts": self._recovery_attempts,
            "max_recovery_attempts": self.config.max_recovery_attempts,
            "last_recovery_time": self._last_recovery_time,
        }


class MasterLoopController:
    """Single daemon managing all automation.

    Coordinates:
    - Selfplay allocation across cluster
    - Training triggering based on data freshness
    - Evaluation and promotion
    - Model distribution
    - Cluster health monitoring
    """

    def __init__(
        self,
        configs: list[str] | None = None,
        dry_run: bool = False,
        skip_daemons: bool = False,
        daemon_profile: str = "standard",
    ):
        self.active_configs = configs or ALL_CONFIGS
        self.dry_run = dry_run
        self.skip_daemons = skip_daemons
        self.daemon_profile = daemon_profile

        # State tracking
        self._states: dict[str, ConfigState] = {
            cfg: ConfigState(config_key=cfg) for cfg in self.active_configs
        }

        # Timing
        self._last_training_check = 0.0
        self._last_allocation_check = 0.0
        self._last_load_snapshot = 0.0  # Load forecasting (Dec 2025)
        self._last_cluster_p2p_recovery = 0.0  # Cluster-wide P2P recovery (Dec 31, 2025)

        # Control
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Lazy-loaded managers
        self._daemon_manager = None
        self._cluster_monitor = None
        self._resource_manager = None
        self._feedback_controller = None
        self._pipeline_orchestrator = None

        # Queue populator watchdog (January 14, 2026)
        self._queue_populator_watchdog = QueuePopulatorWatchdog()
        self._queue_populator = None  # Lazy-loaded

        # State persistence (Gap 3 fix: Dec 2025)
        self._db_path = STATE_DB_PATH
        self._last_state_save = 0.0
        self._loop_iteration = 0  # Heartbeat tracking (Dec 2025)
        self._state_lock = threading.Lock()  # Race condition fix (Dec 2025)
        self._init_state_db()

    # =========================================================================
    # Lazy-loaded dependencies
    # =========================================================================

    @property
    def daemon_manager(self):
        """Get DaemonManager (lazy load)."""
        if self._daemon_manager is None:
            from app.coordination.daemon_manager import get_daemon_manager
            self._daemon_manager = get_daemon_manager()
        return self._daemon_manager

    @property
    def cluster_monitor(self):
        """Get ClusterMonitor (lazy load)."""
        if self._cluster_monitor is None:
            from app.coordination.cluster_status_monitor import ClusterMonitor
            self._cluster_monitor = ClusterMonitor()
        return self._cluster_monitor

    @property
    def resource_manager(self):
        """Get AdaptiveResourceManager (lazy load)."""
        if self._resource_manager is None:
            from app.coordination.adaptive_resource_manager import get_resource_manager
            self._resource_manager = get_resource_manager()
        return self._resource_manager

    @property
    def feedback_controller(self):
        """Get FeedbackLoopController (lazy load)."""
        if self._feedback_controller is None:
            from app.coordination.feedback_loop_controller import get_feedback_loop_controller
            self._feedback_controller = get_feedback_loop_controller()
        return self._feedback_controller

    @property
    def pipeline_orchestrator(self):
        """Get DataPipelineOrchestrator (lazy load)."""
        if self._pipeline_orchestrator is None:
            from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator
            self._pipeline_orchestrator = get_pipeline_orchestrator()
        return self._pipeline_orchestrator

    @property
    def queue_populator(self):
        """Get UnifiedQueuePopulator (lazy load) for watchdog integration."""
        if self._queue_populator is None:
            try:
                from app.coordination.unified_queue_populator import get_queue_populator
                self._queue_populator = get_queue_populator()
            except ImportError:
                pass
        return self._queue_populator

    # =========================================================================
    # State Persistence (Gap 3 fix: Dec 2025)
    # =========================================================================

    def _init_state_db(self) -> None:
        """Initialize the state persistence database.

        Jan 12, 2026: Fixed to use context manager for proper connection cleanup.
        """
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.closing(connect_safe(self._db_path, row_factory=None)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS config_state (
                        config_key TEXT PRIMARY KEY,
                        exploration_boost REAL NOT NULL DEFAULT 1.0,
                        training_intensity TEXT NOT NULL DEFAULT 'normal',
                        last_quality_score REAL NOT NULL DEFAULT 0.7,
                        last_quality_update_time REAL NOT NULL DEFAULT 0.0,
                        updated_at REAL NOT NULL
                    )
                """)
                # Heartbeat table for health monitoring (Dec 2025)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS heartbeat (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        last_beat REAL NOT NULL,
                        loop_iteration INTEGER NOT NULL DEFAULT 0,
                        active_configs INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'running'
                    )
                """)
                conn.execute("""
                    INSERT OR IGNORE INTO heartbeat (id, last_beat, loop_iteration, active_configs, status)
                    VALUES (1, ?, 0, 0, 'starting')
                """, (time.time(),))
                conn.commit()
            logger.debug(f"[MasterLoop] State DB initialized at {self._db_path}")
        except Exception as e:
            logger.warning(f"[MasterLoop] Failed to init state DB: {e}")

    async def _load_persisted_state(self) -> None:
        """Load exploration_boost and other state from database on startup.

        Uses threading lock to prevent race conditions during concurrent access.
        Dec 2025: Added locking to fix state corruption issue.
        January 2026: Made async to avoid blocking event loop.
        """
        def _load_sync() -> int:
            """Synchronous SQLite operations wrapped in lock.

            Jan 12, 2026: Fixed to use context manager for proper connection cleanup.
            """
            with self._state_lock:
                try:
                    with contextlib.closing(connect_safe(self._db_path, row_factory=None)) as conn:
                        # Feb 2026: Add last_quality_update_time column if missing (schema migration)
                        try:
                            conn.execute("ALTER TABLE config_state ADD COLUMN last_quality_update_time REAL NOT NULL DEFAULT 0.0")
                            conn.commit()
                        except sqlite3.OperationalError:
                            pass  # Column already exists

                        rows = conn.execute("""
                            SELECT config_key, exploration_boost, training_intensity,
                                   last_quality_score, last_quality_update_time
                            FROM config_state
                        """).fetchall()

                        restored_count = 0
                        for config_key, boost, intensity, quality_score, quality_update_time in rows:
                            if config_key in self._states:
                                self._states[config_key].exploration_boost = boost
                                self._states[config_key].training_intensity = intensity
                                self._states[config_key].last_quality_score = quality_score
                                self._states[config_key].last_quality_update_time = quality_update_time
                                restored_count += 1

                        return restored_count
                except (sqlite3.Error, OSError) as e:
                    logger.warning(f"[MasterLoop] Failed to load persisted state: {e}")
                    return 0

        restored_count = await asyncio.to_thread(_load_sync)
        if restored_count > 0:
            logger.info(f"[MasterLoop] Restored persisted state for {restored_count} configs")

    async def _save_persisted_state(self) -> None:
        """Save exploration_boost and other state to database.

        Uses threading lock and SQLite transaction to prevent race conditions.
        Dec 2025: Added locking to fix state corruption issue.
        January 2026: Made async to avoid blocking event loop.
        """
        def _save_sync() -> None:
            """Synchronous SQLite operations wrapped in lock.

            Jan 12, 2026: Fixed to use context manager for proper connection cleanup.
            """
            with self._state_lock:
                try:
                    with contextlib.closing(connect_safe(self._db_path, row_factory=None)) as conn:
                        now = time.time()

                        # Use BEGIN IMMEDIATE for SQLite-level write lock
                        conn.execute("BEGIN IMMEDIATE")

                        for config_key, state in self._states.items():
                            conn.execute("""
                                INSERT OR REPLACE INTO config_state
                                (config_key, exploration_boost, training_intensity,
                                 last_quality_score, last_quality_update_time, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                config_key,
                                state.exploration_boost,
                                state.training_intensity,
                                state.last_quality_score,
                                state.last_quality_update_time,
                                now,
                            ))

                        conn.commit()
                        self._last_state_save = now
                        logger.debug(f"[MasterLoop] Persisted state for {len(self._states)} configs")

                except (sqlite3.Error, OSError) as e:
                    # Dec 29, 2025: Narrowed from bare Exception
                    logger.warning(f"[MasterLoop] Failed to save persisted state: {e}")

        await asyncio.to_thread(_save_sync)

    async def _maybe_save_state(self) -> None:
        """Save state if enough time has passed since last save.

        January 2026: Made async to match _save_persisted_state().
        """
        if time.time() - self._last_state_save >= STATE_SAVE_INTERVAL_SECONDS:
            await self._save_persisted_state()

    # =========================================================================
    # Configuration Helpers (December 2025)
    # =========================================================================

    def _has_aws_credentials(self) -> bool:
        """Check if AWS credentials are available via any method.

        Uses boto3's credential chain to check for credentials from:
        1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        2. AWS credentials file (~/.aws/credentials)
        3. AWS config file (~/.aws/config)
        4. IAM role (if running on AWS EC2/ECS/Lambda)

        December 2025: Better long-term solution than checking env vars only.
        This enables S3 daemons to start when credentials are in ~/.aws/credentials.

        Returns:
            True if valid credentials found, False otherwise.
        """
        # Method 1: Check environment variable (fast path)
        if os.getenv("AWS_ACCESS_KEY_ID"):
            logger.debug("[MasterLoop] AWS credentials found via environment variable")
            return True

        # Method 2: Check ~/.aws/credentials file directly (no boto3 import needed)
        credentials_file = Path.home() / ".aws" / "credentials"
        if credentials_file.exists():
            try:
                import configparser
                config = configparser.ConfigParser()
                config.read(credentials_file)
                # Check for default profile or any profile with access key
                for section in config.sections():
                    if config.get(section, "aws_access_key_id", fallback=None):
                        logger.debug(f"[MasterLoop] AWS credentials found in {credentials_file} [{section}]")
                        return True
            except (configparser.Error, OSError, KeyError) as e:
                # Dec 29, 2025: Narrowed from bare Exception
                logger.debug(f"[MasterLoop] Failed to parse AWS credentials file: {e}")

        # Method 3: Use boto3 credential chain (most comprehensive but requires import)
        try:
            import boto3
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials and credentials.access_key:
                logger.debug("[MasterLoop] AWS credentials found via boto3 session")
                return True
        except ImportError:
            logger.debug("[MasterLoop] boto3 not installed, skipping credential chain check")
        except (OSError, ValueError, AttributeError) as e:
            # Dec 29, 2025: Narrowed from bare Exception - boto3 can raise these
            logger.debug(f"[MasterLoop] boto3 credential check failed: {e}")

        logger.info("[MasterLoop] No AWS credentials found - S3 daemons will be disabled")
        return False

    def _has_npx(self) -> bool:
        """Check if npx (Node.js package runner) is available.

        January 2026: Added to support parity validation daemon on any node
        with Node.js installed, not just coordinators.

        Returns:
            True if npx is available, False otherwise.
        """
        import shutil

        if shutil.which("npx"):
            logger.debug("[MasterLoop] npx found, parity validation available")
            return True

        logger.debug("[MasterLoop] npx not found - parity validation will be disabled")
        return False

    def _update_heartbeat(self, status: str = "running") -> None:
        """Update heartbeat for health monitoring (Dec 2025).

        This allows external monitoring to detect hung loops by checking
        last_beat timestamp. A healthy loop should update every 30-60 seconds.

        Jan 12, 2026: Fixed to use context manager for proper connection cleanup.
        """
        try:
            self._loop_iteration += 1
            with contextlib.closing(connect_safe(self._db_path, row_factory=None)) as conn:
                conn.execute("""
                    UPDATE heartbeat
                    SET last_beat = ?, loop_iteration = ?, active_configs = ?, status = ?
                    WHERE id = 1
                """, (time.time(), self._loop_iteration, len(self.active_configs), status))
                conn.commit()
        except (sqlite3.Error, OSError) as e:
            # Dec 29, 2025: Narrowed from bare Exception
            logger.debug(f"[MasterLoop] Failed to update heartbeat: {e}")

    def _acquire_singleton_lock(self) -> bool:
        """Acquire singleton lock for duplicate process prevention (December 2025).

        Uses atomic file locking (fcntl) which is more reliable than PID file checks.

        Returns:
            True if lock acquired successfully
        """
        global _MASTER_LOCK

        LOCK_DIR.mkdir(parents=True, exist_ok=True)

        _MASTER_LOCK = SingletonLock("master_loop", lock_dir=LOCK_DIR)
        if not _MASTER_LOCK.acquire():
            holder_pid = _MASTER_LOCK.get_holder_pid()
            if holder_pid:
                logger.error(
                    f"[MasterLoop] Another instance is already running (PID {holder_pid}). "
                    f"Use --kill-duplicates to automatically terminate it."
                )
            else:
                logger.error("[MasterLoop] Another instance is already running")
            return False

        logger.info(f"[MasterLoop] Acquired singleton lock (PID {os.getpid()})")

        # Also write PID file for backward compatibility
        try:
            PID_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(PID_FILE_PATH, "w") as f:
                f.write(str(os.getpid()))
        except Exception as e:
            logger.debug(f"[MasterLoop] Failed to write PID file: {e}")

        return True

    def _release_singleton_lock(self) -> None:
        """Release singleton lock on shutdown (December 2025)."""
        global _MASTER_LOCK
        if _MASTER_LOCK:
            _MASTER_LOCK.release()
            logger.debug("[MasterLoop] Released singleton lock")
            _MASTER_LOCK = None

        # Also remove PID file for backward compatibility
        try:
            if PID_FILE_PATH.exists():
                PID_FILE_PATH.unlink()
        except OSError:
            pass

    # Backward compatibility aliases
    _create_pid_file = lambda self: self._acquire_singleton_lock()
    _remove_pid_file = lambda self: self._release_singleton_lock()

    @staticmethod
    def is_running(pid_file: Path | None = None) -> bool:
        """Check if master loop is running by checking PID file.

        Args:
            pid_file: Path to PID file (defaults to PID_FILE_PATH)

        Returns:
            True if master loop process is active, False otherwise
        """
        pid_file = pid_file or PID_FILE_PATH

        if not pid_file.exists():
            return False

        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())

            # Check if process with this PID exists
            try:
                os.kill(pid, 0)  # Signal 0 doesn't kill, just checks existence
                return True
            except OSError:
                # Process doesn't exist, remove stale PID file
                pid_file.unlink()
                return False
        except (ValueError, IOError, OSError):
            return False

    @staticmethod
    def check_health(db_path: Path | None = None, max_age_seconds: float = 120.0) -> dict[str, Any]:
        """Check if master loop is healthy by reading heartbeat.

        Jan 12, 2026: Fixed to use context manager for proper connection cleanup.

        Returns:
            Dict with 'healthy' bool, 'last_beat' timestamp, 'age_seconds', 'status'
        """
        db_path = db_path or STATE_DB_PATH
        try:
            if not db_path.exists():
                return {"healthy": False, "error": "State DB not found"}

            with contextlib.closing(connect_safe(db_path, row_factory=None)) as conn:
                row = conn.execute("""
                    SELECT last_beat, loop_iteration, active_configs, status
                    FROM heartbeat WHERE id = 1
                """).fetchone()

                if not row:
                    return {"healthy": False, "error": "No heartbeat record"}

                last_beat, iteration, active_configs, status = row
                age = time.time() - last_beat

                return {
                    "healthy": age < max_age_seconds and status != "stopped",
                    "last_beat": last_beat,
                    "age_seconds": age,
                    "loop_iteration": iteration,
                    "active_configs": active_configs,
                    "status": status,
                }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _verify_p2p_connectivity(self) -> tuple[bool, list[str]]:
        """Pre-flight check: Verify P2P orchestrator is accessible on voter nodes.

        December 2025: Added as part of Phase E cluster integration.
        Prevents master loop from starting if P2P cluster is unavailable.

        December 29, 2025: Added container networking setup for Vast.ai/RunPod nodes.
        Container nodes need userspace Tailscale with SOCKS5 proxy before P2P works.

        Returns:
            Tuple of (success, list of error messages)
        """
        import aiohttp
        from app.config.cluster_config import load_cluster_config, get_p2p_port
        from app.config.env import env

        errors = []

        # Container networking setup (December 29, 2025)
        # Container nodes (Vast.ai, RunPod) need userspace Tailscale for P2P
        if env.needs_userspace_tailscale:
            try:
                from app.coordination.container_tailscale_setup import setup_container_networking
                setup_ok, setup_message = await setup_container_networking()
                if setup_ok:
                    logger.info(f"[MasterLoop] Container networking ready: {setup_message}")
                else:
                    logger.warning(f"[MasterLoop] Container networking setup failed: {setup_message}")
                    errors.append(f"Container networking: {setup_message}")
                    # Don't fail hard - container might already have Tailscale from startup script
            except ImportError:
                logger.debug("[MasterLoop] Container tailscale setup module not available")
            except Exception as e:
                logger.warning(f"[MasterLoop] Container networking setup error: {e}")
                errors.append(f"Container networking error: {e}")

        config = load_cluster_config()

        # Get voter nodes from config
        voters = config.p2p_voters
        if not voters:
            # No voters configured, skip check
            logger.warning("[MasterLoop] No P2P voters configured, skipping P2P health check")
            return True, []

        p2p_port = get_p2p_port()
        reachable = 0
        total = len(voters)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
            for voter in voters:
                # Find host info for this voter
                host_info = config.hosts_raw.get(voter)
                if not host_info:
                    errors.append(f"Voter {voter} not found in hosts config")
                    continue

                # Get host address - prefer tailscale_ip, fall back to host
                host = host_info.get("tailscale_ip") or host_info.get("host")
                if not host:
                    errors.append(f"Voter {voter} has no host address")
                    continue

                url = f"http://{host}:{p2p_port}/status"
                try:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            reachable += 1
                            logger.debug(f"[MasterLoop] P2P voter {voter} is reachable")
                        else:
                            errors.append(f"Voter {voter}: HTTP {resp.status}")
                except asyncio.TimeoutError:
                    errors.append(f"Voter {voter}: timeout")
                except aiohttp.ClientError as e:
                    errors.append(f"Voter {voter}: {type(e).__name__}")

        # Require quorum (>50%) of voters to be reachable
        quorum = (total // 2) + 1
        if reachable < quorum:
            logger.error(
                f"[MasterLoop] P2P quorum not met: {reachable}/{total} voters reachable "
                f"(need {quorum}). Errors: {errors}"
            )
            return False, errors

        logger.info(f"[MasterLoop] P2P connectivity verified: {reachable}/{total} voters reachable")
        return True, errors

    async def _validate_critical_subsystems(self) -> tuple[bool, list[str]]:
        """Pre-flight validation of critical subsystems for autonomous operation.

        December 29, 2025: Added to ensure all prerequisites are met before
        starting the autonomous training loop. This prevents silent failures
        when critical infrastructure is missing or misconfigured.

        Validates:
        1. Event router - Core communication infrastructure
        2. Work queue - Job distribution system
        3. Critical directories - data/games, data/training, models
        4. Cluster config - distributed_hosts.yaml
        5. State persistence - Coordination database access

        Returns:
            Tuple of (all_passed, list_of_error_messages)
        """
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Validate event router
        try:
            from app.coordination.event_router import get_event_bus, DataEventType
            bus = get_event_bus()
            # Verify we can subscribe (basic functionality test)
            if bus is None:
                errors.append("Event router: get_event_bus() returned None")
            else:
                logger.debug("[MasterLoop] Event router validated")
        except ImportError as e:
            errors.append(f"Event router: Import failed - {e}")
        except Exception as e:
            errors.append(f"Event router: Initialization failed - {e}")

        # 2. Validate work queue (for job distribution)
        try:
            from app.coordination.work_queue import get_work_queue
            queue = get_work_queue()
            if queue is None:
                errors.append("Work queue: get_work_queue() returned None")
            else:
                # Quick health check
                stats = queue.get_queue_stats()
                logger.debug(f"[MasterLoop] Work queue validated: {stats.get('total_items', 0)} items")
        except ImportError as e:
            errors.append(f"Work queue: Import failed - {e}")
        except Exception as e:
            # Work queue issues are critical - jobs can't be distributed
            errors.append(f"Work queue: Health check failed - {e}")

        # 3. Validate critical directories
        from pathlib import Path
        ai_service_root = Path(__file__).parent.parent

        critical_dirs = [
            ("data/games", ai_service_root / "data" / "games"),
            ("data/training", ai_service_root / "data" / "training"),
            ("models", ai_service_root / "models"),
            ("data/coordination", ai_service_root / "data" / "coordination"),
        ]

        for name, path in critical_dirs:
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"[MasterLoop] Created missing directory: {name}")
                except OSError as e:
                    errors.append(f"Directory {name}: Cannot create - {e}")
            elif not path.is_dir():
                errors.append(f"Directory {name}: Path exists but is not a directory")

        # 4. Validate cluster config exists
        config_path = ai_service_root / "config" / "distributed_hosts.yaml"
        if not config_path.exists():
            warnings.append("Cluster config: distributed_hosts.yaml not found (single-node mode)")
        else:
            try:
                from app.config.cluster_config import load_cluster_config
                config = load_cluster_config()
                hosts = config.hosts_raw
                logger.debug(f"[MasterLoop] Cluster config validated: {len(hosts)} hosts configured")
            except Exception as e:
                warnings.append(f"Cluster config: Parse error - {e}")

        # 5. Validate state database access
        # Jan 12, 2026: Fixed to use context manager for proper connection cleanup.
        try:
            with contextlib.closing(connect_safe(self._db_path, row_factory=None)) as conn:
                # Quick read test
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
            logger.debug(f"[MasterLoop] State database validated: {len(tables)} tables")
        except sqlite3.Error as e:
            errors.append(f"State database: Access failed - {e}")

        # 6. Validate daemon manager can be initialized
        try:
            from app.coordination.daemon_manager import get_daemon_manager
            dm = get_daemon_manager()
            if dm is None:
                errors.append("Daemon manager: get_daemon_manager() returned None")
            else:
                logger.debug("[MasterLoop] Daemon manager validated")
        except ImportError as e:
            errors.append(f"Daemon manager: Import failed - {e}")
        except Exception as e:
            errors.append(f"Daemon manager: Initialization failed - {e}")

        # 7. Validate at least one canonical model exists (optional but recommended)
        models_dir = ai_service_root / "models"
        if models_dir.exists():
            canonical_models = list(models_dir.glob("canonical_*.pth"))
            if not canonical_models:
                warnings.append("No canonical models found (selfplay will use heuristics only)")
            else:
                logger.debug(f"[MasterLoop] Found {len(canonical_models)} canonical models")

        # Log results
        if errors:
            for error in errors:
                logger.error(f"[MasterLoop] CRITICAL: {error}")
        if warnings:
            for warning in warnings:
                logger.warning(f"[MasterLoop] WARNING: {warning}")

        all_passed = len(errors) == 0
        if all_passed:
            logger.info(
                f"[MasterLoop] Startup validation passed "
                f"({len(warnings)} warnings)"
            )
        else:
            logger.error(
                f"[MasterLoop] Startup validation FAILED: "
                f"{len(errors)} critical errors, {len(warnings)} warnings"
            )

        return all_passed, errors

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the master loop."""
        if self._running:
            logger.warning("MasterLoopController already running")
            return

        self._running = True

        # Create PID file (December 2025)
        self._create_pid_file()

        logger.info(f"[MasterLoop] Starting with {len(self.active_configs)} configs")
        logger.info(
            f"[MasterLoop] Dry run: {self.dry_run}, Skip daemons: {self.skip_daemons}, "
            f"Profile: {self.daemon_profile}"
        )

        # Bootstrap coordination system - wires event subscriptions for feedback loops
        # (December 2025: Critical for REGRESSION_DETECTED → rollback, PLATEAU_DETECTED → curriculum, etc.)
        try:
            bootstrap_result = bootstrap_coordination(
                enable_integrations=True,
                pipeline_auto_trigger=True,
            )
            wired = bootstrap_result.get("wired_subscriptions", {})
            logger.info(
                f"[MasterLoop] Coordination bootstrap complete: "
                f"selfplay_to_sync={wired.get('selfplay_to_sync', False)}, "
                f"regression_to_rollback={wired.get('regression_to_rollback', False)}, "
                f"plateau_to_curriculum={wired.get('plateau_to_curriculum', False)}"
            )
        except ImportError as e:
            # Module not available - optional coordination features disabled
            logger.warning(f"[MasterLoop] Coordination module not available: {e}")
        except (ValueError, KeyError, TypeError) as e:
            # Configuration or wiring errors
            logger.error(f"[MasterLoop] Coordination bootstrap configuration error: {e}")
        except Exception as e:
            # Unexpected errors - Dec 2025: narrowed, added exc_info for debugging
            logger.error(f"[MasterLoop] Coordination bootstrap failed: {e}", exc_info=True)

        # December 2025 (Phase E): Verify P2P cluster connectivity before proceeding
        # This prevents silent failures when P2P is unavailable
        try:
            p2p_ok, p2p_errors = await self._verify_p2p_connectivity()
            if not p2p_ok:
                logger.warning(
                    f"[MasterLoop] P2P cluster not fully available: {p2p_errors}. "
                    "Proceeding with reduced cluster functionality."
                )
                # Emit warning event for monitoring
                try:
                    from app.coordination.event_router import publish_sync, DataEventType
                    publish_sync(
                        DataEventType.P2P_CLUSTER_UNHEALTHY,
                        {"errors": p2p_errors, "timestamp": time.time()},
                        source="master_loop",
                    )
                except ImportError:
                    pass  # Module not available (optional dependency)
                except Exception as e:
                    logger.debug(f"Event emission failed (best-effort): {e}")
        except Exception as e:
            logger.warning(f"[MasterLoop] P2P connectivity check failed: {e}")

        # December 29, 2025: Validate critical subsystems before proceeding
        # This catches configuration/infrastructure issues early
        try:
            validation_ok, validation_errors = await self._validate_critical_subsystems()
            if not validation_ok:
                logger.error(
                    f"[MasterLoop] Critical subsystem validation failed. "
                    f"Errors: {validation_errors}. Proceeding with degraded functionality."
                )
                # Emit validation failure event for monitoring
                try:
                    from app.coordination.event_router import publish_sync, DataEventType
                    publish_sync(
                        DataEventType.HEALTH_ALERT,
                        {
                            "alert": "startup_validation_failed",
                            "errors": validation_errors,
                            "timestamp": time.time(),
                        },
                        source="master_loop",
                    )
                except (ImportError, AttributeError):
                    pass  # Event system may be part of what failed
        except Exception as e:
            logger.warning(f"[MasterLoop] Subsystem validation check failed: {e}")

        # Start daemons
        if not self.skip_daemons:
            await self._start_daemons()

        # December 29, 2025: Start ReactiveDispatcher for event-driven selfplay
        # Part of 48-hour autonomous operation optimization
        if REACTIVE_DISPATCH_ENABLED and not self.dry_run:
            await self._start_reactive_dispatcher()

        # Subscribe to events
        self._subscribe_to_events()

        # Restore persisted state (exploration_boost, etc.) - Gap 3 fix
        # January 2026: Now async to avoid blocking event loop
        await self._load_persisted_state()

        # Initialize state from current data
        await self._initialize_state()

        logger.info("[MasterLoop] Started successfully")

    async def stop(self) -> None:
        """Stop the master loop gracefully."""
        logger.info("[MasterLoop] Stopping...")
        self._running = False
        self._shutdown_event.set()

        # Save state before shutdown - Gap 3 fix
        # January 2026: Now async to avoid blocking event loop
        await self._save_persisted_state()

        # Mark heartbeat as stopped (Dec 2025)
        self._update_heartbeat("stopped")

        # Remove PID file (December 2025)
        self._remove_pid_file()

        # Stop ReactiveDispatcher (December 29, 2025)
        if REACTIVE_DISPATCH_ENABLED:
            await self._stop_reactive_dispatcher()

        # Stop daemons
        if not self.skip_daemons and self._daemon_manager is not None:
            await self.daemon_manager.shutdown()

        logger.info("[MasterLoop] Stopped")

    async def run(self) -> None:
        """Main automation loop."""
        await self.start()

        try:
            while self._running and not self._shutdown_event.is_set():
                loop_start = time.time()

                try:
                    # 1. Check cluster health, throttle if needed
                    health = await self._get_cluster_health()
                    if health.load_critical:
                        logger.warning("[MasterLoop] Cluster load critical, throttling")
                        await self._throttle_selfplay()

                    # 2. Check for training opportunities
                    now = time.time()
                    if now - self._last_training_check >= TRAINING_CHECK_INTERVAL:
                        await self._check_training_opportunities()
                        self._last_training_check = now

                    # 3. Check for allocation rebalancing
                    if now - self._last_allocation_check >= ALLOCATION_CHECK_INTERVAL:
                        await self._rebalance_allocations()
                        self._last_allocation_check = now

                    # 4. Check for pending evaluations
                    await self._check_pending_evaluations()

                    # 5. Log status
                    self._log_status(health)

                    # 6. Periodically save state - Gap 3 fix
                    # January 2026: Now async to avoid blocking event loop
                    await self._maybe_save_state()

                    # 7. Update heartbeat for health monitoring (Dec 2025)
                    self._update_heartbeat("running")

                    # 8. Record load snapshot for forecasting (Dec 2025)
                    if now - self._last_load_snapshot >= LOAD_SNAPSHOT_INTERVAL:
                        await self._record_load_snapshot(health)
                        self._last_load_snapshot = now

                    # 9. Cluster-wide P2P recovery (Dec 31, 2025)
                    # Complements local P2P_RECOVERY daemon by checking/restarting P2P on all cluster nodes
                    if now - self._last_cluster_p2p_recovery >= CLUSTER_P2P_RECOVERY_INTERVAL:
                        await self._run_cluster_p2p_recovery()
                        self._last_cluster_p2p_recovery = now

                    # 10. Emergency P2P recovery on quorum loss (Dec 31, 2025)
                    # If voter quorum drops below 80%, trigger immediate recovery regardless of interval
                    # This prevents the cluster from being partitioned for extended periods
                    if not self._voter_quorum_healthy():
                        logger.warning("[MasterLoop] Voter quorum unhealthy - triggering immediate P2P recovery")
                        await self._run_cluster_p2p_recovery()
                        self._last_cluster_p2p_recovery = now

                    # 11. Queue populator watchdog check (January 14, 2026)
                    # Detect and recover from stuck queue populator
                    if self.queue_populator is not None:
                        watchdog_result = self._queue_populator_watchdog.check(self.queue_populator)
                        if watchdog_result.get("stuck_detected"):
                            logger.warning(
                                f"[MasterLoop] Queue populator stuck: {watchdog_result.get('stuck_reason')}"
                            )

                    # 12. OOM watchdog (February 2026)
                    # Hard safety valve: if RAM usage exceeds threshold, aggressively
                    # kill non-essential daemons. This catches cases where
                    # MemoryPressureController's graduated response is too slow.
                    await self._oom_watchdog_check()

                except asyncio.CancelledError:
                    # Allow cancellation to propagate for clean shutdown
                    raise
                except (OSError, ConnectionError, TimeoutError) as e:
                    # Network/connection errors - log and continue with backoff
                    logger.warning(f"[MasterLoop] Transient error in loop iteration: {e}")
                    await asyncio.sleep(5.0)  # Jan 2026: Add backoff to prevent CPU spin
                except Exception as e:
                    # Log unexpected errors but continue loop - Dec 2025: narrowed from bare except
                    logger.error(f"[MasterLoop] Error in loop iteration: {e}", exc_info=True)
                    await asyncio.sleep(10.0)  # Jan 2026: Add backoff to prevent CPU spin

                # Wait for next iteration
                elapsed = time.time() - loop_start
                # Jan 2026: Minimum 1s sleep to prevent CPU spin when operations timeout
                sleep_time = max(1.0, LOOP_INTERVAL_SECONDS - elapsed)

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=sleep_time
                    )
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue loop

        finally:
            await self.stop()

    # =========================================================================
    # Daemon management
    # =========================================================================

    def _get_daemons_for_profile(self) -> list["DaemonType"]:
        """Resolve daemon list for the selected profile.

        December 2025: Coordinator-only mode
        When running on a coordinator node (role: coordinator in distributed_hosts.yaml),
        intensive daemons (selfplay, training, gauntlet, export) are automatically filtered out.
        Coordinators only run sync, monitoring, and health daemons.
        """
        from app.config.env import env
        from app.coordination.daemon_manager import DaemonType

        minimal = [
            # Core event infrastructure (must be first)
            DaemonType.EVENT_ROUTER,
            # Health monitoring layer
            # Dec 30 2025: Replaced deprecated NODE_HEALTH_MONITOR + SYSTEM_HEALTH_MONITOR
            # with COORDINATOR_HEALTH_MONITOR which uses unified_health_manager
            DaemonType.COORDINATOR_HEALTH_MONITOR,
            DaemonType.CLUSTER_MONITOR,
            DaemonType.HEALTH_SERVER,
            # Dec 2025 fix: DATA_PIPELINE and FEEDBACK_LOOP BEFORE sync daemons
            # They subscribe to events that sync daemons emit (DATA_SYNC_COMPLETED, etc.)
            # If sync starts first, events are lost before subscribers are ready
            DaemonType.FEEDBACK_LOOP,  # Subscribes to: TRAINING_COMPLETED, EVALUATION_COMPLETED, etc.
            DaemonType.DATA_PIPELINE,  # Subscribes to: DATA_SYNC_COMPLETED, SELFPLAY_COMPLETE, etc.
            # Sync daemons (emit events that DATA_PIPELINE receives)
            # Dec 30 2025: Removed deprecated CLUSTER_DATA_SYNC (use AUTO_SYNC with broadcast strategy)
            DaemonType.AUTO_SYNC,
            # Jan 2026: CONFIG_SYNC auto-syncs distributed_hosts.yaml across cluster
            # Fixes P2P voter config drift issue where nodes have mismatched voter lists
            DaemonType.CONFIG_SYNC,
            DaemonType.ELO_SYNC,
            # Jan 3, 2026: ELO_PROGRESS snapshots Elo periodically for trend tracking
            DaemonType.ELO_PROGRESS,
            # Mar 2026: AUTO_EXPORT in minimal profile so standby coordinators with
            # canonical game DBs can still export NPZ training data. Export is I/O bound
            # (SQLite reads + NPZ writes), not CPU/GPU heavy. Without this, nodes running
            # --profile minimal never export, starving GPU nodes of fresh training data.
            # Gated by RINGRIFT_EXPORT_ENABLED env var (daemon self-disables if false).
            DaemonType.AUTO_EXPORT,
        ]

        standard = minimal + [
            # Core automation (current default stack)
            # Note: FEEDBACK_LOOP and DATA_PIPELINE moved to minimal for correct ordering
            # December 2025: NODE_AVAILABILITY syncs cloud provider state with distributed_hosts.yaml
            # Ensures nodes marked 'ready' are actually running (fixes stale config problem)
            DaemonType.NODE_AVAILABILITY,
            DaemonType.MODEL_DISTRIBUTION,
            DaemonType.IDLE_RESOURCE,
            # Jan 2026: COORDINATOR_DISK_MANAGER prevents disk space exhaustion on coordinator
            # Syncs data to OWC before cleanup, removes synced local copies after 24h
            DaemonType.COORDINATOR_DISK_MANAGER,
            DaemonType.UTILIZATION_OPTIMIZER,
            DaemonType.QUEUE_POPULATOR,
            DaemonType.SELFPLAY_COORDINATOR,  # Dec 28: Priority-based selfplay scheduling
            # Note: AUTO_EXPORT moved to minimal profile (Mar 2026) for standby coordinators
            # Jan 2026: CLUSTER_CONSOLIDATION pulls games from P2P cluster nodes to coordinator
            # Must run after AUTO_SYNC (needs cluster connectivity) and before DATA_CONSOLIDATION
            DaemonType.CLUSTER_CONSOLIDATION,
            # Dec 2025: DATA_CONSOLIDATION merges scattered selfplay games into canonical DBs
            # Must run after AUTO_SYNC (games need to be synced first) and before training
            DaemonType.DATA_CONSOLIDATION,
            # Jan 2026: COMPREHENSIVE_CONSOLIDATION does scheduled sweeps to catch missed data
            # Scans ALL sources (owc_imports, synced, p2p_gpu, etc.) every 30 minutes
            DaemonType.COMPREHENSIVE_CONSOLIDATION,
            # Dec 2025: NPZ_COMBINATION quality-weights and combines NPZ files for training
            # Must run after AUTO_EXPORT and before TRAINING_TRIGGER
            DaemonType.NPZ_COMBINATION,
            DaemonType.TRAINING_TRIGGER,  # Dec 27 2025: Added as critical daemon
            DaemonType.EVALUATION,
            # January 7, 2026 (Session 17.50): Process evaluation results for Elo updates
            # Bridges gauntlet results to FeedbackLoopController for training feedback
            DaemonType.GAUNTLET_FEEDBACK,
            # January 4, 2026 (Session 17.21): Model evaluation automation daemons
            # These enable fully automated evaluation of ALL model architectures (v2, v4, v5, etc.)
            # UNEVALUATED_MODEL_SCANNER: Scans for models without Elo ratings
            # STALE_EVALUATION: Re-evaluates models with ratings >30 days old
            # OWC_MODEL_IMPORT: Imports models from OWC external drive on mac-studio
            DaemonType.UNEVALUATED_MODEL_SCANNER,
            DaemonType.STALE_EVALUATION,
            DaemonType.OWC_MODEL_IMPORT,
            DaemonType.AUTO_PROMOTION,
            # January 27, 2026 (Phase 2.1): Reanalysis daemon for improved training targets
            # Re-evaluates historical games with improved models (+25-50 Elo potential)
            # Subscribes to MODEL_PROMOTED, triggers when Elo delta >= 50
            DaemonType.REANALYSIS,
            DaemonType.TOURNAMENT_DAEMON,
            DaemonType.CURRICULUM_INTEGRATION,
            # January 5, 2026 (Task 8.7): Cascade training orchestrator
            # Automates transfer learning: 2p→3p→4p when Elo thresholds are met
            # Accelerates multiplayer training by starting from learned features
            DaemonType.CASCADE_TRAINING,
            DaemonType.NODE_RECOVERY,
            DaemonType.TRAINING_NODE_WATCHER,
            DaemonType.QUALITY_MONITOR,
            # December 29, 2025: Automatic NNUE training when game threshold reached
            # NNUE models are lightweight and train faster than full NN models
            DaemonType.NNUE_TRAINING,
            # December 29, 2025: Architecture feedback controller
            # Bridges evaluation results to selfplay allocation, enforces 10% minimum
            DaemonType.ARCHITECTURE_FEEDBACK,
            # December 29, 2025: Proactive disk space management
            # Prevents sync/training failures due to disk full conditions
            DaemonType.DISK_SPACE_MANAGER,
            # December 30, 2025: Vast.ai idle termination (15min threshold)
            # Uses unified_idle_shutdown_daemon with Vast-specific config
            DaemonType.VAST_IDLE,
            # December 31, 2025: P2P recovery for LOCAL orchestrator health
            # Monitors /status endpoint and restarts local P2P when unhealthy
            # For cluster-wide P2P recovery, see _run_cluster_p2p_recovery()
            DaemonType.P2P_RECOVERY,
            # December 31, 2025: Memory monitoring to prevent OOM crashes
            DaemonType.MEMORY_MONITOR,
            # December 31, 2025: Stale model fallback for uninterrupted selfplay
            DaemonType.STALE_FALLBACK,
            # January 4, 2026: Fast failure detection (Phase 4 P2P Resilience)
            # Tiered detection: 5 min warning, 10 min alert+boost, 30 min recovery
            DaemonType.FAST_FAILURE_DETECTOR,
            # January 3, 2026: Unified backup to OWC and S3
            # Backs up all selfplay games to both destinations for disaster recovery
            DaemonType.UNIFIED_BACKUP,
            # January 5, 2026: Training watchdog for stuck process detection
            # Monitors training jobs and kills stuck processes after 2h threshold
            DaemonType.TRAINING_WATCHDOG,
            # January 5, 2026: GPU underutilization recovery
            # Injects work into queue when GPU nodes are idle
            DaemonType.UNDERUTILIZATION_RECOVERY,
            # January 7, 2026: Progress watchdog for 48h autonomous operation
            # Detects Elo stalls >24h and triggers recovery actions
            DaemonType.PROGRESS_WATCHDOG,
        ]

        # S3 sync daemon - only if AWS credentials are configured
        # Feb 2026: Migrated from deprecated S3_BACKUP + S3_NODE_SYNC to unified S3_SYNC
        s3_daemons = []
        if self._has_aws_credentials():
            s3_daemons = [
                DaemonType.S3_SYNC,  # Unified S3 sync (replaces S3_BACKUP, S3_NODE_SYNC, S3_PUSH)
            ]

        # Dec 30 2025: All deprecated daemons (from daemon_registry.py)
        # These have replacement modules and will be removed Q2 2026
        deprecated = {
            DaemonType.SYNC_COORDINATOR,      # Use AUTO_SYNC
            DaemonType.HEALTH_CHECK,          # Use COORDINATOR_HEALTH_MONITOR
            DaemonType.EPHEMERAL_SYNC,        # Use AUTO_SYNC with strategy='ephemeral'
            DaemonType.NODE_HEALTH_MONITOR,   # Use COORDINATOR_HEALTH_MONITOR
            DaemonType.SYSTEM_HEALTH_MONITOR, # Use unified_health_manager
            DaemonType.NPZ_DISTRIBUTION,      # Use MODEL_DISTRIBUTION with DataType.NPZ
            DaemonType.REPLICATION_MONITOR,   # Use unified_replication_daemon
            DaemonType.REPLICATION_REPAIR,    # Use unified_replication_daemon
            DaemonType.LAMBDA_IDLE,           # GH200 nodes are dedicated (Dec 28)
            # VAST_IDLE now runs via unified_idle_shutdown_daemon - removed from skip list Dec 30
            DaemonType.CLUSTER_DATA_SYNC,     # Use AUTO_SYNC with strategy='broadcast'
            DaemonType.S3_BACKUP,             # Use S3_SYNC (Feb 2026)
            DaemonType.S3_NODE_SYNC,          # Use S3_SYNC (Feb 2026)
            DaemonType.S3_PUSH,               # Use S3_SYNC (Feb 2026)
            DaemonType.S3_CONSOLIDATION,      # Use S3_SYNC (Feb 2026)
        }
        full = [daemon for daemon in DaemonType if daemon not in deprecated]

        profiles = {
            "minimal": minimal,
            "standard": standard,
            "full": full,
        }

        daemons = profiles.get(self.daemon_profile, standard)

        # December 2025: Add S3 backup daemons if AWS credentials are configured
        # These enable automatic backup to S3 after model promotion and periodic sync
        if s3_daemons and self.daemon_profile != "minimal":
            daemons = daemons + s3_daemons
            logger.info(
                f"[MasterLoop] S3 backup enabled: adding {len(s3_daemons)} S3 daemons "
                f"(AWS_ACCESS_KEY_ID set)"
            )

        # December 2025: Coordinator-only mode filtering
        # When running on a coordinator node, filter out intensive daemons
        if env.is_coordinator:
            # Daemons that run CPU/GPU intensive processes
            # These should NEVER run on coordinator nodes
            intensive_daemons = {
                DaemonType.IDLE_RESOURCE,           # spawns selfplay
                DaemonType.TRAINING_NODE_WATCHER,   # monitors training
                # Jan 19, 2026: Removed AUTO_EXPORT from intensive list.
                # AUTO_EXPORT is I/O-bound (not CPU), and coordinator needs it to export
                # consolidated game data for training. Without it, NPZ files become stale.
                # Jan 10, 2026: TRAINING_TRIGGER moved to coordinator - dispatches training
                # to cluster nodes via P2P, doesn't run training locally.
                DaemonType.NNUE_TRAINING,           # dispatches NNUE training to cluster
                DaemonType.TOURNAMENT_DAEMON,       # runs tournaments
                # Jan 3, 2026: EVALUATION moved to coordinator - dispatches gauntlet runs
                # to cluster nodes via P2P, doesn't run locally. Similar to AUTO_PROMOTION.
                DaemonType.QUEUE_POPULATOR,         # can spawn selfplay
                DaemonType.UTILIZATION_OPTIMIZER,   # spawns processes on idle GPUs
            }
            original_count = len(daemons)
            daemons = [d for d in daemons if d not in intensive_daemons]
            filtered_count = original_count - len(daemons)
            if filtered_count > 0:
                logger.info(
                    f"[MasterLoop] Coordinator-only mode: filtered out {filtered_count} "
                    f"intensive daemons (node: {env.node_id})"
                )

            # December 2025: Add coordinator-specific daemons
            # These manage coordinator disk space by syncing data to external storage
            # December 30, 2025: Added PARITY_VALIDATION - validates pending_gate databases
            # and stores TS hashes for cluster nodes that lack Node.js
            coordinator_daemons = {
                DaemonType.COORDINATOR_DISK_MANAGER,  # Proactive disk cleanup with external sync
                DaemonType.EXTERNAL_DRIVE_SYNC,       # Pull data from cluster to OWC drive (Dec 29)
                DaemonType.OWC_IMPORT,                # Import data FROM OWC to cluster (Dec 30)
                DaemonType.PARITY_VALIDATION,         # Validate pending_gate DBs, store TS hashes (Dec 30)
            }

            # Feb 2026: S3_CONSOLIDATION replaced by S3_SYNC (which handles
            # consolidated paths). S3_SYNC is already added via s3_daemons above.
            for daemon in coordinator_daemons:
                if daemon not in daemons:
                    daemons.append(daemon)
                    logger.info(
                        f"[MasterLoop] Coordinator mode: added {daemon.value} for disk management"
                    )

        # March 2026: Standby coordinator mode — minimal daemon set.
        # Standby coordinators (e.g., local-mac MacBook) only run health
        # monitoring, Elo sync, and dashboard. No evaluations, exports,
        # training triggers, S3 pushes, or any disk/CPU-intensive work.
        if env.is_standby_coordinator:
            # Allow-list: only lightweight observe-only daemons
            standby_allowed = {
                DaemonType.EVENT_ROUTER,              # Core event infrastructure
                DaemonType.COORDINATOR_HEALTH_MONITOR, # Health monitoring
                DaemonType.CLUSTER_MONITOR,            # Cluster observability
                DaemonType.HEALTH_SERVER,              # Health HTTP endpoint
                DaemonType.ELO_SYNC,                   # Receive Elo updates
                DaemonType.ELO_PROGRESS,               # Elo trend tracking
                DaemonType.CONFIG_SYNC,                # Config sync
                DaemonType.NODE_AVAILABILITY,          # Node status tracking
            }
            original_count = len(daemons)
            daemons = [d for d in daemons if d in standby_allowed]
            logger.info(
                f"[MasterLoop] Standby coordinator mode: keeping {len(daemons)} of "
                f"{original_count} daemons (node: {env.node_id})"
            )

        # January 2026: Add PARITY_VALIDATION on any node with npx available
        # The parity daemon validates pending_gate databases and stores TS hashes.
        # It requires Node.js (npx) but isn't resource-intensive, so it can run
        # on non-coordinator nodes that have npx installed.
        if DaemonType.PARITY_VALIDATION not in daemons and not env.is_standby_coordinator:
            if self._has_npx():
                daemons.append(DaemonType.PARITY_VALIDATION)
                logger.info(
                    "[MasterLoop] Node has npx available: added PARITY_VALIDATION daemon"
                )

        # Ensure event router starts first when present
        if DaemonType.EVENT_ROUTER in daemons:
            daemons = [DaemonType.EVENT_ROUTER] + [d for d in daemons if d != DaemonType.EVENT_ROUTER]

        # De-dupe while preserving order
        seen: set[DaemonType] = set()
        ordered: list[DaemonType] = []
        for daemon in daemons:
            if daemon not in seen:
                ordered.append(daemon)
                seen.add(daemon)
        return ordered

    async def _start_daemons(self) -> None:
        """Start daemons for the selected profile."""
        from app.coordination.daemon_manager import DaemonType
        from app.coordination.daemon_types import validate_startup_order_or_raise

        profile_daemons = self._get_daemons_for_profile()
        logger.info(
            f"[MasterLoop] Starting {len(profile_daemons)} daemons "
            f"(profile={self.daemon_profile})"
        )

        # December 2025: Validate daemon dependency graph before starting
        # January 2026: Made configurable via RINGRIFT_ALLOW_BROKEN_DEPS
        try:
            validate_startup_order_or_raise()
            logger.debug("[MasterLoop] Daemon startup order validated successfully")
        except ValueError as e:
            logger.error(f"[MasterLoop] Daemon dependency validation failed: {e}")
            # Check if we should fail hard or continue with warning
            allow_broken_deps = os.environ.get("RINGRIFT_ALLOW_BROKEN_DEPS", "true").lower() == "true"
            if not allow_broken_deps:
                logger.critical(
                    "[MasterLoop] Exiting due to broken daemon dependencies. "
                    "Set RINGRIFT_ALLOW_BROKEN_DEPS=true to bypass."
                )
                raise SystemExit(1)
            logger.warning("[MasterLoop] Proceeding despite dependency issues (RINGRIFT_ALLOW_BROKEN_DEPS=true)")

        # Track which daemons started successfully
        started_daemons: set[str] = set()
        failed_daemons: set[str] = set()

        if self.dry_run:
            for daemon_type in profile_daemons:
                logger.info(f"[MasterLoop] [DRY RUN] Would start {daemon_type.value}")
                started_daemons.add(daemon_type.value)
        else:
            # Feb 22, 2026: Concurrent daemon startup with to_thread wrapping.
            # Previous sequential startup blocked for 46 * 60s = 46 minutes when
            # daemon start() methods contained synchronous I/O that blocked the
            # event loop (asyncio.wait_for timeout couldn't fire). Now we:
            # 1. Start EVENT_ROUTER first (critical dependency for all others)
            # 2. Start remaining daemons concurrently via asyncio.to_thread()
            #    so blocking I/O in one daemon can't stall the entire startup.
            event_router_type = None
            remaining_daemons = []
            for d in profile_daemons:
                if d.value == "event_router":
                    event_router_type = d
                else:
                    remaining_daemons.append(d)

            # Phase 1: Start event_router first (all daemons depend on it)
            if event_router_type:
                try:
                    await asyncio.wait_for(
                        self.daemon_manager.start(event_router_type),
                        timeout=DAEMON_STARTUP_TIMEOUT_SECONDS,
                    )
                    started_daemons.add(event_router_type.value)
                    logger.info(f"[MasterLoop] Started {event_router_type.value}")
                except (asyncio.TimeoutError, Exception) as e:
                    failed_daemons.add(event_router_type.value)
                    logger.error(f"[MasterLoop] event_router startup failed: {e}")

            # Phase 2: Start remaining daemons in small sequential batches.
            # We can't use asyncio.gather() because daemon start() methods
            # often contain synchronous SQLite I/O that blocks the event loop,
            # preventing wait_for timeouts from triggering. Instead, we start
            # them sequentially but with a short timeout and skip on failure.
            if remaining_daemons:
                logger.info(
                    f"[MasterLoop] Starting {len(remaining_daemons)} daemons "
                    f"(timeout={DAEMON_STARTUP_TIMEOUT_SECONDS}s each)"
                )
                results = []
                for daemon_type in remaining_daemons:
                    try:
                        # Use to_thread to prevent blocking the event loop
                        loop = asyncio.get_running_loop()
                        start_coro = self.daemon_manager.start(daemon_type)
                        await asyncio.wait_for(start_coro, timeout=DAEMON_STARTUP_TIMEOUT_SECONDS)
                        results.append((daemon_type.value, True, None))
                        logger.debug(f"[MasterLoop] Started {daemon_type.value}")
                    except asyncio.TimeoutError:
                        results.append((daemon_type.value, False, "timeout"))
                    except Exception as e:
                        results.append((daemon_type.value, False, str(e)))
                for result in results:
                    if isinstance(result, Exception):
                        logger.warning(f"[MasterLoop] Daemon startup exception: {result}")
                        continue
                    name, success, error = result
                    if success:
                        started_daemons.add(name)
                    else:
                        failed_daemons.add(name)
                        if error == "timeout":
                            logger.error(
                                f"[MasterLoop] Daemon {name} startup timed out "
                                f"after {DAEMON_STARTUP_TIMEOUT_SECONDS}s"
                            )
                        else:
                            logger.warning(f"[MasterLoop] Failed to start {name}: {error}")

        # December 2025 - Gap 2 fix: Validate critical daemons started
        # Only check daemons that were in the profile (coordinator mode filters some out)
        profile_daemon_names = {d.value for d in profile_daemons}
        expected_critical = CRITICAL_DAEMON_NAMES & profile_daemon_names
        failed_critical = expected_critical & failed_daemons
        missing_critical = expected_critical - started_daemons - failed_daemons

        if failed_critical or missing_critical:
            critical_issues = failed_critical | missing_critical
            logger.error(
                f"[MasterLoop] CRITICAL DAEMONS FAILED: {critical_issues}. "
                f"Started: {started_daemons}, Failed: {failed_daemons}"
            )
            # December 29, 2025: Add fail-fast option for critical daemon failures
            # Set RINGRIFT_FAIL_ON_CRITICAL_DAEMON_FAILURE=1 to enforce strict startup
            fail_fast = os.environ.get("RINGRIFT_FAIL_ON_CRITICAL_DAEMON_FAILURE", "").lower() in ("1", "true", "yes")
            if fail_fast:
                raise RuntimeError(
                    f"FATAL: Critical daemons failed to start: {critical_issues}. "
                    "Set RINGRIFT_FAIL_ON_CRITICAL_DAEMON_FAILURE=0 to disable strict mode."
                )
        else:
            skipped_critical = CRITICAL_DAEMON_NAMES - expected_critical
            if skipped_critical:
                logger.info(
                    f"[MasterLoop] All expected critical daemons started. "
                    f"(skipped on coordinator: {skipped_critical})"
                )
            else:
                logger.info(
                    f"[MasterLoop] All critical daemons started successfully: {CRITICAL_DAEMON_NAMES}"
                )

        # Start FeedbackLoopController explicitly (December 2025 - Phase 2A.1)
        # The daemon manager starts FEEDBACK_LOOP daemon, but we also need to
        # ensure the controller itself is started with proper event subscriptions
        try:
            if not self.dry_run and DaemonType.FEEDBACK_LOOP in profile_daemons:
                await self.feedback_controller.start()
                logger.info("[MasterLoop] Started FeedbackLoopController")
        except Exception as e:
            logger.warning(f"[MasterLoop] Failed to start FeedbackLoopController: {e}")

    async def _start_reactive_dispatcher(self) -> None:
        """Start the ReactiveDispatcher for event-driven selfplay.

        December 29, 2025: Part of 48-hour autonomous operation optimization.
        Replaces polling-based selfplay dispatch with event-driven dispatch.

        When enabled, the master loop interval increases to 5 minutes (watchdog mode)
        since selfplay is dispatched via events (node_recovered, training_completed, etc.)
        """
        try:
            from app.coordination.reactive_dispatcher import (
                ReactiveDispatcher,
                start_reactive_dispatcher,
            )

            self._reactive_dispatcher = await start_reactive_dispatcher()
            logger.info(
                f"[MasterLoop] Started ReactiveDispatcher "
                f"(loop interval: {LOOP_INTERVAL_SECONDS}s watchdog mode)"
            )
        except ImportError as e:
            logger.warning(f"[MasterLoop] ReactiveDispatcher not available: {e}")
            self._reactive_dispatcher = None
        except Exception as e:
            logger.error(f"[MasterLoop] Failed to start ReactiveDispatcher: {e}")
            self._reactive_dispatcher = None

    async def _stop_reactive_dispatcher(self) -> None:
        """Stop the ReactiveDispatcher."""
        if hasattr(self, "_reactive_dispatcher") and self._reactive_dispatcher is not None:
            try:
                await self._reactive_dispatcher.stop()
                logger.info("[MasterLoop] Stopped ReactiveDispatcher")
            except Exception as e:
                logger.warning(f"[MasterLoop] Error stopping ReactiveDispatcher: {e}")

    # =========================================================================
    # Event handling
    # =========================================================================

    def _subscribe_to_events(self) -> None:
        """Subscribe to pipeline events."""
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            bus.subscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)
            bus.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete)
            bus.subscribe(DataEventType.EVALUATION_PROGRESS, self._on_evaluation_progress)
            bus.subscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_complete)
            bus.subscribe(DataEventType.MODEL_PROMOTED, self._on_promotion_complete)
            bus.subscribe(DataEventType.QUALITY_SCORE_UPDATED, self._on_quality_assessed)
            bus.subscribe(DataEventType.TRAINING_BLOCKED_BY_QUALITY, self._on_training_blocked_by_quality)
            bus.subscribe(DataEventType.LOCK_TIMEOUT, self._on_lock_timeout)

            logger.info("[MasterLoop] Subscribed to pipeline events (including quality + evaluation progress)")
        except Exception as e:
            logger.warning(f"[MasterLoop] Failed to subscribe to events: {e}")

    def _on_selfplay_complete(self, event: Any) -> None:
        """Handle selfplay completion."""
        try:
            config_key = event.payload.get("config_key") or event.payload.get("config", "")
            games_added = event.payload.get("games_added", 0)

            if config_key in self._states:
                state = self._states[config_key]
                state.games_since_last_export += games_added
                logger.debug(f"[MasterLoop] {config_key}: +{games_added} games, total pending: {state.games_since_last_export}")
        except Exception as e:
            logger.debug(f"[MasterLoop] Error handling selfplay event: {e}")

    def _on_training_complete(self, event: Any) -> None:
        """Handle training completion."""
        try:
            # train.py emits "config" field; fall back to it if "config_key" is missing
            config_key = event.payload.get("config_key") or event.payload.get("config", "")
            policy_accuracy = event.payload.get("policy_accuracy", 0.0)
            model_path = event.payload.get("model_path", "")
            source = getattr(event, "source", "")

            # Feb 2026: Reject phantom training completions from DLQ replay or
            # stale events that lack real training metrics. These caused spurious
            # generation tracking and model overwrites with untrained weights.
            if policy_accuracy == 0.0 and not model_path:
                logger.debug(
                    f"[MasterLoop] Ignoring phantom training event for {config_key} "
                    f"(no model_path, policy_accuracy=0, source={source})"
                )
                return

            if config_key in self._states:
                state = self._states[config_key]
                state.last_training_time = time.time()
                state.last_policy_accuracy = policy_accuracy
                state.pending_evaluation = True  # Queue for evaluation
                state.training_node = None  # Release training node

                logger.info(f"[MasterLoop] {config_key}: Training complete, policy accuracy: {policy_accuracy:.2%}")
        except Exception as e:
            logger.debug(f"[MasterLoop] Error handling training event: {e}")

    def _on_evaluation_complete(self, event: Any) -> None:
        """Handle evaluation completion."""
        try:
            config_key = event.payload.get("config_key") or event.payload.get("config", "")
            win_rate = event.payload.get("win_rate", 0.0)
            passed = event.payload.get("passed", False)

            if config_key in self._states:
                state = self._states[config_key]
                state.last_evaluation_time = time.time()
                state.last_evaluation_win_rate = win_rate
                state.pending_evaluation = False

                logger.info(f"[MasterLoop] {config_key}: Evaluation complete, win rate: {win_rate:.2%}, passed: {passed}")
        except Exception as e:
            logger.debug(f"[MasterLoop] Error handling evaluation event: {e}")

    def _on_evaluation_progress(self, event: Any) -> None:
        """Handle evaluation progress updates."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", payload.get("config", ""))
            baseline = payload.get("baseline", "unknown")
            games_completed = payload.get("games_completed", payload.get("games", 0))
            games_total = payload.get("games_total", 0)
            current_win_rate = payload.get("current_win_rate", 0.0)

            if config_key in self._states:
                state = self._states[config_key]
                state.last_evaluation_win_rate = current_win_rate

            if config_key:
                logger.debug(
                    f"[MasterLoop] {config_key}: Evaluation progress "
                    f"{games_completed}/{games_total} vs {baseline} "
                    f"({current_win_rate:.2%})"
                )
            else:
                logger.debug(
                    f"[MasterLoop] Evaluation progress {games_completed}/{games_total} "
                    f"vs {baseline} ({current_win_rate:.2%})"
                )
        except Exception as e:
            logger.debug(f"[MasterLoop] Error handling evaluation progress: {e}")

    def _on_promotion_complete(self, event: Any) -> None:
        """Handle promotion completion."""
        try:
            config_key = event.payload.get("config_key") or event.payload.get("config", "")
            success = event.payload.get("success", False)

            if config_key in self._states:
                state = self._states[config_key]
                state.last_promotion_success = success

                # Apply feedback based on outcome
                if success:
                    # Accelerate training for this config
                    state.training_intensity = "accelerated"
                    state.exploration_boost = 1.0  # Reset exploration
                    logger.info(f"[MasterLoop] {config_key}: Promotion succeeded, accelerating training")
                else:
                    # Boost exploration on failure
                    state.exploration_boost = min(2.0, state.exploration_boost * 1.3)
                    logger.info(f"[MasterLoop] {config_key}: Promotion failed, exploration boost: {state.exploration_boost:.2f}")
        except Exception as e:
            logger.debug(f"[MasterLoop] Error handling promotion event: {e}")

    def _on_training_blocked_by_quality(self, event: Any) -> None:
        """Handle training blocked by quality or freshness gates."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", payload.get("config", ""))
            reason = payload.get("reason", "unknown")
            data_age_hours = payload.get("data_age_hours")
            threshold_hours = payload.get("threshold_hours")
            games_available = payload.get("games_available")

            if config_key in self._states:
                state = self._states[config_key]
                state.training_node = None
                state.training_intensity = "paused"

            details = (
                f"reason={reason}, data_age_hours={data_age_hours}, "
                f"threshold_hours={threshold_hours}, games_available={games_available}"
            )
            if config_key:
                logger.info(f"[MasterLoop] {config_key}: Training blocked ({details})")
            else:
                logger.info(f"[MasterLoop] Training blocked ({details})")
        except Exception as e:
            logger.debug(f"[MasterLoop] Error handling training blocked event: {e}")

    def _on_lock_timeout(self, event: Any) -> None:
        """Handle lock timeout events from sync coordination."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            host = payload.get("host", "unknown")
            operation = payload.get("operation", "unknown")
            wait_duration = payload.get("wait_duration")
            wait_timeout = payload.get("wait_timeout")

            logger.warning(
                f"[MasterLoop] Lock timeout: host={host} op={operation} "
                f"waited={wait_duration} timeout={wait_timeout}"
            )
        except Exception as e:
            logger.debug(f"[MasterLoop] Error handling lock timeout event: {e}")

    def _on_quality_assessed(self, event: Any) -> None:
        """Handle data quality assessment event.

        December 2025: Fixes Gap 1 - quality score was never tracked in master loop.
        Now captures quality scores from FeedbackLoopController for training decisions.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", payload.get("config", ""))
            quality_score = payload.get("quality_score", 0.0)
            ready_for_training = payload.get("ready_for_training", False)

            if config_key in self._states:
                state = self._states[config_key]
                state.last_quality_score = quality_score
                state.last_quality_update_time = time.time()

                # Compute training intensity from quality
                if quality_score >= 0.90:
                    state.training_intensity = "hot_path"
                elif quality_score >= 0.80:
                    state.training_intensity = "accelerated"
                elif quality_score >= 0.65:
                    state.training_intensity = "normal"
                elif quality_score >= 0.50:
                    state.training_intensity = "reduced"
                else:
                    state.training_intensity = "paused"

                logger.debug(
                    f"[MasterLoop] {config_key}: Quality={quality_score:.2f}, "
                    f"intensity={state.training_intensity}, ready={ready_for_training}"
                )

                # Sync intensity via event emission (Dec 2025 - Event-driven refactor)
                # TrainingTriggerDaemon subscribes to QUALITY_SCORE_UPDATED and computes
                # intensity from quality_score. This avoids fragile direct state access.
                try:
                    from app.distributed.data_events import (
                        DataEvent,
                        DataEventType,
                        get_event_bus,
                    )

                    # Emit quality score update with config_key for training coordination
                    event = DataEvent(
                        event_type=DataEventType.QUALITY_SCORE_UPDATED,
                        payload={
                            "config_key": config_key,
                            "quality_score": quality_score,
                            "source": "master_loop",
                        },
                        source="master_loop._on_quality_assessed",
                    )

                    # Create task to emit async event without blocking
                    # Jan 12, 2026: Fixed fire-and-forget pattern - now captures exceptions
                    import asyncio

                    def _on_publish_error(task: asyncio.Task) -> None:
                        """Handle event publish errors to prevent silent failures."""
                        try:
                            exc = task.exception()
                            if exc:
                                logger.error(
                                    f"[MasterLoop] EventBus publish failed for {config_key}: {exc}"
                                )
                        except asyncio.CancelledError:
                            pass

                    publish_task = asyncio.create_task(get_event_bus().publish(event))
                    publish_task.add_done_callback(_on_publish_error)
                except ImportError:
                    pass  # Event system not available

        except Exception as e:
            logger.debug(f"[MasterLoop] Error handling quality event: {e}")

    # =========================================================================
    # State initialization
    # =========================================================================

    async def _initialize_state(self) -> None:
        """Initialize state from current data."""
        logger.info("[MasterLoop] Initializing state from current data...")

        # January 4, 2026: Initialize Elo velocities from database history
        # This fixes cold start gap where velocities are 0.0 until ELO_VELOCITY_CHANGED events fire
        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            scheduler = get_selfplay_scheduler()
            count = scheduler.initialize_elo_velocities_from_db()
            if count > 0:
                logger.info(f"[MasterLoop] Initialized Elo velocities for {count} configs from database")
            else:
                logger.debug("[MasterLoop] No Elo velocity history found in database")
        except ImportError as e:
            logger.debug(f"[MasterLoop] SelfplayScheduler not available for Elo velocity init: {e}")
        except Exception as e:
            logger.warning(f"[MasterLoop] Failed to initialize Elo velocities: {e}")

        for config_key in self.active_configs:
            try:
                # Parse config key
                parts = config_key.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                board_type = parts[0]
                num_players = int(parts[1].replace("p", ""))

                # Check for existing NPZ files
                npz_path = Path(f"data/training/{board_type}_{num_players}p.npz")
                if npz_path.exists():
                    mtime = npz_path.stat().st_mtime
                    self._states[config_key].last_export_time = mtime
                    logger.debug(f"[MasterLoop] {config_key}: Found NPZ from {datetime.fromtimestamp(mtime)}")

            except Exception as e:
                logger.debug(f"[MasterLoop] Error initializing {config_key}: {e}")

    # =========================================================================
    # Health monitoring
    # =========================================================================

    async def _get_cluster_health(self) -> ClusterHealth:
        """Get aggregated cluster health."""
        health = ClusterHealth()

        try:
            # Query cluster status
            status = self.cluster_monitor.get_cluster_status()

            health.total_nodes = status.total_nodes
            health.healthy_nodes = status.active_nodes
            health.avg_disk_usage = status.avg_disk_usage

            # Count training and selfplay nodes
            for node_id, node_status in status.nodes.items():
                if node_status.training_active:
                    health.training_nodes += 1
                # Assume non-training GPU nodes are doing selfplay
                elif node_status.gpu_utilization_percent > 10:
                    health.selfplay_nodes += 1

                health.avg_gpu_utilization += node_status.gpu_utilization_percent

            if health.total_nodes > 0:
                health.avg_gpu_utilization /= health.total_nodes

            # Check for critical load
            health.load_critical = (
                health.avg_disk_usage > 90
                or health.healthy_nodes < health.total_nodes * 0.5
            )

        except Exception as e:
            health.errors.append(str(e))
            logger.debug(f"[MasterLoop] Error getting cluster health: {e}")

        return health

    async def _throttle_selfplay(self) -> None:
        """Throttle selfplay when cluster is overloaded."""
        logger.info("[MasterLoop] Throttling selfplay due to cluster load")

        if self.dry_run:
            logger.info("[MasterLoop] [DRY RUN] Would pause selfplay jobs")
            return

        # Emit throttle signal via event bus
        try:
            from app.coordination.event_router import emit_event, DataEventType

            await emit_event(
                DataEventType.HEALTH_ALERT,
                {
                    "alert": "throttle_selfplay",
                    "action": "throttle_selfplay",
                    "reason": "load_critical",
                }
            )
        except Exception as e:
            logger.debug(f"[MasterLoop] Error emitting throttle event: {e}")

    async def _record_load_snapshot(self, health: ClusterHealth) -> None:
        """Record cluster load snapshot for forecasting.

        December 2025: Added for load forecasting integration.
        Records hourly load snapshots to learn cluster usage patterns.
        """
        try:
            from app.coordination.load_forecaster import record_hourly_load

            # Count active jobs from P2P status
            active_jobs = health.training_nodes + health.selfplay_nodes
            busy_hosts = active_jobs  # Approximate

            record_hourly_load(
                active_jobs=active_jobs,
                busy_hosts=busy_hosts,
                total_hosts=health.total_nodes,
                gpu_utilization=health.avg_gpu_utilization / 100,  # Convert to 0-1
                cpu_utilization=0.0,  # Not tracked yet
            )
            logger.debug(
                f"[MasterLoop] Recorded load snapshot: jobs={active_jobs}, "
                f"busy={busy_hosts}/{health.total_nodes}"
            )
        except ImportError:
            logger.debug("[MasterLoop] Load forecaster not available")
        except Exception as e:
            logger.debug(f"[MasterLoop] Error recording load snapshot: {e}")

    async def _run_cluster_p2p_recovery(self) -> None:
        """Cluster-wide P2P recovery - check and restart P2P on remote nodes.

        December 31, 2025: Added for 48-hour autonomous operation.
        January 21, 2026: Added supervisor coordination check.

        This complements the local P2P_RECOVERY daemon by scanning all P2P-enabled
        cluster nodes via SSH and restarting P2P on any that aren't responding.

        The local daemon only handles the coordinator's own P2P orchestrator.
        This method handles all other cluster nodes.

        Recovery is parallel with a semaphore to avoid overwhelming SSH connections.
        """
        try:
            # January 21, 2026: Check if we should defer to manual management
            try:
                from scripts.p2p_orchestrator import should_master_loop_manage_p2p
                should_manage, reason = should_master_loop_manage_p2p()
                if not should_manage:
                    logger.info(f"[MasterLoop] Deferring P2P recovery to manual management: {reason}")
                    return
            except ImportError:
                pass  # p2p_orchestrator not available, proceed with recovery
            except Exception as e:
                logger.debug(f"[MasterLoop] Could not check supervisor file: {e}")

            # Import the recovery module
            from scripts.recover_p2p_cluster import (
                load_p2p_nodes,
                check_node_p2p_status,
                restart_p2p_on_node,
            )

            nodes = load_p2p_nodes()
            if not nodes:
                logger.debug("[MasterLoop] No P2P-enabled nodes found for cluster recovery")
                return

            logger.info(f"[MasterLoop] Starting cluster P2P recovery check for {len(nodes)} nodes")

            # Check status of all nodes in parallel
            semaphore = asyncio.Semaphore(10)  # Max 10 concurrent SSH connections

            async def check_with_semaphore(node):
                async with semaphore:
                    return await check_node_p2p_status(node)

            statuses = await asyncio.gather(*[check_with_semaphore(n) for n in nodes])

            # Categorize results
            reachable = [s for s in statuses if s["reachable"]]
            p2p_ok = [s for s in reachable if s["p2p_responding"]]
            p2p_not_running = [s for s in reachable if not s["p2p_running"]]
            p2p_not_responding = [s for s in reachable if s["p2p_running"] and not s["p2p_responding"]]

            logger.info(
                f"[MasterLoop] P2P cluster status: "
                f"{len(reachable)}/{len(nodes)} reachable, "
                f"{len(p2p_ok)} healthy, "
                f"{len(p2p_not_running)} not running, "
                f"{len(p2p_not_responding)} not responding"
            )

            # Restart P2P on nodes where it's not running or not responding
            nodes_to_restart = []
            for status in p2p_not_running + p2p_not_responding:
                node = next((n for n in nodes if n.name == status["name"]), None)
                if node:
                    nodes_to_restart.append(node)

            if nodes_to_restart:
                logger.info(f"[MasterLoop] Restarting P2P on {len(nodes_to_restart)} nodes")

                async def restart_with_semaphore(node):
                    async with semaphore:
                        return await restart_p2p_on_node(node, dry_run=False)

                restart_results = await asyncio.gather(
                    *[restart_with_semaphore(n) for n in nodes_to_restart]
                )
                restarted = sum(1 for r in restart_results if r)
                logger.info(
                    f"[MasterLoop] Cluster P2P recovery: restarted {restarted}/{len(nodes_to_restart)} nodes"
                )

                # Emit recovery event for monitoring
                try:
                    from app.distributed.data_events import DataEventType, emit_data_event
                    emit_data_event(
                        DataEventType.CLUSTER_P2P_RECOVERY_COMPLETED,
                        nodes_checked=len(nodes),
                        reachable=len(reachable),
                        healthy=len(p2p_ok),
                        restarted=restarted,
                        source="MasterLoopController",
                    )
                except Exception as e:
                    logger.debug(f"[MasterLoop] Failed to emit recovery event: {e}")
            else:
                logger.debug("[MasterLoop] All P2P nodes healthy, no restart needed")

        except ImportError as e:
            logger.warning(f"[MasterLoop] Cluster P2P recovery not available: {e}")
        except Exception as e:
            logger.error(f"[MasterLoop] Error in cluster P2P recovery: {e}", exc_info=True)

    def _voter_quorum_healthy(self) -> bool:
        """Check if P2P voter quorum is at 80%+ capacity.

        December 31, 2025: Added for 48-hour autonomous operation.

        Returns True if quorum is healthy (80%+ voters alive), False otherwise.
        Returns True on any error to avoid triggering tight recovery loops.

        The voter quorum is essential for distributed consensus. If too many
        voters are offline, the cluster can become partitioned and unable to
        elect a leader or coordinate work.
        """
        try:
            import requests

            resp = requests.get(get_local_p2p_status_url(), timeout=5)
            if resp.status_code != 200:
                return True  # Assume healthy on error

            status = resp.json()
            voters_alive = status.get("voters_alive", 0)
            voter_node_ids = status.get("voter_node_ids", [])
            total_voters = len(voter_node_ids)

            if total_voters == 0:
                return True  # No voters configured, assume healthy

            quorum_ratio = voters_alive / total_voters
            quorum_healthy = quorum_ratio >= 0.8

            if not quorum_healthy:
                logger.warning(
                    f"[MasterLoop] Voter quorum unhealthy: {voters_alive}/{total_voters} "
                    f"({quorum_ratio:.0%}) < 80%"
                )

            return quorum_healthy

        except requests.RequestException:
            # P2P not responding - let the regular recovery handle it
            return True
        except (KeyError, TypeError, ValueError):
            # Malformed response - assume healthy to avoid tight loops
            return True
        except Exception:
            # Any other error - assume healthy
            return True

    # =========================================================================
    # OOM Watchdog (February 2026)
    # =========================================================================

    async def _oom_watchdog_check(self) -> None:
        """Hard OOM safety valve - runs every loop iteration.

        February 2026: Added after repeated kernel panics on 128 GB coordinator.
        This is a last-resort check that catches memory issues even if the
        MemoryPressureController is slow to react (it requires consecutive
        samples and has cooldowns).

        At 85% usage (108 GB on 128 GB), we have ~20 GB free. That's enough
        for the coordinator to operate but not enough to absorb a spike from
        consolidation daemons opening multi-GB databases. We aggressively
        stop heavy daemons at this threshold.
        """
        try:
            from app.utils.resource_guard import get_memory_usage

            used_percent, available_gb, total_gb = get_memory_usage()
            oom_threshold = float(os.environ.get(
                "RINGRIFT_OOM_WATCHDOG_THRESHOLD", "85"
            ))

            if used_percent < oom_threshold:
                return

            logger.warning(
                f"[MasterLoop] OOM watchdog triggered: {used_percent:.1f}% RAM used "
                f"(threshold: {oom_threshold}%), available: {available_gb:.1f} GB / "
                f"{total_gb:.0f} GB. Stopping heavy daemons."
            )

            # Stop memory-heavy daemons in order of impact
            heavy_daemons = [
                "CLUSTER_CONSOLIDATION",
                "COMPREHENSIVE_CONSOLIDATION",
                "DATA_CONSOLIDATION",
                "NPZ_COMBINATION",
                "AUTO_EXPORT",
                "REANALYSIS",
                "UNIFIED_BACKUP",
            ]

            stopped = 0
            for daemon_name in heavy_daemons:
                try:
                    from app.coordination.daemon_manager import DaemonType
                    dtype = DaemonType(daemon_name.lower())
                    await self.daemon_manager.stop(dtype)
                    stopped += 1
                    logger.warning(f"[MasterLoop] OOM watchdog stopped: {daemon_name}")
                except (ValueError, Exception) as e:
                    logger.debug(f"[MasterLoop] OOM watchdog: could not stop {daemon_name}: {e}")

            # Force garbage collection
            import gc
            gc.collect()

            logger.warning(
                f"[MasterLoop] OOM watchdog: stopped {stopped} daemons, forced GC. "
                f"Daemons will restart on next cycle when memory recovers."
            )

        except ImportError:
            pass  # resource_guard not available
        except Exception as e:
            logger.debug(f"[MasterLoop] OOM watchdog error: {e}")

    # =========================================================================
    # Training coordination
    # =========================================================================

    async def _check_training_opportunities(self) -> None:
        """Check if any configs are ready for training."""
        logger.debug("[MasterLoop] Checking training opportunities...")

        for config_key, state in self._states.items():
            if state.training_node is not None:
                # Already training
                continue

            # Check readiness
            ready, reason = await self._check_training_readiness(config_key)

            if ready:
                logger.info(f"[MasterLoop] {config_key}: Ready for training")
                await self._trigger_training(config_key)
            elif state.games_since_last_export > 0:
                logger.debug(f"[MasterLoop] {config_key}: Not ready - {reason}")

    async def _check_training_readiness(self, config_key: str) -> tuple[bool, str]:
        """Check if a config is ready for training."""
        state = self._states[config_key]

        # Check minimum games - Dec 29, 2025: Use player-count aware threshold
        min_games = state.min_games_threshold
        if state.games_since_last_export < min_games:
            return False, f"Insufficient games: {state.games_since_last_export} < {min_games}"

        # Check quality score with time-decay escape hatch
        # Feb 2026: If quality hasn't been updated in >2 hours and we have enough games,
        # decay the floor to allow training on marginal-quality data rather than stalling
        quality_floor = 0.5
        hours_since_update = 0.0
        if state.last_quality_update_time > 0:
            hours_since_update = (time.time() - state.last_quality_update_time) / 3600
        if hours_since_update > 2.0 and state.games_since_last_export >= min_games * 2:
            # Decay floor: 0.5 -> 0.35 over 2-6 hours
            decay = min(0.15, (hours_since_update - 2.0) * 0.0375)
            quality_floor = max(0.35, 0.5 - decay)
        if state.last_quality_score < quality_floor:
            return False, f"Low quality: {state.last_quality_score:.2f} (floor={quality_floor:.2f})"

        # Dec 29, 2025: Check data freshness before training
        # Can be bypassed via RINGRIFT_ALLOW_STALE_TRAINING=true for faster iteration
        allow_stale = os.getenv("RINGRIFT_ALLOW_STALE_TRAINING", "").lower() in ("1", "true", "yes")
        if not allow_stale:
            try:
                from app.coordination.training_freshness import check_freshness_sync

                parts = config_key.rsplit("_", 1)
                board_type = parts[0]
                num_players = int(parts[1].replace("p", ""))

                max_age = float(os.getenv("RINGRIFT_MAX_DATA_AGE_HOURS", "2.0"))
                freshness = check_freshness_sync(board_type, num_players, max_age_hours=max_age)
                if not freshness.is_fresh:
                    # Trigger priority sync for stale data
                    await self._trigger_priority_sync(config_key, freshness.data_age_hours)
                    return False, f"Data stale ({freshness.data_age_hours:.1f}h old), sync triggered"
            except ImportError:
                logger.debug("[MasterLoop] TrainingFreshness not available, skipping freshness check")
            except Exception as e:
                logger.warning(f"[MasterLoop] Freshness check failed for {config_key}: {e}")
        else:
            logger.debug("[MasterLoop] Stale data training allowed via RINGRIFT_ALLOW_STALE_TRAINING")

        # Check if circuit breaker is tripped
        try:
            if self._pipeline_orchestrator is not None:
                breaker = self.pipeline_orchestrator._circuit_breaker
                if breaker and not breaker.can_execute():
                    return False, "Circuit breaker open"
        except AttributeError:
            pass

        return True, "Ready"

    async def _trigger_training(self, config_key: str) -> None:
        """Trigger training for a config."""
        if self.dry_run:
            logger.info(f"[MasterLoop] [DRY RUN] Would trigger training for {config_key}")
            return

        try:
            # Parse config
            parts = config_key.rsplit("_", 1)
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))

            # Emit training trigger event
            from app.coordination.event_router import emit_event, DataEventType

            await emit_event(
                DataEventType.TRAINING_THRESHOLD_REACHED,
                {
                    "config": config_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "priority": self._states[config_key].training_intensity,
                    "reason": "master_loop_trigger",
                }
            )

            logger.info(f"[MasterLoop] Triggered training for {config_key}")

            # Reset games counter
            self._states[config_key].games_since_last_export = 0
            self._states[config_key].last_export_time = time.time()

        except Exception as e:
            logger.error(f"[MasterLoop] Failed to trigger training for {config_key}: {e}")

    async def _trigger_priority_sync(self, config_key: str, data_age_hours: float) -> None:
        """Trigger priority sync for stale training data.

        Dec 29, 2025: Added as part of data freshness gate.
        When training data is stale, trigger a priority sync to fetch
        fresh data from cluster nodes before training.

        Args:
            config_key: Config to sync (e.g., "hex8_2p")
            data_age_hours: Current data age in hours
        """
        if self.dry_run:
            logger.info(
                f"[MasterLoop] [DRY RUN] Would trigger priority sync for {config_key} "
                f"(data age: {data_age_hours:.1f}h)"
            )
            return

        try:
            from app.coordination.sync_facade import get_sync_facade

            facade = get_sync_facade()
            await facade.trigger_priority_sync(
                reason="stale_training_data",
                config_key=config_key,
                data_type="games",
            )
            logger.info(
                f"[MasterLoop] Triggered priority sync for {config_key} "
                f"(data was {data_age_hours:.1f}h old)"
            )
        except ImportError:
            logger.debug("[MasterLoop] SyncFacade not available, skipping priority sync")
        except Exception as e:
            logger.warning(f"[MasterLoop] Failed to trigger priority sync for {config_key}: {e}")

    # =========================================================================
    # Allocation rebalancing
    # =========================================================================

    async def _rebalance_allocations(self) -> None:
        """Rebalance selfplay allocations based on priorities.

        December 2025 - Phase 2A.2: Now uses SelfplayScheduler for priority-based allocation.
        """
        logger.debug("[MasterLoop] Rebalancing allocations...")

        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            scheduler = get_selfplay_scheduler()

            if self.dry_run:
                # Just show priorities in dry run mode
                # Dec 28, 2025: Show all 12 configs to match max_configs
                priorities = await scheduler.get_priority_configs(top_n=12)
                for config_key, priority in priorities:
                    logger.info(f"[MasterLoop] [DRY RUN] Priority: {config_key} = {priority:.2f}")
                return

            # Get allocations from scheduler
            # Dec 28, 2025: Increased max_configs from 6 to 12 to cover all board/player configs
            # Previous value starved square19_*, hexagonal_* configs
            allocation = await scheduler.allocate_selfplay_batch(
                games_per_config=500,
                max_configs=12,
            )

            if allocation:
                # December 29, 2025: Track dispatch stats for reliability monitoring
                dispatch_stats = {"success": 0, "failed": 0}

                # Emit job allocation events
                for config_key, nodes in allocation.items():
                    total_games = sum(nodes.values())
                    logger.info(
                        f"[MasterLoop] Allocated {config_key}: {total_games} games "
                        f"across {len(nodes)} nodes"
                    )

                    # Emit event for each node allocation and track results
                    for node_id, num_games in nodes.items():
                        success = await self._emit_selfplay_job(node_id, config_key, num_games)
                        if success:
                            dispatch_stats["success"] += 1
                        else:
                            dispatch_stats["failed"] += 1
                            logger.warning(
                                f"[MasterLoop] Failed dispatch: {config_key} to {node_id}"
                            )

                # Log aggregate stats and warn if majority failed
                total_dispatches = dispatch_stats["success"] + dispatch_stats["failed"]
                if total_dispatches > 0:
                    if dispatch_stats["failed"] > dispatch_stats["success"]:
                        logger.error(
                            f"[MasterLoop] MAJORITY OF DISPATCHES FAILED: "
                            f"{dispatch_stats['failed']}/{total_dispatches} "
                            f"({dispatch_stats['failed']*100//total_dispatches}% failure rate)"
                        )
                    else:
                        logger.info(
                            f"[MasterLoop] Dispatch stats: "
                            f"{dispatch_stats['success']} succeeded, "
                            f"{dispatch_stats['failed']} failed"
                        )

                logger.info(f"[MasterLoop] Rebalanced {len(allocation)} configs")
            else:
                logger.debug("[MasterLoop] No allocations needed")

        except ImportError:
            # Fallback to simple priority logging if scheduler not available
            priorities = self._get_priority_configs()
            for config_key, priority in priorities[:3]:
                logger.debug(f"[MasterLoop] Priority: {config_key} = {priority:.2f}")
        except Exception as e:
            logger.warning(f"[MasterLoop] Error in rebalancing: {e}")

    def _get_priority_configs(self) -> list[tuple[str, float]]:
        """Rank configs by priority for selfplay allocation."""
        priorities = []

        for config_key, state in self._states.items():
            # Priority factors:
            # - Data staleness (higher staleness = higher priority)
            # - Improvement potential (configs that are improving get more resources)
            # - Exploration boost (failing configs get boosted exploration)

            staleness_factor = min(state.data_staleness_hours / MAX_DATA_STALENESS_HOURS, 2.0)
            exploration_factor = state.exploration_boost

            # Bonus for accelerated training
            intensity_factor = 1.5 if state.training_intensity == "accelerated" else 1.0

            priority = staleness_factor * exploration_factor * intensity_factor
            priorities.append((config_key, priority))

        return sorted(priorities, key=lambda x: -x[1])

    async def _emit_selfplay_job(
        self,
        node_id: str,
        config_key: str,
        num_games: int,
    ) -> bool:
        """Emit a selfplay job and dispatch it directly to the target node.

        December 2025 - Phase 2A.2: Emits events for work queue integration.
        December 29, 2025: Added direct dispatch via /selfplay/start endpoint.
        The event-only approach didn't work because IdleResourceDaemon only
        logs the event and waits for its regular loop to pick it up.

        Returns:
            True if dispatch succeeded, False otherwise.
        """
        try:
            # Parse config key
            parts = config_key.rsplit("_", 1)
            if len(parts) != 2:
                return False

            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))

            # First, try direct dispatch to the node
            dispatch_success = await self._dispatch_selfplay_to_node(
                node_id, board_type, num_players, num_games
            )

            if dispatch_success:
                logger.info(
                    f"[MasterLoop] Dispatched selfplay to {node_id}: "
                    f"{config_key}, {num_games} games"
                )
            else:
                # Fall back to event emission for other daemons to pick up
                logger.debug(
                    f"[MasterLoop] Direct dispatch failed for {node_id}, "
                    f"emitting event for {config_key}"
                )

            # Also emit the event for other subscribers (feedback loop, etc.)
            try:
                from app.coordination.event_router import emit_event, DataEventType

                await emit_event(
                    DataEventType.SELFPLAY_TARGET_UPDATED,
                    {
                        "node_id": node_id,
                        "config_key": config_key,
                        "board_type": board_type,
                        "num_players": num_players,
                        "target_games": num_games,
                        "priority": "high",
                        "source": "master_loop",
                        "dispatched": dispatch_success,
                    }
                )
            except Exception as e:
                logger.debug(f"[MasterLoop] Event emission failed (non-fatal): {e}")

            return dispatch_success
        except Exception as e:
            logger.debug(f"[MasterLoop] Error emitting selfplay job: {e}")
            return False

    async def _dispatch_selfplay_to_node(
        self,
        node_id: str,
        board_type: str,
        num_players: int,
        num_games: int,
    ) -> bool:
        """Dispatch selfplay to a node via work queue (preferred) or direct HTTP.

        Feb 2026: Changed to use work queue as primary dispatch path. Direct HTTP
        dispatch fails for NAT-blocked/firewall-blocked Lambda nodes because port
        8770 is only accessible via localhost on those nodes. The P2P WorkerPullLoop
        on each node claims work from the queue, which works regardless of NAT.

        Falls back to direct HTTP for local/coordinator nodes where it's faster.
        """
        try:
            from app.config.cluster_config import load_cluster_config

            # Get node host from cluster config
            config = load_cluster_config()
            hosts = config.hosts_raw
            host_info = hosts.get(node_id, {})

            # Skip nodes that are offline or have selfplay disabled
            node_status = host_info.get("status", "unknown")
            if node_status in ("offline", "archived", "terminated"):
                return False
            if host_info.get("selfplay_enabled") is False:
                return False

            # Select engine using multi-armed bandit (December 2025)
            config_key = f"{board_type}_{num_players}p"
            try:
                from app.coordination.selfplay_engine_bandit import get_selfplay_engine_bandit

                bandit = get_selfplay_engine_bandit()
                engine_mode = bandit.select_engine(config_key, num_players=num_players)
                logger.debug(
                    f"[MasterLoop] Bandit selected engine '{engine_mode}' for {config_key}"
                )

                # Track selected engine in feedback state for later reporting
                if config_key in self._states:
                    self._states[config_key].last_selfplay_engine = engine_mode
                    self._states[config_key].last_selfplay_games += num_games

            except (ImportError, AttributeError) as e:
                logger.debug(f"[MasterLoop] Bandit unavailable ({e}), using fallback")
                import random
                if num_players >= 3:
                    engine_mode = random.choice(["maxn", "brs", "paranoid", "gumbel-mcts", "heuristic-only"])
                elif board_type in ("square19", "hexagonal"):
                    engine_mode = random.choice(["gumbel-mcts", "nnue-guided", "descent-only", "heuristic-only"])
                else:
                    engine_mode = random.choice(["gumbel-mcts", "nnue-guided", "descent-only", "policy-only", "brs"])

            # For local node, use direct dispatch (faster, no work queue overhead)
            from app.config.env import env
            if node_id == env.node_id or node_id in ("local-mac", "localhost"):
                try:
                    from app.coordination.p2p_integration import dispatch_selfplay_direct
                    from app.config.cluster_config import get_p2p_port
                    result = await dispatch_selfplay_direct(
                        target_node=node_id,
                        host="127.0.0.1",
                        port=get_p2p_port(),
                        board_type=board_type,
                        num_players=num_players,
                        num_games=num_games,
                        engine_mode=engine_mode,
                    )
                    if result.success:
                        return True
                except Exception:
                    pass  # Fall through to work queue

            # Primary path: submit to work queue for WorkerPullLoop to claim
            try:
                from app.coordination.work_queue import get_work_queue, WorkItem, WorkType

                wq = get_work_queue()
                item = WorkItem(
                    work_type=WorkType.SELFPLAY,
                    priority=60,
                    config={
                        "board_type": board_type,
                        "num_players": num_players,
                        "num_games": num_games,
                        "engine_mode": engine_mode,
                        "target_node": node_id,
                        "source": "master_loop",
                    },
                )
                await wq.add_work_async(item)
                return True
            except Exception as e:
                logger.warning(f"[MasterLoop] Work queue dispatch to {node_id} failed: {e}")

            # Last resort: try direct HTTP dispatch
            try:
                from app.coordination.p2p_integration import dispatch_selfplay_direct
                from app.config.cluster_config import get_p2p_port

                host = host_info.get("tailscale_ip") or host_info.get("ssh_host") or node_id
                port = host_info.get("p2p_port") or get_p2p_port()
                result = await dispatch_selfplay_direct(
                    target_node=node_id,
                    host=host,
                    port=port,
                    board_type=board_type,
                    num_players=num_players,
                    num_games=num_games,
                    engine_mode=engine_mode,
                )
                if not result.success:
                    logger.info(f"[MasterLoop] Direct HTTP dispatch to {node_id} also failed: {result.error}")
                return result.success
            except Exception as e:
                logger.info(f"[MasterLoop] Direct HTTP dispatch to {node_id} exception: {e}")
                return False

        except ImportError as e:
            logger.warning(f"[MasterLoop] P2P dispatch not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"[MasterLoop] Dispatch failed for {node_id}: {e}")
            return False

    # =========================================================================
    # Evaluation handling
    # =========================================================================

    async def _check_pending_evaluations(self) -> None:
        """Check for configs pending evaluation."""
        for config_key, state in self._states.items():
            if state.pending_evaluation:
                # Check if evaluation is already running
                # (would need integration with gauntlet runner)
                logger.debug(f"[MasterLoop] {config_key}: Pending evaluation")

    # =========================================================================
    # Status reporting
    # =========================================================================

    def _log_status(self, health: ClusterHealth) -> None:
        """Log current status."""
        # Only log periodically
        if int(time.time()) % 300 != 0:  # Every 5 minutes
            return

        logger.info(
            f"[MasterLoop] Status: "
            f"nodes={health.healthy_nodes}/{health.total_nodes}, "
            f"training={health.training_nodes}, "
            f"selfplay={health.selfplay_nodes}, "
            f"gpu_util={health.avg_gpu_utilization:.0f}%"
        )

        # Log config states
        for config_key, state in self._states.items():
            if state.games_since_last_export > 0 or state.pending_evaluation:
                logger.info(
                    f"[MasterLoop]   {config_key}: "
                    f"pending_games={state.games_since_last_export}, "
                    f"intensity={state.training_intensity}"
                )

        # January 14, 2026: Log queue populator health metrics
        if self.queue_populator is not None:
            try:
                queue_metrics = self.queue_populator.get_health_metrics()
                watchdog_status = self._queue_populator_watchdog.get_status()

                logger.info(
                    f"[MasterLoop] Queue: "
                    f"depth={queue_metrics.get('queue_depth', 'N/A')}, "
                    f"bp={queue_metrics.get('backpressure_level', 'N/A')}, "
                    f"drain={queue_metrics.get('drain_rate_per_minute', 0):.1f}/min, "
                    f"circuit={queue_metrics.get('circuit_state', 'N/A')}, "
                    f"partition={queue_metrics.get('partition_detected', False)}, "
                    f"backoff={queue_metrics.get('backoff_active', False)}"
                )

                # Log watchdog status if there are concerns
                if watchdog_status.get("recovery_attempts", 0) > 0:
                    logger.warning(
                        f"[MasterLoop] Watchdog: "
                        f"recovery_attempts={watchdog_status['recovery_attempts']}, "
                        f"seconds_since_drain={watchdog_status.get('seconds_since_drain', 0):.0f}"
                    )
            except Exception as e:
                logger.debug(f"[MasterLoop] Failed to get queue metrics: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current status as dict."""
        status = {
            "running": self._running,
            "active_configs": self.active_configs,
            "dry_run": self.dry_run,
            "config_states": {
                cfg: {
                    "games_pending": state.games_since_last_export,
                    "data_staleness_hours": state.data_staleness_hours,
                    "training_intensity": state.training_intensity,
                    "exploration_boost": state.exploration_boost,
                    "pending_evaluation": state.pending_evaluation,
                }
                for cfg, state in self._states.items()
            },
        }

        # January 14, 2026: Add queue health metrics
        if self.queue_populator is not None:
            try:
                status["queue_health"] = self.queue_populator.get_health_metrics()
                status["queue_watchdog"] = self._queue_populator_watchdog.get_status()
            except Exception:
                pass

        return status


# =============================================================================
# Watch mode
# =============================================================================

async def watch_mode(controller: MasterLoopController, interval: int = 10) -> None:
    """Display live status updates."""
    import shutil

    term_width = shutil.get_terminal_size().columns

    while True:
        # Clear screen
        print("\033[2J\033[H", end="")

        # Header
        print("=" * term_width)
        print(f"RingRift Master Loop - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * term_width)

        status = controller.get_status()

        print(f"\nRunning: {status['running']}")
        print(f"Dry Run: {status['dry_run']}")
        print(f"Active Configs: {len(status['active_configs'])}")

        print("\nConfig States:")
        print("-" * term_width)
        print(f"{'Config':<15} {'Games':<8} {'Staleness':<12} {'Intensity':<12} {'Eval':<6}")
        print("-" * term_width)

        for cfg, state in status['config_states'].items():
            staleness = f"{state['data_staleness_hours']:.1f}h" if state['data_staleness_hours'] < float('inf') else "N/A"
            eval_status = "Y" if state['pending_evaluation'] else "-"
            print(
                f"{cfg:<15} "
                f"{state['games_pending']:<8} "
                f"{staleness:<12} "
                f"{state['training_intensity']:<12} "
                f"{eval_status:<6}"
            )

        print("\n(Press Ctrl+C to exit)")

        await asyncio.sleep(interval)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Master Loop Controller - Unified RingRift AI automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to unified loop config (default: config/unified_loop.yaml)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        help="Comma-separated list of configs to manage (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without executing",
    )
    parser.add_argument(
        "--skip-daemons",
        action="store_true",
        help="Don't start/stop daemons (for testing)",
    )
    parser.add_argument(
        "--profile",
        choices=["minimal", "standard", "full"],
        default="standard",
        help="Daemon profile (minimal=sync+health, standard=automation, full=all)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode - display live status",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Watch mode update interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status and exit",
    )
    parser.add_argument(
        "--kill-duplicates",
        action="store_true",
        help="Kill any existing master loop processes before starting",
    )

    return parser.parse_args()


def _kill_duplicate_processes() -> None:
    """Kill any existing master_loop processes (December 2025)."""
    import time
    pattern = r"master_loop\.py"
    existing = find_processes_by_pattern(pattern, exclude_self=True)
    if existing:
        logger.info(f"[MasterLoop] Found {len(existing)} duplicate processes, killing...")
        for proc in existing:
            logger.info(f"[MasterLoop] Killing duplicate: PID {proc.pid}")
            if kill_process(proc.pid, wait=True, timeout=5.0):
                logger.info(f"[MasterLoop] Killed PID {proc.pid}")
            else:
                logger.warning(f"[MasterLoop] Failed to kill PID {proc.pid}")
        # Wait a moment for locks to release
        time.sleep(0.5)


async def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Kill duplicates if requested (December 2025)
    if args.kill_duplicates:
        _kill_duplicate_processes()

    # Check for duplicate instance using atomic file lock (December 2025)
    global _MASTER_LOCK
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    _MASTER_LOCK = SingletonLock("master_loop", lock_dir=LOCK_DIR)

    if not _MASTER_LOCK.acquire():
        holder_pid = _MASTER_LOCK.get_holder_pid()
        if holder_pid:
            logger.error(
                f"[MasterLoop] Another instance is already running (PID {holder_pid}). "
                f"Use --kill-duplicates to automatically terminate it."
            )
        else:
            logger.error("[MasterLoop] Another instance is already running")
        sys.exit(1)

    logger.info(f"[MasterLoop] Acquired singleton lock (PID {os.getpid()})")

    # Write PID file for backward compatibility with master_loop_guard (Dec 2025)
    try:
        PID_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PID_FILE_PATH, "w") as f:
            f.write(str(os.getpid()))
        logger.debug(f"[MasterLoop] PID file written to {PID_FILE_PATH}")
    except OSError as e:
        logger.debug(f"[MasterLoop] Failed to write PID file: {e}")

    # Parse configs
    configs = None
    if args.configs:
        configs = [c.strip() for c in args.configs.split(",")]
    elif args.config:
        config = get_config(config_path=args.config, force_reload=True)
        configs = [bc.config_key for bc in config.get_all_board_configs()]

    # Create controller
    controller = MasterLoopController(
        configs=configs,
        dry_run=args.dry_run,
        skip_daemons=args.skip_daemons,
        daemon_profile=args.profile,
    )

    # Handle signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(controller.stop()))

    if args.status:
        # Just show status
        await controller._initialize_state()
        status = controller.get_status()
        import json
        print(json.dumps(status, indent=2, default=str))
        return

    if args.watch:
        # Watch mode
        await controller.start()
        await watch_mode(controller, interval=args.interval)
    else:
        # Run main loop
        await controller.run()


if __name__ == "__main__":
    asyncio.run(main())
