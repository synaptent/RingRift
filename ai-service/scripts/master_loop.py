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
import json
import logging
import os
import signal
import sqlite3
import sys
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
)
from app.config.unified_config import get_config

# Import coordination bootstrap for event wiring (December 2025)
# This is critical for feedback loops to function properly
from app.coordination.coordination_bootstrap import bootstrap_coordination

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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

# Loop timing (Dec 2025: Reduced from 60/300/600 for faster event-driven pipeline)
LOOP_INTERVAL_SECONDS = 30  # Check every 30 seconds
TRAINING_CHECK_INTERVAL = 60  # Check training readiness every minute (fallback for events)
ALLOCATION_CHECK_INTERVAL = 120  # Rebalance allocations every 2 minutes (fallback for events)

# Thresholds
MIN_GAMES_FOR_EXPORT = 1000  # Minimum new games before triggering export
MAX_DATA_STALENESS_HOURS = 4.0  # Trigger sync if data older than this

# State persistence path (Gap 3 fix: Dec 2025)
STATE_DB_PATH = Path(__file__).parent.parent / "data" / "coordination" / "master_loop_state.db"
STATE_SAVE_INTERVAL_SECONDS = 300  # Save state every 5 minutes

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
    last_quality_score: float = 0.0
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

    @property
    def data_staleness_hours(self) -> float:
        """Hours since last export."""
        if self.last_export_time == 0:
            return float("inf")
        return (time.time() - self.last_export_time) / 3600

    @property
    def needs_training(self) -> bool:
        """Check if config needs training."""
        # Has enough games and data is fresh enough
        return (
            self.games_since_last_export >= MIN_GAMES_FOR_EXPORT
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

        # Control
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Lazy-loaded managers
        self._daemon_manager = None
        self._cluster_monitor = None
        self._resource_manager = None
        self._feedback_controller = None
        self._pipeline_orchestrator = None

        # State persistence (Gap 3 fix: Dec 2025)
        self._db_path = STATE_DB_PATH
        self._last_state_save = 0.0
        self._loop_iteration = 0  # Heartbeat tracking (Dec 2025)
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

    # =========================================================================
    # State Persistence (Gap 3 fix: Dec 2025)
    # =========================================================================

    def _init_state_db(self) -> None:
        """Initialize the state persistence database."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS config_state (
                    config_key TEXT PRIMARY KEY,
                    exploration_boost REAL NOT NULL DEFAULT 1.0,
                    training_intensity TEXT NOT NULL DEFAULT 'normal',
                    last_quality_score REAL NOT NULL DEFAULT 0.0,
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
            conn.close()
            logger.debug(f"[MasterLoop] State DB initialized at {self._db_path}")
        except Exception as e:
            logger.warning(f"[MasterLoop] Failed to init state DB: {e}")

    def _load_persisted_state(self) -> None:
        """Load exploration_boost and other state from database on startup."""
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute("""
                SELECT config_key, exploration_boost, training_intensity, last_quality_score
                FROM config_state
            """).fetchall()
            conn.close()

            restored_count = 0
            for config_key, boost, intensity, quality_score in rows:
                if config_key in self._states:
                    self._states[config_key].exploration_boost = boost
                    self._states[config_key].training_intensity = intensity
                    self._states[config_key].last_quality_score = quality_score
                    restored_count += 1

            if restored_count > 0:
                logger.info(f"[MasterLoop] Restored persisted state for {restored_count} configs")

        except Exception as e:
            logger.warning(f"[MasterLoop] Failed to load persisted state: {e}")

    def _save_persisted_state(self) -> None:
        """Save exploration_boost and other state to database."""
        try:
            conn = sqlite3.connect(self._db_path)
            now = time.time()

            for config_key, state in self._states.items():
                conn.execute("""
                    INSERT OR REPLACE INTO config_state
                    (config_key, exploration_boost, training_intensity, last_quality_score, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    config_key,
                    state.exploration_boost,
                    state.training_intensity,
                    state.last_quality_score,
                    now,
                ))

            conn.commit()
            conn.close()
            self._last_state_save = now
            logger.debug(f"[MasterLoop] Persisted state for {len(self._states)} configs")

        except Exception as e:
            logger.warning(f"[MasterLoop] Failed to save persisted state: {e}")

    def _maybe_save_state(self) -> None:
        """Save state if enough time has passed since last save."""
        if time.time() - self._last_state_save >= STATE_SAVE_INTERVAL_SECONDS:
            self._save_persisted_state()

    def _update_heartbeat(self, status: str = "running") -> None:
        """Update heartbeat for health monitoring (Dec 2025).

        This allows external monitoring to detect hung loops by checking
        last_beat timestamp. A healthy loop should update every 30-60 seconds.
        """
        try:
            self._loop_iteration += 1
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                UPDATE heartbeat
                SET last_beat = ?, loop_iteration = ?, active_configs = ?, status = ?
                WHERE id = 1
            """, (time.time(), self._loop_iteration, len(self.active_configs), status))
            conn.commit()
            conn.close()
        except Exception as e:
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

        Returns:
            Dict with 'healthy' bool, 'last_beat' timestamp, 'age_seconds', 'status'
        """
        db_path = db_path or STATE_DB_PATH
        try:
            if not db_path.exists():
                return {"healthy": False, "error": "State DB not found"}

            conn = sqlite3.connect(db_path)
            row = conn.execute("""
                SELECT last_beat, loop_iteration, active_configs, status
                FROM heartbeat WHERE id = 1
            """).fetchone()
            conn.close()

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

        Returns:
            Tuple of (success, list of error messages)
        """
        import aiohttp
        from app.config.distributed_hosts import load_hosts_config

        errors = []
        config = load_hosts_config()

        # Get voter nodes from config
        voters = config.get("p2p_cluster", {}).get("voters", [])
        if not voters:
            # No voters configured, skip check
            logger.warning("[MasterLoop] No P2P voters configured, skipping P2P health check")
            return True, []

        p2p_port = config.get("p2p_cluster", {}).get("port", 8770)
        reachable = 0
        total = len(voters)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
            for voter in voters:
                # Find host info for this voter
                host_info = next(
                    (h for h in config.get("hosts", []) if h.get("name") == voter),
                    None,
                )
                if not host_info:
                    errors.append(f"Voter {voter} not found in hosts config")
                    continue

                host = host_info.get("host")
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
        except Exception as e:
            logger.error(f"[MasterLoop] Coordination bootstrap failed: {e}")

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

        # Start daemons
        if not self.skip_daemons:
            await self._start_daemons()

        # Subscribe to events
        self._subscribe_to_events()

        # Restore persisted state (exploration_boost, etc.) - Gap 3 fix
        self._load_persisted_state()

        # Initialize state from current data
        await self._initialize_state()

        logger.info("[MasterLoop] Started successfully")

    async def stop(self) -> None:
        """Stop the master loop gracefully."""
        logger.info("[MasterLoop] Stopping...")
        self._running = False
        self._shutdown_event.set()

        # Save state before shutdown - Gap 3 fix
        self._save_persisted_state()

        # Mark heartbeat as stopped (Dec 2025)
        self._update_heartbeat("stopped")

        # Remove PID file (December 2025)
        self._remove_pid_file()

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
                    self._maybe_save_state()

                    # 7. Update heartbeat for health monitoring (Dec 2025)
                    self._update_heartbeat("running")

                except Exception as e:
                    logger.error(f"[MasterLoop] Error in loop iteration: {e}")

                # Wait for next iteration
                elapsed = time.time() - loop_start
                sleep_time = max(0, LOOP_INTERVAL_SECONDS - elapsed)

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
            DaemonType.NODE_HEALTH_MONITOR,
            DaemonType.CLUSTER_MONITOR,
            DaemonType.SYSTEM_HEALTH_MONITOR,
            DaemonType.HEALTH_SERVER,
            # Dec 2025 fix: DATA_PIPELINE and FEEDBACK_LOOP BEFORE sync daemons
            # They subscribe to events that sync daemons emit (DATA_SYNC_COMPLETED, etc.)
            # If sync starts first, events are lost before subscribers are ready
            DaemonType.FEEDBACK_LOOP,  # Subscribes to: TRAINING_COMPLETED, EVALUATION_COMPLETED, etc.
            DaemonType.DATA_PIPELINE,  # Subscribes to: DATA_SYNC_COMPLETED, SELFPLAY_COMPLETE, etc.
            # Sync daemons (emit events that DATA_PIPELINE receives)
            DaemonType.AUTO_SYNC,
            DaemonType.CLUSTER_DATA_SYNC,
            DaemonType.ELO_SYNC,
        ]

        standard = minimal + [
            # Core automation (current default stack)
            # Note: FEEDBACK_LOOP and DATA_PIPELINE moved to minimal for correct ordering
            DaemonType.MODEL_DISTRIBUTION,
            DaemonType.IDLE_RESOURCE,
            DaemonType.UTILIZATION_OPTIMIZER,
            DaemonType.QUEUE_POPULATOR,
            DaemonType.AUTO_EXPORT,
            # Dec 2025: DATA_CONSOLIDATION merges scattered selfplay games into canonical DBs
            # Must run after AUTO_SYNC (games need to be synced first) and before training
            DaemonType.DATA_CONSOLIDATION,
            DaemonType.TRAINING_TRIGGER,  # Dec 27 2025: Added as critical daemon
            DaemonType.EVALUATION,
            DaemonType.AUTO_PROMOTION,
            DaemonType.TOURNAMENT_DAEMON,
            DaemonType.CURRICULUM_INTEGRATION,
            DaemonType.NODE_RECOVERY,
            DaemonType.TRAINING_NODE_WATCHER,
            DaemonType.QUALITY_MONITOR,
        ]

        deprecated = {
            DaemonType.SYNC_COORDINATOR,
            DaemonType.HEALTH_CHECK,
        }
        full = [daemon for daemon in DaemonType if daemon not in deprecated]

        profiles = {
            "minimal": minimal,
            "standard": standard,
            "full": full,
        }

        daemons = profiles.get(self.daemon_profile, standard)

        # December 2025: Coordinator-only mode filtering
        # When running on a coordinator node, filter out intensive daemons
        if env.is_coordinator:
            # Daemons that run CPU/GPU intensive processes
            # These should NEVER run on coordinator nodes
            intensive_daemons = {
                DaemonType.IDLE_RESOURCE,           # spawns selfplay
                DaemonType.TRAINING_NODE_WATCHER,   # monitors training
                DaemonType.AUTO_EXPORT,             # exports training data (CPU-bound)
                DaemonType.TRAINING_TRIGGER,        # triggers training jobs
                DaemonType.TOURNAMENT_DAEMON,       # runs tournaments
                DaemonType.EVALUATION,              # runs gauntlets
                DaemonType.AUTO_PROMOTION,          # triggers promotion (can spawn gauntlet)
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
        try:
            validate_startup_order_or_raise()
            logger.debug("[MasterLoop] Daemon startup order validated successfully")
        except ValueError as e:
            logger.error(f"[MasterLoop] Daemon dependency validation failed: {e}")
            # Continue with warning - don't block startup, but log prominently
            logger.warning("[MasterLoop] Proceeding despite dependency issues")

        # Track which daemons started successfully
        started_daemons: set[str] = set()
        failed_daemons: set[str] = set()

        for daemon_type in profile_daemons:
            try:
                if self.dry_run:
                    logger.info(f"[MasterLoop] [DRY RUN] Would start {daemon_type.value}")
                    started_daemons.add(daemon_type.value)
                else:
                    await self.daemon_manager.start(daemon_type)
                    started_daemons.add(daemon_type.value)
                    logger.debug(f"[MasterLoop] Started {daemon_type.value}")
            except Exception as e:
                failed_daemons.add(daemon_type.value)
                logger.warning(f"[MasterLoop] Failed to start {daemon_type.value}: {e}")

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
            # Don't raise - log warning but continue (graceful degradation)
            # In production, consider: raise RuntimeError(f"Critical daemons failed: {critical_issues}")
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
            config_key = event.payload.get("config_key", "")
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
            config_key = event.payload.get("config_key", "")
            policy_accuracy = event.payload.get("policy_accuracy", 0.0)

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
            config_key = event.payload.get("config_key", "")
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
            config_key = event.payload.get("config_key", "")
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

                # Sync intensity to TrainingTriggerDaemon (Gap 2 fix: Dec 2025)
                # This ensures the training subprocess uses the correct parameters
                try:
                    from app.coordination.training_trigger_daemon import (
                        get_training_trigger_daemon,
                    )
                    trigger_daemon = get_training_trigger_daemon()
                    if config_key in trigger_daemon._training_states:
                        trigger_daemon._training_states[config_key].training_intensity = (
                            state.training_intensity
                        )
                except ImportError:
                    pass  # Daemon not available

        except Exception as e:
            logger.debug(f"[MasterLoop] Error handling quality event: {e}")

    # =========================================================================
    # State initialization
    # =========================================================================

    async def _initialize_state(self) -> None:
        """Initialize state from current data."""
        logger.info("[MasterLoop] Initializing state from current data...")

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

        # Check minimum games
        if state.games_since_last_export < MIN_GAMES_FOR_EXPORT:
            return False, f"Insufficient games: {state.games_since_last_export} < {MIN_GAMES_FOR_EXPORT}"

        # Check quality score
        if state.last_quality_score < 0.5:
            return False, f"Low quality: {state.last_quality_score:.2f}"

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
                priorities = await scheduler.get_priority_configs(top_n=6)
                for config_key, priority in priorities:
                    logger.info(f"[MasterLoop] [DRY RUN] Priority: {config_key} = {priority:.2f}")
                return

            # Get allocations from scheduler
            allocation = await scheduler.allocate_selfplay_batch(
                games_per_config=500,
                max_configs=6,
            )

            if allocation:
                # Emit job allocation events
                for config_key, nodes in allocation.items():
                    total_games = sum(nodes.values())
                    logger.info(
                        f"[MasterLoop] Allocated {config_key}: {total_games} games "
                        f"across {len(nodes)} nodes"
                    )

                    # Emit event for each node allocation
                    for node_id, num_games in nodes.items():
                        await self._emit_selfplay_job(node_id, config_key, num_games)

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
    ) -> None:
        """Emit a selfplay job allocation event.

        December 2025 - Phase 2A.2: Emits events for work queue integration.
        """
        try:
            from app.coordination.event_router import emit_event, DataEventType

            # Parse config key
            parts = config_key.rsplit("_", 1)
            if len(parts) != 2:
                return

            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))

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
                }
            )
        except Exception as e:
            logger.debug(f"[MasterLoop] Error emitting selfplay job: {e}")

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

    def get_status(self) -> dict[str, Any]:
        """Get current status as dict."""
        return {
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
