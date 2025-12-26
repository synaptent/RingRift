#!/usr/bin/env python3
"""Training Coordinator for Cluster-Wide Training Management.

Provides global coordination for training across the Lambda GH200 cluster
to prevent duplicate training and provide visibility into training status.

Features:
- Global training lock per (board_type, num_players) configuration
- Cluster-wide training status visibility via NFS
- Queue management for training requests
- Automatic cleanup of stale training jobs
- Integration with distributed_lock for low-level locking

Architecture Relationship (December 2025):
-----------------------------------------
This module is part of a layered coordination architecture:

1. **TaskCoordinator** (:mod:`app.coordination.task_coordinator`)
   - Canonical for TASK ADMISSION CONTROL
   - Decides how many tasks can run based on limits/resources

2. **OrchestratorRegistry** (:mod:`app.coordination.orchestrator_registry`)
   - Canonical for ROLE-BASED COORDINATION
   - Ensures only one orchestrator per role

3. **TrainingCoordinator** (this module)
   - Specialized facade for TRAINING COORDINATION
   - Adds NFS-based locking for GH200 cluster
   - Delegates to DistributedLock for low-level locking
   - Answers: "Can I start training this (board_type, num_players) config?"

Usage:
    from app.coordination.training_coordinator import (
        TrainingCoordinator,
        request_training_slot,
        training_slot,
    )

    coordinator = TrainingCoordinator()

    # Check if training is available
    if coordinator.can_start_training("square8", 2):
        with training_slot("square8", 2) as slot:
            if slot:
                # Run training
                train_nnue(...)
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.distributed_lock import DistributedLock
from app.utils.paths import DATA_DIR

# Use centralized event emitters (December 2025)
# Note: event_emitters.py handles all routing to stage_events and cross-process buses
from app.coordination.event_emitters import (
    emit_training_complete_sync,
    emit_training_started,
)

# CoordinatorProtocol support (December 2025 - Phase 14)
from app.coordination.protocols import (
    CoordinatorStatus,
    HealthCheckResult,
    register_coordinator,
    unregister_coordinator,
)

logger = logging.getLogger(__name__)

# Coordinator registration (December 2025)
try:
    from app.coordination.orchestrator_registry import register_coordinator
    HAS_COORDINATOR_REGISTRY = True
except ImportError:
    HAS_COORDINATOR_REGISTRY = False
    register_coordinator = None  # type: ignore

# Cluster event integration (December 2025)
# Subscribe to cluster health events for training decisions
try:
    from app.distributed.data_events import DataEventType, EventBus, get_event_bus
    HAS_CLUSTER_EVENTS = True
except ImportError:
    HAS_CLUSTER_EVENTS = False
    DataEventType = None  # type: ignore
    EventBus = None  # type: ignore
    get_event_bus = None  # type: ignore

# NFS path for cluster-wide coordination (Lambda GH200 nodes)
# Configurable via RINGRIFT_NFS_COORDINATION_PATH environment variable
NFS_COORDINATION_PATH = Path(
    os.environ.get("RINGRIFT_NFS_COORDINATION_PATH", "/lambda/nfs/RingRift/coordination")
)
LOCAL_COORDINATION_PATH = DATA_DIR / "coordination"

# Import centralized timeout thresholds
try:
    from app.config.thresholds import SQLITE_BUSY_TIMEOUT_MS, SQLITE_TIMEOUT
except ImportError:
    SQLITE_BUSY_TIMEOUT_MS = 10000
    SQLITE_TIMEOUT = 30

# Training configuration - use centralized defaults (December 2025)
try:
    from app.config.coordination_defaults import HeartbeatDefaults, TrainingDefaults
    MAX_CONCURRENT_TRAINING_SAME_CONFIG = TrainingDefaults.MAX_CONCURRENT_SAME_CONFIG
    MAX_TOTAL_CONCURRENT_TRAINING = TrainingDefaults.MAX_CONCURRENT_TOTAL
    TRAINING_TIMEOUT_HOURS = TrainingDefaults.TIMEOUT_HOURS
    HEARTBEAT_INTERVAL_SECONDS = HeartbeatDefaults.INTERVAL
    STALE_CHECK_INTERVAL_SECONDS = HeartbeatDefaults.STALE_CLEANUP_INTERVAL * 5  # 5 minutes
except ImportError:
    # Fallback values
    MAX_CONCURRENT_TRAINING_SAME_CONFIG = 1
    MAX_TOTAL_CONCURRENT_TRAINING = 4
    TRAINING_TIMEOUT_HOURS = 12
    HEARTBEAT_INTERVAL_SECONDS = 60
    STALE_CHECK_INTERVAL_SECONDS = 300


@dataclass
class TrainingJob:
    """Represents an active or queued training job."""

    job_id: str
    board_type: str
    num_players: int
    node_name: str
    node_ip: str
    pid: int
    started_at: float
    last_heartbeat: float
    status: str = "running"  # running, queued, completed, failed
    model_version: str = ""
    epochs_completed: int = 0
    best_val_loss: float = float("inf")
    current_elo: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def config_key(self) -> str:
        return f"{self.board_type}_{self.num_players}p"

    @property
    def age_hours(self) -> float:
        return (time.time() - self.started_at) / 3600

    @property
    def heartbeat_age_seconds(self) -> float:
        return time.time() - self.last_heartbeat

    @property
    def is_stale(self) -> bool:
        return (
            self.heartbeat_age_seconds > HEARTBEAT_INTERVAL_SECONDS * 3
            or self.age_hours > TRAINING_TIMEOUT_HOURS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "config_key": self.config_key,
            "age_hours": round(self.age_hours, 2),
            "heartbeat_age_seconds": round(self.heartbeat_age_seconds, 1),
            "is_stale": self.is_stale,
        }


class TrainingCoordinator:
    """Cluster-wide training coordination.

    Uses NFS-backed SQLite for cross-node coordination when available,
    falls back to local SQLite otherwise.
    """

    def __init__(self, use_nfs: bool = True):
        """Initialize the training coordinator.

        Args:
            use_nfs: Whether to try using NFS for cluster-wide coordination
        """
        self._local = threading.local()
        self._use_nfs = use_nfs
        self._db_path = self._get_db_path()
        self._node_name = socket.gethostname()
        self._node_ip = self._get_node_ip()
        self._init_db()

        # Cluster health state (December 2025 - feedback loop integration)
        self._cluster_healthy = True
        self._cluster_capacity = 1.0  # 0.0-1.0, affects training decisions
        self._subscribe_to_cluster_events()

        # CoordinatorProtocol state (December 2025 - Phase 14)
        self._status = CoordinatorStatus.RUNNING
        self._start_time: float = time.time()
        self._errors_count: int = 0
        self._last_error: str = ""
        self._events_processed: int = 0

        # Register with coordinator registry
        register_coordinator(self)

    def _get_db_path(self) -> Path:
        """Determine the best database path."""
        if self._use_nfs and NFS_COORDINATION_PATH.exists():
            db_path = NFS_COORDINATION_PATH / "training_coordination.db"
            try:
                # Test write access
                db_path.parent.mkdir(parents=True, exist_ok=True)
                test_file = db_path.parent / ".write_test"
                test_file.touch()
                test_file.unlink()
                logger.info(f"Using NFS coordination at {db_path}")
                return db_path
            except Exception as e:
                logger.warning(f"NFS not writable, using local: {e}")

        # Fallback to local
        local_path = LOCAL_COORDINATION_PATH / "training_coordination.db"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using local coordination at {local_path}")
        return local_path

    def _get_node_ip(self) -> str:
        """Get the Tailscale IP of this node."""
        try:
            # Try to get Tailscale IP
            import subprocess
            result = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        # Fallback to hostname-based IP
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self._db_path),
                timeout=float(SQLITE_TIMEOUT),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute('PRAGMA journal_mode=WAL')
            self._local.conn.execute(f'PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}')
            self._local.conn.execute('PRAGMA synchronous=NORMAL')
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript('''
            -- Active training jobs
            CREATE TABLE IF NOT EXISTS training_jobs (
                job_id TEXT PRIMARY KEY,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                node_name TEXT NOT NULL,
                node_ip TEXT NOT NULL,
                pid INTEGER NOT NULL,
                started_at REAL NOT NULL,
                last_heartbeat REAL NOT NULL,
                status TEXT DEFAULT 'running',
                model_version TEXT DEFAULT '',
                epochs_completed INTEGER DEFAULT 0,
                best_val_loss REAL DEFAULT 999999,
                current_elo REAL DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            );

            -- Unique constraint: only one active training per config
            CREATE UNIQUE INDEX IF NOT EXISTS idx_training_jobs_config
                ON training_jobs(board_type, num_players)
                WHERE status = 'running';

            -- Index for cleanup queries
            CREATE INDEX IF NOT EXISTS idx_training_jobs_status
                ON training_jobs(status, last_heartbeat);

            -- Training history for analytics
            CREATE TABLE IF NOT EXISTS training_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                node_name TEXT NOT NULL,
                started_at REAL NOT NULL,
                completed_at REAL,
                status TEXT NOT NULL,
                final_val_loss REAL,
                final_elo REAL,
                epochs_completed INTEGER,
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_training_history_config
                ON training_history(board_type, num_players, completed_at DESC);

            -- Training queue for pending requests
            CREATE TABLE IF NOT EXISTS training_queue (
                queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                requester_node TEXT NOT NULL,
                requested_at REAL NOT NULL,
                priority INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_training_queue_priority
                ON training_queue(board_type, num_players, priority DESC, requested_at);
        ''')
        conn.commit()

    def _subscribe_to_cluster_events(self) -> None:
        """Subscribe to cluster health events for training decisions.

        December 2025: Closes the cluster→training feedback loop.
        Training decisions now react to cluster health events and regressions.
        """
        if not HAS_CLUSTER_EVENTS or get_event_bus is None:
            logger.debug("[TrainingCoordinator] Cluster events unavailable, skipping subscriptions")
            return

        try:
            bus = get_event_bus()
            if bus is None:
                return

            # Subscribe to cluster health events
            bus.subscribe(DataEventType.P2P_CLUSTER_HEALTHY, self._on_cluster_healthy)
            bus.subscribe(DataEventType.P2P_CLUSTER_UNHEALTHY, self._on_cluster_unhealthy)
            bus.subscribe(DataEventType.CLUSTER_CAPACITY_CHANGED, self._on_capacity_changed)
            bus.subscribe(DataEventType.NODE_RECOVERED, self._on_node_recovered)
            bus.subscribe(DataEventType.NODE_UNHEALTHY, self._on_node_unhealthy)

            # Subscribe to regression events (December 2025 - feedback loop strengthening)
            bus.subscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected)
            bus.subscribe(DataEventType.REGRESSION_CRITICAL, self._on_regression_critical)

            # Subscribe to quality events (December 2025 - real-time quality feedback)
            bus.subscribe(DataEventType.LOW_QUALITY_DATA_WARNING, self._on_low_quality_warning)
            bus.subscribe(DataEventType.TRAINING_BLOCKED_BY_QUALITY, self._on_training_blocked_by_quality)

            logger.info("[TrainingCoordinator] Subscribed to cluster health, regression, and quality events")
        except Exception as e:
            logger.warning(f"[TrainingCoordinator] Failed to subscribe to cluster events: {e}")

    def _on_cluster_healthy(self, event: Any) -> None:
        """Handle cluster healthy event - resume accepting training requests."""
        self._cluster_healthy = True
        logger.info("[TrainingCoordinator] Cluster is healthy - training enabled")

    def _on_cluster_unhealthy(self, event: Any) -> None:
        """Handle cluster unhealthy event - pause new training requests."""
        self._cluster_healthy = False
        logger.warning("[TrainingCoordinator] Cluster is unhealthy - pausing new training")

    def _on_capacity_changed(self, event: Any) -> None:
        """Handle capacity change event - adjust training intensity."""
        if hasattr(event, 'payload') and event.payload:
            new_capacity = event.payload.get('capacity', 1.0)
            old_capacity = self._cluster_capacity
            self._cluster_capacity = max(0.0, min(1.0, new_capacity))
            logger.info(
                f"[TrainingCoordinator] Cluster capacity changed: "
                f"{old_capacity:.1%} → {self._cluster_capacity:.1%}"
            )

    def _on_node_recovered(self, event: Any) -> None:
        """Handle node recovery - may allow more training."""
        logger.info("[TrainingCoordinator] Node recovered - checking training capacity")
        # Could trigger pending training requests here

    def _on_node_unhealthy(self, event: Any) -> None:
        """Handle node unhealthy - reduce training capacity."""
        logger.warning("[TrainingCoordinator] Node unhealthy - may affect training")
        # Reduce effective capacity when nodes drop

    def _on_regression_detected(self, event: Any) -> None:
        """Handle REGRESSION_DETECTED event - consider pausing training.

        December 2025: Closes the feedback loop from model evaluation.
        When a model regresses (Elo drops), we may need to pause training
        and investigate before wasting more compute.

        Behavior:
        - Elo drop < 30: Log warning, continue training
        - Elo drop 30-50: Reduce cluster capacity for this config
        - Elo drop > 50: Pause training for this config
        """
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config") or payload.get("config_key", "")
        model_id = payload.get("model_id", "")
        elo_drop = payload.get("elo_drop", 0)
        current_elo = payload.get("current_elo", 0)
        previous_elo = payload.get("previous_elo", 0)

        if not config_key:
            # Try to extract from model_id (format: "{board}_{n}p_...")
            if model_id and "_" in model_id:
                parts = model_id.split("_")
                if len(parts) >= 2:
                    config_key = f"{parts[0]}_{parts[1]}"

        if not config_key:
            logger.debug("[TrainingCoordinator] REGRESSION_DETECTED without config_key, ignoring")
            return

        self._events_processed += 1

        if elo_drop < 30:
            # Minor regression - just log
            logger.info(
                f"[TrainingCoordinator] Minor regression for {config_key}: "
                f"{previous_elo:.0f} → {current_elo:.0f} (drop: {elo_drop:.0f})"
            )
        elif elo_drop < 50:
            # Moderate regression - reduce capacity
            old_capacity = self._cluster_capacity
            self._cluster_capacity = max(0.5, self._cluster_capacity * 0.8)
            logger.warning(
                f"[TrainingCoordinator] Moderate regression for {config_key}: "
                f"Elo drop {elo_drop:.0f}, reducing capacity {old_capacity:.1%} → {self._cluster_capacity:.1%}"
            )
            # Emit capacity change event
            self._emit_training_event(
                "capacity_reduced",
                job_id="",
                board_type=config_key.split("_")[0] if "_" in config_key else config_key,
                num_players=int(config_key.split("_")[1].replace("p", "")) if "_" in config_key else 2,
                reason="regression_detected",
                elo_drop=elo_drop,
            )
        else:
            # Severe regression - pause training for this config
            logger.error(
                f"[TrainingCoordinator] SEVERE regression for {config_key}: "
                f"Elo drop {elo_drop:.0f}, pausing training"
            )
            self._pause_training_for_config(config_key, reason=f"severe_regression_elo_drop_{elo_drop:.0f}")

    def _on_regression_critical(self, event: Any) -> None:
        """Handle REGRESSION_CRITICAL event - immediately pause training.

        December 2025: Critical regressions require immediate action.
        This may indicate a corrupted model or training bug.
        """
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config") or payload.get("config_key", "")
        model_id = payload.get("model_id", "")
        elo_drop = payload.get("elo_drop", 0)

        if not config_key and model_id and "_" in model_id:
            parts = model_id.split("_")
            if len(parts) >= 2:
                config_key = f"{parts[0]}_{parts[1]}"

        if not config_key:
            logger.error("[TrainingCoordinator] REGRESSION_CRITICAL without config_key")
            return

        self._events_processed += 1
        self._errors_count += 1
        self._last_error = f"Critical regression: {config_key} dropped {elo_drop:.0f} Elo"

        logger.critical(
            f"[TrainingCoordinator] CRITICAL regression for {config_key}: "
            f"Elo drop {elo_drop:.0f}, halting all training for this config"
        )

        self._pause_training_for_config(config_key, reason=f"critical_regression_elo_drop_{elo_drop:.0f}")

        # Emit training paused event
        self._emit_via_router(
            "TRAINING_PAUSED",
            {
                "config_key": config_key,
                "reason": "critical_regression",
                "elo_drop": elo_drop,
                "timestamp": time.time(),
            },
        )

    def _pause_training_for_config(self, config_key: str, reason: str) -> bool:
        """Pause training for a specific config.

        Marks any active training job for this config as paused and prevents
        new training from starting until the pause is cleared.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")
            reason: Reason for pausing

        Returns:
            True if training was paused, False if no active training
        """
        # Parse config_key
        if "_" not in config_key:
            logger.warning(f"[TrainingCoordinator] Invalid config_key format: {config_key}")
            return False

        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        # Check for active training job
        job = self.get_job(board_type, num_players)
        if not job:
            logger.info(f"[TrainingCoordinator] No active training for {config_key} to pause")
            return False

        # Mark job as paused in database
        conn = self._get_connection()
        conn.execute(
            '''UPDATE training_jobs
               SET status = 'paused', metadata = json_set(metadata, '$.pause_reason', ?)
               WHERE job_id = ? AND status = 'running' ''',
            (reason, job.job_id)
        )
        conn.commit()

        logger.warning(
            f"[TrainingCoordinator] Paused training job {job.job_id} for {config_key}: {reason}"
        )

        return True

    def _on_low_quality_warning(self, event: Any) -> None:
        """Handle LOW_QUALITY_DATA_WARNING event - reduce training intensity.

        December 2025: Real-time quality feedback loop.
        When data quality drops, we reduce training intensity rather than
        continuing to train on low-quality data.

        Actions:
        - Log warning with quality details
        - Reduce cluster capacity allocation
        - Emit event for monitoring
        """
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config") or payload.get("config_key", "")
        quality_score = payload.get("quality_score", 0.0)
        threshold = payload.get("threshold", 0.0)
        reason = payload.get("reason", "unknown")

        self._events_processed += 1

        logger.warning(
            f"[TrainingCoordinator] Low quality data warning for {config_key}: "
            f"score={quality_score:.2f} (threshold={threshold:.2f}), reason={reason}"
        )

        # Reduce capacity for configs with quality issues
        old_capacity = self._cluster_capacity
        self._cluster_capacity = max(0.3, self._cluster_capacity * 0.7)

        if old_capacity != self._cluster_capacity:
            logger.info(
                f"[TrainingCoordinator] Reduced capacity due to quality warning: "
                f"{old_capacity:.1%} → {self._cluster_capacity:.1%}"
            )

            self._emit_training_event(
                "capacity_reduced",
                job_id="",
                board_type=config_key.split("_")[0] if "_" in config_key else "",
                num_players=2,
                reason="low_quality_data",
                quality_score=quality_score,
            )

    def _on_training_blocked_by_quality(self, event: Any) -> None:
        """Handle TRAINING_BLOCKED_BY_QUALITY event - halt training for this config.

        December 2025: Critical quality gating.
        When quality gate blocks training, we must halt any active training
        for this config and prevent new training until quality improves.
        """
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config") or payload.get("config_key", "")
        quality_score = payload.get("quality_score", 0.0)
        reason = payload.get("reason", "Quality gate blocked")

        if not config_key:
            logger.debug("[TrainingCoordinator] TRAINING_BLOCKED_BY_QUALITY without config_key")
            return

        self._events_processed += 1
        self._errors_count += 1
        self._last_error = f"Quality gate blocked for {config_key}: {quality_score:.2f}"

        logger.error(
            f"[TrainingCoordinator] Training blocked by quality for {config_key}: "
            f"score={quality_score:.2f}, reason={reason}"
        )

        # Pause training for this config
        paused = self._pause_training_for_config(
            config_key,
            reason=f"quality_gate_blocked_score_{quality_score:.2f}"
        )

        # Emit event for monitoring
        self._emit_via_router(
            "TRAINING_HALTED",
            {
                "config_key": config_key,
                "reason": "quality_gate_blocked",
                "quality_score": quality_score,
                "timestamp": time.time(),
            },
        )

        if paused:
            logger.warning(
                f"[TrainingCoordinator] Halted training for {config_key} due to quality gate"
            )

    @property
    def cluster_healthy(self) -> bool:
        """Check if cluster is healthy for training."""
        return self._cluster_healthy

    @property
    def cluster_capacity(self) -> float:
        """Get current cluster capacity (0.0-1.0)."""
        return self._cluster_capacity

    # =========================================================================
    # CoordinatorProtocol Implementation (December 2025 - Phase 14)
    # =========================================================================

    @property
    def name(self) -> str:
        """Coordinator name for identification."""
        return "TrainingCoordinator"

    @property
    def status(self) -> CoordinatorStatus:
        """Current status of the coordinator."""
        return self._status

    @property
    def uptime_seconds(self) -> float:
        """Time since coordinator started, in seconds."""
        if self._start_time <= 0:
            return 0.0
        return time.time() - self._start_time

    async def start(self) -> None:
        """Start the coordinator (already running from __init__)."""
        if self._status == CoordinatorStatus.RUNNING:
            return
        self._status = CoordinatorStatus.RUNNING
        self._start_time = time.time()
        register_coordinator(self)
        logger.info(f"[{self.name}] Started")

    async def stop(self) -> None:
        """Stop the coordinator."""
        self._status = CoordinatorStatus.STOPPING
        unregister_coordinator(self.name)
        self._status = CoordinatorStatus.STOPPED
        logger.info(f"[{self.name}] Stopped")

    def get_metrics(self) -> dict[str, Any]:
        """Get coordinator metrics.

        Returns:
            Dictionary of metrics (CoordinatorProtocol compliant)
        """
        active_jobs = self.get_active_jobs()

        return {
            "name": self.name,
            "status": self._status.value,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self._start_time,
            "events_processed": self._events_processed,
            "errors_count": self._errors_count,
            "last_error": self._last_error,
            # Custom metrics
            "active_training_jobs": len(active_jobs),
            "cluster_healthy": self._cluster_healthy,
            "cluster_capacity": self._cluster_capacity,
            "node_name": self._node_name,
            "db_path": str(self._db_path),
            "using_nfs": self._use_nfs and NFS_COORDINATION_PATH.exists(),
        }

    def health_check(self) -> HealthCheckResult:
        """Check coordinator health.

        Returns:
            HealthCheckResult (CoordinatorProtocol compliant)
        """
        if self._status == CoordinatorStatus.ERROR:
            return HealthCheckResult.unhealthy(
                f"Coordinator in error state: {self._last_error}",
                db_path=str(self._db_path),
            )

        if self._status == CoordinatorStatus.STOPPED:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Coordinator is stopped",
            )

        # Check database connectivity
        try:
            conn = self._get_connection()
            conn.execute("SELECT 1")
            db_healthy = True
        except Exception as e:
            db_healthy = False
            return HealthCheckResult.unhealthy(
                f"Database connection failed: {e}",
                db_path=str(self._db_path),
            )

        # Check cluster health
        if not self._cluster_healthy:
            return HealthCheckResult.degraded(
                "Coordinator running but cluster is unhealthy",
                cluster_capacity=self._cluster_capacity,
                db_healthy=db_healthy,
            )

        active_jobs = self.get_active_jobs()
        return HealthCheckResult(
            healthy=True,
            status=self._status,
            details={
                "active_training_jobs": len(active_jobs),
                "cluster_healthy": self._cluster_healthy,
                "cluster_capacity": self._cluster_capacity,
                "uptime_seconds": self.uptime_seconds,
                "db_path": str(self._db_path),
                "db_healthy": db_healthy,
            },
        )

    def can_start_training(self, board_type: str, num_players: int) -> bool:
        """Check if training can be started for this config.

        Returns:
            True if no active training for this config and slots available
        """
        config_key = f"{board_type}_{num_players}p"

        # Check cluster health first (December 2025 - feedback loop)
        if not self._cluster_healthy:
            logger.info("Training blocked: cluster is unhealthy")
            self._emit_slot_unavailable(
                board_type, num_players, reason="cluster_unhealthy"
            )
            return False

        conn = self._get_connection()
        self._cleanup_stale_jobs()

        # Check if this config is already being trained
        cursor = conn.execute(
            '''SELECT job_id, node_name FROM training_jobs
               WHERE board_type = ? AND num_players = ? AND status = 'running' ''',
            (board_type, num_players)
        )
        existing = cursor.fetchone()
        if existing:
            logger.info(
                f"Training for {config_key} already running on {existing['node_name']}"
            )
            self._emit_slot_unavailable(
                board_type, num_players,
                reason="already_running",
                holder_node=existing['node_name'],
                holder_job_id=existing['job_id'],
            )
            return False

        # Check total concurrent training limit
        cursor = conn.execute(
            "SELECT COUNT(*) FROM training_jobs WHERE status = 'running'"
        )
        active_count = cursor.fetchone()[0]
        if active_count >= MAX_TOTAL_CONCURRENT_TRAINING:
            logger.info(
                f"Max concurrent training ({MAX_TOTAL_CONCURRENT_TRAINING}) reached"
            )
            self._emit_slot_unavailable(
                board_type, num_players,
                reason="max_concurrent_reached",
                active_count=active_count,
                max_allowed=MAX_TOTAL_CONCURRENT_TRAINING,
            )
            return False

        return True

    def start_training(
        self,
        board_type: str,
        num_players: int,
        model_version: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Register a new training job.

        Args:
            board_type: Board type (e.g., "square8", "hex8")
            num_players: Number of players
            model_version: Version string for the model
            metadata: Additional metadata

        Returns:
            job_id if registered successfully, None if slot not available
        """
        # First try to acquire distributed lock with retry
        config_key = f"{board_type}_{num_players}p"
        lock = DistributedLock(f"training:{config_key}")

        # Retry with increasing timeouts: 30s, 60s, 90s
        lock_timeouts = [30, 60, 90]
        lock_acquired = False

        for attempt, timeout in enumerate(lock_timeouts):
            if lock.acquire(timeout=timeout, blocking=True):
                lock_acquired = True
                break

            logger.warning(
                f"Lock acquisition attempt {attempt + 1}/{len(lock_timeouts)} failed "
                f"for {config_key} (timeout={timeout}s)"
            )

            # Wait before next attempt (except on last try)
            if attempt < len(lock_timeouts) - 1:
                time.sleep(5)

        if not lock_acquired:
            logger.error(f"Could not acquire distributed lock for {config_key} after {len(lock_timeouts)} attempts")
            # Emit failure event for monitoring
            self._emit_slot_unavailable(
                board_type=board_type,
                num_players=num_players,
                reason="lock_failed",
                attempts=len(lock_timeouts),
            )
            return None

        # Emit lock acquired event for monitoring (December 2025 - Phase 14)
        self._emit_training_event(
            "lock_acquired",
            job_id="pending",  # Job ID not yet assigned
            board_type=board_type,
            num_players=num_players,
        )

        try:
            if not self.can_start_training(board_type, num_players):
                lock.release()
                return None

            conn = self._get_connection()
            now = time.time()
            job_id = f"{config_key}_{int(now)}_{os.getpid()}"

            try:
                conn.execute(
                    '''INSERT INTO training_jobs
                       (job_id, board_type, num_players, node_name, node_ip, pid,
                        started_at, last_heartbeat, status, model_version, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)''',
                    (
                        job_id, board_type, num_players,
                        self._node_name, self._node_ip, os.getpid(),
                        now, now, model_version,
                        json.dumps(metadata or {})
                    )
                )
                conn.commit()
                logger.info(f"Started training job {job_id} on {self._node_name}")

                # Emit TRAINING_STARTED event (December 2025)
                self._emit_training_event(
                    "started",
                    job_id=job_id,
                    board_type=board_type,
                    num_players=num_players,
                    model_version=model_version,
                )

                return job_id

            except sqlite3.IntegrityError:
                # Race condition - another node started training
                logger.warning(f"Race condition: {config_key} training started elsewhere")
                lock.release()
                return None

        except Exception as e:
            logger.error(f"Error starting training: {e}")
            lock.release()
            return None

    def update_progress(
        self,
        job_id: str,
        epochs_completed: int = 0,
        best_val_loss: float = float("inf"),
        current_elo: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update training progress and heartbeat.

        Args:
            job_id: The job ID returned by start_training
            epochs_completed: Number of epochs completed
            best_val_loss: Best validation loss so far
            current_elo: Current Elo rating if evaluated
            metadata: Additional metadata to update

        Returns:
            True if update successful
        """
        conn = self._get_connection()
        now = time.time()

        updates = ["last_heartbeat = ?", "epochs_completed = ?"]
        params: list[Any] = [now, epochs_completed]

        if best_val_loss < float("inf"):
            updates.append("best_val_loss = ?")
            params.append(best_val_loss)

        if current_elo > 0:
            updates.append("current_elo = ?")
            params.append(current_elo)

        if metadata:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        params.append(job_id)
        params.append(os.getpid())

        cursor = conn.execute(
            f'''UPDATE training_jobs
                SET {', '.join(updates)}
                WHERE job_id = ? AND pid = ?''',
            params
        )
        conn.commit()
        return cursor.rowcount > 0

    def complete_training(
        self,
        job_id: str,
        status: str = "completed",
        final_val_loss: float | None = None,
        final_elo: float | None = None,
    ) -> bool:
        """Mark training as complete and archive to history.

        Args:
            job_id: The job ID
            status: Final status (completed, failed)
            final_val_loss: Final validation loss
            final_elo: Final Elo rating

        Returns:
            True if completed successfully
        """
        conn = self._get_connection()
        now = time.time()

        # Get current job info
        cursor = conn.execute(
            "SELECT * FROM training_jobs WHERE job_id = ?", (job_id,)
        )
        job = cursor.fetchone()
        if not job:
            return False

        # Archive to history
        conn.execute(
            '''INSERT INTO training_history
               (job_id, board_type, num_players, node_name, started_at,
                completed_at, status, final_val_loss, final_elo,
                epochs_completed, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                job_id, job["board_type"], job["num_players"],
                job["node_name"], job["started_at"], now, status,
                final_val_loss or job["best_val_loss"],
                final_elo or job["current_elo"],
                job["epochs_completed"], job["metadata"]
            )
        )

        # Remove from active jobs
        conn.execute("DELETE FROM training_jobs WHERE job_id = ?", (job_id,))
        conn.commit()

        # Release distributed lock
        config_key = f"{job['board_type']}_{job['num_players']}p"
        lock = DistributedLock(f"training:{config_key}")
        lock.release()

        logger.info(f"Completed training job {job_id} with status {status}")

        # Emit TRAINING_COMPLETE or TRAINING_FAILED event (December 2025)
        event_type = "complete" if status == "completed" else "failed"
        self._emit_training_event(
            event_type,
            job_id=job_id,
            board_type=job["board_type"],
            num_players=job["num_players"],
            final_val_loss=final_val_loss or job["best_val_loss"],
            final_elo=final_elo or job["current_elo"],
            epochs_completed=job["epochs_completed"],
            status=status,
        )

        return True

    def _emit_training_event(
        self,
        event_type: str,
        job_id: str,
        board_type: str,
        num_players: int,
        **kwargs,
    ) -> None:
        """Emit training-related event via centralized emitters.

        Uses event_emitters.py which handles all routing to stage_events
        and cross-process buses.

        Args:
            event_type: One of "started", "complete", "failed"
            job_id: Training job ID
            board_type: Board type
            num_players: Number of players
            **kwargs: Additional event data
        """
        import asyncio

        if event_type == "started":
            # emit_training_started is async, try to run it
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(
                    emit_training_started(
                        job_id=job_id,
                        board_type=board_type,
                        num_players=num_players,
                        model_version=kwargs.get("model_version", ""),
                        node_name=self._node_name,
                    )
                )
            except RuntimeError:
                # No event loop - run synchronously
                asyncio.run(
                    emit_training_started(
                        job_id=job_id,
                        board_type=board_type,
                        num_players=num_players,
                        model_version=kwargs.get("model_version", ""),
                        node_name=self._node_name,
                    )
                )
            logger.debug(f"Emitted TRAINING_STARTED for job {job_id}")

        elif event_type in ("complete", "failed"):
            # Use sync version since we're in sync context
            success = (event_type == "complete")
            emit_training_complete_sync(
                job_id=job_id,
                board_type=board_type,
                num_players=num_players,
                success=success,
                final_loss=kwargs.get("final_val_loss"),
                final_elo=kwargs.get("final_elo"),
                model_path=kwargs.get("model_path"),
                epochs_completed=kwargs.get("epochs_completed", 0),
                node_name=self._node_name,
                status=kwargs.get("status", "completed" if success else "failed"),
            )
            event_name = "TRAINING_COMPLETE" if success else "TRAINING_FAILED"
            logger.debug(f"Emitted {event_name} for job {job_id}")

        elif event_type == "lock_acquired":
            # Emit TRAINING_LOCK_ACQUIRED for monitoring
            self._emit_via_router(
                "TRAINING_LOCK_ACQUIRED",
                {
                    "job_id": job_id,
                    "board_type": board_type,
                    "num_players": num_players,
                    "node_name": self._node_name,
                    "config": f"{board_type}_{num_players}p",
                    "timestamp": time.time(),
                },
            )
            logger.debug(f"Emitted TRAINING_LOCK_ACQUIRED for {board_type}_{num_players}p")

    def _emit_slot_unavailable(
        self,
        board_type: str,
        num_players: int,
        reason: str,
        **kwargs,
    ) -> None:
        """Emit TRAINING_SLOT_UNAVAILABLE event.

        Provides visibility into why training couldn't start.

        Args:
            board_type: Board type
            num_players: Number of players
            reason: Why slot is unavailable (cluster_unhealthy, already_running, max_concurrent_reached, lock_failed)
            **kwargs: Additional context (holder_node, holder_job_id, active_count, etc.)
        """
        self._emit_via_router(
            "TRAINING_SLOT_UNAVAILABLE",
            {
                "board_type": board_type,
                "num_players": num_players,
                "config": f"{board_type}_{num_players}p",
                "reason": reason,
                "requester_node": self._node_name,
                "timestamp": time.time(),
                **kwargs,
            },
        )
        logger.debug(
            f"Emitted TRAINING_SLOT_UNAVAILABLE for {board_type}_{num_players}p: {reason}"
        )

    def _emit_via_router(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit event via unified router.

        Args:
            event_type: Event type string
            payload: Event payload
        """
        try:
            from app.coordination.event_router import get_router
            import asyncio

            router = get_router()
            if router is None:
                return

            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(
                    router.publish(
                        event_type=event_type,
                        payload=payload,
                        source="TrainingCoordinator",
                    )
                )
            except RuntimeError:
                # No event loop running - use sync publish
                asyncio.run(
                    router.publish(
                        event_type=event_type,
                        payload=payload,
                        source="TrainingCoordinator",
                    )
                )
        except Exception as e:
            logger.debug(f"Failed to emit {event_type}: {e}")

    def get_active_jobs(self) -> list[TrainingJob]:
        """Get all active training jobs."""
        conn = self._get_connection()
        self._cleanup_stale_jobs()

        cursor = conn.execute(
            '''SELECT * FROM training_jobs WHERE status = 'running'
               ORDER BY started_at'''
        )

        jobs = []
        for row in cursor.fetchall():
            jobs.append(TrainingJob(
                job_id=row["job_id"],
                board_type=row["board_type"],
                num_players=row["num_players"],
                node_name=row["node_name"],
                node_ip=row["node_ip"],
                pid=row["pid"],
                started_at=row["started_at"],
                last_heartbeat=row["last_heartbeat"],
                status=row["status"],
                model_version=row["model_version"],
                epochs_completed=row["epochs_completed"],
                best_val_loss=row["best_val_loss"],
                current_elo=row["current_elo"],
                metadata=json.loads(row["metadata"] or "{}"),
            ))
        return jobs

    def get_job(self, board_type: str, num_players: int) -> TrainingJob | None:
        """Get the active training job for a config if any."""
        conn = self._get_connection()
        cursor = conn.execute(
            '''SELECT * FROM training_jobs
               WHERE board_type = ? AND num_players = ? AND status = 'running' ''',
            (board_type, num_players)
        )
        row = cursor.fetchone()
        if not row:
            return None

        return TrainingJob(
            job_id=row["job_id"],
            board_type=row["board_type"],
            num_players=row["num_players"],
            node_name=row["node_name"],
            node_ip=row["node_ip"],
            pid=row["pid"],
            started_at=row["started_at"],
            last_heartbeat=row["last_heartbeat"],
            status=row["status"],
            model_version=row["model_version"],
            epochs_completed=row["epochs_completed"],
            best_val_loss=row["best_val_loss"],
            current_elo=row["current_elo"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def get_training_history(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get training history."""
        conn = self._get_connection()

        query = "SELECT * FROM training_history WHERE 1=1"
        params: list[Any] = []

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)

        query += " ORDER BY completed_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def _cleanup_stale_jobs(self) -> int:
        """Remove stale training jobs."""
        conn = self._get_connection()
        now = time.time()

        # Find stale jobs
        heartbeat_threshold = now - (HEARTBEAT_INTERVAL_SECONDS * 3)
        age_threshold = now - (TRAINING_TIMEOUT_HOURS * 3600)

        cursor = conn.execute(
            '''SELECT job_id, board_type, num_players, node_name, started_at
               FROM training_jobs
               WHERE status = 'running'
                 AND (last_heartbeat < ? OR started_at < ?)''',
            (heartbeat_threshold, age_threshold)
        )
        stale_jobs = cursor.fetchall()

        for job in stale_jobs:
            logger.warning(
                f"Cleaning up stale training job {job['job_id']} "
                f"from {job['node_name']}"
            )
            # Archive with failed status
            conn.execute(
                '''INSERT INTO training_history
                   (job_id, board_type, num_players, node_name, started_at,
                    completed_at, status, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, 'stale', '{}')''',
                (
                    job["job_id"], job["board_type"], job["num_players"],
                    job["node_name"], job["started_at"], now
                )
            )
            conn.execute(
                "DELETE FROM training_jobs WHERE job_id = ?",
                (job["job_id"],)
            )

            # Release the distributed lock
            config_key = f"{job['board_type']}_{job['num_players']}p"
            lock = DistributedLock(f"training:{config_key}")
            lock.release()

        if stale_jobs:
            conn.commit()
        return len(stale_jobs)

    def get_status(self) -> dict[str, Any]:
        """Get overall training coordination status."""
        conn = self._get_connection()
        self._cleanup_stale_jobs()

        cursor = conn.execute(
            "SELECT COUNT(*) FROM training_jobs WHERE status = 'running'"
        )
        active_count = cursor.fetchone()[0]

        cursor = conn.execute(
            '''SELECT board_type, num_players, node_name, epochs_completed,
                      best_val_loss, (? - started_at) / 3600 as hours_running
               FROM training_jobs WHERE status = 'running'
               ORDER BY started_at''',
            (time.time(),)
        )
        active_jobs = [
            {
                "config": f"{row['board_type']}_{row['num_players']}p",
                "node": row["node_name"],
                "epochs": row["epochs_completed"],
                "best_loss": round(row["best_val_loss"], 4),
                "hours": round(row["hours_running"], 2),
            }
            for row in cursor.fetchall()
        ]

        return {
            "active_jobs": active_count,
            "max_concurrent": MAX_TOTAL_CONCURRENT_TRAINING,
            "slots_available": MAX_TOTAL_CONCURRENT_TRAINING - active_count,
            "coordinator_node": self._node_name,
            "db_path": str(self._db_path),
            "using_nfs": "nfs" in str(self._db_path).lower(),
            "jobs": active_jobs,
        }

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Global singleton
_coordinator: TrainingCoordinator | None = None
_coordinator_lock = threading.Lock()


def get_training_coordinator(use_nfs: bool = True) -> TrainingCoordinator:
    """Get the global training coordinator singleton."""
    global _coordinator
    with _coordinator_lock:
        if _coordinator is None:
            _coordinator = TrainingCoordinator(use_nfs=use_nfs)
            # Register with orchestrator registry for visibility (December 2025)
            if HAS_COORDINATOR_REGISTRY and register_coordinator is not None:
                try:
                    register_coordinator(
                        name="training_coordinator",
                        coordinator=_coordinator,
                        health_callback=lambda: True,  # Always healthy if exists
                        shutdown_callback=_coordinator.close,
                        metadata={"use_nfs": use_nfs},
                    )
                    logger.debug("Registered training_coordinator with orchestrator registry")
                except Exception as e:
                    logger.debug(f"Could not register with orchestrator registry: {e}")
        return _coordinator


# Convenience functions

def request_training_slot(
    board_type: str,
    num_players: int,
    model_version: str = "",
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """Request a training slot for a config.

    Returns:
        job_id if slot acquired, None otherwise
    """
    return get_training_coordinator().start_training(
        board_type, num_players, model_version, metadata
    )


def release_training_slot(
    job_id: str,
    status: str = "completed",
    final_val_loss: float | None = None,
    final_elo: float | None = None,
) -> bool:
    """Release a training slot."""
    return get_training_coordinator().complete_training(
        job_id, status, final_val_loss, final_elo
    )


def update_training_progress(
    job_id: str,
    epochs_completed: int = 0,
    best_val_loss: float = float("inf"),
    current_elo: float = 0.0,
) -> bool:
    """Update training progress."""
    return get_training_coordinator().update_progress(
        job_id, epochs_completed, best_val_loss, current_elo
    )


def can_train(board_type: str, num_players: int) -> bool:
    """Check if training can start for a config."""
    return get_training_coordinator().can_start_training(board_type, num_players)


def get_training_status() -> dict[str, Any]:
    """Get cluster-wide training status."""
    return get_training_coordinator().get_status()


@contextmanager
def training_slot(
    board_type: str,
    num_players: int,
    model_version: str = "",
    timeout: int = 60,
) -> Generator[str | None]:
    """Context manager for training slot.

    Usage:
        with training_slot("square8", 2) as job_id:
            if job_id:
                # Run training
                for epoch in range(100):
                    update_training_progress(job_id, epoch, val_loss)
            else:
                print("Training slot not available")
    """
    coordinator = get_training_coordinator()

    # Wait for slot if needed
    start_time = time.time()
    job_id = None

    while time.time() - start_time < timeout:
        job_id = coordinator.start_training(board_type, num_players, model_version)
        if job_id:
            break
        time.sleep(5)

    try:
        yield job_id
    except Exception as e:
        if job_id:
            coordinator.complete_training(job_id, status="failed")
        raise e
    else:
        if job_id:
            coordinator.complete_training(job_id, status="completed")


def wire_training_events() -> TrainingCoordinator:
    """Wire training coordinator to the event bus for automatic updates.

    Subscribes to:
    - TRAINING_STARTED: Track new training jobs
    - TRAINING_PROGRESS: Update training progress
    - TRAINING_COMPLETED: Mark training complete
    - TRAINING_FAILED: Handle training failures

    Returns:
        The configured TrainingCoordinator instance
    """
    coordinator = get_training_coordinator()

    try:
        # Use unified event router (consolidated from data_events)
        from app.coordination.event_router import get_router
        from app.coordination.event_router import DataEventType  # Types still needed

        router = get_router()

        def _event_payload(event: Any) -> dict[str, Any]:
            if isinstance(event, dict):
                return event
            payload = getattr(event, "payload", None)
            return payload if isinstance(payload, dict) else {}

        def _on_training_started(event: Any) -> None:
            """Handle training start event."""
            payload = _event_payload(event)
            job_id = payload.get("job_id")
            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            if job_id and board_type and num_players:
                # Training already registered - just log
                logger.info(f"[TrainingCoordinator] Training started: {job_id}")

        def _on_training_progress(event: Any) -> None:
            """Handle training progress update."""
            payload = _event_payload(event)
            job_id = payload.get("job_id")
            epoch = payload.get("epoch")
            loss = payload.get("loss") or payload.get("val_loss")
            if job_id:
                coordinator.update_progress(job_id, epoch=epoch, current_loss=loss)

        def _on_training_completed(event: Any) -> None:
            """Handle training completion."""
            payload = _event_payload(event)
            job_id = payload.get("job_id")
            final_loss = payload.get("final_loss") or payload.get("val_loss")
            elo = payload.get("elo")
            if job_id:
                coordinator.complete_training(
                    job_id,
                    status="completed",
                    final_val_loss=final_loss,
                    final_elo=elo,
                )

        def _on_training_failed(event: Any) -> None:
            """Handle training failure."""
            payload = _event_payload(event)
            job_id = payload.get("job_id")
            error = payload.get("error", "Unknown error")
            if job_id:
                coordinator.complete_training(job_id, status="failed", error=error)

        router.subscribe(DataEventType.TRAINING_STARTED.value, _on_training_started)
        router.subscribe(DataEventType.TRAINING_PROGRESS.value, _on_training_progress)
        router.subscribe(DataEventType.TRAINING_COMPLETED.value, _on_training_completed)
        router.subscribe(DataEventType.TRAINING_FAILED.value, _on_training_failed)

        logger.info("[TrainingCoordinator] Wired to event router (TRAINING_STARTED, TRAINING_PROGRESS, TRAINING_COMPLETED, TRAINING_FAILED)")

    except ImportError:
        logger.warning("[TrainingCoordinator] data_events not available, running without event bus")

    return coordinator


__all__ = [
    # Main class
    "TrainingCoordinator",
    # Data classes
    "TrainingJob",
    "can_train",
    # Singleton getter
    "get_training_coordinator",
    "get_training_status",
    "release_training_slot",
    # Convenience functions
    "request_training_slot",
    "training_slot",
    "update_training_progress",
    # Event wiring
    "wire_training_events",
]


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training coordination management")
    parser.add_argument("--status", action="store_true", help="Show training status")
    parser.add_argument("--history", action="store_true", help="Show training history")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup stale jobs")
    parser.add_argument("--board", type=str, help="Board type filter")
    parser.add_argument("--players", type=int, help="Number of players filter")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    coordinator = get_training_coordinator()

    if args.status:
        status = coordinator.get_status()
        print(json.dumps(status, indent=2))

    elif args.history:
        history = coordinator.get_training_history(
            board_type=args.board,
            num_players=args.players,
            limit=20
        )
        for entry in history:
            print(
                f"{entry['job_id']}: {entry['status']} "
                f"(loss={entry.get('final_val_loss', 'N/A')}, "
                f"elo={entry.get('final_elo', 'N/A')})"
            )

    elif args.cleanup:
        cleaned = coordinator._cleanup_stale_jobs()
        print(f"Cleaned up {cleaned} stale jobs")

    else:
        parser.print_help()
