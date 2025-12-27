"""Daemon type definitions and data structures.

Provides the core types used by DaemonManager:
- DaemonType enum - all supported daemon types
- DaemonState enum - daemon lifecycle states
- DaemonInfo dataclass - daemon runtime information
- DaemonManagerConfig dataclass - manager configuration

Extracted from daemon_manager.py to improve modularity (Dec 2025).
"""

from __future__ import annotations

import asyncio
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = [
    "CRITICAL_DAEMONS",
    "DaemonInfo",
    "DaemonManagerConfig",
    "DaemonState",
    "DaemonType",
    "MAX_RESTART_DELAY",
    "DAEMON_RESTART_RESET_AFTER",
    "mark_daemon_ready",
]


# =============================================================================
# Deprecated Daemon Type Tracking (December 2025)
# =============================================================================

# Daemon types scheduled for removal in Q2 2026
_DEPRECATED_DAEMON_TYPES: dict[str, tuple[str, str]] = {
    "sync_coordinator": ("AUTO_SYNC", "Q2 2026"),
    "health_check": ("NODE_HEALTH_MONITOR", "Q2 2026"),
}


def _check_deprecated_daemon(daemon_type: "DaemonType") -> None:
    """Emit deprecation warning for deprecated daemon types."""
    if daemon_type.value in _DEPRECATED_DAEMON_TYPES:
        replacement, removal_date = _DEPRECATED_DAEMON_TYPES[daemon_type.value]
        warnings.warn(
            f"DaemonType.{daemon_type.name} is deprecated and will be removed in {removal_date}. "
            f"Use DaemonType.{replacement} instead.",
            DeprecationWarning,
            stacklevel=3,
        )


class DaemonType(Enum):
    """Types of daemons that can be managed."""
    # Sync daemons
    # DEPRECATED (Dec 2025): SYNC_COORDINATOR replaced by AUTO_SYNC - removal Q2 2026
    SYNC_COORDINATOR = "sync_coordinator"
    HIGH_QUALITY_SYNC = "high_quality_sync"
    ELO_SYNC = "elo_sync"
    MODEL_SYNC = "model_sync"

    # Health/monitoring
    # DEPRECATED (Dec 2025): HEALTH_CHECK replaced by NODE_HEALTH_MONITOR - removal Q2 2026
    HEALTH_CHECK = "health_check"
    CLUSTER_MONITOR = "cluster_monitor"
    QUEUE_MONITOR = "queue_monitor"
    # DEPRECATED (Dec 2025): Use UnifiedNodeHealthDaemon (health_check_orchestrator) - removal Q2 2026
    NODE_HEALTH_MONITOR = "node_health_monitor"

    # Event processing
    EVENT_ROUTER = "event_router"
    CROSS_PROCESS_POLLER = "cross_process_poller"
    DLQ_RETRY = "dlq_retry"
    DAEMON_WATCHDOG = "daemon_watchdog"  # Monitors daemon health & restarts

    # Pipeline daemons
    DATA_PIPELINE = "data_pipeline"
    SELFPLAY_COORDINATOR = "selfplay_coordinator"

    # P2P services
    P2P_BACKEND = "p2p_backend"
    GOSSIP_SYNC = "gossip_sync"
    DATA_SERVER = "data_server"

    # Training enhancement daemons (December 2025)
    DISTILLATION = "distillation"
    UNIFIED_PROMOTION = "unified_promotion"
    EXTERNAL_DRIVE_SYNC = "external_drive_sync"
    VAST_CPU_PIPELINE = "vast_cpu_pipeline"

    # Continuous training loop (December 2025)
    CONTINUOUS_TRAINING_LOOP = "continuous_training_loop"

    # DEPRECATED (Dec 2025): Use AutoSyncDaemon(strategy="broadcast") - removal Q2 2026
    CLUSTER_DATA_SYNC = "cluster_data_sync"

    # Model distribution (December 2025) - auto-distribute models after promotion
    MODEL_DISTRIBUTION = "model_distribution"

    # Automated P2P data sync (December 2025)
    AUTO_SYNC = "auto_sync"

    # Training node watcher (December 2025 - Phase 6)
    TRAINING_NODE_WATCHER = "training_node_watcher"

    # DEPRECATED (Dec 2025): Use AutoSyncDaemon(strategy="ephemeral") - removal Q2 2026
    EPHEMERAL_SYNC = "ephemeral_sync"

    # P2P auto-deployment (December 2025) - ensure P2P runs on all nodes
    P2P_AUTO_DEPLOY = "p2p_auto_deploy"

    # Replication monitor (December 2025) - monitor data replication health
    REPLICATION_MONITOR = "replication_monitor"

    # Replication repair (December 2025) - actively repair under-replicated data
    REPLICATION_REPAIR = "replication_repair"

    # Tournament daemon (December 2025) - automatic tournament scheduling
    TOURNAMENT_DAEMON = "tournament_daemon"

    # Feedback loop controller (December 2025) - orchestrates all feedback signals
    FEEDBACK_LOOP = "feedback_loop"

    # NPZ distribution (December 2025) - sync training data after export
    NPZ_DISTRIBUTION = "npz_distribution"

    # Orphan detection (December 2025) - detect orphaned games not in manifest
    ORPHAN_DETECTION = "orphan_detection"

    # Auto-evaluation (December 2025) - trigger evaluation after training completes
    EVALUATION = "evaluation"

    # Auto-promotion (December 2025) - auto-promote models based on evaluation results
    AUTO_PROMOTION = "auto_promotion"

    # S3 backup (December 2025) - backup models to S3 after promotion
    S3_BACKUP = "s3_backup"

    # Quality monitor (December 2025) - continuous selfplay quality monitoring
    QUALITY_MONITOR = "quality_monitor"

    # Model performance watchdog (December 2025) - monitors model win rates
    MODEL_PERFORMANCE_WATCHDOG = "model_performance_watchdog"

    # Job scheduler (December 2025) - centralized job scheduling with PID-based resource allocation
    JOB_SCHEDULER = "job_scheduler"
    RESOURCE_OPTIMIZER = "resource_optimizer"  # Optimizes resource allocation

    # Idle resource daemon (December 2025 - Phase 20) - monitors idle GPUs and spawns selfplay
    IDLE_RESOURCE = "idle_resource"

    # Node recovery daemon (December 2025 - Phase 21) - auto-recovers terminated nodes
    NODE_RECOVERY = "node_recovery"

    # Queue populator (December 2025 - Phase 4) - auto-populates work queue with selfplay/training jobs
    QUEUE_POPULATOR = "queue_populator"

    # Curriculum integration (December 2025) - bridges all feedback loops for self-improvement
    CURRICULUM_INTEGRATION = "curriculum_integration"

    # Auto export (December 2025) - triggers NPZ export when game thresholds met
    AUTO_EXPORT = "auto_export"

    # Training trigger (December 2025) - decides WHEN to trigger training automatically
    TRAINING_TRIGGER = "training_trigger"

    # Gauntlet feedback controller (December 2025) - bridges gauntlet evaluation to training feedback
    GAUNTLET_FEEDBACK = "gauntlet_feedback"

    # Recovery orchestrator (December 2025) - handles model/training state recovery
    RECOVERY_ORCHESTRATOR = "recovery_orchestrator"

    # Cache coordination (December 2025) - coordinates model caching across cluster
    CACHE_COORDINATION = "cache_coordination"

    # Metrics analysis (December 2025) - continuous metrics monitoring and plateau detection
    METRICS_ANALYSIS = "metrics_analysis"

    # Adaptive resource manager (December 2025) - dynamic resource scaling based on workload
    ADAPTIVE_RESOURCES = "adaptive_resources"

    # Multi-provider orchestrator (December 2025) - coordinates across Lambda/Vast/etc
    MULTI_PROVIDER = "multi_provider"

    # DEPRECATED (Dec 2025): Use unified_health_manager.get_system_health_score() - removal Q2 2026
    SYSTEM_HEALTH_MONITOR = "system_health_monitor"

    # Health server (December 2025) - exposes /health, /ready, /metrics HTTP endpoints
    HEALTH_SERVER = "health_server"

    # Maintenance daemon (December 2025) - log rotation, DB vacuum, cleanup
    MAINTENANCE = "maintenance"

    # Utilization optimizer (December 2025) - optimizes cluster workloads
    # Stops CPU selfplay on GPU nodes, spawns appropriate workloads by provider
    UTILIZATION_OPTIMIZER = "utilization_optimizer"

    # Lambda idle shutdown (December 2025) - terminates idle Lambda nodes to save costs
    # NOTE: Lambda account suspended pending support ticket resolution - keep code for restoration
    LAMBDA_IDLE = "lambda_idle"

    # Vast.ai idle shutdown (December 2025) - terminates idle Vast.ai nodes to save costs
    # Important for ephemeral marketplace instances with hourly billing
    VAST_IDLE = "vast_idle"

    # Cluster watchdog (December 2025) - self-healing cluster utilization monitor
    CLUSTER_WATCHDOG = "cluster_watchdog"

    # Data cleanup (December 2025) - auto-quarantine/delete poor quality databases
    DATA_CLEANUP = "data_cleanup"


class DaemonState(Enum):
    """State of a daemon."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    RESTARTING = "restarting"
    IMPORT_FAILED = "import_failed"  # Permanent failure due to missing imports


# Constants for recovery behavior (December 2025: imported from centralized thresholds)
try:
    from app.config.thresholds import (
        DAEMON_RESTART_DELAY_MAX,
        DAEMON_RESTART_RESET_AFTER,
    )
    MAX_RESTART_DELAY = DAEMON_RESTART_DELAY_MAX  # Legacy alias
except ImportError:
    # Fallback if thresholds not available
    DAEMON_RESTART_RESET_AFTER = 3600  # Reset restart count after 1 hour of stability
    MAX_RESTART_DELAY = 300  # Cap exponential backoff at 5 minutes


@dataclass
class DaemonInfo:
    """Information about a registered daemon."""
    daemon_type: DaemonType
    state: DaemonState = DaemonState.STOPPED
    task: asyncio.Task | None = None
    start_time: float = 0.0
    restart_count: int = 0
    last_error: str | None = None
    health_check_interval: float = 60.0
    auto_restart: bool = True
    max_restarts: int = 5
    restart_delay: float = 5.0

    # Dependencies
    depends_on: list[DaemonType] = field(default_factory=list)

    # Stability tracking for restart count reset
    stable_since: float = 0.0  # When daemon became stable (no errors)
    last_failure_time: float = 0.0  # When the last failure occurred

    # Import error tracking
    import_error: str | None = None  # Specific import error message

    # P0.3 Dec 2025: Readiness signal to prevent race conditions
    # When daemons are started, they set ready_event when initialization completes
    ready_event: asyncio.Event | None = None
    ready_timeout: float = 30.0  # Max time to wait for daemon to be ready

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        if self.state == DaemonState.RUNNING and self.start_time > 0:
            return time.time() - self.start_time
        return 0.0


@dataclass
class DaemonManagerConfig:
    """Configuration for DaemonManager."""
    auto_start: bool = False  # Auto-start all daemons on init
    health_check_interval: float = 30.0  # Global health check interval
    shutdown_timeout: float = 10.0  # Max time to wait for graceful shutdown
    force_kill_timeout: float = 5.0  # Additional time after shutdown_timeout before giving up
    auto_restart_failed: bool = True  # Auto-restart failed daemons
    max_restart_attempts: int = 5  # Max restart attempts per daemon
    recovery_cooldown: float = 10.0  # Time before attempting to recover FAILED daemons (reduced from 300s for faster recovery)
    # P11-HIGH-2: Faster health checks for critical daemons
    critical_daemon_health_interval: float = 15.0  # Health check interval for critical daemons


# P11-HIGH-2: Daemons critical for cluster health that need faster failure detection
# NOTE (Dec 2025): Only include daemons that are ACTUALLY used in standard profile.
# Optional daemons like GOSSIP_SYNC, DATA_SERVER, EPHEMERAL_SYNC should not be marked
# critical since they're not started by default.
CRITICAL_DAEMONS: set[DaemonType] = {
    DaemonType.EVENT_ROUTER,  # Core event bus - all coordination depends on this
    DaemonType.AUTO_SYNC,  # Primary data sync mechanism
    DaemonType.QUEUE_POPULATOR,  # Keeps work queue populated
    DaemonType.IDLE_RESOURCE,  # Ensures GPUs stay utilized
    DaemonType.FEEDBACK_LOOP,  # Coordinates training feedback signals
}


def mark_daemon_ready(daemon_type: DaemonType) -> None:
    """Signal that a daemon has completed initialization and is ready.

    Daemons should call this after completing their initialization to unblock
    dependent daemons that are waiting to start.

    Usage in daemon factory:
        async def _create_my_daemon(self) -> None:
            # ... initialization code ...
            await some_initialization()

            # Signal readiness so dependent daemons can start
            mark_daemon_ready(DaemonType.MY_DAEMON)

            # ... main loop ...
            while True:
                await asyncio.sleep(60)
    """
    # Import here to avoid circular dependency
    from app.coordination.daemon_manager import get_daemon_manager

    manager = get_daemon_manager()
    if manager is None:
        return

    info = manager._daemons.get(daemon_type)
    if info is not None and info.ready_event is not None:
        info.ready_event.set()
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[DaemonManager] {daemon_type.value} signaled readiness")
