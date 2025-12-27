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
from typing import TYPE_CHECKING, Any, Callable

from app.config.coordination_defaults import DaemonLoopDefaults

if TYPE_CHECKING:
    pass

__all__ = [
    "CRITICAL_DAEMONS",
    "DAEMON_DEPENDENCIES",
    "DAEMON_STARTUP_ORDER",
    "DaemonInfo",
    "DaemonManagerConfig",
    "DaemonState",
    "DaemonType",
    "MAX_RESTART_DELAY",
    "DAEMON_RESTART_RESET_AFTER",
    "get_daemon_startup_position",
    "mark_daemon_ready",
    "register_mark_ready_callback",
    "validate_daemon_dependencies",
]


# =============================================================================
# Deprecated Daemon Type Tracking (December 2025)
# =============================================================================

# Daemon types scheduled for removal in Q2 2026
_DEPRECATED_DAEMON_TYPES: dict[str, tuple[str, str]] = {
    "sync_coordinator": ("AUTO_SYNC", "Q2 2026"),
    "health_check": ("NODE_HEALTH_MONITOR", "Q2 2026"),
    # December 2025: Added missing deprecated types
    "ephemeral_sync": ("AUTO_SYNC", "Q2 2026"),
    "cluster_data_sync": ("AUTO_SYNC", "Q2 2026"),
    "system_health_monitor": ("HEALTH_SERVER", "Q2 2026"),  # Use unified_health_manager
    # December 2025: Lambda Labs account terminated - use VAST_IDLE or UNIFIED_IDLE
    "lambda_idle": ("VAST_IDLE", "Q2 2026"),
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

    # Work queue monitor (December 2025) - tracks WORK_* lifecycle events
    # Provides queue depth, latency metrics, backpressure signaling, stuck job detection
    WORK_QUEUE_MONITOR = "work_queue_monitor"

    # Coordinator health monitor (December 2025) - tracks COORDINATOR_* lifecycle events
    # Provides coordinator health state, heartbeat monitoring, cluster health summary
    COORDINATOR_HEALTH_MONITOR = "coordinator_health_monitor"

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

    # Multi-provider orchestrator (December 2025) - coordinates across Vast/RunPod/Nebius/etc
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
    # DEPRECATED: Lambda Labs account permanently terminated Dec 2025. No Lambda nodes remain.
    # Code retained for historical reference. Use VAST_IDLE or other provider daemons instead.
    LAMBDA_IDLE = "lambda_idle"

    # Vast.ai idle shutdown (December 2025) - terminates idle Vast.ai nodes to save costs
    # Important for ephemeral marketplace instances with hourly billing
    VAST_IDLE = "vast_idle"

    # Cluster watchdog (December 2025) - self-healing cluster utilization monitor
    CLUSTER_WATCHDOG = "cluster_watchdog"

    # Data cleanup (December 2025) - auto-quarantine/delete poor quality databases
    DATA_CLEANUP = "data_cleanup"

    # Data consolidation (December 2025) - consolidate scattered selfplay games into canonical DBs
    DATA_CONSOLIDATION = "data_consolidation"

    # Disk space manager (December 2025) - proactive disk space management
    # Monitors disk usage and triggers cleanup before reaching critical thresholds
    DISK_SPACE_MANAGER = "disk_space_manager"


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

    # December 2025: Store daemon instance for health_check() calls
    # Set by factory functions in daemon_runners.py
    instance: Any | None = None

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        if self.state == DaemonState.RUNNING and self.start_time > 0:
            return time.time() - self.start_time
        return 0.0


@dataclass
class DaemonManagerConfig:
    """Configuration for DaemonManager.

    Default values sourced from app.config.coordination_defaults.DaemonLoopDefaults
    for centralized configuration management.
    """
    auto_start: bool = False  # Auto-start all daemons on init
    # Dec 2025: Use centralized defaults from coordination_defaults.py
    health_check_interval: float = field(
        default_factory=lambda: float(DaemonLoopDefaults.CHECK_INTERVAL) / 10.0  # 30s (10% of check interval)
    )
    shutdown_timeout: float = field(
        default_factory=lambda: DaemonLoopDefaults.SHUTDOWN_GRACE_PERIOD
    )
    force_kill_timeout: float = field(
        default_factory=lambda: DaemonLoopDefaults.HEALTH_CHECK_TIMEOUT
    )
    auto_restart_failed: bool = True  # Auto-restart failed daemons
    max_restart_attempts: int = field(
        default_factory=lambda: DaemonLoopDefaults.MAX_CONSECUTIVE_ERRORS
    )
    recovery_cooldown: float = field(
        default_factory=lambda: DaemonLoopDefaults.ERROR_BACKOFF_BASE * 2  # 10s (2x base backoff)
    )
    # P11-HIGH-2: Faster health checks for critical daemons
    critical_daemon_health_interval: float = field(
        default_factory=lambda: DaemonLoopDefaults.HEALTH_CHECK_TIMEOUT * 3  # 15s (3x health timeout)
    )


# P11-HIGH-2: Daemons critical for cluster health that need faster failure detection
# NOTE (Dec 2025): Only include daemons that are ACTUALLY used in standard profile.
# Optional daemons like GOSSIP_SYNC, DATA_SERVER, EPHEMERAL_SYNC should not be marked
# critical since they're not started by default.
CRITICAL_DAEMONS: set[DaemonType] = {
    DaemonType.EVENT_ROUTER,  # Core event bus - all coordination depends on this
    DaemonType.DAEMON_WATCHDOG,  # Self-healing for daemon crashes (Dec 2025 fix)
    DaemonType.DATA_PIPELINE,  # Pipeline processor - must start before AUTO_SYNC (Dec 2025 fix)
    DaemonType.AUTO_SYNC,  # Primary data sync mechanism
    DaemonType.QUEUE_POPULATOR,  # Keeps work queue populated
    DaemonType.IDLE_RESOURCE,  # Ensures GPUs stay utilized
    DaemonType.FEEDBACK_LOOP,  # Coordinates training feedback signals
}

# P0 Critical Fix (Dec 2025): Daemon startup order to prevent race conditions
# DATA_PIPELINE and FEEDBACK_LOOP must start BEFORE AUTO_SYNC to avoid event loss.
# Events emitted by AUTO_SYNC (DATA_SYNC_COMPLETED) need handlers ready.
DAEMON_STARTUP_ORDER: list[DaemonType] = [
    DaemonType.EVENT_ROUTER,           # 1. Event system must be first
    DaemonType.DAEMON_WATCHDOG,        # 2. Self-healing for daemon crashes
    DaemonType.DATA_PIPELINE,          # 3. Pipeline processor (before sync!)
    DaemonType.FEEDBACK_LOOP,          # 4. Training feedback (before sync!)
    DaemonType.AUTO_SYNC,              # 5. Data sync (emits events)
    DaemonType.QUEUE_POPULATOR,        # 6. Work queue maintenance
    DaemonType.WORK_QUEUE_MONITOR,     # 7. Queue visibility (after populator)
    DaemonType.COORDINATOR_HEALTH_MONITOR,  # 8. Coordinator visibility
    DaemonType.IDLE_RESOURCE,          # 9. GPU utilization
    DaemonType.TRAINING_TRIGGER,       # 10. Training trigger (after pipeline)
]


# P0.4 Dec 2025: Explicit dependency map for startup validation
# Daemons MUST wait for all dependencies to be running before starting.
# This prevents race conditions where events are emitted before handlers are ready.
DAEMON_DEPENDENCIES: dict[DaemonType, set[DaemonType]] = {
    # Core infrastructure (no dependencies)
    DaemonType.EVENT_ROUTER: set(),
    DaemonType.DAEMON_WATCHDOG: {DaemonType.EVENT_ROUTER},

    # Pipeline processors (depend on event system)
    DaemonType.DATA_PIPELINE: {DaemonType.EVENT_ROUTER},
    DaemonType.FEEDBACK_LOOP: {DaemonType.EVENT_ROUTER},

    # Sync daemons (depend on pipeline being ready to handle events)
    DaemonType.AUTO_SYNC: {
        DaemonType.EVENT_ROUTER,
        DaemonType.DATA_PIPELINE,
        DaemonType.FEEDBACK_LOOP,
    },

    # Queue management (depend on event system)
    DaemonType.QUEUE_POPULATOR: {DaemonType.EVENT_ROUTER},
    DaemonType.WORK_QUEUE_MONITOR: {DaemonType.EVENT_ROUTER, DaemonType.QUEUE_POPULATOR},
    DaemonType.COORDINATOR_HEALTH_MONITOR: {DaemonType.EVENT_ROUTER},

    # Resource management (depend on queue being populated)
    DaemonType.IDLE_RESOURCE: {DaemonType.EVENT_ROUTER, DaemonType.QUEUE_POPULATOR},

    # Training coordination (depend on pipeline and sync)
    DaemonType.TRAINING_TRIGGER: {
        DaemonType.EVENT_ROUTER,
        DaemonType.DATA_PIPELINE,
        DaemonType.AUTO_SYNC,
    },

    # Evaluation daemons
    DaemonType.EVALUATION: {DaemonType.EVENT_ROUTER, DaemonType.TRAINING_TRIGGER},
    DaemonType.AUTO_PROMOTION: {DaemonType.EVENT_ROUTER, DaemonType.EVALUATION},

    # Distribution daemons
    DaemonType.MODEL_DISTRIBUTION: {DaemonType.EVENT_ROUTER, DaemonType.AUTO_PROMOTION},
    DaemonType.NPZ_DISTRIBUTION: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},

    # P2P daemons
    DaemonType.GOSSIP_SYNC: {DaemonType.EVENT_ROUTER},
    DaemonType.P2P_BACKEND: set(),  # Runs independently
    DaemonType.P2P_AUTO_DEPLOY: {DaemonType.EVENT_ROUTER},

    # Monitoring daemons
    DaemonType.CLUSTER_MONITOR: {DaemonType.EVENT_ROUTER},
    DaemonType.QUEUE_MONITOR: {DaemonType.EVENT_ROUTER},
    DaemonType.REPLICATION_MONITOR: {DaemonType.EVENT_ROUTER},

    # Multi-provider orchestrator
    DaemonType.MULTI_PROVIDER: {DaemonType.EVENT_ROUTER, DaemonType.IDLE_RESOURCE},

    # =========================================================================
    # Additional daemon dependencies (December 2025 - P0 CRITICAL fix)
    # =========================================================================

    # Sync daemons
    DaemonType.HIGH_QUALITY_SYNC: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},
    DaemonType.ELO_SYNC: {DaemonType.EVENT_ROUTER},
    DaemonType.MODEL_SYNC: {DaemonType.EVENT_ROUTER},

    # Health/monitoring daemons
    DaemonType.NODE_HEALTH_MONITOR: {DaemonType.EVENT_ROUTER},
    DaemonType.SYSTEM_HEALTH_MONITOR: {DaemonType.EVENT_ROUTER},  # Deprecated but may still be used
    DaemonType.HEALTH_SERVER: {DaemonType.EVENT_ROUTER},
    DaemonType.CLUSTER_WATCHDOG: {DaemonType.EVENT_ROUTER, DaemonType.CLUSTER_MONITOR},
    DaemonType.MODEL_PERFORMANCE_WATCHDOG: {DaemonType.EVENT_ROUTER},

    # Event processing daemons
    DaemonType.CROSS_PROCESS_POLLER: {DaemonType.EVENT_ROUTER},
    DaemonType.DLQ_RETRY: {DaemonType.EVENT_ROUTER},

    # Pipeline/selfplay daemons
    DaemonType.SELFPLAY_COORDINATOR: {DaemonType.EVENT_ROUTER},
    DaemonType.CONTINUOUS_TRAINING_LOOP: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},
    DaemonType.QUALITY_MONITOR: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},

    # P2P/data server daemons
    DaemonType.DATA_SERVER: {DaemonType.EVENT_ROUTER},

    # Training enhancement daemons
    DaemonType.DISTILLATION: {DaemonType.EVENT_ROUTER, DaemonType.TRAINING_TRIGGER},
    DaemonType.UNIFIED_PROMOTION: {DaemonType.EVENT_ROUTER, DaemonType.EVALUATION},
    DaemonType.EXTERNAL_DRIVE_SYNC: {DaemonType.EVENT_ROUTER},
    DaemonType.VAST_CPU_PIPELINE: {DaemonType.EVENT_ROUTER},

    # Node watching/recovery daemons
    DaemonType.TRAINING_NODE_WATCHER: {DaemonType.EVENT_ROUTER},
    DaemonType.NODE_RECOVERY: {DaemonType.EVENT_ROUTER, DaemonType.NODE_HEALTH_MONITOR},

    # Replication daemons
    DaemonType.REPLICATION_REPAIR: {DaemonType.EVENT_ROUTER, DaemonType.REPLICATION_MONITOR},

    # Tournament/evaluation daemons
    DaemonType.TOURNAMENT_DAEMON: {DaemonType.EVENT_ROUTER},
    DaemonType.GAUNTLET_FEEDBACK: {DaemonType.EVENT_ROUTER, DaemonType.EVALUATION},

    # Data management daemons
    DaemonType.ORPHAN_DETECTION: {DaemonType.EVENT_ROUTER},
    DaemonType.AUTO_EXPORT: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},
    DaemonType.DATA_CLEANUP: {DaemonType.EVENT_ROUTER, DaemonType.AUTO_SYNC},

    # Data consolidation depends on event router and data pipeline
    DaemonType.DATA_CONSOLIDATION: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},
    DaemonType.S3_BACKUP: {DaemonType.EVENT_ROUTER, DaemonType.AUTO_PROMOTION},

    # Job/resource daemons
    DaemonType.JOB_SCHEDULER: {DaemonType.EVENT_ROUTER},
    DaemonType.RESOURCE_OPTIMIZER: {DaemonType.EVENT_ROUTER},
    DaemonType.UTILIZATION_OPTIMIZER: {DaemonType.EVENT_ROUTER, DaemonType.IDLE_RESOURCE},

    # Curriculum/feedback daemons
    DaemonType.CURRICULUM_INTEGRATION: {DaemonType.EVENT_ROUTER, DaemonType.FEEDBACK_LOOP},

    # Recovery/coordination daemons
    DaemonType.RECOVERY_ORCHESTRATOR: {DaemonType.EVENT_ROUTER},
    DaemonType.CACHE_COORDINATION: {DaemonType.EVENT_ROUTER},
    DaemonType.METRICS_ANALYSIS: {DaemonType.EVENT_ROUTER},
    DaemonType.ADAPTIVE_RESOURCES: {DaemonType.EVENT_ROUTER, DaemonType.IDLE_RESOURCE},

    # Maintenance daemons
    DaemonType.MAINTENANCE: {DaemonType.EVENT_ROUTER},

    # Disk space management
    DaemonType.DISK_SPACE_MANAGER: {DaemonType.EVENT_ROUTER},

    # Provider-specific idle daemons
    DaemonType.VAST_IDLE: {DaemonType.EVENT_ROUTER},
    DaemonType.LAMBDA_IDLE: {DaemonType.EVENT_ROUTER},  # Deprecated but may still be referenced

    # Deprecated daemons (empty deps - should not be started)
    DaemonType.SYNC_COORDINATOR: set(),  # DEPRECATED: Use AUTO_SYNC
    DaemonType.HEALTH_CHECK: set(),  # DEPRECATED: Use NODE_HEALTH_MONITOR
    DaemonType.CLUSTER_DATA_SYNC: set(),  # DEPRECATED: Use AUTO_SYNC
    DaemonType.EPHEMERAL_SYNC: set(),  # DEPRECATED: Use AUTO_SYNC
}


def validate_daemon_dependencies(
    daemon_type: DaemonType,
    running_daemons: set[DaemonType],
) -> tuple[bool, list[DaemonType]]:
    """Check if all dependencies for a daemon are running.

    Args:
        daemon_type: The daemon to check dependencies for.
        running_daemons: Set of currently running daemon types.

    Returns:
        Tuple of (all_satisfied, missing_deps).
        If all_satisfied is True, missing_deps will be empty.
    """
    required = DAEMON_DEPENDENCIES.get(daemon_type, set())
    missing = [dep for dep in required if dep not in running_daemons]
    return (len(missing) == 0, missing)


def get_daemon_startup_position(daemon_type: DaemonType) -> int:
    """Get the startup position for a daemon in DAEMON_STARTUP_ORDER.

    Args:
        daemon_type: The daemon type to look up.

    Returns:
        Position (0-indexed) in startup order, or -1 if not in order.
        Daemons not in the order list can start after ordered daemons.
    """
    try:
        return DAEMON_STARTUP_ORDER.index(daemon_type)
    except ValueError:
        return -1


# =============================================================================
# Callback Registration Pattern for Breaking Circular Dependencies (Dec 2025)
# =============================================================================
# Instead of importing daemon_manager, we use a callback that daemon_manager
# registers when it initializes. This breaks the daemon_types → daemon_manager
# circular dependency.

_mark_ready_callback: "Callable[[DaemonType], None] | None" = None


def register_mark_ready_callback(callback: "Callable[[DaemonType], None]") -> None:
    """Register the callback for mark_daemon_ready().

    This is called by DaemonManager.__init__() to provide the implementation
    without creating a circular import.

    Args:
        callback: Function that takes a DaemonType and signals readiness.
    """
    global _mark_ready_callback
    _mark_ready_callback = callback


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

    Note:
        This function uses a callback pattern to avoid circular imports.
        The callback is registered by DaemonManager.__init__().
    """
    if _mark_ready_callback is not None:
        _mark_ready_callback(daemon_type)
    else:
        # Fallback: callback not registered yet (rare race condition at startup)
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"[DaemonTypes] mark_daemon_ready called for {daemon_type.value} "
            "before DaemonManager initialized"
        )
