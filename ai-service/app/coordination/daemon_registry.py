"""Data-driven daemon registry for DaemonManager.

December 2025 - Extracted from daemon_manager._register_default_factories().

This module provides a declarative registry of all daemon types and their
configuration. Benefits:
- Declarative over imperative: Easy to see all daemons and their dependencies
- Testable: Registry can be validated independently
- Extensible: Add new daemons by adding entries, not code
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.coordination.daemon_types import DaemonType

if TYPE_CHECKING:
    from app.coordination import daemon_runners


@dataclass(frozen=True)
class DaemonSpec:
    """Specification for a daemon registration.

    Attributes:
        runner_name: Name of the runner function in daemon_runners module.
            Must be a valid attribute name like "create_auto_sync".
        depends_on: List of daemon types that must be running first.
        health_check_interval: Custom health check interval (None = use defaults).
        auto_restart: Whether to auto-restart on failure.
        max_restarts: Maximum restart attempts.
        category: Human-readable category for documentation.
        deprecated: Whether this daemon is deprecated (December 2025).
        deprecated_message: Migration guidance for deprecated daemons.
    """

    runner_name: str
    depends_on: tuple[DaemonType, ...] = field(default_factory=tuple)
    health_check_interval: float | None = None
    auto_restart: bool = True
    max_restarts: int = 5
    category: str = "misc"
    deprecated: bool = False
    deprecated_message: str = ""


# =============================================================================
# DAEMON REGISTRY - Declarative specification of all daemons
# =============================================================================

DAEMON_REGISTRY: dict[DaemonType, DaemonSpec] = {
    # =========================================================================
    # Sync Daemons
    # All sync daemons depend on EVENT_ROUTER since they emit events
    # =========================================================================
    DaemonType.SYNC_COORDINATOR: DaemonSpec(
        runner_name="create_sync_coordinator",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
        deprecated=True,
        deprecated_message="Use AUTO_SYNC daemon instead. Removal: Q2 2026.",
    ),
    DaemonType.HIGH_QUALITY_SYNC: DaemonSpec(
        runner_name="create_high_quality_sync",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
    ),
    DaemonType.ELO_SYNC: DaemonSpec(
        runner_name="create_elo_sync",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
    ),
    # Auto sync (December 2025) - emits 12+ event types including DATA_SYNC_*
    # CRITICAL: Must depend on DATA_PIPELINE and FEEDBACK_LOOP to ensure
    # event handlers are subscribed before AUTO_SYNC emits events.
    DaemonType.AUTO_SYNC: DaemonSpec(
        runner_name="create_auto_sync",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE, DaemonType.FEEDBACK_LOOP),
        category="sync",
    ),
    # Training node watcher (Phase 6)
    DaemonType.TRAINING_NODE_WATCHER: DaemonSpec(
        runner_name="create_training_node_watcher",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE),
        category="sync",
    ),
    # Training data sync (December 2025) - pre-training data sync from OWC/S3
    DaemonType.TRAINING_DATA_SYNC: DaemonSpec(
        runner_name="create_training_data_sync",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
    ),
    # Ephemeral sync for Vast.ai (Phase 4)
    DaemonType.EPHEMERAL_SYNC: DaemonSpec(
        runner_name="create_ephemeral_sync",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE),
        category="sync",
        deprecated=True,
        deprecated_message="Use AutoSyncDaemon(strategy='ephemeral') instead. Removal: Q2 2026.",
    ),
    DaemonType.GOSSIP_SYNC: DaemonSpec(
        runner_name="create_gossip_sync",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
    ),
    # =========================================================================
    # Event Processing
    # =========================================================================
    DaemonType.EVENT_ROUTER: DaemonSpec(
        runner_name="create_event_router",
        category="event",
    ),
    DaemonType.CROSS_PROCESS_POLLER: DaemonSpec(
        runner_name="create_cross_process_poller",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="event",
    ),
    DaemonType.DLQ_RETRY: DaemonSpec(
        runner_name="create_dlq_retry",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="event",
    ),
    # =========================================================================
    # Health & Monitoring
    # =========================================================================
    DaemonType.HEALTH_CHECK: DaemonSpec(
        runner_name="create_health_check",
        category="health",
        deprecated=True,
        deprecated_message="Use NODE_HEALTH_MONITOR daemon instead. Removal: Q2 2026.",
    ),
    DaemonType.QUEUE_MONITOR: DaemonSpec(
        runner_name="create_queue_monitor",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="health",
    ),
    DaemonType.DAEMON_WATCHDOG: DaemonSpec(
        runner_name="create_daemon_watchdog",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="health",
    ),
    DaemonType.NODE_HEALTH_MONITOR: DaemonSpec(
        runner_name="create_node_health_monitor",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="health",
        deprecated=True,
        deprecated_message="Use health_check_orchestrator.py instead. Removal: Q2 2026.",
    ),
    DaemonType.SYSTEM_HEALTH_MONITOR: DaemonSpec(
        runner_name="create_system_health_monitor",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.NODE_HEALTH_MONITOR),
        category="health",
        deprecated=True,
        deprecated_message="Use unified_health_manager.py instead. Removal: Q2 2026.",
    ),
    DaemonType.QUALITY_MONITOR: DaemonSpec(
        runner_name="create_quality_monitor",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="health",
    ),
    DaemonType.MODEL_PERFORMANCE_WATCHDOG: DaemonSpec(
        runner_name="create_model_performance_watchdog",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="health",
    ),
    DaemonType.CLUSTER_MONITOR: DaemonSpec(
        runner_name="create_cluster_monitor",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="health",
    ),
    DaemonType.CLUSTER_WATCHDOG: DaemonSpec(
        runner_name="create_cluster_watchdog",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.CLUSTER_MONITOR),
        category="health",
    ),
    DaemonType.COORDINATOR_HEALTH_MONITOR: DaemonSpec(
        runner_name="create_coordinator_health_monitor",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="health",
    ),
    DaemonType.WORK_QUEUE_MONITOR: DaemonSpec(
        runner_name="create_work_queue_monitor",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.QUEUE_POPULATOR),
        category="health",
    ),
    # December 2025: HEALTH_SERVER now has a runner in daemon_runners.py
    # that wraps the DaemonManager._create_health_server method.
    DaemonType.HEALTH_SERVER: DaemonSpec(
        runner_name="create_health_server",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="health",
        auto_restart=True,
        health_check_interval=30.0,
    ),
    # =========================================================================
    # Training & Pipeline
    # =========================================================================
    DaemonType.DATA_PIPELINE: DaemonSpec(
        runner_name="create_data_pipeline",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="pipeline",
    ),
    DaemonType.CONTINUOUS_TRAINING_LOOP: DaemonSpec(
        runner_name="create_continuous_training_loop",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="pipeline",
    ),
    DaemonType.SELFPLAY_COORDINATOR: DaemonSpec(
        runner_name="create_selfplay_coordinator",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="pipeline",
    ),
    DaemonType.TRAINING_TRIGGER: DaemonSpec(
        runner_name="create_training_trigger",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.AUTO_EXPORT),
        category="pipeline",
    ),
    DaemonType.AUTO_EXPORT: DaemonSpec(
        runner_name="create_auto_export",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="pipeline",
    ),
    DaemonType.TOURNAMENT_DAEMON: DaemonSpec(
        runner_name="create_tournament_daemon",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="pipeline",
    ),
    # =========================================================================
    # Evaluation & Promotion
    # =========================================================================
    DaemonType.EVALUATION: DaemonSpec(
        runner_name="create_evaluation_daemon",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="evaluation",
    ),
    DaemonType.AUTO_PROMOTION: DaemonSpec(
        runner_name="create_auto_promotion",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.EVALUATION),
        category="evaluation",
    ),
    DaemonType.UNIFIED_PROMOTION: DaemonSpec(
        runner_name="create_unified_promotion",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="evaluation",
    ),
    DaemonType.GAUNTLET_FEEDBACK: DaemonSpec(
        runner_name="create_gauntlet_feedback",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="evaluation",
    ),
    # =========================================================================
    # Distribution
    # =========================================================================
    DaemonType.MODEL_SYNC: DaemonSpec(
        runner_name="create_model_sync",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="distribution",
    ),
    DaemonType.MODEL_DISTRIBUTION: DaemonSpec(
        runner_name="create_model_distribution",
        # Dec 2025: Added EVALUATION and AUTO_PROMOTION dependencies to ensure
        # model distribution only runs after evaluation completes and promotion
        # is approved. Prevents race conditions where distribution starts before
        # the model is ready.
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.EVALUATION, DaemonType.AUTO_PROMOTION),
        category="distribution",
    ),
    DaemonType.NPZ_DISTRIBUTION: DaemonSpec(
        runner_name="create_npz_distribution",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="distribution",
        deprecated=True,
        deprecated_message="Use unified_distribution_daemon.py with DataType.NPZ. Removal: Q2 2026.",
    ),
    DaemonType.DATA_SERVER: DaemonSpec(
        runner_name="create_data_server",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="distribution",
    ),
    # =========================================================================
    # Replication
    # =========================================================================
    DaemonType.REPLICATION_MONITOR: DaemonSpec(
        runner_name="create_replication_monitor",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="replication",
        deprecated=True,
        deprecated_message="Use unified_replication_daemon.py instead. Removal: Q2 2026.",
    ),
    DaemonType.REPLICATION_REPAIR: DaemonSpec(
        runner_name="create_replication_repair",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="replication",
        deprecated=True,
        deprecated_message="Use unified_replication_daemon.py instead. Removal: Q2 2026.",
    ),
    # =========================================================================
    # Resource Management
    # =========================================================================
    DaemonType.IDLE_RESOURCE: DaemonSpec(
        runner_name="create_idle_resource",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="resource",
    ),
    DaemonType.NODE_RECOVERY: DaemonSpec(
        runner_name="create_node_recovery",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="resource",
    ),
    DaemonType.RESOURCE_OPTIMIZER: DaemonSpec(
        runner_name="create_resource_optimizer",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.JOB_SCHEDULER),
        category="resource",
    ),
    DaemonType.UTILIZATION_OPTIMIZER: DaemonSpec(
        runner_name="create_utilization_optimizer",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.IDLE_RESOURCE),
        category="resource",
    ),
    DaemonType.ADAPTIVE_RESOURCES: DaemonSpec(
        runner_name="create_adaptive_resources",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.CLUSTER_MONITOR),
        category="resource",
    ),
    # =========================================================================
    # Provider-Specific
    # =========================================================================
    DaemonType.LAMBDA_IDLE: DaemonSpec(
        runner_name="create_lambda_idle",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.CLUSTER_MONITOR),
        category="provider",
        deprecated=True,
        deprecated_message="Lambda GH200 nodes are dedicated training infrastructure (restored Dec 28, 2025). Dedicated nodes don't need idle shutdown. Removal: Q2 2026.",
    ),
    DaemonType.VAST_IDLE: DaemonSpec(
        runner_name="create_vast_idle",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.CLUSTER_MONITOR),
        category="provider",
        deprecated=True,
        deprecated_message="Use create_vast_idle_daemon() from unified_idle_shutdown_daemon. Removal: Q2 2026.",
    ),
    DaemonType.MULTI_PROVIDER: DaemonSpec(
        runner_name="create_multi_provider",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.CLUSTER_MONITOR),
        category="provider",
    ),
    # =========================================================================
    # Queue & Job Management
    # =========================================================================
    DaemonType.QUEUE_POPULATOR: DaemonSpec(
        runner_name="create_queue_populator",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.SELFPLAY_COORDINATOR),
        category="queue",
    ),
    DaemonType.JOB_SCHEDULER: DaemonSpec(
        runner_name="create_job_scheduler",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="queue",
    ),
    # =========================================================================
    # Feedback & Curriculum
    # =========================================================================
    DaemonType.FEEDBACK_LOOP: DaemonSpec(
        runner_name="create_feedback_loop",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="feedback",
    ),
    DaemonType.CURRICULUM_INTEGRATION: DaemonSpec(
        runner_name="create_curriculum_integration",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="feedback",
    ),
    # =========================================================================
    # Recovery & Maintenance
    # =========================================================================
    DaemonType.RECOVERY_ORCHESTRATOR: DaemonSpec(
        runner_name="create_recovery_orchestrator",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.NODE_HEALTH_MONITOR),
        category="recovery",
    ),
    DaemonType.CACHE_COORDINATION: DaemonSpec(
        runner_name="create_cache_coordination",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.CLUSTER_MONITOR),
        category="recovery",
    ),
    DaemonType.MAINTENANCE: DaemonSpec(
        runner_name="create_maintenance",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="recovery",
    ),
    DaemonType.ORPHAN_DETECTION: DaemonSpec(
        runner_name="create_orphan_detection",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="recovery",
    ),
    DaemonType.DATA_CLEANUP: DaemonSpec(
        runner_name="create_data_cleanup",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="recovery",
    ),
    # =========================================================================
    # Miscellaneous
    # =========================================================================
    DaemonType.S3_BACKUP: DaemonSpec(
        runner_name="create_s3_backup",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.MODEL_DISTRIBUTION),
        category="misc",
    ),
    # S3 node sync (December 2025) - bi-directional S3 sync for all cluster nodes
    DaemonType.S3_NODE_SYNC: DaemonSpec(
        runner_name="create_s3_node_sync",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
    ),
    # S3 consolidation (December 2025) - consolidates data from all nodes (coordinator only)
    DaemonType.S3_CONSOLIDATION: DaemonSpec(
        runner_name="create_s3_consolidation",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.S3_NODE_SYNC),
        category="sync",
    ),
    DaemonType.DISTILLATION: DaemonSpec(
        runner_name="create_distillation",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="misc",
    ),
    DaemonType.EXTERNAL_DRIVE_SYNC: DaemonSpec(
        runner_name="create_external_drive_sync",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="misc",
    ),
    DaemonType.VAST_CPU_PIPELINE: DaemonSpec(
        runner_name="create_vast_cpu_pipeline",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="misc",
    ),
    DaemonType.CLUSTER_DATA_SYNC: DaemonSpec(
        runner_name="create_cluster_data_sync",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="misc",
        deprecated=True,
        deprecated_message="Use AutoSyncDaemon(strategy='broadcast') instead. Removal: Q2 2026.",
    ),
    DaemonType.P2P_BACKEND: DaemonSpec(
        runner_name="create_p2p_backend",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="misc",
    ),
    DaemonType.P2P_AUTO_DEPLOY: DaemonSpec(
        runner_name="create_p2p_auto_deploy",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="misc",
    ),
    DaemonType.METRICS_ANALYSIS: DaemonSpec(
        runner_name="create_metrics_analysis",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="misc",
    ),
    # =========================================================================
    # Disk Management (December 2025)
    # =========================================================================
    DaemonType.DISK_SPACE_MANAGER: DaemonSpec(
        runner_name="create_disk_space_manager",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="recovery",
    ),
    DaemonType.COORDINATOR_DISK_MANAGER: DaemonSpec(
        runner_name="create_coordinator_disk_manager",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DISK_SPACE_MANAGER),
        category="recovery",
    ),
    # Sync push daemon (December 28, 2025) - push-based sync for GPU training nodes
    # GPU nodes push data to coordinator before cleanup to prevent data loss
    DaemonType.SYNC_PUSH: DaemonSpec(
        runner_name="create_sync_push",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
        auto_restart=True,
        health_check_interval=60.0,
    ),
    # Unified data plane (December 28, 2025) - consolidated data synchronization
    # Replaces fragmented sync infrastructure (~4,514 LOC consolidated)
    # Components: DataCatalog, SyncPlanner v2, TransportManager, EventBridge
    # IMPORTANT: Depends on DATA_PIPELINE and FEEDBACK_LOOP to ensure event
    # handlers are subscribed before this daemon emits sync events
    DaemonType.UNIFIED_DATA_PLANE: DaemonSpec(
        runner_name="create_unified_data_plane",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE, DaemonType.FEEDBACK_LOOP),
        category="sync",
        auto_restart=True,
        health_check_interval=60.0,
    ),
    # =========================================================================
    # Data Consolidation
    # =========================================================================
    DaemonType.DATA_CONSOLIDATION: DaemonSpec(
        runner_name="create_data_consolidation",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE),
        category="pipeline",
    ),
    # NPZ Combination (December 2025) - quality-weighted NPZ combination
    DaemonType.NPZ_COMBINATION: DaemonSpec(
        runner_name="create_npz_combination",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE),
        category="pipeline",
        health_check_interval=60.0,
    ),
    # =========================================================================
    # Data Integrity (December 2025)
    # =========================================================================
    DaemonType.INTEGRITY_CHECK: DaemonSpec(
        runner_name="create_integrity_check",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="recovery",
        health_check_interval=3600.0,  # 1 hour - matches scan interval
    ),
    # =========================================================================
    # Cluster Availability Manager (December 28, 2025)
    # Provides automated cluster availability management:
    # - NodeMonitor: Multi-layer health checking (P2P, SSH, GPU, Provider API)
    # - RecoveryEngine: Escalating recovery strategies
    # - Provisioner: Auto-provision new instances when capacity drops
    # - CapacityPlanner: Budget-aware capacity management
    # =========================================================================
    DaemonType.NODE_AVAILABILITY: DaemonSpec(
        runner_name="create_node_availability",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="resource",
    ),
    DaemonType.AVAILABILITY_NODE_MONITOR: DaemonSpec(
        runner_name="create_availability_node_monitor",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="health",
        health_check_interval=30.0,  # 30s - quick failure detection
    ),
    DaemonType.AVAILABILITY_RECOVERY_ENGINE: DaemonSpec(
        runner_name="create_availability_recovery_engine",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.AVAILABILITY_NODE_MONITOR),
        category="recovery",
    ),
    DaemonType.AVAILABILITY_CAPACITY_PLANNER: DaemonSpec(
        runner_name="create_availability_capacity_planner",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="resource",
    ),
    DaemonType.AVAILABILITY_PROVISIONER: DaemonSpec(
        runner_name="create_availability_provisioner",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.AVAILABILITY_CAPACITY_PLANNER),
        category="resource",
    ),
    # =========================================================================
    # Cascade Training & PER (December 29, 2025)
    # =========================================================================
    DaemonType.CASCADE_TRAINING: DaemonSpec(
        runner_name="create_cascade_training",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE),
        category="pipeline",
    ),
    DaemonType.PER_ORCHESTRATOR: DaemonSpec(
        runner_name="create_per_orchestrator",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="pipeline",
        health_check_interval=60.0,  # Monitor PER buffer health
    ),
    # =========================================================================
    # 48-Hour Autonomous Operation (December 29, 2025)
    # These daemons enable the system to run unattended for 48+ hours
    # =========================================================================
    DaemonType.PROGRESS_WATCHDOG: DaemonSpec(
        runner_name="create_progress_watchdog",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.SELFPLAY_COORDINATOR),
        category="health",
        health_check_interval=1800.0,  # 30 min - long-running watchdog
    ),
    DaemonType.P2P_RECOVERY: DaemonSpec(
        runner_name="create_p2p_recovery",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="health",
        health_check_interval=300.0,  # 5 min - monitor P2P health
        auto_restart=True,
        max_restarts=10,  # More restarts allowed for critical recovery daemon
    ),
    # =========================================================================
    # Tailscale Health Monitoring (December 29, 2025)
    # Monitors and auto-recovers Tailscale connectivity on each cluster node
    # =========================================================================
    DaemonType.TAILSCALE_HEALTH: DaemonSpec(
        runner_name="create_tailscale_health",
        depends_on=(),  # Runs independently on each node
        category="health",
        health_check_interval=60.0,  # 1 min - check Tailscale every minute
        auto_restart=True,
        max_restarts=20,  # Critical for connectivity - allow many restarts
    ),
}


def get_daemons_by_category(category: str) -> list[DaemonType]:
    """Get all daemon types in a specific category.

    Args:
        category: Category name (sync, event, health, pipeline, etc.)

    Returns:
        List of daemon types in that category
    """
    return [
        daemon_type
        for daemon_type, spec in DAEMON_REGISTRY.items()
        if spec.category == category
    ]


def get_categories() -> list[str]:
    """Get all unique category names.

    Returns:
        Sorted list of unique category names
    """
    return sorted(set(spec.category for spec in DAEMON_REGISTRY.values()))


def get_deprecated_daemons() -> list[tuple[DaemonType, str]]:
    """Get all deprecated daemon types with their migration messages.

    Returns:
        List of (DaemonType, deprecated_message) tuples for deprecated daemons.

    Example:
        >>> for daemon_type, message in get_deprecated_daemons():
        ...     print(f"{daemon_type.name}: {message}")
    """
    return [
        (daemon_type, spec.deprecated_message)
        for daemon_type, spec in DAEMON_REGISTRY.items()
        if spec.deprecated
    ]


def is_daemon_deprecated(daemon_type: DaemonType) -> bool:
    """Check if a daemon type is deprecated.

    Args:
        daemon_type: The daemon type to check

    Returns:
        True if the daemon is marked as deprecated
    """
    spec = DAEMON_REGISTRY.get(daemon_type)
    return spec is not None and spec.deprecated


def get_deprecated_types() -> set[DaemonType]:
    """Get the set of deprecated daemon types.

    This is computed from the registry entries with deprecated=True.

    Returns:
        Set of DaemonType values that are deprecated.

    December 2025: Added to provide a single source of truth for deprecated types.
    """
    return {
        daemon_type
        for daemon_type, spec in DAEMON_REGISTRY.items()
        if spec.deprecated
    }


# Convenience constant - computed lazily on first access
# Use get_deprecated_types() for guaranteed freshness
DEPRECATED_TYPES: set[DaemonType] = set()  # Populated at module load


def _init_deprecated_types() -> None:
    """Initialize DEPRECATED_TYPES constant at module load."""
    global DEPRECATED_TYPES
    DEPRECATED_TYPES = get_deprecated_types()


# Initialize on module load
_init_deprecated_types()


def validate_registry() -> list[str]:
    """Validate the daemon registry for common issues.

    Checks:
        1. All DaemonType enum values have registry entries (except deprecated types)
        2. All registry entries have runner functions in daemon_runners
        3. No dependency references to non-existent daemon types
        4. No self-dependencies

    Returns:
        List of validation error messages (empty if valid)

    December 2025: Enhanced to check ALL DaemonType values have registrations.
    """
    errors: list[str] = []

    # Check for DaemonType values missing from registry
    all_daemon_types = set(DaemonType)
    registered_types = set(DAEMON_REGISTRY.keys())
    deprecated_types = get_deprecated_types()

    # Missing types that are NOT deprecated should be errors
    # Missing types that ARE deprecated are acceptable (may have been removed from registry)
    missing_from_registry = all_daemon_types - registered_types
    for daemon_type in sorted(missing_from_registry, key=lambda x: x.name):
        # Only flag as error if not deprecated
        if daemon_type not in deprecated_types:
            errors.append(
                f"{daemon_type.name}: DaemonType exists but has no DAEMON_REGISTRY entry"
            )

    # Check for missing daemon_runners functions
    # CIRCULAR DEPENDENCY NOTE (Dec 2025):
    # This lazy import of daemon_runners is SAFE because:
    # 1. validate_registry() is only called at runtime (from DaemonManager._register_default_factories)
    # 2. By that point, both daemon_registry and daemon_runners are fully loaded
    # 3. daemon_runners only imports DaemonType in TYPE_CHECKING (not at runtime)
    try:
        from app.coordination import daemon_runners

        for daemon_type, spec in DAEMON_REGISTRY.items():
            if not hasattr(daemon_runners, spec.runner_name):
                errors.append(
                    f"{daemon_type.name}: runner '{spec.runner_name}' not found in daemon_runners"
                )
    except ImportError as e:
        errors.append(f"Cannot import daemon_runners: {e}")

    # Check for circular dependencies
    for daemon_type, spec in DAEMON_REGISTRY.items():
        for dep in spec.depends_on:
            if dep not in DAEMON_REGISTRY:
                # May be HEALTH_SERVER or other inline daemon
                if dep != DaemonType.HEALTH_SERVER:
                    errors.append(
                        f"{daemon_type.name}: dependency '{dep.name}' not in registry"
                    )

    # Check for self-dependencies
    for daemon_type, spec in DAEMON_REGISTRY.items():
        if daemon_type in spec.depends_on:
            errors.append(f"{daemon_type.name}: cannot depend on itself")

    return errors


def validate_registry_or_raise() -> None:
    """Validate registry and raise if there are any errors.

    Raises:
        ValueError: If there are any validation errors.

    Usage:
        Called at daemon_manager startup to catch configuration errors early.

    December 2025: Added for startup validation.
    """
    errors = validate_registry()
    if errors:
        error_msg = (
            "Daemon registry validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nFix: Add missing entries to DAEMON_REGISTRY in daemon_registry.py"
        )
        raise ValueError(error_msg)


def check_registry_health() -> "HealthCheckResult":
    """Check health of the daemon registry.

    Performs comprehensive validation of the daemon registry:
        - Registry structure validation
        - Dependency graph validation
        - Runner function availability
        - Deprecated daemon tracking

    Returns:
        HealthCheckResult with registry status and details.

    Example:
        >>> from app.coordination.daemon_registry import check_registry_health
        >>> result = check_registry_health()
        >>> if not result.healthy:
        ...     print(f"Registry issues: {result.message}")

    December 2025: Added for DaemonManager integration.
    """
    # Import from contracts (zero dependencies)
    from app.coordination.contracts import HealthCheckResult, CoordinatorStatus

    # Run validation
    validation_errors = validate_registry()

    # Count categories
    categories = get_categories()
    category_counts = {cat: len(get_daemons_by_category(cat)) for cat in categories}

    # Count deprecated
    deprecated_daemons = get_deprecated_daemons()
    deprecated_count = len(deprecated_daemons)

    # Check for missing runner functions
    # CIRCULAR DEPENDENCY NOTE (Dec 2025):
    # This lazy import of daemon_runners is SAFE - same reasoning as in validate_registry().
    # check_registry_health() is called at runtime for health monitoring, not at import time.
    missing_runners: list[str] = []
    try:
        from app.coordination import daemon_runners

        for daemon_type, spec in DAEMON_REGISTRY.items():
            if not hasattr(daemon_runners, spec.runner_name):
                missing_runners.append(f"{daemon_type.name}: {spec.runner_name}")
    except ImportError:
        missing_runners.append("daemon_runners module not importable")

    # Check dependency graph for cycles
    cycle_issues: list[str] = []
    visited: set[DaemonType] = set()
    rec_stack: set[DaemonType] = set()

    def has_cycle(daemon_type: DaemonType) -> bool:
        """DFS to detect cycles in dependency graph."""
        visited.add(daemon_type)
        rec_stack.add(daemon_type)

        spec = DAEMON_REGISTRY.get(daemon_type)
        if spec:
            for dep in spec.depends_on:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    cycle_issues.append(f"Cycle detected: {daemon_type.name} -> {dep.name}")
                    return True

        rec_stack.remove(daemon_type)
        return False

    for daemon_type in DAEMON_REGISTRY:
        if daemon_type not in visited:
            has_cycle(daemon_type)

    # Determine overall health
    is_healthy = (
        len(validation_errors) == 0
        and len(missing_runners) == 0
        and len(cycle_issues) == 0
    )

    if not is_healthy:
        status = CoordinatorStatus.ERROR if validation_errors else CoordinatorStatus.DEGRADED
        if validation_errors:
            message = f"Registry validation failed: {len(validation_errors)} error(s)"
        elif missing_runners:
            message = f"Missing runners: {len(missing_runners)}"
        else:
            message = f"Dependency cycle detected: {len(cycle_issues)}"
    else:
        status = CoordinatorStatus.RUNNING
        message = f"Registry healthy: {len(DAEMON_REGISTRY)} daemons, {len(categories)} categories"

    return HealthCheckResult(
        healthy=is_healthy,
        status=status,
        message=message,
        details={
            "total_daemons": len(DAEMON_REGISTRY),
            "categories": category_counts,
            "deprecated_count": deprecated_count,
            "deprecated_daemons": [d[0].name for d in deprecated_daemons[:5]],  # Limit
            "validation_errors": validation_errors[:5] if validation_errors else [],
            "missing_runners": missing_runners[:5] if missing_runners else [],
            "cycle_issues": cycle_issues[:5] if cycle_issues else [],
        },
    )


__all__ = [
    # Core data structures
    "DaemonSpec",
    "DAEMON_REGISTRY",
    "DEPRECATED_TYPES",
    # Query functions
    "get_daemons_by_category",
    "get_categories",
    "get_deprecated_daemons",
    "get_deprecated_types",
    "is_daemon_deprecated",
    # Validation
    "validate_registry",
    "validate_registry_or_raise",
    "check_registry_health",
]
