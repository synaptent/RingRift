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
    # Ephemeral sync for Vast.ai (Phase 4)
    DaemonType.EPHEMERAL_SYNC: DaemonSpec(
        runner_name="create_ephemeral_sync",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE),
        category="sync",
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
    ),
    DaemonType.SYSTEM_HEALTH_MONITOR: DaemonSpec(
        runner_name="create_system_health_monitor",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.NODE_HEALTH_MONITOR),
        category="health",
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
        depends_on=(DaemonType.EVENT_ROUTER,),
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
        deprecated_message="Use create_lambda_idle_daemon() from unified_idle_shutdown_daemon. Removal: Q2 2026.",
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
    # =========================================================================
    # Data Consolidation
    # =========================================================================
    DaemonType.DATA_CONSOLIDATION: DaemonSpec(
        runner_name="create_data_consolidation",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE),
        category="pipeline",
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


def validate_registry() -> list[str]:
    """Validate the daemon registry for common issues.

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    # Check for missing daemon_runners functions
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
