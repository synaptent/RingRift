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
        depends_on: Hard dependencies that MUST be running first.
            Startup fails if these are not ready within timeout.
        soft_depends_on: Soft dependencies that SHOULD be running.
            Jan 2, 2026: Startup continues with warning if these are not ready.
            Useful for optional integrations or degraded operation mode.
        startup_mode: How to handle startup when dependencies are missing.
            - "strict": Fail if any dependency (hard or soft) is missing
            - "degraded": Start in degraded mode if soft deps missing
            - "local": Start in local-only mode (skip cluster deps)
        health_check_interval: Custom health check interval (None = use defaults).
        auto_restart: Whether to auto-restart on failure.
        max_restarts: Maximum restart attempts.
        category: Human-readable category for documentation.
        deprecated: Whether this daemon is deprecated (December 2025).
        deprecated_message: Migration guidance for deprecated daemons.
    """

    runner_name: str
    depends_on: tuple[DaemonType, ...] = field(default_factory=tuple)
    soft_depends_on: tuple[DaemonType, ...] = field(default_factory=tuple)
    startup_mode: str = "degraded"  # Default to degraded for resilience
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
        health_check_interval=60.0,  # Dec 2025: Critical for data movement
    ),
    # Config sync daemon (January 2026) - auto-sync distributed_hosts.yaml
    # Coordinator detects mtime changes and emits CONFIG_UPDATED event
    # Workers pull config via rsync when they receive the event
    DaemonType.CONFIG_SYNC: DaemonSpec(
        runner_name="create_config_sync",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
        health_check_interval=60.0,
        auto_restart=True,  # Critical for cluster config consistency
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
    # OWC external drive import (December 29, 2025) - periodic import from OWC drive
    # Imports training data from external archive on mac-studio for underserved configs
    # December 30, 2025: Removed DATA_PIPELINE dependency - OWC_IMPORT doesn't need pipeline
    # to run, it only imports files. Events are optional enhancement, not core functionality.
    DaemonType.OWC_IMPORT: DaemonSpec(
        runner_name="create_owc_import",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
        health_check_interval=1800.0,  # 30 min - runs hourly import cycles
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
        health_check_interval=30.0,  # Dec 2025: Monitors other daemons - fast checks
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
        health_check_interval=60.0,  # Dec 2025: Cluster health monitoring
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
    # Jan 2, 2026: DATA_PIPELINE uses soft deps for AUTO_SYNC to enable local-only mode
    # when cluster is unavailable. Starts in degraded mode if sync daemons are down.
    DaemonType.DATA_PIPELINE: DaemonSpec(
        runner_name="create_data_pipeline",
        depends_on=(DaemonType.EVENT_ROUTER,),
        soft_depends_on=(DaemonType.AUTO_SYNC,),
        startup_mode="degraded",  # Allow local-only operation
        category="pipeline",
        health_check_interval=30.0,  # Dec 2025: Critical for data flow
    ),
    # Jan 3, 2026: Deprecated - module archived, functionality in p2p_orchestrator.py
    DaemonType.CONTINUOUS_TRAINING_LOOP: DaemonSpec(
        runner_name="create_continuous_training_loop",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="pipeline",
        deprecated=True,
        deprecated_message="Module archived. Functionality now in p2p_orchestrator.py. Removal: Q2 2026.",
    ),
    DaemonType.SELFPLAY_COORDINATOR: DaemonSpec(
        runner_name="create_selfplay_coordinator",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="pipeline",
        health_check_interval=60.0,  # Dec 2025: Coordinates selfplay jobs
    ),
    # Jan 2, 2026: TRAINING_TRIGGER uses soft deps for cluster daemons
    # to enable local-only training when cluster sync is unavailable.
    # Jan 10, 2026: Moved AUTO_EXPORT to soft dependency because on coordinator
    # nodes AUTO_EXPORT is filtered out (CPU-bound), but TRAINING_TRIGGER can
    # still dispatch training based on NPZ files from cluster nodes.
    DaemonType.TRAINING_TRIGGER: DaemonSpec(
        runner_name="create_training_trigger",
        depends_on=(DaemonType.EVENT_ROUTER,),
        soft_depends_on=(DaemonType.AUTO_EXPORT, DaemonType.AUTO_SYNC, DaemonType.DATA_PIPELINE),
        startup_mode="degraded",  # Enable local-only mode
        category="pipeline",
        health_check_interval=120.0,  # Dec 2025: Training jobs are long-running
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
        health_check_interval=120.0,  # Dec 2025: Gauntlet evaluations are slow
    ),
    DaemonType.AUTO_PROMOTION: DaemonSpec(
        runner_name="create_auto_promotion",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.EVALUATION),
        category="evaluation",
    ),
    # Reanalysis daemon (January 27, 2026 - Phase 2.1)
    # Re-evaluates historical games with improved models
    # Subscribes to MODEL_PROMOTED, triggers reanalysis on Elo improvements
    DaemonType.REANALYSIS: DaemonSpec(
        runner_name="create_reanalysis",
        depends_on=(DaemonType.EVENT_ROUTER,),
        soft_depends_on=(DaemonType.AUTO_PROMOTION,),
        category="training",
        health_check_interval=300.0,  # 5 min - matches daemon cycle
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
    # Sprint 15 (Jan 3, 2026): Backlog evaluation daemon
    # Discovers OWC models, queues for evaluation with rate limiting
    DaemonType.BACKLOG_EVALUATION: DaemonSpec(
        runner_name="create_backlog_evaluation",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="evaluation",
        health_check_interval=300.0,  # 5 minutes
        auto_restart=True,
    ),
    # Sprint 17.9 (Jan 9, 2026): Comprehensive model scan daemon
    # Scans all model sources, queues multi-harness evaluations
    DaemonType.COMPREHENSIVE_MODEL_SCAN: DaemonSpec(
        runner_name="create_comprehensive_model_scan",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="evaluation",
        health_check_interval=300.0,  # 5 minutes - scans are periodic
        auto_restart=True,
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
        health_check_interval=120.0,  # Dec 2025: Resource management
    ),
    DaemonType.NODE_RECOVERY: DaemonSpec(
        runner_name="create_node_recovery",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="resource",
        health_check_interval=120.0,  # Dec 2025: Recovery takes time
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
        health_check_interval=120.0,  # Dec 2025: Moderate priority
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
        health_check_interval=60.0,  # Dec 2025: Critical for training signals
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
    # Jan 3, 2026: Deprecated - standalone script at scripts/distillation_daemon.py
    DaemonType.DISTILLATION: DaemonSpec(
        runner_name="create_distillation",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="misc",
        deprecated=True,
        deprecated_message="Standalone script. Use scripts/distillation_daemon.py directly.",
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
    # Cluster Consolidation (January 2026) - Pull games from cluster nodes
    # CRITICAL: Bridges distributed selfplay with training pipeline
    # Note: Daemon internally checks coordinator_only flag from config
    DaemonType.CLUSTER_CONSOLIDATION: DaemonSpec(
        runner_name="create_cluster_consolidation",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE),
        category="sync",
        health_check_interval=300.0,  # 5 minutes (matches sync cycle)
    ),
    # Comprehensive Consolidation (January 2026) - Scheduled full sweep
    # Scans ALL databases (local, OWC, S3, P2P) and consolidates into canonical DBs
    # Unlike event-driven consolidation, runs on schedule to catch missed data
    DaemonType.COMPREHENSIVE_CONSOLIDATION: DaemonSpec(
        runner_name="create_comprehensive_consolidation",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE),
        soft_depends_on=(DaemonType.OWC_IMPORT, DaemonType.S3_BACKUP),
        category="pipeline",
        health_check_interval=1800.0,  # 30 minutes (matches consolidation cycle)
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
    # Voter health monitor (December 30, 2025) - continuous voter probing
    # Multi-transport probing: P2P HTTP → Tailscale → SSH
    # Emits VOTER_OFFLINE, VOTER_ONLINE, QUORUM_LOST/RESTORED/AT_RISK
    DaemonType.VOTER_HEALTH_MONITOR: DaemonSpec(
        runner_name="create_voter_health_monitor",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="health",
        health_check_interval=60.0,  # 1 min - fast voter health tracking
        auto_restart=True,
        max_restarts=10,  # Critical for quorum protection
    ),
    # Memory monitor (December 30, 2025) - prevents OOM crashes
    # Monitors GPU VRAM and process RSS, emits MEMORY_PRESSURE events
    DaemonType.MEMORY_MONITOR: DaemonSpec(
        runner_name="create_memory_monitor",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="health",
        health_check_interval=60.0,  # 1 min - critical for preventing OOM
        auto_restart=True,
        max_restarts=10,
    ),
    # Stale fallback (December 30, 2025) - graceful degradation
    # Uses older models when sync fails to maintain selfplay continuity
    DaemonType.STALE_FALLBACK: DaemonSpec(
        runner_name="create_stale_fallback",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.AUTO_SYNC),
        category="sync",
        health_check_interval=300.0,  # 5 min
        auto_restart=True,
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
    # =========================================================================
    # Connectivity Recovery Coordinator (December 29, 2025)
    # Unified event-driven connectivity recovery
    # Bridges TailscaleHealthDaemon, NodeAvailabilityDaemon, P2P orchestrator
    # =========================================================================
    DaemonType.CONNECTIVITY_RECOVERY: DaemonSpec(
        runner_name="create_connectivity_recovery",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.TAILSCALE_HEALTH),
        category="recovery",
        health_check_interval=60.0,  # 1 min - monitor recovery state
        auto_restart=True,
        max_restarts=10,
    ),
    # =========================================================================
    # NNUE Automatic Training (December 29, 2025)
    # Trains NNUE models when game thresholds are met
    # Subscribes to: NEW_GAMES_AVAILABLE, CONSOLIDATION_COMPLETE, DATA_SYNC_COMPLETED
    # =========================================================================
    DaemonType.NNUE_TRAINING: DaemonSpec(
        runner_name="create_nnue_training",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE),
        category="pipeline",
        health_check_interval=3600.0,  # 1 hour - training jobs are long-running
        auto_restart=True,
        max_restarts=5,
    ),
    # =========================================================================
    # Architecture Feedback Controller (Dec 29, 2025)
    # Bridges evaluation results to selfplay allocation by tracking architecture
    # performance. Enforces 10% minimum allocation per architecture.
    # Subscribes to: EVALUATION_COMPLETED, TRAINING_COMPLETED
    # Emits: ARCHITECTURE_WEIGHTS_UPDATED
    # =========================================================================
    DaemonType.ARCHITECTURE_FEEDBACK: DaemonSpec(
        runner_name="create_architecture_feedback",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="feedback",
        health_check_interval=300.0,  # 5 minutes
        auto_restart=True,
        max_restarts=3,
    ),
    # =========================================================================
    # Parity Validation Daemon (December 30, 2025)
    # Runs on coordinator (has Node.js) to validate TS/Python parity for
    # canonical databases. Cluster nodes lack npx, so they generate databases
    # with "pending_gate" status. This daemon validates them and stores TS
    # reference hashes, enabling hash-based validation on cluster nodes.
    # Subscribes to: DATA_SYNC_COMPLETED
    # Emits: PARITY_VALIDATION_COMPLETED
    # =========================================================================
    DaemonType.PARITY_VALIDATION: DaemonSpec(
        runner_name="create_parity_validation",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE),
        category="pipeline",
        health_check_interval=1800.0,  # 30 minutes - matches validation cycle
        auto_restart=True,
        max_restarts=5,
    ),
    # =========================================================================
    # Jan 3, 2026: Added missing daemon specs
    # These daemon types existed but had no registry entries
    # =========================================================================
    DaemonType.CLUSTER_UTILIZATION_WATCHDOG: DaemonSpec(
        runner_name="create_cluster_utilization_watchdog",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="health",
        health_check_interval=300.0,  # 5 minutes
    ),
    DaemonType.ELO_PROGRESS: DaemonSpec(
        runner_name="create_elo_progress",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="autonomous",
        health_check_interval=600.0,  # 10 minutes - tracks Elo improvement
    ),
    DaemonType.UNIFIED_BACKUP: DaemonSpec(
        runner_name="create_unified_backup",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="distribution",
        health_check_interval=3600.0,  # 1 hour - backup operations are slow
    ),
    DaemonType.S3_PUSH: DaemonSpec(
        runner_name="create_s3_push",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="distribution",
        health_check_interval=1800.0,  # 30 minutes
    ),
    # =========================================================================
    # Jan 3, 2026: Added missing daemon specs for data availability infrastructure
    # These daemon types were declared in DaemonType enum but missing from registry
    # =========================================================================
    DaemonType.DUAL_BACKUP: DaemonSpec(
        runner_name="create_dual_backup",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
        health_check_interval=3600.0,  # 1 hour - backup operations are slow
    ),
    DaemonType.OWC_PUSH: DaemonSpec(
        runner_name="create_owc_push",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
        health_check_interval=3600.0,  # 1 hour - push operations are slow
    ),
    DaemonType.S3_IMPORT: DaemonSpec(
        runner_name="create_s3_import",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
        health_check_interval=1800.0,  # 30 minutes
    ),
    DaemonType.UNIFIED_DATA_CATALOG: DaemonSpec(
        runner_name="create_unified_data_catalog",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
        health_check_interval=600.0,  # 10 minutes
        deprecated=True,
        deprecated_message="Not yet implemented - placeholder for future unified data catalog API",
    ),
    DaemonType.NODE_DATA_AGENT: DaemonSpec(
        runner_name="create_node_data_agent",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="distribution",
        health_check_interval=300.0,  # 5 minutes
        deprecated=True,
        deprecated_message="Not yet implemented - placeholder for per-node data agent",
    ),
    DaemonType.UNIFIED_DATA_SYNC_ORCHESTRATOR: DaemonSpec(
        runner_name="create_unified_data_sync_orchestrator",
        depends_on=(DaemonType.EVENT_ROUTER, DaemonType.AUTO_SYNC),
        category="sync",
        health_check_interval=300.0,  # 5 minutes
    ),
    # Jan 4, 2026: P2P Cluster Resilience - Phase 3
    DaemonType.UNDERUTILIZATION_RECOVERY: DaemonSpec(
        runner_name="create_underutilization_recovery",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="autonomous",
        health_check_interval=60.0,
    ),
    # Jan 4, 2026: P2P Cluster Resilience - Phase 4
    DaemonType.FAST_FAILURE_DETECTOR: DaemonSpec(
        runner_name="create_fast_failure_detector",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="autonomous",
        health_check_interval=60.0,
    ),
    # Jan 2026: Socket leak detection and recovery
    DaemonType.SOCKET_LEAK_RECOVERY: DaemonSpec(
        runner_name="create_socket_leak_recovery",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="autonomous",
        health_check_interval=120.0,
    ),
    # Jan 2026: Training data recovery after corruption
    DaemonType.TRAINING_DATA_RECOVERY: DaemonSpec(
        runner_name="create_training_data_recovery",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="autonomous",
        health_check_interval=120.0,
    ),
    # Jan 4, 2026: Training watchdog for stuck processes
    DaemonType.TRAINING_WATCHDOG: DaemonSpec(
        runner_name="create_training_watchdog",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="autonomous",
        health_check_interval=60.0,
    ),
    # Jan 6, 2026: Export watchdog for stuck export scripts
    DaemonType.EXPORT_WATCHDOG: DaemonSpec(
        runner_name="create_export_watchdog",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="autonomous",
        health_check_interval=60.0,
    ),
    # Model evaluation and promotion daemons
    DaemonType.OWC_MODEL_IMPORT: DaemonSpec(
        runner_name="create_owc_model_import",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
        health_check_interval=300.0,
    ),
    DaemonType.UNEVALUATED_MODEL_SCANNER: DaemonSpec(
        runner_name="create_unevaluated_model_scanner",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="evaluation",
        health_check_interval=300.0,
    ),
    DaemonType.STALE_EVALUATION: DaemonSpec(
        runner_name="create_stale_evaluation",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="evaluation",
        health_check_interval=300.0,
    ),
    # =========================================================================
    # Consolidated Daemons (January 2026)
    # These replace multiple deprecated daemons with unified implementations
    # =========================================================================
    DaemonType.CONFIG_VALIDATOR: DaemonSpec(
        runner_name="create_config_validator",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="misc",
        health_check_interval=300.0,
    ),
    DaemonType.ONLINE_MERGE: DaemonSpec(
        runner_name="create_online_merge",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="training",
        health_check_interval=120.0,
    ),
    DaemonType.OWC_SYNC_MANAGER: DaemonSpec(
        runner_name="create_owc_sync_manager",
        depends_on=(DaemonType.EVENT_ROUTER,),
        soft_depends_on=(DaemonType.AUTO_SYNC,),
        category="sync",
        health_check_interval=300.0,
    ),
    DaemonType.PRODUCTION_GAME_IMPORT: DaemonSpec(
        runner_name="create_production_game_import",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
        health_check_interval=300.0,
    ),
    DaemonType.S3_SYNC: DaemonSpec(
        runner_name="create_s3_sync",
        depends_on=(DaemonType.EVENT_ROUTER,),
        soft_depends_on=(DaemonType.AUTO_SYNC,),
        category="sync",
        health_check_interval=120.0,
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
