"""Daemon runner functions extracted from DaemonManager.

December 2025: Extracted from DaemonManager to reduce module size and improve testability.
December 30, 2025: Refactored to registry-driven factory pattern (Phase 1 consolidation).

This module provides async runner functions for all 89 daemon types in the system.
Runners are now defined declaratively via RUNNER_SPECS registry and instantiated
via the generic _create_runner_from_spec() factory function.

Runner Specification Pattern:
    RUNNER_SPECS["auto_sync"] = RunnerSpec(
        module="app.coordination.auto_sync_daemon",
        class_name="AutoSyncDaemon",
    )

Instantiation Styles:
    - DIRECT: DaemonClass()
    - SINGLETON: DaemonClass.get_instance() or factory function
    - FACTORY: factory_function() returns daemon instance
    - WITH_CONFIG: DaemonClass(config=ConfigClass.from_env())
    - ASYNC_FACTORY: await factory_function() returns daemon instance

Wait Styles:
    - DAEMON: await _wait_for_daemon(daemon)
    - FOREVER_LOOP: while True: await asyncio.sleep(interval)
    - RUN_FOREVER: await daemon.run_forever()
    - NONE: No waiting (initialize and return)

Key functions:
- get_runner(DaemonType) -> Callable: Get runner function for a daemon type
- get_all_runners() -> dict: Get full registry mapping names to runners
- _create_runner_from_spec(name) -> None: Generic runner factory

The registry (_RUNNER_REGISTRY) is built lazily on first access to avoid
import-time circular dependencies with daemon_types.py.

Usage:
    from app.coordination.daemon_runners import get_runner
    from app.coordination.daemon_types import DaemonType

    runner = get_runner(DaemonType.AUTO_SYNC)
    await runner()  # Starts AutoSyncDaemon and blocks until stopped
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from app.coordination.daemon_types import DaemonType

logger = logging.getLogger(__name__)


# =============================================================================
# Runner Specification Infrastructure (December 30, 2025)
# =============================================================================


class InstantiationStyle(Enum):
    """How to instantiate the daemon class."""

    DIRECT = "direct"  # DaemonClass()
    SINGLETON = "singleton"  # DaemonClass.get_instance()
    FACTORY = "factory"  # factory_function() returns daemon
    WITH_CONFIG = "with_config"  # DaemonClass(config=ConfigClass.from_env())
    ASYNC_FACTORY = "async_factory"  # await factory_function()
    CUSTOM = "custom"  # Use custom runner function (create_<daemon_name>())


class WaitStyle(Enum):
    """How to wait for daemon completion."""

    DAEMON = "daemon"  # await _wait_for_daemon(daemon)
    FOREVER_LOOP = "forever_loop"  # while True: await asyncio.sleep(interval)
    RUN_FOREVER = "run_forever"  # await daemon.run_forever()
    NONE = "none"  # No waiting (initialize only)
    CUSTOM = "custom"  # Special handling (keep legacy function)


class StartMethod(Enum):
    """How to start the daemon."""

    ASYNC_START = "async_start"  # await daemon.start()
    SYNC_START = "sync_start"  # daemon.start() (synchronous)
    INITIALIZE = "initialize"  # await daemon.initialize()
    START_SERVER = "start_server"  # await daemon.start_server()
    NONE = "none"  # No start method needed


@dataclass
class RunnerSpec:
    """Specification for a daemon runner.

    Defines how to import, instantiate, start, and wait for a daemon.
    Used by _create_runner_from_spec() to generate runners dynamically.

    Example:
        RunnerSpec(
            module="app.coordination.auto_sync_daemon",
            class_name="AutoSyncDaemon",
        )

        # Results in:
        from app.coordination.auto_sync_daemon import AutoSyncDaemon
        daemon = AutoSyncDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    """

    module: str  # Import path, e.g., "app.coordination.auto_sync_daemon"
    class_name: str  # Class or function name to import

    # Instantiation
    style: InstantiationStyle = InstantiationStyle.DIRECT
    config_class: str | None = None  # For WITH_CONFIG: ConfigClass name
    factory_func: str | None = None  # For FACTORY/SINGLETON: function name

    # Start
    start_method: StartMethod = StartMethod.ASYNC_START

    # Wait
    wait: WaitStyle = WaitStyle.DAEMON
    wait_interval: float = 60.0  # For FOREVER_LOOP

    # Metadata
    deprecated: bool = False
    deprecation_message: str = ""
    notes: str = ""

    # For daemons that need extra imports (e.g., multiple classes from same module)
    extra_imports: list[str] = field(default_factory=list)


# =============================================================================
# Runner Specs Registry
# =============================================================================

# Maps runner name to its specification
# Runner names match DaemonType enum values (lowercase with underscores)
RUNNER_SPECS: dict[str, RunnerSpec] = {
    # --- Sync Daemons ---
    "sync_coordinator": RunnerSpec(
        module="app.distributed.sync_coordinator",
        class_name="SyncCoordinator",
        style=InstantiationStyle.SINGLETON,
        deprecated=True,
        deprecation_message="Use DaemonType.AUTO_SYNC instead. Removal scheduled for Q2 2026.",
    ),
    "high_quality_sync": RunnerSpec(
        module="app.coordination.training_freshness",
        class_name="HighQualityDataSyncWatcher",
    ),
    "elo_sync": RunnerSpec(
        module="app.tournament.elo_sync_manager",
        class_name="EloSyncManager",
        start_method=StartMethod.INITIALIZE,
        wait=WaitStyle.NONE,  # Manager starts its own loop after initialize()
        notes="Calls initialize() then start() - manager handles its own loop",
    ),
    "auto_sync": RunnerSpec(
        module="app.coordination.auto_sync_daemon",
        class_name="AutoSyncDaemon",
    ),
    "config_sync": RunnerSpec(
        module="app.coordination.config_sync_daemon",
        class_name="ConfigSyncDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_config_sync_daemon",
    ),
    "config_validator": RunnerSpec(
        module="app.coordination.config_validator_daemon",
        class_name="ConfigValidatorDaemon",
    ),
    "training_node_watcher": RunnerSpec(
        module="app.coordination.training_activity_daemon",
        class_name="TrainingActivityDaemon",
        style=InstantiationStyle.WITH_CONFIG,
        config_class="TrainingActivityConfig",
    ),
    "training_data_sync": RunnerSpec(
        module="app.coordination.training_data_sync_daemon",
        class_name="TrainingDataSyncDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_training_data_sync_daemon",
        wait=WaitStyle.CUSTOM,  # Special: uses _running attribute check
    ),
    "owc_import": RunnerSpec(
        module="app.coordination.owc_import_daemon",
        class_name="OWCImportDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_owc_import_daemon",
    ),
    "ephemeral_sync": RunnerSpec(
        module="app.coordination.auto_sync_daemon",
        class_name="AutoSyncDaemon",
        style=InstantiationStyle.WITH_CONFIG,
        config_class="AutoSyncConfig",
        deprecated=True,
        deprecation_message="Use AutoSyncDaemon(strategy='ephemeral') instead. Removal scheduled for Q2 2026.",
        wait=WaitStyle.CUSTOM,  # Sets strategy after config creation
    ),
    "gossip_sync": RunnerSpec(
        module="app.distributed.gossip_sync",
        class_name="GossipSyncDaemon",
    ),
    # --- Event Processing Daemons ---
    "event_router": RunnerSpec(
        module="app.coordination.event_router",
        class_name="UnifiedEventRouter",
        style=InstantiationStyle.FACTORY,
        factory_func="get_router",
        wait=WaitStyle.CUSTOM,  # Has validation guards
    ),
    # Jan 3, 2026: Fixed module path - was pointing to event_router
    "cross_process_poller": RunnerSpec(
        module="app.coordination.cross_process_events",
        class_name="CrossProcessEventPoller",
    ),
    "dlq_retry": RunnerSpec(
        module="app.coordination.dead_letter_queue",
        class_name="DLQRetryDaemon",
    ),
    # --- Health & Monitoring Daemons ---
    "health_check": RunnerSpec(
        module="app.coordination.health_check_orchestrator",
        class_name="HealthCheckOrchestrator",
        deprecated=True,
        deprecation_message="Use DaemonType.NODE_HEALTH_MONITOR instead. Removal scheduled for Q2 2026.",
    ),
    "queue_monitor": RunnerSpec(
        module="app.coordination.queue_monitor",
        class_name="QueueMonitor",
    ),
    "daemon_watchdog": RunnerSpec(
        module="app.coordination.daemon_watchdog",
        class_name="DaemonWatchdog",
        style=InstantiationStyle.ASYNC_FACTORY,
        factory_func="start_watchdog",
        start_method=StartMethod.NONE,  # start_watchdog() handles start
    ),
    "node_health_monitor": RunnerSpec(
        module="app.coordination.health_check_orchestrator",
        class_name="HealthCheckOrchestrator",
        deprecated=True,
        deprecation_message="Use HealthCheckOrchestrator (via DaemonType.HEALTH_SERVER) instead. Removal scheduled for Q2 2026.",
    ),
    "system_health_monitor": RunnerSpec(
        module="app.coordination.unified_health_manager",
        class_name="UnifiedHealthManager",
        deprecated=True,
        deprecation_message="Use unified_health_manager.get_system_health_score() instead. Removal scheduled for Q2 2026.",
    ),
    "health_server": RunnerSpec(
        module="app.coordination.daemon_manager",
        class_name="DaemonManager",
        style=InstantiationStyle.FACTORY,
        factory_func="get_daemon_manager",
        wait=WaitStyle.CUSTOM,  # Calls dm._create_health_server()
    ),
    "quality_monitor": RunnerSpec(
        module="app.coordination.quality_monitor_daemon",
        class_name="QualityMonitorDaemon",
    ),
    "model_performance_watchdog": RunnerSpec(
        module="app.coordination.model_performance_watchdog",
        class_name="ModelPerformanceWatchdog",
    ),
    "cluster_monitor": RunnerSpec(
        module="app.coordination.cluster_status_monitor",
        class_name="ClusterMonitor",
        wait=WaitStyle.RUN_FOREVER,
    ),
    "cluster_watchdog": RunnerSpec(
        module="app.coordination.cluster_watchdog_daemon",
        class_name="ClusterWatchdogDaemon",
    ),
    "coordinator_health_monitor": RunnerSpec(
        module="app.coordination.coordinator_health_monitor_daemon",
        class_name="CoordinatorHealthMonitorDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_coordinator_health_monitor",
    ),
    "work_queue_monitor": RunnerSpec(
        module="app.coordination.work_queue_monitor_daemon",
        class_name="WorkQueueMonitorDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_work_queue_monitor",
    ),
    # --- Training & Pipeline Daemons ---
    "data_pipeline": RunnerSpec(
        module="app.coordination.data_pipeline_orchestrator",
        class_name="DataPipelineOrchestrator",
        style=InstantiationStyle.FACTORY,
        factory_func="get_pipeline_orchestrator",
    ),
    "continuous_training_loop": RunnerSpec(
        module="app.coordination.continuous_training_loop",
        class_name="ContinuousTrainingLoop",
    ),
    "selfplay_coordinator": RunnerSpec(
        module="app.coordination.selfplay_scheduler",
        class_name="SelfplayScheduler",
        style=InstantiationStyle.FACTORY,
        factory_func="get_selfplay_scheduler",
        start_method=StartMethod.NONE,  # SelfplayScheduler is utility, not daemon
        wait=WaitStyle.FOREVER_LOOP,
        wait_interval=3600.0,  # 1 hour
    ),
    "training_trigger": RunnerSpec(
        module="app.coordination.training_trigger_daemon",
        class_name="TrainingTriggerDaemon",
    ),
    "auto_export": RunnerSpec(
        module="app.coordination.auto_export_daemon",
        class_name="AutoExportDaemon",
    ),
    "tournament_daemon": RunnerSpec(
        module="app.coordination.tournament_daemon",
        class_name="TournamentDaemon",
    ),
    "nnue_training": RunnerSpec(
        module="app.coordination.nnue_training_daemon",
        class_name="NNUETrainingDaemon",
        style=InstantiationStyle.SINGLETON,
    ),
    "architecture_feedback": RunnerSpec(
        module="app.coordination.architecture_feedback_controller",
        class_name="ArchitectureFeedbackController",
        style=InstantiationStyle.SINGLETON,
    ),
    # --- Evaluation & Promotion Daemons ---
    "evaluation": RunnerSpec(
        module="app.coordination.evaluation_daemon",
        class_name="EvaluationDaemon",
    ),
    # January 3, 2026: Scans for models without Elo ratings and queues them for evaluation
    "unevaluated_model_scanner": RunnerSpec(
        module="app.coordination.unevaluated_model_scanner_daemon",
        class_name="UnevaluatedModelScannerDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_unevaluated_model_scanner_daemon",
    ),
    # January 3, 2026: Imports model files from OWC external drive for Elo evaluation
    "owc_model_import": RunnerSpec(
        module="app.coordination.owc_model_import_daemon",
        class_name="OWCModelImportDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_owc_model_import_daemon",
    ),
    # January 3, 2026: Re-evaluates models with stale Elo ratings (>30 days old)
    "stale_evaluation": RunnerSpec(
        module="app.coordination.stale_evaluation_daemon",
        class_name="StaleEvaluationDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_stale_evaluation_daemon",
    ),
    "auto_promotion": RunnerSpec(
        module="app.coordination.auto_promotion_daemon",
        class_name="AutoPromotionDaemon",
    ),
    # January 27, 2026 (Phase 2.1): Reanalysis daemon
    "reanalysis": RunnerSpec(
        module="app.coordination.reanalysis_daemon",
        class_name="ReanalysisDaemon",
    ),
    "unified_promotion": RunnerSpec(
        module="app.training.promotion_controller",
        class_name="PromotionController",
    ),
    "gauntlet_feedback": RunnerSpec(
        module="app.coordination.gauntlet_feedback_controller",
        class_name="GauntletFeedbackController",
    ),
    # --- Distribution Daemons ---
    "model_sync": RunnerSpec(
        module="app.coordination.unified_distribution_daemon",
        class_name="UnifiedDistributionDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="create_model_distribution_daemon",
    ),
    "model_distribution": RunnerSpec(
        module="app.coordination.unified_distribution_daemon",
        class_name="UnifiedDistributionDaemon",
    ),
    "npz_distribution": RunnerSpec(
        module="app.coordination.unified_distribution_daemon",
        class_name="UnifiedDistributionDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="create_npz_distribution_daemon",
    ),
    # Jan 3, 2026: Changed to CUSTOM runner because SyncCoordinator has
    # start_data_server() not start_server(). The create_data_server() function
    # correctly calls start_data_server() on the singleton instance.
    "data_server": RunnerSpec(
        module="app.coordination.daemon_runners",
        class_name="",  # Not used for CUSTOM
        style=InstantiationStyle.CUSTOM,
    ),
    # --- Replication Daemons ---
    "replication_monitor": RunnerSpec(
        module="app.coordination.unified_replication_daemon",
        class_name="UnifiedReplicationDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="create_replication_monitor",
    ),
    "replication_repair": RunnerSpec(
        module="app.coordination.unified_replication_daemon",
        class_name="UnifiedReplicationDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="create_replication_repair_daemon",
    ),
    # --- Resource Management Daemons ---
    "idle_resource": RunnerSpec(
        module="app.coordination.idle_resource_daemon",
        class_name="IdleResourceDaemon",
    ),
    "cluster_utilization_watchdog": RunnerSpec(
        module="app.coordination.cluster_utilization_watchdog",
        class_name="ClusterUtilizationWatchdog",
        style=InstantiationStyle.SINGLETON,
        notes="Monitors GPU utilization across cluster, emits CLUSTER_UNDERUTILIZED events",
    ),
    "node_recovery": RunnerSpec(
        module="app.coordination.node_recovery_daemon",
        class_name="NodeRecoveryDaemon",
    ),
    "resource_optimizer": RunnerSpec(
        module="app.coordination.resource_optimizer",
        class_name="ResourceOptimizer",
    ),
    "utilization_optimizer": RunnerSpec(
        module="app.coordination.utilization_optimizer",
        class_name="UtilizationOptimizer",
        start_method=StartMethod.NONE,
        wait=WaitStyle.CUSTOM,  # Wraps in periodic optimize_cluster() loop
    ),
    "adaptive_resources": RunnerSpec(
        module="app.coordination.adaptive_resource_manager",
        class_name="AdaptiveResourceManager",
    ),
    # --- Provider-Specific Daemons ---
    "lambda_idle": RunnerSpec(
        module="app.coordination.unified_idle_shutdown_daemon",
        class_name="UnifiedIdleShutdownDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="create_lambda_idle_daemon",
        deprecated=True,
        deprecation_message="Lambda Labs account was terminated Dec 2025. Use DaemonType.VAST_IDLE instead. Removal scheduled for Q2 2026.",
        wait=WaitStyle.CUSTOM,  # Returns None, skips start
    ),
    "vast_idle": RunnerSpec(
        module="app.coordination.unified_idle_shutdown_daemon",
        class_name="UnifiedIdleShutdownDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="create_vast_idle_daemon",
    ),
    "multi_provider": RunnerSpec(
        module="app.coordination.multi_provider_orchestrator",
        class_name="MultiProviderOrchestrator",
    ),
    # --- Queue & Job Daemons ---
    "queue_populator": RunnerSpec(
        module="app.coordination.unified_queue_populator",
        class_name="UnifiedQueuePopulatorDaemon",
    ),
    "job_scheduler": RunnerSpec(
        module="app.coordination.job_scheduler",
        class_name="PriorityJobScheduler",
        style=InstantiationStyle.FACTORY,
        factory_func="get_scheduler",
        start_method=StartMethod.NONE,  # Utility class, no lifecycle
        wait=WaitStyle.NONE,
    ),
    # --- Feedback & Curriculum Daemons ---
    "feedback_loop": RunnerSpec(
        module="app.coordination.feedback_loop_controller",
        class_name="FeedbackLoopController",
    ),
    "curriculum_integration": RunnerSpec(
        module="app.coordination.curriculum_integration",
        class_name="MomentumToCurriculumBridge",
        start_method=StartMethod.SYNC_START,  # start() is synchronous
    ),
    # --- Recovery & Maintenance Daemons ---
    "recovery_orchestrator": RunnerSpec(
        module="app.coordination.recovery_orchestrator",
        class_name="RecoveryOrchestrator",
    ),
    "cache_coordination": RunnerSpec(
        module="app.coordination.cache_coordination_orchestrator",
        class_name="CacheCoordinationOrchestrator",
    ),
    "maintenance": RunnerSpec(
        module="app.coordination.maintenance_daemon",
        class_name="MaintenanceDaemon",
    ),
    "orphan_detection": RunnerSpec(
        module="app.coordination.orphan_detection_daemon",
        class_name="OrphanDetectionDaemon",
    ),
    "data_cleanup": RunnerSpec(
        module="app.coordination.data_cleanup_daemon",
        class_name="DataCleanupDaemon",
    ),
    "disk_space_manager": RunnerSpec(
        module="app.coordination.disk_space_manager_daemon",
        class_name="DiskSpaceManagerDaemon",
    ),
    "coordinator_disk_manager": RunnerSpec(
        module="app.coordination.disk_space_manager_daemon",
        class_name="CoordinatorDiskManager",
    ),
    "node_availability": RunnerSpec(
        module="app.coordination.node_availability.daemon",
        class_name="NodeAvailabilityDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_node_availability_daemon",
    ),
    "sync_push": RunnerSpec(
        module="app.coordination.sync_push_daemon",
        class_name="SyncPushDaemon",
    ),
    "unified_data_plane": RunnerSpec(
        module="app.coordination.unified_data_plane_daemon",
        class_name="UnifiedDataPlaneDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_data_plane_daemon",
    ),
    # --- Miscellaneous Daemons ---
    "s3_backup": RunnerSpec(
        module="app.coordination.s3_backup_daemon",
        class_name="S3BackupDaemon",
    ),
    "s3_node_sync": RunnerSpec(
        module="app.coordination.s3_node_sync_daemon",
        class_name="S3NodeSyncDaemon",
    ),
    "s3_consolidation": RunnerSpec(
        module="app.coordination.s3_node_sync_daemon",
        class_name="S3ConsolidationDaemon",
    ),
    "distillation": RunnerSpec(
        module="app.coordination.distillation_daemon",
        class_name="DistillationDaemon",
    ),
    "external_drive_sync": RunnerSpec(
        module="app.coordination.external_drive_sync",
        class_name="ExternalDriveSyncDaemon",
    ),
    "vast_cpu_pipeline": RunnerSpec(
        module="app.distributed.vast_cpu_pipeline",
        class_name="VastCpuPipelineDaemon",
    ),
    "cluster_data_sync": RunnerSpec(
        module="app.coordination.auto_sync_daemon",
        class_name="AutoSyncDaemon",
        style=InstantiationStyle.WITH_CONFIG,
        config_class="AutoSyncConfig",
        deprecated=True,
        deprecation_message="Use AutoSyncDaemon(strategy='broadcast') instead. Removal scheduled for Q2 2026.",
        wait=WaitStyle.CUSTOM,
    ),
    "p2p_backend": RunnerSpec(
        module="app.coordination.p2p_integration",
        class_name="P2PIntegration",
    ),
    "p2p_auto_deploy": RunnerSpec(
        module="app.coordination.p2p_auto_deployer",
        class_name="P2PAutoDeployer",
    ),
    "metrics_analysis": RunnerSpec(
        module="app.coordination.metrics_analysis_orchestrator",
        class_name="MetricsAnalysisOrchestrator",
    ),
    "per_orchestrator": RunnerSpec(
        module="app.training.per_orchestrator",
        class_name="PEROrchestrator",
        style=InstantiationStyle.FACTORY,
        factory_func="wire_per_events",
        start_method=StartMethod.NONE,
        wait=WaitStyle.FOREVER_LOOP,
        wait_interval=60.0,
    ),
    "data_consolidation": RunnerSpec(
        module="app.coordination.data_consolidation_daemon",
        class_name="DataConsolidationDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_consolidation_daemon",
    ),
    "cluster_consolidation": RunnerSpec(
        module="app.coordination.cluster_consolidation_daemon",
        class_name="ClusterConsolidationDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_cluster_consolidation_daemon",
        notes="Jan 2026: Pulls games from cluster nodes into canonical DBs for training",
    ),
    "npz_combination": RunnerSpec(
        module="app.coordination.npz_combination_daemon",
        class_name="NPZCombinationDaemon",
        style=InstantiationStyle.ASYNC_FACTORY,
        factory_func="get_npz_combination_daemon",
    ),
    "integrity_check": RunnerSpec(
        module="app.coordination.integrity_check_daemon",
        class_name="IntegrityCheckDaemon",
        style=InstantiationStyle.WITH_CONFIG,
        config_class="IntegrityCheckConfig",
    ),
    # --- Cluster Availability Manager ---
    "availability_node_monitor": RunnerSpec(
        module="app.coordination.availability.node_monitor",
        class_name="NodeMonitor",
        style=InstantiationStyle.FACTORY,
        factory_func="get_node_monitor",
    ),
    "availability_recovery_engine": RunnerSpec(
        module="app.coordination.availability.recovery_engine",
        class_name="RecoveryEngine",
        style=InstantiationStyle.FACTORY,
        factory_func="get_recovery_engine",
    ),
    "availability_capacity_planner": RunnerSpec(
        module="app.coordination.availability.capacity_planner",
        class_name="CapacityPlanner",
        style=InstantiationStyle.FACTORY,
        factory_func="get_capacity_planner",
    ),
    "availability_provisioner": RunnerSpec(
        module="app.coordination.availability.provisioner",
        class_name="Provisioner",
        style=InstantiationStyle.FACTORY,
        factory_func="get_provisioner",
    ),
    "cascade_training": RunnerSpec(
        module="app.coordination.cascade_training",
        class_name="CascadeTrainingOrchestrator",
        style=InstantiationStyle.FACTORY,
        factory_func="get_cascade_orchestrator",
    ),
    # --- 48-Hour Autonomous Operation ---
    "progress_watchdog": RunnerSpec(
        module="app.coordination.progress_watchdog_daemon",
        class_name="ProgressWatchdog",
        style=InstantiationStyle.FACTORY,
        factory_func="get_progress_watchdog",
    ),
    "p2p_recovery": RunnerSpec(
        module="app.coordination.p2p_recovery_daemon",
        class_name="P2PRecoveryDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_p2p_recovery_daemon",
    ),
    "voter_health_monitor": RunnerSpec(
        module="app.coordination.voter_health_daemon",
        class_name="VoterHealthMonitorDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_voter_health_daemon",
        notes="Dec 30, 2025: Multi-transport voter health probing for quorum protection",
    ),
    "memory_monitor": RunnerSpec(
        module="app.coordination.memory_monitor_daemon",
        class_name="MemoryMonitorDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_memory_monitor",
    ),
    "stale_fallback": RunnerSpec(
        module="app.coordination.stale_fallback",
        class_name="TrainingFallbackController",
        style=InstantiationStyle.FACTORY,
        factory_func="get_training_fallback_controller",
        start_method=StartMethod.NONE,
        wait=WaitStyle.CUSTOM,  # Subscribes to events and runs forever loop
    ),
    # --- Underutilization Recovery (January 4, 2026 - Phase 3 P2P Resilience) ---
    # Handles CLUSTER_UNDERUTILIZED and WORK_QUEUE_EXHAUSTED events by injecting work items.
    "underutilization_recovery": RunnerSpec(
        module="app.coordination.underutilization_recovery_handler",
        class_name="UnderutilizationRecoveryHandler",
        style=InstantiationStyle.FACTORY,
        factory_func="get_underutilization_handler",
        notes="Jan 4, 2026: Phase 3 P2P Resilience - work injection on underutilization",
    ),
    # --- Fast Failure Detector (January 4, 2026 - Phase 4 P2P Resilience) ---
    # Tiered failure detection: 5min warning → 10min alert → 30min recovery.
    "fast_failure_detector": RunnerSpec(
        module="app.coordination.fast_failure_detector",
        class_name="FastFailureDetector",
        style=InstantiationStyle.FACTORY,
        factory_func="get_fast_failure_detector",
        notes="Jan 4, 2026: Phase 4 P2P Resilience - 5-10 minute failure detection",
    ),
    "tailscale_health": RunnerSpec(
        module="app.coordination.tailscale_health_daemon",
        class_name="TailscaleHealthDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_tailscale_health_daemon",
    ),
    "connectivity_recovery": RunnerSpec(
        module="app.coordination.connectivity_recovery_coordinator",
        class_name="ConnectivityRecoveryCoordinator",
        style=InstantiationStyle.FACTORY,
        factory_func="get_connectivity_recovery_coordinator",
    ),
    # --- Parity Validation (December 30, 2025) ---
    # Runs on coordinator (has Node.js) to validate TS/Python parity.
    # Stores TS reference hashes enabling hash-based validation on cluster nodes.
    "parity_validation": RunnerSpec(
        module="app.coordination.parity_validation_daemon",
        class_name="ParityValidationDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_parity_validation_daemon",
        notes="Dec 30, 2025: Validates pending_gate databases and stores TS hashes",
    ),
    # --- Comprehensive Data Consolidation System (January 2026) ---
    # These daemons provide unified data management across all storage locations.
    "owc_push": RunnerSpec(
        module="app.coordination.owc_push_daemon",
        class_name="OWCPushDaemon",
        style=InstantiationStyle.FACTORY,
        factory_func="get_owc_push_daemon",
        notes="Jan 2026: Push data to OWC external drive for backup",
    ),
    "s3_import": RunnerSpec(
        module="app.coordination.s3_import_daemon",
        class_name="S3ImportDaemon",
        style=InstantiationStyle.SINGLETON,
        notes="Jan 2026: Import data from S3 for recovery/bootstrap",
    ),
    "unified_data_catalog": RunnerSpec(
        module="app.coordination.unified_data_catalog",
        class_name="UnifiedDataCatalog",
        style=InstantiationStyle.SINGLETON,
        notes="Jan 2026: Single API for querying data across all sources",
    ),
    "dual_backup": RunnerSpec(
        module="app.coordination.dual_backup_daemon",
        class_name="DualBackupDaemon",
        style=InstantiationStyle.SINGLETON,
        notes="Jan 2026: Ensures data is backed up to BOTH S3 AND OWC",
    ),
    "node_data_agent": RunnerSpec(
        module="app.coordination.node_data_agent",
        class_name="NodeDataAgent",
        style=InstantiationStyle.FACTORY,
        factory_func="get_node_data_agent",
        notes="Jan 2026: Per-node agent for data discovery and fetching",
    ),
}


async def _create_runner_from_spec(name: str) -> None:
    """Generic runner factory using registry specs.

    This function creates and runs a daemon based on its specification
    in the RUNNER_SPECS registry.

    Args:
        name: The runner name (matches DaemonType enum value in lowercase)

    Raises:
        ValueError: If runner name is not in registry
        ImportError: If daemon module cannot be imported
    """
    spec = RUNNER_SPECS.get(name)
    if not spec:
        raise ValueError(f"Unknown runner: {name}")

    # Emit deprecation warning if applicable
    if spec.deprecated:
        warnings.warn(
            spec.deprecation_message or f"Runner {name} is deprecated.",
            DeprecationWarning,
            stacklevel=3,
        )

    try:
        module = importlib.import_module(spec.module)

        # Instantiate daemon based on style
        # Note: Using if/elif for Python 3.9 compatibility (match requires 3.10+)
        daemon: Any = None
        if spec.style == InstantiationStyle.DIRECT:
            daemon_class = getattr(module, spec.class_name)
            daemon = daemon_class()
        elif spec.style == InstantiationStyle.SINGLETON:
            daemon_class = getattr(module, spec.class_name)
            daemon = daemon_class.get_instance()
        elif spec.style == InstantiationStyle.FACTORY:
            factory = getattr(module, spec.factory_func)
            daemon = factory()
        elif spec.style == InstantiationStyle.ASYNC_FACTORY:
            factory = getattr(module, spec.factory_func)
            daemon = await factory()
        elif spec.style == InstantiationStyle.WITH_CONFIG:
            daemon_class = getattr(module, spec.class_name)
            config_class = getattr(module, spec.config_class)
            config = config_class.from_env()
            daemon = daemon_class(config=config)

        # Start daemon based on start method
        # Note: Using if/elif for Python 3.9 compatibility (match requires 3.10+)
        if spec.start_method == StartMethod.ASYNC_START:
            await daemon.start()
        elif spec.start_method == StartMethod.SYNC_START:
            daemon.start()
        elif spec.start_method == StartMethod.INITIALIZE:
            await daemon.initialize()
            await daemon.start()  # Managers often need start() after initialize()
        elif spec.start_method == StartMethod.START_SERVER:
            await daemon.start_server()
        elif spec.start_method == StartMethod.NONE:
            pass  # No start method needed

        # Wait for daemon based on wait style
        # Note: Using if/elif for Python 3.9 compatibility (match requires 3.10+)
        if spec.wait == WaitStyle.DAEMON:
            await _wait_for_daemon(daemon)
        elif spec.wait == WaitStyle.FOREVER_LOOP:
            while True:
                await asyncio.sleep(spec.wait_interval)
        elif spec.wait == WaitStyle.RUN_FOREVER:
            await daemon.run_forever()
        elif spec.wait == WaitStyle.NONE:
            pass  # No waiting
        elif spec.wait == WaitStyle.CUSTOM:
            # Custom handling - fall through to legacy function
            # This shouldn't happen if we migrated correctly
            logger.warning(f"[{name}] Using CUSTOM wait - check if legacy function is needed")
            await _wait_for_daemon(daemon)

    except ImportError as e:
        logger.error(f"{spec.class_name} not available: {e}")
        raise


def get_runner_spec(name: str) -> RunnerSpec | None:
    """Get the runner specification for a daemon name.

    Args:
        name: The runner name (lowercase with underscores)

    Returns:
        The RunnerSpec if found, None otherwise
    """
    return RUNNER_SPECS.get(name)


def _check_daemon_running(daemon: Any) -> bool:
    """Check if a daemon is running, handling different daemon implementations.

    Supports:
    - Daemons with is_running property (BaseDaemon pattern)
    - Daemons with is_running() method
    - Daemons with _running attribute (legacy pattern)
    """
    # Try is_running as property or method
    if hasattr(daemon, "is_running"):
        attr = getattr(daemon, "is_running")
        if callable(attr):
            return attr()
        return attr
    # Fall back to _running attribute
    if hasattr(daemon, "_running"):
        return daemon._running
    # Default: assume not running if we can't determine
    return False


async def _wait_for_daemon(daemon: Any, check_interval: float = 10.0) -> None:
    """Wait for a daemon to complete or be stopped."""
    while _check_daemon_running(daemon):
        await asyncio.sleep(check_interval)


# =============================================================================
# Sync Daemons
# =============================================================================


async def create_sync_coordinator() -> None:
    """Create and run sync coordinator daemon.

    DEPRECATED (December 2025): Use DaemonType.AUTO_SYNC instead.
    This runner is retained for backward compatibility and will be removed in Q2 2026.
    """
    warnings.warn(
        "DaemonType.SYNC_COORDINATOR is deprecated. Use DaemonType.AUTO_SYNC instead. "
        "Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from app.distributed.sync_coordinator import SyncCoordinator

        sync = SyncCoordinator.get_instance()
        await sync.start()
        await _wait_for_daemon(sync)
    except ImportError as e:
        logger.error(f"SyncCoordinator not available: {e}")
        raise


async def create_high_quality_sync() -> None:
    """Create and run high quality data sync watcher (Phase 2, December 2025)."""
    try:
        from app.coordination.training_freshness import HighQualityDataSyncWatcher

        watcher = HighQualityDataSyncWatcher()
        await watcher.start()
        await _wait_for_daemon(watcher)
    except ImportError as e:
        logger.error(f"HighQualityDataSyncWatcher not available: {e}")
        raise


async def create_elo_sync() -> None:
    """Create and run Elo sync manager (Phase 8, December 2025)."""
    try:
        from app.tournament.elo_sync_manager import EloSyncManager

        manager = EloSyncManager()
        await manager.initialize()
        # December 27, 2025: Fixed - was calling sync_loop() which doesn't exist
        # The start() method from SyncManagerBase runs the sync loop
        await manager.start()
    except ImportError as e:
        logger.error(f"EloSyncManager not available: {e}")
        raise


async def create_auto_sync() -> None:
    """Create and run automated P2P data sync daemon (December 2025)."""
    try:
        from app.coordination.auto_sync_daemon import AutoSyncDaemon

        daemon = AutoSyncDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"AutoSyncDaemon not available: {e}")
        raise


async def create_config_sync() -> None:
    """Create and run config sync daemon (January 2026).

    Automatically syncs distributed_hosts.yaml across cluster nodes.
    - Coordinator: Detects file changes via mtime polling, emits CONFIG_UPDATED
    - Workers: Subscribe to event, pull newer config via rsync
    """
    try:
        from app.coordination.config_sync_daemon import get_config_sync_daemon

        daemon = get_config_sync_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ConfigSyncDaemon not available: {e}")
        raise


async def create_config_validator() -> None:
    """Create and run config validator daemon (January 2026).

    Validates distributed_hosts.yaml against provider APIs (Lambda, Vast,
    RunPod, Tailscale) to detect configuration drift and stale entries.
    """
    try:
        from app.coordination.config_validator_daemon import ConfigValidatorDaemon

        daemon = ConfigValidatorDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ConfigValidatorDaemon not available: {e}")
        raise


async def create_training_node_watcher() -> None:
    """Create and run training activity daemon (December 2025).

    Monitors cluster for training activity and triggers priority sync
    before training starts. Uses TrainingActivityDaemon which:
    - Detects training via P2P status (running_jobs, processes)
    - Detects local training via process monitoring
    - Triggers priority sync when new training detected
    - Emits TRAINING_STARTED events for coordination
    - Graceful shutdown on SIGTERM with final sync
    """
    try:
        from app.coordination.training_activity_daemon import (
            TrainingActivityConfig,
            TrainingActivityDaemon,
        )

        config = TrainingActivityConfig.from_env()
        daemon = TrainingActivityDaemon(config=config)
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"TrainingActivityDaemon not available: {e}")
        raise


async def create_training_data_sync() -> None:
    """Create and run training data sync daemon (December 2025).

    Pre-training data synchronization from OWC drive and S3.
    Ensures training nodes have access to the best available training
    data before starting training jobs.

    Features:
    - Multi-source: OWC external drive, S3 bucket
    - Config-aware: Only syncs data for configs being trained
    - Size-based selection: Downloads largest available dataset
    - Resume support: Skips already-downloaded files
    """
    try:
        from app.coordination.training_data_sync_daemon import (
            TrainingDataSyncDaemon,
            get_training_data_sync_daemon,
        )

        daemon = get_training_data_sync_daemon()
        await daemon.start()

        # Keep running until stopped
        while True:
            await asyncio.sleep(60)
            if not daemon._running:
                break

    except ImportError as e:
        logger.error(f"TrainingDataSyncDaemon not available: {e}")
        raise


async def create_training_data_recovery() -> None:
    """Create and run training data recovery daemon (January 2026 Sprint 13.3).

    Auto-recovery from training data corruption by re-exporting NPZ files.

    Features:
    - Subscribes to TRAINING_FAILED events
    - Detects data corruption patterns
    - Triggers NPZ re-export from canonical databases
    - Emits TRAINING_DATA_RECOVERED / TRAINING_DATA_RECOVERY_FAILED
    """
    try:
        from app.coordination.training_data_recovery_daemon import (
            TrainingDataRecoveryDaemon,
            get_training_data_recovery_daemon,
        )

        daemon = get_training_data_recovery_daemon()
        await daemon.start()

        # Keep running until stopped
        while True:
            await asyncio.sleep(60)
            if not daemon._running:
                break

    except ImportError as e:
        logger.error(f"TrainingDataRecoveryDaemon not available: {e}")
        raise


async def create_training_watchdog() -> None:
    """Create and run training watchdog daemon (January 2026 Sprint 17).

    Monitors training processes for stalls and kills stuck processes.

    Features:
    - Tracks training process heartbeats via TRAINING_HEARTBEAT events
    - Detects stale processes (no heartbeat for 2+ hours)
    - Kills stale processes (SIGTERM, then SIGKILL after grace period)
    - Releases associated training locks
    - Emits TRAINING_PROCESS_KILLED event
    """
    try:
        from app.coordination.training_watchdog_daemon import (
            TrainingWatchdogDaemon,
            get_training_watchdog_daemon,
        )

        daemon = get_training_watchdog_daemon()
        await daemon.start()

        # Keep running until stopped
        while True:
            await asyncio.sleep(60)
            if not daemon._running:
                break

    except ImportError as e:
        logger.error(f"TrainingWatchdogDaemon not available: {e}")
        raise


async def create_export_watchdog() -> None:
    """Create and run export watchdog daemon (January 2026 Session 17.41).

    Monitors export_replay_dataset.py processes for hangs and kills them
    if they exceed the maximum runtime threshold (default 30 minutes).

    Features:
    - Detects export processes via pgrep
    - Tracks runtime of each process
    - Kills processes exceeding max_export_runtime (SIGTERM, then SIGKILL)
    - Emits EXPORT_TIMEOUT event when killing a stuck process
    """
    try:
        from app.coordination.export_watchdog_daemon import (
            ExportWatchdogDaemon,
            get_export_watchdog_daemon,
        )

        daemon = get_export_watchdog_daemon()
        await daemon.start()

        # Keep running until stopped
        while True:
            await asyncio.sleep(60)
            if not daemon._running:
                break

    except ImportError as e:
        logger.error(f"ExportWatchdogDaemon not available: {e}")
        raise


async def create_owc_import() -> None:
    """Create and run OWC import daemon (December 29, 2025).

    Periodically imports training data from OWC external drive on mac-studio
    for underserved configs (those with <100,000 games in canonical databases).
    January 2026: Threshold increased from 500 to 100,000 to enable comprehensive
    import from OWC drive (8.5M games available).

    Features:
    - SSH-based discovery of databases on OWC drive
    - Automatic detection of underserved configs
    - Rsync-based database transfer
    - Event emission for consolidation pipeline integration
    - Emits NEW_GAMES_AVAILABLE and DATA_SYNC_COMPLETED events

    Environment variables:
    - OWC_HOST: Remote host with OWC drive (default: mac-studio)
    - OWC_BASE_PATH: Base path on OWC drive (default: /Volumes/RingRift-Data)
    - OWC_SSH_KEY: SSH key for connection (default: ~/.ssh/id_ed25519)
    - RINGRIFT_OWC_UNDERSERVED_THRESHOLD: Threshold for import (default: 100000)
    """
    try:
        from app.coordination.owc_import_daemon import (
            OWCImportConfig,
            OWCImportDaemon,
            get_owc_import_daemon,
        )

        daemon = get_owc_import_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"OWCImportDaemon not available: {e}")
        raise


async def create_owc_model_import() -> None:
    """Create and run OWC model import daemon (January 3, 2026 - Sprint 13 Session 4).

    Imports MODEL FILES (.pth) from OWC external drive on mac-studio.
    Unlike OWCImportDaemon (which imports databases), this daemon imports
    trained models that need Elo evaluation.

    Features:
    - SSH-based discovery of models on OWC drive
    - Cross-reference with EloService to skip already-rated models
    - Rsync-based model transfer
    - Emits MODEL_IMPORTED and OWC_MODELS_DISCOVERED events
    - Adds models to PersistentEvaluationQueue for evaluation

    Environment variables:
    - OWC_HOST: Remote host with OWC drive (default: mac-studio)
    - OWC_BASE_PATH: Base path on OWC drive (default: /Volumes/RingRift-Data)
    """
    try:
        from app.coordination.owc_model_import_daemon import (
            get_owc_model_import_daemon,
        )

        daemon = get_owc_model_import_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"OWCModelImportDaemon not available: {e}")
        raise


async def create_unevaluated_model_scanner() -> None:
    """Create and run unevaluated model scanner daemon (January 3, 2026 - Sprint 13 Session 4).

    Scans all model sources (local, OWC, cluster, registry) for models
    without Elo ratings. Adds them to the evaluation queue with
    curriculum-aware priority.

    Features:
    - Scans local models/ directory for .pth files
    - Cross-references with EloService (unified_elo.db)
    - Computes priority based on curriculum weights
    - Emits EVALUATION_REQUESTED and UNEVALUATED_MODELS_FOUND events
    - Adds models to PersistentEvaluationQueue
    """
    try:
        from app.coordination.unevaluated_model_scanner_daemon import (
            get_unevaluated_model_scanner_daemon,
        )

        daemon = get_unevaluated_model_scanner_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"UnevaluatedModelScannerDaemon not available: {e}")
        raise


async def create_stale_evaluation() -> None:
    """Create and run stale evaluation daemon (January 4, 2026 - Sprint 17).

    Re-evaluates models with Elo ratings older than a configured threshold
    (default 30 days). This ensures model rankings stay accurate as the
    training landscape evolves.

    Features:
    - Scans EloService for models with old evaluation timestamps
    - Adds to evaluation queue with appropriate priority
    - Prevents unnecessary re-evaluation with configurable cooldowns
    - Emits EVALUATION_REQUESTED events
    """
    try:
        from app.coordination.stale_evaluation_daemon import (
            get_stale_evaluation_daemon,
        )

        daemon = get_stale_evaluation_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"StaleEvaluationDaemon not available: {e}")
        raise


async def create_comprehensive_model_scan() -> None:
    """Create and run comprehensive model scan daemon (January 9, 2026 - Sprint 17.9).

    Discovers ALL models across all sources:
    - Local models/ directory
    - ClusterModelDiscovery (SSH queries to cluster nodes)
    - OWCModelDiscovery (external OWC drive)

    For each discovered model, queues evaluation under all compatible harnesses
    (BRS, MaxN, Descent, Gumbel MCTS). Ensures all models get fresh Elo ratings
    under multiple harnesses, with games saved for training.

    Features:
    - SHA256-based deduplication across sources
    - Priority-based queueing (4-player, underserved configs boosted)
    - Configurable harness types via environment
    - Subscribes to MODEL_IMPORTED and MODEL_PROMOTED for immediate queueing
    """
    try:
        from app.coordination.comprehensive_model_scan_daemon import (
            get_comprehensive_model_scan_daemon,
        )

        daemon = get_comprehensive_model_scan_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ComprehensiveModelScanDaemon not available: {e}")
        raise


async def create_ephemeral_sync() -> None:
    """Create and run ephemeral sync daemon (Phase 4, December 2025).

    DEPRECATED (December 2025): Use AutoSyncDaemon with strategy="ephemeral" instead.
    This runner is retained for backward compatibility and will be removed in Q2 2026.
    """
    warnings.warn(
        "DaemonType.EPHEMERAL_SYNC is deprecated. Use AutoSyncDaemon(strategy='ephemeral') instead. "
        "Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from app.coordination.auto_sync_daemon import (
            AutoSyncConfig,
            AutoSyncDaemon,
            SyncStrategy,
        )

        config = AutoSyncConfig.from_config_file()
        config.strategy = SyncStrategy.EPHEMERAL
        daemon = AutoSyncDaemon(config=config)
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"AutoSyncDaemon not available for ephemeral sync: {e}")
        raise


async def create_gossip_sync() -> None:
    """Create and run gossip sync daemon.

    Jan 3, 2026: Fixed to provide required constructor arguments.
    GossipSyncDaemon requires node_id, data_dir, and peers_config.

    Jan 10, 2026: Skip if P2P orchestrator is running (uses same port 8771).
    """
    import socket

    # Check if port 8771 is already in use (likely P2P orchestrator)
    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        test_sock.bind(("0.0.0.0", 8771))
        test_sock.close()
    except OSError:
        # Port already in use - P2P orchestrator likely running
        logger.info("GossipSync skipped: port 8771 already in use (P2P orchestrator running)")
        # Sleep forever to keep daemon "running" without conflict
        import asyncio
        while True:
            await asyncio.sleep(3600)
        return

    try:
        from pathlib import Path

        from app.distributed.gossip_sync import GossipSyncDaemon

        # Get node_id from hostname
        node_id = socket.gethostname()

        # Use standard data directory
        data_dir = Path("data/games")

        # Load peers config from distributed_hosts.yaml
        peers_config: dict[str, dict] = {}
        try:
            from app.config.cluster_config import get_config_cache
            config = get_config_cache().get_config()
            if config and "hosts" in config:
                for host_name, host_info in config["hosts"].items():
                    if isinstance(host_info, dict):
                        peers_config[host_name] = {
                            "host": host_info.get("tailscale_ip") or host_info.get("ssh_host", ""),
                            "port": host_info.get("gossip_port", 8771),
                        }
        except Exception as e:
            logger.warning(f"Failed to load peers config for GossipSync: {e}")

        daemon = GossipSyncDaemon(
            node_id=node_id,
            data_dir=data_dir,
            peers_config=peers_config,
        )
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"GossipSyncDaemon not available: {e}")
        raise


# =============================================================================
# Event Processing Daemons
# =============================================================================


async def create_event_router() -> None:
    """Create and run unified event router daemon.

    December 27, 2025: Added validation guards to catch import timing issues.
    The .start() method may not be available if circular imports leave
    the UnifiedEventRouter class incomplete during initialization.
    """
    try:
        from app.coordination.event_router import UnifiedEventRouter, get_router

        # Validate class has required method (guards against incomplete import)
        if not hasattr(UnifiedEventRouter, "start"):
            logger.error(
                "UnifiedEventRouter.start() not found - possible circular import issue"
            )
            raise RuntimeError("UnifiedEventRouter missing start() method")

        router = get_router()

        # Double-check instance has method (guards against class mismatch)
        if not hasattr(router, "start"):
            logger.error("Router instance missing start() - possible stale module cache")
            raise RuntimeError("Router instance missing start() method")

        await router.start()
        await _wait_for_daemon(router)
    except ImportError as e:
        logger.error(f"UnifiedEventRouter not available: {e}")
        raise


async def create_cross_process_poller() -> None:
    """Create and run cross-process event poller daemon.

    Jan 3, 2026: Fixed import to use cross_process_events module.
    Jan 3, 2026: Fixed to provide required process_name argument.
    Jan 4, 2026: Fixed await issue - start() is synchronous, not async.
    """
    try:
        from app.coordination.cross_process_events import CrossProcessEventPoller

        # process_name identifies this poller instance for event routing
        poller = CrossProcessEventPoller(
            process_name="daemon_manager",
            event_types=None,  # Subscribe to all event types
        )
        # start() is synchronous - spawns background polling thread
        poller.start()
        # _wait_for_daemon expects async start, but we're already started
        # Just verify the poller is running
        if hasattr(poller, "_running") and poller._running:
            logger.info("CrossProcessEventPoller started successfully")
    except ImportError as e:
        logger.error(f"CrossProcessEventPoller not available: {e}")
        raise


async def create_dlq_retry() -> None:
    """Create and run dead-letter queue retry daemon (December 2025)."""
    try:
        from app.coordination.dead_letter_queue import DLQRetryDaemon

        retry = DLQRetryDaemon()  # Uses get_dead_letter_queue() internally
        await retry.start()
        await _wait_for_daemon(retry)
    except ImportError as e:
        logger.error(f"DLQRetryDaemon not available: {e}")
        raise


# =============================================================================
# Health & Monitoring Daemons
# =============================================================================


async def create_health_check() -> None:
    """Create and run health check daemon.

    DEPRECATED (December 2025): Use DaemonType.NODE_HEALTH_MONITOR instead.
    This runner is retained for backward compatibility and will be removed in Q2 2026.
    """
    warnings.warn(
        "DaemonType.HEALTH_CHECK is deprecated. Use DaemonType.NODE_HEALTH_MONITOR instead. "
        "Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from app.coordination.health_check_orchestrator import HealthCheckOrchestrator

        orchestrator = HealthCheckOrchestrator()
        await orchestrator.start()
        await _wait_for_daemon(orchestrator)
    except ImportError as e:
        logger.error(f"HealthCheckOrchestrator not available: {e}")
        raise


async def create_queue_monitor() -> None:
    """Create and run queue monitor daemon."""
    try:
        from app.coordination.queue_monitor import QueueMonitor

        monitor = QueueMonitor()
        await monitor.start()
        await _wait_for_daemon(monitor)
    except ImportError as e:
        logger.error(f"QueueMonitor not available: {e}")
        raise


async def create_daemon_watchdog() -> None:
    """Create and run daemon watchdog (December 2025)."""
    try:
        from app.coordination.daemon_watchdog import start_watchdog

        # start_watchdog() is async and already calls .start() internally
        watchdog = await start_watchdog()
        await _wait_for_daemon(watchdog)
    except ImportError as e:
        logger.error(f"DaemonWatchdog not available: {e}")
        raise


async def create_node_health_monitor() -> None:
    """Create and run node health monitor daemon.

    DEPRECATED (December 2025): Use HealthCheckOrchestrator directly via HEALTH_SERVER.
    The NODE_HEALTH_MONITOR daemon type is deprecated in favor of the unified
    health_check_orchestrator. This runner is retained for backward compatibility
    and will be removed in Q2 2026.
    """
    warnings.warn(
        "DaemonType.NODE_HEALTH_MONITOR is deprecated. Use HealthCheckOrchestrator "
        "(via DaemonType.HEALTH_SERVER) instead. Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from app.coordination.health_check_orchestrator import HealthCheckOrchestrator

        orchestrator = HealthCheckOrchestrator()
        await orchestrator.start()
        await _wait_for_daemon(orchestrator)
    except ImportError as e:
        logger.error(f"HealthCheckOrchestrator not available: {e}")
        raise


async def create_system_health_monitor() -> None:
    """Create and run system health monitor daemon (December 2025).

    DEPRECATED (December 2025): Use unified_health_manager.get_system_health_score() instead.
    This daemon type is deprecated in favor of unified_health_manager functions.
    This runner is retained for backward compatibility and will be removed in Q2 2026.
    """
    warnings.warn(
        "DaemonType.SYSTEM_HEALTH_MONITOR is deprecated. Use unified_health_manager."
        "get_system_health_score() instead. Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from app.coordination.unified_health_manager import UnifiedHealthManager

        manager = UnifiedHealthManager()
        await manager.start()
        await _wait_for_daemon(manager)
    except ImportError as e:
        logger.error(f"UnifiedHealthManager not available: {e}")
        raise


async def create_health_server() -> None:
    """Create and run HTTP health server (December 2025).

    This runner wraps the DaemonManager's _create_health_server method.
    The health server exposes endpoints at port 8790:
    - GET /health: Liveness probe
    - GET /ready: Readiness probe
    - GET /metrics: Prometheus-style metrics
    - GET /status: Detailed daemon status

    CIRCULAR DEPENDENCY NOTE (Dec 2025):
    This function imports get_daemon_manager() from daemon_manager.py.
    daemon_manager.py imports daemon_runners at top-level.
    This is SAFE because:
    1. This import is LAZY (inside function body, not at module load time)
    2. By the time this function is called, daemon_manager.py is fully loaded
    3. The circular reference is resolved at runtime, not import time
    """
    try:
        # Lazy import to avoid circular dependency with daemon_manager.py
        from app.coordination.daemon_manager import get_daemon_manager

        dm = get_daemon_manager()
        await dm._create_health_server()
    except ImportError as e:
        logger.error(f"DaemonManager not available for health server: {e}")
        raise


async def create_quality_monitor() -> None:
    """Create and run quality monitor daemon (December 2025)."""
    try:
        from app.coordination.quality_monitor_daemon import QualityMonitorDaemon

        daemon = QualityMonitorDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"QualityMonitorDaemon not available: {e}")
        raise


async def create_model_performance_watchdog() -> None:
    """Create and run model performance watchdog (December 2025)."""
    try:
        from app.coordination.model_performance_watchdog import ModelPerformanceWatchdog

        watchdog = ModelPerformanceWatchdog()
        await watchdog.start()
        await _wait_for_daemon(watchdog)
    except ImportError as e:
        logger.error(f"ModelPerformanceWatchdog not available: {e}")
        raise


async def create_cluster_monitor() -> None:
    """Create and run cluster monitor daemon."""
    try:
        from app.coordination.cluster_status_monitor import ClusterMonitor

        monitor = ClusterMonitor()
        await monitor.run_forever()
    except ImportError as e:
        logger.error(f"ClusterMonitor not available: {e}")
        raise


async def create_cluster_watchdog() -> None:
    """Create and run cluster watchdog daemon (December 2025)."""
    try:
        from app.coordination.cluster_watchdog_daemon import ClusterWatchdogDaemon

        daemon = ClusterWatchdogDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ClusterWatchdogDaemon not available: {e}")
        raise


async def create_coordinator_health_monitor() -> None:
    """Create and run coordinator health monitor daemon (December 2025).

    Subscribes to COORDINATOR_* events to track coordinator lifecycle:
    - Coordinator health status (healthy/unhealthy/degraded)
    - Heartbeat freshness monitoring
    - Init failure tracking
    """
    try:
        from app.coordination.coordinator_health_monitor_daemon import (
            get_coordinator_health_monitor,
        )

        monitor = get_coordinator_health_monitor()
        await monitor.start()
        await _wait_for_daemon(monitor)
    except ImportError as e:
        logger.error(f"CoordinatorHealthMonitorDaemon not available: {e}")
        raise


async def create_work_queue_monitor() -> None:
    """Create and run work queue monitor daemon (December 2025).

    Subscribes to WORK_* events to track queue lifecycle:
    - Queue depth tracking
    - Job latency monitoring
    - Stuck job detection
    - Backpressure signaling
    """
    try:
        from app.coordination.work_queue_monitor_daemon import (
            get_work_queue_monitor,
        )

        monitor = get_work_queue_monitor()
        await monitor.start()
        await _wait_for_daemon(monitor)
    except ImportError as e:
        logger.error(f"WorkQueueMonitorDaemon not available: {e}")
        raise


# =============================================================================
# Training & Pipeline Daemons
# =============================================================================


async def create_data_pipeline() -> None:
    """Create and run data pipeline orchestrator (December 2025)."""
    try:
        from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator

        orchestrator = get_pipeline_orchestrator()
        await orchestrator.start()
        await _wait_for_daemon(orchestrator)
    except ImportError as e:
        logger.error(f"DataPipelineOrchestrator not available: {e}")
        raise


async def create_continuous_training_loop() -> None:
    """Create and run continuous training loop daemon.

    Jan 3, 2026: DEPRECATED - Module archived. Functionality moved to p2p_orchestrator.py.
    This function now logs a warning and returns without raising to prevent cascade failures.
    """
    logger.warning(
        "[DEPRECATED] ContinuousTrainingLoop module is archived. "
        "Functionality now provided by p2p_orchestrator.py. "
        "This daemon will be removed in Q2 2026."
    )
    # Don't raise - allow daemon manager to continue with other daemons
    return


async def create_selfplay_coordinator() -> None:
    """Initialize selfplay scheduler singleton (December 2025).

    Note: SelfplayScheduler is a utility class, not a daemon with a lifecycle.
    It provides priority-based selfplay scheduling decisions to other daemons
    like IdleResourceDaemon. This runner initializes the singleton and wires
    up its event subscriptions.
    """
    try:
        from app.coordination.selfplay_scheduler import get_selfplay_scheduler

        # Get singleton and wire up event subscriptions
        scheduler = get_selfplay_scheduler()
        logger.info(
            f"[SelfplayCoordinator] Initialized scheduler with {len(scheduler._config_priorities)} configs"
        )
        # SelfplayScheduler doesn't have a lifecycle - it's used by other daemons
        # Keep running indefinitely
        while True:
            await asyncio.sleep(3600)  # Sleep 1 hour, wake up to check for shutdown
    except ImportError as e:
        logger.error(f"SelfplayScheduler not available: {e}")
        raise


async def create_training_trigger() -> None:
    """Create and run training trigger daemon (December 2025)."""
    try:
        from app.coordination.training_trigger_daemon import TrainingTriggerDaemon

        daemon = TrainingTriggerDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"TrainingTriggerDaemon not available: {e}")
        raise


async def create_auto_export() -> None:
    """Create and run auto export daemon (December 2025)."""
    try:
        from app.coordination.auto_export_daemon import AutoExportDaemon

        daemon = AutoExportDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"AutoExportDaemon not available: {e}")
        raise


async def create_tournament_daemon() -> None:
    """Create and run tournament daemon (December 2025)."""
    try:
        from app.coordination.tournament_daemon import TournamentDaemon

        daemon = TournamentDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"TournamentDaemon not available: {e}")
        raise


async def create_nnue_training() -> None:
    """Create and run NNUE training daemon (December 2025).

    Automatically trains NNUE models when game thresholds are met.
    Per-config thresholds: hex8_2p=5000, hex8_4p=10000, square19_2p=2000.
    Subscribes to: NEW_GAMES_AVAILABLE, CONSOLIDATION_COMPLETE, DATA_SYNC_COMPLETED.
    """
    try:
        from app.coordination.nnue_training_daemon import NNUETrainingDaemon

        daemon = NNUETrainingDaemon.get_instance()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"NNUETrainingDaemon not available: {e}")
        raise


async def create_architecture_feedback() -> None:
    """Create and run architecture feedback controller (December 2025).

    Bridges evaluation results to selfplay allocation by tracking architecture
    performance. Enforces 10% minimum allocation per architecture.
    Subscribes to: EVALUATION_COMPLETED, TRAINING_COMPLETED.
    Emits: ARCHITECTURE_WEIGHTS_UPDATED.
    """
    try:
        from app.coordination.architecture_feedback_controller import (
            ArchitectureFeedbackController,
        )

        controller = ArchitectureFeedbackController.get_instance()
        await controller.start()
        await _wait_for_daemon(controller)
    except ImportError as e:
        logger.error(f"ArchitectureFeedbackController not available: {e}")
        raise


async def create_parity_validation() -> None:
    """Create and run parity validation daemon (December 30, 2025).

    Runs on coordinator (which has Node.js) to validate TS/Python parity for
    canonical databases. Cluster nodes generate databases with "pending_gate"
    status because they lack npx. This daemon validates them and stores TS
    reference hashes, enabling hash-based validation on cluster nodes.

    Subscribes to: DATA_SYNC_COMPLETED (to validate newly synced databases)
    Emits: PARITY_VALIDATION_COMPLETED
    """
    try:
        from app.coordination.parity_validation_daemon import (
            get_parity_validation_daemon,
        )

        daemon = get_parity_validation_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ParityValidationDaemon not available: {e}")


async def create_elo_progress() -> None:
    """Create and run Elo progress tracking daemon (December 31, 2025).

    Periodically snapshots the best model's Elo for each config to track
    improvement over time. Provides evidence of training loop effectiveness.

    - Takes snapshots hourly by default
    - Also triggers on EVALUATION_COMPLETED and MODEL_PROMOTED events
    - Stores data in elo_progress.db for trend analysis

    Subscribes to: EVALUATION_COMPLETED, MODEL_PROMOTED
    """
    try:
        from app.coordination.elo_progress_daemon import get_elo_progress_daemon

        daemon = get_elo_progress_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"EloProgressDaemon not available: {e}")
        raise


# =============================================================================
# Evaluation & Promotion Daemons
# =============================================================================


async def create_evaluation_daemon() -> None:
    """Create and run evaluation daemon (December 2025)."""
    try:
        from app.coordination.evaluation_daemon import get_evaluation_daemon

        daemon = get_evaluation_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"EvaluationDaemon not available: {e}")
        raise


async def create_auto_promotion() -> None:
    """Create and run auto-promotion daemon (December 2025)."""
    try:
        from app.coordination.auto_promotion_daemon import AutoPromotionDaemon

        daemon = AutoPromotionDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"AutoPromotionDaemon not available: {e}")
        raise


async def create_unified_promotion() -> None:
    """Create and run unified promotion daemon (December 2025).

    PromotionController is event-driven and sets up subscriptions in __init__.
    We just need to instantiate and keep it alive.
    """
    try:
        from app.training.promotion_controller import PromotionController

        controller = PromotionController()
        # PromotionController subscribes to events in setup_event_subscriptions()
        # called from __init__ - no start() method needed
        logger.info("PromotionController initialized and subscribed to events")

        # Keep alive by waiting indefinitely
        while True:
            await asyncio.sleep(60)
    except ImportError as e:
        logger.error(f"PromotionController not available: {e}")
        raise


async def create_gauntlet_feedback() -> None:
    """Create and run gauntlet feedback controller (December 2025)."""
    try:
        from app.coordination.gauntlet_feedback_controller import (
            GauntletFeedbackController,
        )

        controller = GauntletFeedbackController()
        await controller.start()
        await _wait_for_daemon(controller)
    except ImportError as e:
        logger.error(f"GauntletFeedbackController not available: {e}")
        raise


async def create_backlog_evaluation() -> None:
    """Create and run backlog evaluation daemon (Sprint 15 - Jan 3, 2026).

    Discovers OWC models and queues them for Elo evaluation.
    """
    try:
        from app.coordination.backlog_evaluation_daemon import (
            BacklogEvaluationDaemon,
        )

        daemon = BacklogEvaluationDaemon.get_instance()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"BacklogEvaluationDaemon not available: {e}")
        raise


# =============================================================================
# Distribution Daemons
# =============================================================================


async def create_model_sync() -> None:
    """Create and run model sync daemon (December 2025)."""
    try:
        from app.coordination.unified_distribution_daemon import (
            create_model_distribution_daemon,
        )

        daemon = create_model_distribution_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"Model sync daemon not available: {e}")
        raise


async def create_model_distribution() -> None:
    """Create and run model distribution daemon (December 2025)."""
    try:
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        daemon = UnifiedDistributionDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"UnifiedDistributionDaemon not available: {e}")
        raise


async def create_npz_distribution() -> None:
    """Create and run NPZ distribution daemon (December 2025)."""
    try:
        from app.coordination.unified_distribution_daemon import (
            create_npz_distribution_daemon,
        )

        daemon = create_npz_distribution_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"NPZ distribution daemon not available: {e}")
        raise


async def create_data_server() -> None:
    """Create and run data server daemon."""
    try:
        from app.distributed.sync_coordinator import SyncCoordinator

        sync = SyncCoordinator.get_instance()
        # Jan 3, 2026: Fixed method name - SyncCoordinator has start_data_server(), not start_server()
        await sync.start_data_server()
        await _wait_for_daemon(sync)
    except ImportError as e:
        logger.error(f"SyncCoordinator data server not available: {e}")
        raise


# =============================================================================
# Replication Daemons
# =============================================================================


async def create_replication_monitor() -> None:
    """Create and run replication monitor daemon (December 2025)."""
    try:
        from app.coordination.unified_replication_daemon import (
            create_replication_monitor,
        )

        daemon = create_replication_monitor()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ReplicationMonitor not available: {e}")
        raise


async def create_replication_repair() -> None:
    """Create and run replication repair daemon (December 2025)."""
    try:
        from app.coordination.unified_replication_daemon import (
            create_replication_repair_daemon,
        )

        daemon = create_replication_repair_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ReplicationRepairDaemon not available: {e}")
        raise


# =============================================================================
# Resource Management Daemons
# =============================================================================


async def create_idle_resource() -> None:
    """Create and run idle resource daemon (Phase 20)."""
    try:
        from app.coordination.idle_resource_daemon import IdleResourceDaemon

        daemon = IdleResourceDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"IdleResourceDaemon not available: {e}")
        raise


async def create_cluster_utilization_watchdog() -> None:
    """Create and run cluster utilization watchdog (Dec 30, 2025).

    Monitors GPU utilization across cluster and emits CLUSTER_UNDERUTILIZED
    events when too many GPUs are idle, enabling proactive remediation.
    """
    try:
        from app.coordination.cluster_utilization_watchdog import (
            ClusterUtilizationWatchdog,
        )

        daemon = ClusterUtilizationWatchdog.get_instance()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ClusterUtilizationWatchdog not available: {e}")
        raise


async def create_node_recovery() -> None:
    """Create and run node recovery daemon (Phase 21)."""
    try:
        from app.coordination.node_recovery_daemon import NodeRecoveryDaemon

        daemon = NodeRecoveryDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"NodeRecoveryDaemon not available: {e}")
        raise


async def create_resource_optimizer() -> None:
    """Create and run resource optimizer daemon (December 2025).

    ResourceOptimizer is a utility class without daemon lifecycle methods,
    so we wrap it in a periodic loop that collects metrics and logs optimization
    recommendations.

    January 2026: Fixed - ResourceOptimizer has no start() method.
    """
    try:
        import asyncio

        from app.coordination.resource_optimizer import ResourceOptimizer

        optimizer = ResourceOptimizer()
        logger.info("[ResourceOptimizer] Starting optimization loop")

        # Run periodic optimization check (every 60 seconds)
        while True:
            try:
                # Update cluster state and get recommendations
                state = optimizer.get_cluster_state(max_age_seconds=120)
                if state and state.total_nodes > 0:
                    recommendation = optimizer.get_optimization_recommendation()
                    if recommendation and recommendation.scale_action != "NONE":
                        logger.info(
                            f"[ResourceOptimizer] Recommendation: {recommendation.scale_action} "
                            f"(CPU: {state.avg_cpu_util:.1f}%, GPU: {state.avg_gpu_util:.1f}%)"
                        )
            except Exception as e:
                logger.debug(f"[ResourceOptimizer] Optimization check failed: {e}")

            await asyncio.sleep(60)
    except ImportError as e:
        logger.error(f"ResourceOptimizer not available: {e}")
        raise


async def create_utilization_optimizer() -> None:
    """Create and run utilization optimizer daemon (December 2025).

    UtilizationOptimizer is a utility class without daemon lifecycle methods,
    so we wrap it in a periodic loop that calls optimize_cluster().
    """
    try:
        import asyncio

        from app.coordination.utilization_optimizer import UtilizationOptimizer

        optimizer = UtilizationOptimizer()
        logger.info("[UtilizationOptimizer] Starting optimization loop")

        # Run optimization loop every 5 minutes
        while True:
            try:
                results = await optimizer.optimize_cluster()
                if results:
                    logger.info(f"[UtilizationOptimizer] Optimized {len(results)} nodes")
            except Exception as e:
                logger.error(f"[UtilizationOptimizer] Optimization failed: {e}")

            await asyncio.sleep(300)  # 5 minute interval
    except ImportError as e:
        logger.error(f"UtilizationOptimizer not available: {e}")
        raise


async def create_adaptive_resources() -> None:
    """Create and run adaptive resource manager (December 2025)."""
    try:
        from app.coordination.adaptive_resource_manager import AdaptiveResourceManager

        manager = AdaptiveResourceManager()
        await manager.start()
        await _wait_for_daemon(manager)
    except ImportError as e:
        logger.error(f"AdaptiveResourceManager not available: {e}")
        raise


# =============================================================================
# Provider-Specific Daemons
# =============================================================================


async def create_lambda_idle() -> None:
    """Create and run Lambda idle shutdown daemon.

    DEPRECATED (December 2025): Lambda Labs account permanently terminated.
    Use DaemonType.VAST_IDLE or other provider-specific idle daemons instead.
    This runner returns immediately without starting any daemon.
    Retained for backward compatibility and will be removed in Q2 2026.
    """
    warnings.warn(
        "DaemonType.LAMBDA_IDLE is deprecated. Lambda Labs account was terminated Dec 2025. "
        "Use DaemonType.VAST_IDLE or other provider idle daemons instead. "
        "Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    from app.coordination.unified_idle_shutdown_daemon import (
        create_lambda_idle_daemon,
    )

    daemon = create_lambda_idle_daemon()  # Returns None with deprecation warning
    if daemon is None:
        logger.info("Lambda idle daemon skipped (Lambda Labs account terminated Dec 2025)")
        return
    await daemon.start()
    await _wait_for_daemon(daemon)


async def create_vast_idle() -> None:
    """Create and run Vast.ai idle shutdown daemon (December 2025)."""
    try:
        from app.coordination.unified_idle_shutdown_daemon import (
            create_vast_idle_daemon,
        )

        daemon = create_vast_idle_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"Vast idle daemon not available: {e}")
        raise


async def create_multi_provider() -> None:
    """Create and run multi-provider orchestrator (December 2025)."""
    try:
        from app.coordination.multi_provider_orchestrator import (
            MultiProviderOrchestrator,
        )

        orchestrator = MultiProviderOrchestrator()
        await orchestrator.start()
        await _wait_for_daemon(orchestrator)
    except ImportError as e:
        logger.error(f"MultiProviderOrchestrator not available: {e}")
        raise


# =============================================================================
# Queue & Job Daemons
# =============================================================================


async def create_queue_populator() -> None:
    """Create and run queue populator daemon (Phase 4)."""
    try:
        from app.coordination.unified_queue_populator import UnifiedQueuePopulatorDaemon

        # Use the daemon wrapper which has start/stop lifecycle
        daemon = UnifiedQueuePopulatorDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"UnifiedQueuePopulatorDaemon not available: {e}")
        raise


async def create_job_scheduler() -> None:
    """Initialize the job scheduler singleton.

    Note: PriorityJobScheduler is a utility class for job prioritization,
    not a daemon. This runner just ensures it's initialized and accessible.
    The scheduler is used by other daemons (like IdleResourceDaemon) to
    get job priorities.
    """
    try:
        from app.coordination.job_scheduler import get_scheduler

        # Initialize the singleton - it's a utility class, not a daemon
        scheduler = get_scheduler()
        logger.info(f"JobScheduler initialized with queue size: {scheduler.get_queue_depth()}")
        # No daemon loop needed - other components use get_scheduler() to access it
    except ImportError as e:
        logger.error(f"JobScheduler not available: {e}")
        raise


# =============================================================================
# Feedback & Curriculum Daemons
# =============================================================================


async def create_feedback_loop() -> None:
    """Create and run feedback loop controller (December 2025)."""
    try:
        from app.coordination.feedback_loop_controller import FeedbackLoopController

        controller = FeedbackLoopController()
        await controller.start()
        await _wait_for_daemon(controller)
    except ImportError as e:
        logger.error(f"FeedbackLoopController not available: {e}")
        raise


async def create_curriculum_integration() -> None:
    """Create and run curriculum integration daemon (December 2025)."""
    try:
        from app.coordination.curriculum_integration import (
            MomentumToCurriculumBridge,
        )

        bridge = MomentumToCurriculumBridge()
        # Dec 2025 fix: start() is synchronous (uses threading internally), don't await
        bridge.start()
        await _wait_for_daemon(bridge)
    except ImportError as e:
        logger.error(f"MomentumToCurriculumBridge not available: {e}")
        raise


# =============================================================================
# Recovery & Maintenance Daemons
# =============================================================================


async def create_recovery_orchestrator() -> None:
    """Create and run recovery orchestrator (December 2025)."""
    try:
        from app.coordination.recovery_orchestrator import RecoveryOrchestrator

        orchestrator = RecoveryOrchestrator()
        await orchestrator.start()
        await _wait_for_daemon(orchestrator)
    except ImportError as e:
        logger.error(f"RecoveryOrchestrator not available: {e}")
        raise


async def create_cache_coordination() -> None:
    """Create and run cache coordination orchestrator (December 2025).

    CacheCoordinationOrchestrator is event-driven. We instantiate it,
    call subscribe_to_events(), and keep it alive.
    """
    try:
        from app.coordination.cache_coordination_orchestrator import (
            CacheCoordinationOrchestrator,
        )

        orchestrator = CacheCoordinationOrchestrator()
        # CacheCoordinationOrchestrator needs explicit subscription call
        orchestrator.subscribe_to_events()
        logger.info("CacheCoordinationOrchestrator initialized and subscribed to events")

        # Keep alive by waiting indefinitely
        while True:
            await asyncio.sleep(60)
    except ImportError as e:
        logger.error(f"CacheCoordinationOrchestrator not available: {e}")
        raise


async def create_maintenance() -> None:
    """Create and run maintenance daemon (December 2025)."""
    try:
        from app.coordination.maintenance_daemon import MaintenanceDaemon

        daemon = MaintenanceDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"MaintenanceDaemon not available: {e}")
        raise


async def create_orphan_detection() -> None:
    """Create and run orphan detection daemon (December 2025)."""
    try:
        from app.coordination.orphan_detection_daemon import OrphanDetectionDaemon

        daemon = OrphanDetectionDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"OrphanDetectionDaemon not available: {e}")
        raise


async def create_data_cleanup() -> None:
    """Create and run data cleanup daemon (December 2025)."""
    try:
        from app.coordination.data_cleanup_daemon import DataCleanupDaemon

        daemon = DataCleanupDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"DataCleanupDaemon not available: {e}")
        raise


async def create_disk_space_manager() -> None:
    """Create and run disk space manager daemon (December 2025).

    Proactive disk space management:
    - Monitors disk usage across nodes
    - Triggers cleanup at 60% (before 70% warning)
    - Removes old logs, empty databases, old checkpoints
    - Emits DISK_SPACE_LOW and DISK_CLEANUP_TRIGGERED events
    """
    try:
        from app.coordination.disk_space_manager_daemon import DiskSpaceManagerDaemon

        daemon = DiskSpaceManagerDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"DiskSpaceManagerDaemon not available: {e}")
        raise


async def create_coordinator_disk_manager() -> None:
    """Create and run coordinator disk manager daemon (December 27, 2025).

    Specialized disk management for coordinator-only nodes:
    - Syncs data to remote storage (OWC on mac-studio) before cleanup
    - More aggressive cleanup thresholds (50% vs 60%)
    - Removes synced training/game files after 24 hours
    - Keeps canonical databases locally for quick access

    January 13, 2026: Added coordinator check to prevent running on GPU nodes.
    Only runs if RINGRIFT_IS_COORDINATOR=true environment variable is set.
    """
    import os

    is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in (
        "true",
        "1",
        "yes",
    )
    if not is_coordinator:
        logger.info(
            "[CoordinatorDiskManager] Skipping - not a coordinator node "
            "(set RINGRIFT_IS_COORDINATOR=true to enable)"
        )
        return

    try:
        from app.coordination.disk_space_manager_daemon import CoordinatorDiskManager

        daemon = CoordinatorDiskManager()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"CoordinatorDiskManager not available: {e}")
        raise


async def create_node_availability() -> None:
    """Create and run node availability daemon (December 28, 2025).

    Synchronizes cloud provider instance state with distributed_hosts.yaml:
    - Queries Vast.ai, Lambda, RunPod APIs for current instance states
    - Updates YAML status when instances change (terminated, stopped, etc.)
    - Atomic YAML updates with backup
    - Dry-run mode by default for testing

    Solves the problem of stale config where nodes are marked 'ready'
    but are actually terminated in the cloud provider.
    """
    try:
        from app.coordination.node_availability.daemon import (
            NodeAvailabilityDaemon,
            get_node_availability_daemon,
        )

        daemon = get_node_availability_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"NodeAvailabilityDaemon not available: {e}")
        raise


async def create_sync_push() -> None:
    """Create and run sync push daemon (December 28, 2025).

    Push-based sync for GPU training nodes:
    - At 50% disk: Start pushing completed games to coordinator
    - At 70% disk: Push urgently (bypass write locks if safe)
    - At 75% disk: Clean up files with 2+ verified copies
    - Never delete files without verified sync receipts

    This is part of the "sync then clean" pattern to prevent data loss
    on GPU nodes that fill up during selfplay.
    """
    try:
        from app.coordination.sync_push_daemon import SyncPushDaemon

        daemon = SyncPushDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"SyncPushDaemon not available: {e}")
        raise


async def create_unified_data_plane() -> None:
    """Create and run unified data plane daemon (December 28, 2025).

    Consolidated data synchronization daemon that replaces fragmented sync
    infrastructure (~4,514 LOC consolidated):
    - AutoSyncDaemon (P2P gossip sync)
    - SyncFacade (backend routing)
    - S3NodeSyncDaemon (S3 backup)
    - dynamic_data_distribution.py (OWC distribution)
    - SyncRouter (intelligent routing)

    Key Components:
    - DataCatalog: Central registry of what data exists where
    - SyncPlanner v2: Intelligent routing (what to sync where and when)
    - TransportManager: Unified transfer layer with fallback chains
    - EventBridge: Event routing with chain completion

    Event Chain Completion:
        SELFPLAY_COMPLETE → DATA_SYNC_STARTED → DATA_SYNC_COMPLETED → NEW_GAMES_AVAILABLE
        TRAINING_COMPLETED → MODEL_PROMOTED → MODEL_DISTRIBUTION_STARTED → MODEL_DISTRIBUTION_COMPLETE
    """
    try:
        from app.coordination.unified_data_plane_daemon import (
            UnifiedDataPlaneDaemon,
            get_data_plane_daemon,
        )

        daemon = get_data_plane_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"UnifiedDataPlaneDaemon not available: {e}")
        raise


# =============================================================================
# Miscellaneous Daemons
# =============================================================================


async def create_s3_backup() -> None:
    """Create and run S3 backup daemon (December 2025)."""
    try:
        from app.coordination.s3_backup_daemon import S3BackupDaemon

        daemon = S3BackupDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"S3BackupDaemon not available: {e}")
        raise


async def create_s3_node_sync() -> None:
    """Create and run S3 node sync daemon (December 2025).

    Bi-directional S3 sync for all cluster nodes.
    """
    try:
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"S3NodeSyncDaemon not available: {e}")
        raise


async def create_s3_consolidation() -> None:
    """Create and run S3 consolidation daemon (December 2025).

    Consolidates data from all nodes (coordinator only).
    """
    try:
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        daemon = S3ConsolidationDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"S3ConsolidationDaemon not available: {e}")
        raise


async def create_unified_backup() -> None:
    """Create and run unified backup daemon (January 2026).

    Backs up all selfplay games to OWC external drive and S3.
    Uses GameDiscovery to find databases from all storage patterns.
    """
    try:
        from app.coordination.unified_backup_daemon import UnifiedBackupDaemon

        daemon = UnifiedBackupDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"UnifiedBackupDaemon not available: {e}")
        raise


async def create_s3_push() -> None:
    """Create and run S3 push daemon (January 2026).

    Pushes all game databases, training NPZ files, and models to S3
    for backup and cluster-wide access.
    """
    try:
        from app.coordination.s3_push_daemon import S3PushDaemon

        daemon = S3PushDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"S3PushDaemon not available: {e}")
        raise


async def create_distillation() -> None:
    """Create and run distillation daemon.

    Jan 3, 2026: DEPRECATED - Standalone script at scripts/distillation_daemon.py.
    """
    logger.warning(
        "[DEPRECATED] DistillationDaemon is a standalone script. "
        "Use scripts/distillation_daemon.py directly. "
        "This daemon will be removed in Q2 2026."
    )
    # Don't raise - allow daemon manager to continue with other daemons
    return


async def create_external_drive_sync() -> None:
    """Create and run external drive sync daemon."""
    try:
        from app.coordination.external_drive_sync import ExternalDriveSyncDaemon

        daemon = ExternalDriveSyncDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ExternalDriveSyncDaemon not available: {e}")
        raise


async def create_vast_cpu_pipeline() -> None:
    """Create and run Vast CPU pipeline daemon."""
    try:
        from app.distributed.vast_cpu_pipeline import VastCpuPipelineDaemon

        daemon = VastCpuPipelineDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"VastCpuPipelineDaemon not available: {e}")
        raise


async def create_cluster_data_sync() -> None:
    """Create and run cluster data sync daemon.

    DEPRECATED (December 2025): Use AutoSyncDaemon with strategy="broadcast" instead.
    This runner is retained for backward compatibility and will be removed in Q2 2026.
    """
    warnings.warn(
        "DaemonType.CLUSTER_DATA_SYNC is deprecated. Use AutoSyncDaemon(strategy='broadcast') "
        "instead. Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from app.coordination.auto_sync_daemon import (
            AutoSyncConfig,
            AutoSyncDaemon,
            SyncStrategy,
        )

        config = AutoSyncConfig.from_config_file()
        config.strategy = SyncStrategy.BROADCAST
        daemon = AutoSyncDaemon(config=config)
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"AutoSyncDaemon not available for cluster sync: {e}")
        raise


async def create_p2p_backend() -> None:
    """Create and run P2P backend daemon (December 2025)."""
    try:
        from app.coordination.p2p_integration import P2PIntegration

        integration = P2PIntegration()
        await integration.start()
        await _wait_for_daemon(integration)
    except ImportError as e:
        logger.error(f"P2PIntegration not available: {e}")
        raise


async def create_p2p_auto_deploy() -> None:
    """Create and run P2P auto-deploy daemon (Phase 21.2)."""
    try:
        from app.coordination.p2p_auto_deployer import P2PAutoDeployer

        deployer = P2PAutoDeployer()
        await deployer.run_daemon()
        await _wait_for_daemon(deployer)
    except ImportError as e:
        logger.error(f"P2PAutoDeployer not available: {e}")
        raise


async def create_metrics_analysis() -> None:
    """Create and run metrics analysis orchestrator (December 2025)."""
    try:
        from app.coordination.metrics_analysis_orchestrator import (
            MetricsAnalysisOrchestrator,
        )

        orchestrator = MetricsAnalysisOrchestrator()
        await orchestrator.start()
        await _wait_for_daemon(orchestrator)
    except ImportError as e:
        logger.error(f"MetricsAnalysisOrchestrator not available: {e}")
        raise


async def create_per_orchestrator() -> None:
    """Create and run PER (Prioritized Experience Replay) orchestrator (December 2025).

    Monitors PER buffer events across the training system. Subscribes to
    PER_BUFFER_REBUILT and PER_PRIORITIES_UPDATED events.
    """
    import asyncio

    try:
        from app.training.per_orchestrator import wire_per_events

        # Wire events and get orchestrator
        orchestrator = wire_per_events()
        if orchestrator:
            logger.info("[PEROrchestrator] Started and subscribed to events")
            # Keep daemon running
            while True:
                await asyncio.sleep(60)
    except ImportError as e:
        logger.error(f"PEROrchestrator not available: {e}")
        raise


async def create_data_consolidation() -> None:
    """Create and run data consolidation daemon (December 2025).

    Consolidates scattered selfplay games into canonical databases.
    Subscribes to SELFPLAY_COMPLETE and NEW_GAMES_AVAILABLE events.
    Emits CONSOLIDATION_COMPLETE event after merging games.
    """
    try:
        from app.coordination.data_consolidation_daemon import get_consolidation_daemon

        daemon = get_consolidation_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"DataConsolidationDaemon not available: {e}")
        raise


async def create_cluster_consolidation() -> None:
    """Create and run cluster consolidation daemon (January 2026).

    Pulls selfplay games from cluster nodes into canonical databases.
    Critical for training pipeline: bridges distributed selfplay with training.

    Flow:
        1. Discovers alive cluster nodes via P2P
        2. Syncs selfplay.db from each node via Tailscale/SSH
        3. Merges games into canonical_*.db databases
        4. Emits CONSOLIDATION_COMPLETE event for training trigger

    Runs on coordinator only (configurable).
    """
    try:
        from app.coordination.cluster_consolidation_daemon import (
            get_cluster_consolidation_daemon,
        )

        daemon = get_cluster_consolidation_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ClusterConsolidationDaemon not available: {e}")
        raise


async def create_comprehensive_consolidation() -> None:
    """Create and run comprehensive consolidation daemon (January 2026).

    Scheduled sweep consolidation that finds ALL games across 14+ storage patterns.
    Unlike DATA_CONSOLIDATION (event-driven), this runs on a schedule to ensure
    no games are missed from:
    - owc_imports/ - Games imported from OWC external drive
    - synced/ - Games synced from P2P cluster
    - p2p_gpu/ - Games from GPU selfplay nodes
    - gumbel/ - High-quality Gumbel MCTS games
    - slurm/ - Games from SLURM cluster runs
    - And 9+ other patterns

    Flow:
        1. Runs on schedule (default: every 30 minutes)
        2. Uses GameDiscovery to scan all 14+ database patterns
        3. Merges all games into canonical_*.db databases
        4. Updates consolidation_tracking.db with progress
        5. Emits COMPREHENSIVE_CONSOLIDATION_COMPLETE event

    Sprint 1 of Unified Data Consolidation plan.
    """
    try:
        from app.coordination.comprehensive_consolidation_daemon import (
            get_comprehensive_consolidation_daemon,
        )

        daemon = get_comprehensive_consolidation_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ComprehensiveConsolidationDaemon not available: {e}")
        raise


async def create_unified_data_sync_orchestrator() -> None:
    """Create and run unified data sync orchestrator daemon (January 2026).

    Central coordinator for all data synchronization operations:
    - Listens to DATA_SYNC_COMPLETED events from AutoSyncDaemon
    - Triggers S3 and OWC backups based on event flags
    - Tracks replication status across all storage destinations
    - Provides unified visibility into data distribution

    Part of unified data sync plan for comprehensive backup and visibility.
    """
    try:
        from app.coordination.unified_data_sync_orchestrator import (
            get_unified_data_sync_orchestrator,
        )

        daemon = get_unified_data_sync_orchestrator()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"UnifiedDataSyncOrchestrator not available: {e}")
        raise


async def create_npz_combination() -> None:
    """Create and run NPZ combination daemon (December 2025).

    Combines multiple NPZ training files with quality-aware weighting.
    Subscribes to NPZ_EXPORT_COMPLETE event to trigger combination.
    Emits NPZ_COMBINATION_COMPLETE event after combining files.
    """
    try:
        from app.coordination.npz_combination_daemon import get_npz_combination_daemon

        daemon = get_npz_combination_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"NPZCombinationDaemon not available: {e}")
        raise


# =============================================================================
# Data Integrity (December 2025)
# =============================================================================


async def create_integrity_check() -> None:
    """Create and run integrity check daemon (December 2025).

    Periodically scans game databases for integrity issues, specifically:
    - Games without move data (orphan games)
    - Quarantines invalid games for review
    - Cleans up old quarantined games

    Part of Phase 6: Move Data Integrity Enforcement.
    """
    try:
        from app.coordination.integrity_check_daemon import (
            IntegrityCheckConfig,
            IntegrityCheckDaemon,
        )

        config = IntegrityCheckConfig.from_env()
        daemon = IntegrityCheckDaemon(config=config)
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"IntegrityCheckDaemon not available: {e}")
        raise


# =============================================================================
# Cluster Availability Manager (December 28, 2025)
# =============================================================================


async def create_availability_node_monitor() -> None:
    """Create and run availability NodeMonitor daemon.

    Multi-layer health checking for cluster nodes:
    - P2P heartbeat checks
    - SSH connectivity
    - GPU health monitoring
    - Provider API status

    Emits NODE_UNHEALTHY, NODE_RECOVERED events.
    """
    try:
        from app.coordination.availability.node_monitor import get_node_monitor

        daemon = get_node_monitor()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"NodeMonitor not available: {e}")
        raise


async def create_availability_recovery_engine() -> None:
    """Create and run availability RecoveryEngine daemon.

    Escalating recovery strategies:
    - RESTART_P2P
    - RESTART_TAILSCALE
    - REBOOT_INSTANCE
    - RECREATE_INSTANCE

    Subscribes to NODE_UNHEALTHY events.
    Emits RECOVERY_INITIATED, RECOVERY_COMPLETED events.
    """
    try:
        from app.coordination.availability.recovery_engine import get_recovery_engine

        daemon = get_recovery_engine()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"RecoveryEngine not available: {e}")
        raise


async def create_availability_capacity_planner() -> None:
    """Create and run availability CapacityPlanner daemon.

    Budget-aware capacity management:
    - Tracks hourly/daily spending
    - Provides scaling recommendations
    - Monitors cluster utilization

    Subscribes to NODE_PROVISIONED, NODE_TERMINATED events.
    Emits BUDGET_ALERT events.
    """
    try:
        from app.coordination.availability.capacity_planner import get_capacity_planner

        daemon = get_capacity_planner()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"CapacityPlanner not available: {e}")
        raise


async def create_cascade_training() -> None:
    """Create and run CascadeTrainingOrchestrator daemon.

    December 29, 2025: Multiplayer bootstrapping via cascade training.
    Orchestrates the 2p → 3p → 4p training cascade:
    - Monitors model quality/Elo for each board type
    - Triggers weight transfer when quality threshold met
    - Boosts selfplay priority for configs blocking cascade

    Subscribes to: TRAINING_COMPLETED, EVALUATION_COMPLETED, MODEL_PROMOTED, ELO_UPDATED
    Emits: CASCADE_TRANSFER_TRIGGERED, TRAINING_REQUESTED
    """
    try:
        from app.coordination.cascade_training import get_cascade_orchestrator

        daemon = get_cascade_orchestrator()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"CascadeTrainingOrchestrator not available: {e}")
        raise


async def create_availability_provisioner() -> None:
    """Create and run availability Provisioner daemon.

    Auto-provision new instances when capacity drops below thresholds.
    Respects budget constraints via CapacityPlanner integration.

    Subscribes to CAPACITY_LOW, NODE_FAILED_PERMANENTLY events.
    Emits NODE_PROVISIONED, PROVISION_FAILED events.
    """
    try:
        from app.coordination.availability.provisioner import get_provisioner

        daemon = get_provisioner()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"Provisioner not available: {e}")
        raise


# =============================================================================
# 48-Hour Autonomous Operation Daemons (December 29, 2025)
# =============================================================================


async def create_progress_watchdog() -> None:
    """Create and run ProgressWatchdog daemon.

    December 29, 2025: Part of 48-hour autonomous operation enablement.
    Monitors Elo velocity across all 12 configs and detects training stalls.

    - Tracks Elo velocity per config
    - Detects stalls (6+ hours without progress)
    - Triggers recovery via PROGRESS_STALL_DETECTED event

    Subscribes to: ELO_UPDATED
    Emits: PROGRESS_STALL_DETECTED, PROGRESS_RECOVERED
    """
    try:
        from app.coordination.progress_watchdog_daemon import get_progress_watchdog

        daemon = get_progress_watchdog()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ProgressWatchdog not available: {e}")
        raise


async def create_p2p_recovery() -> None:
    """Create and run P2PRecovery daemon.

    December 29, 2025: Part of 48-hour autonomous operation enablement.
    Monitors P2P cluster health and auto-restarts on partition/failure.

    - Checks P2P /status endpoint every 60s
    - Tracks consecutive failures
    - Restarts P2P after 3 consecutive failures
    - Respects 5-minute cooldown between restarts

    Emits: P2P_RESTART_TRIGGERED, P2P_HEALTH_RECOVERED
    """
    try:
        from app.coordination.p2p_recovery_daemon import get_p2p_recovery_daemon

        daemon = get_p2p_recovery_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"P2PRecovery not available: {e}")
        raise


async def create_voter_health_monitor() -> None:
    """Create and run VoterHealthMonitor daemon.

    December 30, 2025: Part of 48-hour autonomous operation enablement.
    Monitors individual voter health with multi-transport probing.

    Probing order:
    - P2P HTTP /health endpoint (5s timeout)
    - Tailscale ping (10s timeout)
    - SSH connectivity check (15s timeout)

    Key events:
    - VOTER_OFFLINE: Individual voter became unreachable (after 2 consecutive failures)
    - VOTER_ONLINE: Individual voter recovered
    - QUORUM_LOST: Online voters dropped below quorum threshold
    - QUORUM_RESTORED: Quorum regained
    - QUORUM_AT_RISK: Exactly at quorum threshold (marginal)

    Subscribes to: None (cycle-based, 30s interval)
    Emits: VOTER_OFFLINE, VOTER_ONLINE, QUORUM_LOST, QUORUM_RESTORED, QUORUM_AT_RISK
    """
    try:
        from app.coordination.voter_health_daemon import get_voter_health_daemon

        daemon = get_voter_health_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"VoterHealthMonitor not available: {e}")
        raise


async def create_memory_monitor() -> None:
    """Create and run MemoryMonitor daemon.

    December 30, 2025: Part of 48-hour autonomous operation enablement.
    Monitors GPU VRAM and process RSS to detect memory pressure before OOM.

    Key thresholds:
    - GPU VRAM: 75% warning, 85% critical → Emit MEMORY_PRESSURE, pause spawning
    - System RAM: 80% warning, 90% critical → Emit RESOURCE_CONSTRAINT
    - Process RSS: 32GB critical → SIGTERM → wait 60s → SIGKILL

    Subscribes to: None (cycle-based)
    Emits: MEMORY_PRESSURE, RESOURCE_CONSTRAINT
    """
    try:
        from app.coordination.memory_monitor_daemon import get_memory_monitor

        daemon = get_memory_monitor()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"MemoryMonitor not available: {e}")
        raise


async def create_socket_leak_recovery() -> None:
    """Create and run SocketLeakRecovery daemon.

    January 2026: Part of 48-hour autonomous operation enablement.
    Monitors socket connections and file descriptors to detect and recover
    from resource leaks before they cause system instability.

    Key features:
    - Monitors TIME_WAIT, CLOSE_WAIT socket buildup
    - Monitors file descriptor exhaustion
    - Triggers connection pool cleanup when critical
    - Part of resource recovery infrastructure with MEMORY_MONITOR

    Subscribes to: RESOURCE_CONSTRAINT
    Emits: SOCKET_LEAK_DETECTED, SOCKET_LEAK_RECOVERED, P2P_CONNECTION_RESET_REQUESTED
    """
    try:
        from app.coordination.socket_leak_recovery_daemon import (
            get_socket_leak_recovery_daemon,
        )

        daemon = get_socket_leak_recovery_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"SocketLeakRecoveryDaemon not available: {e}")
        raise


async def create_stale_fallback() -> None:
    """Create and run StaleFallback controller.

    December 30, 2025: Part of 48-hour autonomous operation enablement.
    Enables graceful degradation when sync fails by using older model versions.

    Note: TrainingFallbackController is a utility controller, not a full daemon.
    This runner initializes it and subscribes to relevant events.

    Subscribes to: DATA_SYNC_FAILED, DATA_SYNC_COMPLETED
    Emits: STALE_TRAINING_FALLBACK (via controller.should_allow_training)
    """
    try:
        from app.coordination.stale_fallback import get_training_fallback_controller

        controller = get_training_fallback_controller()

        # Subscribe to sync events
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                router.subscribe(
                    "DATA_SYNC_FAILED",
                    lambda e: controller.record_sync_failure(e.get("config_key", "unknown")),
                    handler_name="stale_fallback_sync_failed",
                )
                router.subscribe(
                    "DATA_SYNC_COMPLETED",
                    lambda e: controller.record_sync_success(e.get("config_key", "unknown")),
                    handler_name="stale_fallback_sync_completed",
                )
                logger.info("[StaleFallback] Subscribed to sync events")
        except Exception as e:
            logger.debug(f"[StaleFallback] Could not subscribe to events: {e}")

        # Keep running (controller is stateful)
        while True:
            await asyncio.sleep(60)
    except ImportError as e:
        logger.error(f"StaleFallback not available: {e}")
        raise


async def create_underutilization_recovery() -> None:
    """Create and run UnderutilizationRecoveryHandler.

    January 4, 2026: Phase 3 of P2P Cluster Resilience plan.

    Handles CLUSTER_UNDERUTILIZED and WORK_QUEUE_EXHAUSTED events by injecting
    high-priority work items for underserved configurations.

    Subscribes to: CLUSTER_UNDERUTILIZED, WORK_QUEUE_EXHAUSTED
    Emits: UTILIZATION_RECOVERY_STARTED, UTILIZATION_RECOVERY_COMPLETED,
           UTILIZATION_RECOVERY_FAILED
    """
    try:
        from app.coordination.underutilization_recovery_handler import (
            get_underutilization_handler,
        )

        handler = get_underutilization_handler()
        await handler.start()
        await _wait_for_daemon(handler)
    except ImportError as e:
        logger.error(f"UnderutilizationRecoveryHandler not available: {e}")
        raise


async def create_fast_failure_detector() -> None:
    """Create and run FastFailureDetector.

    January 4, 2026: Phase 4 of P2P Cluster Resilience plan.

    Detects cluster-wide failures within 5-10 minutes using tiered escalation:
    - Tier 1 (5 min): Warning log
    - Tier 2 (10 min): Emit FAST_FAILURE_ALERT, boost selfplay 1.5x
    - Tier 3 (30 min): Emit FAST_FAILURE_RECOVERY, boost selfplay 2x

    Emits: FAST_FAILURE_ALERT, FAST_FAILURE_RECOVERY, FAST_FAILURE_RECOVERED
    """
    try:
        from app.coordination.fast_failure_detector import get_fast_failure_detector

        detector = get_fast_failure_detector()
        await detector.start()
        await _wait_for_daemon(detector)
    except ImportError as e:
        logger.error(f"FastFailureDetector not available: {e}")
        raise


async def create_tailscale_health() -> None:
    """Create and run TailscaleHealthDaemon.

    December 29, 2025: Monitors and auto-recovers Tailscale connectivity.

    Runs on each cluster node to ensure P2P mesh stability:
    - Checks Tailscale status every 30s (configurable)
    - Detects disconnected/expired states
    - Auto-recovers via tailscale up or tailscaled restart
    - Supports userspace networking for containers

    Emits: TAILSCALE_DISCONNECTED, TAILSCALE_RECOVERED, TAILSCALE_RECOVERY_FAILED
    """
    try:
        from app.coordination.tailscale_health_daemon import get_tailscale_health_daemon

        daemon = get_tailscale_health_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"TailscaleHealthDaemon not available: {e}")
        raise


async def create_connectivity_recovery() -> None:
    """Create and run ConnectivityRecoveryCoordinator.

    December 29, 2025: Unified event-driven connectivity recovery.

    Bridges TailscaleHealthDaemon, NodeAvailabilityDaemon, and P2P orchestrator:
    - Subscribes to TAILSCALE_DISCONNECTED, TAILSCALE_RECOVERED events
    - Subscribes to P2P_NODE_DEAD, HOST_OFFLINE, HOST_ONLINE events
    - Triggers SSH-based Tailscale recovery for masked failures
    - Escalates persistent failures with Slack alerts
    - Tracks connectivity state for all nodes

    Emits: TAILSCALE_RECOVERED (after SSH recovery), CONNECTIVITY_RECOVERY_ESCALATED
    """
    try:
        from app.coordination.connectivity_recovery_coordinator import (
            get_connectivity_recovery_coordinator,
        )

        daemon = get_connectivity_recovery_coordinator()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ConnectivityRecoveryCoordinator not available: {e}")
        raise


# =============================================================================
# Data Availability Infrastructure (Jan 3, 2026)
# =============================================================================


async def create_dual_backup() -> None:
    """Create and run dual backup daemon (S3 + OWC)."""
    try:
        from app.coordination.dual_backup_daemon import (
            create_dual_backup as _create,
        )

        await _create()
    except ImportError as e:
        logger.error(f"DualBackupDaemon not available: {e}")
        raise


async def create_owc_push() -> None:
    """Create and run OWC push daemon."""
    try:
        from app.coordination.owc_push_daemon import (
            create_owc_push as _create,
        )

        await _create()
    except ImportError as e:
        logger.error(f"OWCPushDaemon not available: {e}")
        raise


async def create_s3_import() -> None:
    """Create and run S3 import daemon."""
    try:
        from app.coordination.s3_import_daemon import (
            create_s3_import as _create,
        )

        await _create()
    except ImportError as e:
        logger.error(f"S3ImportDaemon not available: {e}")
        raise


async def create_unified_data_catalog() -> None:
    """Create and run unified data catalog daemon (placeholder)."""
    logger.warning(
        "UNIFIED_DATA_CATALOG daemon is not yet implemented - "
        "this is a placeholder for future unified data catalog API"
    )
    # Placeholder - daemon does nothing but prevents startup failure
    await asyncio.sleep(float("inf"))


async def create_node_data_agent() -> None:
    """Create and run node data agent daemon (placeholder)."""
    logger.warning(
        "NODE_DATA_AGENT daemon is not yet implemented - "
        "this is a placeholder for future per-node data agent"
    )
    # Placeholder - daemon does nothing but prevents startup failure
    await asyncio.sleep(float("inf"))


async def create_online_merge() -> None:
    """Create and run online model merge daemon.

    January 2026: Part of human game training pipeline.
    Validates shadow models that have received online learning updates
    and merges them into canonical models if they show improvement.
    """
    try:
        from app.coordination.online_merge_daemon import get_online_merge_daemon

        daemon = get_online_merge_daemon()
        await daemon.start()
        await daemon.wait_until_stopped()
    except ImportError as e:
        logger.error(f"OnlineMergeDaemon not available: {e}")
        raise


# =============================================================================
# Runner Registry
# =============================================================================

# Lazy-built registry mapping DaemonType to runner function
_RUNNER_REGISTRY: dict[str, Callable[[], Coroutine[None, None, None]]] | None = None


def _build_runner_registry() -> dict[str, Callable[[], Coroutine[None, None, None]]]:
    """Build the runner registry lazily."""
    from app.coordination.daemon_types import DaemonType

    return {
        DaemonType.SYNC_COORDINATOR.name: create_sync_coordinator,
        DaemonType.HIGH_QUALITY_SYNC.name: create_high_quality_sync,
        DaemonType.ELO_SYNC.name: create_elo_sync,
        DaemonType.AUTO_SYNC.name: create_auto_sync,
        DaemonType.CONFIG_SYNC.name: create_config_sync,  # Jan 12, 2026: Config sync
        DaemonType.CONFIG_VALIDATOR.name: create_config_validator,  # Jan 12, 2026: Config validation
        DaemonType.TRAINING_NODE_WATCHER.name: create_training_node_watcher,
        DaemonType.TRAINING_DATA_SYNC.name: create_training_data_sync,
        DaemonType.TRAINING_DATA_RECOVERY.name: create_training_data_recovery,
        DaemonType.TRAINING_WATCHDOG.name: create_training_watchdog,  # Jan 4, 2026: Sprint 17
        DaemonType.EXPORT_WATCHDOG.name: create_export_watchdog,  # Jan 6, 2026: Session 17.41
        DaemonType.OWC_IMPORT.name: create_owc_import,
        # January 3, 2026 (Sprint 13 Session 4): Model evaluation automation
        DaemonType.OWC_MODEL_IMPORT.name: create_owc_model_import,
        DaemonType.UNEVALUATED_MODEL_SCANNER.name: create_unevaluated_model_scanner,
        DaemonType.STALE_EVALUATION.name: create_stale_evaluation,  # Jan 4, 2026: Sprint 17
        DaemonType.COMPREHENSIVE_MODEL_SCAN.name: create_comprehensive_model_scan,  # Jan 9, 2026: Sprint 17.9
        DaemonType.EPHEMERAL_SYNC.name: create_ephemeral_sync,
        DaemonType.GOSSIP_SYNC.name: create_gossip_sync,
        DaemonType.EVENT_ROUTER.name: create_event_router,
        DaemonType.CROSS_PROCESS_POLLER.name: create_cross_process_poller,
        DaemonType.DLQ_RETRY.name: create_dlq_retry,
        DaemonType.HEALTH_CHECK.name: create_health_check,
        DaemonType.QUEUE_MONITOR.name: create_queue_monitor,
        DaemonType.DAEMON_WATCHDOG.name: create_daemon_watchdog,
        DaemonType.NODE_HEALTH_MONITOR.name: create_node_health_monitor,
        DaemonType.SYSTEM_HEALTH_MONITOR.name: create_system_health_monitor,
        DaemonType.HEALTH_SERVER.name: create_health_server,
        DaemonType.QUALITY_MONITOR.name: create_quality_monitor,
        DaemonType.MODEL_PERFORMANCE_WATCHDOG.name: create_model_performance_watchdog,
        DaemonType.CLUSTER_MONITOR.name: create_cluster_monitor,
        DaemonType.CLUSTER_WATCHDOG.name: create_cluster_watchdog,
        DaemonType.COORDINATOR_HEALTH_MONITOR.name: create_coordinator_health_monitor,
        DaemonType.WORK_QUEUE_MONITOR.name: create_work_queue_monitor,
        DaemonType.DATA_PIPELINE.name: create_data_pipeline,
        DaemonType.CONTINUOUS_TRAINING_LOOP.name: create_continuous_training_loop,
        DaemonType.SELFPLAY_COORDINATOR.name: create_selfplay_coordinator,
        DaemonType.TRAINING_TRIGGER.name: create_training_trigger,
        DaemonType.AUTO_EXPORT.name: create_auto_export,
        DaemonType.NPZ_COMBINATION.name: create_npz_combination,  # Dec 2025: NPZ combination
        DaemonType.TOURNAMENT_DAEMON.name: create_tournament_daemon,
        DaemonType.EVALUATION.name: create_evaluation_daemon,
        DaemonType.AUTO_PROMOTION.name: create_auto_promotion,
        DaemonType.UNIFIED_PROMOTION.name: create_unified_promotion,
        DaemonType.GAUNTLET_FEEDBACK.name: create_gauntlet_feedback,
        DaemonType.MODEL_SYNC.name: create_model_sync,
        DaemonType.MODEL_DISTRIBUTION.name: create_model_distribution,
        DaemonType.NPZ_DISTRIBUTION.name: create_npz_distribution,
        DaemonType.DATA_SERVER.name: create_data_server,
        DaemonType.REPLICATION_MONITOR.name: create_replication_monitor,
        DaemonType.REPLICATION_REPAIR.name: create_replication_repair,
        DaemonType.IDLE_RESOURCE.name: create_idle_resource,
        DaemonType.CLUSTER_UTILIZATION_WATCHDOG.name: create_cluster_utilization_watchdog,
        DaemonType.NODE_RECOVERY.name: create_node_recovery,
        DaemonType.RESOURCE_OPTIMIZER.name: create_resource_optimizer,
        DaemonType.UTILIZATION_OPTIMIZER.name: create_utilization_optimizer,
        DaemonType.ADAPTIVE_RESOURCES.name: create_adaptive_resources,
        DaemonType.LAMBDA_IDLE.name: create_lambda_idle,
        DaemonType.VAST_IDLE.name: create_vast_idle,
        DaemonType.MULTI_PROVIDER.name: create_multi_provider,
        DaemonType.QUEUE_POPULATOR.name: create_queue_populator,
        DaemonType.JOB_SCHEDULER.name: create_job_scheduler,
        DaemonType.FEEDBACK_LOOP.name: create_feedback_loop,
        DaemonType.CURRICULUM_INTEGRATION.name: create_curriculum_integration,
        DaemonType.RECOVERY_ORCHESTRATOR.name: create_recovery_orchestrator,
        DaemonType.CACHE_COORDINATION.name: create_cache_coordination,
        DaemonType.MAINTENANCE.name: create_maintenance,
        DaemonType.ORPHAN_DETECTION.name: create_orphan_detection,
        DaemonType.DATA_CLEANUP.name: create_data_cleanup,
        DaemonType.DISK_SPACE_MANAGER.name: create_disk_space_manager,
        DaemonType.COORDINATOR_DISK_MANAGER.name: create_coordinator_disk_manager,
        DaemonType.NODE_AVAILABILITY.name: create_node_availability,
        DaemonType.SYNC_PUSH.name: create_sync_push,
        DaemonType.UNIFIED_DATA_PLANE.name: create_unified_data_plane,
        DaemonType.S3_BACKUP.name: create_s3_backup,
        DaemonType.S3_NODE_SYNC.name: create_s3_node_sync,
        DaemonType.S3_CONSOLIDATION.name: create_s3_consolidation,
        DaemonType.UNIFIED_BACKUP.name: create_unified_backup,  # Jan 2026: OWC + S3 backup
        DaemonType.S3_PUSH.name: create_s3_push,  # Jan 2026: S3 backup push
        DaemonType.DISTILLATION.name: create_distillation,
        DaemonType.EXTERNAL_DRIVE_SYNC.name: create_external_drive_sync,
        DaemonType.VAST_CPU_PIPELINE.name: create_vast_cpu_pipeline,
        DaemonType.CLUSTER_DATA_SYNC.name: create_cluster_data_sync,
        DaemonType.P2P_BACKEND.name: create_p2p_backend,
        DaemonType.P2P_AUTO_DEPLOY.name: create_p2p_auto_deploy,
        DaemonType.METRICS_ANALYSIS.name: create_metrics_analysis,
        DaemonType.DATA_CONSOLIDATION.name: create_data_consolidation,
        DaemonType.CLUSTER_CONSOLIDATION.name: create_cluster_consolidation,
        DaemonType.COMPREHENSIVE_CONSOLIDATION.name: create_comprehensive_consolidation,  # Jan 2026: Scheduled sweep
        DaemonType.UNIFIED_DATA_SYNC_ORCHESTRATOR.name: create_unified_data_sync_orchestrator,
        DaemonType.INTEGRITY_CHECK.name: create_integrity_check,
        # Cluster Availability Manager (December 28, 2025)
        DaemonType.AVAILABILITY_NODE_MONITOR.name: create_availability_node_monitor,
        DaemonType.AVAILABILITY_RECOVERY_ENGINE.name: create_availability_recovery_engine,
        DaemonType.AVAILABILITY_CAPACITY_PLANNER.name: create_availability_capacity_planner,
        DaemonType.AVAILABILITY_PROVISIONER.name: create_availability_provisioner,
        # Cascade training (December 29, 2025)
        DaemonType.CASCADE_TRAINING.name: create_cascade_training,
        # PER orchestrator (December 29, 2025)
        DaemonType.PER_ORCHESTRATOR.name: create_per_orchestrator,
        # 48-Hour Autonomous Operation (December 29-30, 2025)
        DaemonType.PROGRESS_WATCHDOG.name: create_progress_watchdog,
        DaemonType.P2P_RECOVERY.name: create_p2p_recovery,
        DaemonType.VOTER_HEALTH_MONITOR.name: create_voter_health_monitor,
        DaemonType.MEMORY_MONITOR.name: create_memory_monitor,
        DaemonType.SOCKET_LEAK_RECOVERY.name: create_socket_leak_recovery,
        DaemonType.STALE_FALLBACK.name: create_stale_fallback,
        DaemonType.UNDERUTILIZATION_RECOVERY.name: create_underutilization_recovery,  # Jan 4, 2026: Phase 3 P2P Resilience
        DaemonType.FAST_FAILURE_DETECTOR.name: create_fast_failure_detector,  # Jan 4, 2026: Phase 4 P2P Resilience
        # Tailscale health monitoring (December 29, 2025)
        DaemonType.TAILSCALE_HEALTH.name: create_tailscale_health,
        # Connectivity recovery coordinator (December 29, 2025)
        DaemonType.CONNECTIVITY_RECOVERY.name: create_connectivity_recovery,
        # NNUE automatic training (December 29, 2025)
        DaemonType.NNUE_TRAINING.name: create_nnue_training,
        # Architecture feedback controller (December 29, 2025)
        DaemonType.ARCHITECTURE_FEEDBACK.name: create_architecture_feedback,
        # Parity validation daemon (December 30, 2025)
        DaemonType.PARITY_VALIDATION.name: create_parity_validation,
        # Elo progress tracking (December 31, 2025)
        DaemonType.ELO_PROGRESS.name: create_elo_progress,
        # Data availability infrastructure (January 3, 2026)
        DaemonType.DUAL_BACKUP.name: create_dual_backup,
        DaemonType.OWC_PUSH.name: create_owc_push,
        DaemonType.S3_IMPORT.name: create_s3_import,
        DaemonType.UNIFIED_DATA_CATALOG.name: create_unified_data_catalog,
        DaemonType.NODE_DATA_AGENT.name: create_node_data_agent,
        # Online model merge daemon (January 2026)
        DaemonType.ONLINE_MERGE.name: create_online_merge,
    }


def get_runner(
    daemon_type: DaemonType,
) -> Callable[[], Coroutine[None, None, None]] | None:
    """Get the runner function for a daemon type.

    Args:
        daemon_type: The daemon type to get a runner for

    Returns:
        The async runner function, or None if not found
    """
    global _RUNNER_REGISTRY
    if _RUNNER_REGISTRY is None:
        _RUNNER_REGISTRY = _build_runner_registry()
    return _RUNNER_REGISTRY.get(daemon_type.name)


def get_all_runners() -> dict[str, Callable[[], Coroutine[None, None, None]]]:
    """Get all registered runner functions.

    Returns:
        Dictionary mapping daemon type names to runner functions
    """
    global _RUNNER_REGISTRY
    if _RUNNER_REGISTRY is None:
        _RUNNER_REGISTRY = _build_runner_registry()
    return _RUNNER_REGISTRY.copy()
