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
    # --- Pipeline Completeness Monitor (February 2026) ---
    "pipeline_completeness_monitor": RunnerSpec(
        module="app.coordination.pipeline_completeness_monitor",
        class_name="PipelineCompletenessMonitor",
        style=InstantiationStyle.SINGLETON,
        notes="Feb 2026: Tracks pipeline stage completion, emits PIPELINE_STAGE_OVERDUE",
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
# Runner Functions (February 2026: Decomposed into runners/ subpackage)
# =============================================================================
# All create_* runner functions have been moved to category modules under
# app/coordination/runners/ for improved navigability:
#   - sync_runners.py: Sync + Event Processing daemons
#   - health_runners.py: Health & Monitoring daemons
#   - training_runners.py: Training, Pipeline, Evaluation & Promotion daemons
#   - distribution_runners.py: Distribution + Replication daemons
#   - resource_runners.py: Resource Management, Provider-Specific, Queue & Job daemons
#   - operations_runners.py: Feedback, Recovery, Miscellaneous, Data Integrity,
#                            Cluster Availability, 48-Hour Autonomous, Data Availability
#
# They are re-exported via runners/__init__.py so existing imports continue to work.

from app.coordination.runners import (  # noqa: E402, F401
    # Sync runners
    create_sync_coordinator,
    create_high_quality_sync,
    create_elo_sync,
    create_auto_sync,
    create_config_sync,
    create_config_validator,
    create_training_node_watcher,
    create_training_data_sync,
    create_training_data_recovery,
    create_training_watchdog,
    create_export_watchdog,
    create_owc_import,
    create_owc_model_import,
    create_unevaluated_model_scanner,
    create_stale_evaluation,
    create_comprehensive_model_scan,
    create_ephemeral_sync,
    create_gossip_sync,
    # Event processing runners
    create_event_router,
    create_cross_process_poller,
    create_dlq_retry,
    # Health runners
    create_health_check,
    create_queue_monitor,
    create_daemon_watchdog,
    create_node_health_monitor,
    create_system_health_monitor,
    create_health_server,
    create_quality_monitor,
    create_model_performance_watchdog,
    create_cluster_monitor,
    create_cluster_watchdog,
    create_coordinator_health_monitor,
    create_work_queue_monitor,
    # Training runners
    create_data_pipeline,
    create_continuous_training_loop,
    create_selfplay_coordinator,
    create_training_trigger,
    create_auto_export,
    create_tournament_daemon,
    create_nnue_training,
    create_architecture_feedback,
    create_parity_validation,
    create_elo_progress,
    # Evaluation & promotion runners
    create_evaluation_daemon,
    create_auto_promotion,
    create_unified_promotion,
    create_gauntlet_feedback,
    create_backlog_evaluation,
    # Distribution runners
    create_model_sync,
    create_model_distribution,
    create_npz_distribution,
    create_data_server,
    # Replication runners
    create_replication_monitor,
    create_replication_repair,
    # Resource runners
    create_idle_resource,
    create_cluster_utilization_watchdog,
    create_node_recovery,
    create_resource_optimizer,
    create_utilization_optimizer,
    create_adaptive_resources,
    # Provider-specific runners
    create_lambda_idle,
    create_vast_idle,
    create_multi_provider,
    # Queue & job runners
    create_queue_populator,
    create_job_scheduler,
    # Feedback & curriculum runners
    create_feedback_loop,
    create_curriculum_integration,
    # Recovery & maintenance runners
    create_recovery_orchestrator,
    create_cache_coordination,
    create_maintenance,
    create_orphan_detection,
    create_data_cleanup,
    create_disk_space_manager,
    create_coordinator_disk_manager,
    create_node_availability,
    create_sync_push,
    create_unified_data_plane,
    # Miscellaneous runners
    create_s3_backup,
    create_s3_node_sync,
    create_s3_consolidation,
    create_unified_backup,
    create_s3_push,
    create_distillation,
    create_external_drive_sync,
    create_vast_cpu_pipeline,
    create_cluster_data_sync,
    create_p2p_backend,
    create_p2p_auto_deploy,
    create_metrics_analysis,
    create_per_orchestrator,
    create_data_consolidation,
    create_cluster_consolidation,
    create_comprehensive_consolidation,
    create_unified_data_sync_orchestrator,
    create_npz_combination,
    # Data integrity runners
    create_integrity_check,
    # Cluster availability runners
    create_availability_node_monitor,
    create_availability_recovery_engine,
    create_availability_capacity_planner,
    create_cascade_training,
    create_availability_provisioner,
    # 48-hour autonomous operation runners
    create_progress_watchdog,
    create_p2p_recovery,
    create_voter_health_monitor,
    create_memory_monitor,
    create_socket_leak_recovery,
    create_stale_fallback,
    create_underutilization_recovery,
    create_fast_failure_detector,
    create_tailscale_health,
    create_connectivity_recovery,
    # Data availability runners
    create_dual_backup,
    create_owc_push,
    create_s3_import,
    create_unified_data_catalog,
    create_node_data_agent,
    create_online_merge,
    create_owc_sync_manager,
    create_s3_sync,
    create_production_game_import,
    create_reanalysis,
    create_pipeline_completeness_monitor,
)


# NOTE: The following block is intentionally left as a deleted marker.
# The original ~2,580 lines of create_* function definitions (lines 887-3461)
# have been moved to app/coordination/runners/ submodules.
# See the import block above for the complete list of re-exported functions.


_RUNNER_FUNCTIONS_MOVED = True  # Sentinel to confirm decomposition is active

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
        # Sync and import daemons (February 2026)
        DaemonType.OWC_SYNC_MANAGER.name: create_owc_sync_manager,
        DaemonType.S3_SYNC.name: create_s3_sync,
        DaemonType.PRODUCTION_GAME_IMPORT.name: create_production_game_import,
        DaemonType.REANALYSIS.name: create_reanalysis,
        # Pipeline completeness monitor (February 2026)
        DaemonType.PIPELINE_COMPLETENESS_MONITOR.name: create_pipeline_completeness_monitor,
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
