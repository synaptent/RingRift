"""Centralized daemon factory for lazy loading.

This module consolidates all daemon imports into a single location,
breaking circular dependencies and providing a clean registry pattern.

Instead of 78 deferred imports scattered throughout daemon_manager.py,
all daemon creation is now routed through this factory.

Usage:
    from app.coordination.daemon_factory import get_daemon_factory

    factory = get_daemon_factory()
    daemon = factory.create(DaemonType.AUTO_SYNC)
    await daemon.start()

December 2025 - Phase 1.2 architecture cleanup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from app.coordination.daemon_types import DaemonType

logger = logging.getLogger(__name__)


@dataclass
class DaemonImportSpec:
    """Specification for lazy-loading a daemon class.

    Note: This is different from daemon_registry.DaemonSpec which specifies
    daemon runner configuration. This class specifies import paths for
    lazy class loading to avoid circular imports.

    December 2025: Renamed from DaemonSpec to avoid collision with
    daemon_registry.DaemonSpec.
    """

    import_path: str  # Full import path e.g., "app.coordination.auto_sync_daemon"
    class_name: str   # Class name e.g., "AutoSyncDaemon"
    factory_fn: str | None = None  # Optional factory function e.g., "get_auto_sync_daemon"
    singleton: bool = True  # Whether to cache the instance


# Backward compatibility alias
DaemonSpec = DaemonImportSpec


# Registry of all daemon types
# Lazy loaded to avoid circular imports
_DAEMON_REGISTRY: dict[str, DaemonImportSpec] = {}


def _build_registry() -> dict[str, DaemonImportSpec]:
    """Build the daemon registry lazily.

    This function is called once on first use to avoid circular imports
    during module initialization.
    """
    # Import DaemonType here to avoid circular import
    from app.coordination.daemon_types import DaemonType

    return {
        # =================================================================
        # Core Infrastructure Daemons
        # =================================================================
        DaemonType.EVENT_ROUTER.name: DaemonSpec(
            import_path="app.coordination.event_router",
            class_name="UnifiedEventRouter",
            factory_fn="get_router",
        ),
        DaemonType.CROSS_PROCESS_POLLER.name: DaemonSpec(
            import_path="app.coordination.cross_process_events",
            class_name="CrossProcessEventPoller",
        ),
        DaemonType.DAEMON_WATCHDOG.name: DaemonSpec(
            import_path="app.coordination.daemon_watchdog",
            class_name="DaemonWatchdog",
            factory_fn="start_watchdog",
        ),

        # =================================================================
        # Sync Daemons
        # =================================================================
        DaemonType.AUTO_SYNC.name: DaemonSpec(
            import_path="app.coordination.auto_sync_daemon",
            class_name="AutoSyncDaemon",
        ),
        DaemonType.GOSSIP_SYNC.name: DaemonSpec(
            import_path="app.distributed.gossip_sync",
            class_name="GossipSyncDaemon",
        ),
        # EPHEMERAL_SYNC: Updated Dec 2025 to use unified auto_sync_daemon
        DaemonType.EPHEMERAL_SYNC.name: DaemonSpec(
            import_path="app.coordination.auto_sync_daemon",
            class_name="AutoSyncDaemon",
            factory_fn="get_ephemeral_sync_daemon",
        ),
        # MODEL_SYNC: Deprecated (Dec 2025) - use MODEL_DISTRIBUTION instead
        # The unified distribution daemon handles both model and NPZ sync
        DaemonType.MODEL_SYNC.name: DaemonSpec(
            import_path="app.coordination.unified_distribution_daemon",
            class_name="UnifiedDistributionDaemon",
            factory_fn="create_model_distribution_daemon",
        ),
        DaemonType.MODEL_DISTRIBUTION.name: DaemonSpec(
            import_path="app.coordination.unified_distribution_daemon",
            class_name="UnifiedDistributionDaemon",
            factory_fn="create_model_distribution_daemon",
        ),
        DaemonType.NPZ_DISTRIBUTION.name: DaemonSpec(
            import_path="app.coordination.unified_distribution_daemon",
            class_name="UnifiedDistributionDaemon",
            factory_fn="create_npz_distribution_daemon",
        ),
        DaemonType.EXTERNAL_DRIVE_SYNC.name: DaemonSpec(
            import_path="app.distributed.external_drive_sync",
            class_name="ExternalDriveSyncDaemon",
        ),
        # CLUSTER_DATA_SYNC: Updated Dec 2025 to use unified auto_sync_daemon
        DaemonType.CLUSTER_DATA_SYNC.name: DaemonSpec(
            import_path="app.coordination.auto_sync_daemon",
            class_name="AutoSyncDaemon",
            factory_fn="create_cluster_data_sync_daemon",
        ),
        DaemonType.HIGH_QUALITY_SYNC.name: DaemonSpec(
            import_path="app.distributed.sync_coordinator",
            class_name="HighQualityDataSyncWatcher",
        ),
        DaemonType.ELO_SYNC.name: DaemonSpec(
            import_path="app.tournament.elo_sync_manager",
            class_name="EloSyncManager",
        ),

        # =================================================================
        # Training & Pipeline Daemons
        # =================================================================
        DaemonType.DATA_PIPELINE.name: DaemonSpec(
            import_path="app.coordination.data_pipeline_orchestrator",
            class_name="DataPipelineOrchestrator",
        ),
        DaemonType.UNIFIED_PROMOTION.name: DaemonSpec(
            import_path="app.training.promotion_controller",
            class_name="PromotionController",
        ),
        DaemonType.AUTO_PROMOTION.name: DaemonSpec(
            import_path="app.coordination.auto_promotion_daemon",
            class_name="AutoPromotionDaemon",
            factory_fn="get_auto_promotion_daemon",
        ),
        DaemonType.DISTILLATION.name: DaemonSpec(
            import_path="app.coordination.daemon_adapters",
            class_name="DistillationDaemonAdapter",
        ),
        DaemonType.CONTINUOUS_TRAINING_LOOP.name: DaemonSpec(
            import_path="app.coordination.continuous_loop",
            class_name="ContinuousTrainingLoop",
        ),
        DaemonType.SELFPLAY_COORDINATOR.name: DaemonSpec(
            import_path="app.coordination.selfplay_scheduler",
            class_name="SelfplayScheduler",
        ),

        # =================================================================
        # Evaluation & Tournament Daemons
        # =================================================================
        DaemonType.EVALUATION.name: DaemonSpec(
            import_path="app.coordination.evaluation_daemon",
            class_name="EvaluationDaemon",
            factory_fn="get_evaluation_daemon",
        ),
        DaemonType.TOURNAMENT_DAEMON.name: DaemonSpec(
            import_path="app.coordination.tournament_daemon",
            class_name="TournamentDaemon",
            factory_fn="get_tournament_daemon",
        ),
        # Dec 2025: GAUNTLET_FEEDBACK now delegates to UnifiedFeedbackOrchestrator
        # which consolidates all 4 feedback systems. The old GauntletFeedbackController
        # is deprecated. See unified_feedback.py for the replacement.
        DaemonType.GAUNTLET_FEEDBACK.name: DaemonSpec(
            import_path="app.coordination.unified_feedback",
            class_name="UnifiedFeedbackOrchestrator",
            factory_fn="get_unified_feedback",
        ),

        # =================================================================
        # Health & Monitoring Daemons
        # =================================================================
        DaemonType.HEALTH_CHECK.name: DaemonSpec(
            import_path="app.distributed.health_checks",
            class_name="HealthChecker",
        ),
        DaemonType.NODE_HEALTH_MONITOR.name: DaemonSpec(
            import_path="app.coordination.unified_node_health_daemon",
            class_name="UnifiedNodeHealthDaemon",
        ),
        DaemonType.QUALITY_MONITOR.name: DaemonSpec(
            import_path="app.coordination.quality_monitor_daemon",
            class_name="QualityMonitorDaemon",
            factory_fn="create_quality_monitor",
        ),
        DaemonType.MODEL_PERFORMANCE_WATCHDOG.name: DaemonSpec(
            import_path="app.coordination.model_performance_watchdog",
            class_name="ModelPerformanceWatchdog",
        ),
        DaemonType.ORPHAN_DETECTION.name: DaemonSpec(
            import_path="app.coordination.orphan_detection_daemon",
            class_name="OrphanDetectionDaemon",
        ),

        # =================================================================
        # Cluster Management Daemons
        # =================================================================
        DaemonType.CLUSTER_MONITOR.name: DaemonSpec(
            import_path="app.coordination.cluster_status_monitor",
            class_name="ClusterMonitor",
        ),
        DaemonType.P2P_BACKEND.name: DaemonSpec(
            import_path="app.coordination.p2p_backend",
            class_name="P2PBackend",
        ),
        DaemonType.P2P_AUTO_DEPLOY.name: DaemonSpec(
            import_path="app.coordination.p2p_auto_deployer",
            class_name="P2PAutoDeployer",
        ),
        DaemonType.DATA_SERVER.name: DaemonSpec(
            import_path="app.distributed.sync_coordinator",
            class_name="SyncCoordinator",
            factory_fn="get_instance",  # Uses singleton pattern
        ),
        DaemonType.QUEUE_MONITOR.name: DaemonSpec(
            import_path="app.coordination.queue_monitor",
            class_name="QueueMonitor",
        ),
        DaemonType.WORK_QUEUE_MONITOR.name: DaemonSpec(
            import_path="app.coordination.work_queue_monitor_daemon",
            class_name="WorkQueueMonitorDaemon",
            factory_fn="get_work_queue_monitor_sync",
        ),
        DaemonType.COORDINATOR_HEALTH_MONITOR.name: DaemonSpec(
            import_path="app.coordination.coordinator_health_monitor_daemon",
            class_name="CoordinatorHealthMonitorDaemon",
            factory_fn="get_coordinator_health_monitor_sync",
        ),
        DaemonType.SYNC_COORDINATOR.name: DaemonSpec(
            import_path="app.distributed.sync_coordinator",
            class_name="SyncCoordinator",
        ),
        DaemonType.REPLICATION_MONITOR.name: DaemonSpec(
            import_path="app.coordination.unified_replication_daemon",
            class_name="UnifiedReplicationDaemon",
            factory_fn="create_replication_monitor",
        ),
        DaemonType.REPLICATION_REPAIR.name: DaemonSpec(
            import_path="app.coordination.unified_replication_daemon",
            class_name="UnifiedReplicationDaemon",
            factory_fn="create_replication_repair_daemon",
        ),
        # TRAINING_NODE_WATCHER: Updated Dec 2025 to use unified auto_sync_daemon with BROADCAST strategy
        DaemonType.TRAINING_NODE_WATCHER.name: DaemonSpec(
            import_path="app.coordination.auto_sync_daemon",
            class_name="AutoSyncDaemon",
            factory_fn="create_training_sync_daemon",
        ),

        # =================================================================
        # Job Scheduling Daemons
        # =================================================================
        DaemonType.JOB_SCHEDULER.name: DaemonSpec(
            import_path="app.coordination.job_scheduler",
            class_name="PriorityJobScheduler",
        ),
        DaemonType.RESOURCE_OPTIMIZER.name: DaemonSpec(
            import_path="app.coordination.resource_optimizer",
            class_name="ResourceOptimizer",
        ),
        DaemonType.VAST_CPU_PIPELINE.name: DaemonSpec(
            import_path="app.distributed.vast_cpu_pipeline",
            class_name="VastCpuPipelineDaemon",
        ),

        # =================================================================
        # Backup & External Storage
        # =================================================================
        DaemonType.S3_BACKUP.name: DaemonSpec(
            import_path="app.coordination.s3_backup_daemon",
            class_name="S3BackupDaemon",
        ),
        DaemonType.S3_NODE_SYNC.name: DaemonSpec(
            import_path="app.coordination.s3_node_sync_daemon",
            class_name="S3NodeSyncDaemon",
        ),
        DaemonType.S3_CONSOLIDATION.name: DaemonSpec(
            import_path="app.coordination.s3_node_sync_daemon",
            class_name="S3ConsolidationDaemon",
        ),

        # =================================================================
        # Resource & Utilization Daemons
        # =================================================================
        DaemonType.IDLE_RESOURCE.name: DaemonSpec(
            import_path="app.coordination.idle_resource_daemon",
            class_name="IdleResourceDaemon",
        ),
        DaemonType.QUEUE_POPULATOR.name: DaemonSpec(
            import_path="app.coordination.unified_queue_populator",
            class_name="UnifiedQueuePopulator",
        ),
        DaemonType.NODE_RECOVERY.name: DaemonSpec(
            import_path="app.coordination.node_recovery_daemon",
            class_name="NodeRecoveryDaemon",
            factory_fn="get_node_recovery_daemon",  # Dec 2025: Use singleton factory
        ),
        DaemonType.UTILIZATION_OPTIMIZER.name: DaemonSpec(
            import_path="app.coordination.utilization_optimizer",
            class_name="UtilizationOptimizer",
        ),
        DaemonType.CLUSTER_WATCHDOG.name: DaemonSpec(
            import_path="app.coordination.cluster_watchdog_daemon",
            class_name="ClusterWatchdogDaemon",
            factory_fn="get_cluster_watchdog_daemon",  # Dec 2025: Use singleton factory
        ),

        # =================================================================
        # Pipeline Automation Daemons
        # =================================================================
        DaemonType.AUTO_EXPORT.name: DaemonSpec(
            import_path="app.coordination.auto_export_daemon",
            class_name="AutoExportDaemon",
        ),
        DaemonType.TRAINING_TRIGGER.name: DaemonSpec(
            import_path="app.coordination.training_trigger_daemon",
            class_name="TrainingTriggerDaemon",
        ),
        DaemonType.DATA_CLEANUP.name: DaemonSpec(
            import_path="app.coordination.data_cleanup_daemon",
            class_name="DataCleanupDaemon",
        ),
        DaemonType.DATA_CONSOLIDATION.name: DaemonSpec(
            import_path="app.coordination.data_consolidation_daemon",
            class_name="DataConsolidationDaemon",
            factory_fn="get_consolidation_daemon",
        ),
        DaemonType.DISK_SPACE_MANAGER.name: DaemonSpec(
            import_path="app.coordination.disk_space_manager_daemon",
            class_name="DiskSpaceManagerDaemon",
        ),
        DaemonType.COORDINATOR_DISK_MANAGER.name: DaemonSpec(
            import_path="app.coordination.disk_space_manager_daemon",
            class_name="CoordinatorDiskManager",
            factory_fn="get_coordinator_disk_daemon",
        ),
        DaemonType.DLQ_RETRY.name: DaemonSpec(
            import_path="app.coordination.dead_letter_queue",
            class_name="DLQRetryDaemon",
        ),

        # =================================================================
        # Health & System Monitoring
        # =================================================================
        # SYSTEM_HEALTH_MONITOR: Updated Dec 2025 to use unified_health_manager
        DaemonType.SYSTEM_HEALTH_MONITOR.name: DaemonSpec(
            import_path="app.coordination.unified_health_manager",
            class_name="UnifiedHealthManager",
            factory_fn="get_health_manager",
        ),
        DaemonType.MAINTENANCE.name: DaemonSpec(
            import_path="app.coordination.maintenance_daemon",
            class_name="MaintenanceDaemon",
        ),

        # =================================================================
        # Cost Optimization Daemons
        # =================================================================
        DaemonType.LAMBDA_IDLE.name: DaemonSpec(
            import_path="app.coordination.unified_idle_shutdown_daemon",
            class_name="UnifiedIdleShutdownDaemon",
            factory_fn="create_lambda_idle_daemon",  # Uses provider-specific config
        ),
        DaemonType.VAST_IDLE.name: DaemonSpec(
            import_path="app.coordination.unified_idle_shutdown_daemon",
            class_name="UnifiedIdleShutdownDaemon",
            factory_fn="create_vast_idle_daemon",  # Uses provider-specific config
        ),

        # =================================================================
        # Feedback & Curriculum Daemons
        # =================================================================
        # Dec 2025: FEEDBACK_LOOP now uses UnifiedFeedbackOrchestrator which
        # consolidates FeedbackLoopController, GauntletFeedbackController,
        # CurriculumIntegration, and TrainingFreshness into one system.
        DaemonType.FEEDBACK_LOOP.name: DaemonSpec(
            import_path="app.coordination.unified_feedback",
            class_name="UnifiedFeedbackOrchestrator",
            factory_fn="get_unified_feedback",
        ),

        # =================================================================
        # Orchestration Daemons (December 2025)
        # =================================================================
        DaemonType.RECOVERY_ORCHESTRATOR.name: DaemonSpec(
            import_path="app.coordination.recovery_orchestrator",
            class_name="RecoveryOrchestrator",
        ),
        DaemonType.CACHE_COORDINATION.name: DaemonSpec(
            import_path="app.coordination.cache_coordination_orchestrator",
            class_name="CacheCoordinationOrchestrator",
        ),
        DaemonType.METRICS_ANALYSIS.name: DaemonSpec(
            import_path="app.coordination.metrics_analysis_orchestrator",
            class_name="MetricsAnalysisOrchestrator",
        ),
        DaemonType.ADAPTIVE_RESOURCES.name: DaemonSpec(
            import_path="app.coordination.adaptive_resource_manager",
            class_name="AdaptiveResourceManager",
        ),
        DaemonType.MULTI_PROVIDER.name: DaemonSpec(
            import_path="app.coordination.multi_provider_orchestrator",
            class_name="MultiProviderOrchestrator",
        ),
        DaemonType.CURRICULUM_INTEGRATION.name: DaemonSpec(
            import_path="app.coordination.curriculum_integration",
            class_name="MomentumToCurriculumBridge",  # Main integration class
        ),
        DaemonType.HEALTH_SERVER.name: DaemonSpec(
            import_path="app.coordination.unified_health_manager",
            class_name="UnifiedHealthManager",  # Provides health endpoints
        ),

        # =================================================================
        # Data Integrity & Sync Push Daemons (December 2025)
        # =================================================================
        DaemonType.INTEGRITY_CHECK.name: DaemonSpec(
            import_path="app.coordination.integrity_check_daemon",
            class_name="IntegrityCheckDaemon",
        ),
        DaemonType.SYNC_PUSH.name: DaemonSpec(
            import_path="app.coordination.sync_push_daemon",
            class_name="SyncPushDaemon",
            factory_fn="get_sync_push_daemon",  # Uses singleton pattern
        ),

        # =================================================================
        # S3/Cloud Storage Daemons (December 2025)
        # =================================================================
        DaemonType.S3_NODE_SYNC.name: DaemonSpec(
            import_path="app.coordination.s3_node_sync_daemon",
            class_name="S3NodeSyncDaemon",
        ),
        DaemonType.S3_CONSOLIDATION.name: DaemonSpec(
            import_path="app.coordination.s3_node_sync_daemon",
            class_name="S3ConsolidationDaemon",
        ),
        DaemonType.DATA_CONSOLIDATION.name: DaemonSpec(
            import_path="app.coordination.data_consolidation_daemon",
            class_name="DataConsolidationDaemon",
        ),
    }


class DaemonFactory:
    """Factory for creating daemon instances.

    Uses lazy loading to avoid importing daemon modules until needed.
    Supports both singleton (cached) and fresh instance creation.
    """

    def __init__(self) -> None:
        self._registry: dict[str, DaemonSpec] = {}
        self._instances: dict[str, Any] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize registry on first use."""
        if not self._initialized:
            self._registry = _build_registry()
            self._initialized = True

    def create(
        self,
        daemon_type: DaemonType | str,
        *args: Any,
        force_new: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create or get a daemon instance.

        Args:
            daemon_type: Type of daemon to create
            *args: Positional arguments for daemon constructor
            force_new: Force creation of new instance even if singleton exists
            **kwargs: Keyword arguments for daemon constructor

        Returns:
            Daemon instance

        Raises:
            ValueError: If daemon type is not registered
            ImportError: If daemon module cannot be imported
        """
        self._ensure_initialized()

        # Convert to string key
        if hasattr(daemon_type, 'name'):
            key = daemon_type.name
        else:
            key = str(daemon_type)

        # Check cache for singletons
        spec = self._registry.get(key)
        if spec is None:
            raise ValueError(f"Unknown daemon type: {key}")

        if spec.singleton and not force_new and key in self._instances:
            return self._instances[key]

        # Import and create
        instance = self._import_and_create(spec, *args, **kwargs)

        # Cache singletons
        if spec.singleton:
            self._instances[key] = instance

        return instance

    def _import_and_create(
        self,
        spec: DaemonSpec,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Import the daemon module and create instance."""
        import importlib

        try:
            module = importlib.import_module(spec.import_path)
        except ImportError as e:
            logger.error(f"Failed to import {spec.import_path}: {e}")
            raise

        # Use factory function if specified
        if spec.factory_fn:
            factory = getattr(module, spec.factory_fn, None)
            if factory is None:
                logger.warning(
                    f"Factory function {spec.factory_fn} not found in {spec.import_path}, "
                    f"falling back to class {spec.class_name}"
                )
            else:
                try:
                    return factory(*args, **kwargs)
                except TypeError:
                    # Factory might not accept arguments
                    return factory()

        # Direct class instantiation
        cls = getattr(module, spec.class_name)
        return cls(*args, **kwargs)

    def get_spec(self, daemon_type: DaemonType | str) -> DaemonSpec | None:
        """Get the specification for a daemon type."""
        self._ensure_initialized()

        if hasattr(daemon_type, 'name'):
            key = daemon_type.name
        else:
            key = str(daemon_type)

        return self._registry.get(key)

    def register(
        self,
        daemon_type: DaemonType | str,
        spec: DaemonSpec,
    ) -> None:
        """Register a daemon type dynamically.

        Useful for plugins or tests that need to register custom daemons.
        """
        self._ensure_initialized()

        if hasattr(daemon_type, 'name'):
            key = daemon_type.name
        else:
            key = str(daemon_type)

        self._registry[key] = spec

    def clear_cache(self) -> None:
        """Clear the singleton instance cache.

        Useful for testing or when reinitializing daemons.
        """
        self._instances.clear()

    def list_registered(self) -> list[str]:
        """List all registered daemon types."""
        self._ensure_initialized()
        return list(self._registry.keys())


# Module-level singleton
_daemon_factory: DaemonFactory | None = None


def get_daemon_factory() -> DaemonFactory:
    """Get the singleton daemon factory."""
    global _daemon_factory
    if _daemon_factory is None:
        _daemon_factory = DaemonFactory()
    return _daemon_factory


def reset_daemon_factory() -> None:
    """Reset the singleton for testing."""
    global _daemon_factory
    if _daemon_factory is not None:
        _daemon_factory.clear_cache()
    _daemon_factory = None


__all__ = [
    "DaemonFactory",
    "DaemonSpec",
    "get_daemon_factory",
    "reset_daemon_factory",
]
