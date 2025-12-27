"""Daemon runner functions extracted from DaemonManager.

This module contains the async runner functions for each daemon type.
These functions handle:
- Importing the daemon class
- Creating and configuring the daemon instance
- Starting the daemon and waiting for completion

December 2025 - Code quality refactoring to reduce daemon_manager.py size.

Usage:
    from app.coordination.daemon_runners import get_runner

    runner = get_runner(DaemonType.AUTO_SYNC)
    await runner()
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from app.coordination.daemon_types import DaemonType

logger = logging.getLogger(__name__)


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
        await manager.sync_loop()
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
    """Create and run gossip sync daemon."""
    try:
        from app.distributed.gossip_sync import GossipSyncDaemon

        daemon = GossipSyncDaemon()
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
    """Create and run cross-process event poller daemon."""
    try:
        from app.coordination.event_router import CrossProcessEventPoller

        poller = CrossProcessEventPoller()
        await poller.start()
        await _wait_for_daemon(poller)
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
    """
    try:
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
    """Create and run continuous training loop daemon."""
    try:
        from app.coordination.continuous_training_loop import ContinuousTrainingLoop

        loop = ContinuousTrainingLoop()
        await loop.start()
        await _wait_for_daemon(loop)
    except ImportError as e:
        logger.error(f"ContinuousTrainingLoop not available: {e}")
        raise


async def create_selfplay_coordinator() -> None:
    """Create and run selfplay coordinator daemon (December 2025)."""
    try:
        from app.coordination.selfplay_scheduler import SelfplayScheduler

        scheduler = SelfplayScheduler()
        await scheduler.start()
        await _wait_for_daemon(scheduler)
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


# =============================================================================
# Evaluation & Promotion Daemons
# =============================================================================


async def create_evaluation_daemon() -> None:
    """Create and run evaluation daemon (December 2025)."""
    try:
        from app.coordination.evaluation_daemon import EvaluationDaemon

        daemon = EvaluationDaemon()
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
    """Create and run unified promotion daemon (December 2025)."""
    try:
        from app.coordination.promotion_controller import PromotionController

        controller = PromotionController()
        await controller.start()
        await _wait_for_daemon(controller)
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
        await sync.start_server()
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
    """Create and run resource optimizer daemon (December 2025)."""
    try:
        from app.coordination.resource_optimizer import ResourceOptimizer

        optimizer = ResourceOptimizer()
        await optimizer.start()
        await _wait_for_daemon(optimizer)
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

    NOTE: Lambda Labs account was terminated December 2025. This runner
    now logs a warning and returns immediately without starting any daemon.
    Retained for backward compatibility with existing daemon registrations.
    """
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
        from app.coordination.unified_queue_populator import UnifiedQueuePopulator

        populator = UnifiedQueuePopulator()
        await populator.start()
        await _wait_for_daemon(populator)
    except ImportError as e:
        logger.error(f"UnifiedQueuePopulator not available: {e}")
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
        logger.info(f"JobScheduler initialized with queue size: {scheduler.pending_count}")
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
        await bridge.start()
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
    """Create and run cache coordination orchestrator (December 2025)."""
    try:
        from app.coordination.cache_coordination_orchestrator import (
            CacheCoordinationOrchestrator,
        )

        orchestrator = CacheCoordinationOrchestrator()
        await orchestrator.start()
        await _wait_for_daemon(orchestrator)
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
    """
    try:
        from app.coordination.disk_space_manager_daemon import CoordinatorDiskManager

        daemon = CoordinatorDiskManager()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"CoordinatorDiskManager not available: {e}")
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


async def create_distillation() -> None:
    """Create and run distillation daemon."""
    try:
        from app.coordination.distillation_daemon import DistillationDaemon

        daemon = DistillationDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"DistillationDaemon not available: {e}")
        raise


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
        from app.coordination.vast_cpu_pipeline import VastCPUPipelineDaemon

        daemon = VastCPUPipelineDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"VastCPUPipelineDaemon not available: {e}")
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
        await deployer.start()
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
        DaemonType.TRAINING_NODE_WATCHER.name: create_training_node_watcher,
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
        DaemonType.S3_BACKUP.name: create_s3_backup,
        DaemonType.DISTILLATION.name: create_distillation,
        DaemonType.EXTERNAL_DRIVE_SYNC.name: create_external_drive_sync,
        DaemonType.VAST_CPU_PIPELINE.name: create_vast_cpu_pipeline,
        DaemonType.CLUSTER_DATA_SYNC.name: create_cluster_data_sync,
        DaemonType.P2P_BACKEND.name: create_p2p_backend,
        DaemonType.P2P_AUTO_DEPLOY.name: create_p2p_auto_deploy,
        DaemonType.METRICS_ANALYSIS.name: create_metrics_analysis,
        DaemonType.DATA_CONSOLIDATION.name: create_data_consolidation,
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
