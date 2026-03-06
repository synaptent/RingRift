"""Resource Management, Provider-Specific, and Queue & Job daemon runners.

February 2026: Extracted from daemon_runners.py.

Contains runners for:
- Resource Management Daemons (idle_resource through adaptive_resources)
- Provider-Specific Daemons (lambda_idle, vast_idle, multi_provider)
- Queue & Job Daemons (queue_populator, job_scheduler)
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any

from app.coordination.runners import _wait_for_daemon

logger = logging.getLogger(__name__)


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
        import asyncio
        from app.coordination.unified_queue_populator import UnifiedQueuePopulatorDaemon

        # Mar 2026: Wrap in to_thread — UnifiedQueuePopulator.__init__ can block
        # if ensure_game_counts_loaded() is called eagerly by other init paths.
        daemon = await asyncio.to_thread(UnifiedQueuePopulatorDaemon)
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
