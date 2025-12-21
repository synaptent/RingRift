"""Coordination Bootstrap - Unified initialization for all coordinators (December 2025).

This module provides a single entry point for initializing the entire
coordination layer. It handles the correct initialization order, event
wiring, and registry registration for all coordinators.

Usage:
    from app.coordination.coordination_bootstrap import (
        bootstrap_coordination,
        shutdown_coordination,
        get_bootstrap_status,
    )

    # Initialize all coordination components
    bootstrap_coordination()

    # Or initialize specific components
    bootstrap_coordination(
        enable_metrics=True,
        enable_optimization=True,
        enable_leadership=False,  # Disable if single-node
    )

    # Check initialization status
    status = get_bootstrap_status()
    print(f"Coordinators initialized: {status['initialized_count']}")

    # Graceful shutdown
    shutdown_coordination()

Benefits:
- Single entry point for coordination initialization
- Correct initialization order (dependencies first)
- Consistent error handling across all coordinators
- Unified status reporting
- Graceful shutdown support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Initialization State
# =============================================================================

@dataclass
class BootstrapCoordinatorStatus:
    """Status of a single coordinator during bootstrap.

    Note: This is distinct from coordinator_base.CoordinatorStatus (an Enum)
    which tracks runtime status. This dataclass tracks initialization state.
    """

    name: str
    initialized: bool = False
    subscribed: bool = False
    error: str | None = None
    initialized_at: datetime | None = None


@dataclass
class BootstrapState:
    """Global bootstrap state."""

    initialized: bool = False
    started_at: datetime | None = None
    completed_at: datetime | None = None
    coordinators: dict[str, BootstrapCoordinatorStatus] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    shutdown_requested: bool = False


_state = BootstrapState()


# =============================================================================
# Initialization Functions
# =============================================================================

def _init_resource_coordinator() -> BootstrapCoordinatorStatus:
    """Initialize ResourceMonitoringCoordinator."""
    status = BootstrapCoordinatorStatus(name="resource_coordinator")
    try:
        from app.coordination.resource_monitoring_coordinator import wire_resource_events

        coordinator = wire_resource_events()
        status.initialized = True
        status.subscribed = coordinator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] ResourceMonitoringCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] ResourceMonitoringCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize ResourceMonitoringCoordinator: {e}")

    return status


def _init_metrics_orchestrator() -> BootstrapCoordinatorStatus:
    """Initialize MetricsAnalysisOrchestrator."""
    status = BootstrapCoordinatorStatus(name="metrics_orchestrator")
    try:
        from app.coordination.metrics_analysis_orchestrator import wire_metrics_events

        orchestrator = wire_metrics_events()
        status.initialized = True
        status.subscribed = orchestrator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] MetricsAnalysisOrchestrator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] MetricsAnalysisOrchestrator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize MetricsAnalysisOrchestrator: {e}")

    return status


def _init_optimization_coordinator() -> BootstrapCoordinatorStatus:
    """Initialize OptimizationCoordinator."""
    status = BootstrapCoordinatorStatus(name="optimization_coordinator")
    try:
        from app.coordination.optimization_coordinator import wire_optimization_events

        coordinator = wire_optimization_events()
        status.initialized = True
        status.subscribed = coordinator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] OptimizationCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] OptimizationCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize OptimizationCoordinator: {e}")

    return status


def _init_cache_orchestrator() -> BootstrapCoordinatorStatus:
    """Initialize CacheCoordinationOrchestrator."""
    status = BootstrapCoordinatorStatus(name="cache_orchestrator")
    try:
        from app.coordination.cache_coordination_orchestrator import wire_cache_events

        orchestrator = wire_cache_events()
        status.initialized = True
        status.subscribed = orchestrator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] CacheCoordinationOrchestrator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] CacheCoordinationOrchestrator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize CacheCoordinationOrchestrator: {e}")

    return status


def _init_model_coordinator() -> BootstrapCoordinatorStatus:
    """Initialize ModelLifecycleCoordinator."""
    status = BootstrapCoordinatorStatus(name="model_coordinator")
    try:
        from app.coordination.model_lifecycle_coordinator import wire_model_events

        coordinator = wire_model_events()
        status.initialized = True
        status.subscribed = coordinator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] ModelLifecycleCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] ModelLifecycleCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize ModelLifecycleCoordinator: {e}")

    return status


def _init_health_manager() -> BootstrapCoordinatorStatus:
    """Initialize UnifiedHealthManager (replaces ErrorRecoveryCoordinator + RecoveryManager)."""
    status = BootstrapCoordinatorStatus(name="health_manager")
    try:
        from app.coordination.unified_health_manager import wire_health_events

        manager = wire_health_events()
        status.initialized = True
        status.subscribed = manager._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] UnifiedHealthManager initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] UnifiedHealthManager not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize UnifiedHealthManager: {e}")

    return status


def _init_error_coordinator() -> BootstrapCoordinatorStatus:
    """Initialize ErrorRecoveryCoordinator (DEPRECATED - use _init_health_manager)."""
    # For backward compatibility, delegate to health manager
    return _init_health_manager()


def _init_leadership_coordinator() -> BootstrapCoordinatorStatus:
    """Initialize LeadershipCoordinator."""
    status = BootstrapCoordinatorStatus(name="leadership_coordinator")
    try:
        from app.coordination.leadership_coordinator import wire_leadership_events

        coordinator = wire_leadership_events()
        status.initialized = True
        status.subscribed = coordinator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] LeadershipCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] LeadershipCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize LeadershipCoordinator: {e}")

    return status


def _init_selfplay_orchestrator() -> BootstrapCoordinatorStatus:
    """Initialize SelfplayOrchestrator."""
    status = BootstrapCoordinatorStatus(name="selfplay_orchestrator")
    try:
        from app.coordination.selfplay_orchestrator import wire_selfplay_events

        orchestrator = wire_selfplay_events()
        status.initialized = True
        status.subscribed = orchestrator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] SelfplayOrchestrator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] SelfplayOrchestrator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize SelfplayOrchestrator: {e}")

    return status


def _init_pipeline_orchestrator(auto_trigger: bool = False) -> BootstrapCoordinatorStatus:
    """Initialize DataPipelineOrchestrator."""
    status = BootstrapCoordinatorStatus(name="pipeline_orchestrator")
    try:
        from app.coordination.data_pipeline_orchestrator import wire_pipeline_events

        orchestrator = wire_pipeline_events(auto_trigger=auto_trigger)
        status.initialized = True
        status.subscribed = orchestrator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] DataPipelineOrchestrator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] DataPipelineOrchestrator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize DataPipelineOrchestrator: {e}")

    return status


def _init_task_coordinator() -> BootstrapCoordinatorStatus:
    """Initialize TaskLifecycleCoordinator."""
    status = BootstrapCoordinatorStatus(name="task_coordinator")
    try:
        from app.coordination.task_lifecycle_coordinator import wire_task_events

        coordinator = wire_task_events()
        status.initialized = True
        status.subscribed = coordinator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] TaskLifecycleCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] TaskLifecycleCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize TaskLifecycleCoordinator: {e}")

    return status


def _init_sync_coordinator() -> BootstrapCoordinatorStatus:
    """Initialize SyncCoordinator (SyncScheduler)."""
    status = BootstrapCoordinatorStatus(name="sync_coordinator")
    try:
        from app.coordination.sync_coordinator import wire_sync_events

        coordinator = wire_sync_events()
        status.initialized = True
        status.subscribed = getattr(coordinator, "_subscribed", True)
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] SyncCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] SyncCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize SyncCoordinator: {e}")

    return status


def _init_training_coordinator() -> BootstrapCoordinatorStatus:
    """Initialize TrainingCoordinator."""
    status = BootstrapCoordinatorStatus(name="training_coordinator")
    try:
        from app.coordination.training_coordinator import wire_training_events

        coordinator = wire_training_events()
        status.initialized = True
        status.subscribed = getattr(coordinator, "_subscribed", True)
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] TrainingCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] TrainingCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize TrainingCoordinator: {e}")

    return status


def _init_recovery_manager() -> BootstrapCoordinatorStatus:
    """Initialize RecoveryManager (DEPRECATED - use _init_health_manager).

    RecoveryManager functionality is now consolidated into UnifiedHealthManager.
    This function exists for backward compatibility and returns a skip status.
    """
    status = BootstrapCoordinatorStatus(name="recovery_manager")
    status.initialized = True
    status.subscribed = True
    status.initialized_at = datetime.now()
    # Recovery functionality is handled by UnifiedHealthManager
    logger.debug("[Bootstrap] RecoveryManager skipped - using UnifiedHealthManager")
    return status


def _init_transfer_verifier() -> BootstrapCoordinatorStatus:
    """Initialize TransferVerifier."""
    status = BootstrapCoordinatorStatus(name="transfer_verifier")
    try:
        from app.coordination.transfer_verification import wire_transfer_verifier_events

        coordinator = wire_transfer_verifier_events()
        status.initialized = True
        status.subscribed = getattr(coordinator, "_subscribed", True)
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] TransferVerifier initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] TransferVerifier not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize TransferVerifier: {e}")

    return status


def _init_ephemeral_guard() -> BootstrapCoordinatorStatus:
    """Initialize EphemeralDataGuard."""
    status = BootstrapCoordinatorStatus(name="ephemeral_guard")
    try:
        from app.coordination.ephemeral_data_guard import wire_ephemeral_guard_events

        coordinator = wire_ephemeral_guard_events()
        status.initialized = True
        status.subscribed = getattr(coordinator, "_subscribed", True)
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] EphemeralDataGuard initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] EphemeralDataGuard not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize EphemeralDataGuard: {e}")

    return status


def _init_queue_populator() -> BootstrapCoordinatorStatus:
    """Initialize QueuePopulator."""
    status = BootstrapCoordinatorStatus(name="queue_populator")
    try:
        from app.coordination.queue_populator import wire_queue_populator_events

        coordinator = wire_queue_populator_events()
        status.initialized = True
        status.subscribed = getattr(coordinator, "_subscribed", True)
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] QueuePopulator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] QueuePopulator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize QueuePopulator: {e}")

    return status


def _init_multi_provider() -> BootstrapCoordinatorStatus:
    """Initialize MultiProviderOrchestrator."""
    status = BootstrapCoordinatorStatus(name="multi_provider")
    try:
        from app.coordination.multi_provider_orchestrator import wire_orchestrator_events

        coordinator = wire_orchestrator_events()
        status.initialized = True
        status.subscribed = getattr(coordinator, "_subscribed", True)
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] MultiProviderOrchestrator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] MultiProviderOrchestrator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize MultiProviderOrchestrator: {e}")

    return status


def _init_job_scheduler() -> BootstrapCoordinatorStatus:
    """Initialize JobScheduler host-dead migration wiring."""
    status = BootstrapCoordinatorStatus(name="job_scheduler")
    try:
        from app.coordination.job_scheduler import wire_host_dead_to_job_migration

        wire_host_dead_to_job_migration()
        status.initialized = True
        status.subscribed = True
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] JobScheduler host-dead migration wired")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] JobScheduler not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize JobScheduler: {e}")

    return status


def _init_global_task_coordinator() -> BootstrapCoordinatorStatus:
    """Initialize global TaskCoordinator (separate from TaskLifecycleCoordinator)."""
    status = BootstrapCoordinatorStatus(name="global_task_coordinator")
    try:
        from app.coordination.task_coordinator import wire_task_coordinator_events

        _ = wire_task_coordinator_events()
        status.initialized = True
        status.subscribed = True  # wire function always subscribes
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] Global TaskCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] Global TaskCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize global TaskCoordinator: {e}")

    return status


def _register_coordinators() -> bool:
    """Register all coordinators with OrchestratorRegistry."""
    try:
        from app.coordination.orchestrator_registry import auto_register_known_coordinators

        count = auto_register_known_coordinators()
        logger.info(f"[Bootstrap] Registered {count} coordinators with registry")
        return True

    except ImportError:
        logger.warning("[Bootstrap] OrchestratorRegistry not available")
        return False
    except Exception as e:
        logger.error(f"[Bootstrap] Failed to register coordinators: {e}")
        return False


def _wire_integrations() -> bool:
    """Wire integration modules to event router (C2 consolidation).

    Connects standalone integration modules to the unified event system:
    - ModelLifecycleManager (model_lifecycle.py)
    - P2PIntegrationManager (p2p_integration.py)
    - PipelineFeedbackController (pipeline_feedback.py)

    Returns:
        True if wiring succeeded, False otherwise
    """
    try:
        from app.coordination.integration_bridge import wire_all_integrations_sync

        results = wire_all_integrations_sync()
        wired_count = sum(1 for v in results.values() if v is True)
        total_count = len(results)
        logger.info(f"[Bootstrap] Wired {wired_count}/{total_count} integration modules")
        return wired_count > 0

    except ImportError as e:
        logger.debug(f"[Bootstrap] Integration bridge not available: {e}")
        return False
    except Exception as e:
        logger.error(f"[Bootstrap] Failed to wire integrations: {e}")
        return False


# =============================================================================
# Main Bootstrap Function
# =============================================================================

def bootstrap_coordination(
    enable_resources: bool = True,
    enable_metrics: bool = True,
    enable_optimization: bool = True,
    enable_cache: bool = True,
    enable_model: bool = True,
    enable_error: bool = True,  # Deprecated, use enable_health
    enable_health: bool = True,  # New: UnifiedHealthManager (replaces error + recovery)
    enable_leadership: bool = True,
    enable_selfplay: bool = True,
    enable_pipeline: bool = True,
    enable_task: bool = True,
    enable_sync: bool = True,
    enable_training: bool = True,
    enable_recovery: bool = True,
    enable_transfer: bool = True,
    enable_ephemeral: bool = True,
    enable_queue: bool = True,
    enable_multi_provider: bool = True,
    enable_job_scheduler: bool = True,
    enable_global_task: bool = True,
    enable_integrations: bool = True,  # New: Wire integration modules (C2)
    pipeline_auto_trigger: bool = False,
    register_with_registry: bool = True,
) -> dict[str, Any]:
    """Initialize all coordination components.

    Initializes coordinators in the correct dependency order:
    1. Task lifecycle (foundational - no dependencies)
    2. Global task coordinator (foundational - task spawning limits)
    3. Resource monitoring (foundational - no dependencies)
    4. Cache coordination (foundational - no dependencies)
    5. Error recovery (infrastructure support)
    6. Recovery manager (infrastructure support)
    7. Model lifecycle (depends on cache)
    8. Sync coordinator (depends on resources)
    9. Training coordinator (depends on cache, model)
    10. Transfer verifier (depends on sync)
    11. Ephemeral data guard (depends on sync, cache)
    12. Queue populator (depends on cache)
    13. Selfplay orchestrator (depends on task_lifecycle, resources)
    14. Pipeline orchestrator (depends on selfplay, cache)
    15. Multi-provider orchestrator (depends on pipeline)
    16. Job scheduler (depends on selfplay, training)
    17. Metrics analysis (depends on pipeline)
    18. Optimization (depends on metrics)
    19. Leadership (coordinates all others)

    Args:
        enable_resources: Initialize ResourceMonitoringCoordinator
        enable_metrics: Initialize MetricsAnalysisOrchestrator
        enable_optimization: Initialize OptimizationCoordinator
        enable_cache: Initialize CacheCoordinationOrchestrator
        enable_model: Initialize ModelLifecycleCoordinator
        enable_error: DEPRECATED - use enable_health instead
        enable_health: Initialize UnifiedHealthManager (consolidated error + recovery)
        enable_leadership: Initialize LeadershipCoordinator
        enable_selfplay: Initialize SelfplayOrchestrator
        enable_pipeline: Initialize DataPipelineOrchestrator
        enable_task: Initialize TaskLifecycleCoordinator
        enable_sync: Initialize SyncCoordinator
        enable_training: Initialize TrainingCoordinator
        enable_recovery: Initialize RecoveryManager
        enable_transfer: Initialize TransferVerifier
        enable_ephemeral: Initialize EphemeralDataGuard
        enable_queue: Initialize QueuePopulator
        enable_multi_provider: Initialize MultiProviderOrchestrator
        enable_job_scheduler: Initialize JobScheduler host-dead migration
        enable_global_task: Initialize global TaskCoordinator
        enable_integrations: Wire integration modules to event router (C2)
        pipeline_auto_trigger: Auto-trigger pipeline on events
        register_with_registry: Register coordinators with OrchestratorRegistry

    Returns:
        Status dict with initialization results
    """

    if _state.initialized:
        logger.warning("[Bootstrap] Coordination already initialized, skipping")
        return get_bootstrap_status()

    _state.started_at = datetime.now()
    _state.errors = []

    logger.info("[Bootstrap] Starting coordination bootstrap...")

    # Initialize in dependency order
    # Foundational coordinators first (no dependencies), then dependents
    init_order = [
        # Foundational layer (no dependencies)
        ("task_coordinator", enable_task, _init_task_coordinator),
        ("global_task_coordinator", enable_global_task, _init_global_task_coordinator),
        ("resource_coordinator", enable_resources, _init_resource_coordinator),
        ("cache_orchestrator", enable_cache, _init_cache_orchestrator),
        # Infrastructure support layer - UnifiedHealthManager replaces error + recovery
        ("health_manager", enable_health or enable_error, _init_health_manager),
        ("model_coordinator", enable_model, _init_model_coordinator),
        # Sync and training layer
        ("sync_coordinator", enable_sync, _init_sync_coordinator),
        ("training_coordinator", enable_training, _init_training_coordinator),
        # Data integrity layer
        ("transfer_verifier", enable_transfer, _init_transfer_verifier),
        ("ephemeral_guard", enable_ephemeral, _init_ephemeral_guard),
        ("queue_populator", enable_queue, _init_queue_populator),
        # Selfplay layer (depends on task_lifecycle, resources)
        ("selfplay_orchestrator", enable_selfplay, _init_selfplay_orchestrator),
        # Pipeline layer (depends on selfplay, cache)
        ("pipeline_orchestrator", enable_pipeline, lambda: _init_pipeline_orchestrator(pipeline_auto_trigger)),
        # Multi-provider layer
        ("multi_provider", enable_multi_provider, _init_multi_provider),
        # Job scheduler layer
        ("job_scheduler", enable_job_scheduler, _init_job_scheduler),
        # Metrics layer (depends on pipeline)
        ("metrics_orchestrator", enable_metrics, _init_metrics_orchestrator),
        # Optimization layer (depends on metrics)
        ("optimization_coordinator", enable_optimization, _init_optimization_coordinator),
        # Leadership layer (coordinates all others)
        ("leadership_coordinator", enable_leadership, _init_leadership_coordinator),
    ]

    for name, enabled, init_func in init_order:
        if not enabled:
            logger.debug(f"[Bootstrap] Skipping {name} (disabled)")
            continue

        status = init_func()
        _state.coordinators[name] = status

        if status.error:
            _state.errors.append(f"{name}: {status.error}")

    # Register with OrchestratorRegistry
    if register_with_registry:
        _register_coordinators()

    # Wire integration modules to event router (C2 consolidation)
    if enable_integrations:
        _wire_integrations()

    _state.initialized = True
    _state.completed_at = datetime.now()

    # Log summary
    initialized_count = sum(1 for s in _state.coordinators.values() if s.initialized)
    total_count = len(_state.coordinators)
    error_count = len(_state.errors)

    if error_count > 0:
        logger.warning(
            f"[Bootstrap] Coordination bootstrap completed with errors: "
            f"{initialized_count}/{total_count} coordinators, {error_count} errors"
        )
    else:
        logger.info(
            f"[Bootstrap] Coordination bootstrap completed: "
            f"{initialized_count}/{total_count} coordinators initialized"
        )

    return get_bootstrap_status()


def shutdown_coordination() -> dict[str, Any]:
    """Gracefully shutdown all coordination components.

    Returns:
        Status dict with shutdown results
    """

    if not _state.initialized:
        logger.warning("[Bootstrap] Coordination not initialized, nothing to shutdown")
        return {"shutdown": False, "reason": "not initialized"}

    _state.shutdown_requested = True
    logger.info("[Bootstrap] Starting coordination shutdown...")

    # Shutdown coordinators in reverse of initialization order
    shutdown_order = [
        "leadership_coordinator",
        "optimization_coordinator",
        "metrics_orchestrator",
        "job_scheduler",
        "multi_provider",
        "pipeline_orchestrator",
        "selfplay_orchestrator",
        "queue_populator",
        "ephemeral_guard",
        "transfer_verifier",
        "training_coordinator",
        "sync_coordinator",
        "model_coordinator",
        "health_manager",  # Unified: replaces error_coordinator + recovery_manager
        "cache_orchestrator",
        "resource_coordinator",
        "global_task_coordinator",
        "task_coordinator",
    ]

    shutdown_results: dict[str, bool] = {}

    for name in shutdown_order:
        if name not in _state.coordinators:
            continue

        status = _state.coordinators[name]
        if not status.initialized:
            continue

        try:
            # Try to get coordinator and call shutdown if available
            coordinator = None
            if name == "resource_coordinator":
                from app.coordination.resource_monitoring_coordinator import get_resource_coordinator
                coordinator = get_resource_coordinator()
            elif name == "metrics_orchestrator":
                from app.coordination.metrics_analysis_orchestrator import get_metrics_orchestrator
                coordinator = get_metrics_orchestrator()
            elif name == "optimization_coordinator":
                from app.coordination.optimization_coordinator import get_optimization_coordinator
                coordinator = get_optimization_coordinator()
            elif name == "cache_orchestrator":
                from app.coordination.cache_coordination_orchestrator import get_cache_orchestrator
                coordinator = get_cache_orchestrator()
            elif name == "model_coordinator":
                from app.coordination.model_lifecycle_coordinator import get_model_coordinator
                coordinator = get_model_coordinator()
            elif name == "health_manager":
                from app.coordination.unified_health_manager import get_health_manager
                coordinator = get_health_manager()
            elif name == "leadership_coordinator":
                from app.coordination.leadership_coordinator import get_leadership_coordinator
                coordinator = get_leadership_coordinator()
            elif name == "selfplay_orchestrator":
                from app.coordination.selfplay_orchestrator import get_selfplay_orchestrator
                coordinator = get_selfplay_orchestrator()
            elif name == "pipeline_orchestrator":
                from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator
                coordinator = get_pipeline_orchestrator()
            elif name == "task_coordinator":
                from app.coordination.task_lifecycle_coordinator import get_task_lifecycle_coordinator
                coordinator = get_task_lifecycle_coordinator()
            elif name == "sync_coordinator":
                from app.coordination.sync_coordinator import get_sync_scheduler
                coordinator = get_sync_scheduler()
            elif name == "training_coordinator":
                from app.coordination.training_coordinator import get_training_coordinator
                coordinator = get_training_coordinator()
            elif name == "transfer_verifier":
                from app.coordination.transfer_verification import get_transfer_verifier
                coordinator = get_transfer_verifier()
            elif name == "ephemeral_guard":
                from app.coordination.ephemeral_data_guard import get_ephemeral_guard
                coordinator = get_ephemeral_guard()
            elif name == "queue_populator":
                from app.coordination.queue_populator import get_queue_populator
                coordinator = get_queue_populator()
            elif name == "multi_provider":
                from app.coordination.multi_provider_orchestrator import get_multi_provider_orchestrator
                coordinator = get_multi_provider_orchestrator()
            elif name == "global_task_coordinator":
                from app.coordination.task_coordinator import get_coordinator
                coordinator = get_coordinator()
            # job_scheduler doesn't have a coordinator instance to shutdown

            # Call shutdown method if available
            if coordinator and hasattr(coordinator, "shutdown"):
                coordinator.shutdown()
                logger.debug(f"[Bootstrap] Shutdown {name}")

            shutdown_results[name] = True

        except Exception as e:
            logger.error(f"[Bootstrap] Error shutting down {name}: {e}")
            shutdown_results[name] = False

    logger.info("[Bootstrap] Coordination shutdown completed")

    return {
        "shutdown": True,
        "results": shutdown_results,
        "successful": sum(1 for v in shutdown_results.values() if v),
        "failed": sum(1 for v in shutdown_results.values() if not v),
    }


def get_bootstrap_status() -> dict[str, Any]:
    """Get current bootstrap status.

    Returns:
        Status dict with initialization details
    """

    coordinators_summary = {
        name: {
            "initialized": status.initialized,
            "subscribed": status.subscribed,
            "error": status.error,
        }
        for name, status in _state.coordinators.items()
    }

    initialized_count = sum(1 for s in _state.coordinators.values() if s.initialized)
    subscribed_count = sum(1 for s in _state.coordinators.values() if s.subscribed)

    return {
        "initialized": _state.initialized,
        "started_at": _state.started_at.isoformat() if _state.started_at else None,
        "completed_at": _state.completed_at.isoformat() if _state.completed_at else None,
        "initialized_count": initialized_count,
        "subscribed_count": subscribed_count,
        "total_count": len(_state.coordinators),
        "coordinators": coordinators_summary,
        "errors": _state.errors,
        "shutdown_requested": _state.shutdown_requested,
    }


def is_coordination_ready() -> bool:
    """Check if coordination layer is ready for use.

    Returns:
        True if at least the core coordinators are initialized
    """

    if not _state.initialized:
        return False

    # Check that core coordinators are ready
    core_coordinators = [
        "resource_coordinator",
        "metrics_orchestrator",
        "cache_orchestrator",
    ]

    for name in core_coordinators:
        if name not in _state.coordinators:
            return False
        if not _state.coordinators[name].initialized:
            return False

    return True


def reset_bootstrap_state() -> None:
    """Reset bootstrap state for testing purposes.

    WARNING: Only use in tests or development.
    """
    global _state
    _state = BootstrapState()
    logger.warning("[Bootstrap] Bootstrap state reset")


__all__ = [
    "BootstrapCoordinatorStatus",
    "BootstrapState",
    "bootstrap_coordination",
    "get_bootstrap_status",
    "is_coordination_ready",
    "reset_bootstrap_state",
    "shutdown_coordination",
]
