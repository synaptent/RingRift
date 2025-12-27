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
# Critical Import Validation (Phase 8 - December 2025)
# =============================================================================

# Critical modules that must be importable for coordination to work
_CRITICAL_MODULES = [
    # Event system (core to all coordination)
    ("app.coordination.event_router", "Event routing"),
    ("app.distributed.data_events", "Data events"),
    # Pipeline actions (required for automation)
    ("app.coordination.pipeline_actions", "Pipeline triggers"),
    # Core coordinators
    ("app.coordination.sync_coordinator", "Sync scheduling"),
    ("app.coordination.training_coordinator", "Training management"),
]

# Optional modules that enhance functionality but aren't required
_OPTIONAL_MODULES = [
    ("app.coordination.sync_router", "Intelligent sync routing"),
    ("app.coordination.sync_bandwidth", "Bandwidth management"),
    ("app.coordination.ephemeral_sync", "Ephemeral host sync"),
    ("app.coordination.training_freshness", "Data freshness checks"),
]


def _validate_critical_imports() -> dict[str, Any]:
    """Validate that critical modules can be imported.

    Phase 8 (December 2025): Startup validation catches missing modules early,
    preventing silent failures where features are disabled due to ImportError.

    Returns:
        Dict with 'critical_failures', 'optional_failures', and 'validated' lists
    """
    import os

    result = {
        "critical_failures": [],
        "optional_failures": [],
        "validated": [],
    }

    # Check if strict mode is enabled
    strict_mode = os.environ.get("RINGRIFT_REQUIRE_CRITICAL_IMPORTS", "0") == "1"

    # Validate critical modules
    for module_path, description in _CRITICAL_MODULES:
        try:
            __import__(module_path)
            result["validated"].append(f"{description} ({module_path})")
        except ImportError as e:
            failure_msg = f"{description} ({module_path}): {e}"
            result["critical_failures"].append(failure_msg)

    # Validate optional modules (just log warnings)
    for module_path, description in _OPTIONAL_MODULES:
        try:
            __import__(module_path)
            result["validated"].append(f"{description} ({module_path})")
        except ImportError as e:
            failure_msg = f"{description} ({module_path}): {e}"
            result["optional_failures"].append(failure_msg)
            logger.debug(f"[Bootstrap] Optional module not available: {failure_msg}")

    # Log summary
    total_validated = len(result["validated"])
    critical_failed = len(result["critical_failures"])
    optional_failed = len(result["optional_failures"])

    if critical_failed == 0:
        logger.info(
            f"[Bootstrap] Import validation passed: {total_validated} modules validated"
        )
    else:
        logger.warning(
            f"[Bootstrap] Import validation: {total_validated} passed, "
            f"{critical_failed} critical failed, {optional_failed} optional failed"
        )

    # Raise if strict mode and critical failures
    if strict_mode and result["critical_failures"]:
        raise RuntimeError(
            f"Critical imports failed in strict mode: {result['critical_failures']}"
        )

    return result


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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize SelfplayOrchestrator: {e}")

    return status


def _init_selfplay_scheduler() -> BootstrapCoordinatorStatus:
    """Initialize SelfplayScheduler and wire its event subscriptions.

    December 2025: Ensures SelfplayScheduler receives feedback events:
    - SELFPLAY_COMPLETE: Updates config completion counts
    - TRAINING_COMPLETE: Adjusts priority based on training outcomes
    - ELO_VELOCITY_CHANGED: Modifies allocation based on momentum
    - QUALITY_DEGRADED: Triggers exploration boost
    - NODE_UNHEALTHY/NODE_RECOVERED: Adjusts cluster capacity
    """
    status = BootstrapCoordinatorStatus(name="selfplay_scheduler")
    try:
        from app.coordination.selfplay_scheduler import get_selfplay_scheduler

        scheduler = get_selfplay_scheduler()
        # subscribe_to_events() is called in get_selfplay_scheduler()
        # but we verify the subscription state here
        status.initialized = scheduler is not None
        status.subscribed = getattr(scheduler, '_subscribed', False)
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] SelfplayScheduler initialized and subscribed to events")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] SelfplayScheduler not available: {e}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize SelfplayScheduler: {e}")

    return status


def _init_pipeline_orchestrator(
    auto_trigger: bool = False,
    training_epochs: int | None = None,
    training_batch_size: int | None = None,
    training_model_version: str | None = None,
) -> BootstrapCoordinatorStatus:
    """Initialize DataPipelineOrchestrator."""
    status = BootstrapCoordinatorStatus(name="pipeline_orchestrator")
    try:
        from app.coordination.data_pipeline_orchestrator import wire_pipeline_events

        orchestrator = wire_pipeline_events(
            auto_trigger=auto_trigger,
            training_epochs=training_epochs,
            training_batch_size=training_batch_size,
            training_model_version=training_model_version,
        )
        status.initialized = True
        status.subscribed = orchestrator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] DataPipelineOrchestrator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] DataPipelineOrchestrator not available: {e}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize TaskLifecycleCoordinator: {e}")

    return status


def _init_sync_coordinator() -> BootstrapCoordinatorStatus:
    """Initialize SyncCoordinator (SyncScheduler) and SyncRouter."""
    status = BootstrapCoordinatorStatus(name="sync_coordinator")
    try:
        from app.coordination.cluster.sync import wire_sync_events

        coordinator = wire_sync_events()
        status.initialized = True
        status.subscribed = getattr(coordinator, "_subscribed", True)
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] SyncCoordinator initialized")

        # Dec 2025: Also wire SyncRouter to event system (P0 integration gap fix)
        try:
            from app.coordination.sync_router import get_sync_router

            sync_router = get_sync_router()
            sync_router.wire_to_event_router()
            logger.info("[Bootstrap] SyncRouter wired to event router")
        except ImportError:
            logger.debug("[Bootstrap] SyncRouter not available")
        except Exception as e:
            logger.warning(f"[Bootstrap] Failed to wire SyncRouter: {e}")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] SyncCoordinator not available: {e}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize EphemeralDataGuard: {e}")

    return status


def _init_queue_populator() -> BootstrapCoordinatorStatus:
    """Initialize QueuePopulator."""
    status = BootstrapCoordinatorStatus(name="queue_populator")
    try:
        from app.coordination.unified_queue_populator import wire_queue_populator_events

        coordinator = wire_queue_populator_events()
        status.initialized = True
        status.subscribed = getattr(coordinator, "_subscribed", True)
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] QueuePopulator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] QueuePopulator not available: {e}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize global TaskCoordinator: {e}")

    return status


def _init_auto_export_daemon() -> BootstrapCoordinatorStatus:
    """Initialize AutoExportDaemon (December 2025).

    Automatically exports NPZ training data when game thresholds are met.
    Subscribes to SELFPLAY_COMPLETE events.
    """
    status = BootstrapCoordinatorStatus(name="auto_export_daemon")
    try:
        from app.coordination.auto_export_daemon import get_auto_export_daemon

        daemon = get_auto_export_daemon()
        # Note: Daemon start is async, so we just get the singleton here
        # The daemon will be started by DaemonManager
        status.initialized = True
        status.subscribed = True
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] AutoExportDaemon initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] AutoExportDaemon not available: {e}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize AutoExportDaemon: {e}")

    return status


def _init_auto_evaluation_daemon() -> BootstrapCoordinatorStatus:
    """Initialize AutoEvaluationDaemon (December 2025).

    Automatically triggers model evaluation after training completes.
    Subscribes to TRAINING_COMPLETE events.

    December 2025: Uses canonical EvaluationDaemon (via daemon_manager.py).
    AutoEvaluationDaemon is deprecated - use EvaluationDaemon + AutoPromotionDaemon.
    """
    status = BootstrapCoordinatorStatus(name="evaluation_daemon")
    try:
        from app.coordination.evaluation_daemon import get_evaluation_daemon

        daemon = get_evaluation_daemon()
        status.initialized = True
        status.subscribed = True
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] EvaluationDaemon initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] EvaluationDaemon not available: {e}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize EvaluationDaemon: {e}")

    return status


def _init_model_distribution_daemon() -> BootstrapCoordinatorStatus:
    """Initialize UnifiedDistributionDaemon (consolidated Dec 26, 2025).

    Automatically distributes models AND NPZ files to cluster nodes.
    Subscribes to MODEL_PROMOTED and DATA_SYNCED events.
    Note: Just registers the daemon class; actual start happens via DaemonManager.
    """
    status = BootstrapCoordinatorStatus(name="model_distribution_daemon")
    try:
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
        )

        # Just import to verify availability - DaemonManager will start it
        _ = UnifiedDistributionDaemon
        status.initialized = True
        status.subscribed = True  # Will subscribe when started by DaemonManager
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] UnifiedDistributionDaemon registered")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] UnifiedDistributionDaemon not available: {e}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize UnifiedDistributionDaemon: {e}")

    return status


def _init_idle_resource_daemon() -> BootstrapCoordinatorStatus:
    """Initialize IdleResourceDaemon (December 2025).

    Monitors idle GPUs and spawns selfplay jobs to maximize utilization.
    Critical for cluster efficiency.
    Note: Just registers the daemon class; actual start happens via DaemonManager.
    """
    status = BootstrapCoordinatorStatus(name="idle_resource_daemon")
    try:
        from app.coordination.idle_resource_daemon import IdleResourceDaemon

        # Just import to verify availability - DaemonManager will start it
        _ = IdleResourceDaemon
        status.initialized = True
        status.subscribed = True  # Will subscribe when started by DaemonManager
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] IdleResourceDaemon registered")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] IdleResourceDaemon not available: {e}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize IdleResourceDaemon: {e}")

    return status


def _init_quality_monitor_daemon() -> BootstrapCoordinatorStatus:
    """Initialize QualityMonitorDaemon (December 2025).

    Continuously monitors selfplay quality and emits warnings for low-quality data.
    Critical for training data quality.
    Note: Just registers the daemon class; actual start happens via DaemonManager.
    """
    status = BootstrapCoordinatorStatus(name="quality_monitor_daemon")
    try:
        from app.coordination.quality_monitor_daemon import QualityMonitorDaemon

        # Just import to verify availability - DaemonManager will start it
        _ = QualityMonitorDaemon
        status.initialized = True
        status.subscribed = True  # Will subscribe when started by DaemonManager
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] QualityMonitorDaemon registered")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] QualityMonitorDaemon not available: {e}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize QualityMonitorDaemon: {e}")

    return status


def _init_orphan_detection_daemon() -> BootstrapCoordinatorStatus:
    """Initialize OrphanDetectionDaemon (December 2025).

    Detects orphaned game databases not tracked in cluster manifest.
    Ensures all training data is discoverable.
    Note: Just registers the daemon class; actual start happens via DaemonManager.
    """
    status = BootstrapCoordinatorStatus(name="orphan_detection_daemon")
    try:
        from app.coordination.orphan_detection_daemon import OrphanDetectionDaemon

        # Just import to verify availability - DaemonManager will start it
        _ = OrphanDetectionDaemon
        status.initialized = True
        status.subscribed = True  # Will subscribe when started by DaemonManager
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] OrphanDetectionDaemon registered")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] OrphanDetectionDaemon not available: {e}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize OrphanDetectionDaemon: {e}")

    return status


def _init_curriculum_integration() -> BootstrapCoordinatorStatus:
    """Initialize CurriculumIntegration (December 2025).

    Bridges all feedback loops (quality, evaluation, plateau detection) to curriculum.
    Critical for self-improving training loop.
    """
    status = BootstrapCoordinatorStatus(name="curriculum_integration")
    try:
        from app.coordination.curriculum_integration import wire_all_feedback_loops

        wire_all_feedback_loops()
        status.initialized = True
        status.subscribed = True
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] CurriculumIntegration initialized")

        # Dec 2025: Verify OpponentTrackerIntegration was wired
        # This closes a critical feedback loop for weak opponent targeting
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback
            feedback = get_curriculum_feedback()
            if hasattr(feedback, '_opponent_tracker') and feedback._opponent_tracker is not None:
                logger.info("[Bootstrap] OpponentTrackerIntegration verified - wired to curriculum")
            else:
                # Attempt direct wiring as fallback
                from app.training.curriculum_feedback import wire_opponent_tracker_to_curriculum
                wire_opponent_tracker_to_curriculum()
                if hasattr(feedback, '_opponent_tracker') and feedback._opponent_tracker is not None:
                    logger.info("[Bootstrap] OpponentTrackerIntegration wired via fallback")
                else:
                    logger.warning("[Bootstrap] OpponentTrackerIntegration not available")
        except (ImportError, AttributeError) as e:
            logger.debug(f"[Bootstrap] OpponentTrackerIntegration verification skipped: {e}")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] CurriculumIntegration not available: {e}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize CurriculumIntegration: {e}")

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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
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
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        logger.error(f"[Bootstrap] Failed to wire integrations: {e}")
        return False


def _wire_missing_event_subscriptions() -> dict[str, bool]:
    """Wire missing event subscriptions identified in coordination audit (December 2025).

    These are events that are emitted but weren't being handled by appropriate
    orchestrators:
    1. CLUSTER_SYNC_COMPLETE → DataPipelineOrchestrator (triggers NPZ export)
    2. MODEL_SYNC_COMPLETE → ModelLifecycleCoordinator (triggers cache update)
    3. SELFPLAY_COMPLETE → SyncCoordinator (triggers data sync)
    4. MODEL_PROMOTED → SelfplayModelSelector (hot-reload model cache)

    Returns:
        Dict mapping subscription name to success status
    """
    results: dict[str, bool] = {}

    # 1. Wire CLUSTER_SYNC_COMPLETE to DataPipelineOrchestrator
    try:
        from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator
        from app.coordination.event_router import DataEventType, get_event_bus

        pipeline = get_pipeline_orchestrator()
        bus = get_event_bus()

        async def on_cluster_sync_complete(event):
            """Handle CLUSTER_SYNC_COMPLETE - trigger NPZ export."""
            if not pipeline.auto_trigger:
                return
            iteration = event.payload.get("iteration", pipeline._current_iteration)
            # Emit sync complete to pipeline (will trigger export if auto_trigger)
            await pipeline._on_sync_complete(event.payload)

        bus.subscribe(DataEventType.DATA_SYNC_COMPLETED, on_cluster_sync_complete)
        results["cluster_sync_to_pipeline"] = True
        logger.debug("[Bootstrap] Wired CLUSTER_SYNC_COMPLETE -> DataPipelineOrchestrator")

    except (AttributeError, TypeError, KeyError, RuntimeError) as e:
        results["cluster_sync_to_pipeline"] = False
        logger.debug(f"[Bootstrap] Failed to wire cluster sync to pipeline: {e}")

    # 2. Wire MODEL_SYNC_COMPLETE to ModelLifecycleCoordinator
    try:
        from app.coordination.model_lifecycle_coordinator import get_model_coordinator
        from app.coordination.event_router import DataEventType, get_event_bus

        model_coord = get_model_coordinator()
        bus = get_event_bus()

        async def on_model_sync_complete(event):
            """Handle MODEL_SYNC_COMPLETE - trigger cache refresh."""
            if hasattr(model_coord, "_on_model_sync_complete"):
                await model_coord._on_model_sync_complete(event)
            elif hasattr(model_coord, "refresh_model_cache"):
                model_coord.refresh_model_cache()

        bus.subscribe(DataEventType.P2P_MODEL_SYNCED, on_model_sync_complete)
        results["model_sync_to_lifecycle"] = True
        logger.debug("[Bootstrap] Wired MODEL_SYNC_COMPLETE -> ModelLifecycleCoordinator")

    except (AttributeError, TypeError, KeyError, RuntimeError) as e:
        results["model_sync_to_lifecycle"] = False
        logger.debug(f"[Bootstrap] Failed to wire model sync to lifecycle: {e}")

    # 3. Wire SELFPLAY_COMPLETE to SyncCoordinator (for auto-trigger sync)
    try:
        from app.coordination.event_router import DataEventType, get_event_bus
        from app.coordination.cluster.sync import get_sync_scheduler

        sync_coord = get_sync_scheduler()
        bus = get_event_bus()

        async def on_selfplay_complete_for_sync(event):
            """Handle SELFPLAY_COMPLETE - trigger data sync if quality passes gate."""
            payload = event.payload if hasattr(event, "payload") else {}

            # December 2025: Quality gate before sync trigger
            # Prevents low-quality selfplay data from polluting training pipeline
            games_count = payload.get("games_count", 0)
            quality_score = payload.get("quality_score", 0.0)

            QUALITY_GATE_THRESHOLD = 0.5  # Minimum quality to trigger sync
            MIN_GAMES_FOR_SYNC = 50  # Minimum games to bother syncing

            if quality_score > 0 and quality_score < QUALITY_GATE_THRESHOLD:
                logger.info(
                    f"[Bootstrap] Skipping sync - quality {quality_score:.2f} below "
                    f"threshold {QUALITY_GATE_THRESHOLD} (need more diverse games)"
                )
                # Emit low quality event for feedback loop
                try:
                    from app.coordination.event_router import RouterEvent, EventSource
                    low_quality_event = RouterEvent(
                        event_type="LOW_QUALITY_DETECTED",
                        payload={
                            "config": payload.get("config", ""),
                            "quality_score": quality_score,
                            "games_count": games_count,
                            "source": "pre_sync_gate",
                        },
                        source="coordination_bootstrap",
                        origin=EventSource.ROUTER,
                    )
                    await bus.publish(low_quality_event)
                except (AttributeError, TypeError, RuntimeError) as emit_err:
                    logger.warning(f"[Bootstrap] Failed to emit low-quality event: {emit_err}")
                return

            if games_count > 0 and games_count < MIN_GAMES_FOR_SYNC:
                logger.debug(f"[Bootstrap] Skipping sync - only {games_count} games")
                return

            # Quality OK - proceed with sync
            if hasattr(sync_coord, "trigger_sync_on_selfplay_complete"):
                await sync_coord.trigger_sync_on_selfplay_complete(event)
            elif hasattr(sync_coord, "schedule_sync"):
                # Schedule a sync for the board type that finished selfplay
                board_type = payload.get("board_type")
                num_players = payload.get("num_players")
                if board_type and num_players:
                    await sync_coord.schedule_sync(
                        category="games",
                        board_type=board_type,
                        num_players=num_players,
                        priority="normal",
                    )

        bus.subscribe(DataEventType.SELFPLAY_COMPLETE, on_selfplay_complete_for_sync)
        results["selfplay_to_sync"] = True
        logger.debug("[Bootstrap] Wired SELFPLAY_COMPLETE -> SyncCoordinator")

    except (AttributeError, TypeError, KeyError, RuntimeError) as e:
        results["selfplay_to_sync"] = False
        logger.debug(f"[Bootstrap] Failed to wire selfplay to sync: {e}")

    # 4. Initialize model selector events for hot-reload on MODEL_PROMOTED
    # December 2025: Critical fix - handler existed but was never initialized!
    try:
        from app.training.selfplay_model_selector import _init_event_subscription as init_model_selector_events

        init_model_selector_events()
        results["model_selector_events"] = True
        logger.debug("[Bootstrap] Initialized SelfplayModelSelector MODEL_PROMOTED subscription")

    except (ImportError, AttributeError, TypeError, RuntimeError) as e:
        results["model_selector_events"] = False
        logger.debug(f"[Bootstrap] Failed to init model selector events: {e}")

    # 5. Wire LOW_QUALITY_DATA_WARNING to automatic model rollback (December 2025)
    try:
        from app.training.model_registry import get_model_registry
        from app.training.rollback_manager import wire_quality_to_rollback

        registry = get_model_registry()
        watcher = wire_quality_to_rollback(registry)
        results["quality_to_rollback"] = watcher is not None
        if watcher:
            logger.debug("[Bootstrap] Wired LOW_QUALITY_DATA_WARNING -> RollbackManager")
        else:
            logger.debug("[Bootstrap] Quality rollback watcher not subscribed")

    except (AttributeError, TypeError, RuntimeError) as e:
        results["quality_to_rollback"] = False
        logger.debug(f"[Bootstrap] Failed to wire quality to rollback: {e}")

    # 5b. Wire REGRESSION_DETECTED to automatic model rollback (December 2025)
    # This completes the feedback loop: evaluation → regression → rollback
    # Without this, regressions are detected but models aren't recovered
    try:
        from app.training.model_registry import get_model_registry
        from app.training.rollback_manager import wire_regression_to_rollback

        registry = get_model_registry()
        # December 2025: Enable full auto-rollback for both CRITICAL and SEVERE regressions
        # Setting require_approval_for_severe=False closes the feedback loop completely
        handler = wire_regression_to_rollback(
            registry,
            auto_rollback_enabled=True,
            require_approval_for_severe=False,  # Auto-rollback SEVERE regressions too
        )
        results["regression_to_rollback"] = handler is not None
        if handler:
            logger.debug("[Bootstrap] Wired REGRESSION_DETECTED -> RollbackManager (full auto-rollback enabled)")

    except (AttributeError, TypeError, RuntimeError) as e:
        results["regression_to_rollback"] = False
        logger.debug(f"[Bootstrap] Failed to wire regression to rollback: {e}")

    # 6. Wire PLATEAU_DETECTED to curriculum rebalancing (December 2025)
    try:
        from app.training.curriculum_feedback import wire_plateau_to_curriculum

        watcher = wire_plateau_to_curriculum()
        results["plateau_to_curriculum"] = watcher is not None
        if watcher:
            logger.debug("[Bootstrap] Wired PLATEAU_DETECTED -> CurriculumFeedback")

    except (AttributeError, TypeError, RuntimeError) as e:
        results["plateau_to_curriculum"] = False
        logger.debug(f"[Bootstrap] Failed to wire plateau to curriculum: {e}")

    # 7. Wire TRAINING_EARLY_STOPPED to curriculum boost (December 2025)
    # When training early-stops due to stagnation, boost that config's curriculum weight
    # so more selfplay data is generated to help it improve
    try:
        from app.training.curriculum_feedback import wire_early_stop_to_curriculum

        watcher = wire_early_stop_to_curriculum()
        results["early_stop_to_curriculum"] = watcher is not None
        if watcher:
            logger.debug("[Bootstrap] Wired TRAINING_EARLY_STOPPED -> CurriculumFeedback")

    except (AttributeError, TypeError, RuntimeError) as e:
        results["early_stop_to_curriculum"] = False
        logger.debug(f"[Bootstrap] Failed to wire early stop to curriculum: {e}")

    wired = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info(f"[Bootstrap] Wired {wired}/{total} missing event subscriptions")

    return results


def _start_unified_feedback_orchestrator() -> bool:
    """Start the unified feedback orchestrator (December 2025 critical integration gap).

    The UnifiedFeedbackOrchestrator is responsible for:
    1. Computing and emitting feedback signals (exploration, curriculum, etc.)
    2. Bridging training outcomes to selfplay parameter adjustments
    3. Coordinating the self-improvement loop across all training configurations

    This was identified as a missing integration during the December 2025 audit -
    the orchestrator was defined but never started during bootstrap.

    Returns:
        True if started successfully, False otherwise
    """
    try:
        from app.coordination.unified_feedback import get_unified_feedback

        orchestrator = get_unified_feedback()

        # Check if already running (idempotent)
        if hasattr(orchestrator, "_running") and orchestrator._running:
            logger.debug("[Bootstrap] Unified feedback orchestrator already running")
            return True

        # Start the orchestrator (this subscribes to events and begins processing)
        if hasattr(orchestrator, "start"):
            import asyncio

            # Handle sync/async context
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, schedule the start
                asyncio.create_task(orchestrator.start())
                logger.info("[Bootstrap] Unified feedback orchestrator start scheduled")
            except RuntimeError:
                # No running loop - run synchronously
                asyncio.run(orchestrator.start())
                logger.info("[Bootstrap] Unified feedback orchestrator started")
        else:
            # No async start method - just ensure it's initialized
            logger.info("[Bootstrap] Unified feedback orchestrator initialized (no async start)")

        return True

    except ImportError as e:
        logger.warning(f"[Bootstrap] Unified feedback module not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"[Bootstrap] Failed to start unified feedback orchestrator: {e}")
        return False


def _validate_event_wiring() -> dict[str, Any]:
    """Validate event flow to detect orphaned or misconfigured events.

    Phase 21.2 (December 2025): Critical startup validation.
    This validates the event routing system is healthy and logs any issues
    detected. Issues could include:
    - Events not being routed through the system
    - Missing event buses (data, stage, cross-process)
    - High duplicate event rate (potential loops)

    Returns:
        Dict with validation results from event_router.validate_event_flow()
    """
    results: dict[str, Any] = {
        "healthy": False,
        "issues": [],
        "recommendations": [],
        "validated": False,
    }

    try:
        from app.coordination.event_router import validate_event_flow

        validation = validate_event_flow()
        results.update(validation)

        # Check health status
        issues = validation.get("issues", [])
        recommendations = validation.get("recommendations", [])

        if issues:
            for issue in issues[:5]:
                logger.warning(f"[Bootstrap] Event flow issue: {issue}")

        if recommendations:
            for rec in recommendations[:3]:
                logger.info(f"[Bootstrap] Event flow recommendation: {rec}")

        results["validated"] = True

        if validation.get("healthy", False):
            logger.info(
                f"[Bootstrap] Event flow validated: healthy, "
                f"{validation.get('total_routed', 0)} events routed"
            )
        else:
            logger.warning(
                f"[Bootstrap] Event flow validation: {len(issues)} issues detected"
            )

    except ImportError as e:
        logger.debug(f"[Bootstrap] validate_event_flow not available: {e}")
    except (AttributeError, TypeError, KeyError, RuntimeError) as e:
        results["issues"].append(f"Validation failed: {e}")
        logger.warning(f"[Bootstrap] Event flow validation failed: {e}")

    return results


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
    # Critical daemons (December 2025)
    enable_auto_export: bool = True,
    enable_auto_evaluation: bool = True,
    enable_model_distribution: bool = True,
    enable_idle_resource: bool = True,
    enable_quality_monitor: bool = True,
    enable_orphan_detection: bool = True,
    enable_curriculum_integration: bool = True,
    pipeline_auto_trigger: bool = False,
    register_with_registry: bool = True,
    # Training config (December 2025 - CLI connection)
    training_epochs: int | None = None,
    training_batch_size: int | None = None,
    training_model_version: str | None = None,
    # Master loop enforcement (December 2025)
    require_master_loop: bool = False,  # Set True to enforce master loop for full automation
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
        enable_auto_export: Initialize AutoExportDaemon (NPZ export automation)
        enable_auto_evaluation: Initialize AutoEvaluationDaemon (model evaluation)
        enable_model_distribution: Initialize ModelDistributionDaemon (model sync)
        enable_idle_resource: Initialize IdleResourceDaemon (GPU utilization)
        enable_quality_monitor: Initialize QualityMonitorDaemon (data quality)
        enable_orphan_detection: Initialize OrphanDetectionDaemon (data discovery)
        enable_curriculum_integration: Initialize CurriculumIntegration (feedback loops)
        pipeline_auto_trigger: Auto-trigger pipeline on events
        register_with_registry: Register coordinators with OrchestratorRegistry
        training_epochs: Override default training epochs for pipeline
        training_batch_size: Override default training batch size for pipeline
        training_model_version: Override default model version for pipeline
        require_master_loop: If True, enforce that master loop is running

    Returns:
        Status dict with initialization results

    Raises:
        RuntimeError: If require_master_loop=True and master loop is not running
    """

    if _state.initialized:
        logger.warning("[Bootstrap] Coordination already initialized, skipping")
        return get_bootstrap_status()

    # Check if master loop is required (December 2025)
    # Can be overridden with RINGRIFT_SKIP_MASTER_LOOP_CHECK=1 environment variable
    import os
    skip_check = os.environ.get("RINGRIFT_SKIP_MASTER_LOOP_CHECK", "0") == "1"

    if require_master_loop and not skip_check:
        try:
            from app.coordination.master_loop_guard import ensure_master_loop_running

            ensure_master_loop_running(
                require_for_automation=True,
                operation_name="coordination bootstrap with full automation",
            )
            logger.info("[Bootstrap] Master loop check passed")
        except ImportError:
            logger.warning("[Bootstrap] master_loop_guard not available, skipping check")
        except RuntimeError as e:
            logger.error(f"[Bootstrap] {e}")
            raise

    _state.started_at = datetime.now()
    _state.errors = []

    logger.info("[Bootstrap] Starting coordination bootstrap...")

    # Phase 8 (December 2025): Validate critical imports at startup
    # This catches missing modules early before they cause silent failures
    import_status = _validate_critical_imports()
    if import_status.get("critical_failures"):
        for failure in import_status["critical_failures"]:
            _state.errors.append(f"Critical import failed: {failure}")
            logger.error(f"[Bootstrap] CRITICAL: {failure}")
        # Log warning but don't fail - allows partial operation
        logger.warning(
            f"[Bootstrap] {len(import_status['critical_failures'])} critical imports failed. "
            f"Some coordination features will be unavailable."
        )

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
        ("selfplay_scheduler", enable_selfplay, _init_selfplay_scheduler),  # Dec 2025: Feedback loop
        # Pipeline layer (depends on selfplay, cache)
        ("pipeline_orchestrator", enable_pipeline, lambda: _init_pipeline_orchestrator(
            pipeline_auto_trigger, training_epochs, training_batch_size, training_model_version
        )),
        # Multi-provider layer
        ("multi_provider", enable_multi_provider, _init_multi_provider),
        # Job scheduler layer
        ("job_scheduler", enable_job_scheduler, _init_job_scheduler),
        # Daemon layer (December 2025 - critical automation daemons)
        ("auto_export_daemon", enable_auto_export, _init_auto_export_daemon),
        ("auto_evaluation_daemon", enable_auto_evaluation, _init_auto_evaluation_daemon),
        ("model_distribution_daemon", enable_model_distribution, _init_model_distribution_daemon),
        ("idle_resource_daemon", enable_idle_resource, _init_idle_resource_daemon),
        ("quality_monitor_daemon", enable_quality_monitor, _init_quality_monitor_daemon),
        ("orphan_detection_daemon", enable_orphan_detection, _init_orphan_detection_daemon),
        ("curriculum_integration", enable_curriculum_integration, _init_curriculum_integration),
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

    # Wire missing event subscriptions (December 2025 audit findings)
    _wire_missing_event_subscriptions()

    # Start unified feedback orchestrator (December 2025 - critical integration gap)
    _start_unified_feedback_orchestrator()

    # Phase 21.2 (December 2025): Validate event flow to detect orphaned events
    _validate_event_wiring()

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
        # Daemon layer (shutdown before coordinators they depend on)
        "curriculum_integration",
        "orphan_detection_daemon",
        "quality_monitor_daemon",
        "idle_resource_daemon",
        "model_distribution_daemon",
        "auto_evaluation_daemon",
        "auto_export_daemon",
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
                from app.coordination.cluster.sync import get_sync_scheduler
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
                from app.coordination.unified_queue_populator import get_queue_populator
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

        except (AttributeError, TypeError, RuntimeError, OSError) as e:
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


# =============================================================================
# Smoke Test (December 2025)
# =============================================================================


@dataclass
class SmokeTestResult:
    """Result of a single smoke test check."""

    name: str
    passed: bool
    error: str | None = None
    details: dict[str, Any] | None = None


def run_bootstrap_smoke_test() -> dict[str, Any]:
    """Run comprehensive smoke test on daemon subscriptions and wiring.

    Verifies that all critical event subscriptions and integrations
    are properly wired after bootstrap. This is designed to catch
    wiring issues before they cause problems in production.

    Returns:
        Dict with test results:
        - passed: bool - Overall pass/fail
        - checks: list[SmokeTestResult] - Individual check results
        - passed_count: int - Number of passed checks
        - failed_count: int - Number of failed checks
        - warnings: list[str] - Non-fatal warnings

    Example:
        >>> result = run_bootstrap_smoke_test()
        >>> if not result['passed']:
        ...     for check in result['checks']:
        ...         if not check['passed']:
        ...             print(f"FAIL: {check['name']}: {check['error']}")
    """
    checks: list[SmokeTestResult] = []
    warnings: list[str] = []

    # 1. Check event bus is initialized
    try:
        from app.coordination.event_router import get_event_bus

        bus = get_event_bus()
        subscriber_count = len(getattr(bus, "_subscribers", {}))
        checks.append(SmokeTestResult(
            name="event_bus_initialized",
            passed=True,
            details={"subscriber_count": subscriber_count},
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="event_bus_initialized",
            passed=False,
            error=str(e),
        ))

    # 2. Check curriculum feedback watcher
    try:
        from app.training.curriculum_feedback import (
            get_plateau_curriculum_watcher,
            get_curriculum_feedback,
        )

        feedback = get_curriculum_feedback()
        plateau_watcher = get_plateau_curriculum_watcher()
        checks.append(SmokeTestResult(
            name="curriculum_feedback_wired",
            passed=feedback is not None,
            details={
                "has_feedback": feedback is not None,
                "has_plateau_watcher": plateau_watcher is not None,
                "plateau_subscribed": getattr(plateau_watcher, "_subscribed", False) if plateau_watcher else False,
            },
        ))
    except ImportError:
        warnings.append("curriculum_feedback module not available")
        checks.append(SmokeTestResult(
            name="curriculum_feedback_wired",
            passed=True,  # Not a failure if module doesn't exist
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="curriculum_feedback_wired",
            passed=False,
            error=str(e),
        ))

    # 3. Check quality rollback watcher
    try:
        from app.training.rollback_manager import get_quality_rollback_watcher

        watcher = get_quality_rollback_watcher()
        checks.append(SmokeTestResult(
            name="quality_rollback_wired",
            passed=watcher is not None,
            details={
                "has_watcher": watcher is not None,
                "subscribed": getattr(watcher, "_subscribed", False) if watcher else False,
            },
        ))
    except ImportError:
        warnings.append("rollback_manager.QualityRollbackWatcher not available")
        checks.append(SmokeTestResult(
            name="quality_rollback_wired",
            passed=True,
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="quality_rollback_wired",
            passed=False,
            error=str(e),
        ))

    # 4. Check regression detector event bus connection
    try:
        from app.training.regression_detector import get_regression_detector

        detector = get_regression_detector(connect_event_bus=False)
        has_bus = getattr(detector, "_event_bus", None) is not None
        checks.append(SmokeTestResult(
            name="regression_detector_available",
            passed=detector is not None,
            details={
                "has_detector": detector is not None,
                "has_event_bus": has_bus,
            },
        ))
    except ImportError:
        warnings.append("regression_detector module not available")
        checks.append(SmokeTestResult(
            name="regression_detector_available",
            passed=True,
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="regression_detector_available",
            passed=False,
            error=str(e),
        ))

    # 5. Check training coordinator is subscribed
    try:
        from app.coordination.training_coordinator import get_training_coordinator

        coordinator = get_training_coordinator()
        subscribed = getattr(coordinator, "_subscribed", False)
        checks.append(SmokeTestResult(
            name="training_coordinator_subscribed",
            passed=subscribed,
            details={
                "subscribed": subscribed,
                "has_coordinator": coordinator is not None,
            },
        ))
    except ImportError:
        warnings.append("training_coordinator not available")
        checks.append(SmokeTestResult(
            name="training_coordinator_subscribed",
            passed=True,
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="training_coordinator_subscribed",
            passed=False,
            error=str(e),
        ))

    # 6. Check selfplay scheduler curriculum integration
    try:
        from app.coordination.selfplay_scheduler import get_selfplay_scheduler

        scheduler = get_selfplay_scheduler()
        checks.append(SmokeTestResult(
            name="selfplay_scheduler_available",
            passed=scheduler is not None,
            details={
                "has_scheduler": scheduler is not None,
            },
        ))
    except ImportError:
        warnings.append("selfplay_scheduler not available")
        checks.append(SmokeTestResult(
            name="selfplay_scheduler_available",
            passed=True,
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="selfplay_scheduler_available",
            passed=False,
            error=str(e),
        ))

    # 7. Check key event types are defined
    try:
        from app.coordination.event_router import DataEventType

        critical_events = [
            "PLATEAU_DETECTED",
            "REGRESSION_DETECTED",
            "LOW_QUALITY_DATA_WARNING",
            "EVALUATION_COMPLETED",
            "MODEL_PROMOTED",
        ]
        missing = [e for e in critical_events if not hasattr(DataEventType, e)]
        checks.append(SmokeTestResult(
            name="critical_event_types_defined",
            passed=len(missing) == 0,
            details={"missing_events": missing} if missing else None,
            error=f"Missing events: {missing}" if missing else None,
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="critical_event_types_defined",
            passed=False,
            error=str(e),
        ))

    # 8. Check daemon manager has known daemons
    try:
        from app.coordination.daemon_manager import get_daemon_manager

        manager = get_daemon_manager()
        known_daemons = getattr(manager, "_daemons", {})
        checks.append(SmokeTestResult(
            name="daemon_manager_available",
            passed=manager is not None,
            details={
                "daemon_count": len(known_daemons),
            },
        ))
    except ImportError:
        warnings.append("daemon_manager not available")
        checks.append(SmokeTestResult(
            name="daemon_manager_available",
            passed=True,
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="daemon_manager_available",
            passed=False,
            error=str(e),
        ))

    # Compile results
    passed_count = sum(1 for c in checks if c.passed)
    failed_count = len(checks) - passed_count

    result = {
        "passed": failed_count == 0,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "error": c.error,
                "details": c.details,
            }
            for c in checks
        ],
        "passed_count": passed_count,
        "failed_count": failed_count,
        "warnings": warnings,
    }

    # Log summary
    if failed_count > 0:
        logger.warning(
            f"[Bootstrap] Smoke test: {passed_count}/{len(checks)} passed, "
            f"{failed_count} failed"
        )
        for c in checks:
            if not c.passed:
                logger.warning(f"[Bootstrap]   FAIL: {c.name}: {c.error}")
    else:
        logger.info(
            f"[Bootstrap] Smoke test: {passed_count}/{len(checks)} passed"
        )

    return result


__all__ = [
    "BootstrapCoordinatorStatus",
    "BootstrapState",
    "SmokeTestResult",
    "bootstrap_coordination",
    "get_bootstrap_status",
    "is_coordination_ready",
    "reset_bootstrap_state",
    "run_bootstrap_smoke_test",
    "shutdown_coordination",
]
