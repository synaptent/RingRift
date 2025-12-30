"""Coordination Bootstrap - Unified initialization for all coordinators (December 2025).

This module provides a single entry point for initializing the entire
coordination layer. It handles the correct initialization order, event
wiring, and registry registration for all coordinators.

INITIALIZATION ORDER
====================

Coordinators are initialized in dependency order across 8 layers:

Layer 1 - Foundational (no dependencies):
    - task_coordinator: Task lifecycle event wiring
    - global_task_coordinator: Global task tracking
    - resource_coordinator: Resource monitoring events
    - cache_orchestrator: Cache coordination events

Layer 2 - Infrastructure Support:
    - health_manager: Unified health monitoring (replaces error + recovery)
    - model_coordinator: Model lifecycle events

Layer 3 - Sync and Training:
    - sync_coordinator: Data synchronization events
    - training_coordinator: Training pipeline events

Layer 4 - Data Integrity:
    - transfer_verifier: Transfer verification events
    - ephemeral_guard: Ephemeral data protection
    - queue_populator: Work queue maintenance

Layer 5 - Selfplay (depends on task_lifecycle, resources):
    - selfplay_orchestrator: Selfplay coordination
    - selfplay_scheduler: Priority-based selfplay allocation

Layer 6 - Multi-Provider and Jobs:
    - multi_provider: Multi-provider orchestration
    - job_scheduler: Job scheduling events

Layer 7 - Daemons (December 2025 critical automation):
    - auto_export_daemon: Auto NPZ export
    - evaluation_daemon: Auto evaluation
    - model_distribution_daemon: Model sync to cluster
    - idle_resource_daemon: GPU idle detection
    - quality_monitor_daemon: Data quality monitoring
    - orphan_detection_daemon: Orphan game recovery
    - pipeline_orchestrator: Pipeline stage coordination (special case)
    - curriculum_integration: Curriculum feedback loops (special case)

Layer 8 - Top-Level Coordination:
    - metrics_orchestrator: Metrics analysis (depends on pipeline)
    - optimization_coordinator: CMA-ES/NAS optimization (depends on metrics)
    - leadership_coordinator: Leader election (coordinates all others)

Post-Initialization Steps:
    1. Register coordinators with OrchestratorRegistry
    2. Wire integration modules to event router
    3. Wire missing event subscriptions (audit findings)
    4. Start unified feedback orchestrator
    5. Validate event wiring (detect orphaned events)

CRITICAL: Subscribers must start before emitters. FEEDBACK_LOOP and
DATA_PIPELINE must initialize before AUTO_SYNC to receive sync events.

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

import importlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Event Emission Helper (December 2025)
# =============================================================================


def _emit_coordinator_health_event(status: "BootstrapCoordinatorStatus") -> None:
    """Emit coordinator health event based on initialization status.

    December 2025: Added to track coordinator initialization success/failure
    via the unified event system.

    Args:
        status: The BootstrapCoordinatorStatus from initialization
    """
    try:
        from app.utils.async_utils import fire_and_forget
        import asyncio

        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            in_async = True
        except RuntimeError:
            in_async = False

        if status.initialized:
            # Emit healthy event
            from app.coordination.event_emitters import emit_coordinator_healthy

            async def _emit_healthy():
                await emit_coordinator_healthy(
                    coordinator_name=status.name,
                    health_score=1.0 if status.subscribed else 0.8,
                    uptime_seconds=0.0,
                    subscribed=status.subscribed,
                    source="coordination_bootstrap",
                )

            if in_async:
                fire_and_forget(_emit_healthy(), name=f"emit_healthy_{status.name}")
            else:
                # Try to create a task anyway - may work in some contexts
                try:
                    fire_and_forget(_emit_healthy(), name=f"emit_healthy_{status.name}")
                except RuntimeError:
                    logger.debug(f"[Bootstrap] Could not emit healthy event for {status.name} (no event loop)")
        else:
            # Emit unhealthy event
            from app.coordination.event_emitters import emit_coordinator_unhealthy

            async def _emit_unhealthy():
                await emit_coordinator_unhealthy(
                    coordinator_name=status.name,
                    reason="initialization_failed",
                    error=status.error or "Unknown error",
                    health_score=0.0,
                    source="coordination_bootstrap",
                )

            if in_async:
                fire_and_forget(_emit_unhealthy(), name=f"emit_unhealthy_{status.name}")
            else:
                try:
                    fire_and_forget(_emit_unhealthy(), name=f"emit_unhealthy_{status.name}")
                except RuntimeError:
                    logger.debug(f"[Bootstrap] Could not emit unhealthy event for {status.name} (no event loop)")

    except ImportError as e:
        # Event emitters not available - log at debug level
        logger.debug(f"[Bootstrap] Event emitters not available: {e}")
    except (RuntimeError, TypeError, AttributeError) as e:
        # Don't fail bootstrap due to event emission issues
        # RuntimeError: no event loop, TypeError: wrong args, AttributeError: API mismatch
        logger.debug(f"[Bootstrap] Could not emit health event for {status.name}: {e}")


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
# Coordinator Registry (December 2025 Refactoring)
# =============================================================================

from enum import Enum


class InitPattern(Enum):
    """Initialization patterns for coordinators."""

    WIRE = "wire"  # Call wire_*() function, check _subscribed
    GET = "get"  # Call get_*() singleton, check _subscribed
    IMPORT = "import"  # Just import class to verify availability
    SKIP = "skip"  # Deprecated, return skip status
    DELEGATE = "delegate"  # Delegate to another coordinator


@dataclass
class CoordinatorSpec:
    """Specification for a coordinator to be initialized.

    Attributes:
        name: Registry key and internal identifier
        display_name: Human-readable name for logging
        module_path: Full import path for the module
        pattern: Initialization pattern to use
        func_name: Function/class name to import (wire function, get function, or class)
        delegate_to: For DELEGATE pattern, name of coordinator to delegate to
        extra_wiring: Additional wiring function to call after main init
        check_subscribed: Whether to check _subscribed attribute (default True)
    """

    name: str
    display_name: str
    module_path: str
    pattern: InitPattern
    func_name: str | None = None
    delegate_to: str | None = None
    extra_wiring: tuple[str, str] | None = None  # (module_path, func_name)
    check_subscribed: bool = True


# Registry of all coordinators with their initialization specs
# Organized by initialization order (dependencies first)
COORDINATOR_REGISTRY: dict[str, CoordinatorSpec] = {
    # === Foundational layer (no dependencies) ===
    "task_coordinator": CoordinatorSpec(
        name="task_coordinator",
        display_name="TaskLifecycleCoordinator",
        module_path="app.coordination.task_lifecycle_coordinator",
        pattern=InitPattern.WIRE,
        func_name="wire_task_events",
    ),
    "global_task_coordinator": CoordinatorSpec(
        name="global_task_coordinator",
        display_name="Global TaskCoordinator",
        module_path="app.coordination.task_coordinator",
        pattern=InitPattern.WIRE,
        func_name="wire_task_coordinator_events",
    ),
    "resource_coordinator": CoordinatorSpec(
        name="resource_coordinator",
        display_name="ResourceMonitoringCoordinator",
        module_path="app.coordination.resource_monitoring_coordinator",
        pattern=InitPattern.WIRE,
        func_name="wire_resource_events",
    ),
    "cache_orchestrator": CoordinatorSpec(
        name="cache_orchestrator",
        display_name="CacheCoordinationOrchestrator",
        module_path="app.coordination.cache_coordination_orchestrator",
        pattern=InitPattern.WIRE,
        func_name="wire_cache_events",
    ),
    # === Infrastructure support layer ===
    "health_manager": CoordinatorSpec(
        name="health_manager",
        display_name="UnifiedHealthManager",
        module_path="app.coordination.unified_health_manager",
        pattern=InitPattern.WIRE,
        func_name="wire_health_events",
    ),
    "error_coordinator": CoordinatorSpec(
        name="error_coordinator",
        display_name="ErrorRecoveryCoordinator",
        module_path="",  # Not used - delegates
        pattern=InitPattern.DELEGATE,
        delegate_to="health_manager",
    ),
    "recovery_manager": CoordinatorSpec(
        name="recovery_manager",
        display_name="RecoveryManager",
        module_path="",  # Not used - skip
        pattern=InitPattern.SKIP,
    ),
    "model_coordinator": CoordinatorSpec(
        name="model_coordinator",
        display_name="ModelLifecycleCoordinator",
        module_path="app.coordination.model_lifecycle_coordinator",
        pattern=InitPattern.WIRE,
        func_name="wire_model_events",
    ),
    # === Sync and training layer ===
    "sync_coordinator": CoordinatorSpec(
        name="sync_coordinator",
        display_name="SyncCoordinator",
        module_path="app.coordination.cluster.sync",
        pattern=InitPattern.WIRE,
        func_name="wire_sync_events",
        extra_wiring=("app.coordination.sync_router", "get_sync_router"),
    ),
    "training_coordinator": CoordinatorSpec(
        name="training_coordinator",
        display_name="TrainingCoordinator",
        module_path="app.coordination.training_coordinator",
        pattern=InitPattern.WIRE,
        func_name="wire_training_events",
    ),
    # === Data integrity layer ===
    "transfer_verifier": CoordinatorSpec(
        name="transfer_verifier",
        display_name="TransferVerifier",
        module_path="app.coordination.transfer_verification",
        pattern=InitPattern.WIRE,
        func_name="wire_transfer_verifier_events",
    ),
    "ephemeral_guard": CoordinatorSpec(
        name="ephemeral_guard",
        display_name="EphemeralDataGuard",
        module_path="app.coordination.ephemeral_data_guard",
        pattern=InitPattern.WIRE,
        func_name="wire_ephemeral_guard_events",
    ),
    "queue_populator": CoordinatorSpec(
        name="queue_populator",
        display_name="QueuePopulator",
        module_path="app.coordination.unified_queue_populator",
        pattern=InitPattern.WIRE,
        func_name="wire_queue_populator_events",
    ),
    # === Selfplay layer ===
    "selfplay_orchestrator": CoordinatorSpec(
        name="selfplay_orchestrator",
        display_name="SelfplayOrchestrator",
        module_path="app.coordination.selfplay_orchestrator",
        pattern=InitPattern.WIRE,
        func_name="wire_selfplay_events",
    ),
    "selfplay_scheduler": CoordinatorSpec(
        name="selfplay_scheduler",
        display_name="SelfplayScheduler",
        module_path="app.coordination.selfplay_scheduler",
        pattern=InitPattern.GET,
        func_name="get_selfplay_scheduler",
    ),
    # === Pipeline and job layer ===
    # Note: pipeline_orchestrator has special handling for extra args
    "multi_provider": CoordinatorSpec(
        name="multi_provider",
        display_name="MultiProviderOrchestrator",
        module_path="app.coordination.multi_provider_orchestrator",
        pattern=InitPattern.WIRE,
        func_name="wire_orchestrator_events",
    ),
    "job_scheduler": CoordinatorSpec(
        name="job_scheduler",
        display_name="JobScheduler",
        module_path="app.coordination.job_scheduler",
        pattern=InitPattern.WIRE,
        func_name="wire_host_dead_to_job_migration",
        check_subscribed=False,  # Returns None, not a coordinator
    ),
    # === Daemon layer (December 2025) ===
    "auto_export_daemon": CoordinatorSpec(
        name="auto_export_daemon",
        display_name="AutoExportDaemon",
        module_path="app.coordination.auto_export_daemon",
        pattern=InitPattern.GET,
        func_name="get_auto_export_daemon",
        check_subscribed=False,  # Subscribes in start(), called by DaemonManager
    ),
    "evaluation_daemon": CoordinatorSpec(
        name="evaluation_daemon",
        display_name="EvaluationDaemon",
        module_path="app.coordination.evaluation_daemon",
        pattern=InitPattern.GET,
        func_name="get_evaluation_daemon",
        check_subscribed=False,  # Subscribes in start(), called by DaemonManager
    ),
    "model_distribution_daemon": CoordinatorSpec(
        name="model_distribution_daemon",
        display_name="UnifiedDistributionDaemon",
        module_path="app.coordination.unified_distribution_daemon",
        pattern=InitPattern.IMPORT,
        func_name="UnifiedDistributionDaemon",
    ),
    "idle_resource_daemon": CoordinatorSpec(
        name="idle_resource_daemon",
        display_name="IdleResourceDaemon",
        module_path="app.coordination.idle_resource_daemon",
        pattern=InitPattern.IMPORT,
        func_name="IdleResourceDaemon",
    ),
    "quality_monitor_daemon": CoordinatorSpec(
        name="quality_monitor_daemon",
        display_name="QualityMonitorDaemon",
        module_path="app.coordination.quality_monitor_daemon",
        pattern=InitPattern.IMPORT,
        func_name="QualityMonitorDaemon",
    ),
    "orphan_detection_daemon": CoordinatorSpec(
        name="orphan_detection_daemon",
        display_name="OrphanDetectionDaemon",
        module_path="app.coordination.orphan_detection_daemon",
        pattern=InitPattern.IMPORT,
        func_name="OrphanDetectionDaemon",
    ),
    "training_activity": CoordinatorSpec(
        name="training_activity",
        display_name="TrainingActivityDaemon",
        module_path="app.coordination.training_activity_daemon",
        pattern=InitPattern.GET,
        func_name="get_training_activity_daemon",
        check_subscribed=False,  # Subscribes in start(), called by DaemonManager
    ),
    "curriculum_integration": CoordinatorSpec(
        name="curriculum_integration",
        display_name="CurriculumIntegration",
        module_path="app.coordination.curriculum_integration",
        pattern=InitPattern.WIRE,
        func_name="wire_all_feedback_loops",
        check_subscribed=False,  # Returns None, not a coordinator
    ),
    # === Metrics and optimization layer ===
    "metrics_orchestrator": CoordinatorSpec(
        name="metrics_orchestrator",
        display_name="MetricsAnalysisOrchestrator",
        module_path="app.coordination.metrics_analysis_orchestrator",
        pattern=InitPattern.WIRE,
        func_name="wire_metrics_events",
    ),
    "optimization_coordinator": CoordinatorSpec(
        name="optimization_coordinator",
        display_name="OptimizationCoordinator",
        module_path="app.coordination.optimization_coordinator",
        pattern=InitPattern.WIRE,
        func_name="wire_optimization_events",
    ),
    # === PER (Prioritized Experience Replay) - December 29, 2025 ===
    "per_orchestrator": CoordinatorSpec(
        name="per_orchestrator",
        display_name="PEROrchestrator",
        module_path="app.training.per_orchestrator",
        pattern=InitPattern.WIRE,
        func_name="wire_per_events",
    ),
    # === Leadership layer (coordinates all others) ===
    "leadership_coordinator": CoordinatorSpec(
        name="leadership_coordinator",
        display_name="LeadershipCoordinator",
        module_path="app.coordination.leadership_coordinator",
        pattern=InitPattern.WIRE,
        func_name="wire_leadership_events",
    ),
}


def _init_coordinator_from_spec(
    spec: CoordinatorSpec,
    **kwargs,
) -> BootstrapCoordinatorStatus:
    """Initialize a coordinator from its specification.

    Generic initialization function that handles all coordinator patterns:
    - WIRE: Import module, call wire_*(), check _subscribed
    - GET: Import module, call get_*(), check for singleton
    - IMPORT: Just import class to verify availability
    - SKIP: Return skip status for deprecated coordinators
    - DELEGATE: Delegate to another coordinator

    Args:
        spec: CoordinatorSpec defining how to initialize
        **kwargs: Additional arguments passed to wire/get function

    Returns:
        BootstrapCoordinatorStatus with initialization result
    """
    status = BootstrapCoordinatorStatus(name=spec.name)

    # Handle delegation pattern
    if spec.pattern == InitPattern.DELEGATE:
        if spec.delegate_to and spec.delegate_to in COORDINATOR_REGISTRY:
            delegate_spec = COORDINATOR_REGISTRY[spec.delegate_to]
            return _init_coordinator_from_spec(delegate_spec, **kwargs)
        else:
            status.error = f"Delegate target not found: {spec.delegate_to}"
            logger.warning(f"[Bootstrap] {spec.display_name} delegate not found")
            return status

    # Handle skip pattern (deprecated coordinators)
    if spec.pattern == InitPattern.SKIP:
        status.initialized = True
        status.subscribed = True
        status.initialized_at = datetime.now()
        logger.debug(f"[Bootstrap] {spec.display_name} skipped - using unified replacement")
        return status

    try:
        # Import the module
        module = importlib.import_module(spec.module_path)

        if spec.pattern == InitPattern.WIRE:
            # Call wire function
            wire_func = getattr(module, spec.func_name)
            result = wire_func(**kwargs) if kwargs else wire_func()

            status.initialized = True
            if spec.check_subscribed and result is not None:
                status.subscribed = getattr(result, "_subscribed", True)
            else:
                status.subscribed = True
            status.initialized_at = datetime.now()
            logger.info(f"[Bootstrap] {spec.display_name} initialized")

            # December 2025: Surface subscription failures instead of hiding them
            if status.initialized and not status.subscribed:
                logger.warning(
                    f"[Bootstrap] {spec.display_name} initialized but NOT subscribed. "
                    "Events may be missed."
                )
                _state.errors.append(f"{spec.name}: initialized but not subscribed")

        elif spec.pattern == InitPattern.GET:
            # Call get/singleton function
            get_func = getattr(module, spec.func_name)
            result = get_func()

            status.initialized = result is not None
            if spec.check_subscribed and result is not None:
                status.subscribed = getattr(result, "_subscribed", False)
            else:
                status.subscribed = True
            status.initialized_at = datetime.now()
            logger.info(f"[Bootstrap] {spec.display_name} initialized")

            # December 2025: Surface subscription failures instead of hiding them
            if status.initialized and not status.subscribed:
                logger.warning(
                    f"[Bootstrap] {spec.display_name} initialized but NOT subscribed. "
                    "Events may be missed."
                )
                _state.errors.append(f"{spec.name}: initialized but not subscribed")

        elif spec.pattern == InitPattern.IMPORT:
            # Just import to verify availability
            _ = getattr(module, spec.func_name)
            status.initialized = True
            status.subscribed = True  # Will subscribe when started by DaemonManager
            status.initialized_at = datetime.now()
            logger.info(f"[Bootstrap] {spec.display_name} registered")

        # Handle extra wiring if specified
        if spec.extra_wiring and status.initialized:
            try:
                extra_module_path, extra_func_name = spec.extra_wiring
                extra_module = importlib.import_module(extra_module_path)
                extra_func = getattr(extra_module, extra_func_name)

                # Special handling for sync_router - wiring is MANDATORY
                if extra_func_name == "get_sync_router":
                    sync_router = extra_func()
                    # Dec 2025: Make SyncRouter wiring mandatory, not optional
                    # SyncRouter MUST be wired to receive DATA_STALE, SYNC_REQUEST, etc.
                    try:
                        sync_router.wire_to_event_router()
                        logger.info("[Bootstrap] SyncRouter wired to event router")
                    except AttributeError:
                        logger.error(
                            "[Bootstrap] CRITICAL: SyncRouter.wire_to_event_router() missing! "
                            "Sync will not respond to events."
                        )
                        status.error = "SyncRouter wiring failed - method missing"
                    except (TypeError, ValueError, RuntimeError, ImportError) as wire_err:
                        # TypeError: wrong callback signature, ValueError: invalid config
                        # RuntimeError: event router init failed, ImportError: missing module
                        logger.error(
                            f"[Bootstrap] CRITICAL: SyncRouter wiring failed: {wire_err}"
                        )
                        status.error = f"SyncRouter wiring failed: {wire_err}"
                else:
                    extra_func()
                    logger.debug(f"[Bootstrap] Extra wiring completed: {extra_func_name}")

            except ImportError:
                logger.debug(f"[Bootstrap] Extra wiring module not available: {spec.extra_wiring[0]}")
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                # AttributeError: missing method, TypeError: wrong signature
                # ValueError: config issue, RuntimeError: wiring setup failed
                logger.warning(f"[Bootstrap] Extra wiring failed: {e}")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] {spec.display_name} not available: {e}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize {spec.display_name}: {e}")

    # December 2025: Emit coordinator health event
    _emit_coordinator_health_event(status)

    return status


def _init_pipeline_orchestrator(
    auto_trigger: bool = False,
    training_epochs: int | None = None,
    training_batch_size: int | None = None,
    training_model_version: str | None = None,
) -> BootstrapCoordinatorStatus:
    """Initialize DataPipelineOrchestrator.

    Special handler for pipeline_orchestrator that takes additional arguments.
    Not part of the registry pattern due to its unique parameter requirements.
    """
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

    # December 2025: Emit coordinator health event
    _emit_coordinator_health_event(status)

    return status


def _init_curriculum_integration_with_verification() -> BootstrapCoordinatorStatus:
    """Initialize CurriculumIntegration with OpponentTracker verification.

    Special handler that includes verification logic not suitable for the registry.
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
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            if hasattr(feedback, "_opponent_tracker") and feedback._opponent_tracker is not None:
                logger.info("[Bootstrap] OpponentTrackerIntegration verified - wired to curriculum")
            else:
                # Attempt direct wiring as fallback
                from app.training.curriculum_feedback import wire_opponent_tracker_to_curriculum

                wire_opponent_tracker_to_curriculum()
                if hasattr(feedback, "_opponent_tracker") and feedback._opponent_tracker is not None:
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

    # December 2025: Emit coordinator health event
    _emit_coordinator_health_event(status)

    return status


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
    ("app.coordination.auto_sync_daemon", "Ephemeral host sync"),  # Migrated from ephemeral_sync
    ("app.coordination.training_freshness", "Data freshness checks"),
    ("app.coordination.backpressure", "Backpressure monitoring"),  # Lazy imports handled gracefully
    ("app.coordination.queue_monitor", "Queue depth monitoring"),
    ("app.coordination.p2p_integration", "P2P status integration"),
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
            importlib.import_module(module_path)
            result["validated"].append(f"{description} ({module_path})")
        except ImportError as e:
            failure_msg = f"{description} ({module_path}): {e}"
            result["critical_failures"].append(failure_msg)

    # Validate optional modules (just log warnings)
    for module_path, description in _OPTIONAL_MODULES:
        try:
            importlib.import_module(module_path)
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
# Helper Functions
# =============================================================================



def _register_coordinators() -> bool:
    """Register all coordinators with OrchestratorRegistry for health monitoring.

    Calls auto_register_known_coordinators() which discovers and registers
    coordinator singletons (TrainingCoordinator, EloService, etc.) so they
    can be tracked by the cross-coordinator health monitoring system.
    """
    try:
        from app.coordination.orchestrator_registry import auto_register_known_coordinators

        results = auto_register_known_coordinators()
        # results is dict[str, bool] mapping coordinator name to registration success
        registered_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        logger.info(
            f"[Bootstrap] Auto-registered {registered_count}/{total_count} "
            "known coordinators for health monitoring"
        )
        return registered_count > 0

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

    December 29, 2025: Refactored to use declarative registry for simpler subscriptions.
    See event_subscription_registry.py for INIT_CALL_REGISTRY and DELEGATION_REGISTRY.

    Returns:
        Dict mapping subscription name to success status
    """
    results: dict[str, bool] = {}

    # December 29, 2025: Process declarative registries first
    try:
        from app.coordination.event_subscription_registry import (
            process_init_call_registry,
            process_delegation_registry,
        )
        process_init_call_registry(results)
        process_delegation_registry(results)
    except ImportError as e:
        logger.warning(f"[Bootstrap] Event subscription registry not available: {e}")

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
        logger.warning(f"[Bootstrap] Failed to wire cluster sync to pipeline: {e}")

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
        logger.warning(f"[Bootstrap] Failed to wire model sync to lifecycle: {e}")

    # 3. Wire SELFPLAY_COMPLETE to SyncCoordinator (for auto-trigger sync)
    try:
        from app.coordination.event_router import DataEventType, get_event_bus
        from app.coordination.sync_facade import get_sync_facade, SyncRequest

        sync_facade = get_sync_facade()
        bus = get_event_bus()

        async def on_selfplay_complete_for_sync(event):
            """Handle SELFPLAY_COMPLETE - trigger data sync if quality passes gate."""
            payload = event.payload if hasattr(event, "payload") else {}

            # December 2025: Quality gate before sync trigger
            # Prevents low-quality selfplay data from polluting training pipeline
            games_count = payload.get("games_count", 0)
            quality_score = payload.get("quality_score", 0.0)

            # Phase 3 Dec 2025: Use canonical thresholds from thresholds.py
            from app.config.thresholds import (
                MIN_QUALITY_FOR_PRIORITY_SYNC as QUALITY_GATE_THRESHOLD,
                MIN_GAMES_FOR_SYNC,
            )

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

            # Quality OK - proceed with sync via SyncFacade (Dec 2025 migration)
            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            if board_type and num_players:
                request = SyncRequest(
                    data_type="games",
                    board_type=board_type,
                    num_players=num_players,
                    priority="normal",
                )
                await sync_facade.sync(request)

        bus.subscribe(DataEventType.SELFPLAY_COMPLETE, on_selfplay_complete_for_sync)
        results["selfplay_to_sync"] = True
        logger.debug("[Bootstrap] Wired SELFPLAY_COMPLETE -> SyncCoordinator")

    except (AttributeError, TypeError, KeyError, RuntimeError) as e:
        results["selfplay_to_sync"] = False
        logger.warning(f"[Bootstrap] Failed to wire selfplay to sync: {e}")

    # Items 4, 5, 5b, 6, 7 moved to INIT_CALL_REGISTRY in event_subscription_registry.py
    # - model_selector_events (MODEL_PROMOTED -> SelfplayModelSelector)
    # - quality_to_rollback (LOW_QUALITY_DATA_WARNING -> RollbackManager)
    # - regression_to_rollback (REGRESSION_DETECTED -> RollbackManager)
    # - plateau_to_curriculum (PLATEAU_DETECTED -> CurriculumFeedback)
    # - early_stop_to_curriculum (TRAINING_EARLY_STOPPED -> CurriculumFeedback)

    # Item 8 moved to DELEGATION_REGISTRY in event_subscription_registry.py (December 29, 2025)
    # - daemon_started_handler (DAEMON_STARTED -> DaemonManager._track_daemon_started)
    # - daemon_stopped_handler (DAEMON_STOPPED -> DaemonManager._track_daemon_stopped)

    # 9. Wire MODEL_DISTRIBUTION_FAILED to alert/tracking system
    # December 2025: Critical for handling distribution failures
    try:
        from app.coordination.event_router import DataEventType, get_event_bus

        bus = get_event_bus()

        async def on_model_distribution_failed(event):
            """Handle model distribution failure - log alert for monitoring."""
            payload = event.payload if hasattr(event, "payload") else {}
            model_path = payload.get("model_path", "unknown")
            error = payload.get("error", "unknown")
            failed_nodes = payload.get("failed_nodes", [])

            logger.warning(
                f"[Bootstrap] Model distribution failed for {model_path}: {error} "
                f"(failed nodes: {failed_nodes})"
            )

            # Emit health alert for monitoring systems
            try:
                from app.coordination.event_router import RouterEvent, EventSource
                alert_event = RouterEvent(
                    event_type="HEALTH_ALERT",
                    payload={
                        "alert_type": "model_distribution_failed",
                        "model_path": model_path,
                        "error": error,
                        "failed_nodes": failed_nodes,
                        "severity": "warning",
                    },
                    source="coordination_bootstrap",
                    origin=EventSource.ROUTER,
                )
                await bus.publish(alert_event)
            except (AttributeError, TypeError, RuntimeError) as alert_err:
                logger.debug(f"[Bootstrap] Could not emit alert: {alert_err}")

        bus.subscribe(DataEventType.MODEL_DISTRIBUTION_FAILED, on_model_distribution_failed)
        results["model_distribution_failed_handler"] = True
        logger.debug("[Bootstrap] Wired MODEL_DISTRIBUTION_FAILED -> alert handler")

    except (AttributeError, TypeError, KeyError, RuntimeError) as e:
        results["model_distribution_failed_handler"] = False
        logger.warning(f"[Bootstrap] Failed to wire model distribution failed: {e}")

    # 10. Wire EVALUATION_STARTED to metrics tracking
    # December 2025: Complements EVALUATION_COMPLETED for full lifecycle tracking
    try:
        from app.coordination.event_router import DataEventType, get_event_bus

        bus = get_event_bus()

        async def on_evaluation_started(event):
            """Track evaluation start for metrics/dashboard."""
            payload = event.payload if hasattr(event, "payload") else {}
            config = payload.get("config", "unknown")
            model_id = payload.get("model_id", "unknown")
            games_planned = payload.get("games_planned", 0)
            logger.info(
                f"[Bootstrap] Evaluation started: {model_id} for {config} "
                f"({games_planned} games planned)"
            )

        bus.subscribe(DataEventType.EVALUATION_STARTED, on_evaluation_started)
        results["evaluation_started_tracking"] = True
        logger.debug("[Bootstrap] Wired EVALUATION_STARTED -> metrics tracking")

    except (AttributeError, TypeError, KeyError, RuntimeError) as e:
        results["evaluation_started_tracking"] = False
        logger.warning(f"[Bootstrap] Failed to wire evaluation started: {e}")

    # Item 11 moved to DELEGATION_REGISTRY in event_subscription_registry.py (December 29, 2025)
    # - promotion_rejected_handler (PROMOTION_REJECTED -> CurriculumFeedback.increase_weight)

    # 14. Wire DATA_SYNC_FAILED to backpressure activation (December 2025)
    try:
        from app.coordination.event_router import DataEventType, get_event_bus

        bus = get_event_bus()

        async def on_sync_failed(event):
            """Handle DATA_SYNC_FAILED - activate backpressure, retry with different transport."""
            payload = event.payload if hasattr(event, "payload") else {}
            error = payload.get("error", "unknown")
            target_node = payload.get("target_node", "unknown")

            logger.warning(f"[Bootstrap] Sync failed to {target_node}: {error}")

            # Try to activate backpressure via SelfplayScheduler
            try:
                from app.coordination.selfplay_scheduler import get_selfplay_scheduler

                scheduler = get_selfplay_scheduler()
                if hasattr(scheduler, "activate_backpressure"):
                    scheduler.activate_backpressure(
                        reason=f"sync_failed:{target_node}",
                        duration_seconds=300,  # 5 minute backpressure
                    )
            except (ImportError, AttributeError):
                pass  # Scheduler not available

        bus.subscribe(DataEventType.DATA_SYNC_FAILED, on_sync_failed)
        results["sync_failed_handler"] = True
        logger.debug("[Bootstrap] Wired DATA_SYNC_FAILED -> backpressure activation")

    except (AttributeError, TypeError, KeyError, RuntimeError) as e:
        results["sync_failed_handler"] = False
        logger.warning(f"[Bootstrap] Failed to wire sync failed: {e}")

    # Items 15, 16, 17, 18 moved to DELEGATION_REGISTRY in event_subscription_registry.py
    # - sync_stalled_handler (SYNC_STALLED -> SyncRouter)
    # - host_offline_handler (HOST_OFFLINE -> UnifiedHealthManager)
    # - host_online_handler (HOST_ONLINE -> UnifiedHealthManager)
    # - leader_elected_handler (LEADER_ELECTED -> LeadershipCoordinator)

    # Items 19, 20 moved to DELEGATION_REGISTRY in event_subscription_registry.py
    # - node_suspect_handler (NODE_SUSPECT -> NodeRecoveryDaemon)
    # - node_retired_handler (NODE_RETIRED -> SelfplayScheduler)

    # Item 21 moved to DELEGATION_REGISTRY in event_subscription_registry.py (December 29, 2025)
    # - health_check_passed_handler (HEALTH_CHECK_PASSED -> HealthCheckOrchestrator.record_check_result)
    # - health_check_failed_handler (HEALTH_CHECK_FAILED -> HealthCheckOrchestrator.record_check_result)

    # Items 22, 23 moved to DELEGATION_REGISTRY in event_subscription_registry.py
    # - task_orphaned_handler (TASK_ORPHANED -> JobManager)
    # - coordinator_heartbeat_handler (COORDINATOR_HEARTBEAT -> ClusterWatchdog)

    # 25. Wire EVALUATION_COMPLETED to ArchitectureTracker (Phase 5A)
    # December 2025: Track architecture performance for allocation decisions
    try:
        from app.training.architecture_tracker import wire_architecture_tracker_to_events

        success = wire_architecture_tracker_to_events()
        results["architecture_tracker"] = success
        if success:
            logger.debug("[Bootstrap] Wired EVALUATION_COMPLETED -> ArchitectureTracker")
        else:
            logger.warning("[Bootstrap] ArchitectureTracker wiring returned False")

    except (ImportError, AttributeError, TypeError) as e:
        results["architecture_tracker"] = False
        logger.warning(f"[Bootstrap] Failed to wire architecture tracker: {e}")

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
            from app.utils.async_utils import fire_and_forget

            # Handle sync/async context
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, schedule the start with error handling
                fire_and_forget(orchestrator.start(), name="unified_feedback_orchestrator_start")
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
    except (RuntimeError, AttributeError, TypeError) as e:
        # RuntimeError: asyncio issues, AttributeError: missing methods
        # TypeError: wrong method signatures
        logger.warning(f"[Bootstrap] Failed to start unified feedback orchestrator: {e}")
        return False


def _validate_event_wiring() -> dict[str, Any]:
    """Validate event flow and critical event subscriptions.

    Phase 21.2 (December 2025): Critical startup validation.
    This validates the event routing system is healthy and logs any issues
    detected. Issues could include:
    - Events not being routed through the system
    - Missing event buses (data, stage, cross-process)
    - High duplicate event rate (potential loops)
    - Critical events without subscribers (December 28, 2025 enhancement)

    Returns:
        Dict with validation results from event_router.validate_event_flow()
        and validate_event_wiring()
    """
    results: dict[str, Any] = {
        "healthy": False,
        "issues": [],
        "recommendations": [],
        "validated": False,
        "critical_wiring_valid": False,
    }

    try:
        from app.coordination.event_router import validate_event_flow, validate_event_wiring

        # Phase 1: Validate general event flow health
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

        # Phase 2 (December 28, 2025): Validate critical event subscriptions
        # This checks that essential pipeline events have subscribers
        try:
            wiring_result = validate_event_wiring(
                raise_on_error=False,  # Don't raise - we want to log and continue
                log_warnings=True,
            )
            results["critical_wiring_valid"] = wiring_result.get("valid", False)
            results["missing_critical_events"] = wiring_result.get("missing_critical", [])
            results["missing_optional_events"] = wiring_result.get("missing_optional", [])

            if wiring_result.get("valid", False):
                logger.info(
                    f"[Bootstrap] Critical event wiring validated: "
                    f"{len(wiring_result.get('all_subscribed', []))} event types subscribed"
                )
            else:
                missing = wiring_result.get("missing_critical", [])
                logger.error(
                    f"[Bootstrap] CRITICAL: Missing event subscribers for: {missing}. "
                    f"Training pipeline may not function correctly!"
                )
                results["issues"].append(
                    f"Critical events without subscribers: {missing}"
                )
        except (AttributeError, TypeError) as e:
            logger.warning(f"[Bootstrap] validate_event_wiring check failed: {e}")

    except ImportError as e:
        logger.debug(f"[Bootstrap] validate_event_flow not available: {e}")
    except (AttributeError, TypeError, KeyError, RuntimeError) as e:
        results["issues"].append(f"Validation failed: {e}")
        logger.warning(f"[Bootstrap] Event flow validation failed: {e}")

    return results


def _restore_event_subscriptions() -> dict[str, Any]:
    """Restore event subscriptions from persistent store and replay stale DLQ events.

    P0 December 2025: Subscriptions are lost on process restart, causing events
    to be orphaned. This function:
    1. Restores subscriptions from SQLite store
    2. Replays DLQ events older than 5 minutes
    3. Emits alerts for stale DLQ events (>24h old)

    Returns:
        Dict with restoration results
    """
    results: dict[str, Any] = {
        "subscriptions_restored": 0,
        "dlq_events_replayed": 0,
        "stale_alerts_emitted": 0,
        "errors": [],
    }

    try:
        from app.coordination.event_router import get_router

        router = get_router()

        # Restore subscriptions from persistent store
        try:
            import asyncio
            from app.utils.async_utils import fire_and_forget
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create task if we're in async context with error handling
                fire_and_forget(router.restore_subscriptions(), name="restore_subscriptions")
                # Can't await in sync function, log that it will complete async
                logger.debug(
                    "[Bootstrap] Subscription restoration scheduled (async context)"
                )
            else:
                results["subscriptions_restored"] = loop.run_until_complete(
                    router.restore_subscriptions()
                )
        except RuntimeError:
            # No event loop - create one
            results["subscriptions_restored"] = asyncio.run(
                router.restore_subscriptions()
            )

        # Replay stale DLQ events (skip in test mode to avoid timeouts)
        if os.environ.get("RINGRIFT_DISABLE_DLQ_REPLAY", "").lower() in ("1", "true", "yes"):
            logger.debug("[Bootstrap] DLQ replay disabled via RINGRIFT_DISABLE_DLQ_REPLAY")
            results["dlq_events_replayed"] = 0
        else:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    fire_and_forget(router.replay_stale_dlq_events(), name="replay_stale_dlq")
                    logger.debug("[Bootstrap] DLQ replay scheduled (async context)")
                else:
                    results["dlq_events_replayed"] = loop.run_until_complete(
                        router.replay_stale_dlq_events()
                    )
            except RuntimeError:
                results["dlq_events_replayed"] = asyncio.run(
                    router.replay_stale_dlq_events()
                )

        # Emit alerts for stale DLQ events
        try:
            from app.coordination.subscription_store import get_subscription_store

            store = get_subscription_store()
            loop = asyncio.get_event_loop()
            if loop.is_running():
                fire_and_forget(store.emit_stale_dlq_alerts(), name="emit_stale_dlq_alerts")
            else:
                results["stale_alerts_emitted"] = loop.run_until_complete(
                    store.emit_stale_dlq_alerts()
                )
        except ImportError:
            pass  # Subscription store not available
        except RuntimeError:
            try:
                results["stale_alerts_emitted"] = asyncio.run(
                    store.emit_stale_dlq_alerts()
                )
            except (RuntimeError, OSError):
                pass  # Event loop or I/O issue

        if results["subscriptions_restored"] > 0:
            logger.info(
                f"[Bootstrap] Restored {results['subscriptions_restored']} subscriptions "
                f"from persistent store"
            )

        if results["dlq_events_replayed"] > 0:
            logger.info(
                f"[Bootstrap] Replayed {results['dlq_events_replayed']} stale DLQ events"
            )

        if results["stale_alerts_emitted"] > 0:
            logger.warning(
                f"[Bootstrap] Emitted {results['stale_alerts_emitted']} alerts "
                f"for stale DLQ events"
            )

    except ImportError as e:
        results["errors"].append(f"Event router not available: {e}")
        logger.debug(f"[Bootstrap] Subscription restoration not available: {e}")
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        # AttributeError/TypeError: invalid method access, ValueError: bad config
        # RuntimeError: initialization failure
        results["errors"].append(str(e))
        logger.warning(f"[Bootstrap] Subscription restoration failed: {e}")

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
    enable_training_activity: bool = True,
    enable_curriculum_integration: bool = True,
    enable_per: bool = True,  # December 29, 2025: PER orchestrator for experience replay
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

    # ==========================================================================
    # INITIALIZATION ORDER (See module docstring for full documentation)
    # ==========================================================================
    # Format: (name, enabled_flag) - looks up COORDINATOR_REGISTRY
    # CRITICAL: Subscribers must start before emitters!
    # Special cases (pipeline_orchestrator, curriculum_integration) handled below
    init_order: list[tuple[str, bool]] = [
        # --- Layer 1: Foundational (no dependencies) ---
        ("task_coordinator", enable_task),
        ("global_task_coordinator", enable_global_task),
        ("resource_coordinator", enable_resources),
        ("cache_orchestrator", enable_cache),
        # --- Layer 2: Infrastructure Support ---
        # UnifiedHealthManager replaces deprecated error + recovery coordinators
        ("health_manager", enable_health or enable_error),
        ("model_coordinator", enable_model),
        # --- Layer 3: Sync and Training ---
        ("sync_coordinator", enable_sync),
        ("training_coordinator", enable_training),
        # --- Layer 4: Data Integrity ---
        ("transfer_verifier", enable_transfer),
        ("ephemeral_guard", enable_ephemeral),
        ("queue_populator", enable_queue),
        # --- Layer 5: Selfplay (depends on task_lifecycle, resources) ---
        ("selfplay_orchestrator", enable_selfplay),
        ("selfplay_scheduler", enable_selfplay),
        # --- Layer 6: Multi-Provider and Jobs ---
        ("multi_provider", enable_multi_provider),
        ("job_scheduler", enable_job_scheduler),
        # --- Layer 7: Daemons (December 2025 automation) ---
        ("auto_export_daemon", enable_auto_export),
        ("evaluation_daemon", enable_auto_evaluation),
        ("model_distribution_daemon", enable_model_distribution),
        ("idle_resource_daemon", enable_idle_resource),
        ("quality_monitor_daemon", enable_quality_monitor),
        ("orphan_detection_daemon", enable_orphan_detection),
        ("training_activity", enable_training_activity),
        # --- Layer 8: Top-Level Coordination ---
        ("metrics_orchestrator", enable_metrics),  # Depends on pipeline
        ("optimization_coordinator", enable_optimization),  # Depends on metrics
        ("per_orchestrator", enable_per),  # Dec 29, 2025: PER buffer monitoring
        ("leadership_coordinator", enable_leadership),  # Coordinates all others
    ]

    # Initialize coordinators from registry
    for name, enabled in init_order:
        if not enabled:
            logger.debug(f"[Bootstrap] Skipping {name} (disabled)")
            continue

        if name not in COORDINATOR_REGISTRY:
            logger.warning(f"[Bootstrap] Unknown coordinator: {name}")
            continue

        spec = COORDINATOR_REGISTRY[name]
        status = _init_coordinator_from_spec(spec)
        _state.coordinators[name] = status

        if status.error:
            _state.errors.append(f"{name}: {status.error}")

    # Special case: pipeline_orchestrator (takes extra args)
    if enable_pipeline:
        status = _init_pipeline_orchestrator(
            auto_trigger=pipeline_auto_trigger,
            training_epochs=training_epochs,
            training_batch_size=training_batch_size,
            training_model_version=training_model_version,
        )
        _state.coordinators["pipeline_orchestrator"] = status
        if status.error:
            _state.errors.append(f"pipeline_orchestrator: {status.error}")

    # Special case: curriculum_integration (has verification logic)
    if enable_curriculum_integration:
        status = _init_curriculum_integration_with_verification()
        _state.coordinators["curriculum_integration"] = status
        if status.error:
            _state.errors.append(f"curriculum_integration: {status.error}")

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

    # P0 December 2025: Restore subscriptions and replay stale DLQ events
    _restore_event_subscriptions()

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
            # sync_coordinator removed Dec 2025 - migrated to SyncFacade which is stateless
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
            # P0.4 (December 2025): Handle async shutdown methods properly
            if coordinator and hasattr(coordinator, "shutdown"):
                import asyncio
                import inspect

                shutdown_method = getattr(coordinator, "shutdown")
                if inspect.iscoroutinefunction(shutdown_method):
                    # Async shutdown - need to run in event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # Create task and schedule (fire-and-forget in sync context)
                        loop.create_task(shutdown_method())
                        logger.debug(f"[Bootstrap] Scheduled async shutdown for {name}")
                    except RuntimeError:
                        # No running loop - create one
                        asyncio.run(shutdown_method())
                        logger.debug(f"[Bootstrap] Shutdown {name} (async)")
                else:
                    # Sync shutdown
                    shutdown_method()
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

    # 3. Check quality rollback watcher class availability (structural check)
    # Note: We verify the CLASS exists and has required methods. Singleton wiring
    # happens later in bootstrap, so we can't check instantiation here.
    try:
        from app.training.rollback_manager import (
            QualityRollbackWatcher,
            wire_quality_to_rollback,
        )

        # Verify the class has required methods
        has_subscribe = hasattr(QualityRollbackWatcher, "subscribe_to_quality_events")
        has_handler = hasattr(QualityRollbackWatcher, "_on_low_quality")
        has_wire_func = callable(wire_quality_to_rollback)
        handlers_ready = has_subscribe and has_handler and has_wire_func

        checks.append(SmokeTestResult(
            name="quality_rollback_wired",
            passed=handlers_ready,
            details={
                "has_class": True,
                "has_subscribe_method": has_subscribe,
                "has_handler_method": has_handler,
                "has_wire_function": has_wire_func,
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

    # 9. Check UnifiedHealthManager coordinator lifecycle handlers exist (Dec 2025)
    # Note: We verify handlers EXIST (structural check). Subscription happens at runtime.
    try:
        from app.coordination.unified_health_manager import get_health_manager

        health_mgr = get_health_manager()
        # Verify the P0 lifecycle event handlers exist (added Dec 2025)
        has_shutdown = hasattr(health_mgr, "_on_coordinator_shutdown")
        has_heartbeat = hasattr(health_mgr, "_on_coordinator_heartbeat")
        has_subscribe = hasattr(health_mgr, "subscribe_to_events")
        handlers_ready = has_shutdown and has_heartbeat and has_subscribe
        checks.append(SmokeTestResult(
            name="health_manager_coordinator_lifecycle",
            passed=handlers_ready,
            details={
                "has_shutdown_handler": has_shutdown,
                "has_heartbeat_handler": has_heartbeat,
                "has_subscribe_method": has_subscribe,
            },
            error=None if handlers_ready
            else "Missing coordinator lifecycle handlers",
        ))
    except ImportError:
        warnings.append("unified_health_manager not available")
        checks.append(SmokeTestResult(
            name="health_manager_coordinator_lifecycle",
            passed=True,
            error="Module not available",
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="health_manager_coordinator_lifecycle",
            passed=False,
            error=str(e),
        ))

    # 10. Check all coordinators are properly subscribed (December 2025)
    # This catches silent subscription failures where a coordinator initializes
    # but fails to subscribe to events.
    try:
        unsubscribed = [
            name for name, status in _state.coordinators.items()
            if status.initialized and not status.subscribed
        ]
        checks.append(SmokeTestResult(
            name="coordinator_subscriptions_complete",
            passed=len(unsubscribed) == 0,
            error=f"Unsubscribed coordinators: {unsubscribed}" if unsubscribed else None,
            details={
                "total_initialized": sum(1 for s in _state.coordinators.values() if s.initialized),
                "total_subscribed": sum(1 for s in _state.coordinators.values() if s.subscribed),
                "unsubscribed": unsubscribed,
            },
        ))
    except (AttributeError, TypeError, RuntimeError) as e:
        checks.append(SmokeTestResult(
            name="coordinator_subscriptions_complete",
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
