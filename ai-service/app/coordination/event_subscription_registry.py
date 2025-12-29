"""Declarative event subscription registry for coordination bootstrap.

December 29, 2025: Extracted from coordination_bootstrap.py to reduce the
740+ LOC _wire_missing_event_subscriptions() function to a declarative registry.

This module provides:
1. EventSubscriptionSpec - Dataclass for defining event subscriptions
2. INIT_CALL_REGISTRY - Registry for simple init-function subscriptions
3. DELEGATION_REGISTRY - Registry for event->method delegation subscriptions
4. Helper functions for registering subscriptions with standard error handling
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from app.coordination.event_router import DataEventType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InitCallSpec:
    """Specification for init-function style event subscriptions.

    These are subscriptions that just call an initialization function
    (like wire_*() helpers) rather than subscribing directly to events.
    """

    name: str
    import_path: str
    function_name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class DelegationSpec:
    """Specification for event->orchestrator delegation subscriptions.

    These subscriptions forward events to orchestrator methods with
    optional payload extraction and fallback methods.
    """

    name: str
    event_type: str  # DataEventType enum name (e.g., "HOST_OFFLINE")
    orchestrator_import: str  # Module path to import orchestrator
    orchestrator_getter: str  # Function to get orchestrator instance
    primary_method: str  # Primary method to call
    fallback_method: Optional[str] = None  # Fallback if primary doesn't exist
    payload_keys: tuple[str, ...] = ()  # Keys to extract from payload
    log_level: str = "info"  # Log level for event receipt
    description: str = ""
    fixed_kwargs: dict[str, Any] = field(default_factory=dict)  # Fixed kwargs to pass


# Registry of init-function style subscriptions
# These call wire_*() or _init_*() functions during bootstrap
INIT_CALL_REGISTRY: tuple[InitCallSpec, ...] = (
    InitCallSpec(
        name="model_selector_events",
        import_path="app.training.selfplay_model_selector",
        function_name="_init_event_subscription",
        description="Initialize SelfplayModelSelector MODEL_PROMOTED subscription",
    ),
    InitCallSpec(
        name="quality_to_rollback",
        import_path="app.training.rollback_manager",
        function_name="wire_quality_to_rollback",
        description="Wire LOW_QUALITY_DATA_WARNING -> RollbackManager",
    ),
    InitCallSpec(
        name="regression_to_rollback",
        import_path="app.training.rollback_manager",
        function_name="wire_regression_to_rollback",
        kwargs={
            "auto_rollback_enabled": True,
            "require_approval_for_severe": False,
        },
        description="Wire REGRESSION_DETECTED -> RollbackManager (full auto-rollback)",
    ),
    InitCallSpec(
        name="plateau_to_curriculum",
        import_path="app.training.curriculum_feedback",
        function_name="wire_plateau_to_curriculum",
        description="Wire PLATEAU_DETECTED -> CurriculumFeedback",
    ),
    InitCallSpec(
        name="early_stop_to_curriculum",
        import_path="app.training.curriculum_feedback",
        function_name="wire_early_stop_to_curriculum",
        description="Wire TRAINING_EARLY_STOPPED -> CurriculumFeedback",
    ),
)


# Registry of event->orchestrator delegation subscriptions
# These forward events directly to orchestrator methods
DELEGATION_REGISTRY: tuple[DelegationSpec, ...] = (
    DelegationSpec(
        name="host_offline_handler",
        event_type="HOST_OFFLINE",
        orchestrator_import="app.coordination.unified_health_manager",
        orchestrator_getter="get_unified_health_manager",
        primary_method="handle_node_offline",
        fallback_method="mark_node_unhealthy",
        payload_keys=("node_id", "peer_id"),
        description="Wire HOST_OFFLINE -> UnifiedHealthManager",
    ),
    DelegationSpec(
        name="host_online_handler",
        event_type="HOST_ONLINE",
        orchestrator_import="app.coordination.unified_health_manager",
        orchestrator_getter="get_unified_health_manager",
        primary_method="handle_node_online",
        fallback_method="mark_node_healthy",
        payload_keys=("node_id", "peer_id"),
        description="Wire HOST_ONLINE -> UnifiedHealthManager",
    ),
    DelegationSpec(
        name="leader_elected_handler",
        event_type="LEADER_ELECTED",
        orchestrator_import="app.coordination.leadership_coordinator",
        orchestrator_getter="get_leadership_coordinator",
        primary_method="on_leader_change",
        fallback_method="set_leader",
        payload_keys=("leader_id", "new_leader", "previous_leader"),
        description="Wire LEADER_ELECTED -> LeadershipCoordinator",
    ),
    DelegationSpec(
        name="node_suspect_handler",
        event_type="NODE_SUSPECT",
        orchestrator_import="app.coordination.node_recovery_daemon",
        orchestrator_getter="get_node_recovery_daemon",
        primary_method="_on_node_suspect",
        fallback_method="mark_suspect",
        payload_keys=("node_id", "peer_id", "reason"),
        description="Wire NODE_SUSPECT -> NodeRecoveryDaemon",
    ),
    DelegationSpec(
        name="node_retired_handler",
        event_type="NODE_RETIRED",
        orchestrator_import="app.coordination.selfplay_scheduler",
        orchestrator_getter="get_selfplay_scheduler",
        primary_method="_on_node_retired",
        fallback_method="remove_node",
        payload_keys=("node_id", "peer_id"),
        description="Wire NODE_RETIRED -> SelfplayScheduler",
    ),
    DelegationSpec(
        name="task_orphaned_handler",
        event_type="TASK_ORPHANED",
        orchestrator_import="scripts.p2p.managers.job_manager",
        orchestrator_getter="JobManager.get_instance",
        primary_method="_on_task_orphaned",
        fallback_method="cleanup_orphan",
        payload_keys=("task_id", "job_type", "node_id"),
        log_level="warning",
        description="Wire TASK_ORPHANED -> JobManager cleanup",
    ),
    DelegationSpec(
        name="coordinator_heartbeat_handler",
        event_type="COORDINATOR_HEARTBEAT",
        orchestrator_import="app.coordination.cluster_watchdog_daemon",
        orchestrator_getter="get_cluster_watchdog",
        primary_method="record_heartbeat",
        payload_keys=("coordinator_name", "name", "hostname"),
        log_level="debug",
        description="Wire COORDINATOR_HEARTBEAT -> ClusterWatchdog",
    ),
    DelegationSpec(
        name="sync_stalled_handler",
        event_type="SYNC_STALLED",
        orchestrator_import="app.coordination.sync_router",
        orchestrator_getter="get_sync_router",
        primary_method="mark_transport_failed",
        payload_keys=("target_node", "stall_duration_seconds"),
        description="Wire SYNC_STALLED -> SyncRouter transport escalation",
    ),
    # December 29, 2025: Extracted from coordination_bootstrap.py
    DelegationSpec(
        name="daemon_started_handler",
        event_type="DAEMON_STARTED",
        orchestrator_import="app.coordination.daemon_manager",
        orchestrator_getter="get_daemon_manager",
        primary_method="_track_daemon_started",
        payload_keys=("daemon_name", "hostname"),
        log_level="info",
        description="Wire DAEMON_STARTED -> DaemonManager lifecycle tracking",
    ),
    DelegationSpec(
        name="daemon_stopped_handler",
        event_type="DAEMON_STOPPED",
        orchestrator_import="app.coordination.daemon_manager",
        orchestrator_getter="get_daemon_manager",
        primary_method="_track_daemon_stopped",
        payload_keys=("daemon_name", "hostname", "reason"),
        log_level="info",
        description="Wire DAEMON_STOPPED -> DaemonManager lifecycle tracking",
    ),
    DelegationSpec(
        name="health_check_passed_handler",
        event_type="HEALTH_CHECK_PASSED",
        orchestrator_import="app.coordination.health_check_orchestrator",
        orchestrator_getter="get_health_orchestrator",
        primary_method="record_check_result",
        payload_keys=("node_id", "hostname", "check_type"),
        log_level="debug",
        description="Wire HEALTH_CHECK_PASSED -> HealthCheckOrchestrator",
        fixed_kwargs={"passed": True},
    ),
    DelegationSpec(
        name="health_check_failed_handler",
        event_type="HEALTH_CHECK_FAILED",
        orchestrator_import="app.coordination.health_check_orchestrator",
        orchestrator_getter="get_health_orchestrator",
        primary_method="record_check_result",
        payload_keys=("node_id", "hostname", "check_type"),
        log_level="debug",
        description="Wire HEALTH_CHECK_FAILED -> HealthCheckOrchestrator",
        fixed_kwargs={"passed": False},
    ),
    DelegationSpec(
        name="promotion_rejected_handler",
        event_type="PROMOTION_REJECTED",
        orchestrator_import="app.training.curriculum_feedback",
        orchestrator_getter="get_curriculum_feedback",
        primary_method="increase_weight",
        payload_keys=("config", "model_id"),
        log_level="info",
        description="Wire PROMOTION_REJECTED -> CurriculumFeedback weight increase",
        fixed_kwargs={"boost_factor": 1.1},
    ),
)


def process_init_call_registry(results: dict[str, bool]) -> None:
    """Process all init-function subscriptions from INIT_CALL_REGISTRY.

    Args:
        results: Dict to track success/failure of each subscription
    """
    for spec in INIT_CALL_REGISTRY:
        try:
            # Dynamic import
            module = importlib.import_module(spec.import_path)
            func = getattr(module, spec.function_name)

            # Call the init function
            if spec.kwargs:
                # For functions needing extra args (like wire_regression_to_rollback)
                # Some functions need a registry argument
                if "registry" in func.__code__.co_varnames[:func.__code__.co_argcount]:
                    from app.training.model_registry import get_model_registry
                    registry = get_model_registry()
                    result = func(registry, **spec.kwargs)
                else:
                    result = func(**spec.kwargs)
            else:
                result = func()

            # Track success (some functions return the handler, some return None)
            results[spec.name] = result is not None or spec.kwargs == {}
            if results[spec.name]:
                logger.debug(f"[Bootstrap] {spec.description}")
            else:
                logger.warning(f"[Bootstrap] {spec.description} - returned None")

        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            results[spec.name] = False
            logger.debug(f"[Bootstrap] Failed {spec.name}: {e}")


def process_delegation_registry(results: dict[str, bool]) -> None:
    """Process all delegation subscriptions from DELEGATION_REGISTRY.

    Args:
        results: Dict to track success/failure of each subscription
    """
    try:
        from app.coordination.event_router import DataEventType, get_event_bus
        bus = get_event_bus()
    except ImportError as e:
        logger.warning(f"[Bootstrap] Cannot process delegation registry - event bus unavailable: {e}")
        for spec in DELEGATION_REGISTRY:
            results[spec.name] = False
        return

    for spec in DELEGATION_REGISTRY:
        try:
            # Get the event type enum value
            event_type = getattr(DataEventType, spec.event_type, None)
            if event_type is None:
                logger.debug(f"[Bootstrap] Event type {spec.event_type} not found, skipping {spec.name}")
                results[spec.name] = False
                continue

            # Create the handler using closure to capture spec
            handler = _create_delegation_handler(spec)

            # Subscribe to event
            bus.subscribe(event_type, handler)
            results[spec.name] = True
            logger.debug(f"[Bootstrap] {spec.description}")

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            results[spec.name] = False
            logger.debug(f"[Bootstrap] Failed {spec.name}: {e}")


def _create_delegation_handler(spec: DelegationSpec) -> Callable:
    """Create an async handler function for a delegation spec.

    Args:
        spec: The delegation specification

    Returns:
        Async handler function
    """
    async def handler(event) -> None:
        """Generic delegation handler generated from DelegationSpec."""
        payload = event.payload if hasattr(event, "payload") else {}

        # Extract primary identifier from payload
        identifier = None
        for key in spec.payload_keys:
            identifier = payload.get(key)
            if identifier:
                break

        if not identifier and spec.payload_keys:
            return  # Skip if no identifier found

        # Log receipt
        log_func = getattr(logger, spec.log_level, logger.info)
        log_func(f"[Bootstrap] {spec.event_type}: {identifier}")

        try:
            # Import orchestrator module
            module = importlib.import_module(spec.orchestrator_import)

            # Get orchestrator instance (handle Class.get_instance() pattern)
            if "." in spec.orchestrator_getter:
                class_name, method_name = spec.orchestrator_getter.split(".")
                cls = getattr(module, class_name)
                orchestrator = getattr(cls, method_name)()
            else:
                getter = getattr(module, spec.orchestrator_getter)
                orchestrator = getter()

            # Build kwargs from fixed_kwargs and payload
            kwargs = dict(spec.fixed_kwargs) if spec.fixed_kwargs else {}

            # Try primary method
            if hasattr(orchestrator, spec.primary_method):
                method = getattr(orchestrator, spec.primary_method)
                # Check if method is async
                if hasattr(method, "__call__"):
                    import asyncio
                    if asyncio.iscoroutinefunction(method):
                        # Pass args based on method signature
                        if "reason" in spec.payload_keys:
                            reason = payload.get("reason", "unknown")
                            await method(identifier, reason, **kwargs)
                        elif "previous_leader" in spec.payload_keys:
                            previous = payload.get("previous_leader")
                            await method(identifier, previous, **kwargs)
                        elif "check_type" in spec.payload_keys:
                            # Health check methods: record_check_result(node_id, check_type, passed)
                            check_type = payload.get("check_type", "unknown")
                            await method(identifier, check_type, **kwargs)
                        elif "boost_factor" in kwargs:
                            # Curriculum methods: increase_weight(config, boost_factor)
                            await method(identifier, **kwargs)
                        else:
                            await method(identifier, **kwargs) if kwargs else await method(identifier)
                    else:
                        if "reason" in spec.payload_keys:
                            reason = payload.get("reason", "unknown")
                            method(identifier, reason, **kwargs)
                        elif "check_type" in spec.payload_keys:
                            check_type = payload.get("check_type", "unknown")
                            method(identifier, check_type, **kwargs)
                        elif "boost_factor" in kwargs:
                            method(identifier, **kwargs)
                        else:
                            method(identifier, **kwargs) if kwargs else method(identifier)
            elif spec.fallback_method and hasattr(orchestrator, spec.fallback_method):
                method = getattr(orchestrator, spec.fallback_method)
                if hasattr(method, "__call__"):
                    import asyncio
                    if asyncio.iscoroutinefunction(method):
                        await method(identifier, **kwargs) if kwargs else await method(identifier)
                    else:
                        method(identifier, **kwargs) if kwargs else method(identifier)

        except (ImportError, AttributeError) as e:
            logger.debug(f"[Bootstrap] Orchestrator unavailable for {spec.name}: {e}")

    return handler


def get_registry_stats() -> dict[str, int]:
    """Get statistics about the event subscription registries.

    Returns:
        Dict with counts of init calls and delegations
    """
    return {
        "init_call_count": len(INIT_CALL_REGISTRY),
        "delegation_count": len(DELEGATION_REGISTRY),
        "total_subscriptions": len(INIT_CALL_REGISTRY) + len(DELEGATION_REGISTRY),
    }
