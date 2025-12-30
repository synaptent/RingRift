"""Cluster coordination package for unified resource management.

Provides centralized task coordination to prevent uncontrolled task spawning.

Primary modules:
1. task_coordinator - SQLite-backed coordination with rate limiting, backpressure
2. orchestrator_registry - Role-based mutual exclusion with heartbeat liveness
3. safeguards - Circuit breakers, resource monitoring, spawn rate tracking
4. queue_monitor - Queue depth monitoring with backpressure signals
5. bandwidth_manager - Network bandwidth allocation for transfers
6. sync_mutex - Cross-process mutex for rsync operations
7. p2p_backend - REST API client for P2P orchestrator cluster
8. job_scheduler - Priority-based job scheduling with Elo curriculum
9. stage_events - Event-driven pipeline orchestration with callbacks

December 2025: Reorganized into submodule exports for maintainability.
Imports are organized into:
- _exports_core.py - Task coordination, orchestrator registry, resources, health
- _exports_sync.py - Sync operations (bandwidth, mutex, WAL, integrity)
- _exports_daemon.py - Daemon management
- _exports_events.py - Event system (router, emitters, stage events)
- _exports_orchestrators.py - High-level orchestrators
- _exports_utils.py - Utilities and helpers

Usage:
    # Task coordination (canonical)
    from app.coordination import TaskCoordinator, TaskType
    coordinator = TaskCoordinator.get_instance()
    if coordinator.can_spawn_task(TaskType.SELFPLAY, "node-1")[0]:
        coordinator.register_task(task_id, TaskType.SELFPLAY, "node-1")

    # Orchestrator role management
    from app.coordination import acquire_orchestrator_role, OrchestratorRole
    if acquire_orchestrator_role(OrchestratorRole.CLUSTER_ORCHESTRATOR):
        # This process now holds the orchestrator role
        pass

    # Backpressure checking
    from app.coordination import should_throttle_production, QueueType
    if should_throttle_production(QueueType.TRAINING_DATA):
        # Slow down or skip data production
        pass
"""

import contextlib

# =============================================================================
# Submodule Exports (December 2025 - organized for maintainability)
# =============================================================================

# Core coordination (task, orchestrator registry, queue, resources, health, P2P)
from app.coordination._exports_core import *
from app.coordination._exports_core import __all__ as _core_all

# Sync operations (bandwidth, mutex, WAL, integrity, bloom filter)
from app.coordination._exports_sync import *
from app.coordination._exports_sync import __all__ as _sync_all

# Daemon management
from app.coordination._exports_daemon import *
from app.coordination._exports_daemon import __all__ as _daemon_all

# Event system (router, emitters, cross-process, stage events)
from app.coordination._exports_events import *
from app.coordination._exports_events import __all__ as _events_all

# High-level orchestrators
from app.coordination._exports_orchestrators import *
from app.coordination._exports_orchestrators import __all__ as _orchestrators_all

# Utilities and helpers
from app.coordination._exports_utils import *
from app.coordination._exports_utils import __all__ as _utils_all

# Core Utilities - Consolidated Module (December 2025)
# This module consolidates: tracing, distributed_lock, optional_imports, yaml_utils
from app.coordination import core_utils

# Core Events - Consolidated Module (December 2025)
# This module consolidates: event_router, event_mappings, event_emitters, event_normalization
from app.coordination import core_events


# =============================================================================
# Module-level singleton placeholders for cleanup in shutdown_all_coordinators
# =============================================================================
_selfplay_orchestrator = None
_pipeline_orchestrator = None
_task_lifecycle_coordinator = None
_optimization_coordinator = None
_metrics_orchestrator = None
_resource_coordinator = None
_cache_orchestrator = None
_event_coordinator = None


def _init_with_retry(
    name: str,
    init_func,
    max_retries: int = 3,
    base_delay: float = 0.5,
    logger=None,
) -> tuple:
    """Initialize a coordinator with retry logic.

    Args:
        name: Coordinator name for logging
        init_func: Function that returns (instance, subscribed_flag)
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff
        logger: Logger instance

    Returns:
        (instance, success, error_message)
    """
    import time as _time

    last_error = None

    for attempt in range(max_retries):
        try:
            instance, subscribed = init_func()

            if not subscribed:
                raise RuntimeError(f"{name} failed to subscribe to events")

            if logger:
                if attempt > 0:
                    logger.info(f"[init_with_retry] {name} succeeded on attempt {attempt + 1}")
                else:
                    logger.info(f"[initialize_all_coordinators] {name} wired")

            return (instance, True, None)

        except Exception as e:
            last_error = str(e)
            if logger:
                logger.warning(
                    f"[init_with_retry] {name} attempt {attempt + 1}/{max_retries} failed: {e}"
                )

            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                _time.sleep(delay)

    if logger:
        logger.error(f"[initialize_all_coordinators] {name} failed after {max_retries} attempts")

    return (None, False, last_error)


def initialize_all_coordinators(
    auto_trigger_pipeline: bool = False,
    heartbeat_threshold: float = 60.0,
    max_retries: int = 3,
    retry_delay: float = 0.5,
    wrap_handlers: bool = True,
) -> dict:
    """Initialize all orchestrators and coordinators with event wiring (December 2025).

    This is the single entry point to bootstrap all coordination infrastructure.
    It wires all event subscriptions and returns a status dictionary.

    Features:
    - Retry logic with exponential backoff for failed subscriptions
    - Validation that subscriptions actually succeeded
    - Emits COORDINATOR_INIT_FAILED for persistent failures
    - Optionally wraps handlers with resilience (exception boundaries + timeouts)

    Args:
        auto_trigger_pipeline: If True, pipeline stages auto-trigger downstream
        heartbeat_threshold: Seconds without heartbeat to mark tasks orphaned
        max_retries: Maximum retry attempts per coordinator
        retry_delay: Base delay for exponential backoff
        wrap_handlers: If True, wrap handlers with resilience

    Returns:
        Dict with initialization status for each orchestrator
    """
    import asyncio
    import logging

    logger = logging.getLogger(__name__)

    status = {
        "dead_letter_queue": False,
        "task_lifecycle": False,
        "resources": False,
        "cache": False,
        "selfplay": False,
        "pipeline": False,
        "optimization": False,
        "metrics": False,
        "event_coordinator": False,
    }
    errors = {}
    instances = {}

    # Define init functions that return (instance, subscribed)
    def init_task_lifecycle():
        coord = wire_task_events(heartbeat_threshold=heartbeat_threshold)
        return (coord, coord._subscribed)

    def init_resources():
        coord = wire_resource_events()
        return (coord, coord._subscribed)

    def init_cache():
        coord = wire_cache_events()
        return (coord, coord._subscribed)

    def init_selfplay():
        coord = wire_selfplay_events()
        return (coord, coord._subscribed)

    def init_pipeline():
        coord = wire_pipeline_events(auto_trigger=auto_trigger_pipeline)
        return (coord, coord._subscribed)

    def init_optimization():
        coord = wire_optimization_events()
        return (coord, coord._subscribed)

    def init_metrics():
        coord = wire_metrics_events()
        return (coord, coord._subscribed)

    # Initialize in dependency order
    init_order = [
        ("task_lifecycle", init_task_lifecycle, []),
        ("resources", init_resources, []),
        ("cache", init_cache, []),
        ("selfplay", init_selfplay, ["task_lifecycle"]),
        ("pipeline", init_pipeline, ["task_lifecycle", "selfplay"]),
        ("optimization", init_optimization, ["pipeline"]),
        ("metrics", init_metrics, ["task_lifecycle"]),
    ]

    # Initialize Dead Letter Queue first
    dlq = None
    try:
        from app.coordination.dead_letter_queue import enable_dead_letter_queue, get_dead_letter_queue

        dlq = get_dead_letter_queue()
        status["dead_letter_queue"] = True
        instances["dead_letter_queue"] = dlq
        logger.info("[initialize_all_coordinators] Dead letter queue initialized")
    except Exception as e:
        logger.warning(f"[initialize_all_coordinators] Dead letter queue not available: {e}")
        status["dead_letter_queue"] = False

    for name, init_func, dependencies in init_order:
        deps_satisfied = all(status.get(dep, False) for dep in dependencies)
        if not deps_satisfied:
            failed_deps = [dep for dep in dependencies if not status.get(dep, False)]
            logger.warning(
                f"[initialize_all_coordinators] {name} skipped - dependencies failed: {failed_deps}"
            )
            status[name] = False
            errors[name] = f"Dependencies not satisfied: {failed_deps}"
            continue

        instance, success, error = _init_with_retry(
            name,
            init_func,
            max_retries=max_retries,
            base_delay=retry_delay,
            logger=logger,
        )
        status[name] = success
        if instance:
            instances[name] = instance
            if dlq and hasattr(instance, "_bus"):
                try:
                    enable_dead_letter_queue(dlq, instance._bus)
                except (AttributeError, ImportError, TypeError):
                    pass
        if error:
            errors[name] = error

    # Wrap handlers with resilience if requested
    if wrap_handlers:
        try:
            from app.coordination.handler_resilience import make_handlers_resilient

            for name, instance in instances.items():
                make_handlers_resilient(instance, name)
            logger.debug("[initialize_all_coordinators] Wrapped handlers with resilience")
        except ImportError:
            logger.debug("[initialize_all_coordinators] handler_resilience not available")

    # Start UnifiedEventCoordinator
    try:
        from app.core.async_context import fire_and_forget

        stats = get_event_coordinator_stats()
        if not stats.get("is_running", False):
            try:
                asyncio.get_running_loop()
                fire_and_forget(
                    start_event_coordinator(),
                    name="event_coordinator_startup",
                )
                status["event_coordinator"] = True
            except RuntimeError:
                status["event_coordinator"] = asyncio.run(start_event_coordinator())
        else:
            status["event_coordinator"] = True
        logger.info("[initialize_all_coordinators] UnifiedEventCoordinator started")
    except Exception as e:
        logger.error(f"[initialize_all_coordinators] UnifiedEventCoordinator failed: {e}")
        errors["event_coordinator"] = str(e)

    # Emit COORDINATOR_INIT_FAILED for any failures
    if errors:
        try:
            import time as _time

            from app.coordination.event_router import DataEvent, DataEventType, get_event_bus
            from app.core.async_context import fire_and_forget

            bus = get_event_bus()
            for name, error in errors.items():
                event = DataEvent(
                    event_type=DataEventType.COORDINATOR_INIT_FAILED,
                    payload={
                        "coordinator_name": name,
                        "error": error,
                        "timestamp": _time.time(),
                    },
                    source="initialize_all_coordinators",
                )
                try:
                    asyncio.get_running_loop()
                    fire_and_forget(
                        bus.publish(event),
                        name=f"emit_coordinator_init_failed_{name}",
                    )
                except RuntimeError:
                    asyncio.run(bus.publish(event))
        except (AttributeError, ImportError, TypeError):
            pass

    # Log summary
    wired_count = sum(1 for k, v in status.items() if v and not k.startswith("_"))
    total_count = len([k for k in status if not k.startswith("_")])

    if wired_count == total_count:
        logger.info(
            f"[initialize_all_coordinators] All {total_count} orchestrators/coordinators initialized"
        )
    else:
        logger.warning(
            f"[initialize_all_coordinators] Initialized {wired_count}/{total_count} "
            f"orchestrators/coordinators. Failed: {list(errors.keys())}"
        )

    status["_errors"] = errors
    status["_instances"] = list(instances.keys())

    return status


def get_all_coordinator_status() -> dict:
    """Get unified status from all orchestrators and coordinators.

    Returns:
        Dict with status from each orchestrator
    """
    return {
        "selfplay": get_selfplay_orchestrator().get_status(),
        "pipeline": get_pipeline_orchestrator().get_status(),
        "task_lifecycle": get_task_lifecycle_coordinator().get_status(),
        "optimization": get_optimization_coordinator().get_status(),
        "metrics": get_metrics_orchestrator().get_status(),
        "resources": get_resource_coordinator().get_status(),
        "cache": get_cache_orchestrator().get_status(),
        "event_coordinator": get_event_coordinator_stats(),
    }


def get_system_health() -> dict:
    """Get aggregated system health from all coordinators (December 2025).

    Returns:
        Dict with health information including overall_health score,
        status string, per-coordinator health, issues list, and handler_health.
    """
    import time as _time

    issues = []
    coordinator_health = {}
    total_score = 0.0
    coordinator_count = 0

    def _get_health_score(name: str, status: dict) -> float:
        """Calculate health score for a coordinator."""
        score = 1.0

        if not status.get("subscribed", True):
            score -= 0.3
            issues.append(f"{name}: not subscribed to events")

        if status.get("paused", False):
            score -= 0.2
            issues.append(f"{name}: paused ({status.get('pause_reason', 'unknown')})")

        if status.get("resource_constraints"):
            for constraint_type, constraint in status.get("resource_constraints", {}).items():
                if isinstance(constraint, dict) and constraint.get("severity") == "critical":
                    score -= 0.2
                    issues.append(f"{name}: critical {constraint_type} constraint")

        if status.get("backpressure_active"):
            score -= 0.1
            issues.append(f"{name}: backpressure active")

        if status.get("plateaus"):
            score -= 0.1 * min(len(status["plateaus"]), 3)
            for metric in status["plateaus"][:3]:
                issues.append(f"{name}: plateau detected in {metric}")

        if status.get("regressions"):
            score -= 0.2 * min(len(status["regressions"]), 2)
            for metric in status["regressions"][:2]:
                issues.append(f"{name}: regression detected in {metric}")

        if status.get("orphaned", 0) > 0:
            orphan_count = status["orphaned"]
            score -= 0.1 * min(orphan_count, 5)
            issues.append(f"{name}: {orphan_count} orphaned tasks")

        if status.get("failed_tasks", 0) > 10:
            score -= 0.1
            issues.append(f"{name}: high failure count ({status['failed_tasks']})")

        return max(0.0, score)

    coordinators = [
        ("selfplay", get_selfplay_orchestrator),
        ("pipeline", get_pipeline_orchestrator),
        ("task_lifecycle", get_task_lifecycle_coordinator),
        ("optimization", get_optimization_coordinator),
        ("metrics", get_metrics_orchestrator),
        ("resources", get_resource_coordinator),
        ("cache", get_cache_orchestrator),
    ]

    for name, getter in coordinators:
        try:
            status = getter().get_status()
            coordinator_health[name] = _get_health_score(name, status)
            coordinator_count += 1
            total_score += coordinator_health[name]
        except Exception as e:
            coordinator_health[name] = 0.0
            issues.append(f"{name}: failed to get status ({e})")

    overall_health = total_score / coordinator_count if coordinator_count > 0 else 0.0

    if overall_health >= 0.9:
        status_str = "healthy"
    elif overall_health >= 0.7:
        status_str = "degraded"
    else:
        status_str = "unhealthy"

    handler_health = {}
    try:
        from app.coordination.handler_resilience import get_all_handler_metrics

        all_metrics = get_all_handler_metrics()
        total_invocations = sum(m.invocation_count for m in all_metrics.values())
        total_failures = sum(m.failure_count for m in all_metrics.values())
        total_timeouts = sum(m.timeout_count for m in all_metrics.values())

        handler_health = {
            "total_handlers": len(all_metrics),
            "total_invocations": total_invocations,
            "total_failures": total_failures,
            "total_timeouts": total_timeouts,
            "success_rate": (
                (total_invocations - total_failures - total_timeouts) / total_invocations
                if total_invocations > 0
                else 1.0
            ),
            "unhealthy_handlers": [
                name for name, m in all_metrics.items() if m.consecutive_failures >= 3
            ],
        }

        if handler_health["unhealthy_handlers"]:
            for handler in handler_health["unhealthy_handlers"]:
                issues.append(f"handler: {handler} has consecutive failures")
    except (AttributeError, ImportError, KeyError):
        pass

    return {
        "overall_health": round(overall_health, 3),
        "status": status_str,
        "coordinators": coordinator_health,
        "issues": issues[:20],
        "handler_health": handler_health,
        "timestamp": _time.time(),
    }


async def shutdown_all_coordinators(
    timeout_seconds: float = 30.0,
    emit_events: bool = True,
) -> dict:
    """Gracefully shutdown all coordinators (December 2025).

    Args:
        timeout_seconds: Maximum time to wait for graceful shutdown
        emit_events: Whether to emit shutdown events

    Returns:
        Dict with shutdown status for each coordinator
    """
    import asyncio
    import logging
    import time as _time

    logger = logging.getLogger(__name__)
    logger.info("[shutdown_all_coordinators] Starting graceful shutdown...")

    status = {}
    start_time = _time.time()

    if emit_events:
        try:
            from app.coordination.event_emitters import emit_coordinator_shutdown

            coordinators = [
                "optimization",
                "metrics",
                "pipeline",
                "selfplay",
                "cache",
                "resources",
                "task_lifecycle",
            ]
            for coord_name in coordinators:
                with contextlib.suppress(Exception):
                    await emit_coordinator_shutdown(
                        coordinator_name=coord_name,
                        reason="system_shutdown",
                    )
        except ImportError:
            pass

    shutdown_order = [
        ("optimization", get_optimization_coordinator),
        ("metrics", get_metrics_orchestrator),
        ("pipeline", get_pipeline_orchestrator),
        ("selfplay", get_selfplay_orchestrator),
        ("cache", get_cache_orchestrator),
        ("resources", get_resource_coordinator),
        ("task_lifecycle", get_task_lifecycle_coordinator),
    ]

    async def _shutdown_coordinator(name: str, getter) -> tuple:
        """Shutdown a single coordinator with timeout."""
        try:
            coord = getter()

            if hasattr(coord, "shutdown") and asyncio.iscoroutinefunction(coord.shutdown):
                remaining = timeout_seconds - (_time.time() - start_time)
                if remaining > 0:
                    await asyncio.wait_for(coord.shutdown(), timeout=remaining)
                    return (name, True, None)
                return (name, False, "timeout exceeded")

            elif hasattr(coord, "stop") and asyncio.iscoroutinefunction(coord.stop):
                remaining = timeout_seconds - (_time.time() - start_time)
                if remaining > 0:
                    await asyncio.wait_for(coord.stop(), timeout=remaining)
                    return (name, True, None)
                return (name, False, "timeout exceeded")

            return (name, True, "no lifecycle methods")

        except asyncio.TimeoutError:
            return (name, False, "shutdown timed out")
        except Exception as e:
            return (name, False, str(e))

    for name, getter in shutdown_order:
        result = await _shutdown_coordinator(name, getter)
        status[result[0]] = {
            "success": result[1],
            "error": result[2],
        }

        if result[1]:
            logger.info(f"[shutdown_all_coordinators] {name} shutdown complete")
        else:
            logger.warning(f"[shutdown_all_coordinators] {name} shutdown failed: {result[2]}")

    # Cleanup global singletons
    try:
        global _selfplay_orchestrator, _pipeline_orchestrator, _task_lifecycle_coordinator
        global _optimization_coordinator, _metrics_orchestrator, _resource_coordinator
        global _cache_orchestrator, _event_coordinator

        _selfplay_orchestrator = None
        _pipeline_orchestrator = None
        _task_lifecycle_coordinator = None
        _optimization_coordinator = None
        _metrics_orchestrator = None
        _resource_coordinator = None
        _cache_orchestrator = None
        _event_coordinator = None
    except NameError:
        pass

    try:
        from app.coordination.handler_resilience import reset_handler_metrics

        reset_handler_metrics()
    except ImportError:
        pass

    try:
        from app.coordination.coordinator_dependencies import reset_dependency_graph

        reset_dependency_graph()
    except ImportError:
        pass

    total_time = _time.time() - start_time
    success_count = sum(1 for s in status.values() if s["success"])

    logger.info(
        f"[shutdown_all_coordinators] Shutdown complete: {success_count}/{len(status)} "
        f"coordinators in {total_time:.2f}s"
    )

    return {
        "status": status,
        "total_time_seconds": round(total_time, 2),
        "success_count": success_count,
        "total_count": len(status),
    }


# =============================================================================
# Coordinator Heartbeat System (December 2025)
# =============================================================================

_heartbeat_task = None
_heartbeat_running = False


async def _emit_coordinator_heartbeats(interval_seconds: float = 30.0) -> None:
    """Background task to emit heartbeats from all coordinators."""
    import asyncio
    import logging

    logger = logging.getLogger(__name__)

    _heartbeat_running = True
    logger.info(f"[HeartbeatManager] Started with {interval_seconds}s interval")

    while _heartbeat_running:
        try:
            from app.coordination.event_emitters import emit_coordinator_heartbeat

            coordinators = [
                ("selfplay", get_selfplay_orchestrator),
                ("pipeline", get_pipeline_orchestrator),
                ("task_lifecycle", get_task_lifecycle_coordinator),
                ("optimization", get_optimization_coordinator),
                ("metrics", get_metrics_orchestrator),
                ("resources", get_resource_coordinator),
                ("cache", get_cache_orchestrator),
            ]

            for name, getter in coordinators:
                try:
                    coord = getter()
                    status = coord.get_status()

                    health_score = 1.0
                    if not status.get("subscribed", True):
                        health_score = 0.5
                    if status.get("paused", False):
                        health_score = 0.7
                    if status.get("backpressure_active", False):
                        health_score = 0.6

                    await emit_coordinator_heartbeat(
                        coordinator_name=name,
                        health_score=health_score,
                        active_handlers=(
                            status.get("metrics_tracked", 0)
                            if name == "metrics"
                            else status.get("active_tasks", 0)
                        ),
                        events_processed=(
                            status.get("total_invocations", 0)
                            if "total_invocations" in status
                            else 0
                        ),
                    )
                except Exception as e:
                    logger.debug(f"[HeartbeatManager] Failed to emit heartbeat for {name}: {e}")

        except ImportError:
            logger.debug("[HeartbeatManager] event_emitters not available")
        except Exception as e:
            logger.debug(f"[HeartbeatManager] Error in heartbeat loop: {e}")

        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            break

    logger.info("[HeartbeatManager] Stopped")


def start_coordinator_heartbeats(interval_seconds: float = 30.0) -> bool:
    """Start the coordinator heartbeat background task.

    Launches an async task that periodically emits COORDINATOR_HEARTBEAT events.
    These events enable daemon health monitoring, cluster synchronization triggers,
    and leader election participation.

    Args:
        interval_seconds: Time between heartbeat emissions (default: 30.0 seconds)

    Returns:
        True if heartbeat task was started successfully or already running,
        False if no async event loop is available

    Side Effects:
        - Sets global _heartbeat_task reference
        - Starts emitting COORDINATOR_HEARTBEAT events to event router
        - Events include: node_id, uptime, resource_usage, daemon_health summary

    Thread Safety:
        Safe to call multiple times; returns True if already running.
        Call stop_coordinator_heartbeats() to terminate.
    """
    import asyncio

    from app.core.async_context import safe_create_task

    global _heartbeat_task

    if _heartbeat_task is not None and not _heartbeat_task.done():
        return True

    try:
        asyncio.get_running_loop()
        _heartbeat_task = safe_create_task(
            _emit_coordinator_heartbeats(interval_seconds),
            name="coordinator_heartbeat_emitter",
        )
        return True
    except RuntimeError:
        return False


def stop_coordinator_heartbeats() -> None:
    """Stop the coordinator heartbeat background task."""
    global _heartbeat_task, _heartbeat_running

    _heartbeat_running = False

    if _heartbeat_task is not None:
        _heartbeat_task.cancel()
        _heartbeat_task = None


def is_heartbeat_running() -> bool:
    """Check if heartbeat manager is running."""
    return _heartbeat_task is not None and not _heartbeat_task.done()


# =============================================================================
# Combined __all__ from all submodules
# =============================================================================
__all__ = (
    _core_all
    + _sync_all
    + _daemon_all
    + _events_all
    + _orchestrators_all
    + _utils_all
    + [
        # Consolidated modules
        "core_utils",
        "core_events",
        # Functions defined in this file
        "get_all_coordinator_status",
        "get_system_health",
        "initialize_all_coordinators",
        "is_heartbeat_running",
        "shutdown_all_coordinators",
        "start_coordinator_heartbeats",
        "stop_coordinator_heartbeats",
    ]
)
