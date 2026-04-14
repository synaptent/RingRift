"""Bootstrap and lifecycle helpers for coordination package consumers."""

from __future__ import annotations

import asyncio
import logging
import time as _time

from app.utils.retry import RetryConfig

__all__ = [
    "initialize_all_coordinators",
    "shutdown_all_coordinators",
    "start_coordinator_heartbeats",
    "stop_coordinator_heartbeats",
    "is_heartbeat_running",
]

_heartbeat_task = None
_heartbeat_running = False


def _init_with_retry(
    name: str,
    init_func,
    max_retries: int = 3,
    base_delay: float = 0.5,
    logger=None,
) -> tuple:
    """Initialize a coordinator with retry logic."""

    last_error = None
    retry_config = RetryConfig(max_attempts=max_retries, base_delay=base_delay, max_delay=8.0)

    for attempt in retry_config.attempts():
        try:
            instance, subscribed = init_func()

            if not subscribed:
                raise RuntimeError(f"{name} failed to subscribe to events")

            if logger:
                if not attempt.is_first:
                    logger.info(f"[init_with_retry] {name} succeeded on attempt {attempt.number}")
                else:
                    logger.info(f"[initialize_all_coordinators] {name} wired")

            return (instance, True, None)

        except Exception as exc:
            last_error = str(exc)
            if logger:
                logger.warning(
                    f"[init_with_retry] {name} attempt {attempt.number}/{retry_config.max_attempts} failed: {exc}"
                )

            if attempt.should_retry:
                attempt.wait()

    if logger:
        logger.error(
            f"[initialize_all_coordinators] {name} failed after {retry_config.max_attempts} attempts"
        )

    return (None, False, last_error)


def initialize_all_coordinators(
    auto_trigger_pipeline: bool = False,
    heartbeat_threshold: float = 60.0,
    max_retries: int = 3,
    retry_delay: float = 0.5,
    wrap_handlers: bool = True,
) -> dict:
    """Initialize all orchestrators and coordinators with event wiring."""

    logger = logging.getLogger(__name__)

    from app.coordination.cache_coordination_orchestrator import wire_cache_events
    from app.coordination.data_pipeline_orchestrator import wire_pipeline_events
    from app.coordination.event_router import (
        DataEvent,
        DataEventType,
        get_coordinator_stats as get_event_coordinator_stats,
        get_event_bus,
        start_coordinator as start_event_coordinator,
    )
    from app.coordination.metrics_analysis_orchestrator import wire_metrics_events
    from app.coordination.optimization_coordinator import wire_optimization_events
    from app.coordination.resource_monitoring_coordinator import wire_resource_events
    from app.coordination.selfplay_orchestrator import wire_selfplay_events
    from app.coordination.task_lifecycle_coordinator import wire_task_events

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

    init_order = [
        ("task_lifecycle", init_task_lifecycle, []),
        ("resources", init_resources, []),
        ("cache", init_cache, []),
        ("selfplay", init_selfplay, ["task_lifecycle"]),
        ("pipeline", init_pipeline, ["task_lifecycle", "selfplay"]),
        ("optimization", init_optimization, ["pipeline"]),
        ("metrics", init_metrics, ["task_lifecycle"]),
    ]

    dlq = None
    try:
        from app.coordination.dead_letter_queue import enable_dead_letter_queue, get_dead_letter_queue

        dlq = get_dead_letter_queue()
        status["dead_letter_queue"] = True
        instances["dead_letter_queue"] = dlq
        logger.info("[initialize_all_coordinators] Dead letter queue initialized")
    except Exception as exc:
        logger.warning(f"[initialize_all_coordinators] Dead letter queue not available: {exc}")
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

    if wrap_handlers:
        try:
            from app.coordination.handler_resilience import make_handlers_resilient

            for name, instance in instances.items():
                make_handlers_resilient(instance, name)
            logger.debug("[initialize_all_coordinators] Wrapped handlers with resilience")
        except ImportError:
            logger.debug("[initialize_all_coordinators] handler_resilience not available")

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
    except Exception as exc:
        logger.error(f"[initialize_all_coordinators] UnifiedEventCoordinator failed: {exc}")
        errors["event_coordinator"] = str(exc)

    if errors:
        try:
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

    wired_count = sum(1 for key, value in status.items() if value and not key.startswith("_"))
    total_count = len([key for key in status if not key.startswith("_")])

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


async def shutdown_all_coordinators(
    timeout_seconds: float = 30.0,
    emit_events: bool = True,
) -> dict:
    """Gracefully shutdown all coordinators."""

    logger = logging.getLogger(__name__)
    logger.info("[shutdown_all_coordinators] Starting graceful shutdown...")

    status = {}
    start_time = _time.time()

    if emit_events:
        from app.coordination.event_emission_helpers import safe_emit_event_async

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
            await safe_emit_event_async(
                "COORDINATOR_SHUTDOWN",
                {"coordinator_name": coord_name, "reason": "system_shutdown"},
                context="shutdown_all_coordinators",
            )

    from app.coordination.cache_coordination_orchestrator import get_cache_orchestrator
    from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator
    from app.coordination.metrics_analysis_orchestrator import get_metrics_orchestrator
    from app.coordination.optimization_coordinator import get_optimization_coordinator
    from app.coordination.resource_monitoring_coordinator import get_resource_coordinator
    from app.coordination.selfplay_orchestrator import get_selfplay_orchestrator
    from app.coordination.task_lifecycle_coordinator import get_task_lifecycle_coordinator

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
        try:
            coord = getter()

            if hasattr(coord, "shutdown") and asyncio.iscoroutinefunction(coord.shutdown):
                remaining = timeout_seconds - (_time.time() - start_time)
                if remaining > 0:
                    await asyncio.wait_for(coord.shutdown(), timeout=remaining)
                    return (name, True, None)
                return (name, False, "timeout exceeded")

            if hasattr(coord, "stop") and asyncio.iscoroutinefunction(coord.stop):
                remaining = timeout_seconds - (_time.time() - start_time)
                if remaining > 0:
                    await asyncio.wait_for(coord.stop(), timeout=remaining)
                    return (name, True, None)
                return (name, False, "timeout exceeded")

            return (name, True, "no lifecycle methods")

        except asyncio.TimeoutError:
            return (name, False, "shutdown timed out")
        except Exception as exc:
            return (name, False, str(exc))

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
    success_count = sum(1 for item in status.values() if item["success"])

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


async def _emit_coordinator_heartbeats(interval_seconds: float = 30.0) -> None:
    """Background task to emit heartbeats from all coordinators."""

    global _heartbeat_running

    logger = logging.getLogger(__name__)

    from app.coordination.cache_coordination_orchestrator import get_cache_orchestrator
    from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator
    from app.coordination.metrics_analysis_orchestrator import get_metrics_orchestrator
    from app.coordination.optimization_coordinator import get_optimization_coordinator
    from app.coordination.resource_monitoring_coordinator import get_resource_coordinator
    from app.coordination.selfplay_orchestrator import get_selfplay_orchestrator
    from app.coordination.task_lifecycle_coordinator import get_task_lifecycle_coordinator

    _heartbeat_running = True
    logger.info(f"[HeartbeatManager] Started with {interval_seconds}s interval")

    while _heartbeat_running:
        from app.coordination.event_emission_helpers import safe_emit_event_async

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

                await safe_emit_event_async(
                    "COORDINATOR_HEARTBEAT",
                    {
                        "coordinator_name": name,
                        "health_score": health_score,
                        "active_handlers": (
                            status.get("metrics_tracked", 0)
                            if name == "metrics"
                            else status.get("active_tasks", 0)
                        ),
                        "events_processed": (
                            status.get("total_invocations", 0)
                            if "total_invocations" in status
                            else 0
                        ),
                    },
                    context="heartbeat_manager",
                )
            except Exception as exc:
                logger.debug(f"[HeartbeatManager] Failed to emit heartbeat for {name}: {exc}")

        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            break

    logger.info("[HeartbeatManager] Stopped")


def start_coordinator_heartbeats(interval_seconds: float = 30.0) -> bool:
    """Start the coordinator heartbeat background task."""

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
