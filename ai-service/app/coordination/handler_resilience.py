"""Handler Resilience - Exception boundaries and timeout management (December 2025).

This module provides decorators and wrappers to make event handlers resilient:
- Exception boundaries prevent cascading failures
- Timeouts prevent hung handlers
- Automatic emission of HANDLER_FAILED and HANDLER_TIMEOUT events
- Metrics tracking for handler health

Usage:
    from app.coordination.handler_resilience import (
        resilient_handler,
        ResilientHandlerConfig,
    )

    class MyOrchestrator:
        def subscribe_to_events(self):
            bus.subscribe(
                DataEventType.TRAINING_COMPLETED,
                resilient_handler(self._on_some_event, coordinator="MyOrchestrator")
            )

        async def _on_some_event(self, event):
            # Handler code - exceptions caught, timeouts enforced
            pass
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# Type for async handlers
T = TypeVar("T")


@dataclass
class ResilientHandlerConfig:
    """Configuration for resilient handler wrapper."""

    timeout_seconds: float = 30.0  # Default 30s timeout
    emit_failure_events: bool = True  # Emit HANDLER_FAILED on exception
    emit_timeout_events: bool = True  # Emit HANDLER_TIMEOUT on timeout
    log_exceptions: bool = True  # Log exceptions
    log_timeouts: bool = True  # Log timeouts
    retry_on_timeout: bool = False  # Retry once on timeout
    max_consecutive_failures: int = 5  # Threshold for health degradation


@dataclass
class HandlerMetrics:
    """Metrics for a single handler."""

    handler_name: str
    coordinator: str
    invocation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    total_duration_ms: float = 0.0
    last_failure_time: float = 0.0
    last_failure_error: str = ""
    consecutive_failures: int = 0

    @property
    def success_rate(self) -> float:
        if self.invocation_count == 0:
            return 1.0
        return self.success_count / self.invocation_count

    @property
    def avg_duration_ms(self) -> float:
        if self.success_count == 0:
            return 0.0
        return self.total_duration_ms / self.success_count

    def record_success(self, duration_ms: float) -> None:
        self.invocation_count += 1
        self.success_count += 1
        self.total_duration_ms += duration_ms
        self.consecutive_failures = 0

    def record_failure(self, error: str) -> None:
        self.invocation_count += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.last_failure_error = error
        self.consecutive_failures += 1

    def record_timeout(self) -> None:
        self.invocation_count += 1
        self.timeout_count += 1
        self.consecutive_failures += 1


# Global registry of handler metrics
_handler_metrics: dict[str, HandlerMetrics] = {}


def get_handler_metrics(handler_name: str, coordinator: str = "") -> HandlerMetrics:
    """Get or create metrics for a handler."""
    key = f"{coordinator}:{handler_name}"
    if key not in _handler_metrics:
        _handler_metrics[key] = HandlerMetrics(
            handler_name=handler_name,
            coordinator=coordinator,
        )
    return _handler_metrics[key]


def get_all_handler_metrics() -> dict[str, HandlerMetrics]:
    """Get all handler metrics."""
    return dict(_handler_metrics)


def reset_handler_metrics() -> None:
    """Reset all handler metrics (for testing)."""
    _handler_metrics.clear()


async def _emit_failure_event(
    handler_name: str,
    event_type: str,
    error: str,
    coordinator: str,
) -> None:
    """Emit HANDLER_FAILED event."""
    try:
        from app.coordination.event_emitters import emit_handler_failed

        await emit_handler_failed(
            handler_name=handler_name,
            event_type=event_type,
            error=error,
            coordinator=coordinator,
        )
    except Exception as e:
        # Phase 12: Elevated to warning - event emission failures indicate monitoring problems
        logger.warning(f"Failed to emit handler_failed event: {e}")


async def _emit_timeout_event(
    handler_name: str,
    event_type: str,
    timeout_seconds: float,
    coordinator: str,
) -> None:
    """Emit HANDLER_TIMEOUT event."""
    try:
        from app.coordination.event_emitters import emit_handler_timeout

        await emit_handler_timeout(
            handler_name=handler_name,
            event_type=event_type,
            timeout_seconds=timeout_seconds,
            coordinator=coordinator,
        )
    except Exception as e:
        # Phase 12: Elevated to warning - event emission failures indicate monitoring problems
        logger.warning(f"Failed to emit handler_timeout event: {e}")


async def _emit_health_degraded(
    coordinator: str,
    handler_name: str,
    consecutive_failures: int,
) -> None:
    """Emit COORDINATOR_HEALTH_DEGRADED when failure threshold reached."""
    try:
        from app.coordination.event_emitters import emit_coordinator_health_degraded

        await emit_coordinator_health_degraded(
            coordinator_name=coordinator,
            reason=f"Handler {handler_name} has {consecutive_failures} consecutive failures",
            health_score=0.5,
            issues=[f"{handler_name}: {consecutive_failures} consecutive failures"],
        )
    except Exception as e:
        # Phase 12: Elevated to warning - event emission failures indicate monitoring problems
        logger.warning(f"Failed to emit health_degraded event: {e}")


def resilient_handler(
    handler: Callable,
    coordinator: str = "",
    config: ResilientHandlerConfig | None = None,
) -> Callable:
    """Wrap an async event handler with exception boundary and timeout.

    Args:
        handler: The async handler function to wrap
        coordinator: Name of the coordinator (for metrics/events)
        config: Handler configuration

    Returns:
        Wrapped handler that catches exceptions and enforces timeouts
    """
    config = config or ResilientHandlerConfig()
    handler_name = handler.__name__

    @functools.wraps(handler)
    async def wrapper(event) -> None:
        metrics = get_handler_metrics(handler_name, coordinator)
        start_time = time.time()
        event_type_str = ""

        try:
            # Extract event type for logging
            if hasattr(event, "event_type"):
                event_type_str = str(event.event_type.value)
            elif hasattr(event, "event"):
                event_type_str = str(event.event.value)

            # Execute with timeout (compatible with Python 3.10+)
            await asyncio.wait_for(handler(event), timeout=config.timeout_seconds)

            # Record success
            duration_ms = (time.time() - start_time) * 1000
            metrics.record_success(duration_ms)

        except asyncio.TimeoutError:
            # Handle timeout
            metrics.record_timeout()

            if config.log_timeouts:
                logger.warning(
                    f"[{coordinator}] Handler {handler_name} timed out after "
                    f"{config.timeout_seconds}s on {event_type_str}"
                )

            if config.emit_timeout_events:
                await _emit_timeout_event(
                    handler_name, event_type_str, config.timeout_seconds, coordinator
                )

            # Check for health degradation
            if metrics.consecutive_failures >= config.max_consecutive_failures:
                await _emit_health_degraded(
                    coordinator, handler_name, metrics.consecutive_failures
                )

        except Exception as e:
            # Handle exception
            error_msg = f"{type(e).__name__}: {e}"
            metrics.record_failure(error_msg)

            if config.log_exceptions:
                logger.error(
                    f"[{coordinator}] Handler {handler_name} failed on {event_type_str}: {e}"
                )
                logger.debug(traceback.format_exc())

            if config.emit_failure_events:
                await _emit_failure_event(
                    handler_name, event_type_str, error_msg, coordinator
                )

            # Check for health degradation
            if metrics.consecutive_failures >= config.max_consecutive_failures:
                await _emit_health_degraded(
                    coordinator, handler_name, metrics.consecutive_failures
                )

    return wrapper


def make_handlers_resilient(
    coordinator_instance: Any,
    coordinator_name: str,
    handler_prefix: str = "_on_",
    config: ResilientHandlerConfig | None = None,
) -> dict[str, Callable]:
    """Make all handler methods of a coordinator resilient.

    Finds all methods starting with handler_prefix and wraps them.

    Args:
        coordinator_instance: The coordinator instance
        coordinator_name: Name for metrics/events
        handler_prefix: Prefix for handler methods (default "_on_")
        config: Handler configuration

    Returns:
        Dict of wrapped handlers: {original_name: wrapped_handler}
    """
    wrapped = {}

    for attr_name in dir(coordinator_instance):
        if attr_name.startswith(handler_prefix):
            original = getattr(coordinator_instance, attr_name)
            if asyncio.iscoroutinefunction(original):
                wrapped_handler = resilient_handler(
                    original,
                    coordinator=coordinator_name,
                    config=config,
                )
                wrapped[attr_name] = wrapped_handler
                # Replace on instance
                setattr(coordinator_instance, attr_name, wrapped_handler)

    return wrapped


class ResilientCoordinatorMixin:
    """Mixin that adds resilience features to coordinators.

    Usage:
        class MyOrchestrator(ResilientCoordinatorMixin):
            _coordinator_name = "MyOrchestrator"

            def subscribe_to_events(self):
                self._wrap_handlers()  # Wrap all _on_* methods
                # Then subscribe...
    """

    _coordinator_name: str = "unknown"
    _handler_config: ResilientHandlerConfig | None = None
    _events_processed: int = 0
    _handler_failures: int = 0

    def _wrap_handlers(self) -> None:
        """Wrap all event handlers with resilience."""
        make_handlers_resilient(
            self,
            self._coordinator_name,
            config=self._handler_config,
        )

    def _get_handler_health(self) -> dict[str, Any]:
        """Get health summary for this coordinator's handlers."""
        metrics = get_all_handler_metrics()
        coordinator_metrics = {
            k: v for k, v in metrics.items() if k.startswith(f"{self._coordinator_name}:")
        }

        total_invocations = sum(m.invocation_count for m in coordinator_metrics.values())
        total_failures = sum(m.failure_count for m in coordinator_metrics.values())
        total_timeouts = sum(m.timeout_count for m in coordinator_metrics.values())

        return {
            "coordinator": self._coordinator_name,
            "handler_count": len(coordinator_metrics),
            "total_invocations": total_invocations,
            "total_failures": total_failures,
            "total_timeouts": total_timeouts,
            "success_rate": (
                (total_invocations - total_failures - total_timeouts) / total_invocations
                if total_invocations > 0
                else 1.0
            ),
            "handlers": {
                name.split(":", 1)[1]: {
                    "invocations": m.invocation_count,
                    "failures": m.failure_count,
                    "timeouts": m.timeout_count,
                    "avg_duration_ms": round(m.avg_duration_ms, 2),
                    "consecutive_failures": m.consecutive_failures,
                }
                for name, m in coordinator_metrics.items()
            },
        }

    def health_check(self) -> "HealthCheckResult":
        """Check handler resilience health for daemon monitoring.

        December 2025 Phase 4: Added for unified daemon health monitoring.

        Returns:
            HealthCheckResult with handler failure metrics.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        try:
            health_data = self._get_handler_health()
            success_rate = health_data["success_rate"]
            total_failures = health_data["total_failures"]
            total_timeouts = health_data["total_timeouts"]
            handler_count = health_data["handler_count"]

            # Check for degraded handlers (high consecutive failures)
            degraded_handlers = [
                name for name, data in health_data["handlers"].items()
                if data.get("consecutive_failures", 0) >= (
                    self._handler_config.max_consecutive_failures
                    if self._handler_config else 5
                )
            ]

            if degraded_handlers:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"Handlers degraded: {', '.join(degraded_handlers)}",
                    details={
                        "degraded_handlers": degraded_handlers,
                        "success_rate": success_rate,
                        "total_failures": total_failures,
                        "total_timeouts": total_timeouts,
                    },
                )

            if success_rate < 0.8 and health_data["total_invocations"] > 10:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"Low handler success rate: {success_rate:.1%}",
                    details={
                        "success_rate": success_rate,
                        "total_failures": total_failures,
                        "total_timeouts": total_timeouts,
                    },
                )

            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"{self._coordinator_name} handlers healthy: {handler_count} handlers, {success_rate:.1%} success",
                details={
                    "handler_count": handler_count,
                    "success_rate": success_rate,
                    "total_invocations": health_data["total_invocations"],
                },
            )

        except Exception as e:
            logger.error(f"Error checking ResilientCoordinatorMixin health: {e}")
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check error: {e}",
            )


__all__ = [
    "HandlerMetrics",
    "ResilientCoordinatorMixin",
    "ResilientHandlerConfig",
    "get_all_handler_metrics",
    "get_handler_metrics",
    "make_handlers_resilient",
    "reset_handler_metrics",
    "resilient_handler",
]
