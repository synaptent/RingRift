"""
EventBus - Async pub/sub event bus for component coordination.

Extracted from data_events.py for modularity.
"""

import asyncio
import time
from typing import Any, Callable, Coroutine

from .event_types import DataEvent, DataEventType

# Type alias for event callbacks
EventCallback = Callable[[DataEvent], None | Coroutine[Any, Any, None]]

# Global singleton
_event_bus: "EventBus | None" = None


class EventBus:
    """Async event bus for component coordination.

    Supports both sync and async callbacks. Events are delivered
    in order of subscription.

    December 2025: Added subscription registry with warnings for unsubscribed events.
    """

    def __init__(self, max_history: int = 1000, warn_unsubscribed: bool = True):
        self._subscribers: dict[DataEventType, list[EventCallback]] = {}
        self._global_subscribers: list[EventCallback] = []
        self._event_history: list[DataEvent] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()

        # Subscription registry (December 2025)
        self._warn_unsubscribed = warn_unsubscribed
        self._published_event_types: dict[DataEventType, int] = {}  # type -> count
        self._warned_event_types: set = set()  # Types we've already warned about

        # Observability metrics (December 2025)
        self._start_time = time.time()
        self._total_events_published = 0
        self._total_callbacks_invoked = 0
        self._total_callback_errors = 0
        self._callback_latencies: list[float] = []  # Recent latencies in ms
        self._max_latency_samples = 1000
        self._errors_by_type: dict[DataEventType, int] = {}
        self._last_event_time: float = 0.0

    def subscribe(
        self,
        event_type: DataEventType | None,
        callback: EventCallback,
    ) -> None:
        """Subscribe to events.

        Args:
            event_type: Specific event type, or None for all events
            callback: Function to call when event occurs. Can be sync or async.
        """
        if event_type is None:
            self._global_subscribers.append(callback)
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(
        self,
        event_type: DataEventType | None,
        callback: EventCallback,
    ) -> bool:
        """Unsubscribe from events.

        Returns True if callback was found and removed.
        """
        if event_type is None:
            if callback in self._global_subscribers:
                self._global_subscribers.remove(callback)
                return True
        else:
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                return True
        return False

    async def publish(self, event: DataEvent, bridge_cross_process: bool = True) -> None:
        """Publish an event to all subscribers.

        Callbacks are invoked in order. Async callbacks are awaited.
        Errors in callbacks are logged but don't prevent delivery to
        other subscribers.

        Args:
            event: The event to publish
            bridge_cross_process: If True, also bridge to cross-process queue
        """
        # Dec 31, 2025: Defensive handling for bare DataEventType passed instead of DataEvent
        # This fixes AttributeError when callers pass DataEventType enum directly
        if isinstance(event, DataEventType):
            event = DataEvent(event_type=event, payload={}, source="unknown")

        async with self._lock:
            # Store in history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

            # Track published event types (December 2025)
            self._published_event_types[event.event_type] = (
                self._published_event_types.get(event.event_type, 0) + 1
            )

        # Bridge to cross-process queue for multi-daemon coordination
        if bridge_cross_process:
            _bridge_to_cross_process(event)

        # Get all callbacks for this event
        callbacks = list(self._global_subscribers)
        if event.event_type in self._subscribers:
            callbacks.extend(self._subscribers[event.event_type])

        # Warn if no subscribers (December 2025)
        if (
            self._warn_unsubscribed
            and not callbacks
            and event.event_type not in self._warned_event_types
        ):
            self._warned_event_types.add(event.event_type)
            print(
                f"[EventBus] WARNING: Event {event.event_type.value} published "
                f"but has no subscribers. Consider adding a handler."
            )

        # Invoke each callback with latency tracking (December 2025)
        for callback in callbacks:
            self._total_callbacks_invoked += 1
            callback_start = time.time()
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self._total_callback_errors += 1
                self._errors_by_type[event.event_type] = (
                    self._errors_by_type.get(event.event_type, 0) + 1
                )
                print(f"[EventBus] Error in subscriber for {event.event_type.value}: {e}")

                # Capture to dead letter queue if enabled (December 2025)
                if hasattr(self, "_dlq") and self._dlq is not None:
                    try:
                        self._dlq.capture(
                            event_type=event.event_type.value,
                            payload=event.payload,
                            handler_name=getattr(callback, "__name__", "unknown"),
                            error=str(e),
                            source="data_events",
                        )
                    except Exception as dlq_error:
                        print(f"[EventBus] Failed to capture to DLQ: {dlq_error}")
            finally:
                # Track callback latency
                latency_ms = (time.time() - callback_start) * 1000
                self._callback_latencies.append(latency_ms)
                if len(self._callback_latencies) > self._max_latency_samples:
                    self._callback_latencies = self._callback_latencies[-self._max_latency_samples:]

        # Update event metrics
        self._total_events_published += 1
        self._last_event_time = time.time()

    def publish_sync(self, event: DataEvent, bridge_cross_process: bool = True) -> None:
        """Publish an event synchronously (non-async context).

        Creates a new event loop if needed. Use publish() when possible.
        """
        # Bridge immediately in sync context (doesn't need async)
        if bridge_cross_process:
            _bridge_to_cross_process(event)

        try:
            loop = asyncio.get_running_loop()
            # Don't bridge again in the async publish
            loop.create_task(self.publish(event, bridge_cross_process=False))
        except RuntimeError:
            # No running loop - run synchronously
            asyncio.run(self.publish(event, bridge_cross_process=False))

    def emit(
        self,
        event_type: DataEventType,
        payload: dict[str, Any] | None = None,
        source: str = "",
        bridge_cross_process: bool = True,
    ) -> None:
        """Convenience method to emit an event with type and payload.

        This is an alias for publish() that accepts event type and payload
        separately, creating a DataEvent internally. It provides API
        consistency with other event systems.

        December 28, 2025: Added to fix API mismatch where code calls
        bus.emit(DataEventType, payload) but EventBus only had publish(DataEvent).

        Args:
            event_type: The event type to emit
            payload: Event payload dictionary
            source: Component that generated the event
            bridge_cross_process: If True, also bridge to cross-process queue

        Usage:
            bus = get_event_bus()
            bus.emit(DataEventType.TRAINING_COMPLETED, {"config_key": "hex8_2p"})
        """
        event = DataEvent(
            event_type=event_type,
            payload=payload or {},
            source=source,
        )
        # Use sync version since emit() is called from sync contexts
        self.publish_sync(event, bridge_cross_process=bridge_cross_process)

    async def emit_async(
        self,
        event_type: DataEventType,
        payload: dict[str, Any] | None = None,
        source: str = "",
        bridge_cross_process: bool = True,
    ) -> None:
        """Async version of emit() for use in async contexts.

        December 28, 2025: Added for async callers.

        Args:
            event_type: The event type to emit
            payload: Event payload dictionary
            source: Component that generated the event
            bridge_cross_process: If True, also bridge to cross-process queue
        """
        event = DataEvent(
            event_type=event_type,
            payload=payload or {},
            source=source,
        )
        await self.publish(event, bridge_cross_process=bridge_cross_process)

    def get_history(
        self,
        event_type: DataEventType | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[DataEvent]:
        """Get recent events from history.

        Args:
            event_type: Filter by event type (None for all)
            since: Only events after this timestamp
            limit: Maximum number of events to return
        """
        events = self._event_history

        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        if since is not None:
            events = [e for e in events if e.timestamp > since]

        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history = []

    # =========================================================================
    # Subscription Registry (December 2025)
    # =========================================================================

    def get_subscribed_event_types(self) -> list[DataEventType]:
        """Get list of event types that have at least one subscriber."""
        return list(self._subscribers.keys())

    def get_unsubscribed_published_types(self) -> list[DataEventType]:
        """Get event types that have been published but have no subscribers.

        This is useful for debugging to find events that are being published
        but nobody is listening to.
        """
        unsubscribed = []
        for event_type in self._published_event_types:
            has_specific = event_type in self._subscribers and self._subscribers[event_type]
            has_global = bool(self._global_subscribers)
            if not has_specific and not has_global:
                unsubscribed.append(event_type)
        return unsubscribed

    def get_subscriber_count(self, event_type: DataEventType) -> int:
        """Get the number of subscribers for a specific event type."""
        count = len(self._global_subscribers)
        if event_type in self._subscribers:
            count += len(self._subscribers[event_type])
        return count

    def get_subscription_stats(self) -> dict[str, Any]:
        """Get statistics about subscriptions and published events.

        Returns:
            Dict with subscription statistics
        """
        return {
            "subscribed_types": [t.value for t in self.get_subscribed_event_types()],
            "published_types": {t.value: c for t, c in self._published_event_types.items()},
            "unsubscribed_published": [t.value for t in self.get_unsubscribed_published_types()],
            "global_subscribers": len(self._global_subscribers),
            "warned_types": [t.value for t in self._warned_event_types],
            "total_events_published": sum(self._published_event_types.values()),
        }

    # =========================================================================
    # Observability Metrics (December 2025)
    # =========================================================================

    def get_observability_metrics(self) -> dict[str, Any]:
        """Get comprehensive observability metrics for the event bus.

        Returns:
            Dict with metrics including:
            - Throughput (events/callbacks per second)
            - Latency statistics (mean, p50, p95, p99)
            - Error rates
            - Uptime and activity
        """
        import statistics as stats

        uptime = time.time() - self._start_time
        events_per_second = self._total_events_published / uptime if uptime > 0 else 0.0

        # Calculate latency percentiles
        latency_stats = {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
        if self._callback_latencies:
            sorted_latencies = sorted(self._callback_latencies)
            n = len(sorted_latencies)
            latency_stats = {
                "mean_ms": round(stats.mean(sorted_latencies), 2),
                "p50_ms": round(sorted_latencies[int(n * 0.5)], 2),
                "p95_ms": round(sorted_latencies[int(n * 0.95)], 2),
                "p99_ms": round(sorted_latencies[int(n * 0.99)], 2),
                "max_ms": round(max(sorted_latencies), 2),
            }

        # Error rate
        error_rate = (
            self._total_callback_errors / self._total_callbacks_invoked
            if self._total_callbacks_invoked > 0 else 0.0
        )

        return {
            "uptime_seconds": round(uptime, 1),
            "total_events_published": self._total_events_published,
            "total_callbacks_invoked": self._total_callbacks_invoked,
            "total_callback_errors": self._total_callback_errors,
            "error_rate": round(error_rate, 4),
            "events_per_second": round(events_per_second, 2),
            "latency": latency_stats,
            "errors_by_type": {t.value: c for t, c in self._errors_by_type.items()},
            "last_event_time": self._last_event_time,
            "seconds_since_last_event": round(time.time() - self._last_event_time, 1) if self._last_event_time else None,
            "subscriber_counts": {
                t.value: len(subs) for t, subs in self._subscribers.items()
            },
            "history_size": len(self._event_history),
        }

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of the event bus.

        Returns:
            Dict with health indicators
        """
        metrics = self.get_observability_metrics()

        # Determine health score
        health_score = 1.0
        issues = []

        # Check error rate
        if metrics["error_rate"] > 0.1:
            health_score -= 0.3
            issues.append(f"High error rate: {metrics['error_rate']:.1%}")
        elif metrics["error_rate"] > 0.05:
            health_score -= 0.1
            issues.append(f"Elevated error rate: {metrics['error_rate']:.1%}")

        # Check latency
        if metrics["latency"]["p95_ms"] > 1000:
            health_score -= 0.2
            issues.append(f"High p95 latency: {metrics['latency']['p95_ms']}ms")
        elif metrics["latency"]["p95_ms"] > 500:
            health_score -= 0.1
            issues.append(f"Elevated p95 latency: {metrics['latency']['p95_ms']}ms")

        # Check for stale bus
        if metrics["seconds_since_last_event"] and metrics["seconds_since_last_event"] > 300:
            health_score -= 0.1
            issues.append(f"No events for {metrics['seconds_since_last_event']}s")

        # Determine status
        if health_score >= 0.9:
            status = "healthy"
        elif health_score >= 0.7:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "health_score": round(health_score, 2),
            "issues": issues,
            "metrics_summary": {
                "events_published": metrics["total_events_published"],
                "error_rate": metrics["error_rate"],
                "p95_latency_ms": metrics["latency"]["p95_ms"],
                "events_per_second": metrics["events_per_second"],
            },
        }

    def reset_metrics(self) -> None:
        """Reset observability metrics (for testing)."""
        self._total_events_published = 0
        self._total_callbacks_invoked = 0
        self._total_callback_errors = 0
        self._callback_latencies.clear()
        self._errors_by_type.clear()
        self._last_event_time = 0.0
        self._start_time = time.time()

    def has_subscribers(self, event_type: DataEventType) -> bool:
        """Check if an event type has any subscribers (specific or global)."""
        return self.get_subscriber_count(event_type) > 0

    def health_check(self) -> "HealthCheckResult":
        """Return health status for daemon monitoring.

        December 2025: Added for DaemonManager integration.

        Health criteria:
        - Error rate < 10% (healthy) or < 20% (degraded)
        - p95 latency < 1000ms (healthy) or < 2000ms (degraded)
        - Should have processed at least some events

        Returns:
            HealthCheckResult compatible with DaemonManager
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        # Get existing health status
        health_status = self.get_health_status()
        metrics = self.get_observability_metrics()

        # Map status to HealthCheckResult
        is_healthy = health_status["status"] == "healthy"
        is_degraded = health_status["status"] == "degraded"

        # Build detailed status
        details = {
            "health_score": health_status["health_score"],
            "total_events_published": metrics["total_events_published"],
            "total_callbacks_invoked": metrics["total_callbacks_invoked"],
            "total_callback_errors": metrics["total_callback_errors"],
            "error_rate": metrics["error_rate"],
            "p95_latency_ms": metrics["latency"]["p95_ms"],
            "events_per_second": metrics["events_per_second"],
            "subscriber_count": sum(
                len(subs) for subs in self._subscribers.values()
            ) + len(self._global_subscribers),
            "subscribed_types": len(self._subscribers),
            "uptime_seconds": metrics["uptime_seconds"],
            "issues": health_status.get("issues", []),
        }

        if is_healthy:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"EventBus healthy: {metrics['total_events_published']} events, {metrics['error_rate']:.1%} error rate",
                details=details,
            )
        elif is_degraded:
            return HealthCheckResult(
                healthy=True,  # Degraded is still operational
                status=CoordinatorStatus.DEGRADED,
                message=f"EventBus degraded: {'; '.join(health_status.get('issues', []))}",
                details=details,
            )
        else:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.UNHEALTHY,
                message=f"EventBus unhealthy: {'; '.join(health_status.get('issues', []))}",
                details=details,
            )


def get_event_bus() -> EventBus:
    """Get the global event bus singleton.

    .. deprecated:: December 2025
        Use :func:`app.coordination.event_router.get_router` instead for unified
        event routing across all event systems.
    """
    import warnings
    warnings.warn(
        "get_event_bus() from data_events is deprecated. "
        "Use get_router() from app.coordination.event_router instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _event_bus
    _event_bus = None


# Cross-process bridging configuration
# Events that should be propagated to cross-process queue for multi-daemon coordination
CROSS_PROCESS_EVENT_TYPES = {
    # Success events - coordination across processes
    DataEventType.MODEL_PROMOTED,
    DataEventType.TIER_PROMOTION,  # Difficulty ladder progression
    DataEventType.TRAINING_STARTED,
    DataEventType.TRAINING_COMPLETED,
    DataEventType.EVALUATION_COMPLETED,
    DataEventType.CURRICULUM_REBALANCED,
    DataEventType.CURRICULUM_ADVANCED,  # Curriculum tier progression
    DataEventType.CURRICULUM_ROLLBACK_COMPLETED,  # Sprint 16.1: Rollback confirmation
    DataEventType.SELFPLAY_TARGET_UPDATED,  # Dynamic selfplay scaling
    DataEventType.ELO_SIGNIFICANT_CHANGE,
    DataEventType.P2P_MODEL_SYNCED,
    DataEventType.PLATEAU_DETECTED,
    DataEventType.DATA_SYNC_COMPLETED,
    DataEventType.HYPERPARAMETER_UPDATED,
    DataEventType.GAME_SYNCED,  # Ephemeral sync events
    DataEventType.DATA_STALE,  # Training data freshness
    # Failure events - important for distributed health awareness
    DataEventType.TRAINING_FAILED,
    DataEventType.EVALUATION_FAILED,
    DataEventType.PROMOTION_FAILED,
    DataEventType.DATA_SYNC_FAILED,
    # Host/cluster events - topology awareness
    DataEventType.HOST_ONLINE,
    DataEventType.HOST_OFFLINE,
    DataEventType.DAEMON_STARTED,
    DataEventType.DAEMON_STOPPED,
    DataEventType.DAEMON_STATUS_CHANGED,  # Watchdog alerts for daemon health
    DataEventType.DAEMON_PERMANENTLY_FAILED,  # Dec 2025: Exceeded hourly restart limit
    DataEventType.DAEMON_CRASH_LOOP_DETECTED,  # Dec 2025: Early warning for crash loops
    # Trigger events - distributed optimization
    DataEventType.CMAES_TRIGGERED,
    DataEventType.NAS_TRIGGERED,
    DataEventType.TRAINING_THRESHOLD_REACHED,
    DataEventType.CACHE_INVALIDATED,
    # Regression events - unified detection across all processes
    DataEventType.REGRESSION_DETECTED,
    DataEventType.REGRESSION_SEVERE,
    DataEventType.REGRESSION_CRITICAL,
    DataEventType.REGRESSION_CLEARED,
}


def _bridge_to_cross_process(event: DataEvent) -> None:
    """Bridge event to cross-process queue if it's a cross-process event type.

    This allows events published to the in-memory EventBus to also be
    visible to other daemon processes via the SQLite-backed queue.
    """
    if event.event_type not in CROSS_PROCESS_EVENT_TYPES:
        return

    try:
        # Phase 9 (Dec 2025): Import directly from cross_process_events to avoid
        # circular import with event_router (which imports from this module)
        from app.coordination.cross_process_events import bridge_to_cross_process
        bridge_to_cross_process(event.event_type.value, event.payload, event.source)
    except Exception as e:
        # Don't fail the main event if cross-process bridging fails
        print(f"[EventBus] Cross-process bridge failed: {e}")


__all__ = [
    "EventBus",
    "EventCallback",
    "get_event_bus",
    "reset_event_bus",
    "CROSS_PROCESS_EVENT_TYPES",
]
