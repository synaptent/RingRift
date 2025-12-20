"""Data Pipeline Event System.

This module provides an event bus for coordinating components of the
AI self-improvement loop. Events allow loose coupling between:
- Data collection
- Training triggers
- Evaluation
- Model promotion
- Curriculum rebalancing

Usage:
    from app.distributed.data_events import DataEventType, DataEvent, get_event_bus

    # Subscribe to events
    bus = get_event_bus()
    bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, handle_new_games)

    # Publish events
    await bus.publish(DataEvent(
        event_type=DataEventType.NEW_GAMES_AVAILABLE,
        payload={"host": "gh200-a", "new_games": 500}
    ))
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

__all__ = [
    "DataEvent",
    # Event types
    "DataEventType",
    # Event bus
    "EventBus",
    "get_event_bus",
    "reset_event_bus",
]

# Global singleton instance
_event_bus: EventBus | None = None


class DataEventType(Enum):
    """Types of data pipeline events."""

    # Data collection events
    NEW_GAMES_AVAILABLE = "new_games"
    DATA_SYNC_STARTED = "sync_started"
    DATA_SYNC_COMPLETED = "sync_completed"
    DATA_SYNC_FAILED = "sync_failed"

    # Training events
    TRAINING_THRESHOLD_REACHED = "training_threshold"
    TRAINING_STARTED = "training_started"
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"

    # Evaluation events
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_PROGRESS = "evaluation_progress"
    EVALUATION_COMPLETED = "evaluation_completed"
    EVALUATION_FAILED = "evaluation_failed"
    ELO_UPDATED = "elo_updated"

    # Promotion events
    PROMOTION_CANDIDATE = "promotion_candidate"
    PROMOTION_STARTED = "promotion_started"
    MODEL_PROMOTED = "model_promoted"
    PROMOTION_FAILED = "promotion_failed"
    PROMOTION_REJECTED = "promotion_rejected"

    # Curriculum events
    CURRICULUM_REBALANCED = "curriculum_rebalanced"
    WEIGHT_UPDATED = "weight_updated"
    ELO_SIGNIFICANT_CHANGE = "elo_significant_change"  # Triggers curriculum rebalance

    # Optimization events
    CMAES_TRIGGERED = "cmaes_triggered"
    CMAES_COMPLETED = "cmaes_completed"
    NAS_TRIGGERED = "nas_triggered"
    PLATEAU_DETECTED = "plateau_detected"
    HYPERPARAMETER_UPDATED = "hyperparameter_updated"

    # PBT events
    PBT_STARTED = "pbt_started"
    PBT_GENERATION_COMPLETE = "pbt_generation_complete"
    PBT_COMPLETED = "pbt_completed"

    # NAS events
    NAS_STARTED = "nas_started"
    NAS_GENERATION_COMPLETE = "nas_generation_complete"
    NAS_COMPLETED = "nas_completed"
    NAS_BEST_ARCHITECTURE = "nas_best_architecture"

    # PER (Prioritized Experience Replay) events
    PER_BUFFER_REBUILT = "per_buffer_rebuilt"
    PER_PRIORITIES_UPDATED = "per_priorities_updated"

    # Tier gating events
    TIER_PROMOTION = "tier_promotion"

    # Parity validation events
    PARITY_VALIDATION_STARTED = "parity_validation_started"
    PARITY_VALIDATION_COMPLETED = "parity_validation_completed"

    # Data quality events
    DATA_QUALITY_ALERT = "data_quality_alert"
    QUALITY_CHECK_FAILED = "quality_check_failed"
    QUALITY_SCORE_UPDATED = "quality_score_updated"  # Game quality recalculated
    QUALITY_DISTRIBUTION_CHANGED = "quality_distribution_changed"  # Significant shift
    HIGH_QUALITY_DATA_AVAILABLE = "high_quality_data_available"  # Ready for training
    LOW_QUALITY_DATA_WARNING = "low_quality_data_warning"  # Below threshold

    # Registry & metrics events
    REGISTRY_UPDATED = "registry_updated"
    METRICS_UPDATED = "metrics_updated"
    CACHE_INVALIDATED = "cache_invalidated"

    # Regression detection events (from unified RegressionDetector)
    REGRESSION_DETECTED = "regression_detected"  # Any regression
    REGRESSION_MINOR = "regression_minor"  # Severity: minor
    REGRESSION_MODERATE = "regression_moderate"  # Severity: moderate
    REGRESSION_SEVERE = "regression_severe"  # Severity: severe
    REGRESSION_CRITICAL = "regression_critical"  # Severity: critical - rollback recommended
    REGRESSION_CLEARED = "regression_cleared"  # Model recovered from regression

    # P2P/Model sync events
    P2P_MODEL_SYNCED = "p2p_model_synced"
    P2P_CLUSTER_HEALTHY = "p2p_cluster_healthy"
    P2P_CLUSTER_UNHEALTHY = "p2p_cluster_unhealthy"
    P2P_NODES_DEAD = "p2p_nodes_dead"
    P2P_SELFPLAY_SCALED = "p2p_selfplay_scaled"

    # System events
    DAEMON_STARTED = "daemon_started"
    DAEMON_STOPPED = "daemon_stopped"
    HOST_ONLINE = "host_online"
    HOST_OFFLINE = "host_offline"
    ERROR = "error"

    # Health & Recovery events (Phase 10 consolidation)
    HEALTH_CHECK_PASSED = "health_check_passed"
    HEALTH_CHECK_FAILED = "health_check_failed"
    HEALTH_ALERT = "health_alert"  # General health warning
    RESOURCE_CONSTRAINT = "resource_constraint"  # CPU/GPU/Memory/Disk pressure
    RECOVERY_INITIATED = "recovery_initiated"  # Auto-recovery started
    RECOVERY_COMPLETED = "recovery_completed"  # Auto-recovery finished
    RECOVERY_FAILED = "recovery_failed"  # Auto-recovery failed

    # Cluster status events
    CLUSTER_STATUS_CHANGED = "cluster_status_changed"
    NODE_UNHEALTHY = "node_unhealthy"
    NODE_RECOVERED = "node_recovered"

    # Lock/Synchronization events (December 2025)
    LOCK_ACQUIRED = "lock_acquired"
    LOCK_RELEASED = "lock_released"
    LOCK_TIMEOUT = "lock_timeout"
    DEADLOCK_DETECTED = "deadlock_detected"

    # Checkpoint events (December 2025)
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_LOADED = "checkpoint_loaded"

    # Task lifecycle events (December 2025)
    TASK_SPAWNED = "task_spawned"
    TASK_HEARTBEAT = "task_heartbeat"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_ORPHANED = "task_orphaned"
    TASK_CANCELLED = "task_cancelled"

    # Capacity/Resource events (December 2025)
    CLUSTER_CAPACITY_CHANGED = "cluster_capacity_changed"
    NODE_CAPACITY_UPDATED = "node_capacity_updated"
    BACKPRESSURE_ACTIVATED = "backpressure_activated"
    BACKPRESSURE_RELEASED = "backpressure_released"

    # Promotion lifecycle events (December 2025)
    PROMOTION_ROLLED_BACK = "promotion_rolled_back"

    # Quality feedback events (December 2025)
    PARITY_FAILURE_RATE_CHANGED = "parity_failure_rate_changed"

    # Leader election events (December 2025)
    LEADER_ELECTED = "leader_elected"
    LEADER_LOST = "leader_lost"
    LEADER_STEPDOWN = "leader_stepdown"

    # Encoding/Processing events (December 2025)
    ENCODING_BATCH_COMPLETED = "encoding_batch_completed"
    CALIBRATION_COMPLETED = "calibration_completed"

    # Error Recovery & Resilience events (December 2025)
    TRAINING_ROLLBACK_NEEDED = "training_rollback_needed"  # Rollback to previous checkpoint
    TRAINING_ROLLBACK_COMPLETED = "training_rollback_completed"
    MODEL_CORRUPTED = "model_corrupted"  # Model file corruption detected
    COORDINATOR_HEALTH_DEGRADED = "coordinator_health_degraded"  # Coordinator not fully healthy
    COORDINATOR_SHUTDOWN = "coordinator_shutdown"  # Graceful coordinator shutdown
    COORDINATOR_INIT_FAILED = "coordinator_init_failed"  # Coordinator failed to initialize
    HANDLER_TIMEOUT = "handler_timeout"  # Event handler timed out
    HANDLER_FAILED = "handler_failed"  # Event handler threw exception
    TASK_ABANDONED = "task_abandoned"  # Task intentionally abandoned (not orphaned)
    RESOURCE_CONSTRAINT_DETECTED = "resource_constraint_detected"  # Resource limit hit

    # Coordinator heartbeat events (December 2025)
    COORDINATOR_HEARTBEAT = "coordinator_heartbeat"  # Liveness signal from coordinator


@dataclass
class DataEvent:
    """A data pipeline event."""

    event_type: DataEventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Component that generated the event

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataEvent:
        """Create from dictionary."""
        return cls(
            event_type=DataEventType(data["event_type"]),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", ""),
        )


EventCallback = Callable[[DataEvent], Union[None, asyncio.Future]]


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


def get_event_bus() -> EventBus:
    """Get the global event bus singleton."""
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
    DataEventType.ELO_SIGNIFICANT_CHANGE,
    DataEventType.P2P_MODEL_SYNCED,
    DataEventType.PLATEAU_DETECTED,
    DataEventType.DATA_SYNC_COMPLETED,
    DataEventType.HYPERPARAMETER_UPDATED,
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
        from app.coordination.cross_process_events import bridge_to_cross_process
        bridge_to_cross_process(event.event_type.value, event.payload, event.source)
    except Exception as e:
        # Don't fail the main event if cross-process bridging fails
        print(f"[EventBus] Cross-process bridge failed: {e}")


# Convenience functions for common events


async def emit_new_games(host: str, new_games: int, total_games: int, source: str = "") -> None:
    """Emit a NEW_GAMES_AVAILABLE event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NEW_GAMES_AVAILABLE,
        payload={
            "host": host,
            "new_games": new_games,
            "total_games": total_games,
        },
        source=source,
    ))


async def emit_training_threshold(config: str, games: int, source: str = "") -> None:
    """Emit a TRAINING_THRESHOLD_REACHED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_THRESHOLD_REACHED,
        payload={
            "config": config,
            "games": games,
        },
        source=source,
    ))


async def emit_training_completed(
    config: str,
    success: bool,
    duration: float,
    model_path: str | None = None,
    source: str = "",
) -> None:
    """Emit a TRAINING_COMPLETED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_COMPLETED,
        payload={
            "config": config,
            "success": success,
            "duration": duration,
            "model_path": model_path,
        },
        source=source,
    ))


async def emit_evaluation_completed(
    config: str,
    elo: float,
    games_played: int,
    win_rate: float,
    source: str = "",
) -> None:
    """Emit an EVALUATION_COMPLETED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.EVALUATION_COMPLETED,
        payload={
            "config": config,
            "elo": elo,
            "games_played": games_played,
            "win_rate": win_rate,
        },
        source=source,
    ))


async def emit_model_promoted(
    model_id: str,
    config: str,
    elo: float,
    elo_gain: float,
    source: str = "",
) -> None:
    """Emit a MODEL_PROMOTED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.MODEL_PROMOTED,
        payload={
            "model_id": model_id,
            "config": config,
            "elo": elo,
            "elo_gain": elo_gain,
        },
        source=source,
    ))


async def emit_error(
    component: str,
    error: str,
    details: dict[str, Any] | None = None,
    source: str = "",
) -> None:
    """Emit an ERROR event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ERROR,
        payload={
            "component": component,
            "error": error,
            "details": details or {},
        },
        source=source,
    ))


async def emit_elo_updated(
    config: str,
    model_id: str,
    new_elo: float,
    old_elo: float,
    games_played: int,
    source: str = "",
) -> None:
    """Emit an ELO_UPDATED event and check for significant changes.

    If the Elo change exceeds the threshold from unified config,
    also emits an ELO_SIGNIFICANT_CHANGE event to trigger curriculum rebalancing.
    """
    elo_change = new_elo - old_elo

    # Emit the standard Elo update event
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ELO_UPDATED,
        payload={
            "config": config,
            "model_id": model_id,
            "new_elo": new_elo,
            "old_elo": old_elo,
            "elo_change": elo_change,
            "games_played": games_played,
        },
        source=source,
    ))

    # Check for significant change (threshold from unified config)
    try:
        from app.config.unified_config import get_config
        config_obj = get_config()
        threshold = config_obj.curriculum.elo_change_threshold
        should_rebalance = config_obj.curriculum.rebalance_on_elo_change
    except ImportError:
        threshold = 50  # Default if config not available
        should_rebalance = True

    if should_rebalance and abs(elo_change) >= threshold:
        await get_event_bus().publish(DataEvent(
            event_type=DataEventType.ELO_SIGNIFICANT_CHANGE,
            payload={
                "config": config,
                "model_id": model_id,
                "elo_change": elo_change,
                "new_elo": new_elo,
                "threshold": threshold,
            },
            source=source,
        ))


async def emit_quality_score_updated(
    game_id: str,
    quality_score: float,
    quality_category: str,
    training_weight: float,
    game_length: int = 0,
    is_decisive: bool = False,
    source: str = "",
) -> None:
    """Emit a QUALITY_SCORE_UPDATED event.

    Args:
        game_id: Unique game identifier
        quality_score: Computed quality score (0-1)
        quality_category: Category (excellent/good/adequate/poor/unusable)
        training_weight: Weight for training sample selection
        game_length: Number of moves in the game
        is_decisive: Whether game had a clear winner
        source: Component that computed the quality
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.QUALITY_SCORE_UPDATED,
        payload={
            "game_id": game_id,
            "quality_score": quality_score,
            "quality_category": quality_category,
            "training_weight": training_weight,
            "game_length": game_length,
            "is_decisive": is_decisive,
        },
        source=source,
    ))


async def emit_curriculum_rebalanced(
    config: str,
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    trigger: str = "scheduled",
    source: str = "",
) -> None:
    """Emit a CURRICULUM_REBALANCED event.

    Args:
        config: Board configuration (e.g., "square8_2p")
        old_weights: Previous curriculum weights
        new_weights: New curriculum weights
        trigger: What triggered the rebalance ("scheduled", "elo_change", "manual")
        source: Component that triggered the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CURRICULUM_REBALANCED,
        payload={
            "config": config,
            "old_weights": old_weights,
            "new_weights": new_weights,
            "trigger": trigger,
        },
        source=source,
    ))


async def emit_plateau_detected(
    config: str,
    current_elo: float,
    plateau_duration_games: int,
    plateau_duration_seconds: float,
    source: str = "",
) -> None:
    """Emit a PLATEAU_DETECTED event.

    Args:
        config: Board configuration
        current_elo: Current Elo rating
        plateau_duration_games: Number of games in plateau
        plateau_duration_seconds: Time duration of plateau
        source: Component that detected the plateau
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PLATEAU_DETECTED,
        payload={
            "config": config,
            "current_elo": current_elo,
            "plateau_duration_games": plateau_duration_games,
            "plateau_duration_seconds": plateau_duration_seconds,
        },
        source=source,
    ))


async def emit_cmaes_triggered(
    config: str,
    reason: str,
    current_params: dict[str, Any],
    source: str = "",
) -> None:
    """Emit a CMAES_TRIGGERED event.

    Args:
        config: Board configuration
        reason: Why CMA-ES was triggered (e.g., "plateau_detected", "manual")
        current_params: Current hyperparameters before optimization
        source: Component that triggered CMA-ES
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CMAES_TRIGGERED,
        payload={
            "config": config,
            "reason": reason,
            "current_params": current_params,
        },
        source=source,
    ))


async def emit_nas_triggered(
    config: str,
    reason: str,
    search_space: dict[str, Any] | None = None,
    source: str = "",
) -> None:
    """Emit a NAS_TRIGGERED event.

    Args:
        config: Board configuration
        reason: Why NAS was triggered
        search_space: Optional architecture search space configuration
        source: Component that triggered NAS
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NAS_TRIGGERED,
        payload={
            "config": config,
            "reason": reason,
            "search_space": search_space or {},
        },
        source=source,
    ))


async def emit_hyperparameter_updated(
    config: str,
    param_name: str,
    old_value: Any,
    new_value: Any,
    optimizer: str = "manual",
    source: str = "",
) -> None:
    """Emit a HYPERPARAMETER_UPDATED event.

    Args:
        config: Board configuration
        param_name: Name of the parameter that changed
        old_value: Previous value
        new_value: New value
        optimizer: What triggered the update ("cmaes", "nas", "manual")
        source: Component that updated the parameter
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HYPERPARAMETER_UPDATED,
        payload={
            "config": config,
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "optimizer": optimizer,
        },
        source=source,
    ))


# =============================================================================
# Data Sync Events
# =============================================================================

async def emit_data_sync_started(
    host: str,
    sync_type: str = "incremental",
    source: str = "",
) -> None:
    """Emit a DATA_SYNC_STARTED event.

    Args:
        host: Host being synced
        sync_type: Type of sync (incremental, full)
        source: Component initiating the sync
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_SYNC_STARTED,
        payload={
            "host": host,
            "sync_type": sync_type,
        },
        source=source,
    ))


async def emit_data_sync_completed(
    host: str,
    games_synced: int,
    duration: float,
    bytes_transferred: int = 0,
    source: str = "",
    avg_quality_score: float = 0.0,
    high_quality_count: int = 0,
    config: str = "",
) -> None:
    """Emit a DATA_SYNC_COMPLETED event with quality metrics.

    Args:
        host: Host that was synced
        games_synced: Number of games transferred
        duration: Sync duration in seconds
        bytes_transferred: Bytes transferred (if known)
        source: Component that performed the sync
        avg_quality_score: Average quality score of synced games (0-1)
        high_quality_count: Number of games with quality >= 0.7
        config: Configuration key (e.g., "square8_2p")
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_SYNC_COMPLETED,
        payload={
            "host": host,
            "games_synced": games_synced,
            "duration": duration,
            "bytes_transferred": bytes_transferred,
            "avg_quality_score": avg_quality_score,
            "high_quality_count": high_quality_count,
            "config": config,
        },
        source=source,
    ))


async def emit_data_sync_failed(
    host: str,
    error: str,
    retry_count: int = 0,
    source: str = "",
) -> None:
    """Emit a DATA_SYNC_FAILED event.

    Args:
        host: Host that failed to sync
        error: Error message
        retry_count: Number of retries attempted
        source: Component that attempted the sync
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_SYNC_FAILED,
        payload={
            "host": host,
            "error": error,
            "retry_count": retry_count,
        },
        source=source,
    ))


# =============================================================================
# Host Status Events
# =============================================================================

async def emit_host_online(
    host: str,
    capabilities: list[str] | None = None,
    source: str = "",
) -> None:
    """Emit a HOST_ONLINE event.

    Args:
        host: Host that came online
        capabilities: List of host capabilities (gpu, cpu, etc.)
        source: Component that detected the host
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HOST_ONLINE,
        payload={
            "host": host,
            "capabilities": capabilities or [],
        },
        source=source,
    ))


async def emit_host_offline(
    host: str,
    reason: str = "",
    last_seen: float | None = None,
    source: str = "",
) -> None:
    """Emit a HOST_OFFLINE event.

    Args:
        host: Host that went offline
        reason: Reason for going offline (timeout, error, etc.)
        last_seen: Timestamp when host was last seen
        source: Component that detected the offline status
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HOST_OFFLINE,
        payload={
            "host": host,
            "reason": reason,
            "last_seen": last_seen,
        },
        source=source,
    ))


# =============================================================================
# Daemon Lifecycle Events
# =============================================================================

async def emit_daemon_started(
    daemon_name: str,
    hostname: str,
    pid: int,
    source: str = "",
) -> None:
    """Emit a DAEMON_STARTED event.

    Args:
        daemon_name: Name of the daemon
        hostname: Host running the daemon
        pid: Process ID
        source: Component starting the daemon
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DAEMON_STARTED,
        payload={
            "daemon_name": daemon_name,
            "hostname": hostname,
            "pid": pid,
        },
        source=source,
    ))


async def emit_daemon_stopped(
    daemon_name: str,
    hostname: str,
    reason: str = "normal",
    source: str = "",
) -> None:
    """Emit a DAEMON_STOPPED event.

    Args:
        daemon_name: Name of the daemon
        hostname: Host running the daemon
        reason: Reason for stopping (normal, error, signal)
        source: Component reporting the stop
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DAEMON_STOPPED,
        payload={
            "daemon_name": daemon_name,
            "hostname": hostname,
            "reason": reason,
        },
        source=source,
    ))


# =============================================================================
# Training Failure Events
# =============================================================================

async def emit_training_started(
    config: str,
    model_path: str | None = None,
    source: str = "",
) -> None:
    """Emit a TRAINING_STARTED event.

    Args:
        config: Board configuration
        model_path: Path to model being trained
        source: Component starting the training
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_STARTED,
        payload={
            "config": config,
            "model_path": model_path,
        },
        source=source,
    ))


async def emit_training_failed(
    config: str,
    error: str,
    duration: float = 0,
    source: str = "",
) -> None:
    """Emit a TRAINING_FAILED event.

    Args:
        config: Board configuration
        error: Error message
        duration: How long training ran before failing
        source: Component reporting the failure
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_FAILED,
        payload={
            "config": config,
            "error": error,
            "duration": duration,
        },
        source=source,
    ))


# =============================================================================
# Evaluation Failure Events
# =============================================================================

async def emit_evaluation_started(
    config: str,
    model_id: str,
    games_planned: int = 0,
    source: str = "",
) -> None:
    """Emit an EVALUATION_STARTED event.

    Args:
        config: Board configuration
        model_id: Model being evaluated
        games_planned: Number of evaluation games planned
        source: Component starting the evaluation
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.EVALUATION_STARTED,
        payload={
            "config": config,
            "model_id": model_id,
            "games_planned": games_planned,
        },
        source=source,
    ))


async def emit_evaluation_failed(
    config: str,
    model_id: str,
    error: str,
    source: str = "",
) -> None:
    """Emit an EVALUATION_FAILED event.

    Args:
        config: Board configuration
        model_id: Model being evaluated
        error: Error message
        source: Component reporting the failure
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.EVALUATION_FAILED,
        payload={
            "config": config,
            "model_id": model_id,
            "error": error,
        },
        source=source,
    ))


# =============================================================================
# Promotion Events
# =============================================================================

async def emit_promotion_started(
    config: str,
    model_id: str,
    source: str = "",
) -> None:
    """Emit a PROMOTION_STARTED event.

    Args:
        config: Board configuration
        model_id: Model being considered for promotion
        source: Component starting the promotion
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PROMOTION_STARTED,
        payload={
            "config": config,
            "model_id": model_id,
        },
        source=source,
    ))


async def emit_promotion_failed(
    config: str,
    model_id: str,
    error: str,
    source: str = "",
) -> None:
    """Emit a PROMOTION_FAILED event.

    Args:
        config: Board configuration
        model_id: Model that failed promotion
        error: Error message
        source: Component reporting the failure
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PROMOTION_FAILED,
        payload={
            "config": config,
            "model_id": model_id,
            "error": error,
        },
        source=source,
    ))


async def emit_promotion_rejected(
    config: str,
    model_id: str,
    reason: str,
    elo_improvement: float = 0,
    source: str = "",
) -> None:
    """Emit a PROMOTION_REJECTED event.

    Args:
        config: Board configuration
        model_id: Model that was rejected
        reason: Reason for rejection
        elo_improvement: Elo improvement achieved (if any)
        source: Component rejecting the promotion
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PROMOTION_REJECTED,
        payload={
            "config": config,
            "model_id": model_id,
            "reason": reason,
            "elo_improvement": elo_improvement,
        },
        source=source,
    ))


# =============================================================================
# Tier Promotion Events
# =============================================================================

async def emit_tier_promotion(
    config: str,
    old_tier: str,
    new_tier: str,
    model_id: str = "",
    win_rate: float = 0.0,
    elo: float = 0.0,
    games_played: int = 0,
    source: str = "",
) -> None:
    """Emit a TIER_PROMOTION event for difficulty ladder progression.

    Args:
        config: Board configuration (e.g., "square8_2p")
        old_tier: Previous tier (e.g., "D4")
        new_tier: New tier after promotion (e.g., "D5")
        model_id: ID of the model being promoted
        win_rate: Win rate that triggered promotion
        elo: Current Elo rating
        games_played: Number of games played at current tier
        source: Component that triggered the promotion
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TIER_PROMOTION,
        payload={
            "config": config,
            "old_tier": old_tier,
            "new_tier": new_tier,
            "model_id": model_id,
            "win_rate": win_rate,
            "elo": elo,
            "games_played": games_played,
        },
        source=source,
    ))


# =============================================================================
# Lock/Synchronization Events (December 2025)
# =============================================================================

async def emit_lock_acquired(
    resource_id: str,
    holder: str,
    lock_type: str = "exclusive",
    timeout_seconds: float = 0.0,
    source: str = "",
) -> None:
    """Emit a LOCK_ACQUIRED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.LOCK_ACQUIRED,
        payload={
            "resource_id": resource_id,
            "holder": holder,
            "lock_type": lock_type,
            "timeout_seconds": timeout_seconds,
        },
        source=source,
    ))


async def emit_lock_released(
    resource_id: str,
    holder: str,
    held_duration_seconds: float = 0.0,
    source: str = "",
) -> None:
    """Emit a LOCK_RELEASED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.LOCK_RELEASED,
        payload={
            "resource_id": resource_id,
            "holder": holder,
            "held_duration_seconds": held_duration_seconds,
        },
        source=source,
    ))


async def emit_deadlock_detected(
    resources: list[str],
    holders: list[str],
    source: str = "",
) -> None:
    """Emit a DEADLOCK_DETECTED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DEADLOCK_DETECTED,
        payload={
            "resources": resources,
            "holders": holders,
        },
        source=source,
    ))


# =============================================================================
# Checkpoint Events (December 2025)
# =============================================================================

async def emit_checkpoint_saved(
    config: str,
    checkpoint_path: str,
    epoch: int = 0,
    step: int = 0,
    metrics: dict[str, float] | None = None,
    source: str = "",
) -> None:
    """Emit a CHECKPOINT_SAVED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CHECKPOINT_SAVED,
        payload={
            "config": config,
            "checkpoint_path": checkpoint_path,
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
        },
        source=source,
    ))


async def emit_checkpoint_loaded(
    config: str,
    checkpoint_path: str,
    epoch: int = 0,
    step: int = 0,
    source: str = "",
) -> None:
    """Emit a CHECKPOINT_LOADED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CHECKPOINT_LOADED,
        payload={
            "config": config,
            "checkpoint_path": checkpoint_path,
            "epoch": epoch,
            "step": step,
        },
        source=source,
    ))


# =============================================================================
# Task Lifecycle Events (December 2025)
# =============================================================================

async def emit_task_spawned(
    task_id: str,
    task_type: str,
    node_id: str,
    config: str = "",
    priority: int = 0,
    source: str = "",
) -> None:
    """Emit a TASK_SPAWNED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_SPAWNED,
        payload={
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "config": config,
            "priority": priority,
        },
        source=source,
    ))


async def emit_task_heartbeat(
    task_id: str,
    node_id: str,
    progress: float = 0.0,
    status: str = "running",
    source: str = "",
) -> None:
    """Emit a TASK_HEARTBEAT event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_HEARTBEAT,
        payload={
            "task_id": task_id,
            "node_id": node_id,
            "progress": progress,
            "status": status,
        },
        source=source,
    ))


async def emit_task_completed(
    task_id: str,
    task_type: str,
    node_id: str,
    duration_seconds: float = 0.0,
    result: dict[str, Any] | None = None,
    source: str = "",
) -> None:
    """Emit a TASK_COMPLETED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_COMPLETED,
        payload={
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "duration_seconds": duration_seconds,
            "result": result or {},
        },
        source=source,
    ))


async def emit_task_failed(
    task_id: str,
    task_type: str,
    node_id: str,
    error: str,
    duration_seconds: float = 0.0,
    source: str = "",
) -> None:
    """Emit a TASK_FAILED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_FAILED,
        payload={
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "error": error,
            "duration_seconds": duration_seconds,
        },
        source=source,
    ))


async def emit_task_orphaned(
    task_id: str,
    task_type: str,
    node_id: str,
    last_heartbeat_seconds_ago: float = 0.0,
    source: str = "",
) -> None:
    """Emit a TASK_ORPHANED event for tasks that stopped sending heartbeats."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_ORPHANED,
        payload={
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "last_heartbeat_seconds_ago": last_heartbeat_seconds_ago,
        },
        source=source,
    ))


# =============================================================================
# Capacity/Backpressure Events (December 2025)
# =============================================================================

async def emit_cluster_capacity_changed(
    total_gpus: int,
    available_gpus: int,
    total_nodes: int,
    healthy_nodes: int,
    source: str = "",
) -> None:
    """Emit a CLUSTER_CAPACITY_CHANGED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CLUSTER_CAPACITY_CHANGED,
        payload={
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
        },
        source=source,
    ))


async def emit_backpressure_activated(
    reason: str,
    queue_depth: int = 0,
    utilization_percent: float = 0.0,
    source: str = "",
) -> None:
    """Emit a BACKPRESSURE_ACTIVATED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BACKPRESSURE_ACTIVATED,
        payload={
            "reason": reason,
            "queue_depth": queue_depth,
            "utilization_percent": utilization_percent,
        },
        source=source,
    ))


async def emit_backpressure_released(
    reason: str,
    duration_seconds: float = 0.0,
    source: str = "",
) -> None:
    """Emit a BACKPRESSURE_RELEASED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BACKPRESSURE_RELEASED,
        payload={
            "reason": reason,
            "duration_seconds": duration_seconds,
        },
        source=source,
    ))


# =============================================================================
# Leader Election Events (December 2025)
# =============================================================================

async def emit_leader_elected(
    leader_id: str,
    term: int = 0,
    source: str = "",
) -> None:
    """Emit a LEADER_ELECTED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.LEADER_ELECTED,
        payload={
            "leader_id": leader_id,
            "term": term,
        },
        source=source,
    ))


async def emit_leader_lost(
    old_leader_id: str,
    reason: str = "",
    source: str = "",
) -> None:
    """Emit a LEADER_LOST event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.LEADER_LOST,
        payload={
            "old_leader_id": old_leader_id,
            "reason": reason,
        },
        source=source,
    ))


# Encoding/Processing Events (December 2025)

async def emit_encoding_batch_completed(
    games_count: int,
    samples_count: int,
    errors_count: int,
    board_type: str = "",
    duration_seconds: float = 0.0,
    source: str = "parallel_encoding",
) -> None:
    """Emit an ENCODING_BATCH_COMPLETED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ENCODING_BATCH_COMPLETED,
        payload={
            "games_count": games_count,
            "samples_count": samples_count,
            "errors_count": errors_count,
            "board_type": board_type,
            "duration_seconds": duration_seconds,
        },
        source=source,
    ))


async def emit_calibration_completed(
    config_key: str,
    calibration_type: str,
    old_value: float = 0.0,
    new_value: float = 0.0,
    games_analyzed: int = 0,
    source: str = "tier_calibrator",
) -> None:
    """Emit a CALIBRATION_COMPLETED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CALIBRATION_COMPLETED,
        payload={
            "config_key": config_key,
            "calibration_type": calibration_type,
            "old_value": old_value,
            "new_value": new_value,
            "games_analyzed": games_analyzed,
        },
        source=source,
    ))
