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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

# Global singleton instance
_event_bus: Optional["EventBus"] = None


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


@dataclass
class DataEvent:
    """A data pipeline event."""

    event_type: DataEventType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Component that generated the event

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataEvent":
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
    """

    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[DataEventType, List[EventCallback]] = {}
        self._global_subscribers: List[EventCallback] = []
        self._event_history: List[DataEvent] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()

    def subscribe(
        self,
        event_type: Optional[DataEventType],
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
        event_type: Optional[DataEventType],
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
            if event_type in self._subscribers:
                if callback in self._subscribers[event_type]:
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

        # Bridge to cross-process queue for multi-daemon coordination
        if bridge_cross_process:
            _bridge_to_cross_process(event)

        # Get all callbacks for this event
        callbacks = list(self._global_subscribers)
        if event.event_type in self._subscribers:
            callbacks.extend(self._subscribers[event.event_type])

        # Invoke each callback
        for callback in callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"[EventBus] Error in subscriber for {event.event_type.value}: {e}")

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
        event_type: Optional[DataEventType] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[DataEvent]:
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
    model_path: Optional[str] = None,
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
    details: Optional[Dict[str, Any]] = None,
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


async def emit_curriculum_rebalanced(
    config: str,
    old_weights: Dict[str, float],
    new_weights: Dict[str, float],
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
    current_params: Dict[str, Any],
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
    search_space: Optional[Dict[str, Any]] = None,
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
) -> None:
    """Emit a DATA_SYNC_COMPLETED event.

    Args:
        host: Host that was synced
        games_synced: Number of games transferred
        duration: Sync duration in seconds
        bytes_transferred: Bytes transferred (if known)
        source: Component that performed the sync
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_SYNC_COMPLETED,
        payload={
            "host": host,
            "games_synced": games_synced,
            "duration": duration,
            "bytes_transferred": bytes_transferred,
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
    capabilities: Optional[List[str]] = None,
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
    last_seen: Optional[float] = None,
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
    model_path: Optional[str] = None,
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
