"""PER Orchestrator - Unified monitoring for Prioritized Experience Replay (December 2025).

This module provides centralized monitoring of PER (Prioritized Experience Replay)
buffer events across the system. It subscribes to PER-related events and provides:

1. Buffer state tracking across training runs
2. Priority distribution monitoring
3. Buffer rebuild coordination
4. Integration with training pipeline

Event Integration:
- Subscribes to PER_BUFFER_REBUILT: Tracks when buffers are rebuilt
- Subscribes to PER_PRIORITIES_UPDATED: Tracks priority updates
- Emits aggregate statistics for monitoring dashboards

Usage:
    from app.training.per_orchestrator import (
        PEROrchestrator,
        wire_per_events,
        get_per_orchestrator,
    )

    # Wire PER events to orchestrator
    orchestrator = wire_per_events()

    # Get buffer status
    status = orchestrator.get_status()
    print(f"Active buffers: {status['active_buffers']}")

    # Get buffer history
    history = orchestrator.get_buffer_history()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PERBufferState:
    """State of a PER buffer."""

    buffer_path: str
    buffer_size: int
    tree_depth: int = 0
    priority_sum: float = 0.0
    min_priority: float = 0.0
    max_priority: float = 1.0
    last_rebuild_time: float = 0.0
    last_priority_update_time: float = 0.0
    rebuild_count: int = 0
    priority_update_count: int = 0
    config_key: str = ""


@dataclass
class PERStats:
    """Aggregate PER statistics."""

    total_buffers_tracked: int = 0
    total_rebuilds: int = 0
    total_priority_updates: int = 0
    active_buffers: int = 0
    total_samples: int = 0
    avg_buffer_size: float = 0.0
    last_activity_time: float = 0.0


class PEROrchestrator:
    """Orchestrates PER buffer monitoring across the training system.

    Subscribes to PER events and maintains a unified view of buffer states
    across all training runs.
    """

    def __init__(
        self,
        buffer_stale_threshold_seconds: float = 3600.0,
        max_history_per_buffer: int = 100,
    ):
        """Initialize PER orchestrator.

        Args:
            buffer_stale_threshold_seconds: Time after which buffer is considered stale
            max_history_per_buffer: Maximum history entries per buffer
        """
        self.buffer_stale_threshold_seconds = buffer_stale_threshold_seconds
        self.max_history_per_buffer = max_history_per_buffer

        # Buffer tracking
        self._buffers: dict[str, PERBufferState] = {}
        self._buffer_history: dict[str, list[dict[str, Any]]] = {}

        # Statistics
        self._total_rebuilds: int = 0
        self._total_priority_updates: int = 0
        self._subscribed: bool = False

    def subscribe_to_events(self) -> bool:
        """Subscribe to PER-related events from the event bus.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()

            # Subscribe to PER buffer events
            bus.subscribe(DataEventType.PER_BUFFER_REBUILT, self._on_buffer_rebuilt)
            bus.subscribe(DataEventType.PER_PRIORITIES_UPDATED, self._on_priorities_updated)

            self._subscribed = True
            logger.info("[PEROrchestrator] Subscribed to PER events")
            return True

        except Exception as e:
            logger.warning(f"[PEROrchestrator] Failed to subscribe to events: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from PER events."""
        if not self._subscribed:
            return

        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.PER_BUFFER_REBUILT, self._on_buffer_rebuilt)
            bus.unsubscribe(DataEventType.PER_PRIORITIES_UPDATED, self._on_priorities_updated)
            self._subscribed = False

        except Exception:
            pass

    def _on_buffer_rebuilt(self, event: Any) -> None:
        """Handle PER_BUFFER_REBUILT event.

        Args:
            event: DataEvent with buffer rebuild information
        """
        payload = event.payload if hasattr(event, 'payload') else {}

        buffer_path = payload.get("buffer_path", "unknown")
        buffer_size = payload.get("buffer_size", 0)
        tree_depth = payload.get("tree_depth", 0)
        config_key = payload.get("config", "")

        logger.info(
            f"[PEROrchestrator] Buffer rebuilt: {buffer_path} "
            f"(size={buffer_size}, depth={tree_depth})"
        )

        # Update or create buffer state
        if buffer_path not in self._buffers:
            self._buffers[buffer_path] = PERBufferState(
                buffer_path=buffer_path,
                buffer_size=buffer_size,
                tree_depth=tree_depth,
                config_key=config_key,
            )
        else:
            state = self._buffers[buffer_path]
            state.buffer_size = buffer_size
            state.tree_depth = tree_depth
            if config_key:
                state.config_key = config_key

        state = self._buffers[buffer_path]
        state.last_rebuild_time = time.time()
        state.rebuild_count += 1

        self._total_rebuilds += 1

        # Track history
        self._add_to_history(buffer_path, "rebuild", {
            "buffer_size": buffer_size,
            "tree_depth": tree_depth,
            "timestamp": time.time(),
        })

        # Emit aggregate event
        self._emit_stats_event()

    def _on_priorities_updated(self, event: Any) -> None:
        """Handle PER_PRIORITIES_UPDATED event.

        Args:
            event: DataEvent with priority update information
        """
        payload = event.payload if hasattr(event, 'payload') else {}

        buffer_path = payload.get("buffer_path", "unknown")
        priority_sum = payload.get("priority_sum", 0.0)
        min_priority = payload.get("min_priority", 0.0)
        max_priority = payload.get("max_priority", 1.0)
        samples_updated = payload.get("samples_updated", 0)

        logger.debug(
            f"[PEROrchestrator] Priorities updated: {buffer_path} "
            f"(sum={priority_sum:.2f}, range=[{min_priority:.4f}, {max_priority:.4f}])"
        )

        # Update buffer state
        if buffer_path not in self._buffers:
            self._buffers[buffer_path] = PERBufferState(buffer_path=buffer_path)

        state = self._buffers[buffer_path]
        state.priority_sum = priority_sum
        state.min_priority = min_priority
        state.max_priority = max_priority
        state.last_priority_update_time = time.time()
        state.priority_update_count += 1

        self._total_priority_updates += 1

        # Track history (less frequently for priority updates)
        if state.priority_update_count % 10 == 0:
            self._add_to_history(buffer_path, "priority_update", {
                "priority_sum": priority_sum,
                "min_priority": min_priority,
                "max_priority": max_priority,
                "samples_updated": samples_updated,
                "timestamp": time.time(),
            })

    def _add_to_history(self, buffer_path: str, event_type: str, data: dict[str, Any]) -> None:
        """Add entry to buffer history.

        Args:
            buffer_path: Path to buffer
            event_type: Type of event
            data: Event data
        """
        if buffer_path not in self._buffer_history:
            self._buffer_history[buffer_path] = []

        history = self._buffer_history[buffer_path]
        history.append({
            "event_type": event_type,
            **data,
        })

        # Trim history if needed
        if len(history) > self.max_history_per_buffer:
            self._buffer_history[buffer_path] = history[-self.max_history_per_buffer:]

    def _emit_stats_event(self) -> None:
        """Emit aggregate statistics event."""
        try:
            from app.distributed.data_events import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            stats = self.get_stats()

            event = DataEvent(
                event_type=DataEventType.METRICS_UPDATED,
                payload={
                    "metric_type": "per_orchestrator",
                    "total_buffers": stats.total_buffers_tracked,
                    "active_buffers": stats.active_buffers,
                    "total_samples": stats.total_samples,
                    "total_rebuilds": stats.total_rebuilds,
                    "timestamp": time.time(),
                },
                source="per_orchestrator",
            )

            bus = get_event_bus()
            bus.publish_sync(event)

        except Exception as e:
            logger.debug(f"Failed to emit stats event: {e}")

    def get_buffer_state(self, buffer_path: str) -> PERBufferState | None:
        """Get state of a specific buffer.

        Args:
            buffer_path: Path to buffer

        Returns:
            PERBufferState or None if not found
        """
        return self._buffers.get(buffer_path)

    def get_active_buffers(self) -> list[PERBufferState]:
        """Get all active (non-stale) buffers.

        Returns:
            List of active buffer states
        """
        now = time.time()
        active = []

        for state in self._buffers.values():
            last_activity = max(state.last_rebuild_time, state.last_priority_update_time)
            if now - last_activity < self.buffer_stale_threshold_seconds:
                active.append(state)

        return active

    def get_buffer_history(self, buffer_path: str | None = None) -> dict[str, list[dict]]:
        """Get buffer history.

        Args:
            buffer_path: Specific buffer to get history for (all if None)

        Returns:
            Dict mapping buffer paths to history entries
        """
        if buffer_path:
            return {buffer_path: self._buffer_history.get(buffer_path, [])}
        return dict(self._buffer_history)

    def get_stats(self) -> PERStats:
        """Get aggregate PER statistics.

        Returns:
            PERStats with current metrics
        """
        active_buffers = self.get_active_buffers()
        total_samples = sum(b.buffer_size for b in active_buffers)

        last_activity = 0.0
        for state in self._buffers.values():
            last_activity = max(
                last_activity,
                state.last_rebuild_time,
                state.last_priority_update_time,
            )

        return PERStats(
            total_buffers_tracked=len(self._buffers),
            total_rebuilds=self._total_rebuilds,
            total_priority_updates=self._total_priority_updates,
            active_buffers=len(active_buffers),
            total_samples=total_samples,
            avg_buffer_size=total_samples / len(active_buffers) if active_buffers else 0.0,
            last_activity_time=last_activity,
        )

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status for monitoring.

        Returns:
            Dict with status information
        """
        stats = self.get_stats()

        return {
            "subscribed": self._subscribed,
            "total_buffers_tracked": stats.total_buffers_tracked,
            "active_buffers": stats.active_buffers,
            "total_samples": stats.total_samples,
            "total_rebuilds": stats.total_rebuilds,
            "total_priority_updates": stats.total_priority_updates,
            "avg_buffer_size": stats.avg_buffer_size,
            "last_activity_time": stats.last_activity_time,
            "buffer_paths": list(self._buffers.keys()),
        }


# Singleton instance
_per_orchestrator: PEROrchestrator | None = None


def wire_per_events(
    buffer_stale_threshold_seconds: float = 3600.0,
) -> PEROrchestrator:
    """Wire PER events to the orchestrator.

    This enables centralized monitoring of PER buffer state across
    all training runs.

    Args:
        buffer_stale_threshold_seconds: Time after which buffer is stale

    Returns:
        PEROrchestrator instance
    """
    global _per_orchestrator

    if _per_orchestrator is None:
        _per_orchestrator = PEROrchestrator(
            buffer_stale_threshold_seconds=buffer_stale_threshold_seconds,
        )
        _per_orchestrator.subscribe_to_events()

        logger.info(
            f"[wire_per_events] PER events wired to orchestrator "
            f"(stale_threshold={buffer_stale_threshold_seconds}s)"
        )

    return _per_orchestrator


def get_per_orchestrator() -> PEROrchestrator | None:
    """Get the global PER orchestrator if configured."""
    return _per_orchestrator


def reset_per_orchestrator() -> None:
    """Reset the PER orchestrator singleton (for testing)."""
    global _per_orchestrator
    if _per_orchestrator:
        _per_orchestrator.unsubscribe()
    _per_orchestrator = None


__all__ = [
    "PERBufferState",
    "PEROrchestrator",
    "PERStats",
    "get_per_orchestrator",
    "reset_per_orchestrator",
    "wire_per_events",
]
