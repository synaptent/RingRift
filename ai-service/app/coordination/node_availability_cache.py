"""Unified Node Availability Cache.

Single source of truth for node availability, aggregating signals from:
- P2P heartbeats (fastest, ~10s detection)
- SSH probes (reliable, on-demand)
- HealthCheckOrchestrator (comprehensive, 60-120s)
- Event system (P2P_NODE_DEAD, HOST_OFFLINE)

Usage:
    from app.coordination.node_availability_cache import (
        NodeAvailabilityCache,
        get_availability_cache,
    )

    cache = get_availability_cache()

    # Check if node is available
    if cache.is_available("my-node"):
        dispatch_job("my-node")

    # Get all available nodes
    available = cache.get_available_nodes()

December 2025: Created as part of cluster node availability fix.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AvailabilityReason(Enum):
    """Reason for node availability status."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    P2P_DEAD = "p2p_dead"
    SSH_FAILED = "ssh_failed"
    SSH_TIMEOUT = "ssh_timeout"
    GPU_ERROR = "gpu_error"
    HOST_OFFLINE = "host_offline"
    MANUALLY_DISABLED = "manually_disabled"
    HEALTH_CHECK_FAILED = "health_check_failed"
    RECOVERED = "recovered"


@dataclass
class NodeAvailabilityEntry:
    """Availability status for a single node."""

    node_id: str
    is_available: bool = True
    reason: AvailabilityReason = AvailabilityReason.UNKNOWN
    last_update: float = field(default_factory=time.time)
    last_successful_contact: float = field(default_factory=time.time)
    consecutive_failures: int = 0

    # Source of last status update
    source: str = "unknown"  # "p2p", "ssh_probe", "health_check", "event", "manual"

    # Additional context
    error_message: str | None = None

    def mark_available(self, source: str = "unknown") -> None:
        """Mark node as available."""
        self.is_available = True
        self.reason = AvailabilityReason.HEALTHY
        self.last_update = time.time()
        self.last_successful_contact = time.time()
        self.consecutive_failures = 0
        self.source = source
        self.error_message = None

    def mark_unavailable(
        self,
        reason: AvailabilityReason,
        source: str = "unknown",
        error_message: str | None = None,
    ) -> None:
        """Mark node as unavailable."""
        self.is_available = False
        self.reason = reason
        self.last_update = time.time()
        self.consecutive_failures += 1
        self.source = source
        self.error_message = error_message

    def seconds_since_contact(self) -> float:
        """Seconds since last successful contact."""
        return time.time() - self.last_successful_contact

    def is_stale(self, max_age_seconds: float = 300.0) -> bool:
        """Check if entry is stale (hasn't been updated recently)."""
        return (time.time() - self.last_update) > max_age_seconds


class NodeAvailabilityCache:
    """Unified cache for node availability status.

    Thread-safe singleton that aggregates availability signals from multiple sources.
    """

    _instance: NodeAvailabilityCache | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._cache: dict[str, NodeAvailabilityEntry] = {}
        self._lock = threading.RLock()
        self._event_subscribed = False

        # Callbacks for availability changes
        self._on_unavailable_callbacks: list[Callable[[str, AvailabilityReason], None]] = []
        self._on_available_callbacks: list[Callable[[str], None]] = []

        # Configuration
        self.stale_threshold_seconds = 300.0  # 5 minutes
        self.auto_recover_after_seconds = 600.0  # 10 minutes - auto-retry after this

    @classmethod
    def get_instance(cls) -> NodeAvailabilityCache:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._instance_lock:
            cls._instance = None

    def wire_to_events(self) -> None:
        """Subscribe to availability-related events.

        Subscribes to P2P_NODE_DEAD, HOST_OFFLINE, NODE_RECOVERED events
        for automatic cache updates.

        Thread-safe: Uses lock to prevent duplicate subscriptions in concurrent calls.
        """
        # Fast path: check without lock first (safe to read bool atomically)
        if self._event_subscribed:
            return

        # Double-checked locking with proper synchronization
        with self._lock:
            # Re-check after acquiring lock to prevent race condition
            if self._event_subscribed:
                return

            try:
                from app.coordination.event_router import get_router
                from app.distributed.data_events import DataEventType

                router = get_router()
                router.subscribe(DataEventType.P2P_NODE_DEAD, self._on_node_dead_event)
                router.subscribe(DataEventType.HOST_OFFLINE, self._on_host_offline_event)
                router.subscribe(DataEventType.NODE_RECOVERED, self._on_node_recovered_event)

                self._event_subscribed = True
                logger.info("[NodeAvailabilityCache] Subscribed to availability events")
            except ImportError as e:
                logger.debug(f"[NodeAvailabilityCache] Event router not available: {e}")
            except (RuntimeError, AttributeError, TypeError, KeyError) as e:
                # RuntimeError: router in invalid state
                # AttributeError: subscribe method missing
                # TypeError: callback signature mismatch
                # KeyError: DataEventType enum value missing
                logger.warning(f"[NodeAvailabilityCache] Failed to subscribe to events: {e}")

    async def _on_node_dead_event(self, event) -> None:
        """Handle P2P_NODE_DEAD event."""
        # December 29, 2025: Fix 'RouterEvent' object has no attribute 'get' bug
        # Event can be either a dict or a RouterEvent object
        payload = event if isinstance(event, dict) else getattr(event, "payload", {})
        if not isinstance(payload, dict):
            payload = {}
        node_id = payload.get("node_id")
        if node_id:
            self.mark_unavailable(
                node_id,
                AvailabilityReason.P2P_DEAD,
                source="event",
                error_message=payload.get("reason"),
            )

    async def _on_host_offline_event(self, event) -> None:
        """Handle HOST_OFFLINE event."""
        # December 29, 2025: Fix 'RouterEvent' object has no attribute 'get' bug
        payload = event if isinstance(event, dict) else getattr(event, "payload", {})
        if not isinstance(payload, dict):
            payload = {}
        node_id = payload.get("node_id") or payload.get("host")
        if node_id:
            self.mark_unavailable(
                node_id,
                AvailabilityReason.HOST_OFFLINE,
                source="event",
                error_message=payload.get("reason"),
            )

    async def _on_node_recovered_event(self, event) -> None:
        """Handle NODE_RECOVERED event."""
        # December 29, 2025: Fix 'RouterEvent' object has no attribute 'get' bug
        payload = event if isinstance(event, dict) else getattr(event, "payload", {})
        if not isinstance(payload, dict):
            payload = {}
        node_id = payload.get("node_id")
        if node_id:
            self.mark_available(node_id, source="event")

    def is_available(self, node_id: str) -> bool:
        """Check if a node is available.

        Args:
            node_id: The node ID to check

        Returns:
            True if node is available or unknown, False if explicitly unavailable
        """
        with self._lock:
            entry = self._cache.get(node_id)
            if entry is None:
                # Unknown nodes are assumed available (optimistic)
                return True

            # Check for auto-recovery (long time since last failure)
            if not entry.is_available and entry.seconds_since_contact() > self.auto_recover_after_seconds:
                logger.info(
                    f"[NodeAvailabilityCache] Auto-recovering {node_id} after "
                    f"{entry.seconds_since_contact():.0f}s"
                )
                entry.mark_available(source="auto_recovery")

            return entry.is_available

    def mark_available(self, node_id: str, source: str = "unknown") -> None:
        """Mark a node as available.

        Args:
            node_id: The node ID
            source: Source of the availability update
        """
        with self._lock:
            entry = self._cache.get(node_id)
            was_unavailable = entry is not None and not entry.is_available

            if entry is None:
                entry = NodeAvailabilityEntry(node_id=node_id)
                self._cache[node_id] = entry

            entry.mark_available(source)

            if was_unavailable:
                logger.info(f"[NodeAvailabilityCache] Node {node_id} marked AVAILABLE via {source}")
                self._notify_available(node_id)

    def mark_unavailable(
        self,
        node_id: str,
        reason: AvailabilityReason,
        source: str = "unknown",
        error_message: str | None = None,
    ) -> None:
        """Mark a node as unavailable.

        Args:
            node_id: The node ID
            reason: Reason for unavailability
            source: Source of the unavailability update
            error_message: Optional error message for diagnostics
        """
        with self._lock:
            entry = self._cache.get(node_id)
            was_available = entry is None or entry.is_available

            if entry is None:
                entry = NodeAvailabilityEntry(node_id=node_id)
                self._cache[node_id] = entry

            entry.mark_unavailable(reason, source, error_message)

            if was_available:
                logger.warning(
                    f"[NodeAvailabilityCache] Node {node_id} marked UNAVAILABLE: "
                    f"{reason.value} via {source}"
                    + (f" - {error_message}" if error_message else "")
                )
                self._notify_unavailable(node_id, reason)

    def get_available_nodes(self) -> list[str]:
        """Get list of all available node IDs."""
        with self._lock:
            return [
                node_id for node_id, entry in self._cache.items()
                if entry.is_available
            ]

    def get_unavailable_nodes(self) -> list[str]:
        """Get list of all unavailable node IDs."""
        with self._lock:
            return [
                node_id for node_id, entry in self._cache.items()
                if not entry.is_available
            ]

    def get_entry(self, node_id: str) -> NodeAvailabilityEntry | None:
        """Get availability entry for a node."""
        with self._lock:
            return self._cache.get(node_id)

    def get_all_entries(self) -> dict[str, NodeAvailabilityEntry]:
        """Get copy of all entries."""
        with self._lock:
            return dict(self._cache)

    def get_status_summary(self) -> dict[str, Any]:
        """Get summary of cache status."""
        with self._lock:
            available = sum(1 for e in self._cache.values() if e.is_available)
            unavailable = len(self._cache) - available

            by_reason: dict[str, int] = {}
            for entry in self._cache.values():
                if not entry.is_available:
                    reason_name = entry.reason.value
                    by_reason[reason_name] = by_reason.get(reason_name, 0) + 1

            return {
                "total_nodes": len(self._cache),
                "available": available,
                "unavailable": unavailable,
                "by_reason": by_reason,
                "event_subscribed": self._event_subscribed,
            }

    def clear_stale_entries(self) -> int:
        """Remove stale entries from cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            stale_nodes = [
                node_id for node_id, entry in self._cache.items()
                if entry.is_stale(self.stale_threshold_seconds)
            ]
            for node_id in stale_nodes:
                del self._cache[node_id]

            if stale_nodes:
                logger.debug(f"[NodeAvailabilityCache] Cleared {len(stale_nodes)} stale entries")

            return len(stale_nodes)

    def on_unavailable(self, callback: Callable[[str, AvailabilityReason], None]) -> None:
        """Register callback for when a node becomes unavailable."""
        self._on_unavailable_callbacks.append(callback)

    def on_available(self, callback: Callable[[str], None]) -> None:
        """Register callback for when a node becomes available."""
        self._on_available_callbacks.append(callback)

    def _notify_unavailable(self, node_id: str, reason: AvailabilityReason) -> None:
        """Notify callbacks that a node became unavailable."""
        for callback in self._on_unavailable_callbacks:
            try:
                callback(node_id, reason)
            except Exception as e:
                logger.error(f"[NodeAvailabilityCache] Callback error: {e}")

    def _notify_available(self, node_id: str) -> None:
        """Notify callbacks that a node became available."""
        for callback in self._on_available_callbacks:
            try:
                callback(node_id)
            except Exception as e:
                logger.error(f"[NodeAvailabilityCache] Callback error: {e}")

    def health_check(self) -> "HealthCheckResult":
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult with healthy status, message, and details
        """
        # Import from contracts (zero dependencies)
        from app.coordination.contracts import HealthCheckResult

        summary = self.get_status_summary()
        total = summary["total_nodes"]
        available = summary["available"]
        unavailable = summary["unavailable"]

        # Healthy if more than 50% of known nodes are available
        healthy = total == 0 or (available / total) >= 0.5

        if total == 0:
            message = "No nodes tracked yet"
        elif unavailable == 0:
            message = f"All {total} nodes available"
        else:
            message = f"{available}/{total} nodes available"

        return HealthCheckResult(
            healthy=healthy,
            message=message,
            details={
                "total_nodes": total,
                "available": available,
                "unavailable": unavailable,
                "by_reason": summary["by_reason"],
                "event_subscribed": summary["event_subscribed"],
            },
        )


# Module-level singleton accessor
def get_availability_cache() -> NodeAvailabilityCache:
    """Get the singleton NodeAvailabilityCache instance."""
    return NodeAvailabilityCache.get_instance()
