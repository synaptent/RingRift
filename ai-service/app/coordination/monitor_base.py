"""Monitor Base Class for RingRift Health Monitoring.

This module provides a unified base class for all health monitors, consolidating
patterns found in 6+ monitor implementations to reduce ~1,150 LOC duplication.

MonitorBase combines:
- Lifecycle management from BaseDaemon (start/stop, coordinator registration)
- Event subscription from HandlerBase (deduplication, safe fallback)
- Health check logic from HealthCheckMixin (error thresholds, stale detection)
- Singleton pattern for monitor instances

December 2025 - Phase 3 consolidation.

Usage:
    from app.coordination.monitor_base import MonitorBase, MonitorConfig

    @dataclass
    class MyMonitorConfig(MonitorConfig):
        custom_setting: str = "default"

    class MyMonitor(MonitorBase[MyMonitorConfig]):
        async def _run_cycle(self) -> None:
            # Implement monitoring logic
            pass

        def _get_event_subscriptions(self) -> dict[str, Callable]:
            return {
                "some_event": self._on_some_event,
            }

    # Get singleton instance
    monitor = MyMonitor.get_instance()
    await monitor.start()

Migrated from:
- CoordinatorHealthMonitorDaemon (546 LOC -> ~300 LOC)
- QualityMonitorDaemon (602 LOC -> ~350 LOC)
- ClusterStatusMonitor (~500 LOC -> ~250 LOC)
- WorkQueueMonitorDaemon (~300 LOC -> ~150 LOC)
- DaemonWatchdog (~350 LOC -> ~180 LOC)
- HealthCheckOrchestrator (~400 LOC -> ~200 LOC)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Generic, TypeVar

from app.coordination.base_daemon import BaseDaemon, DaemonConfig
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MonitorConfig(DaemonConfig):
    """Configuration for monitors.

    Extends DaemonConfig with monitor-specific settings.

    Note: Uses kw_only=True for safe inheritance.
    """

    # Health check thresholds
    unhealthy_error_threshold: float = 0.5  # 50% error rate = unhealthy
    degraded_error_threshold: float = 0.1   # 10% error rate = degraded
    stale_threshold_seconds: float = 1800.0  # 30 minutes without activity = stale

    # Event handling
    subscribe_to_events: bool = True
    dedup_enabled: bool = True
    dedup_ttl_seconds: float = 300.0  # 5 minutes
    dedup_max_size: int = 1000  # Max cached hashes


MonitorConfigT = TypeVar("MonitorConfigT", bound=MonitorConfig)


# =============================================================================
# Monitor Stats
# =============================================================================


@dataclass
class MonitorStats:
    """Statistics tracking for monitors.

    Consolidates patterns from handler_base.py and base_daemon.py.
    """

    # Event processing
    events_processed: int = 0
    events_deduplicated: int = 0

    # Health metrics
    healthy_checks: int = 0
    unhealthy_checks: int = 0
    degraded_checks: int = 0

    # Error tracking
    errors_count: int = 0
    last_error: str = ""
    last_error_time: float = 0.0

    # Lifecycle
    cycles_completed: int = 0
    started_at: float = 0.0
    last_activity: float = field(default_factory=time.time)

    # Custom stats from subclasses
    custom: dict[str, Any] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        """Calculate error rate (0.0 to 1.0)."""
        total = self.cycles_completed
        if total == 0:
            return 0.0
        return min(1.0, self.errors_count / total)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "events_processed": self.events_processed,
            "events_deduplicated": self.events_deduplicated,
            "healthy_checks": self.healthy_checks,
            "unhealthy_checks": self.unhealthy_checks,
            "degraded_checks": self.degraded_checks,
            "errors_count": self.errors_count,
            "error_rate": round(self.error_rate, 4),
            "last_error": self.last_error,
            "last_error_time": self.last_error_time,
            "cycles_completed": self.cycles_completed,
            "started_at": self.started_at,
            "last_activity": self.last_activity,
        }
        result.update(self.custom)
        return result


# =============================================================================
# Monitor Base Class
# =============================================================================


class MonitorBase(BaseDaemon[MonitorConfigT], Generic[MonitorConfigT]):
    """Unified base class for all health monitors.

    Combines patterns from:
    - BaseDaemon: Lifecycle, coordinator registration, main loop
    - HandlerBase: Event subscription, deduplication, singleton
    - HealthCheckMixin: Error thresholds, stale detection

    Subclasses must implement:
    - _run_cycle(): The monitoring logic

    Subclasses may override:
    - _get_event_subscriptions(): Event handlers mapping
    - _get_monitor_name(): Custom monitor name
    - _get_default_config(): Default configuration
    - _on_start() / _on_stop(): Lifecycle hooks
    """

    # Singleton storage per class
    _instances: ClassVar[dict[type, "MonitorBase"]] = {}
    _singleton_lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, config: MonitorConfigT | None = None):
        """Initialize the monitor.

        Args:
            config: Monitor configuration. If None, uses default from _get_default_config().
        """
        super().__init__(config)

        # Monitor-specific state
        self._monitor_stats = MonitorStats()
        self._monitor_stats.started_at = time.time()

        # Event subscription
        self._event_subscriptions: dict[str, Callable] = {}
        self._event_subscribed = False
        self._event_router: Any | None = None

        # Deduplication
        self._seen_events: dict[str, float] = {}  # hash -> timestamp

        logger.debug(f"[{self._get_daemon_name()}] MonitorBase initialized")

    # =========================================================================
    # Singleton Management
    # =========================================================================

    @classmethod
    def get_instance(cls, config: MonitorConfigT | None = None) -> "MonitorBase[MonitorConfigT]":
        """Get or create singleton instance.

        Thread-safe lazy initialization.
        """
        with cls._singleton_lock:
            if cls not in cls._instances:
                cls._instances[cls] = cls(config)  # type: ignore
            return cls._instances[cls]  # type: ignore

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._singleton_lock:
            if cls in cls._instances:
                instance = cls._instances.pop(cls)
                if hasattr(instance, "_running") and instance._running:
                    logger.warning(f"Resetting running instance: {cls.__name__}")

    @classmethod
    def has_instance(cls) -> bool:
        """Check if singleton instance exists."""
        with cls._singleton_lock:
            return cls in cls._instances

    # =========================================================================
    # Event Subscription
    # =========================================================================

    def _subscribe_all_events(self) -> bool:
        """Subscribe to all events from _get_event_subscriptions().

        Returns:
            True if subscribed successfully, False otherwise.
        """
        if self._event_subscribed:
            return True

        if not self.config.subscribe_to_events:
            logger.debug(f"[{self._get_daemon_name()}] Event subscription disabled")
            return True

        subscriptions = self._get_event_subscriptions()
        if not subscriptions:
            logger.debug(f"[{self._get_daemon_name()}] No event subscriptions defined")
            return True

        try:
            from app.coordination.event_router import get_router

            router = get_router()
            self._event_router = router

            for event_type, handler in subscriptions.items():
                router.subscribe(event_type, handler)
                self._event_subscriptions[event_type] = handler
                logger.debug(f"[{self._get_daemon_name()}] Subscribed to: {event_type}")

            self._event_subscribed = True
            logger.info(
                f"[{self._get_daemon_name()}] Subscribed to {len(subscriptions)} events"
            )
            return True

        except ImportError as e:
            logger.warning(f"[{self._get_daemon_name()}] Event router not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"[{self._get_daemon_name()}] Event subscription failed: {e}")
            return False

    def _unsubscribe_all_events(self) -> None:
        """Unsubscribe from all events."""
        if not self._event_subscribed or not self._event_router:
            return

        try:
            for event_type in self._event_subscriptions:
                self._event_router.unsubscribe(event_type, self._event_subscriptions[event_type])
            self._event_subscriptions.clear()
            self._event_subscribed = False
            logger.debug(f"[{self._get_daemon_name()}] Unsubscribed from all events")
        except Exception as e:
            logger.warning(f"[{self._get_daemon_name()}] Event unsubscription failed: {e}")

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event handlers to subscribe to.

        Override in subclass to define event handlers.

        Returns:
            Dict mapping event type names to handler callables.
        """
        return {}

    # =========================================================================
    # Event Deduplication
    # =========================================================================

    def _is_duplicate_event(
        self,
        event: dict[str, Any],
        key_fields: list[str] | None = None,
    ) -> bool:
        """Check if event is a duplicate based on content hash.

        Args:
            event: Event dictionary or RouterEvent to check
            key_fields: Optional list of fields to use for hashing.
                       If None, uses entire event payload.

        Returns:
            True if this event has been seen recently.
        """
        if not self.config.dedup_enabled:
            return False

        # January 2026: Extract payload from RouterEvent objects
        # Event may be a RouterEvent object or a dict
        if hasattr(event, 'payload'):
            payload = event.payload if isinstance(event.payload, dict) else {}
        elif isinstance(event, dict):
            payload = event
        else:
            payload = {}

        # Clean up old entries
        now = time.time()
        cutoff = now - self.config.dedup_ttl_seconds
        self._seen_events = {
            h: t for h, t in self._seen_events.items()
            if t > cutoff
        }

        # Limit size
        if len(self._seen_events) > self.config.dedup_max_size:
            # Keep newest entries
            sorted_events = sorted(self._seen_events.items(), key=lambda x: x[1])
            self._seen_events = dict(sorted_events[-self.config.dedup_max_size // 2:])

        # Compute hash
        if key_fields:
            hash_data = {k: payload.get(k) for k in key_fields}
        else:
            hash_data = payload

        try:
            event_hash = hashlib.sha256(
                json.dumps(hash_data, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
        except (TypeError, ValueError):
            # If serialization fails, don't deduplicate
            return False

        if event_hash in self._seen_events:
            self._monitor_stats.events_deduplicated += 1
            return True

        self._seen_events[event_hash] = now
        return False

    # =========================================================================
    # Health Check (Enhanced from HealthCheckMixin)
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health status for DaemonManager integration.

        Uses config thresholds for error rate-based health determination.

        Returns:
            HealthCheckResult with status, message, and details.
        """
        # Base state check
        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Monitor is stopped",
            )

        error_rate = self._monitor_stats.error_rate
        cycles = self._monitor_stats.cycles_completed
        uptime = self.uptime_seconds

        # Build details dict
        details = {
            "cycles_completed": cycles,
            "error_rate": round(error_rate, 4),
            "uptime_seconds": round(uptime, 1),
            "events_processed": self._monitor_stats.events_processed,
            "events_deduplicated": self._monitor_stats.events_deduplicated,
            "subscriptions": len(self._event_subscriptions),
        }

        # Add custom stats if available
        details.update(self._monitor_stats.custom)

        # Check for stale activity
        last_activity = self._monitor_stats.last_activity
        time_since_activity = time.time() - last_activity
        if time_since_activity > self.config.stale_threshold_seconds:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Stale: no activity for {time_since_activity:.0f}s",
                details=details,
            )

        # Check error rate thresholds
        if error_rate > self.config.unhealthy_error_threshold:
            self._monitor_stats.unhealthy_checks += 1
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"High error rate: {error_rate:.1%}",
                details=details,
            )

        if error_rate > self.config.degraded_error_threshold:
            self._monitor_stats.degraded_checks += 1
            return HealthCheckResult(
                healthy=True,  # Still functional but degraded
                status=CoordinatorStatus.DEGRADED,
                message=f"Elevated error rate: {error_rate:.1%}",
                details=details,
            )

        self._monitor_stats.healthy_checks += 1
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="Healthy",
            details=details,
        )

    # =========================================================================
    # Lifecycle Hooks
    # =========================================================================

    async def _on_start(self) -> None:
        """Called when monitor starts. Subscribes to events."""
        self._subscribe_all_events()
        await super()._on_start()

    async def _on_stop(self) -> None:
        """Called when monitor stops. Unsubscribes from events."""
        self._unsubscribe_all_events()
        await super()._on_stop()

    # =========================================================================
    # Status and Stats
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get monitor status for reporting.

        Override in subclass to add monitor-specific status.
        """
        return {
            "name": self._get_daemon_name(),
            "running": self._running,
            "uptime_seconds": self.uptime_seconds,
            "stats": self._monitor_stats.to_dict(),
            "event_subscribed": self._event_subscribed,
            "subscription_count": len(self._event_subscriptions),
        }

    def record_event(self) -> None:
        """Record that an event was processed."""
        self._monitor_stats.events_processed += 1
        self._monitor_stats.last_activity = time.time()

    def record_error(self, error: Exception | str) -> None:
        """Record an error occurrence."""
        self._monitor_stats.errors_count += 1
        self._monitor_stats.last_error = str(error)
        self._monitor_stats.last_error_time = time.time()

    def record_cycle(self) -> None:
        """Record that a monitoring cycle completed."""
        self._monitor_stats.cycles_completed += 1
        self._monitor_stats.last_activity = time.time()

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    async def _run_cycle(self) -> None:
        """Execute one monitoring cycle.

        Must be implemented by subclasses.
        Called repeatedly at config.check_interval_seconds intervals.
        """
        raise NotImplementedError

    def _get_default_config(self) -> MonitorConfigT:
        """Return default configuration.

        Override in subclass for custom defaults.
        """
        return MonitorConfig()  # type: ignore


# =============================================================================
# Factory Functions
# =============================================================================


def get_monitor(monitor_cls: type[MonitorBase[MonitorConfigT]]) -> MonitorBase[MonitorConfigT]:
    """Get singleton instance of a monitor class.

    Convenience function for accessing monitors.

    Args:
        monitor_cls: The monitor class to get instance of.

    Returns:
        Singleton instance of the monitor.
    """
    return monitor_cls.get_instance()


def reset_all_monitors() -> None:
    """Reset all monitor singleton instances.

    Primarily for testing.
    """
    with MonitorBase._singleton_lock:
        instances = list(MonitorBase._instances.values())
        for instance in instances:
            instance.__class__.reset_instance()


__all__ = [
    "MonitorBase",
    "MonitorConfig",
    "MonitorConfigT",
    "MonitorStats",
    "get_monitor",
    "reset_all_monitors",
]
