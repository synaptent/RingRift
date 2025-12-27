"""Handler Base Class for RingRift Coordination Daemons.

Provides a unified base class for handlers/daemons with common patterns:
- Event subscription via EventSubscribingDaemonMixin
- Singleton pattern with thread-safe access
- Standardized health check format (HealthCheckResult)
- Event deduplication (hash-based)
- Error tracking with bounded log
- Lifecycle management (start/stop/shutdown)

This class consolidates patterns found in 15+ daemon files to reduce duplication.

Usage:
    from app.coordination.handler_base import HandlerBase, HealthCheckResult

    class MyDaemon(HandlerBase):
        def __init__(self, config: Optional[MyConfig] = None):
            super().__init__(name="my_daemon", config=config)

        def _get_event_subscriptions(self) -> dict[str, Callable]:
            return {
                "training_completed": self._on_training_completed,
            }

        async def _run_cycle(self) -> None:
            # Main work loop
            pass

        async def _on_training_completed(self, event: dict) -> None:
            # Handle event with deduplication
            if self._is_duplicate_event(event):
                return
            # Process event
            pass

    # Get singleton
    daemon = MyDaemon.get_instance()
    await daemon.start()

December 2025 - Phase 2 handler consolidation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar

logger = logging.getLogger(__name__)


class CoordinatorStatus(str, Enum):
    """Standard daemon/coordinator status values."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass
class HealthCheckResult:
    """Standard health check result format.

    Used by DaemonManager for health monitoring.
    Compatible with health_check_orchestrator.py protocol.
    """

    healthy: bool
    status: CoordinatorStatus = CoordinatorStatus.STOPPED
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "healthy": self.healthy,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class HandlerStats:
    """Standard stats tracking for handlers."""

    events_processed: int = 0
    events_deduplicated: int = 0
    cycles_completed: int = 0
    errors_count: int = 0
    last_error: str = ""
    last_error_time: float = 0.0
    started_at: float = 0.0
    last_activity: float = field(default_factory=time.time)


class HandlerBase(ABC):
    """Base class for daemon handlers with common patterns.

    Provides:
    - Singleton management with thread-safe access
    - Event subscription infrastructure
    - Event deduplication (hash-based with TTL)
    - Standardized health check
    - Error tracking with bounded log
    - Async lifecycle management

    Subclasses must implement:
    - _run_cycle(): Main work loop iteration
    - _get_event_subscriptions(): Event handlers mapping (optional)

    Subclasses may override:
    - _on_start(): Called before main loop starts
    - _on_stop(): Called after main loop stops
    - health_check(): Custom health check logic
    """

    # Singleton storage per class
    _instances: ClassVar[dict[type, "HandlerBase"]] = {}
    _singleton_lock: ClassVar[threading.RLock] = threading.RLock()

    # Deduplication defaults
    DEDUP_TTL_SECONDS: ClassVar[float] = 300.0  # 5 minutes
    DEDUP_MAX_SIZE: ClassVar[int] = 1000  # Max cached hashes

    def __init__(
        self,
        name: str,
        config: Any | None = None,
        *,
        cycle_interval: float = 60.0,
        dedup_enabled: bool = True,
    ):
        """Initialize handler base.

        Args:
            name: Handler name for logging
            config: Optional configuration object
            cycle_interval: Seconds between _run_cycle calls
            dedup_enabled: Enable event deduplication
        """
        self._name = name
        self._config = config
        self._cycle_interval = cycle_interval
        self._dedup_enabled = dedup_enabled

        # State
        self._running = False
        self._status = CoordinatorStatus.STOPPED
        self._task: asyncio.Task[None] | None = None
        self._stats = HandlerStats()

        # Event subscription (from EventSubscribingDaemonMixin pattern)
        self._event_subscriptions: dict[str, Any] = {}
        self._event_subscribed = False
        self._event_router: Any | None = None

        # Deduplication
        self._seen_events: dict[str, float] = {}  # hash -> timestamp

        # Error log (bounded)
        self._error_log: list[dict[str, Any]] = []
        self._max_error_log = 50

    @property
    def name(self) -> str:
        """Get handler name."""
        return self._name

    @property
    def is_running(self) -> bool:
        """Check if handler is running."""
        return self._running

    @property
    def stats(self) -> HandlerStats:
        """Get handler stats."""
        return self._stats

    # =========================================================================
    # Singleton Management
    # =========================================================================

    @classmethod
    def get_instance(cls, *args: Any, **kwargs: Any) -> "HandlerBase":
        """Get or create singleton instance.

        Thread-safe lazy initialization.
        """
        with cls._singleton_lock:
            if cls not in cls._instances:
                cls._instances[cls] = cls(*args, **kwargs)
            return cls._instances[cls]

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._singleton_lock:
            if cls in cls._instances:
                instance = cls._instances.pop(cls)
                # Try to stop if running
                if hasattr(instance, "_running") and instance._running:
                    logger.warning(f"Resetting running instance: {cls.__name__}")

    @classmethod
    def has_instance(cls) -> bool:
        """Check if singleton instance exists."""
        with cls._singleton_lock:
            return cls in cls._instances

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self) -> None:
        """Start the handler.

        Subscribes to events and starts the main loop.
        """
        if self._running:
            logger.warning(f"[{self._name}] Already running")
            return

        self._running = True
        self._status = CoordinatorStatus.STARTING
        self._stats.started_at = time.time()

        # Subscribe to events
        self._subscribe_all_events()

        # Hook for subclasses
        await self._on_start()

        # Start main loop
        self._status = CoordinatorStatus.RUNNING
        self._task = asyncio.create_task(self._main_loop())
        logger.info(f"[{self._name}] Started")

    async def stop(self) -> None:
        """Stop the handler gracefully."""
        if not self._running:
            return

        self._running = False
        self._status = CoordinatorStatus.STOPPING

        # Cancel task
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Unsubscribe from events
        self._unsubscribe_all_events()

        # Hook for subclasses
        await self._on_stop()

        self._status = CoordinatorStatus.STOPPED
        self._task = None
        logger.info(f"[{self._name}] Stopped")

    async def shutdown(self) -> None:
        """Alias for stop() for compatibility."""
        await self.stop()

    async def _on_start(self) -> None:
        """Hook called before main loop starts. Override in subclass."""
        pass

    async def _on_stop(self) -> None:
        """Hook called after main loop stops. Override in subclass."""
        pass

    async def _main_loop(self) -> None:
        """Main work loop - runs _run_cycle periodically."""
        while self._running:
            try:
                await self._run_cycle()
                self._stats.cycles_completed += 1
                self._stats.last_activity = time.time()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._record_error(f"Error in _run_cycle: {e}")
                logger.exception(f"[{self._name}] Error in cycle: {e}")

            if self._running:
                await asyncio.sleep(self._cycle_interval)

    @abstractmethod
    async def _run_cycle(self) -> None:
        """Main work loop iteration. Implement in subclass."""
        pass

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Get handler health status.

        Override in subclass for custom logic.
        """
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=self._status,
                message="Handler not running",
            )

        # Check error rate
        total_ops = self._stats.cycles_completed + self._stats.events_processed
        if total_ops > 0:
            error_rate = self._stats.errors_count / total_ops
        else:
            error_rate = 0.0

        # Degraded if error rate > 20%, unhealthy if > 50%
        if error_rate > 0.5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"High error rate: {error_rate:.1%}",
                details=self._get_health_details(),
            )
        elif error_rate > 0.2:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Elevated error rate: {error_rate:.1%}",
                details=self._get_health_details(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="Operating normally",
            details=self._get_health_details(),
        )

    def _get_health_details(self) -> dict[str, Any]:
        """Get detailed health info for health check."""
        return {
            "name": self._name,
            "running": self._running,
            "uptime_seconds": time.time() - self._stats.started_at if self._stats.started_at else 0,
            "cycles_completed": self._stats.cycles_completed,
            "events_processed": self._stats.events_processed,
            "events_deduplicated": self._stats.events_deduplicated,
            "errors_count": self._stats.errors_count,
            "last_error": self._stats.last_error,
            "subscription_count": len(self._event_subscriptions),
        }

    def get_status(self) -> dict[str, Any]:
        """Get handler status dictionary."""
        health = self.health_check()
        return {
            "name": self._name,
            "running": self._running,
            "status": self._status.value,
            "health": health.to_dict(),
            "stats": {
                "events_processed": self._stats.events_processed,
                "events_deduplicated": self._stats.events_deduplicated,
                "cycles_completed": self._stats.cycles_completed,
                "errors_count": self._stats.errors_count,
            },
        }

    # =========================================================================
    # Event Subscription (from EventSubscribingDaemonMixin)
    # =========================================================================

    def _get_event_subscriptions(self) -> dict[str, Callable[[dict[str, Any]], Any]]:
        """Return event_type -> handler mapping.

        Override in subclass to specify subscriptions.

        Example:
            def _get_event_subscriptions(self) -> dict:
                return {
                    "training_completed": self._on_training_completed,
                }
        """
        return {}

    def _subscribe_all_events(self) -> bool:
        """Subscribe to all events from _get_event_subscriptions()."""
        if self._event_subscribed:
            return True

        subscriptions = self._get_event_subscriptions()
        if not subscriptions:
            return True  # No subscriptions needed

        try:
            from app.coordination.event_router import get_router

            router = get_router()
            self._event_router = router

            for event_type, handler in subscriptions.items():
                try:
                    unsub = router.subscribe(event_type, handler)
                    self._event_subscriptions[event_type] = unsub
                    logger.debug(f"[{self._name}] Subscribed to {event_type}")
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(f"[{self._name}] Failed to subscribe to {event_type}: {e}")

            self._event_subscribed = bool(self._event_subscriptions)
            if self._event_subscribed:
                logger.info(f"[{self._name}] Subscribed to {len(self._event_subscriptions)} events")
            return self._event_subscribed

        except ImportError as e:
            logger.debug(f"[{self._name}] Event router not available: {e}")
            return False
        except (RuntimeError, OSError) as e:
            logger.warning(f"[{self._name}] Event subscription failed: {e}")
            return False

    def _unsubscribe_all_events(self) -> None:
        """Unsubscribe from all events."""
        if not self._event_subscriptions:
            return

        for event_type, unsub in list(self._event_subscriptions.items()):
            try:
                if callable(unsub):
                    unsub()
            except (TypeError, RuntimeError, OSError) as e:
                logger.debug(f"[{self._name}] Error unsubscribing from {event_type}: {e}")

        self._event_subscriptions.clear()
        self._event_subscribed = False
        self._event_router = None

    # =========================================================================
    # Event Deduplication
    # =========================================================================

    def _is_duplicate_event(self, event: dict[str, Any], key_fields: list[str] | None = None) -> bool:
        """Check if event is a duplicate based on content hash.

        Args:
            event: Event payload
            key_fields: Optional list of fields to include in hash (default: all)

        Returns:
            True if duplicate (should skip), False if new
        """
        if not self._dedup_enabled:
            return False

        # Compute hash
        if key_fields:
            hash_data = {k: event.get(k) for k in key_fields}
        else:
            hash_data = event

        try:
            event_hash = hashlib.sha256(
                json.dumps(hash_data, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
        except (TypeError, ValueError):
            # Can't hash, assume not duplicate
            return False

        now = time.time()

        # Prune old entries
        if len(self._seen_events) > self.DEDUP_MAX_SIZE:
            cutoff = now - self.DEDUP_TTL_SECONDS
            self._seen_events = {
                h: t for h, t in self._seen_events.items() if t > cutoff
            }

        # Check for duplicate
        if event_hash in self._seen_events:
            last_seen = self._seen_events[event_hash]
            if now - last_seen < self.DEDUP_TTL_SECONDS:
                self._stats.events_deduplicated += 1
                return True

        # Record new event
        self._seen_events[event_hash] = now
        return False

    def _mark_event_processed(self, event: dict[str, Any]) -> None:
        """Mark event as processed, incrementing counter."""
        self._stats.events_processed += 1
        self._stats.last_activity = time.time()

    # =========================================================================
    # Error Tracking
    # =========================================================================

    def _record_error(self, error: str, exc: Exception | None = None) -> None:
        """Record an error with bounded log.

        Args:
            error: Error message
            exc: Optional exception
        """
        self._stats.errors_count += 1
        self._stats.last_error = error
        self._stats.last_error_time = time.time()

        entry = {
            "timestamp": time.time(),
            "error": error,
            "exception": str(exc) if exc else None,
        }
        self._error_log.append(entry)

        # Bound log size
        if len(self._error_log) > self._max_error_log:
            self._error_log = self._error_log[-self._max_error_log:]

    def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent errors from log."""
        return self._error_log[-limit:]


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    "CoordinatorStatus",
    "HandlerBase",
    "HandlerStats",
    "HealthCheckResult",
]
