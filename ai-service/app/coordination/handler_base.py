"""Handler Base Class for RingRift Coordination Daemons.

Provides a unified base class for handlers/daemons with common patterns:
- Event subscription via EventSubscribingDaemonMixin
- Safe event emission via SafeEventEmitterMixin
- Singleton pattern with thread-safe access
- Standardized health check format (HealthCheckResult)
- Event deduplication (hash-based)
- Error tracking with bounded log
- Lifecycle management (start/stop/shutdown)
- Fire-and-forget task helpers (January 2026)

This class consolidates patterns found in 15+ daemon files to reduce duplication.

Usage:
    from app.coordination.handler_base import HandlerBase, HealthCheckResult

    class MyDaemon(HandlerBase):
        _event_source = "MyDaemon"  # For safe event emission

        def __init__(self, config: Optional[MyConfig] = None):
            super().__init__(name="my_daemon", config=config)

        def _get_event_subscriptions(self) -> dict[str, Callable]:
            return {
                "training_completed": self._on_training_completed,
            }

        async def _run_cycle(self) -> None:
            # Main work loop
            # Use safe event emission:
            await self._safe_emit_event_async("MY_EVENT", {"key": "value"})

            # Fire-and-forget async operations:
            self._safe_create_task(
                self._background_operation(),
                context="background_op"
            )

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
December 30, 2025 - Added SafeEventEmitterMixin for unified event emission.
January 3, 2026 - Added fire-and-forget task helpers (_safe_create_task, _try_emit_event).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

# Import canonical types from contracts.py (single source of truth)
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.event_utils import make_config_key
from app.coordination.safe_event_emitter import SafeEventEmitterMixin

logger = logging.getLogger(__name__)


@dataclass
class HandlerStats:
    """Unified stats tracking for handlers.

    December 2025: Consolidated from handler_base.py and base_handler.py.
    This is the canonical version with all fields.
    """

    # Event processing
    events_processed: int = 0
    events_deduplicated: int = 0
    success_count: int = 0  # From base_handler.py

    # Error tracking
    errors_count: int = 0
    last_error: str = ""
    last_error_time: float = 0.0

    # Lifecycle tracking
    cycles_completed: int = 0
    started_at: float = 0.0
    last_activity: float = field(default_factory=time.time)
    last_event_time: float = 0.0  # From base_handler.py

    # State
    subscribed: bool = False  # From base_handler.py
    custom_stats: dict[str, Any] = field(default_factory=dict)  # From base_handler.py

    @property
    def error_count(self) -> int:
        """Backward-compat alias for errors_count (base_handler.py naming)."""
        return self.errors_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0). From base_handler.py."""
        if self.events_processed == 0:
            return 1.0
        return self.success_count / self.events_processed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization. From base_handler.py."""
        result = {
            "subscribed": self.subscribed,
            "events_processed": self.events_processed,
            "events_deduplicated": self.events_deduplicated,
            "success_count": self.success_count,
            "errors_count": self.errors_count,
            "success_rate": round(self.success_rate, 3),
            "cycles_completed": self.cycles_completed,
            "last_event_time": self.last_event_time,
            "last_error_time": self.last_error_time,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "last_activity": self.last_activity,
        }
        result.update(self.custom_stats)
        return result


class HandlerBase(SafeEventEmitterMixin, ABC):
    """Base class for daemon handlers with common patterns.

    Provides:
    - Singleton management with thread-safe access
    - Event subscription infrastructure
    - Safe event emission (via SafeEventEmitterMixin)
    - Event deduplication (hash-based with TTL)
    - Standardized health check
    - Error tracking with bounded log
    - Async lifecycle management

    Event Emission:
        Subclasses inherit _safe_emit_event() and _safe_emit_event_async()
        from SafeEventEmitterMixin. Set _event_source class attribute to
        identify the source in emitted events.

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

        # MultiEventHandler compatibility (backward-compat for base_handler.py)
        self._event_handlers: dict[str, Callable[[Any], Any]] = {}

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
    def handler_name(self) -> str:
        """Backward-compat alias for name (base_handler.py naming)."""
        return self._name

    @property
    def is_running(self) -> bool:
        """Check if handler is running."""
        return self._running

    @property
    def is_subscribed(self) -> bool:
        """Backward-compat alias for _event_subscribed (base_handler.py naming)."""
        return self._event_subscribed

    @property
    def emit_metrics(self) -> bool:
        """Backward-compat property (base_handler.py naming). Always True."""
        return True

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds since handler started."""
        if self._stats.started_at:
            return time.time() - self._stats.started_at
        return 0.0

    @property
    def stats(self) -> HandlerStats:
        """Get handler stats."""
        return self._stats

    # =========================================================================
    # Backward-Compatible Methods (base_handler.py API)
    # =========================================================================

    def _record_success(self) -> None:
        """Record a successful event processing (backward-compat)."""
        self._stats.events_processed += 1
        self._stats.success_count += 1
        self._stats.last_activity = time.time()

    def get_stats(self) -> dict[str, Any]:
        """Get stats as dict (backward-compat for base_handler.py API).

        Returns:
            Dictionary with handler statistics
        """
        result = {
            "subscribed": self._event_subscribed,
            "events_processed": self._stats.events_processed,
            "success_count": self._stats.success_count,
            "errors_count": self._stats.errors_count,
            "success_rate": self._stats.success_rate,
            "last_event_time": self._stats.last_activity,
            "last_error": self._stats.last_error,
        }
        # Add custom stats
        if hasattr(self._stats, "custom_stats"):
            result.update(self._stats.custom_stats)
        return result

    def subscribe(self) -> bool:
        """Subscribe to events (backward-compat for base_handler.py API).

        Returns:
            True if subscribed successfully
        """
        if self._event_subscribed:
            return True
        if hasattr(self, "_do_subscribe"):
            result = self._do_subscribe()
            if result:
                self._event_subscribed = True
                self._stats.subscribed = True
            return result
        # Default: just mark as subscribed
        self._event_subscribed = True
        self._stats.subscribed = True
        return True

    def unsubscribe(self) -> None:
        """Unsubscribe from events (backward-compat for base_handler.py API)."""
        self._event_subscribed = False
        self._stats.subscribed = False
        if hasattr(self, "_subscribed"):
            self._subscribed = False

    def _get_payload(self, event: Any) -> dict[str, Any]:
        """Extract payload from event with fallback chain.

        December 30, 2025: Enhanced to check multiple attribute names
        for compatibility with different event formats across the codebase.

        Args:
            event: Event object or dict

        Returns:
            Payload dictionary (empty dict if extraction fails)
        """
        if hasattr(event, "payload"):
            return event.payload
        if hasattr(event, "metadata"):
            return event.metadata
        if isinstance(event, dict):
            return event
        return {}

    def _extract_model_path(self, payload: dict[str, Any]) -> str | None:
        """Extract model path from event payload with fallback chain.

        December 30, 2025: Consolidates model path extraction pattern
        found in 20+ handlers. Checks multiple field names in priority order.

        Args:
            payload: Event payload dictionary

        Returns:
            Model path string, or None if not found
        """
        return (
            payload.get("checkpoint_path")
            or payload.get("model_path")
            or payload.get("model_id")
        )

    def _extract_config_key(self, payload: dict[str, Any]) -> str:
        """Extract config key from event payload.

        December 30, 2025: Consolidates config key extraction pattern.
        Constructs from board_type/num_players if config_key not present.

        Args:
            payload: Event payload dictionary

        Returns:
            Config key (e.g., "hex8_2p") or "unknown" if not found
        """
        if "config_key" in payload:
            return payload["config_key"]
        board_type = payload.get("board_type")
        num_players = payload.get("num_players")
        if board_type and num_players:
            return make_config_key(board_type, num_players)
        return "unknown"

    def _extract_board_config(self, payload: dict[str, Any]) -> tuple[str, int]:
        """Extract board type and num players from event payload.

        December 30, 2025: Consolidates board config extraction pattern.

        Args:
            payload: Event payload dictionary

        Returns:
            Tuple of (board_type, num_players) with defaults if not found
        """
        board_type = payload.get("board_type", "unknown")
        num_players = payload.get("num_players", 2)
        return (board_type, num_players)

    def _normalize_event_payload(self, event: Any) -> dict[str, Any]:
        """Normalize event to a consistent payload dictionary.

        January 4, 2026 (Sprint 17.2): Consolidates event normalization
        patterns found across 15+ handlers. Handles multiple event formats:
        - Objects with .payload attribute
        - Objects with .data attribute
        - Objects with .metadata attribute
        - Plain dictionaries
        - Dataclass instances

        Args:
            event: Event object in any supported format

        Returns:
            Normalized payload dictionary
        """
        # Try common attribute patterns
        if hasattr(event, "payload") and isinstance(event.payload, dict):
            return event.payload
        if hasattr(event, "data") and isinstance(event.data, dict):
            return event.data
        if hasattr(event, "metadata") and isinstance(event.metadata, dict):
            return event.metadata

        # Handle plain dictionaries
        if isinstance(event, dict):
            return event

        # Handle dataclass instances
        if hasattr(event, "__dataclass_fields__"):
            from dataclasses import asdict
            return asdict(event)

        # Fallback: empty dict
        return {}

    def _extract_event_fields(
        self,
        event: Any,
        fields: list[str],
        defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract specific fields from an event with defaults.

        January 4, 2026 (Sprint 17.2): Consolidates field extraction
        patterns that were duplicated across daemons. Provides a
        type-safe way to extract multiple fields at once.

        Args:
            event: Event object in any supported format
            fields: List of field names to extract
            defaults: Optional dict of default values for missing fields

        Returns:
            Dictionary with extracted field values

        Example:
            >>> data = self._extract_event_fields(
            ...     event,
            ...     ["config_key", "model_path", "elo"],
            ...     defaults={"elo": 1200}
            ... )
            >>> print(data["config_key"])  # Extracted or None
            >>> print(data["elo"])  # Extracted or 1200
        """
        defaults = defaults or {}
        payload = self._normalize_event_payload(event)
        result: dict[str, Any] = {}

        for field in fields:
            if field in payload:
                result[field] = payload[field]
            elif field in defaults:
                result[field] = defaults[field]
            else:
                result[field] = None

        return result

    # -------------------------------------------------------------------------
    # Staleness Check Helpers (Sprint 17.2 - January 4, 2026)
    # -------------------------------------------------------------------------

    def _is_stale(self, timestamp: float, threshold_seconds: float) -> bool:
        """Check if a timestamp is older than the threshold.

        January 4, 2026 (Sprint 17.2): Consolidates staleness checks
        found in p2p_recovery_daemon, training_data_sync_daemon, s3_backup_daemon.

        Args:
            timestamp: Unix timestamp to check
            threshold_seconds: Maximum age in seconds before considered stale

        Returns:
            True if timestamp is stale (older than threshold)
        """
        return time.time() - timestamp > threshold_seconds

    def _get_staleness_ratio(
        self,
        timestamp: float,
        threshold_seconds: float,
    ) -> float:
        """Get how stale a timestamp is as a ratio of the threshold.

        Useful for progressive degradation based on staleness.

        Args:
            timestamp: Unix timestamp to check
            threshold_seconds: Threshold in seconds

        Returns:
            Ratio >= 0.0. Value > 1.0 means stale, < 1.0 means fresh.
            Example: 0.5 = half as old as threshold, 2.0 = twice as old.
        """
        if threshold_seconds <= 0:
            return 0.0
        age = time.time() - timestamp
        return max(0.0, age / threshold_seconds)

    def _get_age_seconds(self, timestamp: float) -> float:
        """Get the age of a timestamp in seconds.

        Args:
            timestamp: Unix timestamp

        Returns:
            Age in seconds (time.time() - timestamp)
        """
        return time.time() - timestamp

    def _get_age_hours(self, timestamp: float) -> float:
        """Get the age of a timestamp in hours.

        Args:
            timestamp: Unix timestamp

        Returns:
            Age in hours
        """
        return (time.time() - timestamp) / 3600.0

    def add_custom_stat(self, key: str, value: Any) -> None:
        """Add a custom stat (backward-compat for base_handler.py API).

        Args:
            key: Stat key
            value: Stat value
        """
        if not hasattr(self._stats, "custom_stats"):
            self._stats.custom_stats = {}
        self._stats.custom_stats[key] = value

    def reset(self) -> None:
        """Reset stats counters (backward-compat for base_handler.py API).

        Preserves subscription state.
        """
        # Preserve both subscription flags (old and new API)
        was_subscribed = self._event_subscribed
        if hasattr(self, "_subscribed"):
            was_subscribed = was_subscribed or self._subscribed
        self._stats.events_processed = 0
        self._stats.success_count = 0
        self._stats.errors_count = 0
        self._stats.last_error = ""
        self._stats.last_activity = 0.0
        if hasattr(self._stats, "custom_stats"):
            self._stats.custom_stats = {}
        # Restore subscription state
        self._event_subscribed = was_subscribed
        self._stats.subscribed = was_subscribed

    async def _handle_event(self, event: dict[str, Any]) -> None:
        """Route event to registered handler (MultiEventHandler compatibility).

        Args:
            event: Event dict with 'type' key for routing
        """
        # Jan 2026: Handle both RouterEvent and dict types
        event_type = event.event_type if hasattr(event, "event_type") else event.get("type", "")
        handler = self._event_handlers.get(event_type)
        if handler:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)

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
    def reset_singleton(cls) -> None:
        """Alias for reset_instance() (backward-compat for base_handler.py API)."""
        cls.reset_instance()

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

    async def wait_until_stopped(self) -> None:
        """Wait until the handler's main loop task completes.

        Used by daemon runners to keep the runner alive while the handler
        is running its internal cycle loop.
        """
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def shutdown(self) -> None:
        """Alias for stop() for compatibility."""
        await self.stop()

    async def _on_start(self) -> None:
        """Hook called before main loop starts. Override for initialization.

        This method is called once during start(), after event subscriptions
        are set up but before the main _run_cycle loop begins.

        Use Cases:
            - Load persistent state from disk/database
            - Initialize connections (database, network, etc.)
            - Emit startup events
            - Validate configuration

        Error Handling:
            - Exceptions here will prevent the handler from starting
            - If an error occurs, _on_stop() will NOT be called
            - Use try/finally patterns for cleanup if needed

        Example:
            async def _on_start(self) -> None:
                self._db = await self._connect_database()
                self._state = await self._load_state()
                logger.info(f"[{self.name}] Loaded {len(self._state)} items")
        """
        pass

    async def _on_stop(self) -> None:
        """Hook called after main loop stops. Override for cleanup.

        This method is called once during stop(), after the main _run_cycle
        loop has been cancelled and all pending work is complete.

        Use Cases:
            - Save persistent state to disk/database
            - Close connections cleanly
            - Emit shutdown events
            - Flush queues or buffers

        Error Handling:
            - Exceptions here are caught and logged (won't prevent stop)
            - Handler will be marked as stopped even if _on_stop fails
            - Use best-effort cleanup patterns

        Example:
            async def _on_stop(self) -> None:
                try:
                    await self._save_state()
                    await self._db.close()
                except Exception as e:
                    logger.warning(f"[{self.name}] Cleanup error: {e}")
                logger.info(f"[{self.name}] Shutdown complete")
        """
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
        """Main work loop iteration. Must be implemented by subclasses.

        This method is called periodically (every `cycle_interval` seconds)
        by the main loop. It should perform one iteration of the handler's
        core work.

        Error Handling:
            - Exceptions are caught and logged by the main loop
            - The cycle count is only incremented on success
            - After an error, the handler continues running (no crash)
            - Errors are recorded via _record_error() for health checks

        Cancellation:
            - asyncio.CancelledError is caught and used to stop the loop
            - Do NOT catch CancelledError within this method
            - Use `self._running` check for graceful shutdown awareness

        Guidelines:
            - Keep cycles short (< 30 seconds recommended)
            - Use asyncio.sleep for delays, not time.sleep
            - Access shared state atomically or use locks
            - Emit events for significant state changes
            - Update self._stats.last_activity for health monitoring

        Example:
            async def _run_cycle(self) -> None:
                # Check for pending work
                items = await self._get_pending_items()
                for item in items:
                    if not self._running:  # Graceful shutdown
                        break
                    await self._process_item(item)
        """
        pass

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Get handler health status. Override for custom health logic.

        The default implementation checks:
        1. Running state: Returns unhealthy if handler is stopped
        2. Error rate: Calculated as errors / (cycles + events)
           - > 50% error rate: UNHEALTHY (healthy=False)
           - > 20% error rate: DEGRADED (healthy=True)
           - <= 20% error rate: RUNNING (healthy=True)

        Returns:
            HealthCheckResult with fields:
            - healthy (bool): Overall health flag for DaemonManager
            - status (CoordinatorStatus): RUNNING, DEGRADED, ERROR, STOPPED
            - message (str): Human-readable status description
            - details (dict): Metrics including uptime, cycles, errors

        Health Details Include:
            - name: Handler name
            - running: Current running state
            - uptime_seconds: Time since start
            - cycles_completed: Number of _run_cycle completions
            - events_processed: Number of events handled
            - events_deduplicated: Number of duplicate events skipped
            - errors_count: Total error count
            - last_error: Most recent error message
            - subscription_count: Number of event subscriptions

        Override Guidelines:
            - Call super().health_check() to get base metrics
            - Add domain-specific health checks (e.g., queue depth)
            - Return DEGRADED instead of ERROR for recoverable issues
            - Include actionable information in the message

        Example:
            def health_check(self) -> HealthCheckResult:
                base = super().health_check()
                if not base.healthy:
                    return base
                # Add custom check
                if self._queue_depth > 1000:
                    return HealthCheckResult(
                        healthy=True,
                        status=CoordinatorStatus.DEGRADED,
                        message=f"Queue depth high: {self._queue_depth}",
                        details={**base.details, "queue_depth": self._queue_depth},
                    )
                return base
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
        """Return event_type -> handler mapping. Override to subscribe to events.

        This method is called once during start() to set up event subscriptions.
        Each returned mapping creates a subscription to the UnifiedEventRouter.

        Returns:
            dict mapping event type names to async handler coroutines.
            Handler signature: async def handler(event: dict) -> None

        Event Types:
            Use lowercase snake_case event names matching DataEventType enum:
            - "training_completed", "evaluation_completed", "model_promoted"
            - "data_sync_completed", "regression_detected", "node_recovered"
            See app/distributed/data_events.py for full list (118 types).

        Event Deduplication:
            If self._dedup_enabled is True (default), use _is_duplicate_event()
            in handlers to skip duplicate events:

            async def _on_training_completed(self, event: dict) -> None:
                if self._is_duplicate_event(event):
                    return  # Skip - already processed this event
                # Process unique event...

        Handler Guidelines:
            - Handlers must be async coroutines
            - Keep handlers short (<5 seconds) to avoid blocking other events
            - Catch and log exceptions - don't let handlers crash
            - Use _record_success() and _record_error() for metrics

        Backward Compatibility:
            - Legacy _get_subscriptions() is still supported with deprecation warning
            - New code should use _get_event_subscriptions() (no leading underscore)

        Example:
            def _get_event_subscriptions(self) -> dict:
                return {
                    "training_completed": self._on_training_completed,
                    "model_promoted": self._on_model_promoted,
                }

            async def _on_training_completed(self, event: dict) -> None:
                if self._is_duplicate_event(event):
                    return
                config_key = event.get("config_key", "")
                logger.info(f"Training completed for {config_key}")
                self._record_success()
        """
        # Check if subclass overrode legacy _get_subscriptions method
        if "_get_subscriptions" in type(self).__dict__:
            legacy_subs = type(self)._get_subscriptions(self)
            if legacy_subs:
                logger.warning(
                    f"{type(self).__name__}._get_subscriptions() is deprecated. "
                    f"Use _get_event_subscriptions() instead. "
                    f"This will be removed in Q2 2026."
                )
                return legacy_subs
        return {}

    def _get_subscriptions(self) -> dict[str, Callable[[dict[str, Any]], Any]]:
        """Legacy method name - deprecated, use _get_event_subscriptions().

        DEPRECATED: This method exists only for backward compatibility.
        Override _get_event_subscriptions() in new code instead.
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
            event: Event payload or RouterEvent object
            key_fields: Optional list of fields to include in hash (default: all)

        Returns:
            True if duplicate (should skip), False if new
        """
        if not self._dedup_enabled:
            return False

        # January 2026: Extract payload from RouterEvent objects
        # Event may be a RouterEvent object or a dict
        if hasattr(event, 'payload'):
            payload = event.payload if isinstance(event.payload, dict) else {}
        elif isinstance(event, dict):
            payload = event
        else:
            payload = {}

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
        self._stats.events_processed += 1  # Backward-compat: errors count as processed
        self._stats.errors_count += 1
        self._stats.last_error = error
        self._stats.last_error_time = time.time()
        self._stats.last_activity = time.time()

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

    # =========================================================================
    # Fire-and-Forget Task Helpers (January 2026)
    # =========================================================================

    def _handle_task_error(self, task: asyncio.Task[Any], context: str = "") -> None:
        """Handle errors from fire-and-forget tasks.

        Attach this as a done callback to async tasks that should not crash
        the handler if they fail. Errors are logged and recorded in stats.

        January 2026: Consolidated from 10+ handlers that had identical patterns.

        Args:
            task: The completed asyncio.Task
            context: Optional context string for error message (e.g., "emit_event")

        Usage:
            task = asyncio.create_task(some_coroutine())
            task.add_done_callback(lambda t: self._handle_task_error(t, "my_op"))
        """
        try:
            exc = task.exception()
            if exc is not None:
                ctx = f" in {context}" if context else ""
                error_msg = f"Task error{ctx}: {exc}"
                self._record_error(error_msg, exc)
                logger.error(f"[{self._name}] {error_msg}")
        except asyncio.CancelledError:
            # Task was cancelled, not an error
            pass
        except asyncio.InvalidStateError:
            # Task not done yet (shouldn't happen in done callback)
            pass

    def _safe_create_task(
        self,
        coro: Any,
        context: str = "",
        *,
        name: str | None = None,
    ) -> asyncio.Task[Any] | None:
        """Create an asyncio task with error handling callback.

        Use this for fire-and-forget operations that should not block the
        caller. The task runs in the background; errors are logged but don't
        crash the handler.

        January 2026: Consolidated from 10+ handlers with identical patterns.

        Args:
            coro: Coroutine to run as a task
            context: Optional context string for error messages
            name: Optional task name for debugging

        Returns:
            The created Task, or None if task creation failed

        Usage:
            # Fire-and-forget event emission
            self._safe_create_task(
                self._emit_my_event(payload),
                context="emit_my_event"
            )

            # With task name for debugging
            self._safe_create_task(
                self._process_async(),
                context="process",
                name="my_handler_process"
            )

        Error Handling:
            - RuntimeError: Caught if event loop is closed
            - Other errors: Caught by done callback, logged and recorded
        """
        try:
            task = asyncio.create_task(coro, name=name)
            task.add_done_callback(lambda t: self._handle_task_error(t, context))
            return task
        except RuntimeError as e:
            # Event loop is closed or not running
            ctx = f" for {context}" if context else ""
            logger.debug(f"[{self._name}] Could not create task{ctx}: {e}")
            return None

    def _try_emit_event(
        self,
        event_name: str,
        payload: dict[str, Any],
        emitter_fn: Callable[..., Any] | None,
        context: str = "",
    ) -> bool:
        """Try to emit an event with graceful fallback if emitter unavailable.

        Use this when calling optional event emitter functions that may not
        be available (e.g., due to import errors or feature flags).

        January 2026: Consolidated pattern for handlers with optional dependencies.

        Args:
            event_name: Event type name for logging
            payload: Event payload dictionary
            emitter_fn: Event emitter function (may be None)
            context: Optional context for error messages

        Returns:
            True if event was emitted successfully, False otherwise

        Usage:
            # With optional emitter
            try:
                from app.coordination.event_router import emit_exploration_boost
                HAS_EXPLORATION_EVENTS = True
            except ImportError:
                emit_exploration_boost = None
                HAS_EXPLORATION_EVENTS = False

            # Later in handler:
            if HAS_EXPLORATION_EVENTS:
                self._try_emit_event(
                    "EXPLORATION_BOOST",
                    {"config_key": config_key, "boost": 1.5},
                    emit_exploration_boost,
                    context="exploration_boost"
                )
        """
        if emitter_fn is None:
            logger.debug(f"[{self._name}] Emitter not available for {event_name}")
            return False

        try:
            result = emitter_fn(**payload)
            if asyncio.iscoroutine(result):
                self._safe_create_task(result, context=context or event_name)
            return True
        except (RuntimeError, asyncio.CancelledError) as e:
            ctx = f" ({context})" if context else ""
            logger.debug(f"[{self._name}] Failed to emit {event_name}{ctx}: {e}")
            return False
        except (AttributeError, TypeError) as e:
            ctx = f" ({context})" if context else ""
            logger.warning(f"[{self._name}] Error emitting {event_name}{ctx}: {e}")
            return False

    # -------------------------------------------------------------------------
    # Thread-Safe Queue Helpers (Sprint 17.2 - January 4, 2026)
    # -------------------------------------------------------------------------

    def _append_to_queue(
        self,
        queue: list,
        item: Any,
        lock: threading.Lock | None = None,
    ) -> None:
        """Append an item to a queue with optional thread-safety.

        January 4, 2026 (Sprint 17.2): Consolidates queue append patterns
        found in s3_backup_daemon, unified_idle_shutdown_daemon, and others.

        Args:
            queue: The list to append to
            item: Item to append
            lock: Optional threading.Lock for thread-safe access
        """
        if lock:
            with lock:
                queue.append(item)
        else:
            queue.append(item)

    def _pop_queue_copy(
        self,
        queue: list,
        lock: threading.Lock | None = None,
        clear: bool = True,
    ) -> list:
        """Get a copy of queue contents, optionally clearing the original.

        January 4, 2026 (Sprint 17.2): Consolidates the pattern of
        copying queue contents under lock and clearing the original.

        Args:
            queue: The list to copy from
            lock: Optional threading.Lock for thread-safe access
            clear: If True, clear the original queue after copying

        Returns:
            Copy of queue contents
        """
        if lock:
            with lock:
                items = list(queue)
                if clear:
                    queue.clear()
                return items
        else:
            items = list(queue)
            if clear:
                queue.clear()
            return items

    def _get_queue_length(
        self,
        queue: list,
        lock: threading.Lock | None = None,
    ) -> int:
        """Get the length of a queue with optional thread-safety.

        Args:
            queue: The list to measure
            lock: Optional threading.Lock for thread-safe access

        Returns:
            Length of the queue
        """
        if lock:
            with lock:
                return len(queue)
        else:
            return len(queue)

    # -------------------------------------------------------------------------
    # Retry Queue Helpers (Sprint 17.5 - January 4, 2026)
    # -------------------------------------------------------------------------

    def _add_to_retry_queue(
        self,
        queue: deque,
        item: tuple,
        delay_seconds: float,
    ) -> None:
        """Add an item to a retry queue with calculated next_retry_time.

        January 4, 2026 (Sprint 17.5): Consolidates retry queue pattern from
        evaluation_daemon.py, training_trigger_daemon.py, selfplay_upload_daemon.py.

        The item should be a tuple of domain-specific data. This method appends
        the next_retry_time as the final element.

        Args:
            queue: The deque to append to (must be collections.deque)
            item: Tuple of domain-specific data (e.g., (model_path, board_type, ...))
            delay_seconds: Delay in seconds before retry should occur

        Example:
            # Domain-specific data tuple
            item = (model_path, board_type, num_players, attempts)
            self._add_to_retry_queue(self._retry_queue, item, delay=60.0)
            # Queue now contains: (model_path, board_type, num_players, attempts, next_retry_time)
        """
        next_retry_time = time.time() + delay_seconds
        queue.append((*item, next_retry_time))

    def _get_ready_retry_items(
        self,
        queue: deque,
    ) -> tuple[list[tuple], list[tuple]]:
        """Separate ready retry items from not-yet-ready items.

        January 4, 2026 (Sprint 17.5): Consolidates the pattern of iterating
        through a retry queue and separating items based on their next_retry_time.

        Items are expected to have next_retry_time as their LAST tuple element.
        Ready items are returned WITHOUT the next_retry_time element, as they
        are ready for processing.

        Args:
            queue: The deque containing retry items (will be modified)

        Returns:
            Tuple of (ready_items, remaining_items):
            - ready_items: List of tuples WITHOUT next_retry_time (ready for processing)
            - remaining_items: List of tuples WITH next_retry_time (still waiting)

        Example:
            ready, remaining = self._get_ready_retry_items(self._retry_queue)

            # Put remaining items back
            for item in remaining:
                self._retry_queue.append(item)

            # Process ready items (they don't have next_retry_time anymore)
            for model_path, board_type, num_players, attempts in ready:
                await self._process_retry(model_path, board_type, num_players, attempts)
        """
        now = time.time()
        ready_items: list[tuple] = []
        remaining_items: list[tuple] = []

        while queue:
            item = queue.popleft()
            # Last element is next_retry_time
            next_retry_time = item[-1]
            data_without_time = item[:-1]

            if next_retry_time <= now:
                ready_items.append(data_without_time)
            else:
                remaining_items.append(item)  # Keep full tuple with time

        return ready_items, remaining_items

    def _process_retry_queue_items(
        self,
        queue: deque,
    ) -> list[tuple]:
        """Process retry queue and return items ready for retry.

        January 4, 2026 (Sprint 17.5): Convenience method that handles the
        common pattern of separating ready items and putting remaining back.

        This is a simpler API than _get_ready_retry_items() when you don't
        need access to the remaining items.

        Args:
            queue: The deque containing retry items

        Returns:
            List of tuples WITHOUT next_retry_time (ready for processing)

        Example:
            for model_path, board_type, num_players, attempts in self._process_retry_queue_items(self._retry_queue):
                await self._process_retry(model_path, board_type, num_players, attempts)
        """
        ready_items, remaining_items = self._get_ready_retry_items(queue)

        # Put remaining items back
        for item in remaining_items:
            queue.append(item)

        return ready_items

    # -------------------------------------------------------------------------
    # SQLite Async Safety Helpers (Sprint 17.3 - January 4, 2026)
    # -------------------------------------------------------------------------

    async def _sqlite_query(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a blocking SQLite read operation in thread pool.

        January 4, 2026 (Sprint 17.3): Provides async-safe SQLite query execution.
        Wraps blocking sqlite3 operations in asyncio.to_thread() to prevent
        event loop blocking.

        Args:
            func: Callable that performs the SQLite operation (e.g., cursor.execute)
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result of the function call

        Example:
            # Before (blocking):
            cursor.execute("SELECT * FROM games WHERE config_key = ?", (key,))
            rows = cursor.fetchall()

            # After (async-safe):
            rows = await self._sqlite_query(
                lambda: cursor.execute("SELECT * FROM games WHERE config_key = ?", (key,)).fetchall()
            )

            # Or with connection:
            async def get_games(self, config_key: str) -> list:
                def _query():
                    conn = sqlite3.connect(self._db_path)
                    try:
                        cursor = conn.execute(
                            "SELECT * FROM games WHERE config_key = ?",
                            (config_key,)
                        )
                        return cursor.fetchall()
                    finally:
                        conn.close()
                return await self._sqlite_query(_query)
        """
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _sqlite_execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a blocking SQLite write operation in thread pool.

        January 4, 2026 (Sprint 17.3): Provides async-safe SQLite write execution.
        Alias for _sqlite_query() but semantically indicates a write operation.
        Use this for INSERT, UPDATE, DELETE, and other modifying operations.

        Args:
            func: Callable that performs the SQLite operation
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result of the function call

        Example:
            # Insert with commit:
            async def save_result(self, config_key: str, elo: int) -> None:
                def _insert():
                    conn = sqlite3.connect(self._db_path)
                    try:
                        conn.execute(
                            "INSERT INTO results (config_key, elo) VALUES (?, ?)",
                            (config_key, elo)
                        )
                        conn.commit()
                    finally:
                        conn.close()
                await self._sqlite_execute(_insert)

            # Transaction with rollback:
            async def update_batch(self, updates: list) -> int:
                def _update():
                    conn = sqlite3.connect(self._db_path)
                    try:
                        cursor = conn.cursor()
                        for item in updates:
                            cursor.execute("UPDATE ...", item)
                        conn.commit()
                        return cursor.rowcount
                    except Exception:
                        conn.rollback()
                        raise
                    finally:
                        conn.close()
                return await self._sqlite_execute(_update)
        """
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _sqlite_with_connection(
        self,
        db_path: str,
        func: Callable[[Any], Any],
        readonly: bool = False,
    ) -> Any:
        """Execute a function with a managed SQLite connection.

        January 4, 2026 (Sprint 17.3): Higher-level helper that manages
        connection lifecycle. Opens connection, executes function, handles
        commit/rollback, and closes connection - all in thread pool.

        Args:
            db_path: Path to SQLite database file
            func: Callable that takes a connection and returns a result
            readonly: If True, opens in read-only mode (no commit needed)

        Returns:
            Result of the function call

        Example:
            # Read operation:
            result = await self._sqlite_with_connection(
                "data/games.db",
                lambda conn: conn.execute("SELECT COUNT(*) FROM games").fetchone()[0],
                readonly=True
            )

            # Write operation (auto-commits):
            await self._sqlite_with_connection(
                "data/games.db",
                lambda conn: conn.execute("DELETE FROM games WHERE stale = 1")
            )
        """
        import sqlite3

        def _execute() -> Any:
            # Open connection (read-only mode if specified)
            if readonly:
                uri = f"file:{db_path}?mode=ro"
                conn = sqlite3.connect(uri, uri=True)
            else:
                conn = sqlite3.connect(db_path)

            try:
                result = func(conn)
                if not readonly:
                    conn.commit()
                return result
            except Exception:
                if not readonly:
                    conn.rollback()
                raise
            finally:
                conn.close()

        return await asyncio.to_thread(_execute)


# =============================================================================
# Backward-Compatible Config (from base_event_handler.py)
# =============================================================================


class EventHandlerConfig:
    """Configuration for event handlers.

    Override in subclasses for custom behavior.

    Consolidated from base_event_handler.py (December 2025).
    """

    # Whether to register with coordinator registry
    register_with_registry: bool = True

    # Whether handler methods are async (True) or sync (False)
    async_handlers: bool = True

    # Whether to use fire_and_forget for sync event callbacks
    use_fire_and_forget: bool = True

    # Timeout for handler execution (0 = no timeout)
    handler_timeout_seconds: float = 0.0


# =============================================================================
# Backward-Compatible Helper Functions (base_handler.py API)
# =============================================================================


def create_handler_stats(**custom_stats: Any) -> HandlerStats:
    """Factory function to create HandlerStats with custom stats.

    Args:
        **custom_stats: Custom statistics to include

    Returns:
        HandlerStats instance with custom_stats populated
    """
    stats = HandlerStats()
    stats.custom_stats = custom_stats
    return stats


def safe_subscribe(handler: HandlerBase, fallback: bool = True) -> bool:
    """Safely subscribe a handler, catching any exceptions.

    Args:
        handler: Handler to subscribe
        fallback: Value to return on error

    Returns:
        True if subscribed successfully, fallback value on error
    """
    try:
        return handler.subscribe()
    except Exception as e:
        logger.warning(f"safe_subscribe failed for {type(handler).__name__}: {e}")
        return fallback


# =============================================================================
# Backward-Compatible Aliases (December 2025)
# =============================================================================

# Alias for daemons that import BaseEventHandler from base_event_handler.py
BaseEventHandler = HandlerBase

# Alias for base_handler.py imports
BaseSingletonHandler = HandlerBase
MultiEventHandler = HandlerBase


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    # Canonical exports
    "CoordinatorStatus",
    "EventHandlerConfig",
    "HandlerBase",
    "HandlerStats",
    "HealthCheckResult",
    "SafeEventEmitterMixin",
    # Backward-compatible aliases
    "BaseEventHandler",
    "BaseSingletonHandler",
    "MultiEventHandler",
    # Helper functions
    "create_handler_stats",
    "safe_subscribe",
]
