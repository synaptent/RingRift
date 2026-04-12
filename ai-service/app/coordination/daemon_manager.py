"""Unified Daemon Manager - Coordinates lifecycle of all background services.

Provides centralized management for all daemons and background services:
- Sync daemons (data, model, elo)
- Health check services
- Event watchers
- Background pipelines

Features:
- Unified start/stop lifecycle
- Health monitoring with auto-restart
- Graceful shutdown handling
- Integration with OrchestratorRegistry
- Status reporting

Usage:
    from app.coordination.daemon_manager import (
        DaemonManager,
        get_daemon_manager,
        DaemonType,
    )

    # Get the singleton manager
    manager = get_daemon_manager()

    # Start all daemons
    await manager.start_all()

    # Start specific daemon
    await manager.start(DaemonType.AUTO_SYNC)  # Primary data sync

    # Get status
    status = manager.get_status()

    # Graceful shutdown
    await manager.shutdown()
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib

# Python 3.10 compatibility shim for asyncio.timeout (added in 3.11)
try:
    from asyncio import timeout as async_timeout
except ImportError:
    # Python 3.10 fallback using async-timeout library or simple wrapper
    try:
        from async_timeout import timeout as async_timeout
    except ImportError:
        # Minimal fallback - no actual timeout enforcement
        @contextlib.asynccontextmanager
        async def async_timeout(seconds: float):
            """Compatibility shim for asyncio.timeout (Python 3.11+)."""
            yield
import importlib
import json
import logging
import os
import signal
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiohttp import web
    from app.coordination.daemon_health_types import AnalysisResult

from app.config.coordination_defaults import (
    CircuitBreakerDefaults,
    DaemonHealthDefaults,
    DegradedModeDefaults,
)
from app.core.async_context import fire_and_forget, safe_create_task

# Singleton mixin for thread-safe singleton pattern (Dec 2025)
from app.coordination.singleton_mixin import SingletonMixin

# Hierarchical cascade circuit breaker (Dec 30, 2025)
from app.coordination.cascade_breaker import (
    CascadeBreakerManager,
    get_cascade_breaker,
)

# January 3, 2026: Unified circuit breaker base for daemon status checks
from app.coordination.circuit_breaker_base import (
    CircuitConfig,
    CircuitState,
    OperationCircuitBreaker,
)

# Daemon types extracted to dedicated module (Dec 2025)
from app.coordination.daemon_types import (
    CRITICAL_DAEMONS,
    DaemonCategory,
    DaemonInfo,
    DaemonManagerConfig,
    DaemonState,
    DaemonType,
    RestartTier,  # December 2025: Graceful degradation
    get_daemon_category,
    mark_daemon_ready,
    register_mark_ready_callback,
)

# Lifecycle management extracted to dedicated module (Dec 2025)
from app.coordination.daemon_lifecycle import DaemonLifecycleManager
from app.coordination.daemon_manager_lifecycle import DaemonManagerLifecycleMixin

# Event handlers extracted to dedicated module (Dec 2025)
from app.coordination.daemon_event_handlers import DaemonEventHandlers

# Daemon runner functions extracted to dedicated module (Dec 2025)
# This reduces daemon_manager.py by ~1,700 LOC
#
# CIRCULAR DEPENDENCY NOTE (Dec 2025):
# daemon_manager.py imports daemon_runners at top-level.
# daemon_runners.py imports get_daemon_manager() LAZILY inside create_health_server().
# This is SAFE because:
# 1. daemon_runners.py only uses TYPE_CHECKING for DaemonType (not executed at import)
# 2. The import of get_daemon_manager is inside a function body (lazy evaluation)
# 3. By the time create_health_server() is called, daemon_manager.py is fully loaded
from app.coordination import daemon_runners

logger = logging.getLogger(__name__)

# Restart count persistence (Dec 2025)
# Persists restart counts to disk so they survive daemon manager restarts
# Dec 29, 2025: Moved from /tmp to COORDINATION_DIR to survive reboots
try:
    from app.utils.paths import COORDINATION_DIR
    _restart_state_dir = COORDINATION_DIR
except ImportError:
    # Fallback if paths module not available
    _restart_state_dir = Path(__file__).parent.parent.parent / "data" / "coordination"

# Ensure the directory exists
_restart_state_dir.mkdir(parents=True, exist_ok=True)
RESTART_STATE_FILE = _restart_state_dir / "daemon_restarts.json"
RESTART_COUNTS_EXPIRY_SECONDS = 86400  # 24 hours - counts older than this are reset
MAX_RESTARTS_PER_HOUR = 10  # If exceeded, daemon is permanently failed
PERMANENT_FAILURE_RECOVERY_SECONDS = 86400  # 24 hours - permanently failed daemons auto-recover

# Cascade restart circuit breaker (Dec 2025)
# Prevents "thundering herd" effect when many daemons fail simultaneously
# If too many restarts happen globally (across all daemons), pause all restarts
CASCADE_RESTART_WINDOW_SECONDS = 300  # 5 minutes - window for counting global restarts
CASCADE_RESTART_THRESHOLD = 25  # Max total restarts in window before circuit trips (was 15, Session 17.48)
CASCADE_COOLDOWN_SECONDS = 120  # 2 minutes - cooldown period when circuit is open
CASCADE_STARTUP_GRACE_PERIOD = 300  # 5 minutes - higher threshold during startup (was 180s, Session 17.48)
CASCADE_STARTUP_THRESHOLD = 100  # Allow many restarts during startup (was 50, Session 17.48)


# =============================================================================
# Daemon Status Circuit Breaker (December 30, 2025)
# =============================================================================
# Prevents repeated slow health checks from blocking the health server.
# If a daemon's status collection times out multiple times, the circuit opens
# and future checks return cached/unavailable status until the reset timeout.


class DaemonStatusCircuitBreaker:
    """Circuit breaker for daemon status collection.

    December 30, 2025: Added to fix P2P cluster connectivity issues where
    slow daemon status collection blocked HTTP health endpoints.

    January 3, 2026: Refactored to use OperationCircuitBreaker internally
    (Sprint 13.2 circuit breaker consolidation).

    When a daemon's status check fails (timeout or error) multiple times,
    the circuit opens and subsequent checks immediately return "circuit_open"
    until the reset timeout expires.

    Usage:
        breaker = DaemonStatusCircuitBreaker()

        if breaker.is_open("my_daemon"):
            # Skip health check, use cached/unavailable status
            return {"status": "circuit_open"}

        try:
            status = await asyncio.timeout(1.0):
                get_daemon_status(daemon)
            breaker.record_success("my_daemon")
        except asyncio.TimeoutError:
            breaker.record_failure("my_daemon")
    """

    def __init__(
        self,
        failure_threshold: int | None = None,
        reset_timeout: float | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before circuit opens.
                Defaults to CircuitBreakerDefaults.FAILURE_THRESHOLD.
            reset_timeout: Seconds until circuit closes after opening.
                Defaults to CircuitBreakerDefaults.RECOVERY_TIMEOUT.

        Jan 2, 2026: Consolidated to use CircuitBreakerDefaults for consistency.
        Jan 3, 2026: Refactored to use OperationCircuitBreaker internally.
        """
        if failure_threshold is None:
            failure_threshold = CircuitBreakerDefaults.FAILURE_THRESHOLD
        if reset_timeout is None:
            reset_timeout = CircuitBreakerDefaults.RECOVERY_TIMEOUT
        # Use unified OperationCircuitBreaker internally
        config = CircuitConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=reset_timeout,
            operation_type="daemon_status",
            emit_events=False,  # Don't emit events for status checks
        )
        self._breaker = OperationCircuitBreaker(config=config)
        # Keep threshold/timeout for get_status() backward compatibility
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout

    def is_open(self, daemon_name: str) -> bool:
        """Check if circuit is open (should skip health check).

        Returns True if circuit is open (too many recent failures),
        False if circuit is closed (ok to try health check).
        """
        # Inverted logic: can_execute returns True if CLOSED, we return True if OPEN
        return not self._breaker.can_execute(daemon_name)

    def record_failure(self, daemon_name: str) -> None:
        """Record a health check failure (timeout or error)."""
        self._breaker.record_failure(daemon_name)

    def record_success(self, daemon_name: str) -> None:
        """Record a successful health check."""
        self._breaker.record_success(daemon_name)

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        summary = self._breaker.get_summary()
        all_states = self._breaker.get_all_states()
        # Build failure counts from circuit data
        failure_counts = {
            target: status.failure_count
            for target, status in all_states.items()
        }
        return {
            "open_circuits": summary["open_targets"],
            "failure_counts": failure_counts,
            "threshold": self._failure_threshold,
            "reset_timeout": self._reset_timeout,
            # Additional info from OperationCircuitBreaker
            "total_circuits": summary["total_targets"],
            "half_open_circuits": summary["half_open"],
        }


# Global circuit breaker instance for daemon status checks
_daemon_status_breaker: DaemonStatusCircuitBreaker | None = None


def get_daemon_status_breaker() -> DaemonStatusCircuitBreaker:
    """Get the global daemon status circuit breaker."""
    global _daemon_status_breaker
    if _daemon_status_breaker is None:
        _daemon_status_breaker = DaemonStatusCircuitBreaker()
    return _daemon_status_breaker


# Lazy import for daemon lifecycle events to avoid circular imports
def _get_daemon_event_emitters():
    """Lazy import daemon event emitters.

    Returns tuple of (emit_daemon_started, emit_daemon_stopped) or (None, None)
    if import fails.
    """
    try:
        from app.distributed.data_events import (
            emit_daemon_started,
            emit_daemon_stopped,
        )
        return emit_daemon_started, emit_daemon_stopped
    except ImportError:
        logger.debug("data_events not available for daemon lifecycle events")
        return None, None

# Note: Deprecated daemon tracking is now in daemon_types.py._DEPRECATED_DAEMON_TYPES
# The legacy re-export was removed Dec 2025 as it was unused dead code.


class DaemonManager(DaemonManagerLifecycleMixin, SingletonMixin["DaemonManager"]):
    """Unified manager for all background daemons and services.

    Provides centralized lifecycle management, health monitoring, and
    coordinated shutdown for all background services.

    December 2025: Now uses SingletonMixin for thread-safe singleton pattern.
    Use get_instance() for singleton access. The reset_instance() method
    includes async shutdown for proper cleanup.
    """

    def __init__(self, config: DaemonManagerConfig | None = None):
        """Initialize the DaemonManager.

        Args:
            config: Optional configuration. If None, uses DaemonManagerConfig defaults.

        Sets up:
            - Daemon registry (_daemons) for tracking daemon state
            - Factory registry (_factories) for daemon runner functions
            - Health monitoring task infrastructure
            - Shutdown event and async lock for coordination
            - DaemonLifecycleManager for start/stop/restart operations
            - Default daemon factories from daemon_runners.py module
            - atexit handler for graceful cleanup on process exit

        December 2025: Lifecycle operations delegated to DaemonLifecycleManager.
        Runner functions extracted to daemon_runners.py for testability.
        """
        self.config = config or DaemonManagerConfig()
        self._daemons: dict[DaemonType, DaemonInfo] = {}
        self._factories: dict[DaemonType, Callable[[], Coroutine[Any, Any, None]]] = {}
        self._running = False
        self._health_task: asyncio.Task | None = None
        self._start_time: float = time.time()
        self._shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()

        # Dec 2025: Track if coordination events have been wired
        # This ensures SyncRouter.wire_to_event_router() is called even when
        # daemons are started individually (not via start_all())
        self._coordination_wired = False

        # Dec 2025: Restart count persistence
        # Load persisted restart counts from disk to prevent infinite restart loops
        # after daemon manager restarts. Tracks hourly restart timestamps for
        # detecting permanently failing daemons.
        # Dec 2025 (updated): _permanently_failed now tracks WHEN daemon was marked failed
        # to enable auto-recovery after 24 hours.
        self._persisted_restart_counts: dict[str, int] = {}
        self._restart_timestamps: dict[str, list[float]] = {}  # Daemon -> list of restart times
        self._permanently_failed: dict[str, float] = {}  # Daemon -> timestamp when marked failed

        # Sprint 5 (Jan 2, 2026): Deadlock detection
        # Track lock acquisitions to detect potential deadlocks (locks held > 5 minutes)
        self._lock_acquired_at: float | None = None
        self._lock_holder_operation: str | None = None
        self._deadlock_threshold_seconds: float = 300.0  # 5 minutes
        self._deadlock_detected_at: float | None = None  # Prevent duplicate alerts

        # Dec 2025: Degraded mode tracking (48-hour autonomous operation)
        # Tracks daemons in degraded mode with their next retry time and tier
        # Key: daemon_name, Value: (next_retry_time, restart_tier, entered_degraded_at)
        self._degraded_daemons: dict[str, tuple[float, RestartTier, float]] = {}

        # Dec 30, 2025: Hierarchical cascade circuit breaker
        # Replaces global cascade breaker with per-category breakers
        # Critical daemons exempt, category-specific thresholds and cooldowns
        self._cascade_breaker: CascadeBreakerManager = get_cascade_breaker()

        # Legacy variables kept for backward compatibility during transition
        # Will be removed once all callers migrate to _cascade_breaker
        self._cascade_breaker_open: bool = False
        self._cascade_breaker_opened_at: float = 0.0
        self._global_restart_timestamps: list[float] = []

        self._load_restart_counts()

        # Register cleanup
        atexit.register(self._sync_shutdown)

        # Register default factories
        self._register_default_factories()

        # Lifecycle management extracted to DaemonLifecycleManager (Dec 2025)
        self._lifecycle = DaemonLifecycleManager(
            daemons=self._daemons,
            factories=self._factories,
            config=self.config,
            shutdown_event=self._shutdown_event,
            lock=self._lock,
            update_daemon_state=self._update_daemon_state,
            running_flag_getter=lambda: self._running,
            running_flag_setter=lambda v: setattr(self, "_running", v),
            record_restart=self.record_restart,  # Dec 2025: Restart count persistence
        )

        # Register callback for mark_daemon_ready() to break circular dependency (Dec 2025)
        # This allows daemon_types.py to signal readiness without importing daemon_manager
        register_mark_ready_callback(self._handle_daemon_ready)

        # Event handlers extracted to DaemonEventHandlers (Dec 2025)
        self._event_handlers = DaemonEventHandlers(self)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing).

        Overrides SingletonMixin.reset_instance() to ensure proper cleanup:
        1. Cancels _health_task directly (avoids async task leaks)
        2. Clears restart state tracking
        3. Attempts graceful shutdown if event loop available

        December 2025: Enhanced for singleton registry test cleanup.
        """
        from app.coordination.singleton_mixin import SingletonMixin

        with cls._get_lock():
            if cls in SingletonMixin._instances:
                instance = SingletonMixin._instances[cls]

                # 1. Cancel health task directly (prevents async leaks)
                if hasattr(instance, "_health_task") and instance._health_task:
                    if not instance._health_task.done():
                        instance._health_task.cancel()
                    instance._health_task = None

                # 2. Clear restart state tracking
                if hasattr(instance, "_persisted_restart_counts"):
                    instance._persisted_restart_counts.clear()
                if hasattr(instance, "_restart_timestamps"):
                    instance._restart_timestamps.clear()
                if hasattr(instance, "_permanently_failed"):
                    instance._permanently_failed.clear()
                if hasattr(instance, "_degraded_daemons"):
                    instance._degraded_daemons.clear()

                # 3. Mark as not running
                if hasattr(instance, "_running"):
                    instance._running = False

                # 4. Set shutdown event to stop loops
                if hasattr(instance, "_shutdown_event"):
                    instance._shutdown_event.set()

                # 5. Try graceful shutdown (best effort)
                try:
                    loop = asyncio.get_running_loop()
                    fire_and_forget(
                        instance.shutdown(),
                        name="daemon_manager_reset_shutdown",
                    )
                except RuntimeError:
                    # No running loop - sync cleanup already done above
                    pass

        # Call parent to clear the singleton reference
        super().reset_instance()

    # =========================================================================
    # Lock Helpers (December 30, 2025)
    # =========================================================================
    # Added to prevent deadlocks during health checks and daemon operations.

    async def _acquire_lock_with_timeout(self, timeout: float = 5.0) -> bool:
        """Acquire the internal lock with timeout.

        December 30, 2025: Added to fix P2P cluster connectivity issues where
        lock contention in health check loops blocked HTTP endpoints.

        Args:
            timeout: Maximum seconds to wait for lock acquisition.

        Returns:
            True if lock was acquired, False if timeout expired.
        """
        try:
            return await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"DaemonManager lock acquisition timed out after {timeout}s")
            return False

    @contextlib.asynccontextmanager
    async def _with_lock_timeout(
        self,
        operation_name: str,
        timeout: float = 5.0,
    ):
        """Context manager for timeout-protected lock acquisition.

        Usage:
            async with self._with_lock_timeout("health_check") as acquired:
                if acquired:
                    # ... protected code
                    pass

        Args:
            operation_name: Name for logging on timeout.
            timeout: Maximum seconds to wait for lock.

        Yields:
            True if lock was acquired, False if timeout expired.
        """
        acquired = await self._acquire_lock_with_timeout(timeout)
        if not acquired:
            logger.warning(f"Skipping {operation_name}: lock timeout")
            yield False
            return
        try:
            # Sprint 5: Track lock acquisition for deadlock detection
            self._lock_acquired_at = time.time()
            self._lock_holder_operation = operation_name
            yield True
        finally:
            # Sprint 5: Clear lock tracking on release
            self._lock_acquired_at = None
            self._lock_holder_operation = None
            self._lock.release()

    # =========================================================================
    # Deadlock Detection (Sprint 5 - Jan 2, 2026)
    # =========================================================================

    async def _check_for_deadlocks(self) -> None:
        """Check for potential deadlocks (locks held > threshold).

        Sprint 5 (Jan 2, 2026): Monitors lock durations and emits DEADLOCK_DETECTED
        event if any lock is held for more than 5 minutes.

        This is called from the health loop to detect stuck operations that may
        be blocking other daemon operations.
        """
        if self._lock_acquired_at is None:
            # No lock currently held - clear any previous detection
            self._deadlock_detected_at = None
            return

        now = time.time()
        lock_duration = now - self._lock_acquired_at

        if lock_duration < self._deadlock_threshold_seconds:
            # Lock not held long enough to be a deadlock
            return

        # Potential deadlock detected
        if self._deadlock_detected_at is not None:
            # Already detected and reported, don't spam
            return

        self._deadlock_detected_at = now
        operation = self._lock_holder_operation or "unknown"
        duration_minutes = lock_duration / 60.0

        logger.error(
            f"[DaemonManager] DEADLOCK SUSPECTED: Lock held for {duration_minutes:.1f} minutes "
            f"by operation '{operation}'"
        )

        # Emit DEADLOCK_DETECTED event
        try:
            from app.distributed.data_events import emit_deadlock_detected
            await emit_deadlock_detected(
                resources=["daemon_manager_lock"],
                holders=[operation],
                source="daemon_manager",
            )
        except (ImportError, RuntimeError) as e:
            logger.debug(f"Could not emit DEADLOCK_DETECTED: {e}")

    def get_lock_status(self) -> dict[str, Any]:
        """Get current lock status for debugging.

        Returns:
            Dict with lock status information including:
            - locked: Whether lock is currently held
            - operation: Name of operation holding the lock (if any)
            - duration_seconds: How long lock has been held (if any)
            - deadlock_detected: Whether deadlock was detected
        """
        now = time.time()
        locked = self._lock_acquired_at is not None
        duration = (now - self._lock_acquired_at) if locked else 0.0

        return {
            "locked": locked,
            "operation": self._lock_holder_operation,
            "duration_seconds": round(duration, 2),
            "deadlock_detected": self._deadlock_detected_at is not None,
            "threshold_seconds": self._deadlock_threshold_seconds,
        }

    # =========================================================================
    # Restart Count Persistence (Dec 2025)
    # =========================================================================





















    # =========================================================================
    # Factory Registration
    # =========================================================================

    def _register_default_factories(self) -> None:
        """Register default daemon factories from the declarative registry.

        December 2025: Refactored to data-driven pattern using daemon_registry.py.
        All daemon specifications are now declarative, reducing code duplication
        and making configuration easier to test and introspect.

        Only _create_health_server remains inline (needs self access).

        December 28, 2025: Added strict_registry_validation config option.
        When enabled, raises ValueError if any DaemonType lacks a registry entry.
        """
        from app.coordination.daemon_registry import (
            DAEMON_REGISTRY,
            validate_registry,
            validate_registry_or_raise,
        )

        # Validate registry at startup to catch configuration errors early
        # December 2025: Added to prevent silent failures from typos/missing runners
        # December 28, 2025: Added strict mode via config
        if self.config.strict_registry_validation:
            # Strict mode: raise on any validation errors
            validate_registry_or_raise()
            logger.info("[DaemonManager] Registry validation passed (strict mode)")
        else:
            # Lenient mode: log errors but continue
            validation_errors = validate_registry()
            if validation_errors:
                for error in validation_errors:
                    logger.error(f"[DaemonManager] Registry validation error: {error}")
                # Don't raise - allow system to start with partial registry
                # but log errors prominently for visibility

        # Register all daemons from the declarative registry
        for daemon_type, spec in DAEMON_REGISTRY.items():
            # Get the runner function from daemon_runners module
            runner = getattr(daemon_runners, spec.runner_name, None)
            if runner is None:
                logger.warning(
                    f"[DaemonManager] Runner '{spec.runner_name}' not found for {daemon_type.name}"
                )
                continue

            self.register_factory(
                daemon_type,
                runner,
                depends_on=list(spec.depends_on) if spec.depends_on else None,
                soft_depends_on=list(spec.soft_depends_on) if spec.soft_depends_on else None,
                startup_mode=spec.startup_mode,
                health_check_interval=spec.health_check_interval,
                auto_restart=spec.auto_restart,
                max_restarts=spec.max_restarts,
            )

        # Health server needs self access - kept inline
        self.register_factory(DaemonType.HEALTH_SERVER, self._create_health_server)

    def register_factory(
        self,
        daemon_type: DaemonType,
        factory: Callable[[], Coroutine[Any, Any, None]],
        depends_on: list[DaemonType] | None = None,
        soft_depends_on: list[DaemonType] | None = None,
        startup_mode: str = "degraded",
        health_check_interval: float | None = None,
        auto_restart: bool = True,
        max_restarts: int = 5,
        startup_grace_period: float | None = None,
    ) -> None:
        """Register a factory function for creating a daemon.

        Args:
            daemon_type: Type of daemon
            factory: Async function that runs the daemon
            depends_on: List of daemons that must be running first (hard deps)
            soft_depends_on: List of daemons that should be running if available.
                Jan 2, 2026: Startup continues with warning if these are missing.
            startup_mode: How to handle missing soft deps.
                - "strict": Fail startup if soft deps missing
                - "degraded": Start in degraded mode (default)
                - "local": Start in local-only mode
            health_check_interval: Health check interval for this daemon.
                If None, uses critical_daemon_health_interval (15s) for critical
                daemons, or 60s for others. (P11-HIGH-2 Dec 2025)
            auto_restart: Whether to auto-restart on failure
            max_restarts: Maximum restart attempts
            startup_grace_period: Seconds before health checks begin after startup.
                If None, uses default_startup_grace_period from config (60s).
                December 2025: Prevents premature health check failures.
        """
        # P11-HIGH-2: Use faster health check interval for critical daemons
        if health_check_interval is None:
            if daemon_type in CRITICAL_DAEMONS:
                health_check_interval = self.config.critical_daemon_health_interval
                logger.debug(
                    f"[DaemonManager] Using critical daemon health interval "
                    f"({health_check_interval}s) for {daemon_type.value}"
                )
            else:
                health_check_interval = 60.0

        # December 2025: Use config default if not specified
        if startup_grace_period is None:
            startup_grace_period = self.config.default_startup_grace_period

        self._factories[daemon_type] = factory

        # Dec 2025: Apply persisted restart count if available
        # This ensures restart counts survive daemon manager restarts
        daemon_name = daemon_type.value
        persisted_count = self._persisted_restart_counts.get(daemon_name, 0)

        self._daemons[daemon_type] = DaemonInfo(
            daemon_type=daemon_type,
            depends_on=depends_on or [],
            soft_depends_on=soft_depends_on or [],
            startup_mode=startup_mode,
            health_check_interval=health_check_interval,
            auto_restart=auto_restart,
            max_restarts=max_restarts,
            restart_count=persisted_count,
            startup_grace_period=startup_grace_period,
        )

        # Log if daemon has non-zero restart count from persistence
        if persisted_count > 0:
            logger.info(
                f"[DaemonManager] {daemon_name} starting with persisted restart count: "
                f"{persisted_count}/{max_restarts}"
            )

    def _handle_daemon_ready(self, daemon_type: DaemonType) -> None:
        """Handle daemon readiness signal from mark_daemon_ready().

        This is called via the callback registered in __init__() to break
        the circular dependency between daemon_types and daemon_manager.

        Args:
            daemon_type: The daemon that signaled readiness.
        """
        info = self._daemons.get(daemon_type)
        if info is not None and info.ready_event is not None:
            info.ready_event.set()
            logger.debug(f"[DaemonManager] {daemon_type.value} signaled readiness")

    def _update_daemon_state(
        self,
        info: DaemonInfo,
        new_state: DaemonState,
        reason: str = "",
        error: str | None = None,
    ) -> None:
        """Update daemon state and emit status changed event.

        P0.5 Dec 2025: Centralizes state transitions and emits DAEMON_STATUS_CHANGED
        events for watchdog integration.

        Args:
            info: Daemon info to update
            new_state: New state to set
            reason: Why the state changed (timeout, exception, signal, restart)
            error: Error message if applicable
        """
        import socket
        old_state = info.state
        info.state = new_state

        # Skip event emission for minor transitions
        if old_state == new_state:
            return

        # Only emit for significant transitions
        significant_transitions = {
            (DaemonState.RUNNING, DaemonState.FAILED),
            (DaemonState.RUNNING, DaemonState.RESTARTING),
            (DaemonState.RESTARTING, DaemonState.RUNNING),
            (DaemonState.STARTING, DaemonState.FAILED),
            (DaemonState.STOPPED, DaemonState.RUNNING),
            (DaemonState.RUNNING, DaemonState.STOPPED),
            (DaemonState.RUNNING, DaemonState.IMPORT_FAILED),
        }

        if (old_state, new_state) not in significant_transitions:
            return

        try:
            from app.coordination.event_router import emit_daemon_status_changed

            if emit_daemon_status_changed is None:
                logger.debug("emit_daemon_status_changed not available, skipping event emission")
                return

            hostname = socket.gethostname()

            # Fire and forget - don't block state transitions on event emission
            fire_and_forget(
                emit_daemon_status_changed(
                    daemon_name=info.daemon_type.value,
                    hostname=hostname,
                    old_status=old_state.value,
                    new_status=new_state.value,
                    reason=reason,
                    error=error,
                    source="daemon_manager",
                ),
                name=f"emit_daemon_status_{info.daemon_type.value}",
            )
        except (RuntimeError, OSError, ConnectionError) as e:
            # Don't fail state transition if event emission fails
            logger.debug(f"Failed to emit daemon status event: {e}")

    def _validate_critical_subsystems(self) -> list[str]:
        """Validate critical subsystems before starting daemons.

        Returns:
            List of validation error messages (empty if all OK).

        December 2025: Added as part of Phase 8 startup validation.
        Critical subsystems that must be importable for daemons to function.
        December 27, 2025: Added startup order validation (Wave 4 Phase 1).
        """
        errors = []

        # Validate daemon startup order consistency first
        # This catches dependency violations early before any daemon starts
        try:
            from app.coordination.daemon_types import validate_startup_order_consistency
            is_consistent, violations = validate_startup_order_consistency()
            if not is_consistent:
                for violation in violations:
                    error_msg = f"Startup order violation: {violation}"
                    logger.error(f"[DaemonManager] {error_msg}")
                    errors.append(error_msg)
            else:
                logger.debug("[DaemonManager] Daemon startup order validated successfully")
        except ImportError as e:
            logger.warning(f"[DaemonManager] Could not validate startup order: {e}")

        critical_modules = [
            ("app.coordination.event_router", "Event routing"),
            ("app.coordination.sync_router", "Sync routing"),
            ("app.coordination.sync_facade", "Sync coordination"),
            ("app.coordination.protocols", "Health check protocols"),
        ]

        for module_path, description in critical_modules:
            try:
                importlib.import_module(module_path)
            except ImportError as e:
                error_msg = f"Critical subsystem unavailable: {description} ({module_path}): {e}"
                logger.error(f"[DaemonManager] {error_msg}")
                errors.append(error_msg)

        # Optional modules - log warning but don't block startup
        optional_modules = [
            ("app.coordination.health_facade", "Health monitoring"),
            ("app.coordination.daemon_watchdog", "Daemon watchdog"),
        ]

        for module_path, description in optional_modules:
            try:
                importlib.import_module(module_path)
            except ImportError as e:
                logger.warning(f"[DaemonManager] Optional subsystem unavailable: {description} ({module_path}): {e}")

        # December 2025: Verify critical event subscriptions are ready
        # This catches issues where emitters start before subscribers are wired
        missing_subs = self._verify_critical_subscriptions()
        if missing_subs:
            for event_type in missing_subs:
                logger.warning(f"[DaemonManager] Critical event {event_type} has no subscribers yet")

        if errors:
            logger.error(f"[DaemonManager] {len(errors)} critical subsystem(s) failed validation")
        else:
            logger.debug("[DaemonManager] All critical subsystems validated successfully")

        return errors

    def _verify_critical_subscriptions(self) -> list[str]:
        """Verify critical event subscriptions are wired.

        December 2025: Added to ensure subscriber daemons are started before
        emitter daemons. This prevents event loss where emitters fire before
        handlers are subscribed.

        Returns:
            List of event types that are missing subscribers
        """
        # Critical events that must have subscribers before sync/training daemons start
        critical_events = [
            "DATA_SYNC_COMPLETED",
            "TRAINING_COMPLETED",
            "MODEL_PROMOTED",
            "EVALUATION_COMPLETED",
            "NEW_GAMES_AVAILABLE",
        ]

        missing = []
        try:
            from app.coordination.event_router import has_subscribers

            for event_type in critical_events:
                if not has_subscribers(event_type):
                    missing.append(event_type)

            if missing:
                logger.debug(
                    f"[DaemonManager] {len(missing)} critical events missing subscribers: {missing}"
                )
            else:
                logger.debug("[DaemonManager] All critical event subscriptions verified")
        except ImportError:
            logger.debug("[DaemonManager] Cannot verify event subscriptions - event_router not available")

        return missing

    # NOTE: _verify_p2p_subscriptions was removed Dec 27, 2025
    # P2P event verification is now consolidated in _verify_subscriptions() at line ~1156
    # which is called in the post-start callback of start_all()













    async def _subscribe_to_critical_events(self) -> None:
        """Subscribe to critical events via DaemonEventHandlers.

        December 2025: Delegated to DaemonEventHandlers class for maintainability.
        See daemon_event_handlers.py for the full implementation.
        """
        await self._event_handlers.subscribe_to_events()

    # =========================================================================
    # Event handlers moved to daemon_event_handlers.py (December 2025)
    # The following handlers are now in DaemonEventHandlers class:
    # - _on_regression_critical
    # - _on_selfplay_target_updated
    # - _on_exploration_boost
    # - _on_daemon_status_changed
    # - _on_host_offline
    # - _on_host_online
    # - _on_leader_elected
    # - _on_backpressure_activated
    # - _on_backpressure_released
    # - _on_disk_space_low
    #
    # Forwarding methods below for backward compatibility with existing tests.
    # =========================================================================

    async def _on_regression_critical(self, event: Any) -> None:
        """Forward to DaemonEventHandlers."""
        await self._event_handlers._on_regression_critical(event)

    async def _on_selfplay_target_updated(self, event: Any) -> None:
        """Forward to DaemonEventHandlers."""
        await self._event_handlers._on_selfplay_target_updated(event)

    async def _on_exploration_boost(self, event: Any) -> None:
        """Forward to DaemonEventHandlers."""
        await self._event_handlers._on_exploration_boost(event)

    async def _on_daemon_status_changed(self, event: Any) -> None:
        """Forward to DaemonEventHandlers."""
        await self._event_handlers._on_daemon_status_changed(event)

    async def _on_host_offline(self, event: Any) -> None:
        """Forward to DaemonEventHandlers."""
        await self._event_handlers._on_host_offline(event)

    async def _on_host_online(self, event: Any) -> None:
        """Forward to DaemonEventHandlers."""
        await self._event_handlers._on_host_online(event)

    async def _on_leader_elected(self, event: Any) -> None:
        """Forward to DaemonEventHandlers."""
        await self._event_handlers._on_leader_elected(event)

    async def _on_backpressure_activated(self, event: Any) -> None:
        """Forward to DaemonEventHandlers."""
        await self._event_handlers._on_backpressure_activated(event)

    async def _on_backpressure_released(self, event: Any) -> None:
        """Forward to DaemonEventHandlers."""
        await self._event_handlers._on_backpressure_released(event)

    async def _on_disk_space_low(self, event: Any) -> None:
        """Forward to DaemonEventHandlers."""
        await self._event_handlers._on_disk_space_low(event)

    async def _ensure_coordination_wired(self) -> None:
        """Ensure coordination events are wired exactly once.

        December 2025: Fixes critical integration gap where SyncRouter was not
        auto-wired when daemons were started individually (via start()) instead
        of via start_all().

        This method is idempotent - calling it multiple times is safe.
        It tracks whether wiring has already been done and skips if so.

        Tests can disable the bootstrap via DaemonManagerConfig.enable_coordination_wiring.
        """
        if self._coordination_wired:
            return

        if not getattr(self.config, "enable_coordination_wiring", True):
            self._coordination_wired = True
            logger.debug("[DaemonManager] Coordination wiring disabled by config")
            return

        # Wire coordination events (includes SyncRouter.wire_to_event_router())
        await self._wire_coordination_events()
        self._coordination_wired = True
        logger.debug("[DaemonManager] Coordination events wired on first daemon start")

    async def _wire_coordination_events(self) -> None:
        """Wire ALL coordination event subscriptions at startup.

        Phase 8 (December 2025): Ensures critical event subscriptions are wired
        BEFORE daemons start processing, preventing race conditions where daemons
        emit events that have no subscribers.

        This calls bootstrap_coordination() with appropriate flags to initialize:
        - Sync coordinator (DATA_SYNC_COMPLETED, NEW_GAMES_AVAILABLE)
        - Training coordinator (TRAINING_*, REGRESSION_*)
        - Pipeline orchestrator (stage events)
        - Selfplay orchestrator (SELFPLAY_COMPLETE)
        - And other critical coordinators

        The wiring is idempotent - calling multiple times is safe.
        """
        try:
            from app.coordination.coordination_bootstrap import bootstrap_coordination

            # Wire critical event subscriptions
            # Use lightweight init - we're called from start_all() which is async
            result = bootstrap_coordination(
                # Essential event sources
                enable_sync=True,           # DATA_SYNC_COMPLETED, NEW_GAMES_AVAILABLE
                enable_training=True,       # TRAINING_*, REGRESSION_*
                enable_pipeline=True,       # Stage events
                enable_selfplay=True,       # SELFPLAY_COMPLETE
                enable_model=True,          # MODEL_PROMOTED
                enable_health=True,         # Health events
                # Disable heavy initializations (already handled by daemons)
                enable_resources=False,     # ResourceMonitoringCoordinator is heavy
                enable_metrics=False,       # MetricsAnalysisOrchestrator is heavy
                enable_optimization=False,  # OptimizationCoordinator is heavy
                enable_cache=False,         # CacheCoordinator is heavy
                enable_leadership=False,    # LeadershipCoordinator handled elsewhere
                # Disable daemons (they're managed by DaemonManager)
                enable_auto_export=False,
                enable_auto_evaluation=False,
                enable_model_distribution=False,
                enable_idle_resource=False,
                enable_quality_monitor=False,
                enable_orphan_detection=False,
                enable_curriculum_integration=False,
                # Other settings
                register_with_registry=False,  # We do this ourselves
            )

            initialized = result.get("initialized_count", 0)
            errors = result.get("errors", [])

            if errors:
                for err in errors[:3]:  # Log first 3 errors
                    logger.warning(f"[DaemonManager] Coordination wiring error: {err}")

            logger.info(
                f"[DaemonManager] Wired {initialized} coordination event subscriptions (Phase 8)"
            )

        except ImportError as e:
            logger.debug(f"[DaemonManager] coordination_bootstrap not available: {e}")
        except (RuntimeError, OSError, ConnectionError) as e:
            logger.warning(f"[DaemonManager] Failed to wire coordination events: {e}")

    async def _verify_subscriptions(self) -> None:
        """Verify that critical event subscriptions are active.

        Phase 5 (December 2025): Startup verification catches missing wiring early.
        Logs warnings for any critical events that have no subscribers.
        """
        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()
            if router is None:
                logger.warning("[DaemonManager] Event router not available for subscription verification")
                return

            # Critical events that should have subscribers for feedback loop to work
            critical_events = [
                (DataEventType.HYPERPARAMETER_UPDATED, "FeedbackAccelerator"),
                (DataEventType.CURRICULUM_ADVANCED, "CurriculumFeedback, SelfplayRunner"),
                (DataEventType.ADAPTIVE_PARAMS_CHANGED, "SelfplayRunner"),
                (DataEventType.REGRESSION_CRITICAL, "DaemonManager, TrainingCoordinator"),
                (DataEventType.EVALUATION_COMPLETED, "FeedbackAccelerator, MomentumBridge"),
                (DataEventType.MODEL_PROMOTED, "SelfplayRunner, ModelDistribution"),
            ]

            # Dec 27, 2025: Add P2P cluster events for daemon lifecycle coordination
            p2p_events = [
                (DataEventType.HOST_OFFLINE, "DaemonManager"),
                (DataEventType.HOST_ONLINE, "DaemonManager"),
                (DataEventType.LEADER_ELECTED, "DaemonManager"),
                (DataEventType.NEW_GAMES_AVAILABLE, "TrainingCoordinator, DataPipeline"),
                # Dec 27, 2025: Added missing P2P events for cluster health monitoring
                (DataEventType.P2P_CLUSTER_HEALTHY, "TrainingCoordinator, SelfplayScheduler"),
                (DataEventType.P2P_CLUSTER_UNHEALTHY, "TrainingCoordinator, SelfplayScheduler, FeedbackLoop"),
                (DataEventType.NODE_RECOVERED, "SyncRouter, JobManager"),
            ]

            # Combine all events to verify
            all_events = critical_events + [
                (evt, desc) for evt, desc in p2p_events
                if hasattr(DataEventType, evt.name if hasattr(evt, 'name') else str(evt).split('.')[-1])
            ]

            missing = []
            active = []

            for event_type, expected_subscribers in all_events:
                event_key = event_type.value if hasattr(event_type, 'value') else str(event_type)

                # Check if router has subscribers for this event
                subscriber_count = 0
                if hasattr(router, '_subscribers'):
                    subscriber_count = len(router._subscribers.get(event_key, []))
                elif hasattr(router, 'get_subscriber_count'):
                    subscriber_count = router.get_subscriber_count(event_key)

                if subscriber_count == 0:
                    missing.append(f"{event_key} (expected: {expected_subscribers})")
                else:
                    active.append(f"{event_key}: {subscriber_count} subscribers")

            if missing:
                logger.warning(
                    f"[DaemonManager] Missing event subscribers ({len(missing)}/{len(all_events)}):\n"
                    f"  {chr(10).join('- ' + m for m in missing)}"
                )
            else:
                logger.info(
                    f"[DaemonManager] All {len(all_events)} events have subscribers "
                    f"({len(critical_events)} critical + {len(p2p_events)} P2P)"
                )

            if active:
                logger.debug(
                    f"[DaemonManager] Active subscriptions:\n"
                    f"  {chr(10).join('- ' + a for a in active)}"
                )

        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"[DaemonManager] Subscription verification failed: {e}")

    async def _emit_daemons_ready(self) -> None:
        """Emit ALL_CRITICAL_DAEMONS_READY signal after startup completes.

        Phase 12 (December 2025): This closes the startup race condition where
        events could be emitted before handlers were registered. External systems
        (P2P orchestrator, training pipeline) can wait for this signal before
        emitting events that need to be handled.

        The event includes:
        - ready_daemons: List of daemons that started successfully
        - timestamp: When readiness was achieved
        - subscription_status: Whether critical subscriptions are active
        """
        try:
            from app.coordination.event_router import publish, DataEventType

            # Collect ready daemons
            ready_daemons = [
                dtype.value for dtype, state in self._lifecycle.get_daemon_states().items()
                if state.name == "RUNNING"
            ]

            # Check if we have critical daemons ready
            critical_types = [
                DaemonType.EVENT_ROUTER,
                DaemonType.SELFPLAY_SCHEDULER,
                DaemonType.FEEDBACK_LOOP,
            ]
            critical_ready = sum(
                1 for dt in critical_types
                if dt in self._lifecycle.get_daemon_states()
                and self._lifecycle.get_daemon_states()[dt].name == "RUNNING"
            )

            event_data = {
                "ready_daemons": ready_daemons,
                "total_ready": len(ready_daemons),
                "critical_ready": critical_ready,
                "critical_total": len(critical_types),
                "timestamp": time.time(),
                "fully_ready": critical_ready == len(critical_types),
            }

            all_critical_ready_event = getattr(DataEventType, "ALL_CRITICAL_DAEMONS_READY", None)
            if all_critical_ready_event is not None:
                await publish(
                    all_critical_ready_event.value,
                    event_data,
                    source="DaemonManager",
                )
                logger.info(
                    f"[DaemonManager] Emitted ALL_CRITICAL_DAEMONS_READY: "
                    f"{len(ready_daemons)} daemons ready, {critical_ready}/{len(critical_types)} critical"
                )
            else:
                # Fallback: emit as generic SYSTEM_STATUS event
                await publish(
                    "system.daemons_ready",
                    event_data,
                    source="DaemonManager",
                )
                logger.info(
                    f"[DaemonManager] Emitted system.daemons_ready: "
                    f"{len(ready_daemons)} daemons ready"
                )

        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"[DaemonManager] Failed to emit readiness signal: {e}")


















    # =========================================================================
    # Lifecycle Tracking Methods (December 2025)
    # =========================================================================





    # =========================================================================
    # Liveness and Readiness Probes (December 2025)
    # =========================================================================







# =============================================================================
# Daemon Profiles (December 2025)
# =============================================================================
# Profiles group daemons by use case for easier management.

DAEMON_PROFILES: dict[str, list[DaemonType]] = {
    # Coordinator node profile - runs on central MacBook
    "coordinator": [
        DaemonType.EVENT_ROUTER,
        DaemonType.HEALTH_SERVER,  # HTTP health endpoints (/health, /ready, /metrics)
        DaemonType.DAEMON_WATCHDOG,  # Dec 2025: Monitor daemon health & auto-restart failed daemons
        DaemonType.P2P_BACKEND,
        DaemonType.TOURNAMENT_DAEMON,
        DaemonType.MODEL_DISTRIBUTION,
        DaemonType.S3_SYNC,  # Consolidated S3 sync (replaces S3_BACKUP)
        DaemonType.CLUSTER_MONITOR,
        DaemonType.QUEUE_MONITOR,  # Monitor queue depths and apply backpressure
        DaemonType.FEEDBACK_LOOP,
        DaemonType.QUALITY_MONITOR,  # Monitor selfplay data quality
        DaemonType.MODEL_PERFORMANCE_WATCHDOG,  # Monitor model win rates
        DaemonType.ORPHAN_DETECTION,  # Detect unregistered game databases
        # NOTE: NODE_HEALTH_MONITOR and SYSTEM_HEALTH_MONITOR removed Dec 2025
        # HEALTH_SERVER (line 3157) + health_check_orchestrator handle both use cases
        DaemonType.UNIFIED_PROMOTION,  # Phase 18.4: Auto-promote models after evaluation
        DaemonType.JOB_SCHEDULER,  # Phase 3: Centralized job scheduling with PID-based allocation
        DaemonType.IDLE_RESOURCE,  # Phase 20: Monitor idle GPUs and spawn selfplay
        DaemonType.NODE_RECOVERY,  # Phase 21: Auto-recover terminated nodes
        # NOTE: LAMBDA_IDLE removed Dec 29, 2025 - GH200 nodes are dedicated training, don't need idle shutdown
        DaemonType.QUEUE_POPULATOR,  # Phase 4: Auto-populate work queue with jobs
        DaemonType.CURRICULUM_INTEGRATION,  # Bridges feedback loops for self-improvement
        DaemonType.AUTO_EXPORT,  # Auto-export NPZ when game threshold met
        DaemonType.NPZ_COMBINATION,  # Dec 2025: Quality-weighted NPZ combination for training
        DaemonType.TRAINING_TRIGGER,  # Decide when to trigger training
        DaemonType.DLQ_RETRY,  # P0.3: Dead letter queue remediation (Dec 2025)
        DaemonType.GAUNTLET_FEEDBACK,  # Dec 2025: Process evaluation results → emit REGRESSION_CRITICAL
        DaemonType.AUTO_SYNC,  # Dec 2025: CRITICAL - Pull game data from remote nodes
        # NOTE: CLUSTER_DATA_SYNC removed Dec 2025 - AUTO_SYNC handles broadcast sync
        DaemonType.CLUSTER_WATCHDOG,  # Dec 2025: Self-healing cluster utilization
        DaemonType.METRICS_ANALYSIS,  # Phase 21.2: Analyze training metrics for feedback
        DaemonType.ELO_SYNC,  # Dec 2025: Sync Elo ratings across cluster nodes
        DaemonType.DATA_CONSOLIDATION,  # Dec 27, 2025: Consolidate scattered data files
        DaemonType.CLUSTER_CONSOLIDATION,  # Jan 2026: Pull games from cluster nodes (coordinator-only)
        DaemonType.COORDINATOR_DISK_MANAGER,  # Dec 27, 2025: Manage coordinator disk space
    ],

    # Training node profile - runs on GPU nodes
    "training_node": [
        DaemonType.EVENT_ROUTER,
        DaemonType.HEALTH_SERVER,  # HTTP health endpoints (/health, /ready, /metrics)
        DaemonType.DATA_PIPELINE,
        DaemonType.AUTO_SYNC,
        DaemonType.TRAINING_NODE_WATCHER,
        DaemonType.EVALUATION,  # Auto-evaluate after training completes
        DaemonType.QUALITY_MONITOR,  # Monitor local selfplay quality
        DaemonType.ORPHAN_DETECTION,  # Detect local orphaned databases
        DaemonType.UNIFIED_PROMOTION,  # Phase 18.4: Auto-promote models after evaluation
        DaemonType.P2P_AUTO_DEPLOY,  # Phase 21.2: Ensure P2P runs on recovered nodes
        DaemonType.IDLE_RESOURCE,  # Phase 4: Detect idle GPUs and auto-spawn selfplay
        DaemonType.UTILIZATION_OPTIMIZER,  # Phase 4: Match GPU capabilities to workloads
        DaemonType.CURRICULUM_INTEGRATION,  # Bridges feedback loops for local self-improvement
        DaemonType.AUTO_EXPORT,  # Auto-export NPZ when game threshold met
        DaemonType.NPZ_COMBINATION,  # Dec 2025: Quality-weighted NPZ combination for training
        DaemonType.TRAINING_TRIGGER,  # Decide when to trigger training
        DaemonType.FEEDBACK_LOOP,  # Phase 21.2: Orchestrate all feedback signals
        DaemonType.METRICS_ANALYSIS,  # Phase 21.2: Analyze training metrics for feedback
        DaemonType.DLQ_RETRY,  # P0.3: Dead letter queue remediation (Dec 2025)
        DaemonType.DISK_SPACE_MANAGER,  # Dec 27, 2025: Manage disk space on training nodes
    ],

    # Ephemeral node profile - runs on Vast.ai/spot instances
    # Phase 21.2: Expanded from 4 to 9 daemons for better data safety & observability
    "ephemeral": [
        DaemonType.EVENT_ROUTER,
        DaemonType.HEALTH_SERVER,  # HTTP health endpoints (/health, /ready, /metrics)
        # NOTE: EPHEMERAL_SYNC removed Dec 2025 - AUTO_SYNC with strategy="ephemeral" handles this
        DaemonType.DATA_PIPELINE,
        DaemonType.IDLE_RESOURCE,  # Phase 4: Detect idle GPUs and auto-spawn selfplay
        DaemonType.QUALITY_MONITOR,  # Phase 21.2: Monitor quality for throttling feedback
        DaemonType.ORPHAN_DETECTION,  # Phase 21.2: Detect orphaned databases before termination
        DaemonType.AUTO_SYNC,  # Phase 21.2: Ensure regular sync alongside ephemeral sync
        DaemonType.FEEDBACK_LOOP,  # Phase 21.2: Orchestrate all feedback signals
        DaemonType.DISK_SPACE_MANAGER,  # Dec 27, 2025: Manage disk space (critical for ephemeral)
    ],

    # Selfplay-only profile - just generates games
    "selfplay": [
        DaemonType.EVENT_ROUTER,
        DaemonType.HEALTH_SERVER,  # HTTP health endpoints (/health, /ready, /metrics)
        DaemonType.AUTO_SYNC,
        DaemonType.QUALITY_MONITOR,  # Monitor quality to trigger throttling feedback
        DaemonType.IDLE_RESOURCE,  # Phase 4: Detect idle GPUs and auto-spawn selfplay
        DaemonType.FEEDBACK_LOOP,  # Orchestrate all feedback signals
    ],

    # Full profile - all daemons (for testing)
    "full": [dt for dt in DaemonType],

    # Minimal profile - just event routing
    "minimal": [
        DaemonType.EVENT_ROUTER,
    ],
}


async def start_profile(profile: str) -> dict[DaemonType, bool]:
    """Start all daemons in a profile.

    Args:
        profile: Profile name from DAEMON_PROFILES

    Returns:
        Dict mapping daemon type to start success

    Raises:
        ValueError: If profile not found
    """
    if profile not in DAEMON_PROFILES:
        raise ValueError(f"Unknown profile: {profile}. Available: {list(DAEMON_PROFILES.keys())}")

    manager = get_daemon_manager()
    daemon_types = DAEMON_PROFILES[profile]

    logger.info(f"Starting daemon profile '{profile}' with {len(daemon_types)} daemons")
    return await manager.start_all(daemon_types)


# Module-level singleton accessors (delegate to SingletonMixin methods)
# December 2025: These now delegate to DaemonManager.get_instance() and reset_instance()
# instead of maintaining a separate module-level cache.


def get_daemon_manager(config: DaemonManagerConfig | None = None) -> DaemonManager:
    """Get the singleton DaemonManager instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        DaemonManager instance
    """
    return DaemonManager.get_instance(config=config)


def reset_daemon_manager() -> None:
    """Reset the singleton (for testing)."""
    DaemonManager.reset_instance()


# Signal handlers for graceful shutdown
def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown."""
    def handle_signal(signum: int, frame: Any) -> None:
        """Handle SIGTERM/SIGINT for graceful daemon shutdown.

        Args:
            signum: Signal number (e.g., signal.SIGTERM, signal.SIGINT)
            frame: Current stack frame (unused)
        """
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        manager = get_daemon_manager()
        try:
            loop = asyncio.get_running_loop()
            fire_and_forget(
                manager.shutdown(),
                name="daemon_manager_signal_shutdown",
            )
        except RuntimeError:
            # No running loop
            manager._sync_shutdown()

    try:
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
    except (OSError, RuntimeError, ValueError) as e:
        logger.debug(f"Could not set up signal handlers: {e}")


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Data classes
    "DaemonInfo",
    # Main class
    "DaemonManager",
    "DaemonManagerConfig",
    "DaemonState",
    # Enums
    "DaemonType",
    # Profiles
    "DAEMON_PROFILES",
    # Functions
    "get_daemon_manager",
    "mark_daemon_ready",  # P0.3 Dec 2025: Readiness signaling
    "reset_daemon_manager",
    "setup_signal_handlers",
    "start_profile",
]
