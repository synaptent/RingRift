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
import concurrent.futures
import json
import logging
import os
import signal
import time
import warnings
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from app.config.ports import DATA_SERVER_PORT
from app.core.async_context import fire_and_forget, safe_create_task

# Singleton mixin for thread-safe singleton pattern (Dec 2025)
from app.coordination.singleton_mixin import SingletonMixin

# Daemon types extracted to dedicated module (Dec 2025)
from app.coordination.daemon_types import (
    CRITICAL_DAEMONS,
    DAEMON_STARTUP_ORDER,
    DaemonInfo,
    DaemonManagerConfig,
    DaemonState,
    DaemonType,
    mark_daemon_ready,
    register_mark_ready_callback,
)

# Lifecycle management extracted to dedicated module (Dec 2025)
from app.coordination.daemon_lifecycle import DaemonLifecycleManager

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
CASCADE_RESTART_THRESHOLD = 15  # Max total restarts in window before circuit trips
CASCADE_COOLDOWN_SECONDS = 120  # 2 minutes - cooldown period when circuit is open


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


class DaemonManager(SingletonMixin["DaemonManager"]):
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

        # Dec 2025: Cascade restart circuit breaker state
        # Tracks global restart activity to prevent thundering herd effect
        self._cascade_breaker_open: bool = False
        self._cascade_breaker_opened_at: float = 0.0
        self._global_restart_timestamps: list[float] = []  # All daemon restarts globally
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
    # Restart Count Persistence (Dec 2025)
    # =========================================================================

    def _load_restart_counts(self) -> None:
        """Load persisted restart counts from disk.

        December 2025: Added to prevent infinite restart loops after daemon manager
        process restarts. Counts older than 24 hours are discarded to allow recovery
        after transient failures.

        Data structure:
            {
                "timestamp": <unix_time>,
                "counts": {"daemon_name": <count>, ...},
                "restart_timestamps": {"daemon_name": [<ts1>, <ts2>, ...], ...},
                "permanently_failed": ["daemon1", "daemon2", ...]
            }
        """
        try:
            if not RESTART_STATE_FILE.exists():
                logger.debug("[DaemonManager] No persisted restart counts found")
                return

            with open(RESTART_STATE_FILE, "r") as f:
                data = json.load(f)

            # Check if data is expired (older than 24 hours)
            saved_timestamp = data.get("timestamp", 0)
            if saved_timestamp < time.time() - RESTART_COUNTS_EXPIRY_SECONDS:
                logger.info(
                    "[DaemonManager] Persisted restart counts expired (>24h), starting fresh"
                )
                RESTART_STATE_FILE.unlink(missing_ok=True)
                return

            # Load counts
            self._persisted_restart_counts = data.get("counts", {})

            # Load restart timestamps (for hourly limit tracking)
            raw_timestamps = data.get("restart_timestamps", {})
            current_time = time.time()
            for daemon_name, timestamps in raw_timestamps.items():
                # Only keep timestamps from the last hour
                recent_timestamps = [
                    ts for ts in timestamps
                    if ts > current_time - 3600
                ]
                if recent_timestamps:
                    self._restart_timestamps[daemon_name] = recent_timestamps

            # Load permanently failed daemons (dict format: daemon_name -> timestamp)
            # Dec 2025: Auto-recover daemons that have been failed for >24 hours
            current_time = time.time()
            raw_failed = data.get("permanently_failed", {})

            # Handle legacy format (list) and new format (dict)
            if isinstance(raw_failed, list):
                # Legacy format: no timestamps, convert to dict with current time
                # (these will expire in 24h from now)
                raw_failed = {daemon: current_time for daemon in raw_failed}

            # Filter out expired failures (auto-recovery after 24 hours)
            self._permanently_failed = {}
            for daemon_name, failed_at in raw_failed.items():
                age_seconds = current_time - failed_at
                if age_seconds < PERMANENT_FAILURE_RECOVERY_SECONDS:
                    self._permanently_failed[daemon_name] = failed_at
                else:
                    logger.info(
                        f"[DaemonManager] {daemon_name} auto-recovered after "
                        f"{age_seconds / 3600:.1f}h in permanent failure state"
                    )

            logger.info(
                f"[DaemonManager] Loaded restart counts: "
                f"{len(self._persisted_restart_counts)} daemons tracked, "
                f"{len(self._permanently_failed)} permanently failed"
            )

            # Log any permanently failed daemons
            if self._permanently_failed:
                logger.warning(
                    f"[DaemonManager] Permanently failed daemons (require manual intervention): "
                    f"{list(self._permanently_failed)}"
                )

        except json.JSONDecodeError as e:
            logger.warning(f"[DaemonManager] Failed to parse restart counts file: {e}")
            RESTART_STATE_FILE.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"[DaemonManager] Failed to load restart counts: {e}")

    def _save_restart_counts(self) -> None:
        """Persist restart counts to disk.

        December 2025: Saves current restart counts and timestamps so they
        survive daemon manager restarts. This prevents infinite restart loops
        for daemons that consistently fail.
        """
        try:
            # Collect current counts from daemon info
            counts = {}
            for daemon_type, info in self._daemons.items():
                if info.restart_count > 0:
                    counts[daemon_type.value] = info.restart_count

            # Merge with persisted counts (for daemons not yet registered)
            for daemon_name, count in self._persisted_restart_counts.items():
                if daemon_name not in counts and count > 0:
                    counts[daemon_name] = count

            data = {
                "timestamp": time.time(),
                "counts": counts,
                "restart_timestamps": self._restart_timestamps,
                "permanently_failed": self._permanently_failed,  # Dict[str, float]: daemon -> fail timestamp
            }

            with open(RESTART_STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(
                f"[DaemonManager] Saved restart counts: {len(counts)} daemons"
            )

        except OSError as e:
            logger.warning(f"[DaemonManager] Failed to save restart counts: {e}")

    def record_restart(self, daemon_type: DaemonType) -> bool:
        """Record a daemon restart and check if it should be permanently failed.

        December 2025: Tracks restart timestamps to detect daemons that are
        failing repeatedly within a short time window. If a daemon restarts
        more than MAX_RESTARTS_PER_HOUR times in an hour, it is marked as
        permanently failed and requires manual intervention.

        Args:
            daemon_type: Type of daemon being restarted

        Returns:
            True if the daemon should be allowed to restart,
            False if it has exceeded the hourly limit and should be marked permanently failed
        """
        daemon_name = daemon_type.value
        current_time = time.time()

        # Check if already permanently failed
        if daemon_name in self._permanently_failed:
            failed_at = self._permanently_failed[daemon_name]
            age_seconds = current_time - failed_at

            # Auto-recover after 24 hours
            if age_seconds >= PERMANENT_FAILURE_RECOVERY_SECONDS:
                logger.info(
                    f"[DaemonManager] {daemon_name} auto-recovered after "
                    f"{age_seconds / 3600:.1f}h in permanent failure state"
                )
                del self._permanently_failed[daemon_name]
                # Clear restart timestamps to give it a fresh start
                self._restart_timestamps.pop(daemon_name, None)
                self._save_restart_counts()
            else:
                logger.error(
                    f"[DaemonManager] {daemon_name} is permanently failed "
                    f"({age_seconds / 3600:.1f}h ago), not restarting. "
                    f"Will auto-recover in {(PERMANENT_FAILURE_RECOVERY_SECONDS - age_seconds) / 3600:.1f}h"
                )
                return False

        # Get or create timestamp list for this daemon
        if daemon_name not in self._restart_timestamps:
            self._restart_timestamps[daemon_name] = []

        # Add current restart timestamp
        self._restart_timestamps[daemon_name].append(current_time)

        # Remove timestamps older than 1 hour
        self._restart_timestamps[daemon_name] = [
            ts for ts in self._restart_timestamps[daemon_name]
            if ts > current_time - 3600
        ]

        # Check for crash loop early warning (3+ restarts in 5 minutes)
        # December 2025: Emit warning before permanent failure to enable proactive intervention
        CRASH_LOOP_WINDOW_SECONDS = 300  # 5 minutes
        CRASH_LOOP_THRESHOLD = 3  # restarts in window
        recent_timestamps = [
            ts for ts in self._restart_timestamps[daemon_name]
            if ts > current_time - CRASH_LOOP_WINDOW_SECONDS
        ]
        recent_restarts = len(recent_timestamps)

        # Emit crash loop warning if threshold exceeded (but not yet permanent failure)
        if recent_restarts >= CRASH_LOOP_THRESHOLD:
            logger.warning(
                f"[DaemonManager] {daemon_name} is crash looping "
                f"({recent_restarts} restarts in {CRASH_LOOP_WINDOW_SECONDS // 60}min)"
            )
            self._emit_crash_loop_warning(
                daemon_type,
                recent_restarts,
                CRASH_LOOP_WINDOW_SECONDS // 60,
            )

        # Check if hourly limit exceeded
        hourly_restarts = len(self._restart_timestamps[daemon_name])
        if hourly_restarts > MAX_RESTARTS_PER_HOUR:
            logger.error(
                f"[DaemonManager] {daemon_name} exceeded hourly restart limit "
                f"({hourly_restarts} > {MAX_RESTARTS_PER_HOUR}), marking permanently failed. "
                f"Will auto-recover in {PERMANENT_FAILURE_RECOVERY_SECONDS / 3600:.0f}h"
            )
            self._permanently_failed[daemon_name] = current_time  # Track WHEN failed for auto-recovery
            self._save_restart_counts()

            # Emit DAEMON_PERMANENTLY_FAILED event
            self._emit_permanently_failed_event(daemon_type)

            return False

        # Save updated state
        self._save_restart_counts()
        return True

    def _emit_permanently_failed_event(self, daemon_type: DaemonType) -> None:
        """Emit DAEMON_PERMANENTLY_FAILED event for monitoring/alerting.

        December 2025: Notifies external systems when a daemon has exceeded
        its restart limit and requires manual intervention.

        Uses fire_and_forget since the emitter is async but this is called from sync context.
        If no event loop is running, the event is logged but not emitted.
        """
        try:
            from app.distributed.data_events import emit_daemon_permanently_failed

            import socket
            hostname = socket.gethostname()
            restart_count = len(self._restart_timestamps.get(daemon_type.value, []))

            # Check if we have an event loop running
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No event loop - can't emit async event
                logger.info(
                    f"DAEMON_PERMANENTLY_FAILED: {daemon_type.value} "
                    f"(restart_count={restart_count}, no event loop)"
                )
                return

            fire_and_forget(
                emit_daemon_permanently_failed(
                    daemon_name=daemon_type.value,
                    hostname=hostname,
                    restart_count=restart_count,
                    source="DaemonManager",
                ),
                name=f"emit_permanently_failed_{daemon_type.value}",
            )
            logger.info(f"Emitted DAEMON_PERMANENTLY_FAILED for {daemon_type.value}")
        except ImportError:
            logger.debug("emit_daemon_permanently_failed not available")
        except Exception as e:
            logger.debug(f"Failed to emit DAEMON_PERMANENTLY_FAILED: {e}")

    def _emit_crash_loop_warning(
        self,
        daemon_type: DaemonType,
        restart_count: int,
        window_minutes: int,
    ) -> None:
        """Emit DAEMON_CRASH_LOOP_DETECTED event as early warning.

        December 2025: Emits an early warning when a daemon is crash looping
        (3+ restarts in 5 minutes) before it reaches permanent failure status.
        This enables proactive intervention and investigation.

        Uses fire_and_forget since the emitter is async but this is called from sync context.
        If no event loop is running, the event is logged but not emitted.
        """
        try:
            from app.distributed.data_events import emit_daemon_crash_loop_detected

            import socket
            hostname = socket.gethostname()

            # Check if we have an event loop running
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No event loop - can't emit async event
                logger.info(
                    f"DAEMON_CRASH_LOOP_DETECTED: {daemon_type.value} "
                    f"({restart_count} restarts in {window_minutes}min, no event loop)"
                )
                return

            fire_and_forget(
                emit_daemon_crash_loop_detected(
                    daemon_name=daemon_type.value,
                    hostname=hostname,
                    restart_count=restart_count,
                    window_minutes=window_minutes,
                    max_restarts=MAX_RESTARTS_PER_HOUR,
                    source="DaemonManager",
                ),
                name=f"emit_crash_loop_{daemon_type.value}",
            )
            logger.info(
                f"Emitted DAEMON_CRASH_LOOP_DETECTED for {daemon_type.value} "
                f"({restart_count} restarts in {window_minutes}min)"
            )
        except ImportError:
            logger.debug("emit_daemon_crash_loop_detected not available")
        except Exception as e:
            logger.debug(f"Failed to emit DAEMON_CRASH_LOOP_DETECTED: {e}")

    def is_permanently_failed(self, daemon_type: DaemonType) -> bool:
        """Check if a daemon is permanently failed.

        Args:
            daemon_type: Type of daemon to check

        Returns:
            True if the daemon has exceeded its hourly restart limit
        """
        return daemon_type.value in self._permanently_failed

    def clear_permanently_failed(self, daemon_type: DaemonType) -> None:
        """Clear permanent failure status for a daemon.

        December 2025: Allows manual intervention to reset a daemon's status.
        This is typically called after the underlying issue is fixed.

        Args:
            daemon_type: Type of daemon to clear
        """
        daemon_name = daemon_type.value
        if daemon_name in self._permanently_failed:
            del self._permanently_failed[daemon_name]  # Fixed: was .discard() which is set method
            self._restart_timestamps.pop(daemon_name, None)
            if daemon_name in self._persisted_restart_counts:
                del self._persisted_restart_counts[daemon_name]

            # Reset the daemon's restart count in DaemonInfo
            if daemon_type in self._daemons:
                self._daemons[daemon_type].restart_count = 0

            self._save_restart_counts()
            logger.info(
                f"[DaemonManager] Cleared permanent failure status for {daemon_name}"
            )

    def _check_cascade_circuit_breaker(self) -> bool:
        """Check if cascade circuit breaker allows restarts.

        Dec 2025: Implements a global circuit breaker to prevent thundering herd effect.
        When too many daemons restart in a short window, pause all restarts to allow
        the system to stabilize.

        Returns:
            True if restarts are allowed, False if circuit breaker is open
        """
        current_time = time.time()

        # If circuit breaker is open, check if cooldown has passed
        if self._cascade_breaker_open:
            elapsed = current_time - self._cascade_breaker_opened_at
            if elapsed >= CASCADE_COOLDOWN_SECONDS:
                # Cooldown complete - close circuit breaker (half-open -> closed)
                self._cascade_breaker_open = False
                logger.info(
                    f"[DaemonManager] Cascade circuit breaker CLOSED after "
                    f"{elapsed:.0f}s cooldown - restarts allowed"
                )
            else:
                # Still in cooldown
                remaining = CASCADE_COOLDOWN_SECONDS - elapsed
                logger.warning(
                    f"[DaemonManager] Cascade circuit breaker OPEN - "
                    f"restarts blocked for {remaining:.0f}s more"
                )
                return False

        # Clean up old timestamps outside window
        cutoff = current_time - CASCADE_RESTART_WINDOW_SECONDS
        self._global_restart_timestamps = [
            ts for ts in self._global_restart_timestamps if ts > cutoff
        ]

        return True

    def _record_global_restart(self, daemon_type: DaemonType) -> None:
        """Record a restart in the global tracker and check if circuit should trip.

        Dec 2025: Tracks all restarts globally to detect cascade failures.
        If too many restarts happen in a short window, trips the circuit breaker.

        Args:
            daemon_type: Type of daemon being restarted
        """
        current_time = time.time()
        self._global_restart_timestamps.append(current_time)

        # Check if we've exceeded threshold
        recent_restarts = len(self._global_restart_timestamps)
        if recent_restarts > CASCADE_RESTART_THRESHOLD:
            if not self._cascade_breaker_open:
                self._cascade_breaker_open = True
                self._cascade_breaker_opened_at = current_time
                logger.error(
                    f"[DaemonManager] CASCADE CIRCUIT BREAKER TRIPPED! "
                    f"{recent_restarts} restarts in {CASCADE_RESTART_WINDOW_SECONDS}s "
                    f"(threshold: {CASCADE_RESTART_THRESHOLD}). "
                    f"Pausing all restarts for {CASCADE_COOLDOWN_SECONDS}s. "
                    f"Triggered by: {daemon_type.value}"
                )

                # Emit event for alerting/monitoring
                try:
                    from app.distributed.data_events import DataEventType

                    fire_and_forget(
                        self._emit_circuit_breaker_event(recent_restarts, daemon_type),
                        name="emit_cascade_breaker_tripped",
                    )
                except (ImportError, RuntimeError):
                    pass

    async def _emit_circuit_breaker_event(
        self, restart_count: int, triggered_by: DaemonType
    ) -> None:
        """Emit event when cascade circuit breaker trips.

        Args:
            restart_count: Number of restarts that triggered the breaker
            triggered_by: Daemon that caused the breaker to trip
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.publish(
                "daemon.cascade_breaker_tripped",
                {
                    "restart_count": restart_count,
                    "threshold": CASCADE_RESTART_THRESHOLD,
                    "window_seconds": CASCADE_RESTART_WINDOW_SECONDS,
                    "cooldown_seconds": CASCADE_COOLDOWN_SECONDS,
                    "triggered_by": triggered_by.value,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit circuit breaker event: {e}")

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """Get status of the cascade circuit breaker.

        Returns:
            Dict with breaker state, recent restart count, and timing info
        """
        current_time = time.time()

        # Count recent restarts
        cutoff = current_time - CASCADE_RESTART_WINDOW_SECONDS
        recent_restarts = [
            ts for ts in self._global_restart_timestamps if ts > cutoff
        ]

        result = {
            "breaker_open": self._cascade_breaker_open,
            "recent_restart_count": len(recent_restarts),
            "threshold": CASCADE_RESTART_THRESHOLD,
            "window_seconds": CASCADE_RESTART_WINDOW_SECONDS,
        }

        if self._cascade_breaker_open:
            elapsed = current_time - self._cascade_breaker_opened_at
            result["cooldown_remaining"] = max(0, CASCADE_COOLDOWN_SECONDS - elapsed)
        else:
            result["cooldown_remaining"] = 0

        return result

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
        health_check_interval: float | None = None,
        auto_restart: bool = True,
        max_restarts: int = 5,
        startup_grace_period: float | None = None,
    ) -> None:
        """Register a factory function for creating a daemon.

        Args:
            daemon_type: Type of daemon
            factory: Async function that runs the daemon
            depends_on: List of daemons that must be running first
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
                __import__(module_path)
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
                __import__(module_path)
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

    async def start(
        self, daemon_type: DaemonType, *, wait_for_deps: bool = True
    ) -> bool:
        """Start a specific daemon.

        Delegates to DaemonLifecycleManager (Dec 2025 extraction).

        Args:
            daemon_type: Type of daemon to start
            wait_for_deps: If True, wait for dependencies to be ready first

        Returns:
            True if started successfully
        """
        # Dec 2025: Ensure coordination events (including SyncRouter) are wired
        # before any daemon starts. This fixes the integration gap where
        # master_loop.py calls start() individually instead of start_all().
        await self._ensure_coordination_wired()

        # Dec 2025: Check memory pressure before spawning daemon
        # Log warning if memory is high, but don't block critical daemons
        if self._check_memory_pressure(threshold_percent=90.0):
            if daemon_type not in CRITICAL_DAEMONS:
                logger.warning(
                    f"Skipping non-critical daemon {daemon_type.value} due to memory pressure"
                )
                return False

        # Dec 27, 2025: Wait for dependencies before starting
        if wait_for_deps:
            await self._wait_for_dependencies(daemon_type)

        result = await self._lifecycle.start(daemon_type)
        if result:
            # Dec 2025 fix: Ensure health loop is running after any daemon starts
            # Previously only started via start_all() callback, causing health loop
            # to never start when master_loop.py called start() individually.
            await self._ensure_health_loop_running()

            # Dec 2025: Emit DAEMON_STARTED event for coordination_bootstrap handlers
            await self._emit_daemon_started(daemon_type)
        return result

    async def _emit_daemon_started(self, daemon_type: DaemonType) -> None:
        """Emit DAEMON_STARTED event after successful daemon start.

        December 2025: Wires the orphaned DAEMON_STARTED event that has
        handlers in coordination_bootstrap.py but was never emitted.
        """
        emit_started, _ = _get_daemon_event_emitters()
        if emit_started is None:
            return

        import socket
        try:
            await emit_started(
                daemon_name=daemon_type.value,
                hostname=socket.gethostname(),
                pid=os.getpid(),
                source="DaemonManager",
            )
            logger.debug(f"Emitted DAEMON_STARTED for {daemon_type.value}")
        except Exception as e:
            # Non-critical - log and continue
            logger.debug(f"Failed to emit DAEMON_STARTED: {e}")

    async def _ensure_health_loop_running(self) -> None:
        """Ensure the health monitoring loop is running.

        Dec 2025: Extracted from start_all() callback to allow individual
        start() calls to also start the health loop. This fixes an issue
        where master_loop.py calling start() individually would never start
        the health monitoring, causing crashed daemons to never be restarted.

        Dec 2025 update: Added crash detection and logging. If the health loop
        crashed (task.done() with exception), we log the exception before
        restarting to aid debugging.

        Safe to call multiple times - will only start health loop once.
        """
        if not self._health_task or self._health_task.done():
            # Check if the previous health loop crashed
            if self._health_task and self._health_task.done():
                try:
                    exception = self._health_task.exception()
                    if exception:
                        logger.error(
                            f"[DaemonManager] Health loop crashed, restarting: {exception}"
                        )
                except asyncio.CancelledError:
                    # Task was cancelled, not crashed
                    logger.debug("[DaemonManager] Health loop was cancelled")
                except asyncio.InvalidStateError:
                    # Task still pending (shouldn't happen after done() check)
                    pass

            self._running = True
            self._health_task = safe_create_task(
                self._health_loop(),
                name="daemon_health_loop"
            )
            logger.info("[DaemonManager] Started health monitoring loop")

    async def _wait_for_dependencies(self, daemon_type: DaemonType) -> None:
        """Wait for all dependencies of a daemon to be ready before starting.

        December 27, 2025: Added to fix startup order issues where daemons
        start before their dependencies are ready, causing lost events.

        Important: use the *currently registered* dependencies (DaemonInfo.depends_on)
        rather than a static registry lookup.

        This keeps production behavior the same (default daemons are registered from
        the declarative registry with their dependencies), while allowing tests to
        register ad-hoc daemons without accidentally inheriting registry dependencies.

        Args:
            daemon_type: Type of daemon being started
        """
        info = self._daemons.get(daemon_type)
        if info is None:
            return

        deps = list(info.depends_on or [])
        if not deps:
            return

        timeout = getattr(self.config, "dependency_wait_timeout", 30.0)

        for dep in deps:
            if not await self._wait_for_daemon_ready(dep, timeout=timeout):
                logger.warning(
                    f"[DaemonManager] Dependency {dep.name} not ready for {daemon_type.name}, "
                    "proceeding anyway (may lose early events)"
                )

    async def _wait_for_daemon_ready(
        self, daemon_type: DaemonType, timeout: float = 30.0
    ) -> bool:
        """Wait for a specific daemon to be running and healthy.

        December 27, 2025: Added to support dependency-based startup ordering.

        Args:
            daemon_type: Type of daemon to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            True if daemon is ready, False if timed out
        """
        import time

        poll_interval = getattr(self.config, "dependency_poll_interval", 0.5)

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_running(daemon_type):
                # Check health if the daemon supports it
                try:
                    health = await self.get_daemon_health(daemon_type)
                    status = health.get("status", "unknown")
                    if status in ("healthy", "ok", "running"):
                        logger.debug(
                            f"[DaemonManager] Dependency {daemon_type.name} is ready"
                        )
                        return True
                    # If health check exists but returns unhealthy, keep waiting
                    if status == "unhealthy":
                        await asyncio.sleep(poll_interval)
                        continue
                except (AttributeError, ValueError, RuntimeError, asyncio.TimeoutError):
                    # No health check available, just check running state
                    pass
                # Running but no health check or unknown status - consider ready
                logger.debug(
                    f"[DaemonManager] Dependency {daemon_type.name} is running (no health status)"
                )
                return True
            await asyncio.sleep(poll_interval)

        logger.warning(
            f"[DaemonManager] Timed out waiting {timeout}s for {daemon_type.name}"
        )
        return False

    def mark_daemon_ready(self, daemon_type: DaemonType) -> bool:
        """Explicitly mark a daemon as ready for dependent daemons.

        Daemons should call this after completing critical initialization.
        This is safer than relying on auto-ready (which triggers after 2s).

        Dec 2025: Added for explicit readiness signaling to prevent
        race conditions where dependent daemons start before their
        dependencies are truly initialized.

        Args:
            daemon_type: Type of daemon to mark as ready

        Returns:
            True if successfully marked, False if daemon not found
        """
        info = self._daemons.get(daemon_type)
        if info is None:
            logger.warning(f"Cannot mark {daemon_type.value} ready: daemon not found")
            return False

        if info.ready_event is None:
            logger.warning(f"Cannot mark {daemon_type.value} ready: no ready_event")
            return False

        if info.ready_event.is_set():
            logger.debug(f"{daemon_type.value} already marked as ready")
            return True

        info.ready_event.set()
        logger.info(f"{daemon_type.value} explicitly marked as ready")
        return True

    async def stop(self, daemon_type: DaemonType) -> bool:
        """Stop a specific daemon with timeout escalation.

        Delegates to DaemonLifecycleManager (Dec 2025 extraction).

        Args:
            daemon_type: Type of daemon to stop

        Returns:
            True if stopped successfully (or was already stopped)
        """
        result = await self._lifecycle.stop(daemon_type)
        if result:
            # Dec 2025: Emit DAEMON_STOPPED event for coordination_bootstrap handlers
            await self._emit_daemon_stopped(daemon_type, reason="normal")
        return result

    async def _emit_daemon_stopped(
        self, daemon_type: DaemonType, reason: str = "normal"
    ) -> None:
        """Emit DAEMON_STOPPED event after successful daemon stop.

        December 2025: Wires the orphaned DAEMON_STOPPED event that has
        handlers in coordination_bootstrap.py but was never emitted.
        """
        _, emit_stopped = _get_daemon_event_emitters()
        if emit_stopped is None:
            return

        import socket
        try:
            await emit_stopped(
                daemon_name=daemon_type.value,
                hostname=socket.gethostname(),
                reason=reason,
                source="DaemonManager",
            )
            logger.debug(f"Emitted DAEMON_STOPPED for {daemon_type.value}")
        except Exception as e:
            # Non-critical - log and continue
            logger.debug(f"Failed to emit DAEMON_STOPPED: {e}")

    async def restart_failed_daemon(
        self,
        daemon_type: DaemonType,
        force: bool = False,
    ) -> bool:
        """Restart a failed daemon, optionally resetting its restart count.

        Delegates to DaemonLifecycleManager (Dec 2025 extraction).

        Args:
            daemon_type: Type of daemon to restart
            force: If True, reset restart count and clear import error

        Returns:
            True if restart initiated successfully
        """
        return await self._lifecycle.restart_failed_daemon(daemon_type, force=force)

    async def start_all(self, types: list[DaemonType] | None = None) -> dict[DaemonType, bool]:
        """Start all (or specified) daemons in dependency order.

        Delegates core lifecycle to DaemonLifecycleManager (Dec 2025 extraction).
        DaemonManager-specific post-start hooks (health loop, watchdog, events)
        are passed via callback.

        Args:
            types: Specific daemon types to start (all if None)

        Returns:
            Dict mapping daemon type to start success
        """
        # Phase 8 (Dec 2025): Validate critical subsystems before starting
        validation_errors = self._validate_critical_subsystems()
        if validation_errors:
            logger.warning(
                f"[DaemonManager] Starting with {len(validation_errors)} subsystem validation error(s). "
                "Some daemons may fail to start."
            )

        # Define callback for DaemonManager-specific post-start operations
        async def _post_start_callback():
            # Start health check loop (uses centralized helper)
            await self._ensure_health_loop_running()

            # Start daemon watchdog for active monitoring
            try:
                from app.coordination.daemon_watchdog import start_watchdog
                safe_create_task(start_watchdog(), name="daemon_watchdog")
                logger.info("Daemon watchdog started")
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Failed to start daemon watchdog: {e}")

            # Phase 8 (Dec 2025): Wire ALL coordination event subscriptions at startup
            # This ensures daemons receive events they need before verification
            # Uses _ensure_coordination_wired for consistency with individual start() calls
            await self._ensure_coordination_wired()

            # Phase 5: Subscribe to REGRESSION_CRITICAL events for centralized handling
            await self._subscribe_to_critical_events()

            # Phase 5: Verify critical subscriptions are active
            await self._verify_subscriptions()

            # Phase 12 (Dec 2025): Emit readiness signal after critical daemons initialized
            # This closes the startup race condition where events were lost before handlers ready
            await self._emit_daemons_ready()

            # December 29, 2025: Start active circuit breaker probing
            # This allows circuits to recover faster when services become available
            # instead of waiting for the full recovery timeout
            try:
                from app.distributed.circuit_breaker import start_recovery_probing
                task = start_recovery_probing(interval=30.0)
                if task:
                    logger.info("[DaemonManager] Started circuit breaker recovery probing")
            except (ImportError, RuntimeError) as e:
                logger.debug(f"[DaemonManager] Circuit breaker probing not started: {e}")

        return await self._lifecycle.start_all(
            types=types,
            on_started_callback=_post_start_callback,
        )

    async def _subscribe_to_critical_events(self) -> None:
        """Subscribe to REGRESSION_CRITICAL and other critical events.

        Phase 5 (December 2025): Centralized handling of critical events
        that require daemon-level coordination response.

        Phase 7 (December 2025): Also wires AutoRollbackHandler to actually
        perform rollbacks when REGRESSION_CRITICAL events are received.
        """
        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()
            if router is None:
                logger.debug("[DaemonManager] Event router not available for critical event subscription")
                return

            # Subscribe to critical events
            router.subscribe(DataEventType.REGRESSION_CRITICAL.value, self._on_regression_critical)

            # P0.3 (December 2025): Subscribe to feedback loop events for daemon coordination
            # SELFPLAY_TARGET_UPDATED - adjust workload scaling based on priority changes
            if hasattr(DataEventType, 'SELFPLAY_TARGET_UPDATED'):
                router.subscribe(DataEventType.SELFPLAY_TARGET_UPDATED.value, self._on_selfplay_target_updated)
                logger.debug("[DaemonManager] Subscribed to SELFPLAY_TARGET_UPDATED")

            # EXPLORATION_BOOST - coordinate temperature adjustments across daemons
            if hasattr(DataEventType, 'EXPLORATION_BOOST'):
                router.subscribe(DataEventType.EXPLORATION_BOOST.value, self._on_exploration_boost)
                logger.debug("[DaemonManager] Subscribed to EXPLORATION_BOOST")

            # DAEMON_STATUS_CHANGED - self-healing when daemons fail
            if hasattr(DataEventType, 'DAEMON_STATUS_CHANGED'):
                router.subscribe(DataEventType.DAEMON_STATUS_CHANGED.value, self._on_daemon_status_changed)
                logger.debug("[DaemonManager] Subscribed to DAEMON_STATUS_CHANGED")

            # Dec 27, 2025: P2P cluster events for daemon lifecycle coordination
            # HOST_OFFLINE - pause affected daemons when nodes leave cluster
            if hasattr(DataEventType, 'HOST_OFFLINE'):
                router.subscribe(DataEventType.HOST_OFFLINE.value, self._on_host_offline)
                logger.debug("[DaemonManager] Subscribed to HOST_OFFLINE")

            # HOST_ONLINE - resume/restart daemons when nodes rejoin
            if hasattr(DataEventType, 'HOST_ONLINE'):
                router.subscribe(DataEventType.HOST_ONLINE.value, self._on_host_online)
                logger.debug("[DaemonManager] Subscribed to HOST_ONLINE")

            # LEADER_ELECTED - trigger leader-only daemons when leadership changes
            if hasattr(DataEventType, 'LEADER_ELECTED'):
                router.subscribe(DataEventType.LEADER_ELECTED.value, self._on_leader_elected)
                logger.debug("[DaemonManager] Subscribed to LEADER_ELECTED")

            # December 2025: Backpressure events for daemon workload coordination
            # BACKPRESSURE_ACTIVATED - pause non-essential daemons to reduce load
            if hasattr(DataEventType, 'BACKPRESSURE_ACTIVATED'):
                router.subscribe(DataEventType.BACKPRESSURE_ACTIVATED.value, self._on_backpressure_activated)
                logger.debug("[DaemonManager] Subscribed to BACKPRESSURE_ACTIVATED")

            # BACKPRESSURE_RELEASED - resume normal daemon operations
            if hasattr(DataEventType, 'BACKPRESSURE_RELEASED'):
                router.subscribe(DataEventType.BACKPRESSURE_RELEASED.value, self._on_backpressure_released)
                logger.debug("[DaemonManager] Subscribed to BACKPRESSURE_RELEASED")

            # December 2025: Disk space events for daemon workload coordination
            # DISK_SPACE_LOW - pause data-generating daemons when disk is low
            if hasattr(DataEventType, 'DISK_SPACE_LOW'):
                router.subscribe(DataEventType.DISK_SPACE_LOW.value, self._on_disk_space_low)
                logger.debug("[DaemonManager] Subscribed to DISK_SPACE_LOW")

            logger.info("[DaemonManager] Subscribed to critical events (Phase 5, P0.3, P2P cluster, backpressure, disk space)")

            # Phase 7: Wire AutoRollbackHandler to actually perform model rollbacks
            # Without this, REGRESSION_CRITICAL events are logged but no rollback happens
            try:
                from app.training.model_registry import get_model_registry
                from app.training.rollback_manager import wire_regression_to_rollback

                registry = get_model_registry()
                handler = wire_regression_to_rollback(registry)
                if handler:
                    logger.info("[DaemonManager] Wired AutoRollbackHandler for automatic model rollback (Phase 7)")
                else:
                    logger.warning("[DaemonManager] Failed to wire AutoRollbackHandler")
            except (ImportError, RuntimeError, AttributeError) as rollback_err:
                logger.warning(f"[DaemonManager] Could not wire AutoRollbackHandler: {rollback_err}")

        except (ImportError, RuntimeError, ConnectionError) as e:
            logger.warning(f"[DaemonManager] Failed to subscribe to critical events: {e}")

    async def _on_regression_critical(self, event) -> None:
        """Handle REGRESSION_CRITICAL event - centralized response.

        Phase 5 (December 2025): When a critical regression is detected,
        coordinate daemon-level response:
        1. Log the critical event prominently
        2. Pause selfplay for the affected config to prevent bad data
        3. Alert cluster nodes via P2P
        4. Trigger model rollback if configured

        Args:
            event: The REGRESSION_CRITICAL event
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", payload.get("config", "unknown"))
            model_id = payload.get("model_id", "unknown")
            elo_drop = payload.get("elo_drop", 0)
            current_elo = payload.get("current_elo", 0)
            previous_elo = payload.get("previous_elo", 0)

            # Prominent logging - this is critical
            logger.critical(
                f"[REGRESSION_CRITICAL] Model regression detected!\n"
                f"  Config: {config_key}\n"
                f"  Model: {model_id}\n"
                f"  ELO: {previous_elo:.0f} → {current_elo:.0f} (drop: {elo_drop:.0f})"
            )

            # Emit to P2P for cluster-wide awareness
            try:
                from app.coordination.event_router import get_router, DataEventType, DataEvent

                router = get_router()
                if router:
                    # Emit an alert event for monitoring/dashboards
                    alert_event = DataEvent(
                        event_type=DataEventType.HEALTH_ALERT,
                        payload={
                            "alert": "regression_critical",
                            "alert_type": "regression_critical",
                            "config_key": config_key,
                            "model_id": model_id,
                            "message": f"Critical model regression: {config_key} dropped {elo_drop:.0f} ELO",
                            "severity": "critical",
                        },
                        source="DaemonManager",
                    )
                    await router.publish_async(DataEventType.HEALTH_ALERT.value, alert_event)
            except (RuntimeError, OSError, ConnectionError) as alert_err:
                logger.debug(f"[DaemonManager] Failed to emit cluster alert: {alert_err}")

            # Check if rollback daemon is running and healthy
            if DaemonType.MODEL_DISTRIBUTION in self._daemons:
                info = self._daemons[DaemonType.MODEL_DISTRIBUTION]
                if info.state == DaemonState.RUNNING:
                    logger.info(
                        f"[DaemonManager] Model distribution daemon running - "
                        f"rollback should be handled by RollbackManager"
                    )

        except (RuntimeError, OSError, ConnectionError, ImportError) as e:
            logger.error(f"[DaemonManager] Error handling REGRESSION_CRITICAL: {e}")

    async def _on_selfplay_target_updated(self, event) -> None:
        """Handle SELFPLAY_TARGET_UPDATED event - adjust daemon workloads.

        P0.3 (December 2025): When selfplay targets change (due to priority
        shifts, feedback loop adjustments, or backpressure), coordinate
        daemon-level response:
        1. Log the target change for monitoring
        2. Adjust idle resource daemon behavior if needed
        3. Propagate priority to relevant daemons

        Args:
            event: The SELFPLAY_TARGET_UPDATED event with payload containing
                   config_key, priority, reason, target_jobs, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "unknown")
            priority = payload.get("priority", "normal")
            reason = payload.get("reason", "unknown")
            target_jobs = payload.get("target_jobs")

            logger.info(
                f"[DaemonManager] SELFPLAY_TARGET_UPDATED: {config_key} "
                f"priority={priority} reason={reason}"
                + (f" target_jobs={target_jobs}" if target_jobs else "")
            )

            # If priority is urgent or high, consider scaling up idle resource daemon
            if priority in ("urgent", "high"):
                if DaemonType.IDLE_RESOURCE in self._daemons:
                    info = self._daemons[DaemonType.IDLE_RESOURCE]
                    if info.state == DaemonState.RUNNING and hasattr(info, 'instance'):
                        # Signal to check for idle resources more frequently
                        daemon = info.instance
                        if hasattr(daemon, 'trigger_immediate_check'):
                            daemon.trigger_immediate_check()
                            logger.debug(
                                f"[DaemonManager] Triggered immediate idle resource check "
                                f"for {config_key}"
                            )

        except (RuntimeError, OSError, AttributeError) as e:
            logger.debug(f"[DaemonManager] Error handling SELFPLAY_TARGET_UPDATED: {e}")

    async def _on_exploration_boost(self, event) -> None:
        """Handle EXPLORATION_BOOST event - coordinate temperature adjustments.

        P0.3 (December 2025): When exploration boost is requested (due to
        training stalls, Elo plateau, or quality issues), coordinate
        daemon-level response:
        1. Log the boost for monitoring
        2. Propagate to selfplay daemons for temperature adjustment
        3. Track boost duration for expiry

        Args:
            event: The EXPLORATION_BOOST event with payload containing
                   config_key, boost_factor, reason, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "unknown")
            boost_factor = payload.get("boost_factor", 1.0)
            reason = payload.get("reason", "unknown")
            duration_seconds = payload.get("duration_seconds", 3600)

            logger.info(
                f"[DaemonManager] EXPLORATION_BOOST: {config_key} "
                f"factor={boost_factor:.2f}x reason={reason} "
                f"duration={duration_seconds}s"
            )

            # Propagate boost to selfplay scheduler if available
            if DaemonType.SELFPLAY_SCHEDULER in self._daemons:
                info = self._daemons[DaemonType.SELFPLAY_SCHEDULER]
                if info.state == DaemonState.RUNNING and hasattr(info, 'instance'):
                    scheduler = info.instance
                    if hasattr(scheduler, 'apply_exploration_boost'):
                        scheduler.apply_exploration_boost(
                            config_key=config_key,
                            boost_factor=boost_factor,
                            duration_seconds=duration_seconds,
                        )
                        logger.debug(
                            f"[DaemonManager] Applied exploration boost to SelfplayScheduler"
                        )

        except (RuntimeError, OSError, AttributeError) as e:
            logger.debug(f"[DaemonManager] Error handling EXPLORATION_BOOST: {e}")

    async def _on_daemon_status_changed(self, event) -> None:
        """Handle DAEMON_STATUS_CHANGED event - self-healing response.

        P0.3 (December 2025): When a daemon status changes (failure, restart,
        health degradation), coordinate daemon-level response:
        1. Log the status change
        2. Attempt restart for failed daemons if configured
        3. Update cluster health awareness

        Args:
            event: The DAEMON_STATUS_CHANGED event with payload containing
                   daemon_type, old_status, new_status, reason, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            daemon_type_str = payload.get("daemon_type", "unknown")
            old_status = payload.get("old_status", "unknown")
            new_status = payload.get("new_status", "unknown")
            reason = payload.get("reason", "")

            logger.info(
                f"[DaemonManager] DAEMON_STATUS_CHANGED: {daemon_type_str} "
                f"{old_status} -> {new_status}"
                + (f" ({reason})" if reason else "")
            )

            # Self-healing: attempt restart for failed daemons
            if new_status in ("FAILED", "CRASHED", "STOPPED"):
                try:
                    daemon_type = DaemonType(daemon_type_str)
                    if daemon_type in self._daemons:
                        info = self._daemons[daemon_type]

                        # Only restart if we haven't exceeded retry limits
                        max_restarts = 3
                        if info.restart_count < max_restarts:
                            logger.warning(
                                f"[DaemonManager] Attempting self-healing restart for "
                                f"{daemon_type_str} (attempt {info.restart_count + 1}/{max_restarts})"
                            )
                            # Schedule restart via lifecycle manager
                            if hasattr(self, '_lifecycle') and self._lifecycle:
                                await self._lifecycle.restart_daemon(daemon_type)
                        else:
                            logger.error(
                                f"[DaemonManager] Daemon {daemon_type_str} exceeded max restarts "
                                f"({max_restarts}), not attempting self-healing"
                            )
                except ValueError:
                    logger.debug(f"[DaemonManager] Unknown daemon type: {daemon_type_str}")

        except (RuntimeError, OSError, AttributeError) as e:
            logger.debug(f"[DaemonManager] Error handling DAEMON_STATUS_CHANGED: {e}")

    async def _on_host_offline(self, event) -> None:
        """Handle HOST_OFFLINE event - pause affected daemons when nodes leave cluster.

        Dec 27, 2025: P2P cluster event integration. When a host goes offline,
        pause daemons that depend on that host (e.g., sync daemons targeting that host,
        distribution daemons, etc.).

        Args:
            event: The HOST_OFFLINE event with payload containing host_id, reason, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            host_id = payload.get("host_id", "unknown")
            reason = payload.get("reason", "")

            logger.info(
                f"[DaemonManager] HOST_OFFLINE: {host_id}"
                + (f" ({reason})" if reason else "")
            )

            # Notify sync-related daemons to exclude this host
            for daemon_type in [DaemonType.AUTO_SYNC, DaemonType.MODEL_DISTRIBUTION]:
                if daemon_type in self._daemons:
                    info = self._daemons[daemon_type]
                    if info.state == DaemonState.RUNNING and hasattr(info, 'instance'):
                        daemon = info.instance
                        if hasattr(daemon, 'mark_host_offline'):
                            daemon.mark_host_offline(host_id)
                            logger.debug(
                                f"[DaemonManager] Marked host {host_id} offline for {daemon_type.value}"
                            )

        except (RuntimeError, OSError, AttributeError, KeyError) as e:
            logger.debug(f"[DaemonManager] Error handling HOST_OFFLINE: {e}")

    async def _on_host_online(self, event) -> None:
        """Handle HOST_ONLINE event - resume/restart daemons when nodes rejoin.

        Dec 27, 2025: P2P cluster event integration. When a host comes back online,
        resume daemons and re-include the host in sync targets.

        Args:
            event: The HOST_ONLINE event with payload containing host_id, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            host_id = payload.get("host_id", "unknown")

            logger.info(f"[DaemonManager] HOST_ONLINE: {host_id}")

            # Notify sync-related daemons to re-include this host
            for daemon_type in [DaemonType.AUTO_SYNC, DaemonType.MODEL_DISTRIBUTION]:
                if daemon_type in self._daemons:
                    info = self._daemons[daemon_type]
                    if info.state == DaemonState.RUNNING and hasattr(info, 'instance'):
                        daemon = info.instance
                        if hasattr(daemon, 'mark_host_online'):
                            daemon.mark_host_online(host_id)
                            logger.debug(
                                f"[DaemonManager] Marked host {host_id} online for {daemon_type.value}"
                            )

        except (RuntimeError, OSError, AttributeError, KeyError) as e:
            logger.debug(f"[DaemonManager] Error handling HOST_ONLINE: {e}")

    async def _on_leader_elected(self, event) -> None:
        """Handle LEADER_ELECTED event - trigger leader-only daemons when leadership changes.

        Dec 27, 2025: P2P cluster event integration. When a new leader is elected:
        1. If we are the new leader: start leader-only daemons
        2. If we lost leadership: stop leader-only daemons
        3. Update all daemons with new leader info

        Args:
            event: The LEADER_ELECTED event with payload containing leader_id,
                   previous_leader_id, is_self, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            leader_id = payload.get("leader_id", "unknown")
            previous_leader_id = payload.get("previous_leader_id", "")
            is_self = payload.get("is_self", False)

            logger.info(
                f"[DaemonManager] LEADER_ELECTED: {leader_id}"
                + (f" (previous: {previous_leader_id})" if previous_leader_id else "")
                + (" [THIS NODE]" if is_self else "")
            )

            # Leader-only daemons that should only run on the leader node
            leader_only_daemons = [
                DaemonType.DATA_PIPELINE,
                DaemonType.AUTO_PROMOTION,
                DaemonType.EVALUATION,
                DaemonType.TRAINING_TRIGGER,
            ]

            if is_self:
                # We became the leader - start leader-only daemons
                for daemon_type in leader_only_daemons:
                    if daemon_type in self._daemons:
                        info = self._daemons[daemon_type]
                        if info.state != DaemonState.RUNNING:
                            logger.info(
                                f"[DaemonManager] Starting leader-only daemon: {daemon_type.value}"
                            )
                            try:
                                await self.start(daemon_type)
                            except (RuntimeError, OSError) as e:
                                logger.warning(
                                    f"[DaemonManager] Failed to start {daemon_type.value}: {e}"
                                )
            else:
                # We lost leadership or another node became leader - stop leader-only daemons
                for daemon_type in leader_only_daemons:
                    if daemon_type in self._daemons:
                        info = self._daemons[daemon_type]
                        if info.state == DaemonState.RUNNING:
                            logger.info(
                                f"[DaemonManager] Stopping leader-only daemon: {daemon_type.value}"
                            )
                            try:
                                await self.stop(daemon_type)
                            except (RuntimeError, OSError) as e:
                                logger.warning(
                                    f"[DaemonManager] Failed to stop {daemon_type.value}: {e}"
                                )

            # Notify all running daemons of the leadership change
            for daemon_type, info in self._daemons.items():
                if info.state == DaemonState.RUNNING and hasattr(info, 'instance'):
                    daemon = info.instance
                    if hasattr(daemon, 'on_leader_changed'):
                        daemon.on_leader_changed(leader_id=leader_id, is_self=is_self)

        except (RuntimeError, OSError, AttributeError, KeyError) as e:
            logger.debug(f"[DaemonManager] Error handling LEADER_ELECTED: {e}")

    async def _on_backpressure_activated(self, event) -> None:
        """Handle BACKPRESSURE_ACTIVATED event - reduce daemon workload.

        December 2025: When backpressure is activated (queue depth exceeds threshold
        or resource contention detected), pause non-essential daemons to reduce load.

        Essential daemons that must keep running:
        - EVENT_ROUTER - core event bus (always needed)
        - DAEMON_WATCHDOG - health monitoring (always needed)
        - QUEUE_MONITOR - needs to track when to release backpressure
        - CLUSTER_WATCHDOG - cluster health (always needed)

        Non-essential daemons that can be paused:
        - IDLE_RESOURCE - spawns new selfplay jobs
        - SELFPLAY_COORDINATOR - schedules selfplay
        - TRAINING_ACTIVITY - detects training (informational)
        - AUTO_SYNC - can wait until backpressure clears

        Args:
            event: The BACKPRESSURE_ACTIVATED event with payload containing
                   reason, threshold, current_value, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            reason = payload.get("reason", "unknown")
            threshold = payload.get("threshold", 0)
            current_value = payload.get("current_value", 0)

            logger.warning(
                f"[DaemonManager] BACKPRESSURE_ACTIVATED: {reason} "
                f"(value: {current_value}, threshold: {threshold})"
            )

            # Non-essential daemons that can be paused during backpressure
            pausable_daemons = [
                DaemonType.IDLE_RESOURCE,
                DaemonType.SELFPLAY_COORDINATOR,
                DaemonType.TRAINING_ACTIVITY,
                DaemonType.AUTO_SYNC,
            ]

            paused_count = 0
            for daemon_type in pausable_daemons:
                if daemon_type in self._daemons:
                    info = self._daemons[daemon_type]
                    if info.state == DaemonState.RUNNING:
                        # Mark as paused (don't fully stop - just suspend work)
                        daemon = getattr(info, 'instance', None)
                        if daemon and hasattr(daemon, 'pause'):
                            try:
                                daemon.pause()
                                paused_count += 1
                                logger.debug(
                                    f"[DaemonManager] Paused {daemon_type.value} due to backpressure"
                                )
                            except (RuntimeError, OSError) as e:
                                logger.debug(
                                    f"[DaemonManager] Failed to pause {daemon_type.value}: {e}"
                                )

            if paused_count > 0:
                logger.info(
                    f"[DaemonManager] Paused {paused_count} daemons due to backpressure"
                )

        except (RuntimeError, OSError, AttributeError, KeyError) as e:
            logger.debug(f"[DaemonManager] Error handling BACKPRESSURE_ACTIVATED: {e}")

    async def _on_backpressure_released(self, event) -> None:
        """Handle BACKPRESSURE_RELEASED event - resume normal daemon operations.

        December 2025: When backpressure is released (queue depth dropped below
        threshold and resources recovered), resume paused daemons.

        Args:
            event: The BACKPRESSURE_RELEASED event with payload containing
                   duration_seconds, peak_value, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            duration = payload.get("duration_seconds", 0)

            logger.info(
                f"[DaemonManager] BACKPRESSURE_RELEASED after {duration:.1f}s"
            )

            # Resume daemons that were paused
            resumable_daemons = [
                DaemonType.IDLE_RESOURCE,
                DaemonType.SELFPLAY_COORDINATOR,
                DaemonType.TRAINING_ACTIVITY,
                DaemonType.AUTO_SYNC,
            ]

            resumed_count = 0
            for daemon_type in resumable_daemons:
                if daemon_type in self._daemons:
                    info = self._daemons[daemon_type]
                    daemon = getattr(info, 'instance', None)
                    if daemon and hasattr(daemon, 'resume'):
                        try:
                            daemon.resume()
                            resumed_count += 1
                            logger.debug(
                                f"[DaemonManager] Resumed {daemon_type.value} after backpressure"
                            )
                        except (RuntimeError, OSError) as e:
                            logger.debug(
                                f"[DaemonManager] Failed to resume {daemon_type.value}: {e}"
                            )

            if resumed_count > 0:
                logger.info(
                    f"[DaemonManager] Resumed {resumed_count} daemons after backpressure release"
                )

        except (RuntimeError, OSError, AttributeError, KeyError) as e:
            logger.debug(f"[DaemonManager] Error handling BACKPRESSURE_RELEASED: {e}")

    async def _on_disk_space_low(self, event) -> None:
        """Handle DISK_SPACE_LOW event - pause data-generating daemons.

        December 2025: When disk space is low, pause daemons that generate data
        to prevent disk from filling up completely. This gives the cleanup
        daemons (MaintenanceDaemon, DiskSpaceManagerDaemon) time to free space.

        Daemons paused during low disk:
        - SELFPLAY_COORDINATOR - generates game data
        - IDLE_RESOURCE - spawns selfplay jobs
        - AUTO_SYNC - downloads data from other nodes
        - TRAINING_ACTIVITY - may trigger training data export

        Args:
            event: The DISK_SPACE_LOW event with payload containing
                   host, usage_percent, free_gb, threshold
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            host = payload.get("host", "unknown")
            usage_percent = payload.get("usage_percent", 0)
            free_gb = payload.get("free_gb", 0)
            threshold = payload.get("threshold", 70)

            logger.warning(
                f"[DaemonManager] DISK_SPACE_LOW: {host} at {usage_percent:.1f}% "
                f"(threshold: {threshold}%, free: {free_gb:.1f}GB)"
            )

            # Only respond to events for this host
            import socket
            local_hostname = socket.gethostname()
            if host not in (local_hostname, "localhost", "127.0.0.1"):
                logger.debug(f"[DaemonManager] Ignoring disk event for other host: {host}")
                return

            # If disk usage is critical (>85%), pause data-generating daemons
            if usage_percent >= 85:
                logger.warning(
                    f"[DaemonManager] Critical disk usage ({usage_percent:.1f}%), "
                    f"pausing data-generating daemons"
                )

                data_generating_daemons = [
                    DaemonType.SELFPLAY_COORDINATOR,
                    DaemonType.IDLE_RESOURCE,
                    DaemonType.AUTO_SYNC,
                    DaemonType.TRAINING_ACTIVITY,
                ]

                paused_count = 0
                for daemon_type in data_generating_daemons:
                    if daemon_type in self._daemons:
                        info = self._daemons[daemon_type]
                        if info.state == DaemonState.RUNNING:
                            daemon = getattr(info, 'instance', None)
                            if daemon and hasattr(daemon, 'pause'):
                                try:
                                    daemon.pause()
                                    paused_count += 1
                                    logger.warning(
                                        f"[DaemonManager] Paused {daemon_type.value} "
                                        f"due to low disk space"
                                    )
                                except (RuntimeError, OSError) as e:
                                    logger.debug(
                                        f"[DaemonManager] Failed to pause {daemon_type.value}: {e}"
                                    )

                if paused_count > 0:
                    logger.warning(
                        f"[DaemonManager] Paused {paused_count} data-generating daemons "
                        f"due to critical disk space"
                    )

        except (RuntimeError, OSError, AttributeError, KeyError) as e:
            logger.debug(f"[DaemonManager] Error handling DISK_SPACE_LOW: {e}")

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
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()
            if router is None:
                logger.debug("[DaemonManager] Event router not available for readiness signal")
                return

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
                "timestamp": __import__("time").time(),
                "fully_ready": critical_ready == len(critical_types),
            }

            # Check if ALL_CRITICAL_DAEMONS_READY event type exists
            if hasattr(DataEventType, "ALL_CRITICAL_DAEMONS_READY"):
                router.publish(
                    DataEventType.ALL_CRITICAL_DAEMONS_READY.value,
                    event_data,
                )
                logger.info(
                    f"[DaemonManager] Emitted ALL_CRITICAL_DAEMONS_READY: "
                    f"{len(ready_daemons)} daemons ready, {critical_ready}/{len(critical_types)} critical"
                )
            else:
                # Fallback: emit as generic SYSTEM_STATUS event
                router.publish(
                    "system.daemons_ready",
                    event_data,
                )
                logger.info(
                    f"[DaemonManager] Emitted system.daemons_ready: "
                    f"{len(ready_daemons)} daemons ready"
                )

        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"[DaemonManager] Failed to emit readiness signal: {e}")

    async def stop_all(self) -> dict[DaemonType, bool]:
        """Stop all running daemons.

        Delegates to DaemonLifecycleManager (Dec 2025 extraction).

        Returns:
            Dict mapping daemon type to stop success
        """
        return await self._lifecycle.stop_all()

    async def shutdown(self) -> None:
        """Gracefully shutdown all daemons.

        Delegates to DaemonLifecycleManager (Dec 2025 extraction).
        DaemonManager-specific pre-shutdown hooks (watchdog) are passed via callback.
        """
        # Define callback for DaemonManager-specific pre-shutdown operations
        async def _pre_shutdown_callback():
            # December 29, 2025: Stop circuit breaker recovery probing
            try:
                from app.distributed.circuit_breaker import stop_recovery_probing
                stop_recovery_probing()
            except (ImportError, RuntimeError):
                pass

            try:
                from app.coordination.daemon_watchdog import stop_watchdog
                await stop_watchdog()
            except (ImportError, RuntimeError, AttributeError) as e:
                logger.debug(f"Watchdog stop error (expected if not started): {e}")

        await self._lifecycle.shutdown(
            health_task=self._health_task,
            pre_shutdown_callback=_pre_shutdown_callback,
        )

    def _sync_shutdown(self) -> None:
        """Synchronous shutdown for atexit."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                fire_and_forget(
                    self.shutdown(),
                    name="daemon_manager_atexit_shutdown",
                )
        except RuntimeError:
            # No running loop
            pass

    def _sort_by_dependencies(self, types: list[DaemonType]) -> list[DaemonType]:
        """Sort daemon types by dependencies (topological sort).

        Delegates to DaemonLifecycleManager (Dec 2025 extraction).
        """
        return self._lifecycle._sort_by_dependencies(types)

    def _get_dependents(self, daemon_type: DaemonType) -> list[DaemonType]:
        """Get all daemons that depend on the given daemon type.

        Delegates to DaemonLifecycleManager (Dec 2025 extraction).
        """
        return self._lifecycle._get_dependents(daemon_type)

    async def _health_loop(self) -> None:
        """Background health check loop."""
        while self._running and not self._shutdown_event.is_set():
            try:
                await self._check_health()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError) as e:
                logger.error(f"Health check error: {e}")

    async def _check_health(self) -> None:
        """Check health of all daemons and attempt recovery of FAILED ones.

        Note: Acquires self._lock to prevent race conditions with start/stop/register.
        Restarts are done outside the lock to avoid deadlock since start() also acquires lock.
        """
        daemons_to_restart: list[DaemonType] = []

        async with self._lock:
            current_time = time.time()

            for daemon_type, info in list(self._daemons.items()):
                # Attempt recovery of FAILED daemons after cooldown period
                if info.state == DaemonState.FAILED:
                    # Skip daemons with import errors - they can't be recovered
                    if info.import_error:
                        continue

                    time_since_failure = current_time - info.last_failure_time
                    if time_since_failure >= self.config.recovery_cooldown:
                        logger.info(
                            f"Attempting recovery of {daemon_type.value} after "
                            f"{time_since_failure:.0f}s cooldown"
                        )
                        # Reset restart count to allow recovery attempts
                        info.restart_count = 0
                        info.state = DaemonState.STOPPED  # Reset state before restart
                        daemons_to_restart.append(daemon_type)
                    continue

                if info.state != DaemonState.RUNNING:
                    continue

                # December 2025: Skip health checks during startup grace period
                # Slow-starting daemons (e.g., loading large state files) need time
                # to initialize before health checks begin
                uptime = current_time - info.start_time
                if uptime < info.startup_grace_period:
                    logger.debug(
                        f"Skipping health check for {daemon_type.value}: "
                        f"in startup grace period ({uptime:.0f}s / {info.startup_grace_period:.0f}s)"
                    )
                    continue

                # Check if task is still alive
                if info.task is None or info.task.done():
                    if info.task and info.task.exception():
                        info.last_error = str(info.task.exception())
                        info.last_failure_time = current_time

                    if self.config.auto_restart_failed and info.restart_count < info.max_restarts:
                        logger.warning(f"{daemon_type.value} died, restarting...")
                        daemons_to_restart.append(daemon_type)
                    else:
                        info.state = DaemonState.FAILED
                        info.last_failure_time = current_time
                    continue

                # December 2025: Call daemon's health_check() if available
                # This enables daemon-specific health monitoring beyond just task liveness
                # December 2025 (P1): Add timeout protection to prevent health loop hanging
                # December 2025: Enhanced to handle both sync and async methods with timeout
                if info.instance is not None and hasattr(info.instance, 'health_check'):
                    try:
                        health_check_timeout = 5.0  # seconds
                        health_method = info.instance.health_check
                        loop = asyncio.get_event_loop()

                        # Check if health_check is a coroutine function (async def)
                        if asyncio.iscoroutinefunction(health_method):
                            # Async health check - call and await with timeout
                            try:
                                health_result = await asyncio.wait_for(
                                    health_method(), timeout=health_check_timeout
                                )
                            except asyncio.TimeoutError:
                                logger.warning(
                                    f"{daemon_type.value} async health_check() timed out "
                                    f"({health_check_timeout}s)"
                                )
                                health_result = {
                                    "healthy": False,
                                    "message": f"health_check() timeout ({health_check_timeout}s)",
                                }
                        else:
                            # Sync health check - run in executor with timeout
                            # This prevents blocking sync calls from hanging the health loop
                            try:
                                health_result = await asyncio.wait_for(
                                    loop.run_in_executor(None, health_method),
                                    timeout=health_check_timeout
                                )
                            except asyncio.TimeoutError:
                                logger.warning(
                                    f"{daemon_type.value} sync health_check() timed out "
                                    f"({health_check_timeout}s)"
                                )
                                health_result = {
                                    "healthy": False,
                                    "message": f"health_check() timeout ({health_check_timeout}s)",
                                }

                        # Handle case where sync method returns a coroutine (unusual but possible)
                        if asyncio.iscoroutine(health_result):
                            try:
                                health_result = await asyncio.wait_for(
                                    health_result, timeout=health_check_timeout
                                )
                            except asyncio.TimeoutError:
                                logger.warning(
                                    f"{daemon_type.value} returned coroutine timed out "
                                    f"({health_check_timeout}s)"
                                )
                                health_result = {
                                    "healthy": False,
                                    "message": f"health_check() timeout ({health_check_timeout}s)",
                                }
                        elif callable(getattr(health_result, '__await__', None)):
                            # Awaitable object (not coroutine but can be awaited)
                            try:
                                health_result = await asyncio.wait_for(
                                    health_result, timeout=health_check_timeout
                                )
                            except asyncio.TimeoutError:
                                logger.warning(
                                    f"{daemon_type.value} awaitable health_check() timed out "
                                    f"({health_check_timeout}s)"
                                )
                                health_result = {
                                    "healthy": False,
                                    "message": f"health_check() timeout ({health_check_timeout}s)",
                                }

                        # Check if result indicates unhealthy state
                        is_healthy = True
                        if hasattr(health_result, 'healthy'):
                            is_healthy = health_result.healthy
                        elif isinstance(health_result, dict):
                            is_healthy = health_result.get('healthy', True)
                        elif isinstance(health_result, bool):
                            is_healthy = health_result

                        if not is_healthy:
                            message = ""
                            if hasattr(health_result, 'message'):
                                message = health_result.message
                            elif isinstance(health_result, dict):
                                message = health_result.get('message', 'unhealthy')

                            logger.warning(
                                f"{daemon_type.value} health check failed: {message}"
                            )
                            info.last_error = f"Health check failed: {message}"
                            # Mark for restart if auto-restart is enabled
                            if self.config.auto_restart_failed and info.restart_count < info.max_restarts:
                                daemons_to_restart.append(daemon_type)
                            else:
                                info.state = DaemonState.FAILED
                                info.last_failure_time = current_time
                    except (RuntimeError, OSError, AttributeError) as e:
                        logger.debug(
                            f"Error calling health_check for {daemon_type.value}: {e}"
                        )

        # Handle restarts outside lock to prevent deadlock (start() also acquires lock)
        # Dec 2025: Check cascade circuit breaker first - if too many restarts are happening
        # globally, pause all restarts to let the system stabilize
        if not daemons_to_restart:
            return  # Nothing to restart

        if not self._check_cascade_circuit_breaker():
            # Circuit breaker is open - skip all restarts this cycle
            logger.warning(
                f"[DaemonManager] Skipping {len(daemons_to_restart)} daemon restart(s) "
                f"due to cascade circuit breaker: {[d.value for d in daemons_to_restart]}"
            )
            return

        # Also cascade restart to dependent daemons when a dependency fails
        all_to_restart: set[DaemonType] = set(daemons_to_restart)
        for daemon_type in daemons_to_restart:
            dependents = self._get_dependents(daemon_type)
            if dependents:
                logger.info(
                    f"Cascading restart: {daemon_type.value} failed, "
                    f"also restarting {len(dependents)} dependents: "
                    f"{[d.value for d in dependents]}"
                )
                all_to_restart.update(dependents)

        # Restart in dependency order (dependencies first, then dependents)
        # Dec 29, 2025: Fix - stop unhealthy daemons before restarting.
        # Without this, start() returns early for RUNNING daemons without restarting.
        sorted_restarts = self._sort_by_dependencies(list(all_to_restart))
        for daemon_type in sorted_restarts:
            # Dec 2025: Record global restart for circuit breaker tracking
            self._record_global_restart(daemon_type)

            # Check if circuit breaker tripped mid-batch (could happen with large cascades)
            if self._cascade_breaker_open:
                logger.warning(
                    f"[DaemonManager] Circuit breaker tripped mid-restart - "
                    f"stopping restart batch"
                )
                break

            # First stop the daemon if it's still running (unhealthy but not crashed)
            info = self._daemons.get(daemon_type)
            if info and info.state == DaemonState.RUNNING:
                logger.info(
                    f"Stopping unhealthy daemon {daemon_type.value} before restart"
                )
                await self.stop(daemon_type)
            # Now start (or restart) the daemon
            await self.start(daemon_type)

    def get_status(self) -> dict[str, Any]:
        """Get status of all daemons.

        Returns:
            Status dict with daemon states and metrics
        """
        daemon_status = {}
        for daemon_type, info in self._daemons.items():
            daemon_status[daemon_type.value] = {
                "state": info.state.value,
                "uptime_seconds": info.uptime_seconds,
                "restart_count": info.restart_count,
                "last_error": info.last_error,
                "auto_restart": info.auto_restart,
            }

        running_count = sum(1 for i in self._daemons.values() if i.state == DaemonState.RUNNING)
        failed_count = sum(1 for i in self._daemons.values() if i.state == DaemonState.FAILED)

        return {
            "running": self._running,
            "daemons": daemon_status,
            "summary": {
                "total": len(self._daemons),
                "running": running_count,
                "failed": failed_count,
                "stopped": len(self._daemons) - running_count - failed_count,
            },
            # Dec 2025: Include cascade circuit breaker status
            "circuit_breaker": self.get_circuit_breaker_status(),
        }

    def health_check(self) -> "HealthCheckResult":
        """Perform health check (CoordinatorProtocol compliance).

        Returns standardized HealthCheckResult for unified monitoring.
        DaemonManager is healthy if it's running and has few failed daemons.

        Returns:
            HealthCheckResult with health status and details
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        running_count = sum(
            1 for i in self._daemons.values() if i.state == DaemonState.RUNNING
        )
        failed_count = sum(
            1 for i in self._daemons.values() if i.state == DaemonState.FAILED
        )
        total = len(self._daemons)

        # Healthy if running and not too many failures
        is_healthy = self._running and (
            total == 0 or failed_count < max(1, total * 0.2)
        )

        if is_healthy:
            status = CoordinatorStatus.RUNNING
            message = ""
        elif self._running and failed_count >= max(1, total * 0.2):
            status = CoordinatorStatus.DEGRADED
            message = f"High failure rate: {failed_count}/{total} daemons failed"
        else:
            status = CoordinatorStatus.STOPPED
            message = "DaemonManager not running"

        # Dec 2025: Include memory pressure info
        memory_info = self._get_memory_info()

        return HealthCheckResult(
            healthy=is_healthy,
            status=status,
            message=message,
            details={
                "running": self._running,
                "daemons_total": total,
                "daemons_running": running_count,
                "daemons_failed": failed_count,
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "memory_percent": memory_info.get("percent", 0),
                "memory_available_gb": memory_info.get("available_gb", 0),
            },
        )

    def _get_memory_info(self) -> dict[str, float]:
        """Get current memory usage info using psutil.

        Returns:
            Dict with memory stats: percent (used), available_gb
        """
        try:
            import psutil

            mem = psutil.virtual_memory()
            return {
                "percent": round(mem.percent, 1),
                "available_gb": round(mem.available / (1024**3), 2),
            }
        except ImportError:
            # psutil not available, skip memory monitoring
            return {"percent": 0, "available_gb": 0}
        except Exception as e:
            logger.debug(f"Memory info unavailable: {e}")
            return {"percent": 0, "available_gb": 0}

    def _check_memory_pressure(self, threshold_percent: float = 90.0) -> bool:
        """Check if system is under memory pressure.

        December 2025: Added to prevent spawning daemons when memory is low.

        Args:
            threshold_percent: Memory usage threshold (default 90%)

        Returns:
            True if memory pressure is HIGH (should not spawn more daemons)
        """
        memory_info = self._get_memory_info()
        if memory_info["percent"] >= threshold_percent:
            logger.warning(
                f"Memory pressure HIGH: {memory_info['percent']:.1f}% used, "
                f"{memory_info['available_gb']:.1f}GB available"
            )
            # Emit RESOURCE_CONSTRAINT event
            self._emit_memory_constraint(memory_info)
            return True
        return False

    def _emit_memory_constraint(self, memory_info: dict[str, float]) -> None:
        """Emit RESOURCE_CONSTRAINT event for memory pressure.

        December 2025: Integrated with event system for pipeline coordination.
        """
        try:
            from app.coordination.event_router import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            event = DataEvent(
                event_type=DataEventType.RESOURCE_CONSTRAINT,
                payload={
                    "constraint_type": "memory",
                    "memory_percent": memory_info.get("percent", 0),
                    "available_gb": memory_info.get("available_gb", 0),
                    "source": "daemon_manager",
                },
                source="DaemonManager",
            )

            import asyncio

            bus = get_event_bus()
            try:
                loop = asyncio.get_running_loop()
                fire_and_forget(bus.publish(event), name="memory_constraint_event")
            except RuntimeError:
                # Not in async context - cannot emit event
                logger.debug("Skipping memory constraint event - not in async context")
        except Exception as e:
            logger.debug(f"Best-effort memory constraint event failed: {e}")

    def is_running(self, daemon_type: DaemonType) -> bool:
        """Check if a daemon is running."""
        info = self._daemons.get(daemon_type)
        return info is not None and info.state == DaemonState.RUNNING

    def get_daemon_info(self, daemon_type: DaemonType) -> DaemonInfo | None:
        """Get daemon info by type (public API).

        Used by DaemonWatchdog to inspect daemon state and detect stuck tasks.

        Args:
            daemon_type: The type of daemon to look up

        Returns:
            DaemonInfo if daemon is registered, None otherwise
        """
        return self._daemons.get(daemon_type)

    # =========================================================================
    # Lifecycle Tracking Methods (December 2025)
    # =========================================================================

    def get_lifecycle_summary(self) -> dict[str, Any]:
        """Get aggregated lifecycle statistics for all daemons.

        Returns:
            Dict with total restarts, average uptime, oldest/newest daemon info
        """
        total_restarts = sum(info.restart_count for info in self._daemons.values())
        uptimes = [
            info.uptime_seconds
            for info in self._daemons.values()
            if info.state == DaemonState.RUNNING
        ]
        avg_uptime = sum(uptimes) / len(uptimes) if uptimes else 0.0
        max_uptime = max(uptimes) if uptimes else 0.0
        min_uptime = min(uptimes) if uptimes else 0.0

        # Find daemon with most restarts
        most_restarts = max(
            self._daemons.values(),
            key=lambda i: i.restart_count,
            default=None,
        )

        return {
            "manager_uptime_seconds": time.time() - self._start_time,
            "total_restarts": total_restarts,
            "average_uptime_seconds": round(avg_uptime, 1),
            "max_uptime_seconds": round(max_uptime, 1),
            "min_uptime_seconds": round(min_uptime, 1),
            "most_restarts_daemon": most_restarts.daemon_type.value if most_restarts else None,
            "most_restarts_count": most_restarts.restart_count if most_restarts else 0,
        }

    def get_failed_daemons(self) -> list[tuple[DaemonType, str | None]]:
        """Get list of currently failed daemons with their error messages.

        Returns:
            List of (DaemonType, error_message) tuples for failed daemons
        """
        return [
            (info.daemon_type, info.last_error)
            for info in self._daemons.values()
            if info.state == DaemonState.FAILED
        ]

    def get_recent_restarts(self, within_seconds: float = 300.0) -> list[DaemonType]:
        """Get list of daemons that restarted recently.

        Args:
            within_seconds: Time window to check (default 5 minutes)

        Returns:
            List of DaemonTypes that have restarted within the window
        """
        cutoff = time.time() - within_seconds
        recent = []
        for daemon_name, timestamps in self._restart_timestamps.items():
            if any(ts > cutoff for ts in timestamps):
                try:
                    daemon_type = DaemonType(daemon_name)
                    recent.append(daemon_type)
                except ValueError:
                    pass  # Ignore unknown daemon types
        return recent

    def get_daemon_uptime(self, daemon_type: DaemonType) -> float:
        """Get uptime in seconds for a specific daemon.

        Args:
            daemon_type: The daemon to check

        Returns:
            Uptime in seconds, or 0.0 if not running
        """
        info = self._daemons.get(daemon_type)
        return info.uptime_seconds if info else 0.0

    # =========================================================================
    # Liveness and Readiness Probes (December 2025)
    # =========================================================================

    def liveness_probe(self) -> dict[str, Any]:
        """Liveness probe for health check endpoints.

        Returns True if the daemon manager is alive and responsive.
        This is a lightweight check suitable for frequent polling.

        Returns:
            Dict with 'alive' bool and optional 'details'
        """
        return {
            "alive": True,
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._start_time if hasattr(self, "_start_time") else 0,
        }

    def readiness_probe(
        self,
        required_daemons: list[DaemonType] | None = None,
    ) -> dict[str, Any]:
        """Readiness probe for health check endpoints.

        Returns True if the system is ready to handle requests.
        Checks that critical daemons are running.

        Args:
            required_daemons: List of daemon types that must be running.
                             If None, checks that at least one daemon is running.

        Returns:
            Dict with 'ready' bool, 'reason' if not ready, and 'details'
        """
        if not self._running:
            return {
                "ready": False,
                "reason": "DaemonManager not started",
                "timestamp": time.time(),
            }

        running_daemons = [
            dt for dt, info in self._daemons.items()
            if info.state == DaemonState.RUNNING
        ]
        failed_daemons = [
            dt for dt, info in self._daemons.items()
            if info.state == DaemonState.FAILED
        ]

        if required_daemons:
            missing = [dt for dt in required_daemons if dt not in running_daemons]
            if missing:
                return {
                    "ready": False,
                    "reason": f"Required daemons not running: {[d.value for d in missing]}",
                    "running": [d.value for d in running_daemons],
                    "failed": [d.value for d in failed_daemons],
                    "timestamp": time.time(),
                }
        elif not running_daemons:
            return {
                "ready": False,
                "reason": "No daemons running",
                "failed": [d.value for d in failed_daemons],
                "timestamp": time.time(),
            }

        return {
            "ready": True,
            "running_count": len(running_daemons),
            "failed_count": len(failed_daemons),
            "running": [d.value for d in running_daemons],
            "timestamp": time.time(),
        }

    def health_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary for monitoring dashboards.

        Returns:
            Dict with detailed health information
        """
        status = self.get_status()

        # Calculate health score (0.0 - 1.0)
        total = status["summary"]["total"]
        running = status["summary"]["running"]
        failed = status["summary"]["failed"]

        if total == 0:
            health_score = 1.0
        else:
            health_score = running / total

        # Determine overall health status
        if health_score >= 0.9:
            health_status = "healthy"
        elif health_score >= 0.5:
            health_status = "degraded"
        else:
            health_status = "unhealthy"

        return {
            "status": health_status,
            "score": health_score,
            "running": running,
            "failed": failed,
            "total": total,
            "daemons": status["daemons"],
            "liveness": self.liveness_probe(),
            "readiness": self.readiness_probe(),
            "timestamp": time.time(),
        }

    def render_metrics(self) -> str:
        """Render Prometheus-style metrics for the health server."""
        metrics_blob = ""
        try:
            from app.utils.optional_imports import (
                PROMETHEUS_AVAILABLE,
                generate_latest,
            )
            if PROMETHEUS_AVAILABLE:
                payload = generate_latest()
                if isinstance(payload, bytes):
                    metrics_blob = payload.decode("utf-8", errors="replace")
                else:
                    metrics_blob = str(payload)
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to collect Prometheus metrics: {e}")
            metrics_blob = ""

        summary = self.health_summary()

        running = summary.get("running", 0)
        failed = summary.get("failed", 0)
        total = summary.get("total", 0)
        stopped = max(0, total - running - failed)
        health_score = summary.get("score", 0.0)

        lines = [
            "# HELP daemon_count Number of daemons",
            "# TYPE daemon_count gauge",
            f'daemon_count{{state="running"}} {running}',
            f'daemon_count{{state="stopped"}} {stopped}',
            f'daemon_count{{state="failed"}} {failed}',
            "",
            "# HELP daemon_health_score Overall health score (0-1)",
            "# TYPE daemon_health_score gauge",
            f"daemon_health_score {health_score}",
            "",
            "# HELP daemon_uptime_seconds Daemon manager uptime",
            "# TYPE daemon_uptime_seconds counter",
            f'daemon_uptime_seconds {summary.get("liveness", {}).get("uptime_seconds", 0)}',
        ]

        # Selfplay throughput metrics
        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            metrics = get_selfplay_scheduler().get_metrics()
            lines.extend([
                "",
                "# HELP selfplay_games_allocated_total Total selfplay games allocated",
                "# TYPE selfplay_games_allocated_total counter",
                f"selfplay_games_allocated_total {metrics.get('games_allocated_total', 0)}",
                "# HELP selfplay_games_allocated_last_hour Selfplay games allocated in last hour",
                "# TYPE selfplay_games_allocated_last_hour gauge",
                f"selfplay_games_allocated_last_hour {metrics.get('games_allocated_last_hour', 0)}",
                "# HELP selfplay_games_per_hour Current selfplay allocation rate",
                "# TYPE selfplay_games_per_hour gauge",
                f"selfplay_games_per_hour {metrics.get('games_per_hour', 0.0)}",
            ])
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to collect selfplay scheduler metrics: {e}")

        # Cluster sync throughput metrics
        try:
            from app.coordination.auto_sync_daemon import get_auto_sync_daemon

            metrics = get_auto_sync_daemon().get_metrics()
            lines.extend([
                "",
                "# HELP cluster_sync_count_total Total sync cycles executed",
                "# TYPE cluster_sync_count_total counter",
                f"cluster_sync_count_total {metrics.get('sync_count', 0)}",
                "# HELP cluster_sync_bytes_last_cycle Bytes synced in last cycle",
                "# TYPE cluster_sync_bytes_last_cycle gauge",
                f"cluster_sync_bytes_last_cycle {metrics.get('last_sync_bytes', 0)}",
                "# HELP cluster_sync_throughput_bytes_per_sec Last cycle throughput (bytes/sec)",
                "# TYPE cluster_sync_throughput_bytes_per_sec gauge",
                f"cluster_sync_throughput_bytes_per_sec {metrics.get('last_sync_throughput_bps', 0.0)}",
                "# HELP cluster_sync_total_bytes Total bytes synced",
                "# TYPE cluster_sync_total_bytes counter",
                f"cluster_sync_total_bytes {metrics.get('total_bytes_synced', 0)}",
            ])
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to collect cluster sync metrics: {e}")

        # Event router metrics
        try:
            from app.coordination.event_router import get_router

            stats = get_router().get_stats()
            lines.extend([
                "",
                "# HELP event_router_events_routed_total Total events routed",
                "# TYPE event_router_events_routed_total counter",
                f"event_router_events_routed_total {stats.get('total_events_routed', 0)}",
                "# HELP event_router_duplicates_prevented_total Duplicate events prevented",
                "# TYPE event_router_duplicates_prevented_total counter",
                f"event_router_duplicates_prevented_total {stats.get('duplicates_prevented', 0)}",
                "# HELP event_router_content_duplicates_prevented_total Content-hash duplicates prevented",
                "# TYPE event_router_content_duplicates_prevented_total counter",
                f"event_router_content_duplicates_prevented_total {stats.get('content_duplicates_prevented', 0)}",
                "# HELP event_router_events_routed_by_type_total Events routed by type",
                "# TYPE event_router_events_routed_by_type_total counter",
            ])
            for event_type, count in stats.get("events_routed_by_type", {}).items():
                safe_event = str(event_type).replace('"', "'")
                lines.append(
                    f'event_router_events_routed_by_type_total{{event="{safe_event}"}} {count}'
                )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to collect event router metrics: {e}")

        except ImportError as e:
            logger.warning(f"MultiProviderOrchestrator dependencies not available: {e}")
        except (RuntimeError, OSError, ConnectionError) as e:
            logger.error(f"MultiProviderOrchestrator failed: {e}")

    async def _create_health_server(self) -> None:
        """Create and run HTTP health server (December 2025).

        Exposes health check endpoints for monitoring:
        - GET /health: Liveness probe
        - GET /ready: Readiness probe
        - GET /metrics: Prometheus-style metrics
        - GET /status: Detailed daemon status

        Default port: 8790 (configurable via RINGRIFT_HEALTH_PORT env var)
        """
        try:
            from aiohttp import web
        except ImportError:
            logger.warning("aiohttp not available for health server: pip install aiohttp")
            return

        port = int(os.environ.get("RINGRIFT_HEALTH_PORT", "8790"))

        async def handle_health(request):
            """Liveness probe - returns 200 if alive."""
            probe = self.liveness_probe()
            return web.json_response(probe)

        async def handle_ready(request):
            """Readiness probe - returns 200 if ready to serve."""
            probe = self.readiness_probe()
            status = 200 if probe.get("ready", False) else 503
            return web.json_response(probe, status=status)

        async def handle_metrics(request):
            """Prometheus-style metrics."""
            try:
                from app.utils.optional_imports import (
                    CONTENT_TYPE_LATEST,
                    PROMETHEUS_AVAILABLE,
                )
                content_type = CONTENT_TYPE_LATEST if PROMETHEUS_AVAILABLE else "text/plain"
            except (ImportError, AttributeError) as e:
                logger.warning(f"Failed to resolve Prometheus content type: {e}")
                content_type = "text/plain"
            return web.Response(text=self.render_metrics(), content_type=content_type)

        async def handle_status(request):
            """Detailed daemon status."""
            summary = self.health_summary()
            # Dec 2025: Fixed to use self._daemons instead of undefined _daemon_states
            # and self._factories (which contains Callables, not DaemonInfo)
            summary["daemons"] = {}
            for daemon_type, info in self._daemons.items():
                summary["daemons"][daemon_type.value] = {
                    "state": info.state.value,
                    "auto_restart": info.auto_restart,
                    "uptime_seconds": info.uptime_seconds,
                    "restart_count": info.restart_count,
                }
            return web.json_response(summary)

        try:
            app = web.Application()
            app.router.add_get('/health', handle_health)
            app.router.add_get('/ready', handle_ready)
            app.router.add_get('/metrics', handle_metrics)
            app.router.add_get('/status', handle_status)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', port)
            await site.start()

            logger.info(f"Health server listening on http://0.0.0.0:{port}")

            # Keep running
            while True:
                await asyncio.sleep(3600)

        except OSError as e:
            if "address already in use" in str(e).lower():
                logger.warning(f"Health server port {port} already in use, skipping")
            else:
                logger.error(f"Health server failed: {e}")
        except (RuntimeError, ConnectionError) as e:
            logger.error(f"Health server failed: {e}")


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
        DaemonType.S3_BACKUP,  # Dec 2025: Backup models to S3 after promotion
        DaemonType.REPLICATION_MONITOR,
        DaemonType.REPLICATION_REPAIR,  # Actively repair under-replicated data
        DaemonType.CLUSTER_MONITOR,
        DaemonType.QUEUE_MONITOR,  # Monitor queue depths and apply backpressure
        DaemonType.FEEDBACK_LOOP,
        DaemonType.QUALITY_MONITOR,  # Monitor selfplay data quality
        DaemonType.MODEL_PERFORMANCE_WATCHDOG,  # Monitor model win rates
        DaemonType.NPZ_DISTRIBUTION,  # Distribute training data after export
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
        DaemonType.COORDINATOR_DISK_MANAGER,  # Dec 27, 2025: Manage coordinator disk space
    ],

    # Training node profile - runs on GPU nodes
    "training_node": [
        DaemonType.EVENT_ROUTER,
        DaemonType.HEALTH_SERVER,  # HTTP health endpoints (/health, /ready, /metrics)
        DaemonType.DATA_PIPELINE,
        DaemonType.CONTINUOUS_TRAINING_LOOP,
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
    def handle_signal(signum, frame):
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
