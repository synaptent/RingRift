"""DaemonManager lifecycle, health, and restart-policy helpers.

April 2026: Extracted from daemon_manager.py as part of the Part 3
coordination decomposition. This module is intentionally separate from
``daemon_lifecycle.py``: that file contains the existing composition-based
``DaemonLifecycleManager``, while this mixin keeps DaemonManager's public
lifecycle and health-check surface small and testable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config.coordination_defaults import (
    DaemonHealthDefaults,
    DegradedModeDefaults,
)
from app.core.async_context import fire_and_forget, safe_create_task
from app.coordination.daemon_types import (
    CRITICAL_DAEMONS,
    DaemonInfo,
    DaemonState,
    DaemonType,
    RestartTier,
    get_daemon_category,
)

if TYPE_CHECKING:
    from app.coordination.daemon_health_types import AnalysisResult

logger = logging.getLogger(__name__)

# Restart count persistence shared by DaemonManager lifecycle helpers.
try:
    from app.utils.paths import COORDINATION_DIR
    _restart_state_dir = COORDINATION_DIR
except ImportError:
    _restart_state_dir = Path(__file__).parent.parent.parent / "data" / "coordination"

_restart_state_dir.mkdir(parents=True, exist_ok=True)
RESTART_STATE_FILE = _restart_state_dir / "daemon_restarts.json"
RESTART_COUNTS_EXPIRY_SECONDS = 86400
MAX_RESTARTS_PER_HOUR = 10
PERMANENT_FAILURE_RECOVERY_SECONDS = 86400
CASCADE_RESTART_WINDOW_SECONDS = 300
CASCADE_RESTART_THRESHOLD = 25
CASCADE_COOLDOWN_SECONDS = 120
CASCADE_STARTUP_GRACE_PERIOD = 300
CASCADE_STARTUP_THRESHOLD = 100

# Lazy import for daemon lifecycle events to avoid circular imports.
def _get_daemon_event_emitters():
    """Return optional daemon lifecycle event emitters."""
    try:
        from app.distributed.data_events import (
            emit_daemon_started,
            emit_daemon_stopped,
        )
        return emit_daemon_started, emit_daemon_stopped
    except ImportError:
        logger.debug("data_events not available for daemon lifecycle events")
        return None, None


class DaemonManagerLifecycleMixin:
    """DaemonManager lifecycle, health, and probe helpers.

    April 2026: Extracted from daemon_manager.py (Part 3 Phase 3). The
    manager keeps registration, profile, event wiring, and restart-policy state;
    this mixin keeps the operational lifecycle wrappers and health surfaces.
    """

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

            # January 2026 Sprint 10: Ensure recovery probing is running
            # Previously only started via start_all() callback, not when using
            # individual start() calls from master_loop.py.
            await self._ensure_recovery_probing_running()

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

    async def _ensure_recovery_probing_running(self) -> None:
        """Ensure circuit breaker recovery probing is running.

        January 2026 Sprint 10: Extracted from start_all() callback to allow
        individual start() calls to also start recovery probing. This fixes
        an issue where master_loop.py calling start() individually would never
        start recovery probing, causing circuits to take longer to recover.

        Safe to call multiple times - will only start probing once.
        """
        # Use a class attribute to track if we've started probing
        if getattr(self, "_recovery_probing_started", False):
            return

        try:
            from app.distributed.circuit_breaker import start_recovery_probing

            task = start_recovery_probing(interval=30.0)
            if task:
                self._recovery_probing_started = True
                logger.info("[DaemonManager] Started circuit breaker recovery probing")
        except (ImportError, RuntimeError) as e:
            logger.debug(f"[DaemonManager] Circuit breaker probing not started: {e}")

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

    async def _emit_daemon_failure_event(
        self, daemon_type: DaemonType, analysis: "AnalysisResult"
    ) -> None:
        """Emit DAEMON_FAILURE_CLASSIFIED event for failure pattern tracking.

        Jan 5, 2026 (Sprint 17.9): Emits events when DaemonHealthAnalyzer
        detects significant failure patterns (escalation, recovery, critical).

        Args:
            daemon_type: Type of daemon that failed
            analysis: Analysis result from DaemonHealthAnalyzer
        """
        try:
            from app.coordination.event_emission_helpers import safe_emit_event
            from app.distributed.data_events import DataEventType

            import socket
            event_type = DataEventType.DAEMON_FAILURE_CLASSIFIED

            safe_emit_event(
                event_type,
                {
                    "daemon_name": daemon_type.value,
                    "category": analysis.category.value,
                    "recommended_action": analysis.recommended_action,
                    "consecutive_failures": analysis.details.get("consecutive_failures", 0),
                    "failure_rate": analysis.details.get("failure_rate", 0.0),
                    "needs_intervention": analysis.needs_intervention,
                    "hostname": socket.gethostname(),
                    "source": "DaemonManager",
                },
                source="daemon_manager",
            )
            logger.debug(
                f"Emitted DAEMON_FAILURE_CLASSIFIED for {daemon_type.value}: "
                f"{analysis.category.value}"
            )
        except ImportError as e:
            # Non-critical - log and continue
            logger.debug(f"Failed to emit DAEMON_FAILURE_CLASSIFIED: {e}")
        except (RuntimeError, OSError, AttributeError) as e:
            # Non-critical - log and continue
            logger.debug(f"Error emitting DAEMON_FAILURE_CLASSIFIED: {e}")

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
                await start_watchdog()
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
        """Background health check loop.

        Sprint 5 (Jan 2, 2026): Added deadlock detection check.
        Sprint 17.9 (Jan 4, 2026): Added circuit breaker TTL decay.
        Session 17.25 (Jan 5, 2026): Reduced CB decay from hourly to 15-min.
        """
        health_check_count = 0
        # Decay old circuits every ~15 health checks (~15 min at 60s intervals)
        # Session 17.25: Reduced from 60 (1h) to 15 (15min) for faster recovery
        CB_DECAY_INTERVAL = 15

        while self._running and not self._shutdown_event.is_set():
            try:
                # Sprint 5: Check for potential deadlocks before health checks
                await self._check_for_deadlocks()
                await self._check_health()

                # Sprint 17.9: Hourly circuit breaker TTL decay
                health_check_count += 1
                if health_check_count >= CB_DECAY_INTERVAL:
                    health_check_count = 0
                    await self._decay_old_circuit_breakers()

                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError) as e:
                logger.error(f"Health check error: {e}")

    async def _decay_old_circuit_breakers(self) -> None:
        """Decay old circuit breakers to prevent permanent node exclusion.

        Sprint 17.9 (Jan 4, 2026): Wraps decay_all_circuit_breakers() for hourly
        execution from the health loop. Uses asyncio.to_thread() to avoid
        blocking the event loop during SQLite operations.

        Circuit breakers opened >6 hours ago are reset to CLOSED state,
        allowing previously-failed nodes to be retried.
        """
        try:
            from app.coordination.circuit_breaker_base import decay_all_circuit_breakers

            # Run in thread pool to avoid blocking
            results = await asyncio.to_thread(decay_all_circuit_breakers)

            # Log summary if any circuits were reset
            total_decayed = results.get("total_decayed", 0)
            if total_decayed > 0:
                logger.info(
                    f"[DaemonManager] Circuit breaker TTL decay: {total_decayed} circuits reset"
                )
        except ImportError:
            logger.debug("[DaemonManager] circuit_breaker_base not available for decay")
        except (RuntimeError, OSError, AttributeError) as e:
            logger.debug(f"[DaemonManager] Circuit breaker decay failed: {e}")

    async def _check_single_daemon_health(
        self,
        daemon_type: DaemonType,
        info: DaemonInfo,
        health_check_timeout: float,
    ) -> tuple[DaemonType, dict[str, Any] | None]:
        """Check health of a single daemon.

        Returns (daemon_type, health_result) where health_result is None if no check was performed.
        Dec 29, 2025: Extracted for parallel health checks.
        """
        if info.instance is None or not hasattr(info.instance, 'health_check'):
            return (daemon_type, None)

        try:
            health_method = info.instance.health_check

            # Check if health_check is a coroutine function (async def)
            if asyncio.iscoroutinefunction(health_method):
                try:
                    health_result = await asyncio.wait_for(
                        health_method(), timeout=health_check_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"{daemon_type.value} async health_check() timed out ({health_check_timeout}s)"
                    )
                    return (daemon_type, {"healthy": False, "message": f"timeout ({health_check_timeout}s)"})
            else:
                try:
                    health_result = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(None, health_method),
                        timeout=health_check_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"{daemon_type.value} sync health_check() timed out ({health_check_timeout}s)"
                    )
                    return (daemon_type, {"healthy": False, "message": f"timeout ({health_check_timeout}s)"})

            # Handle coroutine or awaitable results
            if asyncio.iscoroutine(health_result) or callable(getattr(health_result, '__await__', None)):
                try:
                    health_result = await asyncio.wait_for(health_result, timeout=health_check_timeout)
                except asyncio.TimeoutError:
                    return (daemon_type, {"healthy": False, "message": f"timeout ({health_check_timeout}s)"})

            # Normalize result to dict
            if hasattr(health_result, 'healthy'):
                return (daemon_type, {"healthy": health_result.healthy, "message": getattr(health_result, 'message', '')})
            elif isinstance(health_result, dict):
                return (daemon_type, health_result)
            elif isinstance(health_result, bool):
                return (daemon_type, {"healthy": health_result, "message": ""})
            return (daemon_type, {"healthy": True, "message": ""})

        except (RuntimeError, OSError, AttributeError) as e:
            logger.debug(f"Error calling health_check for {daemon_type.value}: {e}")
            return (daemon_type, None)

    async def _check_health(self) -> None:
        """Check health of all daemons and attempt recovery of FAILED ones.

        Dec 29, 2025: Parallelized health checks for O(t) instead of O(n*t) time.
        - Phase 1: Under lock, identify failed daemons and collect those needing health check
        - Phase 2: Outside lock, run health checks in parallel
        - Phase 3: Under lock, process health check results

        Note: Restarts are done outside lock to avoid deadlock (start() also acquires lock).
        """
        daemons_to_restart: list[DaemonType] = []
        daemons_to_check: list[tuple[DaemonType, DaemonInfo]] = []

        # Jan 3, 2026 (Sprint 15.1): Use adaptive timeout based on system load
        try:
            from app.config.coordination_defaults import get_adaptive_health_timeout
            health_check_timeout = max(
                await asyncio.to_thread(get_adaptive_health_timeout),
                DaemonHealthDefaults.HEALTH_CHECK_TIMEOUT,
            )
        except ImportError:
            health_check_timeout = DaemonHealthDefaults.HEALTH_CHECK_TIMEOUT

        # Phase 1: Collect daemons needing checks (under lock with timeout)
        # December 30, 2025: Added timeout to prevent indefinite blocking
        async with self._with_lock_timeout("health_check_phase1", timeout=30.0) as acquired:
            if not acquired:
                # Lock timeout - skip this health check cycle
                logger.warning("Skipping health check cycle: lock acquisition timeout (30s)")
                return

            current_time = time.time()

            for daemon_type, info in list(self._daemons.items()):
                # Attempt recovery of FAILED daemons after cooldown period
                if info.state == DaemonState.FAILED:
                    if info.import_error or not info.auto_restart or not self.config.auto_restart_failed:
                        continue
                    time_since_failure = current_time - info.last_failure_time
                    if time_since_failure >= self.config.recovery_cooldown:
                        logger.info(
                            f"Attempting recovery of {daemon_type.value} after "
                            f"{time_since_failure:.0f}s cooldown"
                        )
                        info.restart_count = 0
                        info.state = DaemonState.STOPPED
                        daemons_to_restart.append(daemon_type)
                    continue

                # December 30, 2025: Auto-restart STOPPED daemons with auto_restart=True
                # when their dependencies are now satisfied. This handles daemons that
                # failed to start initially due to missing dependencies.
                if (
                    info.state == DaemonState.STOPPED
                    and info.auto_restart
                    and self.config.auto_restart_failed
                ):
                    # Check if all dependencies are now running
                    deps = list(info.depends_on or [])
                    all_deps_running = all(
                        self._daemons.get(dep) is not None
                        and self._daemons[dep].state == DaemonState.RUNNING
                        for dep in deps
                    )
                    if all_deps_running:
                        logger.info(
                            f"Auto-restarting {daemon_type.value}: dependencies now satisfied"
                        )
                        daemons_to_restart.append(daemon_type)
                    continue

                if info.state != DaemonState.RUNNING:
                    continue

                # Skip during startup grace period
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

                    if (
                        self.config.auto_restart_failed
                        and info.auto_restart
                        and info.restart_count < info.max_restarts
                    ):
                        logger.warning(f"{daemon_type.value} died, restarting...")
                        daemons_to_restart.append(daemon_type)
                    else:
                        info.state = DaemonState.FAILED
                        info.last_failure_time = current_time
                    continue

                # Queue for health check if has health_check method
                if info.instance is not None and hasattr(info.instance, 'health_check'):
                    daemons_to_check.append((daemon_type, info))

        # Phase 2: Run health checks in parallel (outside lock)
        # Dec 29, 2025: Parallel health checks reduce time from O(n*t) to O(t)
        health_results: dict[DaemonType, dict[str, Any] | None] = {}
        if daemons_to_check and DaemonHealthDefaults.PARALLEL_HEALTH_CHECKS:
            max_concurrent = DaemonHealthDefaults.MAX_PARALLEL_HEALTH_CHECKS
            if max_concurrent <= 0:
                max_concurrent = len(daemons_to_check)

            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def check_with_semaphore(dt: DaemonType, di: DaemonInfo) -> tuple[DaemonType, dict[str, Any] | None]:
                async with semaphore:
                    return await self._check_single_daemon_health(dt, di, health_check_timeout)

            # Run all health checks concurrently with overall timeout protection
            # Dec 29, 2025: Added wait_for to prevent indefinite hang if all checks stall
            overall_timeout = health_check_timeout * 2 + 5.0  # Safety margin
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(
                        *[check_with_semaphore(dt, di) for dt, di in daemons_to_check],
                        return_exceptions=True
                    ),
                    timeout=overall_timeout
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"[DaemonManager] Health check batch timed out after {overall_timeout:.1f}s "
                    f"({len(daemons_to_check)} daemons). Marking all as unhealthy."
                )
                # Mark all as failed due to timeout
                results = [
                    (dt, {"healthy": False, "message": f"batch timeout ({overall_timeout:.1f}s)"})
                    for dt, _ in daemons_to_check
                ]

            for result in results:
                if isinstance(result, Exception):
                    logger.debug(f"Health check exception: {result}")
                else:
                    daemon_type, health_result = result
                    health_results[daemon_type] = health_result
        elif daemons_to_check:
            # Fallback to sequential (if parallel disabled)
            for daemon_type, info in daemons_to_check:
                _, health_result = await self._check_single_daemon_health(daemon_type, info, health_check_timeout)
                health_results[daemon_type] = health_result

        # Phase 3: Process health check results (under lock with timeout)
        # December 30, 2025: Added timeout to prevent indefinite blocking
        # Jan 5, 2026 (Sprint 17.9): Integrated DaemonHealthAnalyzer for failure classification
        if health_results:
            # Lazy import to avoid circular dependencies
            try:
                from app.coordination.daemon_health_analyzer import get_daemon_health_analyzer
                from app.coordination.daemon_health_types import FailureCategory
                analyzer = get_daemon_health_analyzer()
            except ImportError:
                analyzer = None

            async with self._with_lock_timeout("health_check_phase3", timeout=30.0) as acquired:
                if not acquired:
                    # Lock timeout - skip processing, will retry next cycle
                    logger.warning("Skipping health result processing: lock acquisition timeout")
                else:
                    current_time = time.time()
                    for daemon_type, health_result in health_results.items():
                        if health_result is None:
                            continue
                        info = self._daemons.get(daemon_type)
                        if info is None:
                            continue

                        is_healthy = health_result.get('healthy', True)

                        # Classify the health result using DaemonHealthAnalyzer
                        if analyzer is not None:
                            analysis = analyzer.analyze(daemon_type.value, health_result)
                            health_result["failure_category"] = analysis.category.value
                            health_result["recommended_action"] = analysis.recommended_action

                            # Emit event if analyzer recommends it
                            if analysis.should_emit_event:
                                await self._emit_daemon_failure_event(daemon_type, analysis)

                        if not is_healthy:
                            message = health_result.get('message', 'unhealthy')
                            category = health_result.get('failure_category', 'unknown')
                            logger.warning(f"{daemon_type.value} health check failed ({category}): {message}")
                            info.last_error = f"Health check failed: {message}"
                            if (
                                self.config.auto_restart_failed
                                and info.auto_restart
                                and info.restart_count < info.max_restarts
                            ):
                                daemons_to_restart.append(daemon_type)
                            else:
                                info.state = DaemonState.FAILED
                                info.last_failure_time = current_time

        # Handle restarts outside lock to prevent deadlock (start() also acquires lock)
        # Dec 30, 2025: Use hierarchical circuit breaker with per-daemon checks
        # Critical daemons and exempt categories can restart even when others are blocked
        if not daemons_to_restart:
            return  # Nothing to restart

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
        blocked_daemons: list[str] = []

        for daemon_type in sorted_restarts:
            # Dec 30, 2025: Per-daemon circuit breaker check
            # Critical daemons bypass all breakers, others check their category
            allowed, reason = self._cascade_breaker.can_restart(daemon_type)
            if not allowed:
                blocked_daemons.append(f"{daemon_type.value}({reason})")
                continue

            # Record restart in hierarchical breaker
            self._record_global_restart(daemon_type)

            # First stop the daemon if it's still running (unhealthy but not crashed)
            info = self._daemons.get(daemon_type)
            if info and info.state == DaemonState.RUNNING:
                logger.info(
                    f"Stopping unhealthy daemon {daemon_type.value} before restart"
                )
                await self.stop(daemon_type)
            # Now start (or restart) the daemon
            await self.start(daemon_type)

        # Log blocked daemons summary
        if blocked_daemons:
            logger.warning(
                f"[DaemonManager] {len(blocked_daemons)} daemon(s) blocked by circuit breaker: "
                f"{blocked_daemons[:5]}{'...' if len(blocked_daemons) > 5 else ''}"
            )

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

        # Dec 30, 2025: Include config freshness info
        config_info = self._get_config_freshness_info()

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
                "config_version": config_info.get("hash", "unknown"),
                "config_age_seconds": config_info.get("age_seconds", 0),
                "config_status": config_info.get("status", "unknown"),
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

    def _get_config_freshness_info(self) -> dict[str, Any]:
        """Get cluster config freshness info for health reporting.

        December 30, 2025: Added as part of distributed config sync infrastructure.
        Reports config version hash, age, and freshness status.

        Returns:
            Dict with config freshness info:
                - hash: Content hash of config (first 16 chars of SHA256)
                - age_seconds: Seconds since config was last loaded
                - status: "fresh" (< 1 hour), "stale" (1-24 hours), "outdated" (> 24 hours)
        """
        try:
            from app.config.cluster_config import get_config_version

            version = get_config_version()
            age_seconds = time.time() - version.timestamp

            # Determine status based on age
            if age_seconds < 3600:  # < 1 hour
                status = "fresh"
            elif age_seconds < 86400:  # < 24 hours
                status = "stale"
            else:
                status = "outdated"

            return {
                "hash": version.content_hash,
                "age_seconds": round(age_seconds, 1),
                "status": status,
            }
        except ImportError:
            # Config cache not available
            return {"hash": "unavailable", "age_seconds": 0, "status": "unknown"}
        except (OSError, AttributeError) as e:
            logger.debug(f"Config freshness info unavailable: {e}")
            return {"hash": "error", "age_seconds": 0, "status": "error"}

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
            from app.coordination.event_router import DataEventType, publish_sync

            publish_sync(
                DataEventType.RESOURCE_CONSTRAINT,
                {
                    "constraint_type": "memory",
                    "memory_percent": memory_info.get("percent", 0),
                    "available_gb": memory_info.get("available_gb", 0),
                    "source": "daemon_manager",
                },
                source="DaemonManager",
            )
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

        if metrics_blob:
            lines.extend(["", metrics_blob.rstrip()])

        return "\n".join(lines) + "\n"

    async def _create_health_server(self) -> None:
        """Create and run HTTP health server (December 2025).

        Exposes health check endpoints for monitoring:
        - GET /health: Liveness probe
        - GET /ready: Readiness probe
        - GET /metrics: Prometheus-style metrics
        - GET /status: Detailed daemon status

        Default port: 8790 (configurable via RINGRIFT_HEALTH_PORT env var)

        January 2026: Runs in a separate thread with its own event loop to isolate
        from main loop blocking. This ensures health endpoints always respond even
        when the main event loop is blocked by synchronous operations.
        """
        import threading

        try:
            from aiohttp import web
        except ImportError:
            logger.warning("aiohttp not available for health server: pip install aiohttp")
            return

        port = int(os.environ.get("RINGRIFT_HEALTH_PORT", "8790"))

        # Reference to self for closures in thread
        manager = self

        def _run_health_server_in_thread() -> None:
            """Run the health server in a separate thread with its own event loop.

            This isolates the health server from main event loop blocking.
            """
            import asyncio as thread_asyncio

            async def handle_health(request: web.Request) -> web.Response:
                """Liveness probe - returns 200 if alive.

                January 2026: Lightweight implementation that doesn't call any blocking methods.
                """
                del request
                return web.json_response({
                    "alive": manager._running,
                    "timestamp": time.time(),
                    "node_id": os.environ.get("RINGRIFT_NODE_ID", "unknown"),
                })

            async def handle_ready(request: web.Request) -> web.Response:
                """Readiness probe - returns 200 if ready to serve."""
                del request
                # Simple readiness check - don't block
                critical_daemons_running = sum(
                    1 for d in list(manager._daemons.values())
                    if d.state == DaemonState.RUNNING
                )
                ready = manager._running and critical_daemons_running > 0
                return web.json_response({
                    "ready": ready,
                    "running_daemons": critical_daemons_running,
                    "timestamp": time.time(),
                }, status=200 if ready else 503)

            async def handle_metrics(request: web.Request) -> web.Response:
                """Prometheus-style metrics (lightweight version)."""
                del request
                # Return basic metrics without blocking
                metrics = f"""# HELP daemon_manager_running DaemonManager running status
# TYPE daemon_manager_running gauge
daemon_manager_running {1 if manager._running else 0}
# HELP daemon_manager_daemons_total Total registered daemons
# TYPE daemon_manager_daemons_total gauge
daemon_manager_daemons_total {len(manager._daemons)}
# HELP daemon_manager_running_daemons Number of running daemons
# TYPE daemon_manager_running_daemons gauge
daemon_manager_running_daemons {sum(1 for d in manager._daemons.values() if d.state == DaemonState.RUNNING)}
"""
                return web.Response(text=metrics, content_type="text/plain")

            async def handle_status(request: web.Request) -> web.Response:
                """Detailed daemon status (lightweight version)."""
                del request
                # Quick snapshot without calling expensive methods
                daemons = {}
                for dtype, info in list(manager._daemons.items()):
                    daemons[dtype.value] = {
                        "state": info.state.value,
                        "auto_restart": info.auto_restart,
                        "restart_count": info.restart_count,
                    }
                return web.json_response({
                    "running": manager._running,
                    "daemon_count": len(daemons),
                    "daemons": daemons,
                    "timestamp": time.time(),
                })

            async def handle_event(request: web.Request) -> web.Response:
                """Receive events from P2P nodes."""
                try:
                    data = await request.json()
                    event_type = data.get("event_type")
                    payload = data.get("payload", {})
                    source = data.get("source", "p2p_remote")

                    if not event_type:
                        return web.json_response({"error": "missing event_type"}, status=400)

                    try:
                        from app.coordination.event_router import publish_sync
                        # Use thread-safe event emission
                        thread_asyncio.get_event_loop().run_in_executor(
                            None, lambda: publish_sync(event_type, payload, source=source)
                        )
                        return web.json_response({"status": "ok", "event_type": event_type})
                    except Exception as e:
                        logger.error(f"Failed to publish cross-node event {event_type}: {e}")
                        return web.json_response({"error": f"publish failed: {e}"}, status=500)
                except Exception as e:
                    return web.json_response({"error": f"invalid request: {e}"}, status=400)

            async def run_server() -> None:
                """Set up and run the aiohttp server."""
                try:
                    app = web.Application()
                    app.router.add_get('/health', handle_health)
                    app.router.add_get('/ready', handle_ready)
                    app.router.add_get('/metrics', handle_metrics)
                    app.router.add_get('/status', handle_status)
                    app.router.add_post('/event', handle_event)

                    runner = web.AppRunner(app)
                    await runner.setup()

                    # Try dual-stack IPv6 first, fall back to IPv4
                    try:
                        site = web.TCPSite(runner, '::', port)
                        await site.start()
                        logger.info(f"Health server listening on http://[::]:{port} (dual-stack, isolated thread)")
                    except OSError:
                        site = web.TCPSite(runner, '0.0.0.0', port)
                        await site.start()
                        logger.info(f"Health server listening on http://0.0.0.0:{port} (IPv4-only, isolated thread)")

                    # Keep running
                    while manager._running:
                        await thread_asyncio.sleep(1)

                    await runner.cleanup()

                except OSError as e:
                    if "address already in use" in str(e).lower():
                        logger.warning(f"Health server port {port} already in use, skipping")
                    else:
                        logger.error(f"Health server failed: {e}")

            # Create new event loop for this thread and run the server
            loop = thread_asyncio.new_event_loop()
            thread_asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_server())
            finally:
                loop.close()

        # Start health server in separate thread
        health_thread = threading.Thread(
            target=_run_health_server_in_thread,
            name="health-server",
            daemon=True,
        )
        health_thread.start()
        logger.info("Health server thread started")

        # Keep this coroutine alive while the thread runs
        while self._running and health_thread.is_alive():
            await asyncio.sleep(10)

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

    def _get_backoff_delay(
        self, daemon_type: DaemonType, tier: RestartTier, restart_count: int
    ) -> float:
        """Compute restart backoff delay based on tier and restart count.

        December 2025: Part of 48-hour autonomous operation plan.

        Args:
            daemon_type: Type of daemon
            tier: Current restart tier
            restart_count: Number of restarts in current tier

        Returns:
            Backoff delay in seconds
        """
        is_critical = daemon_type in CRITICAL_DAEMONS

        if tier == RestartTier.NORMAL:
            # Exponential backoff: 5, 10, 20, 40, 80
            base = DegradedModeDefaults.NORMAL_BACKOFF_BASE
            max_backoff = DegradedModeDefaults.NORMAL_BACKOFF_MAX
            delay = min(base * (2 ** min(restart_count - 1, 4)), max_backoff)
            return delay

        elif tier == RestartTier.ELEVATED:
            # Extended backoff: 160, 320
            base = DegradedModeDefaults.ELEVATED_BACKOFF_BASE
            max_backoff = DegradedModeDefaults.ELEVATED_BACKOFF_MAX
            # restart_count here is 6-10, so normalize to 0-4 for tier
            tier_count = restart_count - DegradedModeDefaults.NORMAL_MAX_RESTARTS
            delay = min(base * (2 ** min(tier_count - 1, 1)), max_backoff)
            return delay

        else:  # DEGRADED
            # Long intervals based on criticality
            if is_critical:
                return DegradedModeDefaults.CRITICAL_RETRY_INTERVAL
            else:
                return DegradedModeDefaults.NONCRITICAL_RETRY_INTERVAL

    def _get_restart_tier(self, hourly_restarts: int) -> RestartTier:
        """Get the restart tier based on hourly restart count.

        December 2025: Part of 48-hour autonomous operation plan.

        Args:
            hourly_restarts: Number of restarts in the last hour

        Returns:
            RestartTier based on restart count
        """
        if hourly_restarts <= DegradedModeDefaults.NORMAL_MAX_RESTARTS:
            return RestartTier.NORMAL
        elif hourly_restarts <= DegradedModeDefaults.ELEVATED_MAX_RESTARTS:
            return RestartTier.ELEVATED
        else:
            return RestartTier.DEGRADED

    def _is_degraded_ready_to_retry(self, daemon_name: str) -> bool:
        """Check if a degraded daemon is ready to retry.

        December 2025: Part of 48-hour autonomous operation plan.

        Args:
            daemon_name: Name of the daemon

        Returns:
            True if the daemon should retry, False if still waiting
        """
        if daemon_name not in self._degraded_daemons:
            return True

        next_retry, tier, entered_at = self._degraded_daemons[daemon_name]
        current_time = time.time()

        # Check if auto-recovery period has passed (reset to normal)
        if current_time - entered_at >= DegradedModeDefaults.RESET_AFTER_HOURS * 3600:
            logger.info(
                f"[DaemonManager] {daemon_name} auto-recovered from degraded mode after "
                f"{DegradedModeDefaults.RESET_AFTER_HOURS}h"
            )
            del self._degraded_daemons[daemon_name]
            self._restart_timestamps.pop(daemon_name, None)
            self._save_restart_counts()
            return True

        return current_time >= next_retry

    def get_degraded_daemons(self) -> dict[str, dict]:
        """Get information about daemons in degraded mode.

        December 2025: Part of 48-hour autonomous operation plan.

        Returns:
            Dict of daemon_name -> {tier, next_retry, entered_at, is_critical}
        """
        result = {}
        current_time = time.time()

        for daemon_name, (next_retry, tier, entered_at) in self._degraded_daemons.items():
            # Find daemon type
            daemon_type = None
            for dt in DaemonType:
                if dt.value == daemon_name:
                    daemon_type = dt
                    break

            result[daemon_name] = {
                "tier": tier.value,
                "next_retry_in_seconds": max(0, next_retry - current_time),
                "entered_degraded_at": entered_at,
                "time_in_degraded_seconds": current_time - entered_at,
                "is_critical": daemon_type in CRITICAL_DAEMONS if daemon_type else False,
                "auto_recovery_in_seconds": max(
                    0,
                    (entered_at + DegradedModeDefaults.RESET_AFTER_HOURS * 3600) - current_time
                ),
            }

        return result

    def record_restart(self, daemon_type: DaemonType) -> bool:
        """Record a daemon restart and determine restart action.

        December 2025: Updated for graceful degradation (48-hour autonomous operation).
        Instead of marking daemons as "permanently failed" after 10 restarts/hour,
        we now use tiered restart policies:
        - NORMAL (1-5 restarts): Standard exponential backoff (5s → 80s)
        - ELEVATED (6-10 restarts): Extended backoff (160s → 320s)
        - DEGRADED (>10 restarts): Keep retrying with longer intervals
          - Critical daemons: 30 min retry interval
          - Non-critical daemons: 4 hour retry interval

        Args:
            daemon_type: Type of daemon being restarted

        Returns:
            True if the daemon should be allowed to restart now,
            False if it should wait (in degraded mode) or is blocked
        """
        daemon_name = daemon_type.value
        current_time = time.time()

        # Check if degraded mode is disabled (legacy behavior)
        if not DegradedModeDefaults.ENABLED:
            return self._record_restart_legacy(daemon_type)

        # Check if in degraded mode and not ready to retry
        if daemon_name in self._degraded_daemons:
            if not self._is_degraded_ready_to_retry(daemon_name):
                next_retry, tier, _ = self._degraded_daemons[daemon_name]
                wait_seconds = next_retry - current_time
                logger.info(
                    f"[DaemonManager] {daemon_name} is in {tier.value} mode, "
                    f"next retry in {wait_seconds:.0f}s"
                )
                return False
            else:
                # Ready to retry - will update next_retry below
                logger.info(f"[DaemonManager] {daemon_name} degraded mode retry starting")

        # Legacy: Check if permanently failed (for backward compatibility during migration)
        if daemon_name in self._permanently_failed:
            failed_at = self._permanently_failed[daemon_name]
            age_seconds = current_time - failed_at

            if age_seconds >= PERMANENT_FAILURE_RECOVERY_SECONDS:
                logger.info(
                    f"[DaemonManager] {daemon_name} auto-recovered after "
                    f"{age_seconds / 3600:.1f}h in permanent failure state"
                )
                del self._permanently_failed[daemon_name]
                self._restart_timestamps.pop(daemon_name, None)
                self._save_restart_counts()
            else:
                # Migrate to degraded mode if enabled
                self._enter_degraded_mode(daemon_type, current_time)
                del self._permanently_failed[daemon_name]
                self._save_restart_counts()
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

        hourly_restarts = len(self._restart_timestamps[daemon_name])
        tier = self._get_restart_tier(hourly_restarts)

        # Check for crash loop early warning (3+ restarts in 5 minutes)
        CRASH_LOOP_WINDOW_SECONDS = 300
        CRASH_LOOP_THRESHOLD = 3
        recent_timestamps = [
            ts for ts in self._restart_timestamps[daemon_name]
            if ts > current_time - CRASH_LOOP_WINDOW_SECONDS
        ]
        recent_restarts = len(recent_timestamps)

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

        # Handle based on tier
        if tier == RestartTier.DEGRADED:
            self._enter_degraded_mode(daemon_type, current_time)
            logger.warning(
                f"[DaemonManager] {daemon_name} entered DEGRADED mode "
                f"({hourly_restarts} restarts/hour). Will keep retrying with "
                f"{'30min' if daemon_type in CRITICAL_DAEMONS else '4hr'} intervals."
            )
            self._emit_degraded_mode_event(daemon_type, hourly_restarts)
            self._save_restart_counts()
            return False  # Don't restart immediately, wait for next retry
        else:
            # Normal or elevated tier - compute backoff and allow restart
            backoff = self._get_backoff_delay(daemon_type, tier, hourly_restarts)
            logger.info(
                f"[DaemonManager] {daemon_name} restart #{hourly_restarts} "
                f"(tier={tier.value}, backoff={backoff:.1f}s)"
            )

            # Update daemon state if it was in degraded mode
            if daemon_name in self._degraded_daemons:
                del self._degraded_daemons[daemon_name]
                if daemon_type in self._daemons:
                    self._daemons[daemon_type].state = DaemonState.RESTARTING

            self._save_restart_counts()
            return True

    def _enter_degraded_mode(self, daemon_type: DaemonType, current_time: float) -> None:
        """Enter degraded mode for a daemon.

        December 2025: Part of 48-hour autonomous operation plan.

        Args:
            daemon_type: Type of daemon entering degraded mode
            current_time: Current timestamp
        """
        daemon_name = daemon_type.value
        is_critical = daemon_type in CRITICAL_DAEMONS
        hourly_restarts = len(self._restart_timestamps.get(daemon_name, []))

        # Compute next retry time
        if is_critical:
            retry_delay = DegradedModeDefaults.CRITICAL_RETRY_INTERVAL
        else:
            retry_delay = DegradedModeDefaults.NONCRITICAL_RETRY_INTERVAL

        next_retry = current_time + retry_delay

        # Track when we entered degraded mode (for first entry) or keep original
        if daemon_name in self._degraded_daemons:
            _, _, entered_at = self._degraded_daemons[daemon_name]
        else:
            entered_at = current_time

        self._degraded_daemons[daemon_name] = (next_retry, RestartTier.DEGRADED, entered_at)

        # Update daemon state
        if daemon_type in self._daemons:
            self._daemons[daemon_type].state = DaemonState.DEGRADED

        logger.info(
            f"[DaemonManager] {daemon_name} entered degraded mode "
            f"(critical={is_critical}, next_retry_in={retry_delay/60:.0f}min, "
            f"restarts={hourly_restarts})"
        )

    def _emit_degraded_mode_event(self, daemon_type: DaemonType, restart_count: int) -> None:
        """Emit DAEMON_DEGRADED_MODE event for monitoring/alerting.

        December 2025: Part of 48-hour autonomous operation plan.
        Notifies external systems when a daemon enters degraded mode.
        """
        try:
            from app.distributed.data_events import DataEventType

            import socket
            hostname = socket.gethostname()

            is_critical = daemon_type in CRITICAL_DAEMONS
            retry_interval = (
                DegradedModeDefaults.CRITICAL_RETRY_INTERVAL
                if is_critical
                else DegradedModeDefaults.NONCRITICAL_RETRY_INTERVAL
            )

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                logger.info(
                    f"DAEMON_DEGRADED_MODE: {daemon_type.value} "
                    f"(restart_count={restart_count}, retry_in={retry_interval/60:.0f}min, "
                    f"no event loop)"
                )
                return

            # Emit via event router if available
            try:
                from app.coordination.event_router import publish_sync

                publish_sync(
                    "daemon.degraded_mode",
                    {
                        "daemon_name": daemon_type.value,
                        "hostname": hostname,
                        "restart_count": restart_count,
                        "is_critical": is_critical,
                        "retry_interval_seconds": retry_interval,
                    },
                    source="DaemonManager",
                )
                logger.info(f"Emitted DAEMON_DEGRADED_MODE for {daemon_type.value}")
            except Exception as e:
                logger.debug(f"Failed to emit DAEMON_DEGRADED_MODE via router: {e}")

        except ImportError:
            logger.debug("data_events not available for DAEMON_DEGRADED_MODE")
        except Exception as e:
            logger.debug(f"Failed to emit DAEMON_DEGRADED_MODE: {e}")

    def _record_restart_legacy(self, daemon_type: DaemonType) -> bool:
        """Legacy restart recording (when degraded mode is disabled).

        December 2025: Preserved for backward compatibility when
        RINGRIFT_DEGRADED_MODE_ENABLED=false.
        """
        daemon_name = daemon_type.value
        current_time = time.time()

        # Check if already permanently failed
        if daemon_name in self._permanently_failed:
            failed_at = self._permanently_failed[daemon_name]
            age_seconds = current_time - failed_at

            if age_seconds >= PERMANENT_FAILURE_RECOVERY_SECONDS:
                logger.info(
                    f"[DaemonManager] {daemon_name} auto-recovered after "
                    f"{age_seconds / 3600:.1f}h in permanent failure state"
                )
                del self._permanently_failed[daemon_name]
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

        self._restart_timestamps[daemon_name].append(current_time)
        self._restart_timestamps[daemon_name] = [
            ts for ts in self._restart_timestamps[daemon_name]
            if ts > current_time - 3600
        ]

        hourly_restarts = len(self._restart_timestamps[daemon_name])
        if hourly_restarts > MAX_RESTARTS_PER_HOUR:
            logger.error(
                f"[DaemonManager] {daemon_name} exceeded hourly restart limit "
                f"({hourly_restarts} > {MAX_RESTARTS_PER_HOUR}), marking permanently failed."
            )
            self._permanently_failed[daemon_name] = current_time
            self._save_restart_counts()
            self._emit_permanently_failed_event(daemon_type)
            return False

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

    def _get_adaptive_cascade_threshold(self) -> int:
        """Compute adaptive cascade threshold based on critical daemon health.

        December 29, 2025: Added for cascade prevention enhancement.
        If most critical daemons are healthy, allow more restarts before tripping.
        If many critical daemons are failing, trip sooner to protect the system.

        Returns:
            Adaptive threshold (10-20 restarts based on health)
        """
        try:
            # Count healthy critical daemons
            healthy_critical = 0
            total_critical = len(CRITICAL_DAEMONS)

            for daemon_type in CRITICAL_DAEMONS:
                if daemon_type in self._daemons:
                    info = self._daemons[daemon_type]
                    if info.state == DaemonState.RUNNING:
                        healthy_critical += 1

            # Dec 30, 2025: During startup grace period, allow many restarts
            # This prevents circuit breaker from blocking normal daemon initialization
            uptime = time.time() - self._start_time
            if uptime < CASCADE_STARTUP_GRACE_PERIOD:
                logger.debug(
                    f"[DaemonManager] Startup grace period ({uptime:.0f}s of "
                    f"{CASCADE_STARTUP_GRACE_PERIOD}s), using higher threshold "
                    f"{CASCADE_STARTUP_THRESHOLD}"
                )
                return CASCADE_STARTUP_THRESHOLD

            # Adjust threshold based on health ratio
            if total_critical == 0:
                return CASCADE_RESTART_THRESHOLD  # Default if no critical daemons defined

            health_ratio = healthy_critical / total_critical

            if health_ratio >= 0.8:
                # 80%+ critical daemons healthy - allow more restarts
                return 20
            elif health_ratio >= 0.6:
                # 60-80% healthy - use default threshold
                return CASCADE_RESTART_THRESHOLD
            elif health_ratio >= 0.4:
                # 40-60% healthy - be more protective
                return 12
            else:
                # <40% healthy - trip early to stabilize
                return 10

        except (KeyError, AttributeError, TypeError) as e:
            # Dec 29, 2025: Narrowed from bare Exception
            # KeyError: daemon not in _daemons dict
            # AttributeError: missing state attribute
            # TypeError: invalid comparison
            logger.debug(f"Error computing adaptive threshold: {e}")
            return CASCADE_RESTART_THRESHOLD

    def _check_cascade_circuit_breaker(self, daemon_type: DaemonType | None = None) -> bool:
        """Check if cascade circuit breaker allows restarts.

        Dec 30, 2025: Delegates to hierarchical CascadeBreakerManager.
        Now supports per-daemon checks with category-based and critical exemptions.

        Args:
            daemon_type: Optional daemon to check. If None, performs legacy global check.

        Returns:
            True if restarts are allowed, False if circuit breaker is open
        """
        if daemon_type is not None:
            # Use new hierarchical breaker
            allowed, reason = self._cascade_breaker.can_restart(daemon_type)
            if not allowed:
                logger.debug(
                    f"[DaemonManager] Cascade breaker blocked {daemon_type.value}: {reason}"
                )
            return allowed

        # Legacy fallback: check if any global breaker is open
        status = self._cascade_breaker.get_status()
        if status["global"]["breaker_open"]:
            remaining = status["global"].get("cooldown_remaining", 0)
            logger.warning(
                f"[DaemonManager] Global cascade breaker OPEN - "
                f"restarts blocked for {remaining:.0f}s more"
            )
            return False

        return True

    def _record_global_restart(self, daemon_type: DaemonType) -> None:
        """Record a restart in the hierarchical cascade breaker.

        Dec 30, 2025: Delegates to CascadeBreakerManager for per-category tracking.
        The new breaker handles both category-level and global threshold checking.

        Args:
            daemon_type: Type of daemon being restarted
        """
        # Record restart in hierarchical breaker
        self._cascade_breaker.record_restart(daemon_type)

        # Update legacy state for backward compatibility during transition
        # Can be removed once all code migrates to _cascade_breaker
        status = self._cascade_breaker.get_status()
        self._cascade_breaker_open = status["global"]["breaker_open"]
        if self._cascade_breaker_open:
            # Legacy code may check this variable
            self._cascade_breaker_opened_at = time.time()

    async def _emit_circuit_breaker_event(
        self, restart_count: int, triggered_by: DaemonType
    ) -> None:
        """Emit event when cascade circuit breaker trips.

        Args:
            restart_count: Number of restarts that triggered the breaker
            triggered_by: Daemon that caused the breaker to trip
        """
        try:
            from app.coordination.event_router import publish

            await publish(
                "daemon.cascade_breaker_tripped",
                {
                    "restart_count": restart_count,
                    "threshold": CASCADE_RESTART_THRESHOLD,
                    "window_seconds": CASCADE_RESTART_WINDOW_SECONDS,
                    "cooldown_seconds": CASCADE_COOLDOWN_SECONDS,
                    "triggered_by": triggered_by.value,
                },
                source="DaemonManager",
            )
        except Exception as e:
            logger.debug(f"Failed to emit circuit breaker event: {e}")

    async def _cascade_recovery(self) -> None:
        """Proactively restart critical daemons that were blocked during cascade cooldown.

        December 29, 2025: Added to improve recovery time after cascade events.
        When the circuit breaker closes, some critical daemons may still be in
        a failed state. This method proactively restarts them to ensure the
        system recovers quickly.
        """
        try:
            restarted = []
            for daemon_type in CRITICAL_DAEMONS:
                if daemon_type in self._daemons:
                    info = self._daemons[daemon_type]
                    # Only restart if not already running
                    if info.state != DaemonState.RUNNING:
                        logger.info(
                            f"[DaemonManager] Cascade recovery: restarting "
                            f"{daemon_type.value} (state: {info.state.value})"
                        )
                        success = await self.start(daemon_type)
                        if success:
                            restarted.append(daemon_type.value)

            if restarted:
                logger.info(
                    f"[DaemonManager] Cascade recovery complete: "
                    f"restarted {len(restarted)} critical daemons: {restarted}"
                )

                # Emit recovery event
                try:
                    from app.coordination.event_router import publish

                    await publish(
                        "daemon.cascade_recovery_complete",
                        {
                            "restarted_daemons": restarted,
                            "count": len(restarted),
                        },
                        source="DaemonManager",
                    )
                except (ImportError, RuntimeError, AttributeError):
                    pass
            else:
                logger.debug("[DaemonManager] Cascade recovery: all critical daemons already running")

        except Exception as e:
            logger.warning(f"[DaemonManager] Cascade recovery failed: {e}")

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """Get status of the cascade circuit breaker.

        Dec 30, 2025: Returns comprehensive status from hierarchical breaker manager.
        Includes both global and per-category breaker states.

        Returns:
            Dict with hierarchical breaker state, including:
            - global: Global breaker status
            - categories: Per-category breaker status
            - uptime_seconds: Manager uptime
            - total_allowed/total_blocked: Overall stats
        """
        return self._cascade_breaker.get_status()
