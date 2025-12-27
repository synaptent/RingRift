"""Daemon Lifecycle Management - Extracted from DaemonManager.

This module handles the lifecycle operations for daemons:
- Starting individual daemons and all daemons
- Stopping individual daemons and all daemons
- Graceful shutdown coordination
- Restart logic for failed daemons
- Dependency-based ordering (topological sort)

December 2025: Extracted from daemon_manager.py to reduce file size and
improve maintainability. The DaemonLifecycleManager receives references
to the daemon registry and uses composition (not inheritance).

Usage:
    from app.coordination.daemon_lifecycle import DaemonLifecycleManager

    # Create with references to parent manager's state
    lifecycle = DaemonLifecycleManager(
        daemons=self._daemons,
        factories=self._factories,
        config=self.config,
        shutdown_event=self._shutdown_event,
        lock=self._lock,
        update_daemon_state=self._update_daemon_state,
    )

    # Delegate lifecycle calls
    await lifecycle.start(DaemonType.EVENT_ROUTER)
    await lifecycle.start_all()
    await lifecycle.shutdown()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, Protocol

from app.core.async_context import fire_and_forget, safe_create_task
from app.coordination.daemon_types import (
    DAEMON_RESTART_RESET_AFTER,
    DAEMON_STARTUP_ORDER,
    MAX_RESTART_DELAY,
    DaemonInfo,
    DaemonManagerConfig,
    DaemonState,
    DaemonType,
    _check_deprecated_daemon,
    validate_startup_order_consistency,
)


class DependencyValidationError(Exception):
    """Raised when daemon dependencies are invalid.

    Phase 12 (December 2025): Strict validation prevents startup with:
    - Circular dependencies (would cause deadlock)
    - Missing dependencies (would cause startup failure)
    - Self-dependencies (configuration error)
    """

    pass

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StateUpdateCallback(Protocol):
    """Protocol for the state update callback from DaemonManager."""

    def __call__(
        self,
        info: DaemonInfo,
        new_state: DaemonState,
        reason: str = "",
        error: str | None = None,
    ) -> None:
        """Update daemon state and emit status changed event."""
        ...


class DaemonLifecycleManager:
    """Manages lifecycle operations for daemons.

    This class is designed to be used by composition within DaemonManager.
    It receives references to the shared state (daemon registry, factories,
    config, events, lock) rather than owning them.

    Features:
    - Start/stop individual daemons with dependency checking
    - Start/stop all daemons in correct dependency order
    - Graceful shutdown with timeout escalation
    - Restart failed daemons with exponential backoff
    - Topological sort for dependency ordering
    """

    def __init__(
        self,
        daemons: dict[DaemonType, DaemonInfo],
        factories: dict[DaemonType, Callable[[], Coroutine[Any, Any, None]]],
        config: DaemonManagerConfig,
        shutdown_event: asyncio.Event,
        lock: asyncio.Lock,
        update_daemon_state: StateUpdateCallback,
        running_flag_getter: Callable[[], bool],
        running_flag_setter: Callable[[bool], None],
    ):
        """Initialize the lifecycle manager.

        Args:
            daemons: Reference to the daemon registry dict
            factories: Reference to the factory registry dict
            config: Daemon manager configuration
            shutdown_event: Event signaling shutdown
            lock: Lock for thread-safe operations
            update_daemon_state: Callback to update daemon state and emit events
            running_flag_getter: Callable to get the running flag
            running_flag_setter: Callable to set the running flag
        """
        self._daemons = daemons
        self._factories = factories
        self.config = config
        self._shutdown_event = shutdown_event
        self._lock = lock
        self._update_daemon_state = update_daemon_state
        self._get_running = running_flag_getter
        self._set_running = running_flag_setter

    def _emit_daemon_lifecycle_event(
        self, daemon_type: DaemonType, event_type: str
    ) -> None:
        """Emit DAEMON_STARTED or DAEMON_STOPPED event (Dec 2025).

        Uses fire-and-forget pattern since lifecycle events are informational
        and should not block daemon startup/shutdown.

        Args:
            daemon_type: The daemon type that changed state
            event_type: Either "DAEMON_STARTED" or "DAEMON_STOPPED"
        """
        try:
            from app.coordination.event_router import emit_sync

            emit_sync(
                event_type=event_type,
                data={
                    "daemon_type": daemon_type.value,
                    "timestamp": time.time(),
                },
            )
            logger.debug(f"Emitted {event_type} for {daemon_type.value}")
        except (ImportError, RuntimeError, TypeError) as e:
            # Non-critical - don't fail daemon operations for event failures
            logger.debug(f"Failed to emit {event_type}: {e}")

    async def start(self, daemon_type: DaemonType) -> bool:
        """Start a specific daemon.

        P0.3 Dec 2025: Now waits for daemon to signal readiness before returning.
        This prevents race conditions where dependent daemons start before
        their dependencies have completed initialization.

        Args:
            daemon_type: Type of daemon to start

        Returns:
            True if started successfully
        """
        # Check for deprecated daemon types and emit warning
        _check_deprecated_daemon(daemon_type)

        async with self._lock:
            info = self._daemons.get(daemon_type)
            if info is None:
                logger.error(f"Unknown daemon type: {daemon_type}")
                return False

            if info.state == DaemonState.RUNNING:
                logger.debug(f"{daemon_type.value} already running")
                return True

            # Check dependencies - wait for them to be READY not just RUNNING
            # Dec 2025: Added timeout to prevent deadlocks
            DEPENDENCY_READY_TIMEOUT = 30.0  # seconds
            for dep in info.depends_on:
                dep_info = self._daemons.get(dep)
                if dep_info is None or dep_info.state != DaemonState.RUNNING:
                    logger.warning(f"Cannot start {daemon_type.value}: dependency {dep.value} not running")
                    return False
                # P0.3: Wait for dependency to signal readiness with timeout
                if dep_info.ready_event and not dep_info.ready_event.is_set():
                    logger.info(
                        f"Waiting for {dep.value} to be ready before starting {daemon_type.value}"
                    )
                    try:
                        # Release lock while waiting to prevent deadlock
                        self._lock.release()
                        try:
                            await asyncio.wait_for(
                                dep_info.ready_event.wait(),
                                timeout=DEPENDENCY_READY_TIMEOUT,
                            )
                        finally:
                            await self._lock.acquire()

                        # December 2025: Re-validate state after re-acquiring lock
                        # Another coroutine may have modified daemon state while waiting
                        info = self._daemons.get(daemon_type)
                        if info is None:
                            logger.warning(
                                f"{daemon_type.value} was removed while waiting for dependencies"
                            )
                            return False
                        if info.state == DaemonState.RUNNING:
                            logger.debug(
                                f"{daemon_type.value} was started by another coroutine while waiting"
                            )
                            return True  # Already started, success

                    except asyncio.TimeoutError:
                        logger.error(
                            f"Dependency {dep.value} not ready after {DEPENDENCY_READY_TIMEOUT}s, "
                            f"cannot start {daemon_type.value}"
                        )
                        return False

            # Get factory
            factory = self._factories.get(daemon_type)
            if factory is None:
                logger.error(f"No factory registered for {daemon_type.value}")
                return False

            # P0.3: Create readiness event for this daemon.
            #
            # NOTE: Most daemon factories in this codebase do not explicitly call
            # mark_daemon_ready(). To avoid deadlocking dependency start order,
            # we auto-set ready after a short delay for backward compatibility.
            info.ready_event = asyncio.Event()

            # Start daemon
            info.state = DaemonState.STARTING
            try:
                info.task = safe_create_task(
                    self._run_daemon(daemon_type, factory),
                    name=f"daemon_{daemon_type.value}",
                )
                info.start_time = time.time()
                info.state = DaemonState.RUNNING

                # Dec 2025: Auto-set readiness after delay for backward compatibility.
                # Daemons can call mark_daemon_ready() earlier for explicit signaling.
                # This delay allows daemons to complete critical initialization.
                # Dec 27, 2025: Increased from 0.5s to 2s - some daemons need more time
                # for database connections, event subscriptions, and other I/O.
                async def _auto_set_ready():
                    await asyncio.sleep(2.0)  # Allow initialization to complete
                    if info.ready_event and not info.ready_event.is_set():
                        info.ready_event.set()
                        logger.debug(f"{daemon_type.value} auto-marked as ready")

                safe_create_task(
                    _auto_set_ready(),
                    name=f"auto_ready_{daemon_type.value}",
                )

                logger.info(f"Started daemon: {daemon_type.value}")
                # Emit DAEMON_STARTED event (Dec 2025)
                self._emit_daemon_lifecycle_event(daemon_type, "DAEMON_STARTED")
                return True

            except (RuntimeError, OSError, ImportError) as e:
                info.state = DaemonState.FAILED
                info.last_error = str(e)
                logger.error(f"Failed to start {daemon_type.value}: {e}")
                return False

    async def _run_daemon(
        self,
        daemon_type: DaemonType,
        factory: Callable[[], Coroutine[Any, Any, None]]
    ) -> None:
        """Run a daemon with error handling and restart logic.

        Features:
        - Import errors are treated as permanent failures (IMPORT_FAILED state)
        - Restart count resets after DAEMON_RESTART_RESET_AFTER seconds of stability
        - Exponential backoff for restart delays (capped at MAX_RESTART_DELAY)
        """
        info = self._daemons[daemon_type]

        while not self._shutdown_event.is_set():
            # Reset restart counter after period of stability
            if info.last_failure_time > 0:
                time_since_failure = time.time() - info.last_failure_time
                if time_since_failure > DAEMON_RESTART_RESET_AFTER:
                    if info.restart_count > 0:
                        logger.info(
                            f"{daemon_type.value} stable for {DAEMON_RESTART_RESET_AFTER}s, "
                            f"resetting restart count from {info.restart_count} to 0"
                        )
                        info.restart_count = 0
                        info.last_failure_time = 0.0

            try:
                info.stable_since = time.time()  # Mark start of stable period
                await factory()
                # December 2025 fix: If factory returns normally (no exception),
                # the daemon completed (either finished its work or skipped on coordinator).
                # Break the loop to prevent infinite restarts.
                logger.debug(f"{daemon_type.value} factory completed normally, stopping")
                break
            except asyncio.CancelledError:
                logger.debug(f"{daemon_type.value} cancelled")
                break
            except ImportError as e:
                # Import errors are permanent - require code/environment fix
                info.last_error = str(e)
                info.import_error = str(e)
                # P0.5: Use helper for event emission
                self._update_daemon_state(
                    info, DaemonState.IMPORT_FAILED,
                    reason="import_error", error=str(e)
                )
                info.last_failure_time = time.time()
                logger.error(
                    f"{daemon_type.value} import failed permanently: {e}. "
                    f"Fix the import and restart the daemon manager."
                )
                # Don't retry import failures - they need manual intervention
                break
            except (RuntimeError, OSError, ConnectionError, asyncio.CancelledError) as e:
                info.last_error = str(e)
                info.last_failure_time = time.time()
                info.stable_since = 0.0
                if isinstance(e, asyncio.CancelledError):
                    raise  # Re-raise cancellation
                logger.error(f"{daemon_type.value} failed: {e}")

                if not info.auto_restart:
                    # P0.5: Use helper for event emission
                    self._update_daemon_state(
                        info, DaemonState.FAILED,
                        reason="exception", error=str(e)
                    )
                    break

                if info.restart_count >= info.max_restarts:
                    logger.error(f"{daemon_type.value} exceeded max restarts, stopping")
                    # P0.5: Use helper for event emission
                    self._update_daemon_state(
                        info, DaemonState.FAILED,
                        reason="max_restarts_exceeded", error=str(e)
                    )
                    break

                # Restart with exponential backoff + jitter to prevent thundering herd
                info.restart_count += 1
                # P0.5: Use helper for event emission
                self._update_daemon_state(
                    info, DaemonState.RESTARTING,
                    reason="auto_restart", error=str(e)
                )
                base_delay = min(info.restart_delay * (2 ** (info.restart_count - 1)), MAX_RESTART_DELAY)
                # Add +/-10% jitter to prevent all daemons restarting at same time
                jitter = base_delay * 0.1 * (random.random() * 2 - 1)  # -10% to +10%
                delay = max(1.0, base_delay + jitter)  # Minimum 1 second
                logger.info(f"Restarting {daemon_type.value} (attempt {info.restart_count}) in {delay:.1f}s (base={base_delay:.1f}s)")
                await asyncio.sleep(delay)
                # P0.5: Use helper for event emission
                self._update_daemon_state(
                    info, DaemonState.RUNNING,
                    reason="restart_complete"
                )
                info.start_time = time.time()

        if info.state not in (DaemonState.FAILED, DaemonState.IMPORT_FAILED):
            info.state = DaemonState.STOPPED

    async def stop(self, daemon_type: DaemonType) -> bool:
        """Stop a specific daemon with timeout escalation.

        Uses a three-phase shutdown approach:
        1. Cancel task and wait for graceful shutdown (shutdown_timeout)
        2. If still running, wait additional grace period (force_kill_timeout)
        3. If still stuck, log error but continue (task may be leaked)

        Args:
            daemon_type: Type of daemon to stop

        Returns:
            True if stopped successfully (or was already stopped)
        """
        async with self._lock:
            info = self._daemons.get(daemon_type)
            if info is None:
                return False

            if info.state == DaemonState.STOPPED:
                return True

            info.state = DaemonState.STOPPING

            if info.task is not None:
                # Phase 1: Cancel and wait for graceful shutdown
                info.task.cancel()
                try:
                    await asyncio.wait_for(info.task, timeout=self.config.shutdown_timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout stopping {daemon_type.value} after {self.config.shutdown_timeout}s, "
                        f"waiting additional {self.config.force_kill_timeout}s"
                    )
                    # Phase 2: Additional grace period for stubborn tasks
                    try:
                        await asyncio.wait_for(info.task, timeout=self.config.force_kill_timeout)
                    except asyncio.TimeoutError:
                        # Phase 3: Task is truly stuck - log but don't block
                        logger.error(
                            f"Daemon {daemon_type.value} failed to stop after "
                            f"{self.config.shutdown_timeout + self.config.force_kill_timeout}s total. "
                            f"Task may be leaked. Consider investigating the daemon's shutdown handler."
                        )
                        # Clear the task reference to prevent memory leaks
                        # The task is likely stuck in a blocking operation
                        info.task = None
                    except asyncio.CancelledError:
                        pass
                except asyncio.CancelledError:
                    pass

            info.state = DaemonState.STOPPED
            info.task = None
            logger.info(f"Stopped daemon: {daemon_type.value}")
            # Emit DAEMON_STOPPED event (Dec 2025)
            self._emit_daemon_lifecycle_event(daemon_type, "DAEMON_STOPPED")
            return True

    async def restart_failed_daemon(
        self,
        daemon_type: DaemonType,
        force: bool = False,
    ) -> bool:
        """Restart a failed daemon, optionally resetting its restart count.

        This method allows manual recovery of daemons that have exceeded their
        restart limit or have import errors.

        Args:
            daemon_type: Type of daemon to restart
            force: If True, reset restart count and clear import error

        Returns:
            True if restart initiated successfully
        """
        async with self._lock:
            info = self._daemons.get(daemon_type)
            if info is None:
                logger.error(f"Unknown daemon type: {daemon_type}")
                return False

            if info.state not in (DaemonState.FAILED, DaemonState.IMPORT_FAILED, DaemonState.STOPPED):
                logger.warning(
                    f"Cannot restart {daemon_type.value}: state is {info.state.value}, "
                    f"expected FAILED, IMPORT_FAILED, or STOPPED"
                )
                return False

            # Import failures need force=True to retry
            if info.state == DaemonState.IMPORT_FAILED and not force:
                logger.warning(
                    f"{daemon_type.value} has import error: {info.import_error}. "
                    f"Use force=True to retry after fixing the import."
                )
                return False

            if force:
                logger.info(f"Force restarting {daemon_type.value}, resetting counters")
                info.restart_count = 0
                info.import_error = None
                info.last_error = None
                info.last_failure_time = 0.0

            logger.info(f"Restarting failed daemon: {daemon_type.value}")

        # Release lock before starting (start() acquires its own lock)
        return await self.start(daemon_type)

    async def start_all(
        self,
        types: list[DaemonType] | None = None,
        on_started_callback: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> dict[DaemonType, bool]:
        """Start all (or specified) daemons in dependency order.

        Phase 12 (December 2025): Now validates dependency graph before startup.
        Raises DependencyValidationError if invalid dependencies are detected.

        Args:
            types: Specific daemon types to start (all if None)
            on_started_callback: Optional async callback to run after starting daemons
                (e.g., for starting health loop, wiring events, etc.)

        Returns:
            Dict mapping daemon type to start success

        Raises:
            DependencyValidationError: If circular, missing, or self-dependencies detected
        """
        # Check if master loop is running (December 2025)
        # Warn user if starting daemons outside of master loop
        try:
            from app.coordination.master_loop_guard import check_or_warn

            if not check_or_warn("daemon management"):
                logger.warning(
                    "[DaemonManager] For full automation, use: python scripts/master_loop.py"
                )
        except ImportError:
            pass  # master_loop_guard not available

        results: dict[DaemonType, bool] = {}
        types_to_start = types or list(self._factories.keys())

        # December 2025: Validate DAEMON_STARTUP_ORDER is consistent with DAEMON_DEPENDENCIES
        # This catches configuration bugs where startup order violates dependency constraints
        is_consistent, violations = validate_startup_order_consistency()
        if not is_consistent:
            for violation in violations:
                logger.error(f"[DaemonManager] Startup order violation: {violation}")
            raise DependencyValidationError(
                "DAEMON_STARTUP_ORDER is inconsistent with DAEMON_DEPENDENCIES:\n"
                + "\n".join(f"  - {v}" for v in violations)
            )

        # Phase 12: Validate dependency graph before startup
        # This fails fast with clear errors instead of proceeding with broken config
        self.validate_dependency_graph(types_to_start)

        # Sort by dependencies (topological sort)
        sorted_types = self._sort_by_dependencies(types_to_start)

        # P0 Critical Fix (Dec 2025): Apply critical startup order
        # DATA_PIPELINE and FEEDBACK_LOOP must start BEFORE AUTO_SYNC
        # to ensure event subscribers are ready when emitters start.
        # This prevents sync events from being lost.
        ordered_types = self._apply_critical_startup_order(sorted_types)

        for daemon_type in ordered_types:
            results[daemon_type] = await self.start(daemon_type)

        # Run callback if provided (for health loop, event wiring, etc.)
        if on_started_callback:
            await on_started_callback()

        return results

    async def stop_all(self) -> dict[DaemonType, bool]:
        """Stop all running daemons.

        Returns:
            Dict mapping daemon type to stop success
        """
        self._set_running(False)
        results: dict[DaemonType, bool] = {}

        # Stop in reverse dependency order
        sorted_types = list(reversed(self._sort_by_dependencies(list(self._daemons.keys()))))

        for daemon_type in sorted_types:
            results[daemon_type] = await self.stop(daemon_type)

        return results

    async def shutdown(
        self,
        health_task: asyncio.Task | None = None,
        pre_shutdown_callback: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        """Gracefully shutdown all daemons.

        Args:
            health_task: Optional health check task to cancel
            pre_shutdown_callback: Optional callback to run before stopping daemons
                (e.g., for stopping watchdog)
        """
        logger.info("DaemonManager shutting down...")
        self._shutdown_event.set()
        self._set_running(False)

        # Run pre-shutdown callback (e.g., stop watchdog)
        if pre_shutdown_callback:
            await pre_shutdown_callback()

        # Stop health check
        if health_task:
            health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await health_task

        # Stop all daemons
        await self.stop_all()

        logger.info("DaemonManager shutdown complete")

    def _sort_by_dependencies(self, types: list[DaemonType]) -> list[DaemonType]:
        """Sort daemon types by dependencies (topological sort)."""
        result: list[DaemonType] = []
        visited: set[DaemonType] = set()
        visiting: set[DaemonType] = set()

        def visit(dt: DaemonType) -> None:
            if dt in visited:
                return
            if dt in visiting:
                logger.warning(f"Circular dependency detected for {dt.value}")
                return

            visiting.add(dt)
            info = self._daemons.get(dt)
            if info:
                for dep in info.depends_on:
                    if dep in types:
                        visit(dep)

            visiting.remove(dt)
            visited.add(dt)
            result.append(dt)

        for dt in types:
            visit(dt)

        return result

    def _apply_critical_startup_order(self, types: list[DaemonType]) -> list[DaemonType]:
        """Reorder daemon list to respect DAEMON_STARTUP_ORDER.

        P0 Critical Fix (Dec 2025): This ensures that event subscribers
        (DATA_PIPELINE, FEEDBACK_LOOP) start BEFORE event emitters (AUTO_SYNC).
        Without this ordering, sync completion events would be lost because
        no handlers are registered when AUTO_SYNC emits them.

        The algorithm:
        1. Extract daemons that are in DAEMON_STARTUP_ORDER (in that order)
        2. Append remaining daemons (in their original dependency-sorted order)

        Args:
            types: Dependency-sorted daemon types

        Returns:
            Reordered list with critical daemons first
        """
        types_set = set(types)

        # 1. First, add daemons in DAEMON_STARTUP_ORDER (preserving that order)
        ordered: list[DaemonType] = []
        for critical_daemon in DAEMON_STARTUP_ORDER:
            if critical_daemon in types_set:
                ordered.append(critical_daemon)

        # 2. Then add remaining daemons (preserving dependency order)
        ordered_set = set(ordered)
        for daemon_type in types:
            if daemon_type not in ordered_set:
                ordered.append(daemon_type)

        if ordered != types:
            logger.debug(
                f"[DaemonManager] Applied critical startup order: "
                f"{[d.value for d in ordered[:len(DAEMON_STARTUP_ORDER)]]}"
            )

        return ordered

    def _get_dependents(self, daemon_type: DaemonType) -> list[DaemonType]:
        """Get all daemons that depend on the given daemon type.

        Used for cascading restarts when a dependency fails.
        Returns daemons in order so that direct dependents come first.
        """
        dependents: list[DaemonType] = []
        for dt, info in self._daemons.items():
            if daemon_type in info.depends_on:
                dependents.append(dt)
        return dependents

    def validate_dependency_graph(self, types: list[DaemonType] | None = None) -> None:
        """Validate the dependency graph before startup.

        Phase 12 (December 2025): Strict pre-flight validation that prevents
        startup with invalid dependencies. Raises DependencyValidationError
        instead of silently proceeding with broken configuration.

        Validates:
        1. No self-dependencies (daemon depending on itself)
        2. All dependencies are registered (no dangling references)
        3. No circular dependencies (would cause deadlock)

        Args:
            types: Specific daemon types to validate (all if None)

        Raises:
            DependencyValidationError: If any validation fails
        """
        types_to_check = set(types) if types else set(self._daemons.keys())

        # Phase 1: Check for self-dependencies and missing dependencies
        for dt in types_to_check:
            info = self._daemons.get(dt)
            if info is None:
                continue

            for dep in info.depends_on:
                # Self-dependency check
                if dep == dt:
                    raise DependencyValidationError(
                        f"Self-dependency detected: {dt.value} depends on itself. "
                        f"Remove {dt.value} from its own depends_on list."
                    )

                # Missing dependency check
                if dep not in self._daemons:
                    raise DependencyValidationError(
                        f"Missing dependency: {dt.value} depends on {dep.value}, "
                        f"but {dep.value} is not registered. "
                        f"Register {dep.value} before starting daemons."
                    )

                # Dependency not in types_to_check but should be started first
                if dep not in types_to_check and dep in self._daemons:
                    dep_info = self._daemons[dep]
                    if dep_info.state != DaemonState.RUNNING:
                        raise DependencyValidationError(
                            f"Dependency not included: {dt.value} depends on {dep.value}, "
                            f"but {dep.value} is not in the startup list and not running. "
                            f"Either include {dep.value} in types or start it first."
                        )

        # Phase 2: Detect circular dependencies using DFS
        # States: 0=unvisited, 1=visiting (in current path), 2=visited
        state: dict[DaemonType, int] = {dt: 0 for dt in types_to_check}
        path: list[DaemonType] = []

        def dfs(dt: DaemonType) -> list[DaemonType] | None:
            """DFS to detect cycles. Returns cycle path if found, None otherwise."""
            if state[dt] == 2:  # Already fully visited
                return None
            if state[dt] == 1:  # Found cycle - in current path
                # Extract cycle from path
                cycle_start = path.index(dt)
                return path[cycle_start:] + [dt]

            state[dt] = 1
            path.append(dt)

            info = self._daemons.get(dt)
            if info:
                for dep in info.depends_on:
                    if dep in state:  # Only check deps in our validation set
                        cycle = dfs(dep)
                        if cycle:
                            return cycle

            path.pop()
            state[dt] = 2
            return None

        for dt in types_to_check:
            if state[dt] == 0:
                cycle = dfs(dt)
                if cycle:
                    cycle_str = " -> ".join(d.value for d in cycle)
                    raise DependencyValidationError(
                        f"Circular dependency detected: {cycle_str}. "
                        f"Break the cycle by removing one dependency."
                    )

        logger.debug(
            f"Dependency graph validated: {len(types_to_check)} daemons, "
            f"no cycles or missing dependencies"
        )

    def health_check(self) -> "HealthCheckResult":
        """Perform health check for the lifecycle manager (CoordinatorProtocol compliance).

        Returns:
            HealthCheckResult with lifecycle manager health status
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        # Count daemon states
        running = sum(
            1 for info in self._daemons.values() if info.state == DaemonState.RUNNING
        )
        failed = sum(
            1 for info in self._daemons.values() if info.state == DaemonState.FAILED
        )
        total = len(self._daemons)

        # Unhealthy if more than 20% of daemons failed
        is_healthy = failed < max(1, total * 0.2) if total > 0 else True
        status = CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.DEGRADED

        return HealthCheckResult(
            healthy=is_healthy,
            status=status,
            message="" if is_healthy else f"High failure rate: {failed}/{total} daemons failed",
            details={
                "running_daemons": running,
                "failed_daemons": failed,
                "total_daemons": total,
                "shutdown_requested": self._shutdown_event.is_set(),
            },
        )
