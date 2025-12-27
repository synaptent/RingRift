"""Base Loop Framework for P2P Orchestrator Background Loops.

Created as part of Phase 4 of the P2P orchestrator decomposition (December 2025).

This module provides:
- BaseLoop: Abstract base class with consistent error handling
- Exponential backoff on errors
- Loop lifecycle management (running flag, graceful shutdown)
- Metrics/logging for loop execution
- Abstract _run_once method for subclasses

Usage:
    class MyLoop(BaseLoop):
        def __init__(self):
            super().__init__(name="my_loop", interval=60.0)

        async def _run_once(self) -> None:
            # Do work here
            pass

    loop = MyLoop()
    await loop.run_forever()  # Runs until stop() is called

    # Graceful shutdown
    loop.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from scripts.p2p.metrics_manager import MetricsManager

logger = logging.getLogger(__name__)


@dataclass
class LoopStats:
    """Statistics for loop execution tracking."""

    name: str
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    consecutive_errors: int = 0
    last_run_time: float = 0.0
    last_success_time: float = 0.0
    last_error_time: float = 0.0
    last_error_message: str = ""
    total_run_duration: float = 0.0
    last_run_duration: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_runs == 0:
            return 100.0
        return (self.successful_runs / self.total_runs) * 100.0

    @property
    def avg_run_duration(self) -> float:
        """Calculate average run duration in seconds."""
        if self.successful_runs == 0:
            return 0.0
        return self.total_run_duration / self.successful_runs

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "consecutive_errors": self.consecutive_errors,
            "last_run_time": self.last_run_time,
            "last_success_time": self.last_success_time,
            "last_error_time": self.last_error_time,
            "last_error_message": self.last_error_message,
            "success_rate": self.success_rate,
            "avg_run_duration_ms": self.avg_run_duration * 1000,
            "last_run_duration_ms": self.last_run_duration * 1000,
        }


@dataclass
class BackoffConfig:
    """Configuration for exponential backoff behavior."""

    initial_delay: float = 1.0  # Initial delay after first error (seconds)
    max_delay: float = 300.0  # Maximum delay cap (5 minutes)
    multiplier: float = 2.0  # Exponential multiplier
    jitter: float = 0.1  # Random jitter factor (0.1 = +/-10%)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.initial_delay <= 0:
            raise ValueError("initial_delay must be > 0")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay")
        if self.multiplier <= 1:
            raise ValueError("multiplier must be > 1")
        if self.jitter < 0 or self.jitter > 1:
            raise ValueError("jitter must be between 0 and 1")

    def calculate_delay(self, consecutive_errors: int) -> float:
        """Calculate backoff delay based on consecutive error count.

        Uses exponential backoff with jitter to prevent thundering herd.

        Args:
            consecutive_errors: Number of consecutive errors

        Returns:
            Delay in seconds
        """
        if consecutive_errors <= 0:
            return 0.0

        # Exponential backoff: initial * multiplier^(errors-1)
        delay = self.initial_delay * (self.multiplier ** (consecutive_errors - 1))

        # Cap at max delay
        delay = min(delay, self.max_delay)

        # Add jitter
        import random

        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)

        return max(0.0, delay)


class BaseLoop(ABC):
    """Abstract base class for all P2P orchestrator background loops.

    Provides:
    - Consistent error handling with exponential backoff
    - Loop lifecycle management (running flag, graceful shutdown)
    - Metrics/logging for loop execution
    - Abstract _run_once method for subclasses

    Subclasses must implement:
    - _run_once(): The core loop logic executed each interval

    Optionally override:
    - _on_start(): Called when loop starts
    - _on_stop(): Called when loop stops
    - _on_error(error): Called when an error occurs
    """

    def __init__(
        self,
        name: str,
        interval: float,
        *,
        backoff_config: BackoffConfig | None = None,
        metrics_manager: MetricsManager | None = None,
        enabled: bool = True,
        depends_on: list[str] | None = None,
    ):
        """Initialize the base loop.

        Args:
            name: Human-readable name for logging and metrics
            interval: Normal interval between runs in seconds
            backoff_config: Configuration for exponential backoff (uses defaults if None)
            metrics_manager: Optional MetricsManager for recording metrics
            enabled: Whether the loop is enabled (can be toggled at runtime)
            depends_on: List of loop names that must start before this loop (Dec 2025)
        """
        self.name = name
        self.interval = interval
        self.backoff_config = backoff_config or BackoffConfig()
        self.metrics_manager = metrics_manager
        self.enabled = enabled
        self.depends_on = depends_on or []

        # Lifecycle state
        self._running = False
        self._task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

        # Statistics
        self._stats = LoopStats(name=name)

        # Callbacks
        self._error_callbacks: list[Callable[[Exception], None]] = []

    @property
    def running(self) -> bool:
        """Check if the loop is currently running."""
        return self._running

    @property
    def stats(self) -> LoopStats:
        """Get loop execution statistics."""
        return self._stats

    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add a callback to be invoked when an error occurs.

        Args:
            callback: Function that takes an Exception as argument
        """
        self._error_callbacks.append(callback)

    def remove_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Remove an error callback.

        Args:
            callback: Previously registered callback function
        """
        if callback in self._error_callbacks:
            self._error_callbacks.remove(callback)

    async def run_forever(self) -> None:
        """Run the loop until stop() is called.

        This method:
        1. Calls _on_start() once
        2. Repeatedly calls _run_once() with interval delays
        3. Applies exponential backoff on errors
        4. Calls _on_stop() when shutting down
        """
        if self._running:
            logger.warning(f"[{self.name}] Loop is already running")
            return

        self._running = True
        self._shutdown_event.clear()
        logger.info(f"[{self.name}] Starting loop (interval={self.interval}s)")

        try:
            await self._on_start()

            while self._running and not self._shutdown_event.is_set():
                if not self.enabled:
                    # Loop is disabled, wait for interval and check again
                    await self._interruptible_sleep(self.interval)
                    continue

                run_start = time.time()

                try:
                    await self._run_once()

                    # Success - update stats
                    run_duration = time.time() - run_start
                    self._stats.total_runs += 1
                    self._stats.successful_runs += 1
                    self._stats.consecutive_errors = 0
                    self._stats.last_run_time = run_start
                    self._stats.last_success_time = time.time()
                    self._stats.last_run_duration = run_duration
                    self._stats.total_run_duration += run_duration

                    # Record metric if manager available
                    if self.metrics_manager:
                        self.metrics_manager.record_metric(
                            f"loop_{self.name}_duration_ms",
                            run_duration * 1000,
                        )

                    # Normal interval delay
                    await self._interruptible_sleep(self.interval)

                except asyncio.CancelledError:
                    # Task was cancelled - exit cleanly
                    logger.info(f"[{self.name}] Loop cancelled")
                    raise

                except Exception as e:
                    # Error - update stats and apply backoff
                    self._stats.total_runs += 1
                    self._stats.failed_runs += 1
                    self._stats.consecutive_errors += 1
                    self._stats.last_run_time = run_start
                    self._stats.last_error_time = time.time()
                    self._stats.last_error_message = str(e)

                    # Log error
                    logger.error(
                        f"[{self.name}] Error (attempt {self._stats.consecutive_errors}): {e}",
                        exc_info=True if self._stats.consecutive_errors == 1 else False,
                    )

                    # Record error metric
                    if self.metrics_manager:
                        self.metrics_manager.record_metric(
                            f"loop_{self.name}_errors",
                            1,
                            metadata={"error": str(e)[:200]},
                        )

                    # Call error callbacks
                    await self._on_error(e)
                    for callback in self._error_callbacks:
                        try:
                            callback(e)
                        except Exception as cb_err:
                            logger.warning(f"[{self.name}] Error callback failed: {cb_err}")

                    # Calculate and apply backoff delay
                    backoff_delay = self.backoff_config.calculate_delay(
                        self._stats.consecutive_errors
                    )
                    if backoff_delay > 0:
                        logger.info(
                            f"[{self.name}] Backing off for {backoff_delay:.1f}s "
                            f"after {self._stats.consecutive_errors} consecutive errors"
                        )
                        await self._interruptible_sleep(backoff_delay)
                    else:
                        await self._interruptible_sleep(self.interval)

        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            try:
                await self._on_stop()
            except Exception as e:
                logger.error(f"[{self.name}] Error in _on_stop: {e}")
            logger.info(f"[{self.name}] Loop stopped")

    async def _interruptible_sleep(self, duration: float) -> None:
        """Sleep that can be interrupted by shutdown event.

        Args:
            duration: Sleep duration in seconds
        """
        try:
            await asyncio.wait_for(
                self._shutdown_event.wait(),
                timeout=duration,
            )
        except asyncio.TimeoutError:
            pass  # Normal timeout, continue

    def stop(self) -> None:
        """Request graceful shutdown of the loop.

        This sets the shutdown event, which will interrupt the current
        sleep interval. The loop will complete its current _run_once()
        call (if in progress) and then exit.
        """
        if not self._running:
            return
        logger.info(f"[{self.name}] Stop requested")
        self._running = False
        self._shutdown_event.set()

    async def stop_async(self, timeout: float = 10.0) -> bool:
        """Request graceful shutdown and wait for loop to stop.

        Args:
            timeout: Maximum time to wait for shutdown in seconds

        Returns:
            True if loop stopped within timeout, False otherwise
        """
        self.stop()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
                return True
            except asyncio.TimeoutError:
                logger.warning(f"[{self.name}] Stop timed out after {timeout}s")
                return False
        return True

    def start_background(self) -> asyncio.Task:
        """Start the loop as a background task.

        Returns:
            The asyncio.Task running the loop
        """
        if self._task is not None and not self._task.done():
            logger.warning(f"[{self.name}] Background task already running")
            return self._task

        self._task = asyncio.create_task(
            self.run_forever(),
            name=f"loop_{self.name}",
        )
        return self._task

    def reset_stats(self) -> None:
        """Reset loop statistics to initial values."""
        self._stats = LoopStats(name=self.name)

    def get_status(self) -> dict[str, Any]:
        """Get current loop status for monitoring.

        Returns:
            Dictionary with loop status information
        """
        return {
            "name": self.name,
            "running": self._running,
            "enabled": self.enabled,
            "interval": self.interval,
            "stats": self._stats.to_dict(),
            "backoff": {
                "initial_delay": self.backoff_config.initial_delay,
                "max_delay": self.backoff_config.max_delay,
                "multiplier": self.backoff_config.multiplier,
            },
        }

    def health_check(self) -> "HealthCheckResult":
        """Check loop health for CoordinatorProtocol compliance.

        Dec 2025: Added to enable DaemonManager integration and observability.

        Returns:
            HealthCheckResult with status and loop metrics
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            # Fallback if protocols not available
            return {  # type: ignore[return-value]
                "healthy": self._running,
                "status": "running" if self._running else "stopped",
                "message": f"Loop {self.name} {'running' if self._running else 'stopped'}",
                "details": self._stats.to_dict(),
            }

        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message=f"Loop {self.name} is stopped",
            )

        # Check consecutive errors
        if self._stats.consecutive_errors > 5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Too many consecutive errors: {self._stats.consecutive_errors}",
                details={
                    "last_error": self._stats.last_error_message,
                    "consecutive_errors": self._stats.consecutive_errors,
                },
            )

        # Check success rate
        if self._stats.total_runs > 0 and self._stats.success_rate < 50:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"High failure rate: {self._stats.success_rate:.1f}%",
                details={
                    "total_runs": self._stats.total_runs,
                    "failed_runs": self._stats.failed_runs,
                    "success_rate": f"{self._stats.success_rate:.1f}%",
                },
            )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Loop {self.name} operational",
            details={
                "total_runs": self._stats.total_runs,
                "successful_runs": self._stats.successful_runs,
                "consecutive_errors": self._stats.consecutive_errors,
                "success_rate": f"{self._stats.success_rate:.1f}%",
                "last_success": self._stats.last_success_time,
                "avg_duration_ms": f"{self._stats.avg_run_duration * 1000:.1f}",
            },
        )

    # Abstract method that subclasses must implement

    @abstractmethod
    async def _run_once(self) -> None:
        """Execute one iteration of the loop.

        This method is called repeatedly by run_forever() with appropriate
        interval delays. Subclasses must implement this method with their
        specific loop logic.

        Raises:
            Any exception will be caught, logged, and trigger backoff.
        """
        raise NotImplementedError

    # Optional hooks that subclasses can override

    async def _on_start(self) -> None:
        """Called when the loop starts, before the first _run_once().

        Override this to perform initialization tasks.
        """
        pass

    async def _on_stop(self) -> None:
        """Called when the loop stops, after the last _run_once().

        Override this to perform cleanup tasks.
        """
        pass

    async def _on_error(self, error: Exception) -> None:
        """Called when _run_once() raises an exception.

        Override this to perform custom error handling.

        Args:
            error: The exception that was raised
        """
        pass


class LoopManager:
    """Manager for coordinating multiple background loops.

    Provides:
    - Centralized start/stop for all loops
    - Status aggregation
    - Graceful shutdown coordination
    """

    def __init__(self, name: str = "loop_manager"):
        """Initialize the loop manager.

        Args:
            name: Name for logging
        """
        self.name = name
        self._loops: dict[str, BaseLoop] = {}
        self._started = False

    def register(self, loop: BaseLoop) -> None:
        """Register a loop with the manager.

        Args:
            loop: Loop instance to manage
        """
        if loop.name in self._loops:
            logger.warning(f"[{self.name}] Loop '{loop.name}' already registered, replacing")
        self._loops[loop.name] = loop
        logger.debug(f"[{self.name}] Registered loop: {loop.name}")

    def unregister(self, name: str) -> BaseLoop | None:
        """Unregister a loop from the manager.

        Args:
            name: Name of the loop to unregister

        Returns:
            The unregistered loop, or None if not found
        """
        return self._loops.pop(name, None)

    def get(self, name: str) -> BaseLoop | None:
        """Get a loop by name.

        Args:
            name: Name of the loop

        Returns:
            The loop instance, or None if not found
        """
        return self._loops.get(name)

    async def start_all(
        self, verify_startup: bool = True, startup_timeout: float = 5.0
    ) -> dict[str, bool]:
        """Start all registered loops in dependency order as background tasks.

        Dec 2025: Modified to return startup success per loop and use dependency ordering.

        Args:
            verify_startup: If True, verify all loops started successfully
            startup_timeout: Seconds to wait for loops to start running

        Returns:
            Dictionary mapping loop names to whether they started successfully
        """
        results: dict[str, bool] = {}

        if self._started:
            logger.warning(f"[{self.name}] Already started")
            # Return current running status
            for name, loop in self._loops.items():
                results[name] = loop.running
            return results

        # Get loops in dependency order (topological sort)
        ordered_names = self._get_startup_order()
        logger.info(f"[{self.name}] Starting {len(self._loops)} loops in order: {ordered_names}")

        started_loops: set[str] = set()
        for loop_name in ordered_names:
            loop = self._loops.get(loop_name)
            if loop is None:
                continue

            # Wait for dependencies to be running
            for dep_name in loop.depends_on:
                if dep_name not in started_loops and dep_name in self._loops:
                    # Dependency hasn't started yet - wait briefly
                    dep_loop = self._loops[dep_name]
                    wait_start = time.time()
                    while not dep_loop.running and time.time() - wait_start < 2.0:
                        await asyncio.sleep(0.1)

            # Start this loop
            loop.start_background()
            await asyncio.sleep(0.1)  # Brief delay for startup

            # Check if it started
            if loop.running:
                started_loops.add(loop_name)
                results[loop_name] = True
            else:
                results[loop_name] = False

        self._started = True

        # Dec 2025: Verify loops actually started running
        if verify_startup and self._loops:
            await self._verify_loops_running(timeout=startup_timeout)
            # Update results with verified status
            for name, loop in self._loops.items():
                results[name] = loop.running or self._loops[name]._stats.total_runs > 0

        return results

    def _get_startup_order(self) -> list[str]:
        """Get loop names in topological order based on dependencies.

        Returns:
            List of loop names in order they should be started
        """
        # Simple topological sort using Kahn's algorithm
        in_degree: dict[str, int] = {name: 0 for name in self._loops}
        graph: dict[str, list[str]] = {name: [] for name in self._loops}

        for name, loop in self._loops.items():
            for dep in loop.depends_on:
                if dep in self._loops:
                    graph[dep].append(name)
                    in_degree[name] += 1

        # Start with nodes that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for dependent in graph[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Add any remaining loops (handles cycles by just appending them)
        for name in self._loops:
            if name not in result:
                result.append(name)

        return result

    async def _verify_loops_running(self, timeout: float = 5.0) -> None:
        """Verify all loops have started running.

        Dec 2025: Added to detect startup failures that would otherwise
        be silent. Logs warnings for loops that haven't started.

        Args:
            timeout: Maximum seconds to wait for loops to start
        """
        import asyncio
        start_time = time.time()
        failed_loops: list[str] = []

        while time.time() - start_time < timeout:
            failed_loops = []
            for name, loop in self._loops.items():
                status = loop.get_status()
                # Dec 2025: Fixed field name mismatch - use "running" not "is_running",
                # and check stats.total_runs for iteration count
                stats = status.get("stats", {})
                is_running = status.get("running", False) or stats.get("total_runs", 0) > 0
                if not is_running and not status.get("stopped", False):
                    failed_loops.append(name)

            if not failed_loops:
                logger.info(f"[{self.name}] All {len(self._loops)} loops verified running")
                return

            await asyncio.sleep(0.5)

        # Some loops didn't start
        if failed_loops:
            logger.warning(
                f"[{self.name}] {len(failed_loops)} loops not yet running after {timeout}s: "
                f"{', '.join(failed_loops)}"
            )

    async def stop_all(self, timeout: float = 30.0) -> dict[str, bool]:
        """Stop all registered loops gracefully.

        Args:
            timeout: Maximum time to wait for all loops to stop

        Returns:
            Dictionary mapping loop names to whether they stopped successfully
        """
        if not self._started:
            return {}

        logger.info(f"[{self.name}] Stopping {len(self._loops)} loops")

        # Request all loops to stop
        for loop in self._loops.values():
            loop.stop()

        # Wait for all to stop with timeout
        results: dict[str, bool] = {}
        per_loop_timeout = timeout / max(len(self._loops), 1)

        for name, loop in self._loops.items():
            success = await loop.stop_async(timeout=per_loop_timeout)
            results[name] = success

        self._started = False
        return results

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all registered loops.

        Returns:
            Dictionary mapping loop names to their status
        """
        return {name: loop.get_status() for name, loop in self._loops.items()}

    @property
    def loop_names(self) -> list[str]:
        """Get names of all registered loops."""
        return list(self._loops.keys())

    @property
    def is_started(self) -> bool:
        """Check if the manager has been started."""
        return self._started

    def health_check(self) -> dict[str, Any]:
        """Check health of all managed loops.

        Returns a standardized health check result compatible with daemon protocols.

        Returns:
            Dict with status, loop metrics, and error info
        """
        loops_running = sum(1 for loop in self._loops.values() if loop.running)
        total_loops = len(self._loops)

        # Aggregate loop stats
        total_runs = 0
        total_errors = 0
        failing_loops: list[str] = []

        for name, loop in self._loops.items():
            stats = loop.stats
            total_runs += stats.total_runs
            total_errors += stats.failed_runs
            # Dec 2025: Fixed bug - use (100 - success_rate) instead of non-existent failure_rate
            failure_rate = (100.0 - stats.success_rate) / 100.0 if stats.total_runs > 0 else 0.0
            if failure_rate > 0.5:  # >50% failure rate
                failing_loops.append(name)

        # Determine overall status
        if not self._started:
            status = "stopped"
        elif loops_running == 0 and total_loops > 0:
            status = "unhealthy"
        elif failing_loops:
            status = "degraded"
        elif loops_running < total_loops:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "started": self._started,
            "loops_running": loops_running,
            "total_loops": total_loops,
            "total_runs": total_runs,
            "total_errors": total_errors,
            "failing_loops": failing_loops,
            "loop_status": self.get_all_status(),
        }
