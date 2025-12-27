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
    ):
        """Initialize the base loop.

        Args:
            name: Human-readable name for logging and metrics
            interval: Normal interval between runs in seconds
            backoff_config: Configuration for exponential backoff (uses defaults if None)
            metrics_manager: Optional MetricsManager for recording metrics
            enabled: Whether the loop is enabled (can be toggled at runtime)
        """
        self.name = name
        self.interval = interval
        self.backoff_config = backoff_config or BackoffConfig()
        self.metrics_manager = metrics_manager
        self.enabled = enabled

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

    async def start_all(self, verify_startup: bool = True, startup_timeout: float = 5.0) -> None:
        """Start all registered loops as background tasks.

        Args:
            verify_startup: If True, verify all loops started successfully
            startup_timeout: Seconds to wait for loops to start running

        Raises:
            RuntimeError: If verify_startup is True and any loop fails to start
        """
        if self._started:
            logger.warning(f"[{self.name}] Already started")
            return

        logger.info(f"[{self.name}] Starting {len(self._loops)} loops")
        for loop in self._loops.values():
            loop.start_background()
        self._started = True

        # Dec 2025: Verify loops actually started running
        if verify_startup and self._loops:
            await self._verify_loops_running(timeout=startup_timeout)

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
