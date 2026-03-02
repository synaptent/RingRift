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

from app.core.async_context import safe_create_task

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
    consecutive_timeouts: int = 0  # Feb 2026: Track consecutive timeouts for self-recovery
    total_timeouts: int = 0  # Feb 2026: Track total timeouts
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
            "consecutive_timeouts": self.consecutive_timeouts,
            "total_timeouts": self.total_timeouts,
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
    multiplier: float = 1.5  # Exponential multiplier (Jan 2026: reduced from 2.0 for faster recovery)
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
        executor_category: str | None = None,
        run_timeout: float | None = None,
    ):
        """Initialize the base loop.

        Args:
            name: Human-readable name for logging and metrics
            interval: Normal interval between runs in seconds
            backoff_config: Configuration for exponential backoff (uses defaults if None)
            metrics_manager: Optional MetricsManager for recording metrics
            enabled: Whether the loop is enabled (can be toggled at runtime)
            depends_on: List of loop names that must start before this loop (Dec 2025)
            executor_category: Thread pool category for heavy operations (Jan 2026)
                Categories: network, sync, jobs, health, compute
                If None, uses default asyncio.to_thread()
            run_timeout: Maximum time for _run_once() to complete (Feb 2026)
                If None, defaults to max(interval * 10, 300) seconds.
                Prevents zombie processes when operations hang indefinitely.
        """
        self.name = name
        self.interval = interval
        self.backoff_config = backoff_config or BackoffConfig()
        self.metrics_manager = metrics_manager
        self.enabled = enabled
        self.depends_on = depends_on or []
        self.executor_category = executor_category
        # Feb 2026: Timeout protection to prevent indefinite hangs
        # Default: 10x interval or 300s (5 min), whichever is larger
        self.run_timeout = run_timeout if run_timeout is not None else max(interval * 10, 300.0)

        # Lifecycle state
        self._running = False
        self._task: asyncio.Task | None = None
        # Jan 12, 2026: Lazy-init to avoid "no running event loop" error
        # asyncio.Event() requires an event loop, which may not exist at __init__ time
        self._shutdown_event: asyncio.Event | None = None

        # State synchronization lock (Jan 2026 - Phase 1.1 P2P Stability)
        # Protects _running flag and _stats from concurrent modification
        self._state_lock = asyncio.Lock()

        # Statistics
        # IMPORTANT: Subclasses must NOT shadow _stats with incompatible types.
        # Use a different attribute name for custom stats (e.g., _custom_stats).
        # Jan 24, 2026: Added validation in __setattr__ to catch shadowing issues.
        self._stats = LoopStats(name=name)

        # Performance degradation tracking (Jan 2026 - Phase 5.1 P2P Observability)
        self._performance_degraded_emitted = False

        # Callbacks
        self._error_callbacks: list[Callable[[Exception], None]] = []

    def __setattr__(self, name: str, value: Any) -> None:
        """Validate _stats is not shadowed with incompatible type.

        Jan 24, 2026: Added to prevent bugs like the GossipPeerPromotionLoop
        _stats shadowing issue (fixed in de0b9bf8e).

        Subclasses should use different attribute names for custom stats
        (e.g., _promotion_stats, _custom_stats) instead of overriding _stats.
        """
        if name == "_stats" and hasattr(self, "_stats"):
            # Check if the new value is compatible (has to_dict method)
            if not isinstance(value, LoopStats) and not hasattr(value, "to_dict"):
                import warnings

                warnings.warn(
                    f"{self.__class__.__name__}: Assigning incompatible type to _stats. "
                    f"Expected LoopStats or object with to_dict() method, got {type(value).__name__}. "
                    f"Use a different attribute name for custom stats (e.g., _custom_stats).",
                    stacklevel=2,
                )
        super().__setattr__(name, value)

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
        # Jan 12, 2026: Lazy-init shutdown event in async context
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()
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
                    # Feb 2026: Wrap _run_once() with timeout protection
                    # This prevents zombie processes when operations hang indefinitely
                    # Note: Using asyncio.wait_for() for Python 3.10 compatibility
                    # (asyncio.timeout was added in Python 3.11)
                    await asyncio.wait_for(self._run_once(), timeout=self.run_timeout)

                    # Success - update stats
                    run_duration = time.time() - run_start
                    self._stats.total_runs += 1
                    self._stats.successful_runs += 1
                    self._stats.consecutive_errors = 0
                    self._stats.consecutive_timeouts = 0  # Reset on success
                    self._stats.last_run_time = run_start
                    self._stats.last_success_time = time.time()
                    self._stats.last_run_duration = run_duration
                    self._stats.total_run_duration += run_duration

                    # Phase 5.1: Check for performance degradation
                    self._check_performance_degradation()

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

                except TimeoutError:
                    # Feb 2026: _run_once() timed out - track separately from errors
                    run_duration = time.time() - run_start
                    self._stats.total_runs += 1
                    self._stats.failed_runs += 1
                    self._stats.consecutive_timeouts += 1
                    self._stats.total_timeouts += 1
                    self._stats.last_run_time = run_start
                    self._stats.last_error_time = time.time()
                    self._stats.last_error_message = f"Timeout after {self.run_timeout}s"

                    logger.warning(
                        f"[{self.name}] _run_once() timed out after {self.run_timeout}s "
                        f"(consecutive: {self._stats.consecutive_timeouts})"
                    )

                    # Record timeout metric
                    if self.metrics_manager:
                        self.metrics_manager.record_metric(
                            f"loop_{self.name}_timeouts",
                            1,
                            metadata={"timeout_seconds": self.run_timeout},
                        )

                    # Emit timeout event for observability
                    self._emit_timeout_event()

                    # Check if we should trigger self-recovery
                    if self._stats.consecutive_timeouts >= 3:
                        logger.error(
                            f"[{self.name}] {self._stats.consecutive_timeouts} consecutive timeouts, "
                            "triggering self-recovery"
                        )
                        await self._on_consecutive_timeouts()

                    # Apply shorter backoff for timeouts (just the normal interval)
                    await self._interruptible_sleep(self.interval)

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
        # Jan 12, 2026: Guard for lazy-init event
        if self._shutdown_event is None:
            await asyncio.sleep(duration)
            return
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
        # Jan 12, 2026: Guard for lazy-init event
        if self._shutdown_event is not None:
            self._shutdown_event.set()

    async def stop_async(self, timeout: float = 10.0) -> bool:
        """Request graceful shutdown and wait for loop to stop.

        Jan 2026: Added state lock to prevent race conditions during shutdown.

        Args:
            timeout: Maximum time to wait for shutdown in seconds

        Returns:
            True if loop stopped within timeout, False otherwise
        """
        async with self._state_lock:
            self.stop()
            task = self._task

        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=timeout)
                return True
            except asyncio.TimeoutError:
                logger.warning(f"[{self.name}] Stop timed out after {timeout}s")
                # Phase 1.3: Explicitly cancel task after timeout to prevent leak
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                return False
        return True

    def start_background(self) -> asyncio.Task:
        """Start the loop as a background task.

        Note: For race-safe startup from async code, use start_background_async().

        Returns:
            The asyncio.Task running the loop
        """
        if self._task is not None and not self._task.done():
            logger.warning(f"[{self.name}] Background task already running")
            return self._task

        self._task = safe_create_task(
            self.run_forever(),
            name=f"loop_{self.name}",
        )
        return self._task

    async def start_background_async(self, timeout: float = 5.0) -> asyncio.Task | None:
        """Start the loop as a background task with state lock protection.

        Jan 2026: Added for Phase 1.1 P2P Stability - prevents race conditions
        when starting loops from async code.

        Args:
            timeout: Maximum time to wait for loop to start running (Phase 1.5)

        Returns:
            The asyncio.Task running the loop, or None if startup failed
        """
        async with self._state_lock:
            if self._task is not None and not self._task.done():
                logger.warning(f"[{self.name}] Background task already running")
                return self._task

            self._task = safe_create_task(
                self.run_forever(),
                name=f"loop_{self.name}",
            )

        # Jan 29, 2026: Yield to event loop immediately after task creation
        # to allow the task to start before we poll for _running. Without this,
        # there's a race condition where start_background_async returns None
        # because _running hasn't been set yet.
        await asyncio.sleep(0)

        # Phase 1.5: Wait for loop to actually start with timeout protection
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._running:
                return self._task
            await asyncio.sleep(0.1)

        # Timeout - loop didn't start
        logger.error(f"[{self.name}] Loop failed to start within {timeout}s")
        # Phase 1.4: Emit startup failure event
        self._emit_startup_failure_event()
        return None

    def _emit_startup_failure_event(self) -> None:
        """Emit event when loop fails to start (Phase 1.4)."""
        try:
            from app.coordination.event_router import emit_event
            from app.distributed.data_events import DataEventType

            emit_event(
                DataEventType.P2P_LOOP_STARTUP_FAILED,
                {
                    "loop_name": self.name,
                    "interval": self.interval,
                    "depends_on": self.depends_on,
                },
            )
        except ImportError:
            logger.debug(f"[{self.name}] Event system not available for startup failure")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to emit startup failure event: {e}")

    def _check_performance_degradation(self) -> None:
        """Check if loop performance is degraded and emit event if so.

        Jan 7, 2026 - Phase 5.1 P2P Observability: Emit LOOP_PERFORMANCE_DEGRADED
        if avg_run_duration > interval * 0.5 (loop taking >50% of its interval).
        """
        if self._stats.successful_runs < 5:
            # Need at least 5 successful runs to have meaningful average
            return

        threshold = self.interval * 0.5
        avg_duration = self._stats.avg_run_duration

        if avg_duration > threshold:
            if not self._performance_degraded_emitted:
                self._performance_degraded_emitted = True
                self._emit_performance_degraded_event(avg_duration, threshold)
        else:
            # Performance recovered
            if self._performance_degraded_emitted:
                self._performance_degraded_emitted = False
                logger.info(
                    f"[{self.name}] Performance recovered: "
                    f"avg={avg_duration:.2f}s < threshold={threshold:.2f}s"
                )

    def _emit_performance_degraded_event(
        self, avg_duration: float, threshold: float
    ) -> None:
        """Emit event when loop performance degrades (Phase 5.1)."""
        try:
            from app.coordination.event_router import emit_event
            from app.distributed.data_events import DataEventType

            logger.warning(
                f"[{self.name}] Performance degraded: "
                f"avg={avg_duration:.2f}s > threshold={threshold:.2f}s"
            )
            emit_event(
                DataEventType.P2P_LOOP_PERFORMANCE_DEGRADED,
                {
                    "loop_name": self.name,
                    "avg_run_duration": avg_duration,
                    "interval": self.interval,
                    "threshold": threshold,
                    "total_runs": self._stats.total_runs,
                    "success_rate": self._stats.success_rate,
                },
            )
        except ImportError:
            logger.debug(f"[{self.name}] Event system not available for perf degradation")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to emit perf degradation event: {e}")

    def reset_stats(self) -> None:
        """Reset loop statistics to initial values.

        Note: This is a sync method. For race-safe reset from async code,
        use reset_stats_async().
        """
        self._stats = LoopStats(name=self.name)

    async def reset_stats_async(self) -> None:
        """Reset loop statistics with state lock protection.

        Jan 2026: Added for Phase 1.1 P2P Stability.
        """
        async with self._state_lock:
            self._stats = LoopStats(name=self.name)

    def get_status(self) -> dict[str, Any]:
        """Get current loop status for monitoring.

        Returns:
            Dictionary with loop status information including computed status string
        """
        # Compute status string based on state
        if not self._running:
            status = "stopped"
        elif self._stats.consecutive_errors >= 5:
            status = "error"
        elif self._stats.total_runs > 0 and self._stats.success_rate < 0.5:
            status = "degraded"
        else:
            status = "running"

        return {
            "name": self.name,
            "status": status,  # Added: computed status string for monitoring
            "running": self._running,
            "enabled": self.enabled,
            "interval": self.interval,
            "stats": self._stats.to_dict() if hasattr(self._stats, 'to_dict') else self._stats,
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

    async def _on_consecutive_timeouts(self) -> None:
        """Called when _run_once() times out 3+ consecutive times.

        Feb 2026: Override this to implement self-recovery behavior.
        Default implementation logs a warning. Subclasses can override
        to restart connections, clear caches, or take other recovery actions.
        """
        pass

    def _emit_timeout_event(self) -> None:
        """Emit event when _run_once() times out (Feb 2026).

        Emits P2P_LOOP_TIMEOUT event for observability and alerting.
        """
        try:
            from app.coordination.event_router import emit_event
            from app.distributed.data_events import DataEventType

            emit_event(
                DataEventType.P2P_LOOP_TIMEOUT,
                {
                    "loop_name": self.name,
                    "timeout_seconds": self.run_timeout,
                    "consecutive_timeouts": self._stats.consecutive_timeouts,
                    "total_timeouts": self._stats.total_timeouts,
                    "interval": self.interval,
                },
            )
        except ImportError:
            logger.debug(f"[{self.name}] Event system not available for timeout event")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to emit timeout event: {e}")

    # Jan 2026: Phase 2 - Thread pool helper for heavy operations

    async def run_in_executor(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Run a blocking function in the appropriate thread pool.

        Jan 2026: Phase 2 multi-core parallelization. This method routes
        blocking operations to the loop's configured executor category,
        or falls back to asyncio.to_thread() if no category is set.

        Args:
            func: Blocking function to run
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Example:
            class MySyncLoop(BaseLoop):
                def __init__(self):
                    super().__init__(
                        name="my_sync_loop",
                        interval=60.0,
                        executor_category="sync",  # Use sync pool
                    )

                async def _run_once(self):
                    # Heavy blocking operation runs in sync pool
                    result = await self.run_in_executor(
                        self._heavy_sync_operation
                    )
        """
        if self.executor_category:
            try:
                from scripts.p2p.loop_executors import LoopExecutors

                return await LoopExecutors.run_in_pool(
                    self.executor_category, func, *args, **kwargs
                )
            except ImportError:
                # Fall back to asyncio.to_thread if executor module not available
                pass

        # Default: use asyncio.to_thread
        if kwargs:
            # asyncio.to_thread doesn't support kwargs directly
            import functools
            func = functools.partial(func, **kwargs)
        return await asyncio.to_thread(func, *args)


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

            # Start this loop with proper async startup verification
            # Jan 2026: Fixed race condition - use start_background_async() which
            # polls until _running=True, instead of start_background() + 0.1s sleep
            # which was insufficient for the asyncio task to actually start.
            per_loop_timeout = 2.0
            start_time = time.time()
            task = await loop.start_background_async(timeout=per_loop_timeout)
            elapsed = time.time() - start_time

            # Check if it started
            if task is not None and loop.running:
                started_loops.add(loop_name)
                results[loop_name] = True
                if elapsed > 0.5:
                    logger.info(f"[{self.name}] Loop '{loop_name}' took {elapsed:.2f}s to start")
            else:
                results[loop_name] = False
                logger.warning(
                    f"[{self.name}] Loop '{loop_name}' failed to start within {per_loop_timeout}s"
                )

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

        Jan 7, 2026 - Phase 5.3: Enhanced with cycle detection and logging.

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

        # Phase 5.3: Detect and report dependency cycles
        remaining = [name for name in self._loops if name not in result]
        if remaining:
            # We have loops with unresolved dependencies - likely a cycle
            cycles = self._detect_dependency_cycles(graph, remaining)
            if cycles:
                for cycle in cycles:
                    logger.warning(
                        f"[{self.name}] Dependency cycle detected: {' -> '.join(cycle)}"
                    )
            else:
                logger.warning(
                    f"[{self.name}] Loops with unresolved dependencies: {remaining}"
                )
            # Still add them to result so they can start (with warning)
            result.extend(remaining)

        return result

    def _detect_dependency_cycles(
        self, graph: dict[str, list[str]], candidates: list[str]
    ) -> list[list[str]]:
        """Detect cycles among candidate nodes using DFS.

        Jan 7, 2026 - Phase 5.3: Helper for cycle detection.

        Args:
            graph: Dependency graph (node -> dependents)
            candidates: Nodes to check for cycles

        Returns:
            List of detected cycles (each cycle is a list of node names)
        """
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node: str, path: list[str]) -> None:
            if node in rec_stack:
                # Found cycle - extract it from path
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Check dependencies (reverse direction - who depends on this node)
            loop = self._loops.get(node)
            if loop:
                for dep in loop.depends_on:
                    if dep in self._loops:
                        dfs(dep, path)

            path.pop()
            rec_stack.remove(node)

        for candidate in candidates:
            if candidate not in visited:
                dfs(candidate, [])

        return cycles

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

        Mar 2026: Changed from sequential to concurrent waiting. Previously,
        each loop got timeout/N seconds (e.g., 30/45 = 0.67s per loop), which
        was far too short for loops with in-flight _run_once() calls. Now all
        loops are stopped concurrently with the full timeout budget, so loops
        that are just sleeping wake up immediately via shutdown_event while
        loops with active work get the full timeout to finish.

        Args:
            timeout: Maximum time to wait for all loops to stop

        Returns:
            Dictionary mapping loop names to whether they stopped successfully
        """
        if not self._started:
            return {}

        logger.info(f"[{self.name}] Stopping {len(self._loops)} loops")

        # Request all loops to stop (sets _running=False and shutdown_event)
        for loop in self._loops.values():
            loop.stop()

        # Wait for all loops concurrently with the full timeout budget.
        # Most loops will stop almost immediately (they're sleeping between
        # iterations and the shutdown_event wakes them up). Only loops with
        # active _run_once() calls need real time to finish.
        results: dict[str, bool] = {}

        async def _wait_for_loop(name: str, loop: BaseLoop) -> tuple[str, bool]:
            success = await loop.stop_async(timeout=timeout)
            return name, success

        try:
            wait_results = await asyncio.wait_for(
                asyncio.gather(
                    *[_wait_for_loop(name, loop) for name, loop in self._loops.items()],
                    return_exceptions=True,
                ),
                timeout=timeout,
            )
            for result in wait_results:
                if isinstance(result, Exception):
                    continue
                name, success = result
                results[name] = success
        except asyncio.TimeoutError:
            logger.warning(
                f"[{self.name}] stop_all timed out after {timeout}s, "
                f"force-cancelling remaining loops"
            )
            # Mark any loops not in results as failed
            for name in self._loops:
                if name not in results:
                    results[name] = False

        stopped = sum(1 for ok in results.values() if ok)
        logger.info(f"[{self.name}] stopped {stopped}/{len(self._loops)} loops")

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

    def restart_loop(self, name: str) -> bool:
        """Restart a stopped loop.

        Jan 2026: Added to support 48h autonomous operation by allowing
        crashed/stopped loops to be restarted without full P2P restart.

        Args:
            name: Name of the loop to restart

        Returns:
            True if loop was restarted, False if not found or already running
        """
        loop = self._loops.get(name)
        if loop is None:
            logger.warning(f"[{self.name}] Cannot restart unknown loop: {name}")
            return False

        if loop.running:
            logger.debug(f"[{self.name}] Loop '{name}' is already running")
            return True

        if not loop.enabled:
            logger.info(f"[{self.name}] Not restarting disabled loop: {name}")
            return False

        logger.info(f"[{self.name}] Restarting stopped loop: {name}")
        loop.start_background()
        return True

    async def restart_stopped_loops(self) -> dict[str, bool]:
        """Restart all enabled loops that have stopped.

        Jan 2026: Added for 48h autonomous operation. This method should
        be called periodically to recover loops that crashed unexpectedly.

        Returns:
            Dictionary mapping restarted loop names to success status
        """
        results: dict[str, bool] = {}

        for name, loop in self._loops.items():
            if not loop.running and loop.enabled:
                logger.info(f"[{self.name}] Auto-restarting stopped loop: {name}")
                loop.start_background()
                # Brief delay to let loop initialize
                await asyncio.sleep(0.1)
                results[name] = loop.running
                if loop.running:
                    logger.info(f"[{self.name}] Successfully restarted loop: {name}")
                else:
                    logger.warning(f"[{self.name}] Failed to restart loop: {name}")

        return results
