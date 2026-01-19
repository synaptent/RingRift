"""Run heavy loops in dedicated threads with their own event loops.

Jan 2026: Phase 3 of P2P multi-core parallelization.

This module provides ThreadedLoopRunner for running heavy P2P loops in isolated
threads with their own asyncio event loops. This prevents CPU-intensive operations
from blocking the main P2P event loop.

Key benefits:
- Heavy loops get dedicated CPU cores on high-core systems (GH200 has 64 cores)
- Main event loop remains responsive for health checks and heartbeats
- State sharing via thread-safe queues and copy-on-read patterns

Candidate loops for isolation:
- EloSyncLoop: Database merging, HTTP calls
- DataAggregationLoop: Disk scanning, file counting
- ModelSyncLoop: Large file transfers
- ManifestCollectionLoop: Directory traversal

Usage:
    from scripts.p2p.threaded_loop_runner import ThreadedLoopRunner

    # Create runner for a heavy loop
    runner = ThreadedLoopRunner(
        loop_factory=lambda: EloSyncLoop(orchestrator_ctx),
        name="elo_sync",
    )

    # Start in dedicated thread
    runner.start()

    # Send commands from main thread
    await runner.send_command({"action": "force_sync"})

    # Get results
    result = await runner.get_result(timeout=5.0)

    # Stop gracefully
    await runner.stop()
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

# Feature flag for rollback
THREADED_LOOPS_ENABLED = os.environ.get("RINGRIFT_THREADED_LOOPS_ENABLED", "true").lower() in ("true", "1", "yes")

T = TypeVar("T")


class RunnerState(Enum):
    """State of the threaded loop runner."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass
class RunnerStats:
    """Statistics for a threaded loop runner."""

    name: str
    state: RunnerState = RunnerState.STOPPED
    thread_id: int | None = None
    start_time: float | None = None
    stop_time: float | None = None
    commands_received: int = 0
    commands_processed: int = 0
    results_sent: int = 0
    errors: int = 0
    last_error: str = ""
    loop_iterations: int = 0

    @property
    def uptime(self) -> float:
        """Get uptime in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.stop_time or time.time()
        return end - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "state": self.state.value,
            "thread_id": self.thread_id,
            "uptime_s": round(self.uptime, 2),
            "commands_received": self.commands_received,
            "commands_processed": self.commands_processed,
            "results_sent": self.results_sent,
            "errors": self.errors,
            "last_error": self.last_error,
            "loop_iterations": self.loop_iterations,
        }


@dataclass
class Command:
    """Command to send to a threaded loop."""
    action: str
    data: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def __post_init__(self) -> None:
        if not self.request_id:
            import secrets
            self.request_id = secrets.token_hex(4)


@dataclass
class Result:
    """Result from a threaded loop."""
    request_id: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class ThreadedLoopRunner(Generic[T]):
    """Runs a BaseLoop in a dedicated thread with its own event loop.

    This class provides:
    - Isolated event loop in a dedicated thread
    - Thread-safe command/result queues for communication
    - Graceful startup and shutdown
    - Statistics and health monitoring

    The loop_factory is called in the dedicated thread to create the loop,
    ensuring all asyncio primitives are created in the correct event loop.
    """

    def __init__(
        self,
        loop_factory: Callable[[], T],
        name: str,
        *,
        command_queue_size: int = 100,
        result_queue_size: int = 100,
        executor_category: str | None = None,
        pin_to_cores: bool = True,
    ) -> None:
        """Initialize the threaded loop runner.

        Args:
            loop_factory: Factory function that creates the BaseLoop instance.
                         Called in the dedicated thread.
            name: Name for logging and metrics
            command_queue_size: Max pending commands (default 100)
            result_queue_size: Max pending results (default 100)
            executor_category: Pool category for CPU affinity (network, sync, etc.)
            pin_to_cores: Whether to pin thread to CPU cores (default True)
        """
        self.loop_factory = loop_factory
        self.name = name
        self.executor_category = executor_category or name
        self.pin_to_cores = pin_to_cores

        # Thread-safe queues for communication
        self._command_queue: queue.Queue[Command | None] = queue.Queue(maxsize=command_queue_size)
        self._result_queue: queue.Queue[Result] = queue.Queue(maxsize=result_queue_size)

        # Thread state
        self._thread: threading.Thread | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._loop_instance: T | None = None
        self._state_lock = threading.Lock()
        self._pinned_cores: list[int] | None = None

        # Statistics
        self._stats = RunnerStats(name=name)

        # Shutdown coordination
        self._stop_requested = threading.Event()

    @property
    def state(self) -> RunnerState:
        """Get current runner state."""
        return self._stats.state

    @property
    def stats(self) -> RunnerStats:
        """Get runner statistics."""
        return self._stats

    @property
    def is_running(self) -> bool:
        """Check if runner is currently running."""
        return self._stats.state == RunnerState.RUNNING

    def start(self) -> bool:
        """Start the loop in a dedicated thread.

        Returns:
            True if started successfully, False if already running or failed
        """
        if not THREADED_LOOPS_ENABLED:
            logger.info(f"[{self.name}] Threaded loops disabled via feature flag")
            return False

        with self._state_lock:
            if self._stats.state in (RunnerState.RUNNING, RunnerState.STARTING):
                logger.warning(f"[{self.name}] Already running or starting")
                return False

            self._stats.state = RunnerState.STARTING
            self._stop_requested.clear()

        # Start dedicated thread
        self._thread = threading.Thread(
            target=self._run_in_thread,
            name=f"p2p_loop_{self.name}",
            daemon=True,
        )
        self._thread.start()

        # Wait briefly for startup
        start_wait = time.time()
        while time.time() - start_wait < 5.0:
            if self._stats.state == RunnerState.RUNNING:
                logger.info(f"[{self.name}] Started in thread {self._thread.ident}")
                return True
            if self._stats.state == RunnerState.FAILED:
                logger.error(f"[{self.name}] Failed to start: {self._stats.last_error}")
                return False
            time.sleep(0.1)

        logger.error(f"[{self.name}] Startup timed out")
        self._stats.state = RunnerState.FAILED
        return False

    async def stop(self, timeout: float = 10.0) -> bool:
        """Stop the loop gracefully.

        Args:
            timeout: Maximum time to wait for shutdown

        Returns:
            True if stopped cleanly, False if timed out
        """
        if self._stats.state not in (RunnerState.RUNNING, RunnerState.STARTING):
            return True

        logger.info(f"[{self.name}] Stop requested")

        with self._state_lock:
            self._stats.state = RunnerState.STOPPING

        # Signal stop
        self._stop_requested.set()

        # Send None to unblock command queue
        try:
            self._command_queue.put_nowait(None)
        except queue.Full:
            pass

        # Wait for thread to finish
        if self._thread is not None:
            start_wait = time.time()
            while self._thread.is_alive() and (time.time() - start_wait) < timeout:
                await asyncio.sleep(0.1)

            if self._thread.is_alive():
                logger.warning(f"[{self.name}] Thread didn't stop within timeout")
                return False

        with self._state_lock:
            self._stats.state = RunnerState.STOPPED
            self._stats.stop_time = time.time()

        logger.info(f"[{self.name}] Stopped")
        return True

    async def send_command(self, command: Command | dict[str, Any], timeout: float = 5.0) -> bool:
        """Send a command to the loop.

        Args:
            command: Command to send (Command instance or dict with action/data)
            timeout: Maximum time to wait for queue space

        Returns:
            True if command was queued, False if queue full or not running
        """
        if not self.is_running:
            return False

        if isinstance(command, dict):
            command = Command(
                action=command.get("action", ""),
                data=command.get("data", {}),
            )

        try:
            # Use run_in_executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._command_queue.put(command, timeout=timeout),
            )
            self._stats.commands_received += 1
            return True
        except queue.Full:
            logger.warning(f"[{self.name}] Command queue full")
            return False

    async def get_result(self, timeout: float = 5.0) -> Result | None:
        """Get a result from the loop.

        Args:
            timeout: Maximum time to wait for result

        Returns:
            Result if available, None if timeout or not running
        """
        if not self.is_running and self._result_queue.empty():
            return None

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._result_queue.get(timeout=timeout),
            )
            return result
        except queue.Empty:
            return None

    def _run_in_thread(self) -> None:
        """Thread entry point - creates event loop and runs the loop."""
        try:
            # Set CPU affinity if enabled (Phase 4)
            if self.pin_to_cores:
                try:
                    from scripts.p2p.cpu_affinity import (
                        get_affinity_manager,
                        set_thread_affinity,
                    )
                    manager = get_affinity_manager()
                    if manager.is_enabled:
                        # Allocate 1 core for dedicated thread
                        cores = manager.allocate_cores(
                            self.name,
                            num_cores=1,
                            category=self.executor_category,
                        )
                        if set_thread_affinity(cores):
                            self._pinned_cores = cores
                            logger.debug(f"[{self.name}] Pinned to cores {cores}")
                except ImportError:
                    pass  # CPU affinity module not available
                except Exception as e:
                    logger.debug(f"[{self.name}] CPU affinity setup failed: {e}")

            # Create dedicated event loop for this thread
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)

            with self._state_lock:
                self._stats.thread_id = threading.get_ident()
                self._stats.start_time = time.time()

            # Run the async entry point
            self._event_loop.run_until_complete(self._async_main())

        except Exception as e:
            logger.exception(f"[{self.name}] Thread error: {e}")
            with self._state_lock:
                self._stats.state = RunnerState.FAILED
                self._stats.last_error = str(e)
                self._stats.errors += 1
        finally:
            if self._event_loop is not None:
                try:
                    self._event_loop.close()
                except Exception:
                    pass
                self._event_loop = None

    async def _async_main(self) -> None:
        """Async entry point in the dedicated thread."""
        try:
            # Create loop instance in this thread
            self._loop_instance = self.loop_factory()

            with self._state_lock:
                self._stats.state = RunnerState.RUNNING

            # Run loop and command processor concurrently
            await asyncio.gather(
                self._run_loop(),
                self._process_commands(),
            )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"[{self.name}] Async main error: {e}")
            with self._state_lock:
                self._stats.last_error = str(e)
                self._stats.errors += 1
                self._stats.state = RunnerState.FAILED

    async def _run_loop(self) -> None:
        """Run the actual BaseLoop."""
        if self._loop_instance is None:
            return

        try:
            # If the instance has a run_forever method, use it
            if hasattr(self._loop_instance, "run_forever"):
                await self._loop_instance.run_forever()
            else:
                # Otherwise, it might be a different kind of loop
                logger.warning(f"[{self.name}] Loop instance has no run_forever method")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"[{self.name}] Loop error: {e}")
            with self._state_lock:
                self._stats.last_error = str(e)
                self._stats.errors += 1

    async def _process_commands(self) -> None:
        """Process incoming commands from the main thread."""
        while not self._stop_requested.is_set():
            try:
                # Check for command with timeout
                loop = asyncio.get_running_loop()
                try:
                    command = await loop.run_in_executor(
                        None,
                        lambda: self._command_queue.get(timeout=1.0),
                    )
                except queue.Empty:
                    continue

                if command is None:
                    # Stop signal
                    break

                # Process command
                result = await self._handle_command(command)

                # Send result
                try:
                    self._result_queue.put_nowait(result)
                    self._stats.results_sent += 1
                except queue.Full:
                    logger.warning(f"[{self.name}] Result queue full, dropping result")

                self._stats.commands_processed += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"[{self.name}] Command processing error: {e}")
                self._stats.errors += 1

        # Stop the loop if it has a stop method
        if self._loop_instance is not None and hasattr(self._loop_instance, "stop"):
            self._loop_instance.stop()

    async def _handle_command(self, command: Command) -> Result:
        """Handle a single command.

        Override in subclasses for custom command handling.
        """
        try:
            action = command.action

            if action == "get_status":
                # Return loop status
                status = {}
                if self._loop_instance is not None and hasattr(self._loop_instance, "get_status"):
                    status = self._loop_instance.get_status()
                return Result(
                    request_id=command.request_id,
                    success=True,
                    data={"status": status},
                )

            elif action == "health_check":
                # Return health check
                health = {}
                if self._loop_instance is not None and hasattr(self._loop_instance, "health_check"):
                    health = self._loop_instance.health_check()
                return Result(
                    request_id=command.request_id,
                    success=True,
                    data={"health": health},
                )

            elif action == "force_run":
                # Trigger immediate loop iteration
                if self._loop_instance is not None and hasattr(self._loop_instance, "_run_once"):
                    await self._loop_instance._run_once()
                    self._stats.loop_iterations += 1
                return Result(
                    request_id=command.request_id,
                    success=True,
                    data={"ran": True},
                )

            else:
                # Unknown command
                return Result(
                    request_id=command.request_id,
                    success=False,
                    error=f"Unknown action: {action}",
                )

        except Exception as e:
            logger.exception(f"[{self.name}] Error handling command {command.action}: {e}")
            return Result(
                request_id=command.request_id,
                success=False,
                error=str(e),
            )

    def get_stats_dict(self) -> dict[str, Any]:
        """Get statistics as dictionary."""
        return self._stats.to_dict()


class ThreadedLoopRegistry:
    """Registry for managing multiple threaded loop runners.

    Provides centralized start/stop and status aggregation for all
    threaded loops in the P2P orchestrator.
    """

    _runners: dict[str, ThreadedLoopRunner] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, runner: ThreadedLoopRunner) -> None:
        """Register a runner."""
        with cls._lock:
            cls._runners[runner.name] = runner
            logger.debug(f"[ThreadedLoopRegistry] Registered: {runner.name}")

    @classmethod
    def unregister(cls, name: str) -> ThreadedLoopRunner | None:
        """Unregister a runner."""
        with cls._lock:
            return cls._runners.pop(name, None)

    @classmethod
    def get(cls, name: str) -> ThreadedLoopRunner | None:
        """Get a runner by name."""
        with cls._lock:
            return cls._runners.get(name)

    @classmethod
    def start_all(cls) -> dict[str, bool]:
        """Start all registered runners."""
        results = {}
        with cls._lock:
            runners = list(cls._runners.items())

        for name, runner in runners:
            results[name] = runner.start()

        return results

    @classmethod
    async def stop_all(cls, timeout: float = 30.0) -> dict[str, bool]:
        """Stop all registered runners."""
        results = {}
        with cls._lock:
            runners = list(cls._runners.items())

        per_runner_timeout = timeout / max(len(runners), 1)

        for name, runner in runners:
            results[name] = await runner.stop(timeout=per_runner_timeout)

        return results

    @classmethod
    def get_all_stats(cls) -> dict[str, dict[str, Any]]:
        """Get stats for all runners."""
        with cls._lock:
            return {name: runner.get_stats_dict() for name, runner in cls._runners.items()}

    @classmethod
    def get_summary(cls) -> dict[str, Any]:
        """Get summary of all runners."""
        with cls._lock:
            runners = list(cls._runners.values())

        running = sum(1 for r in runners if r.is_running)
        total_commands = sum(r.stats.commands_processed for r in runners)
        total_errors = sum(r.stats.errors for r in runners)

        return {
            "enabled": THREADED_LOOPS_ENABLED,
            "total_runners": len(runners),
            "running": running,
            "total_commands_processed": total_commands,
            "total_errors": total_errors,
            "runners": {r.name: r.state.value for r in runners},
        }
