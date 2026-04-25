"""Async Raft Manager for P2P Orchestrator.

January 24, 2026: Created as part of the Raft re-enablement plan (Phase 5).

Problem:
PySyncObj's autoTick=True creates busy-wait polling threads that consume
100% CPU. Each Raft instance (ReplicatedWorkQueue, ReplicatedEloStore,
ReplicatedJobAssignments) runs its own tick loop, causing 3x the CPU usage.

Solution:
AsyncRaftManager provides asyncio-compatible tick control:
1. Disables autoTick in PySyncObj instances
2. Manually ticks all instances in a single async loop
3. Uses asyncio.to_thread() to avoid blocking the event loop
4. Configurable tick interval (default 50ms for ~20 ticks/second)

Usage:
    from app.p2p.async_raft_wrapper import AsyncRaftManager

    manager = AsyncRaftManager(tick_interval=0.05)

    # Register Raft instances (must be created with autoTick=False)
    manager.register(work_queue)
    manager.register(elo_store)

    await manager.start()

    # ... later ...
    await manager.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    try:
        from pysyncobj import SyncObj
    except ImportError:
        SyncObj = Any

logger = logging.getLogger(__name__)


@dataclass
class RaftTickStats:
    """Statistics for Raft tick loop."""

    total_ticks: int = 0
    successful_ticks: int = 0
    failed_ticks: int = 0
    total_tick_duration_ms: float = 0.0
    last_tick_time: float = 0.0
    last_tick_duration_ms: float = 0.0
    instances_registered: int = 0

    @property
    def avg_tick_duration_ms(self) -> float:
        """Average tick duration in milliseconds."""
        if self.successful_ticks == 0:
            return 0.0
        return self.total_tick_duration_ms / self.successful_ticks

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_ticks": self.total_ticks,
            "successful_ticks": self.successful_ticks,
            "failed_ticks": self.failed_ticks,
            "avg_tick_duration_ms": self.avg_tick_duration_ms,
            "last_tick_time": self.last_tick_time,
            "last_tick_duration_ms": self.last_tick_duration_ms,
            "instances_registered": self.instances_registered,
        }


@dataclass
class AsyncRaftConfig:
    """Configuration for AsyncRaftManager."""

    # Tick interval in seconds (50ms = 20 ticks/second)
    # Lower values = faster consensus but higher CPU
    # Higher values = slower consensus but lower CPU
    tick_interval: float = 0.05

    # Whether to run ticks in parallel for multiple instances
    # True: tick all instances concurrently (faster, more CPU)
    # False: tick instances sequentially (slower, less CPU)
    parallel_ticks: bool = False

    # Timeout for each tick operation in seconds
    tick_timeout: float = 1.0

    # Maximum consecutive tick failures before logging warning
    max_consecutive_failures: int = 5

    # Whether to continue on tick errors (True) or stop the loop (False)
    continue_on_error: bool = True


class AsyncRaftManager:
    """Manages Raft instances with asyncio-compatible tick control.

    Key features:
    - Single async tick loop for all Raft instances
    - Runs doTick() in thread pool to avoid blocking event loop
    - Configurable tick interval and error handling
    - Statistics tracking for observability

    Important: Raft instances must be created with autoTick=False
    for this manager to work correctly.
    """

    def __init__(self, config: AsyncRaftConfig | None = None):
        """Initialize AsyncRaftManager.

        Args:
            config: Configuration for tick behavior. Uses defaults if None.
        """
        self.config = config or AsyncRaftConfig()

        # Registered Raft instances
        self._instances: list[Any] = []  # SyncObj instances
        self._instance_names: dict[int, str] = {}  # id(instance) -> name

        # Lifecycle state
        self._running = False
        self._tick_task: asyncio.Task | None = None

        # Statistics
        self._stats = RaftTickStats()
        self._consecutive_failures = 0

        logger.info(
            f"AsyncRaftManager initialized with tick_interval={self.config.tick_interval}s, "
            f"parallel_ticks={self.config.parallel_ticks}"
        )

    def register(self, instance: Any, name: str = "unknown") -> None:
        """Register a Raft instance for managed ticking.

        Args:
            instance: PySyncObj SyncObj instance (must have autoTick=False)
            name: Human-readable name for logging
        """
        self._instances.append(instance)
        self._instance_names[id(instance)] = name
        self._stats.instances_registered = len(self._instances)
        logger.info(f"Registered Raft instance: {name}")

    def unregister(self, instance: Any) -> None:
        """Unregister a Raft instance.

        Args:
            instance: Previously registered SyncObj instance
        """
        if instance in self._instances:
            self._instances.remove(instance)
            name = self._instance_names.pop(id(instance), "unknown")
            self._stats.instances_registered = len(self._instances)
            logger.info(f"Unregistered Raft instance: {name}")

    async def start(self) -> None:
        """Start the async tick loop."""
        if self._running:
            logger.warning("AsyncRaftManager already running")
            return

        if not self._instances:
            logger.warning("No Raft instances registered, tick loop not started")
            return

        self._running = True
        self._tick_task = asyncio.create_task(
            self._tick_loop(),
            name="raft_tick_loop"
        )
        logger.info(
            f"AsyncRaftManager started with {len(self._instances)} instances"
        )

    async def stop(self) -> None:
        """Stop the tick loop gracefully."""
        if not self._running:
            return

        self._running = False

        if self._tick_task:
            self._tick_task.cancel()
            try:
                await asyncio.wait_for(self._tick_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._tick_task = None

        logger.info(
            f"AsyncRaftManager stopped. Stats: {self._stats.total_ticks} ticks, "
            f"{self._stats.avg_tick_duration_ms:.2f}ms avg"
        )

    async def _tick_loop(self) -> None:
        """Manually tick all Raft instances at controlled interval."""
        logger.debug("Raft tick loop started")

        while self._running:
            tick_start = time.time()
            self._stats.total_ticks += 1
            self._stats.last_tick_time = tick_start

            try:
                if self.config.parallel_ticks:
                    await self._tick_all_parallel()
                else:
                    await self._tick_all_sequential()

                # Success
                tick_duration_ms = (time.time() - tick_start) * 1000
                self._stats.successful_ticks += 1
                self._stats.total_tick_duration_ms += tick_duration_ms
                self._stats.last_tick_duration_ms = tick_duration_ms
                self._consecutive_failures = 0

            except asyncio.CancelledError:
                # Normal shutdown
                raise

            except Exception as e:
                self._stats.failed_ticks += 1
                self._consecutive_failures += 1

                if self._consecutive_failures >= self.config.max_consecutive_failures:
                    logger.warning(
                        f"Raft tick failed {self._consecutive_failures} consecutive times: {e}"
                    )

                if not self.config.continue_on_error:
                    logger.error(f"Raft tick loop stopping due to error: {e}")
                    break

            # Sleep until next tick
            elapsed = time.time() - tick_start
            sleep_time = max(0, self.config.tick_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        logger.debug("Raft tick loop exited")

    async def _tick_all_sequential(self) -> None:
        """Tick all instances sequentially."""
        for instance in self._instances:
            await self._tick_instance(instance)

    async def _tick_all_parallel(self) -> None:
        """Tick all instances in parallel."""
        tasks = [
            self._tick_instance(instance)
            for instance in self._instances
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _tick_instance(self, instance: Any) -> None:
        """Tick a single Raft instance in thread pool.

        Args:
            instance: SyncObj instance to tick
        """
        try:
            # doTick(0.0) does one tick iteration without internal sleeping
            # We run it in a thread to avoid blocking the event loop
            await asyncio.wait_for(
                asyncio.to_thread(instance.doTick, 0.0),
                timeout=self.config.tick_timeout,
            )
        except asyncio.TimeoutError:
            name = self._instance_names.get(id(instance), "unknown")
            logger.debug(f"Raft tick timeout for {name}")
            raise
        except Exception as e:
            name = self._instance_names.get(id(instance), "unknown")
            logger.debug(f"Raft tick error for {name}: {e}")
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get tick loop statistics."""
        return {
            **self._stats.to_dict(),
            "running": self._running,
            "config": {
                "tick_interval": self.config.tick_interval,
                "parallel_ticks": self.config.parallel_ticks,
                "tick_timeout": self.config.tick_timeout,
            },
            "instances": [
                self._instance_names.get(id(inst), "unknown")
                for inst in self._instances
            ],
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for monitoring integration."""
        healthy = self._running and self._consecutive_failures < self.config.max_consecutive_failures

        if not self._running:
            status = "stopped"
            message = "Tick loop not running"
        elif self._consecutive_failures >= self.config.max_consecutive_failures:
            status = "degraded"
            message = f"High failure rate: {self._consecutive_failures} consecutive failures"
        else:
            status = "healthy"
            message = f"Running with {len(self._instances)} instances"

        return {
            "healthy": healthy,
            "status": status,
            "message": message,
            "details": {
                "running": self._running,
                "instances": len(self._instances),
                "total_ticks": self._stats.total_ticks,
                "failed_ticks": self._stats.failed_ticks,
                "consecutive_failures": self._consecutive_failures,
                "avg_tick_duration_ms": self._stats.avg_tick_duration_ms,
            },
        }


# Singleton instance for global access
_manager_instance: AsyncRaftManager | None = None


def get_async_raft_manager() -> AsyncRaftManager:
    """Get or create the global AsyncRaftManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = AsyncRaftManager()
    return _manager_instance


def reset_async_raft_manager() -> None:
    """Reset the global AsyncRaftManager instance (for testing)."""
    global _manager_instance
    _manager_instance = None
