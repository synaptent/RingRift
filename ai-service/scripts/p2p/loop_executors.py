"""Dedicated thread pools for P2P loop categories.

Jan 2026: Phase 2 of P2P multi-core parallelization.

This module provides category-based thread pool management for P2P background loops.
Instead of all loops sharing the default asyncio thread pool, heavy operations can
be offloaded to dedicated pools sized appropriately for their workload.

Pool Categories:
- network: Fast, latency-sensitive operations (HTTP calls, health checks)
- sync: Heavy I/O operations (file transfers, database sync)
- jobs: Job management operations (job dispatching, monitoring)
- health: Periodic health monitoring operations
- compute: CPU-intensive operations (data processing, checksums)

Usage:
    from scripts.p2p.loop_executors import LoopExecutors

    # Run blocking operation in appropriate pool
    result = await LoopExecutors.run_in_pool("sync", my_blocking_func, arg1, arg2)

    # Get pool stats
    stats = LoopExecutors.get_pool_stats()

    # Shutdown all pools (on P2P shutdown)
    LoopExecutors.shutdown_all()
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Get CPU count, with fallback
CPU_COUNT = os.cpu_count() or 4

# Environment variable to disable pool feature (for rollback)
LOOP_POOLS_ENABLED = os.environ.get("RINGRIFT_LOOP_POOLS_ENABLED", "true").lower() in ("true", "1", "yes")


@dataclass
class PoolConfig:
    """Configuration for a thread pool category."""

    name: str
    max_workers: int
    thread_name_prefix: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.thread_name_prefix:
            self.thread_name_prefix = f"p2p_{self.name}"


@dataclass
class PoolStats:
    """Statistics for a thread pool."""

    name: str
    max_workers: int
    active_threads: int = 0
    pending_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time: float = 0.0
    peak_active_threads: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def avg_execution_time(self) -> float:
        """Average execution time in seconds."""
        total = self.completed_tasks + self.failed_tasks
        if total == 0:
            return 0.0
        return self.total_execution_time / total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "max_workers": self.max_workers,
            "active_threads": self.active_threads,
            "pending_tasks": self.pending_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "total_execution_time_s": round(self.total_execution_time, 2),
            "avg_execution_time_ms": round(self.avg_execution_time * 1000, 2),
            "peak_active_threads": self.peak_active_threads,
            "uptime_s": round(time.time() - self.created_at, 2),
        }


# Default pool configurations
# Sized based on typical workloads and GH200 having 64+ cores
DEFAULT_POOL_CONFIGS: dict[str, PoolConfig] = {
    "network": PoolConfig(
        name="network",
        max_workers=min(8, CPU_COUNT),
        description="Fast, latency-sensitive operations (HTTP, health checks)",
    ),
    "sync": PoolConfig(
        name="sync",
        max_workers=min(16, CPU_COUNT // 2),
        description="Heavy I/O operations (file transfers, database sync)",
    ),
    "jobs": PoolConfig(
        name="jobs",
        max_workers=min(8, CPU_COUNT // 4),
        description="Job management operations (dispatching, monitoring)",
    ),
    "health": PoolConfig(
        name="health",
        max_workers=min(4, CPU_COUNT // 8 or 2),
        description="Periodic health monitoring operations",
    ),
    "compute": PoolConfig(
        name="compute",
        max_workers=min(8, CPU_COUNT // 4),
        description="CPU-intensive operations (data processing, checksums)",
    ),
}


class _PoolWrapper:
    """Wrapper around ThreadPoolExecutor with statistics tracking."""

    def __init__(self, config: PoolConfig) -> None:
        self.config = config
        self.stats = PoolStats(name=config.name, max_workers=config.max_workers)
        self._pool: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._pinned_cores: list[int] | None = None

    def _thread_initializer(self) -> None:
        """Initialize worker threads with CPU affinity (Phase 4)."""
        if self._pinned_cores is None:
            return
        try:
            from scripts.p2p.cpu_affinity import set_thread_affinity
            set_thread_affinity(self._pinned_cores)
        except ImportError:
            pass
        except Exception:
            pass  # Non-critical, ignore failures

    def _get_or_create_pool(self) -> ThreadPoolExecutor:
        """Get or lazily create the thread pool."""
        if self._pool is None:
            with self._lock:
                if self._pool is None:
                    # Try to allocate CPU cores for this pool (Phase 4)
                    initializer = None
                    try:
                        from scripts.p2p.cpu_affinity import get_affinity_manager
                        manager = get_affinity_manager()
                        if manager.is_enabled:
                            cores = manager.allocate_cores(
                                self.config.name,
                                num_cores=self.config.max_workers,
                                category=self.config.name,
                            )
                            self._pinned_cores = cores
                            initializer = self._thread_initializer
                            logger.debug(
                                f"[LoopExecutors] Pool '{self.config.name}' "
                                f"allocated cores {cores}"
                            )
                    except ImportError:
                        pass
                    except Exception as e:
                        logger.debug(f"[LoopExecutors] CPU affinity setup failed: {e}")

                    self._pool = ThreadPoolExecutor(
                        max_workers=self.config.max_workers,
                        thread_name_prefix=self.config.thread_name_prefix,
                        initializer=initializer,
                    )
                    logger.info(
                        f"[LoopExecutors] Created pool '{self.config.name}' "
                        f"with {self.config.max_workers} workers"
                    )
        return self._pool

    def submit(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        """Submit a task to the pool."""
        pool = self._get_or_create_pool()

        # Track pending task
        with self._lock:
            self.stats.pending_tasks += 1

        def _wrapped() -> T:
            start_time = time.time()
            try:
                with self._lock:
                    self.stats.pending_tasks -= 1
                    self.stats.active_threads += 1
                    if self.stats.active_threads > self.stats.peak_active_threads:
                        self.stats.peak_active_threads = self.stats.active_threads

                result = func(*args, **kwargs)

                with self._lock:
                    self.stats.active_threads -= 1
                    self.stats.completed_tasks += 1
                    self.stats.total_execution_time += time.time() - start_time

                return result
            except Exception as e:
                with self._lock:
                    self.stats.active_threads -= 1
                    self.stats.failed_tasks += 1
                    self.stats.total_execution_time += time.time() - start_time
                raise e

        return pool.submit(_wrapped)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the pool."""
        if self._pool is not None:
            with self._lock:
                if self._pool is not None:
                    self._pool.shutdown(wait=wait)
                    self._pool = None
                    logger.info(f"[LoopExecutors] Shutdown pool '{self.config.name}'")


class LoopExecutorRegistry:
    """Registry for category-based thread pools.

    This class manages dedicated thread pools for different categories of
    P2P loop operations, allowing heavy operations to be offloaded without
    blocking the main event loop.

    Thread-safe: All operations are protected by locks.
    """

    _pools: dict[str, _PoolWrapper] = {}
    _lock = threading.Lock()
    _shutdown = False

    @classmethod
    def get_pool(cls, category: str) -> _PoolWrapper:
        """Get or create a pool for the given category.

        Args:
            category: Pool category name (network, sync, jobs, health, compute)

        Returns:
            Pool wrapper for the category
        """
        if cls._shutdown:
            raise RuntimeError("LoopExecutorRegistry has been shutdown")

        with cls._lock:
            if category not in cls._pools:
                config = DEFAULT_POOL_CONFIGS.get(category)
                if config is None:
                    # Unknown category - create with default settings
                    logger.warning(
                        f"[LoopExecutors] Unknown category '{category}', using defaults"
                    )
                    config = PoolConfig(
                        name=category,
                        max_workers=min(4, CPU_COUNT // 4),
                    )
                cls._pools[category] = _PoolWrapper(config)

            return cls._pools[category]

    @classmethod
    async def run_in_pool(
        cls, category: str, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        """Run a blocking function in the appropriate thread pool.

        This is the primary API for offloading blocking operations to
        dedicated thread pools.

        Args:
            category: Pool category (network, sync, jobs, health, compute)
            func: Blocking function to run
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            Any exception raised by func
        """
        if not LOOP_POOLS_ENABLED:
            # Feature disabled - fall back to default asyncio.to_thread()
            return await asyncio.to_thread(func, *args, **kwargs)

        pool = cls.get_pool(category)
        loop = asyncio.get_running_loop()
        future = pool.submit(func, *args, **kwargs)
        return await loop.run_in_executor(None, future.result)

    @classmethod
    def get_pool_stats(cls) -> dict[str, dict[str, Any]]:
        """Get statistics for all pools.

        Returns:
            Dictionary mapping category names to their stats
        """
        with cls._lock:
            return {name: wrapper.stats.to_dict() for name, wrapper in cls._pools.items()}

    @classmethod
    def get_all_stats_summary(cls) -> dict[str, Any]:
        """Get aggregated summary of all pools.

        Returns:
            Summary statistics across all pools
        """
        pool_stats = cls.get_pool_stats()

        total_completed = sum(s.get("completed_tasks", 0) for s in pool_stats.values())
        total_failed = sum(s.get("failed_tasks", 0) for s in pool_stats.values())
        total_pending = sum(s.get("pending_tasks", 0) for s in pool_stats.values())
        total_active = sum(s.get("active_threads", 0) for s in pool_stats.values())

        return {
            "enabled": LOOP_POOLS_ENABLED,
            "cpu_count": CPU_COUNT,
            "pool_count": len(pool_stats),
            "total_completed_tasks": total_completed,
            "total_failed_tasks": total_failed,
            "total_pending_tasks": total_pending,
            "total_active_threads": total_active,
            "pools": pool_stats,
        }

    @classmethod
    def shutdown_all(cls, wait: bool = True) -> None:
        """Shutdown all pools.

        Should be called during P2P orchestrator shutdown.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        with cls._lock:
            cls._shutdown = True
            for name, wrapper in cls._pools.items():
                try:
                    wrapper.shutdown(wait=wait)
                except Exception as e:
                    logger.error(f"[LoopExecutors] Error shutting down pool '{name}': {e}")
            cls._pools.clear()
            logger.info("[LoopExecutors] All pools shutdown")

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (for testing).

        Shuts down all pools and clears the registry.
        """
        cls.shutdown_all(wait=False)
        with cls._lock:
            cls._shutdown = False


# Convenience alias
LoopExecutors = LoopExecutorRegistry
