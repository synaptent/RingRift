"""CPU affinity management for P2P loop parallelization.

Jan 2026: Phase 4 of P2P multi-core parallelization.

This module provides CPU core pinning for thread pools and dedicated threads,
improving cache locality on high-core systems like the GH200 (64 cores).

Key benefits:
- Better L1/L2 cache utilization when threads stay on assigned cores
- Reduced context switching overhead
- More predictable latency for time-critical operations

Usage:
    from scripts.p2p.cpu_affinity import (
        CPUAffinityManager,
        set_thread_affinity,
        get_recommended_cores,
    )

    # Get cores for a pool category
    cores = get_recommended_cores("sync", pool_size=8)

    # Set affinity for current thread
    set_thread_affinity(cores)

    # Or use the manager for automatic assignment
    manager = CPUAffinityManager()
    cores = manager.allocate_cores("sync", num_cores=8)
"""

from __future__ import annotations

import logging
import os
import platform
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Feature flag for rollback
CPU_AFFINITY_ENABLED = os.environ.get(
    "RINGRIFT_CPU_AFFINITY_ENABLED", "true"
).lower() in ("true", "1", "yes")

# Minimum cores before enabling affinity (don't bother on small machines)
MIN_CORES_FOR_AFFINITY = int(os.environ.get("RINGRIFT_MIN_CORES_FOR_AFFINITY", "8"))

# Get system info
CPU_COUNT = os.cpu_count() or 1
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"


@dataclass
class CoreAllocation:
    """Tracks a core allocation for a pool or thread."""

    name: str
    cores: list[int]
    thread_ids: list[int] = field(default_factory=list)
    allocated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "cores": self.cores,
            "thread_ids": self.thread_ids,
            "num_cores": len(self.cores),
        }


# Core allocation strategy for different pool categories
# Higher priority categories get lower-numbered cores (typically faster on NUMA)
POOL_CORE_PRIORITIES: dict[str, int] = {
    "health": 0,    # Highest priority - critical health checks
    "network": 1,   # Fast, latency-sensitive operations
    "jobs": 2,      # Job management
    "sync": 3,      # Heavy I/O, can tolerate some latency
    "compute": 4,   # CPU-intensive, bulk work
}


def _get_available_cores() -> list[int]:
    """Get list of available CPU cores.

    On Linux, respects cgroup CPU restrictions.
    On macOS, returns all cores (no affinity support, but useful for planning).
    """
    if IS_LINUX:
        try:
            # Try to get cores from cgroup (container-aware)
            with open("/sys/fs/cgroup/cpuset/cpuset.cpus", "r") as f:
                cpuset = f.read().strip()
                return _parse_cpuset(cpuset)
        except FileNotFoundError:
            pass

        try:
            # cgroup v2 path
            with open("/sys/fs/cgroup/cpuset.cpus.effective", "r") as f:
                cpuset = f.read().strip()
                return _parse_cpuset(cpuset)
        except FileNotFoundError:
            pass

    # Fallback: all cores
    return list(range(CPU_COUNT))


def _parse_cpuset(cpuset: str) -> list[int]:
    """Parse a cpuset string like '0-3,8-11' into a list of CPU numbers."""
    cores = []
    for part in cpuset.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cores.extend(range(int(start), int(end) + 1))
        else:
            cores.append(int(part))
    return sorted(cores)


def set_thread_affinity(cores: list[int]) -> bool:
    """Set CPU affinity for the current thread.

    Args:
        cores: List of CPU core numbers to pin to

    Returns:
        True if affinity was set, False if not supported or disabled
    """
    if not CPU_AFFINITY_ENABLED:
        return False

    if not cores:
        return False

    if CPU_COUNT < MIN_CORES_FOR_AFFINITY:
        logger.debug(
            f"CPU affinity disabled: only {CPU_COUNT} cores "
            f"(minimum {MIN_CORES_FOR_AFFINITY})"
        )
        return False

    if IS_LINUX:
        try:
            os.sched_setaffinity(0, cores)
            logger.debug(f"Set thread affinity to cores {cores}")
            return True
        except (OSError, AttributeError) as e:
            logger.debug(f"Failed to set thread affinity: {e}")
            return False

    # macOS doesn't support thread affinity via standard APIs
    # (would need mach thread_policy_set which is complex)
    if IS_MACOS:
        logger.debug("CPU affinity not supported on macOS")
        return False

    return False


def get_thread_affinity() -> list[int] | None:
    """Get CPU affinity for the current thread.

    Returns:
        List of CPU cores, or None if not supported
    """
    if IS_LINUX:
        try:
            return list(os.sched_getaffinity(0))
        except (OSError, AttributeError):
            return None
    return None


def get_recommended_cores(
    category: str,
    pool_size: int,
    *,
    avoid_core_zero: bool = True,
) -> list[int]:
    """Get recommended CPU cores for a pool category.

    Args:
        category: Pool category (network, sync, jobs, health, compute)
        pool_size: Number of cores needed
        avoid_core_zero: Whether to avoid core 0 (often used by kernel)

    Returns:
        List of recommended CPU core numbers
    """
    available = _get_available_cores()

    if avoid_core_zero and 0 in available and len(available) > 1:
        available = [c for c in available if c != 0]

    if pool_size >= len(available):
        return available

    # Get priority (lower = higher priority = lower core numbers)
    priority = POOL_CORE_PRIORITIES.get(category, 5)

    # Calculate starting offset based on priority
    # Higher priority categories get lower-numbered cores
    total_pools = len(POOL_CORE_PRIORITIES)
    cores_per_priority = max(1, len(available) // (total_pools + 1))
    start_offset = priority * cores_per_priority

    # Wrap around if necessary
    start_offset = start_offset % len(available)

    # Select cores starting from the offset
    cores = []
    for i in range(pool_size):
        idx = (start_offset + i) % len(available)
        cores.append(available[idx])

    return sorted(cores)


class CPUAffinityManager:
    """Manages CPU core allocation across multiple pools and threads.

    This class provides centralized tracking of core allocations to avoid
    conflicts and ensure optimal distribution across pool categories.

    Thread-safe: All operations are protected by a lock.
    """

    _instance: CPUAffinityManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> CPUAffinityManager:
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the manager."""
        if getattr(self, "_initialized", False):
            return

        self._allocations: dict[str, CoreAllocation] = {}
        self._allocation_lock = threading.Lock()
        self._available_cores = _get_available_cores()
        self._initialized = True

        logger.info(
            f"[CPUAffinityManager] Initialized with {len(self._available_cores)} cores "
            f"(affinity enabled: {CPU_AFFINITY_ENABLED})"
        )

    @property
    def is_enabled(self) -> bool:
        """Check if CPU affinity is enabled and useful."""
        return (
            CPU_AFFINITY_ENABLED
            and CPU_COUNT >= MIN_CORES_FOR_AFFINITY
            and IS_LINUX  # Only Linux supports thread affinity
        )

    @property
    def available_cores(self) -> list[int]:
        """Get list of available CPU cores."""
        return self._available_cores.copy()

    def allocate_cores(
        self,
        name: str,
        num_cores: int,
        *,
        category: str | None = None,
    ) -> list[int]:
        """Allocate CPU cores for a pool or thread group.

        Args:
            name: Unique name for this allocation
            num_cores: Number of cores to allocate
            category: Optional category for priority-based allocation

        Returns:
            List of allocated CPU core numbers
        """
        import time

        with self._allocation_lock:
            if name in self._allocations:
                # Already allocated, return existing
                return self._allocations[name].cores.copy()

            # Determine which cores to use
            effective_category = category or name
            cores = get_recommended_cores(
                effective_category,
                num_cores,
                avoid_core_zero=True,
            )

            # Record allocation
            self._allocations[name] = CoreAllocation(
                name=name,
                cores=cores,
                allocated_at=time.time(),
            )

            logger.debug(f"[CPUAffinityManager] Allocated cores {cores} for '{name}'")
            return cores.copy()

    def register_thread(self, allocation_name: str, thread_id: int | None = None) -> bool:
        """Register a thread with an allocation and optionally set affinity.

        Args:
            allocation_name: Name of the allocation to register with
            thread_id: Thread ID (defaults to current thread)

        Returns:
            True if registered and affinity set, False otherwise
        """
        if thread_id is None:
            thread_id = threading.get_ident()

        with self._allocation_lock:
            allocation = self._allocations.get(allocation_name)
            if allocation is None:
                logger.warning(f"[CPUAffinityManager] Unknown allocation: {allocation_name}")
                return False

            if thread_id not in allocation.thread_ids:
                allocation.thread_ids.append(thread_id)

            # Set affinity if enabled
            return set_thread_affinity(allocation.cores)

    def release_allocation(self, name: str) -> bool:
        """Release a core allocation.

        Args:
            name: Name of the allocation to release

        Returns:
            True if released, False if not found
        """
        with self._allocation_lock:
            if name in self._allocations:
                del self._allocations[name]
                logger.debug(f"[CPUAffinityManager] Released allocation '{name}'")
                return True
            return False

    def get_allocation(self, name: str) -> CoreAllocation | None:
        """Get an allocation by name."""
        with self._allocation_lock:
            allocation = self._allocations.get(name)
            return allocation

    def get_all_allocations(self) -> dict[str, dict[str, Any]]:
        """Get all allocations as dictionaries."""
        with self._allocation_lock:
            return {name: alloc.to_dict() for name, alloc in self._allocations.items()}

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        with self._allocation_lock:
            total_allocated = sum(
                len(alloc.cores) for alloc in self._allocations.values()
            )
            total_threads = sum(
                len(alloc.thread_ids) for alloc in self._allocations.values()
            )

            return {
                "enabled": self.is_enabled,
                "cpu_count": CPU_COUNT,
                "available_cores": len(self._available_cores),
                "total_allocated": total_allocated,
                "total_threads": total_threads,
                "allocations": len(self._allocations),
                "is_linux": IS_LINUX,
            }

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None


def get_affinity_manager() -> CPUAffinityManager:
    """Get the CPU affinity manager singleton."""
    return CPUAffinityManager()
