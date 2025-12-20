"""Unified Locking Infrastructure for RingRift AI Service.

Provides a consolidated locking framework with:
- Unified protocol for all lock types (sync/async, local/distributed)
- Lock ordering for deadlock prevention
- Lock hierarchy enforcement
- Lock pool for efficient resource management
- Comprehensive diagnostics

Usage:
    from app.core.locking import (
        acquire_locks,
        LockFactory,
        OrderedLockGroup,
    )

    # Simple locking
    async with acquire_locks("resource_a", "resource_b"):
        # Both locks held in consistent order
        pass

    # Lock hierarchy
    hierarchy = LockHierarchy()
    hierarchy.define_level("database", 1)
    hierarchy.define_level("cache", 2)
    hierarchy.define_level("api", 3)

    # Prevents acquiring database lock while holding api lock
    async with hierarchy.acquire("cache"):
        async with hierarchy.acquire("database"):  # OK
            pass

See also:
    - app.coordination.distributed_lock: Cross-process Redis/file locks
    - app.coordination.lock_manager: In-process async locks with deadlock detection
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AsyncLockProtocol",
    "LockDiagnostics",
    # Exceptions
    "LockError",
    # Classes
    "LockFactory",
    "LockHierarchy",
    "LockHierarchyViolation",
    "LockOrderViolation",
    "LockPool",
    "LockProtocol",
    "OrderedLockGroup",
    # Protocols
    "SyncLockProtocol",
    # Functions
    "acquire_locks",
    "acquire_locks_sync",
]

T = TypeVar("T")


# =============================================================================
# Exceptions
# =============================================================================

class LockError(Exception):
    """Base exception for lock errors."""


class LockOrderViolation(LockError):
    """Raised when locks are acquired in incorrect order."""

    def __init__(self, expected_order: list[str], actual: str):
        self.expected_order = expected_order
        self.actual = actual
        super().__init__(
            f"Lock order violation: acquired '{actual}' but expected order is {expected_order}"
        )


class LockHierarchyViolation(LockError):
    """Raised when lock hierarchy is violated."""

    def __init__(self, current_level: int, requested_level: int, lock_name: str):
        self.current_level = current_level
        self.requested_level = requested_level
        self.lock_name = lock_name
        super().__init__(
            f"Cannot acquire '{lock_name}' (level {requested_level}) while holding level {current_level}"
        )


# =============================================================================
# Lock Protocols
# =============================================================================

@runtime_checkable
class SyncLockProtocol(Protocol):
    """Protocol for synchronous locks."""

    def acquire(self, timeout: float = -1, blocking: bool = True) -> bool:
        """Acquire the lock."""
        ...

    def release(self) -> None:
        """Release the lock."""
        ...

    @property
    def name(self) -> str:
        """Get lock name."""
        ...


@runtime_checkable
class AsyncLockProtocol(Protocol):
    """Protocol for asynchronous locks."""

    async def acquire(self, timeout: float = -1) -> bool:
        """Acquire the lock asynchronously."""
        ...

    async def release(self) -> None:
        """Release the lock."""
        ...

    @property
    def name(self) -> str:
        """Get lock name."""
        ...


# Union type for any lock
LockProtocol = Union[SyncLockProtocol, AsyncLockProtocol]


# =============================================================================
# Lock Ordering
# =============================================================================

class OrderedLockGroup:
    """Manages acquiring multiple locks in consistent order.

    Prevents deadlocks by always acquiring locks in the same order
    (sorted by name/key).

    Example:
        group = OrderedLockGroup()
        group.add("lock_b", lock_b)
        group.add("lock_a", lock_a)

        async with group:
            # Locks acquired in order: lock_a, lock_b
            pass
    """

    def __init__(self):
        self._locks: dict[str, AsyncLockProtocol] = {}
        self._acquired: list[str] = []

    def add(self, name: str, lock: AsyncLockProtocol) -> OrderedLockGroup:
        """Add a lock to the group.

        Args:
            name: Lock name (used for ordering)
            lock: Lock instance

        Returns:
            Self for chaining
        """
        self._locks[name] = lock
        return self

    async def acquire_all(self, timeout: float = 30.0) -> bool:
        """Acquire all locks in sorted order.

        Args:
            timeout: Timeout per lock

        Returns:
            True if all locks acquired
        """
        # Sort by name for consistent order
        ordered_names = sorted(self._locks.keys())

        for name in ordered_names:
            lock = self._locks[name]
            try:
                acquired = await asyncio.wait_for(
                    lock.acquire(),
                    timeout=timeout,
                )
                if acquired:
                    self._acquired.append(name)
                else:
                    await self.release_all()
                    return False
            except asyncio.TimeoutError:
                await self.release_all()
                return False

        return True

    async def release_all(self) -> None:
        """Release all acquired locks in reverse order."""
        for name in reversed(self._acquired):
            try:
                lock = self._locks[name]
                await lock.release()
            except Exception as e:
                logger.warning(f"Error releasing lock {name}: {e}")

        self._acquired.clear()

    async def __aenter__(self) -> OrderedLockGroup:
        if not await self.acquire_all():
            raise LockError("Failed to acquire all locks")
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.release_all()


# =============================================================================
# Lock Hierarchy
# =============================================================================

class LockHierarchy:
    """Enforces lock acquisition hierarchy to prevent deadlocks.

    Locks are organized into levels. A thread can only acquire locks
    at a higher level than any locks it currently holds.

    Example:
        hierarchy = LockHierarchy()
        hierarchy.define_level("database", 1)  # Low level
        hierarchy.define_level("cache", 2)     # Medium level
        hierarchy.define_level("api", 3)       # High level

        # OK: acquiring in ascending order
        async with hierarchy.acquire("database"):
            async with hierarchy.acquire("cache"):
                pass

        # ERROR: acquiring lower level while holding higher
        async with hierarchy.acquire("cache"):
            async with hierarchy.acquire("database"):  # Raises!
                pass
    """

    def __init__(self):
        self._levels: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._held: dict[int, set[str]] = {}  # thread_id -> held lock names
        self._lock = threading.RLock()

    def define_level(self, name: str, level: int) -> None:
        """Define a lock level.

        Args:
            name: Lock name
            level: Hierarchy level (lower = must acquire first)
        """
        self._levels[name] = level
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()

    def _get_thread_id(self) -> int:
        """Get current thread identifier."""
        return threading.current_thread().ident or 0

    def _get_current_level(self) -> int:
        """Get highest level currently held by this thread."""
        thread_id = self._get_thread_id()
        with self._lock:
            held = self._held.get(thread_id, set())
            if not held:
                return 0
            return max(self._levels.get(name, 0) for name in held)

    def _check_hierarchy(self, name: str) -> None:
        """Check if acquiring this lock would violate hierarchy."""
        level = self._levels.get(name, 0)
        current_level = self._get_current_level()

        if level <= current_level and current_level > 0:
            raise LockHierarchyViolation(current_level, level, name)

    def _mark_acquired(self, name: str) -> None:
        """Mark lock as acquired by current thread."""
        thread_id = self._get_thread_id()
        with self._lock:
            if thread_id not in self._held:
                self._held[thread_id] = set()
            self._held[thread_id].add(name)

    def _mark_released(self, name: str) -> None:
        """Mark lock as released by current thread."""
        thread_id = self._get_thread_id()
        with self._lock:
            if thread_id in self._held:
                self._held[thread_id].discard(name)

    @asynccontextmanager
    async def acquire(
        self,
        name: str,
        timeout: float = 30.0,
    ) -> AsyncGenerator[None, None]:
        """Acquire a hierarchical lock.

        Args:
            name: Lock name
            timeout: Acquisition timeout

        Yields:
            Nothing, but lock is held

        Raises:
            LockHierarchyViolation: If acquisition would violate hierarchy
        """
        self._check_hierarchy(name)

        lock = self._locks.get(name)
        if lock is None:
            raise LockError(f"Lock '{name}' not defined in hierarchy")

        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise LockError(f"Timeout acquiring hierarchical lock: {name}")

        self._mark_acquired(name)

        try:
            yield
        finally:
            lock.release()
            self._mark_released(name)


# =============================================================================
# Lock Pool
# =============================================================================

class LockPool:
    """Pool of reusable locks for dynamic resources.

    Efficiently manages locks for resources that are created/destroyed
    dynamically, avoiding lock proliferation.

    Example:
        pool = LockPool(max_size=1000)

        async with pool.acquire("user:123"):
            # Lock for user 123
            pass

        async with pool.acquire("user:456"):
            # Different lock for user 456
            pass
    """

    def __init__(
        self,
        max_size: int = 10000,
        cleanup_interval: float = 60.0,
    ):
        """Initialize lock pool.

        Args:
            max_size: Maximum number of locks to keep
            cleanup_interval: Seconds between cleanup runs
        """
        self._max_size = max_size
        self._locks: dict[str, tuple[asyncio.Lock, float]] = {}  # name -> (lock, last_used)
        self._lock = threading.RLock()
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

    def _maybe_cleanup(self) -> None:
        """Cleanup old locks if needed."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        with self._lock:
            if len(self._locks) <= self._max_size * 0.9:
                self._last_cleanup = now
                return

            # Sort by last used, remove oldest
            sorted_locks = sorted(
                self._locks.items(),
                key=lambda x: x[1][1],
            )

            # Remove oldest 10%
            to_remove = len(sorted_locks) // 10
            for name, (lock, _) in sorted_locks[:to_remove]:
                if not lock.locked():
                    del self._locks[name]

            self._last_cleanup = now

    def _get_lock(self, name: str) -> asyncio.Lock:
        """Get or create lock for a name."""
        self._maybe_cleanup()

        with self._lock:
            if name in self._locks:
                lock, _ = self._locks[name]
                self._locks[name] = (lock, time.time())
                return lock

            lock = asyncio.Lock()
            self._locks[name] = (lock, time.time())
            return lock

    @asynccontextmanager
    async def acquire(
        self,
        name: str,
        timeout: float = 30.0,
    ) -> AsyncGenerator[None, None]:
        """Acquire a pooled lock.

        Args:
            name: Resource name to lock
            timeout: Acquisition timeout

        Yields:
            Nothing, but lock is held
        """
        lock = self._get_lock(name)

        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise LockError(f"Timeout acquiring pooled lock: {name}")

        try:
            yield
        finally:
            lock.release()

    def stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            active = sum(1 for _, (lock, _) in self._locks.items() if lock.locked())
            return {
                "total_locks": len(self._locks),
                "active_locks": active,
                "max_size": self._max_size,
            }


# =============================================================================
# Lock Factory
# =============================================================================

class LockType(Enum):
    """Types of locks available."""
    LOCAL_ASYNC = "local_async"      # In-process async lock
    LOCAL_SYNC = "local_sync"        # In-process threading lock
    DISTRIBUTED = "distributed"       # Cross-process (Redis/file)


class LockFactory:
    """Factory for creating appropriate lock types.

    Provides a unified way to create locks based on requirements.

    Example:
        factory = LockFactory()

        # Local async lock
        lock = factory.create("my_resource", LockType.LOCAL_ASYNC)

        # Distributed lock
        dist_lock = factory.create("training:config", LockType.DISTRIBUTED)
    """

    def __init__(self):
        self._pool = LockPool()

    def create(
        self,
        name: str,
        lock_type: LockType = LockType.LOCAL_ASYNC,
        timeout: int = 3600,
    ) -> Union[asyncio.Lock, threading.Lock, Any]:
        """Create a lock of the specified type.

        Args:
            name: Lock name
            lock_type: Type of lock to create
            timeout: Lock timeout for distributed locks

        Returns:
            Lock instance
        """
        if lock_type == LockType.LOCAL_ASYNC:
            return asyncio.Lock()

        if lock_type == LockType.LOCAL_SYNC:
            return threading.Lock()

        if lock_type == LockType.DISTRIBUTED:
            try:
                from app.coordination.distributed_lock import DistributedLock
                return DistributedLock(name, lock_timeout=timeout)
            except ImportError:
                logger.warning("DistributedLock not available, using local lock")
                return threading.Lock()

        raise ValueError(f"Unknown lock type: {lock_type}")

    @asynccontextmanager
    async def acquire(
        self,
        name: str,
        lock_type: LockType = LockType.LOCAL_ASYNC,
        timeout: float = 30.0,
    ) -> AsyncGenerator[None, None]:
        """Create and acquire a lock in one step.

        Args:
            name: Lock name
            lock_type: Type of lock
            timeout: Acquisition timeout

        Yields:
            Nothing, but lock is held
        """
        if lock_type == LockType.LOCAL_ASYNC:
            async with self._pool.acquire(name, timeout):
                yield
        else:
            lock = self.create(name, lock_type)
            try:
                if hasattr(lock, '__aenter__'):
                    async with lock:
                        yield
                else:
                    with lock:
                        yield
            finally:
                pass


# =============================================================================
# Convenience Functions
# =============================================================================

@asynccontextmanager
async def acquire_locks(
    *names: str,
    timeout: float = 30.0,
    factory: LockFactory | None = None,
) -> AsyncGenerator[None, None]:
    """Acquire multiple locks in consistent order.

    Locks are acquired in sorted order to prevent deadlocks.

    Args:
        names: Lock names to acquire
        timeout: Timeout per lock
        factory: Lock factory (uses default if None)

    Yields:
        Nothing, but all locks are held

    Example:
        async with acquire_locks("resource_a", "resource_b", "resource_c"):
            # All locks held
            pass
    """
    if factory is None:
        factory = LockFactory()

    sorted_names = sorted(names)
    acquired: list[str] = []

    try:
        for name in sorted_names:
            async with factory.acquire(name, timeout=timeout):
                acquired.append(name)
                if len(acquired) == len(names):
                    yield
    finally:
        pass  # Locks released by context managers


@contextmanager
def acquire_locks_sync(
    *names: str,
    timeout: float = 30.0,
) -> Generator[None, None, None]:
    """Acquire multiple sync locks in consistent order.

    Args:
        names: Lock names to acquire
        timeout: Total timeout

    Yields:
        Nothing, but all locks are held
    """
    sorted_names = sorted(names)
    locks = [threading.Lock() for _ in sorted_names]
    acquired: list[threading.Lock] = []

    try:
        for lock in locks:
            if lock.acquire(timeout=timeout):
                acquired.append(lock)
            else:
                raise LockError("Failed to acquire all locks")
        yield
    finally:
        for lock in reversed(acquired):
            lock.release()


# =============================================================================
# Diagnostics
# =============================================================================

@dataclass
class LockStats:
    """Statistics about lock usage."""
    name: str
    acquisitions: int = 0
    releases: int = 0
    contentions: int = 0  # Failed immediate acquisitions
    total_hold_time: float = 0.0
    max_hold_time: float = 0.0
    current_holder: str | None = None


class LockDiagnostics:
    """Diagnostics and monitoring for locks.

    Tracks lock usage patterns for debugging and optimization.

    Example:
        diag = LockDiagnostics()
        diag.track(my_lock)

        # Later
        stats = diag.get_stats("my_lock")
        print(f"Contentions: {stats.contentions}")
    """

    def __init__(self):
        self._stats: dict[str, LockStats] = {}
        self._lock = threading.RLock()

    def track(self, name: str) -> None:
        """Start tracking a lock."""
        with self._lock:
            if name not in self._stats:
                self._stats[name] = LockStats(name=name)

    def record_acquisition(
        self,
        name: str,
        holder: str = "",
        waited: bool = False,
    ) -> None:
        """Record a lock acquisition."""
        with self._lock:
            stats = self._stats.setdefault(name, LockStats(name=name))
            stats.acquisitions += 1
            stats.current_holder = holder
            if waited:
                stats.contentions += 1

    def record_release(
        self,
        name: str,
        hold_time: float,
    ) -> None:
        """Record a lock release."""
        with self._lock:
            stats = self._stats.get(name)
            if stats:
                stats.releases += 1
                stats.total_hold_time += hold_time
                stats.max_hold_time = max(stats.max_hold_time, hold_time)
                stats.current_holder = None

    def get_stats(self, name: str) -> LockStats | None:
        """Get stats for a specific lock."""
        with self._lock:
            return self._stats.get(name)

    def get_all_stats(self) -> dict[str, LockStats]:
        """Get stats for all tracked locks."""
        with self._lock:
            return dict(self._stats)

    def get_contentious_locks(self, min_contentions: int = 10) -> list[LockStats]:
        """Get locks with high contention."""
        with self._lock:
            return [
                stats for stats in self._stats.values()
                if stats.contentions >= min_contentions
            ]

    def get_long_held_locks(self, threshold: float = 5.0) -> list[LockStats]:
        """Get locks with long average hold times."""
        with self._lock:
            results = []
            for stats in self._stats.values():
                if stats.releases > 0:
                    avg_hold = stats.total_hold_time / stats.releases
                    if avg_hold >= threshold:
                        results.append(stats)
            return results
