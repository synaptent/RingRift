"""Unified Lock Manager with Deadlock Detection.

Provides in-process async lock management with:
- Unified interface for async lock coordination
- Lock tracking for observability
- Deadlock detection via wait-for graph analysis
- Event emission for monitoring

When to Use This vs DistributedLock (December 2025):
- **LockManager** (this module): Use for in-process async coordination
  with deadlock detection. Suitable for coordinating async tasks within
  a single Python process. Does NOT provide cross-process locking.

- **DistributedLock** (app.coordination.distributed_lock): Use for
  cross-node/cross-process coordination via Redis or file locks.
  Suitable for training locks, model registry, resource allocation.

Usage:
    from app.coordination.lock_manager import get_lock_manager

    manager = get_lock_manager()

    # Acquire lock
    async with manager.acquire("resource_id", timeout=30):
        # Do work
        pass

    # Or manual management
    lock = await manager.acquire_lock("resource_id", timeout=30)
    try:
        # Do work
        pass
    finally:
        await manager.release_lock("resource_id")

December 2025 - Consolidation effort
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LockType(Enum):
    """Types of locks."""
    EXCLUSIVE = "exclusive"
    SHARED = "shared"


class LockStatus(Enum):
    """Lock status."""
    HELD = "held"
    WAITING = "waiting"
    RELEASED = "released"
    TIMEOUT = "timeout"


@dataclass
class LockInfo:
    """Information about a held or pending lock."""
    resource_id: str
    holder_id: str
    lock_type: LockType
    acquired_at: float
    timeout_seconds: float
    status: LockStatus = LockStatus.WAITING
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def held_duration(self) -> float:
        """Seconds since lock was acquired."""
        if self.status == LockStatus.HELD:
            return time.time() - self.acquired_at
        return 0.0

    @property
    def is_expired(self) -> bool:
        """Check if lock has exceeded its timeout."""
        if self.timeout_seconds <= 0:
            return False
        return self.held_duration > self.timeout_seconds


class DeadlockError(Exception):
    """Raised when a deadlock is detected."""
    def __init__(self, cycle: list[str], message: str = "Deadlock detected"):
        self.cycle = cycle
        super().__init__(f"{message}: {' -> '.join(cycle)}")


class LockManager:
    """Unified lock manager with deadlock detection.

    Thread-safe and async-compatible lock manager that tracks all locks
    and can detect potential deadlocks using wait-for graph analysis.
    """

    def __init__(self):
        # Lock storage
        self._locks: dict[str, LockInfo] = {}
        self._lock = threading.RLock()

        # Wait-for graph: holder_id -> set of resource_ids they're waiting for
        self._waiting_for: dict[str, set[str]] = defaultdict(set)

        # Resource -> holder mapping
        self._resource_holders: dict[str, str] = {}

        # Async locks per resource
        self._async_locks: dict[str, asyncio.Lock] = {}

        # Statistics
        self._stats = {
            "locks_acquired": 0,
            "locks_released": 0,
            "deadlocks_detected": 0,
            "timeouts": 0,
        }

        # Event emission
        try:
            from app.distributed.data_events import (
                emit_deadlock_detected,
                emit_lock_acquired,
                emit_lock_released,
            )
            self._emit_acquired = emit_lock_acquired
            self._emit_released = emit_lock_released
            self._emit_deadlock = emit_deadlock_detected
        except ImportError:
            self._emit_acquired = None
            self._emit_released = None
            self._emit_deadlock = None

    def _get_async_lock(self, resource_id: str) -> asyncio.Lock:
        """Get or create async lock for a resource."""
        with self._lock:
            if resource_id not in self._async_locks:
                self._async_locks[resource_id] = asyncio.Lock()
            return self._async_locks[resource_id]

    def _detect_deadlock(self, holder_id: str, waiting_for_resource: str) -> list[str] | None:
        """Detect if acquiring this lock would cause a deadlock.

        Uses DFS to find cycles in the wait-for graph.

        Returns:
            Cycle path if deadlock detected, None otherwise
        """
        # Build wait-for graph with the new edge
        # holder_id wants waiting_for_resource, which is held by some other holder

        current_holder = self._resource_holders.get(waiting_for_resource)
        if not current_holder:
            return None  # Resource not held, no deadlock

        if current_holder == holder_id:
            return None  # Same holder, not a deadlock (reentrant)

        # DFS to find cycle
        visited = set()
        path = [holder_id]

        def dfs(current: str) -> list[str] | None:
            if current in visited:
                return None

            if current == holder_id and len(path) > 1:
                # Found cycle back to original holder
                return [*path, holder_id]

            visited.add(current)
            path.append(current)

            # Check what resources this holder is waiting for
            for resource in self._waiting_for.get(current, set()):
                next_holder = self._resource_holders.get(resource)
                if next_holder:
                    result = dfs(next_holder)
                    if result:
                        return result

            path.pop()
            return None

        return dfs(current_holder)

    async def acquire_lock(
        self,
        resource_id: str,
        holder_id: str = "",
        lock_type: LockType = LockType.EXCLUSIVE,
        timeout_seconds: float = 30.0,
        check_deadlock: bool = True,
    ) -> LockInfo:
        """Acquire a lock on a resource.

        Args:
            resource_id: Unique identifier for the resource
            holder_id: Identifier for the lock holder (default: thread id)
            lock_type: Type of lock (exclusive or shared)
            timeout_seconds: Maximum time to wait for lock
            check_deadlock: Whether to check for deadlocks before waiting

        Returns:
            LockInfo for the acquired lock

        Raises:
            DeadlockError: If acquiring would cause a deadlock
            asyncio.TimeoutError: If lock cannot be acquired within timeout
        """
        if not holder_id:
            holder_id = f"thread-{threading.current_thread().ident}"

        # Check for potential deadlock
        if check_deadlock:
            with self._lock:
                cycle = self._detect_deadlock(holder_id, resource_id)
                if cycle:
                    self._stats["deadlocks_detected"] += 1

                    # Emit deadlock event
                    if self._emit_deadlock:
                        with suppress(RuntimeError):
                            asyncio.create_task(self._emit_deadlock(
                                resources=[resource_id],
                                holders=cycle,
                                source="lock_manager",
                            ))

                    raise DeadlockError(cycle)

                # Record that we're waiting for this resource
                self._waiting_for[holder_id].add(resource_id)

        # Create lock info
        lock_info = LockInfo(
            resource_id=resource_id,
            holder_id=holder_id,
            lock_type=lock_type,
            acquired_at=time.time(),
            timeout_seconds=timeout_seconds,
            status=LockStatus.WAITING,
        )

        try:
            # Acquire the async lock
            async_lock = self._get_async_lock(resource_id)

            try:
                await asyncio.wait_for(
                    async_lock.acquire(),
                    timeout=timeout_seconds if timeout_seconds > 0 else None
                )
            except asyncio.TimeoutError:
                self._stats["timeouts"] += 1
                lock_info.status = LockStatus.TIMEOUT
                raise

            # Lock acquired
            with self._lock:
                lock_info.acquired_at = time.time()
                lock_info.status = LockStatus.HELD
                self._locks[resource_id] = lock_info
                self._resource_holders[resource_id] = holder_id
                self._waiting_for[holder_id].discard(resource_id)
                self._stats["locks_acquired"] += 1

            # Emit event
            if self._emit_acquired:
                with suppress(RuntimeError):
                    asyncio.create_task(self._emit_acquired(
                        resource_id=resource_id,
                        holder=holder_id,
                        lock_type=lock_type.value,
                        timeout_seconds=timeout_seconds,
                        source="lock_manager",
                    ))

            logger.debug(f"[LockManager] Acquired lock: {resource_id} by {holder_id}")
            return lock_info

        except Exception:
            # Clean up waiting state on failure
            with self._lock:
                self._waiting_for[holder_id].discard(resource_id)
            raise

    async def release_lock(self, resource_id: str) -> None:
        """Release a held lock.

        Args:
            resource_id: Resource to release
        """
        with self._lock:
            lock_info = self._locks.pop(resource_id, None)
            holder_id = self._resource_holders.pop(resource_id, "")

        if not lock_info:
            logger.warning(f"[LockManager] Attempted to release unheld lock: {resource_id}")
            return

        # Release the async lock
        async_lock = self._async_locks.get(resource_id)
        if async_lock and async_lock.locked():
            async_lock.release()

        held_duration = time.time() - lock_info.acquired_at

        with self._lock:
            self._stats["locks_released"] += 1

        # Emit event
        if self._emit_released:
            with suppress(RuntimeError):
                asyncio.create_task(self._emit_released(
                    resource_id=resource_id,
                    holder=holder_id,
                    held_duration_seconds=held_duration,
                    source="lock_manager",
                ))

        logger.debug(f"[LockManager] Released lock: {resource_id} (held {held_duration:.2f}s)")

    @asynccontextmanager
    async def acquire(
        self,
        resource_id: str,
        holder_id: str = "",
        lock_type: LockType = LockType.EXCLUSIVE,
        timeout_seconds: float = 30.0,
    ):
        """Context manager for acquiring and releasing locks.

        Usage:
            async with manager.acquire("my_resource"):
                # Do work
                pass
        """
        await self.acquire_lock(resource_id, holder_id, lock_type, timeout_seconds)
        try:
            yield
        finally:
            await self.release_lock(resource_id)

    def get_held_locks(self) -> list[LockInfo]:
        """Get all currently held locks."""
        with self._lock:
            return list(self._locks.values())

    def get_lock_stats(self) -> dict[str, Any]:
        """Get lock statistics."""
        with self._lock:
            return {
                **self._stats,
                "currently_held": len(self._locks),
                "waiting_count": sum(len(w) for w in self._waiting_for.values()),
            }

    def is_locked(self, resource_id: str) -> bool:
        """Check if a resource is currently locked."""
        with self._lock:
            return resource_id in self._locks

    def get_expired_locks(self) -> list[LockInfo]:
        """Get locks that have exceeded their timeout."""
        with self._lock:
            return [lock for lock in self._locks.values() if lock.is_expired]

    def force_release(self, resource_id: str) -> bool:
        """Force release a lock (for recovery/admin use).

        Returns:
            True if lock was released, False if not held
        """
        with self._lock:
            if resource_id not in self._locks:
                return False

            self._locks.pop(resource_id)
            self._resource_holders.pop(resource_id, None)

        # Release async lock
        async_lock = self._async_locks.get(resource_id)
        if async_lock and async_lock.locked():
            async_lock.release()

        logger.warning(f"[LockManager] Force released lock: {resource_id}")
        return True


# Singleton instance
_lock_manager: LockManager | None = None
_init_lock = threading.Lock()


def get_lock_manager() -> LockManager:
    """Get the global lock manager singleton."""
    global _lock_manager
    if _lock_manager is None:
        with _init_lock:
            if _lock_manager is None:
                _lock_manager = LockManager()
    return _lock_manager


def reset_lock_manager() -> None:
    """Reset the lock manager (for testing)."""
    global _lock_manager
    with _init_lock:
        _lock_manager = None


__all__ = [
    "DeadlockError",
    "LockInfo",
    "LockManager",
    "LockStatus",
    "LockType",
    "get_lock_manager",
    "reset_lock_manager",
]
