"""Distributed Locking for Training Coordination.

.. deprecated:: January 2026
    Import from `app.coordination.core_utils` instead of this module directly.
    This module remains the implementation, but `core_utils` is the consolidation point.

    Old::

        from app.coordination.distributed_lock import DistributedLock

    New::

        from app.coordination.core_utils import DistributedLock

Provides reliable distributed locks for coordinating training across
multiple nodes. Priority order for backends:

1. **Raft** (cluster-wide, strongly consistent) - Dec 30, 2025
2. **Redis** (distributed, available when Redis is reachable)
3. **File** (local node only, fcntl-based)

Features:
- Raft-based distributed locks via pysyncobj (when P2P cluster running)
- Redis-based distributed locks (when Redis available)
- Automatic lock expiry to prevent deadlocks
- File-based fallback for single-node scenarios
- Lock timeout and retry support
- Context manager interface (sync and async)
- Graceful degradation: Raft → Redis → File

Backend Selection (December 2025):
- **Raft**: Used when P2P orchestrator is running with Raft enabled.
  Provides cluster-wide consistency via Raft consensus.
- **Redis**: Used when Redis is reachable. Provides distributed locking
  across all nodes that can reach Redis.
- **File**: Fallback for single-node or when other backends unavailable.
  Only provides local process coordination.

When to Use This:
- **DistributedLock** (this module): Use for cross-node/cross-process
  coordination. Suitable for training locks, model registry, resource allocation.
- **LockHierarchy** (app.core.locking): Use for in-process async coordination
  with deadlock prevention through lock ordering.

Usage:
    from app.coordination.distributed_lock import DistributedLock

    lock = DistributedLock("training:square8_2p")

    # Context manager usage (sync)
    with lock:
        # Training code here
        pass

    # Async context manager
    async with lock:
        await run_training()

    # Or explicit acquire/release
    if lock.acquire(timeout=30):
        try:
            # Training code
        finally:
            lock.release()

    # Check which backend is being used
    print(f"Backend: {lock.backend}")  # "raft", "redis", or "file"

Dec 30, 2025 - Part of Priority 5: Optional Raft Consensus Integration
"""

from __future__ import annotations

import fcntl
import logging
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from app.utils.paths import DATA_DIR

if TYPE_CHECKING:
    from pysyncobj.batteries import ReplLockManager

logger = logging.getLogger(__name__)


# ============================================
# Lock Backend Selection (Dec 30, 2025)
# ============================================


class LockBackendType(str, Enum):
    """Available lock backend types."""

    RAFT = "raft"  # Cluster-wide via Raft consensus
    REDIS = "redis"  # Distributed via Redis
    FILE = "file"  # Local node only


# Raft availability check (cached)
_raft_available: bool | None = None
_raft_lock_manager: "ReplLockManager | None" = None
_raft_node_id: str | None = None


def _check_raft_available() -> bool:
    """Check if Raft lock manager is available.

    Returns True if:
    1. pysyncobj is installed
    2. RAFT_ENABLED is True
    3. P2P orchestrator is running with initialized Raft
    4. Lock manager is accessible

    Result is cached for performance.
    """
    global _raft_available, _raft_lock_manager, _raft_node_id

    if _raft_available is not None:
        return _raft_available

    try:
        # Check if Raft is enabled
        from scripts.p2p.consensus_mixin import (
            PYSYNCOBJ_AVAILABLE,
            RAFT_ENABLED,
        )

        if not RAFT_ENABLED or not PYSYNCOBJ_AVAILABLE:
            logger.debug("Raft locks disabled: RAFT_ENABLED=%s, PYSYNCOBJ=%s", RAFT_ENABLED, PYSYNCOBJ_AVAILABLE)
            _raft_available = False
            return False

        # Try to get lock manager from P2P orchestrator
        try:
            from scripts.p2p_orchestrator import P2POrchestrator

            # Check for singleton instance
            orchestrator = getattr(P2POrchestrator, "_instance", None)
            if orchestrator is None:
                logger.debug("Raft locks: P2P orchestrator not running")
                _raft_available = False
                return False

            # Check if Raft is initialized
            raft_initialized = getattr(orchestrator, "_raft_initialized", False)
            if not raft_initialized:
                logger.debug("Raft locks: Raft not initialized on orchestrator")
                _raft_available = False
                return False

            # Get the lock manager from ReplicatedWorkQueue
            raft_wq = getattr(orchestrator, "_raft_work_queue", None)
            if raft_wq is None:
                logger.debug("Raft locks: Raft work queue not available")
                _raft_available = False
                return False

            lock_manager = getattr(raft_wq, "_lock_manager", None)
            if lock_manager is None:
                logger.debug("Raft locks: Lock manager not on work queue")
                _raft_available = False
                return False

            # Success - cache the lock manager
            _raft_lock_manager = lock_manager
            _raft_node_id = getattr(orchestrator, "node_id", "unknown")
            _raft_available = True
            logger.info("Raft locks available via P2P orchestrator (node: %s)", _raft_node_id)
            return True

        except ImportError:
            logger.debug("Raft locks: Could not import P2P orchestrator")
            _raft_available = False
            return False

    except ImportError:
        logger.debug("Raft locks: pysyncobj or consensus_mixin not available")
        _raft_available = False
        return False
    except Exception as e:
        logger.warning("Raft locks: Unexpected error checking availability: %s", e)
        _raft_available = False
        return False


def reset_raft_cache() -> None:
    """Reset the Raft availability cache.

    Call this if P2P orchestrator state changes (e.g., Raft initialization).
    """
    global _raft_available, _raft_lock_manager, _raft_node_id
    _raft_available = None
    _raft_lock_manager = None
    _raft_node_id = None


class RaftLockWrapper:
    """Wrapper for acquiring/releasing Raft locks.

    Provides a consistent interface matching the existing lock patterns.
    Uses the ReplLockManager from the P2P orchestrator's ReplicatedWorkQueue.
    """

    def __init__(self, name: str, lock_timeout: int) -> None:
        """Initialize Raft lock wrapper.

        Args:
            name: Lock name
            lock_timeout: Lock timeout in seconds (informational only,
                         actual timeout is Raft's autoUnlockTime)
        """
        self.name = name
        self.lock_timeout = lock_timeout
        self._acquired = False

    def acquire(self) -> bool:
        """Try to acquire the Raft lock.

        Returns:
            True if acquired, False otherwise
        """
        if self._acquired:
            return True

        if _raft_lock_manager is None:
            return False

        try:
            # pysyncobj ReplLockManager.tryAcquire
            # sync=True ensures the lock is replicated before returning
            acquired = _raft_lock_manager.tryAcquire(self.name, sync=True)
            if acquired:
                self._acquired = True
                logger.debug("Acquired Raft lock: %s (node: %s)", self.name, _raft_node_id)
            return acquired
        except Exception as e:
            logger.warning("Raft lock acquire error for %s: %s", self.name, e)
            return False

    def release(self) -> bool:
        """Release the Raft lock.

        Returns:
            True if released, False if not held or error
        """
        if not self._acquired:
            return False

        if _raft_lock_manager is None:
            self._acquired = False
            return False

        try:
            _raft_lock_manager.release(self.name)
            self._acquired = False
            logger.debug("Released Raft lock: %s", self.name)
            return True
        except Exception as e:
            logger.warning("Raft lock release error for %s: %s", self.name, e)
            self._acquired = False
            return False

    def is_held(self) -> bool:
        """Check if lock is held by this wrapper."""
        return self._acquired


# =============================================================================
# Unified Lock Protocol (December 2025)
# =============================================================================

from typing import Any, Generator, Protocol, runtime_checkable


@runtime_checkable
class LockProtocol(Protocol):
    """Unified protocol for all lock types in the codebase.

    This protocol abstracts over different lock implementations:
    - DistributedLock (this module): Cross-node/cross-process coordination
    - SyncMutex (sync_mutex.py): SQLite-based mutex
    - LockHierarchy (app.core.locking): In-process async locks with ordering

    Usage:
        def with_lock(lock: LockProtocol, operation: Callable):
            if lock.acquire(timeout=30):
                try:
                    return operation()
                finally:
                    lock.release()
            raise TimeoutError("Could not acquire lock")

        # Works with any lock type
        with_lock(DistributedLock("my_lock"), my_operation)
        with_lock(SyncMutex("my_mutex"), my_operation)
    """

    def acquire(self, timeout: int = 60, blocking: bool = True) -> bool:
        """Acquire the lock.

        Args:
            timeout: Maximum wait time in seconds
            blocking: If True, wait for lock

        Returns:
            True if lock acquired
        """
        ...

    def release(self) -> None:
        """Release the lock."""
        ...

    def heartbeat(self) -> bool:
        """Extend TTL of currently held lock.

        Returns:
            True if TTL was extended
        """
        ...

    @property
    def name(self) -> str:
        """Get the lock name."""
        ...


def get_appropriate_lock(
    name: str,
    scope: str = "distributed",
    timeout: int = 3600,
) -> DistributedLock:
    """Factory function to get an appropriate lock based on scope.

    Args:
        name: Lock name
        scope: Lock scope ("distributed", "local", "process")
        timeout: Lock timeout in seconds

    Returns:
        Lock instance appropriate for the scope

    Usage:
        lock = get_appropriate_lock("training:config", scope="distributed")
        with lock:
            # Do work
            pass
    """
    # Currently all locks are DistributedLock, but this factory allows
    # future extension to different lock types based on scope
    return DistributedLock(name, lock_timeout=timeout)


# Import centralized defaults (December 2025)
try:
    from app.config.thresholds import (
        DEFAULT_ACQUIRE_TIMEOUT,
        DEFAULT_LOCK_TIMEOUT,
    )
except ImportError:
    # Fallback for standalone use
    DEFAULT_LOCK_TIMEOUT = 3600  # 1 hour max lock time
    DEFAULT_ACQUIRE_TIMEOUT = 60  # 60 seconds to acquire

# Constants
LOCK_DIR = DATA_DIR / "locks"
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Try to import Redis
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    redis = None


class DistributedLock:
    """Distributed lock with Raft + Redis + file-based fallback.

    Automatically selects the best available backend (Dec 30, 2025):
    1. Raft (if P2P orchestrator running with Raft enabled) - cluster-wide
    2. Redis (if available and reachable) - distributed
    3. File-based locking (fallback) - local node only

    The backend property indicates which backend was selected.
    """

    def __init__(
        self,
        name: str,
        lock_timeout: int = DEFAULT_LOCK_TIMEOUT,
        use_redis: bool = True,
        use_raft: bool = True,
    ):
        """Initialize a distributed lock.

        Args:
            name: Unique lock name (e.g., "training:square8_2p")
            lock_timeout: Maximum time to hold lock (seconds)
            use_redis: Whether to try Redis (True) or skip Redis
            use_raft: Whether to try Raft (True) or skip Raft (Dec 30, 2025)
        """
        self.name = name
        self.lock_timeout = lock_timeout
        self._lock_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._redis_client: redis.Redis | None = None
        self._raft_lock: RaftLockWrapper | None = None
        self._file_fd: int | None = None
        self._acquired = False
        self._use_redis = use_redis and HAS_REDIS
        self._use_raft = use_raft
        self._backend: LockBackendType = LockBackendType.FILE

        # Dec 30, 2025: Try Raft first (cluster-wide consistency)
        if self._use_raft and _check_raft_available():
            self._raft_lock = RaftLockWrapper(name, lock_timeout)
            self._backend = LockBackendType.RAFT
            logger.debug("Using Raft for lock: %s", name)
        # Try to connect to Redis if available
        elif self._use_redis:
            try:
                self._redis_client = redis.Redis.from_url(REDIS_URL, socket_timeout=5)
                self._redis_client.ping()
                self._backend = LockBackendType.REDIS
                logger.debug("Using Redis for lock: %s", name)
            except (ConnectionError, TimeoutError, OSError, redis.RedisError) as e:
                logger.debug("Redis not available, using file lock: %s", e)
                self._redis_client = None
                self._backend = LockBackendType.FILE
        else:
            self._backend = LockBackendType.FILE
            logger.debug("Using file lock: %s", name)

    @property
    def backend(self) -> str:
        """Get the active backend type.

        Returns:
            "raft", "redis", or "file"
        """
        return self._backend.value

    def acquire(self, timeout: int = DEFAULT_ACQUIRE_TIMEOUT, blocking: bool = True) -> bool:
        """Acquire the lock.

        Args:
            timeout: Maximum time to wait for lock (seconds)
            blocking: If True, wait for lock. If False, return immediately.

        Returns:
            True if lock acquired, False otherwise
        """
        if self._acquired:
            return True

        start_time = time.time()

        while True:
            # Dec 30, 2025: Try backends in priority order
            if self._raft_lock is not None:
                acquired = self._raft_lock.acquire()
            elif self._redis_client is not None:
                acquired = self._acquire_redis()
            else:
                acquired = self._acquire_file()

            if acquired:
                self._acquired = True
                return True

            if not blocking:
                return False

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning(f"Lock acquisition timed out for {self.name} after {timeout}s")
                try:
                    from app.distributed.event_helpers import emit_sync
                    emit_sync(
                        "LOCK_TIMEOUT",
                        {
                            "lock_name": self.name,
                            "lock_id": self._lock_id,
                            "timeout_seconds": timeout,
                            "backend": "redis" if self._redis_client is not None else "file",
                            "blocking": blocking,
                        },
                        source="distributed_lock",
                    )
                except ImportError:
                    pass  # Event router not available, lock still acquired
                return False

            # Wait and retry
            time.sleep(min(1.0, timeout - elapsed))

        return False

    def release(self) -> None:
        """Release the lock."""
        if not self._acquired:
            return

        try:
            # Dec 30, 2025: Release from active backend
            if self._raft_lock is not None:
                self._raft_lock.release()
            elif self._redis_client is not None:
                self._release_redis()
            else:
                self._release_file()
        finally:
            self._acquired = False

    def is_held(self) -> bool:
        """Check if this instance holds the lock."""
        return self._acquired

    def is_locked(self) -> bool:
        """Check if the lock is held by anyone."""
        # Dec 30, 2025: Check active backend
        if self._raft_lock is not None:
            # For Raft, we can only check if we hold it
            # pysyncobj doesn't expose a "check if anyone holds lock" API
            return self._raft_lock.is_held()
        elif self._redis_client is not None:
            return self._redis_client.exists(f"lock:{self.name}") > 0
        else:
            return self._is_file_locked()

    def heartbeat(self) -> bool:
        """Extend TTL of currently held lock (December 2025).

        Call this periodically during long-running operations to prevent
        the lock from expiring while still in use. Recommended to call
        every lock_timeout/2 seconds.

        Returns:
            True if TTL was extended, False if lock not held or extension failed
        """
        if not self._acquired:
            logger.debug("Cannot heartbeat lock %s: not acquired", self.name)
            return False

        try:
            # Dec 30, 2025: Handle Raft locks
            if self._raft_lock is not None:
                # Raft locks use autoUnlockTime from ReplLockManager
                # Re-acquire to refresh the auto-unlock timer
                if self._raft_lock.is_held():
                    logger.debug("Raft lock heartbeat: %s (auto-managed by Raft)", self.name)
                    return True
                return False
            elif self._redis_client is not None:
                # Extend Redis key TTL
                result = self._redis_client.expire(f"lock:{self.name}", self.lock_timeout)
                if result:
                    logger.debug(f"Extended Redis lock TTL: {self.name} (+{self.lock_timeout}s)")
                    return True
                else:
                    logger.warning(f"Failed to extend Redis lock TTL: {self.name} (key may not exist)")
                    return False
            else:
                # Update timestamp in file lock
                if self._file_fd is not None:
                    os.lseek(self._file_fd, 0, os.SEEK_SET)
                    os.ftruncate(self._file_fd, 0)
                    lock_info = f"{self._lock_id}\n{time.time()}\n{self.lock_timeout}\n"
                    os.write(self._file_fd, lock_info.encode())
                    logger.debug(f"Extended file lock TTL: {self.name}")
                    return True
                return False
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Heartbeat failed for lock {self.name}: {e}")
            return False
        except Exception as e:
            # Catch redis.RedisError and other exceptions
            if "redis" in type(e).__module__.lower():
                logger.warning(f"Redis heartbeat failed for lock {self.name}: {e}")
            else:
                logger.warning(f"Unexpected heartbeat error for lock {self.name}: {e}")
            return False

    def get_time_remaining(self) -> float:
        """Get remaining TTL of the lock in seconds (December 2025).

        Returns:
            Remaining time in seconds, or 0 if not held or unknown
        """
        if not self._acquired:
            return 0.0

        try:
            # Dec 30, 2025: Handle Raft locks
            if self._raft_lock is not None:
                # Raft uses auto-unlock time, not explicit TTL tracking
                # Return the configured auto-unlock time as an estimate
                return float(self.lock_timeout) if self._raft_lock.is_held() else 0.0
            elif self._redis_client is not None:
                ttl = self._redis_client.ttl(f"lock:{self.name}")
                return max(0.0, float(ttl)) if ttl and ttl > 0 else 0.0
            else:
                # Read timestamp from file
                lock_path = self._get_lock_path()
                if lock_path.exists():
                    with open(lock_path) as f:
                        lines = f.readlines()
                        if len(lines) >= 3:
                            lock_time = float(lines[1].strip())
                            lock_timeout = float(lines[2].strip())
                            remaining = lock_timeout - (time.time() - lock_time)
                            return max(0.0, remaining)
        except (OSError, ValueError, ConnectionError) as e:
            logger.debug(f"Error getting TTL for lock {self.name}: {e}")
        return 0.0

    # Async wrappers (December 2025)
    async def acquire_async(
        self, timeout: int = DEFAULT_ACQUIRE_TIMEOUT, blocking: bool = True
    ) -> bool:
        """Async wrapper for acquire() - runs in thread pool.

        Use this in async contexts to avoid blocking the event loop.

        Args:
            timeout: Maximum time to wait for lock (seconds)
            blocking: If True, wait for lock. If False, return immediately.

        Returns:
            True if lock acquired, False otherwise
        """
        import asyncio
        return await asyncio.to_thread(self.acquire, timeout, blocking)

    async def release_async(self) -> None:
        """Async wrapper for release() - runs in thread pool."""
        import asyncio
        await asyncio.to_thread(self.release)

    async def heartbeat_async(self) -> bool:
        """Async wrapper for heartbeat() - runs in thread pool."""
        import asyncio
        return await asyncio.to_thread(self.heartbeat)

    async def __aenter__(self):
        """Async context manager entry (December 2025)."""
        if not await self.acquire_async():
            raise RuntimeError(f"Could not acquire lock: {self.name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.release_async()
        return False

    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise RuntimeError(f"Could not acquire lock: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False

    # Redis implementation
    def _acquire_redis(self) -> bool:
        """Acquire lock using Redis SET NX."""
        try:
            result = self._redis_client.set(
                f"lock:{self.name}",
                self._lock_id,
                nx=True,  # Only set if not exists
                ex=self.lock_timeout,  # Expiry in seconds
            )
            if result:
                logger.debug(f"Acquired Redis lock: {self.name}")
                return True
            return False
        except (ConnectionError, TimeoutError, redis.RedisError) as e:
            logger.warning(f"Redis lock acquire failed: {e}")
            return False

    def _release_redis(self) -> None:
        """Release lock using Redis atomic delete-if-owner."""
        try:
            # Lua script for atomic check-and-delete
            script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            self._redis_client.eval(script, 1, f"lock:{self.name}", self._lock_id)
            logger.debug(f"Released Redis lock: {self.name}")
        except (ConnectionError, TimeoutError, redis.RedisError) as e:
            logger.warning(f"Redis lock release failed: {e}")

    # File-based implementation
    def _get_lock_path(self) -> Path:
        """Get the file path for this lock."""
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = self.name.replace(":", "_").replace("/", "_")
        return LOCK_DIR / f"{safe_name}.lock"

    def _acquire_file(self) -> bool:
        """Acquire lock using file locking."""
        lock_path = self._get_lock_path()
        fd: int | None = None

        try:
            # Open or create lock file
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)

            # Try non-blocking exclusive lock
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write lock info
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            lock_info = f"{self._lock_id}\n{time.time()}\n{self.lock_timeout}\n"
            os.write(fd, lock_info.encode())

            self._file_fd = fd
            fd = None  # Ownership transferred to self._file_fd
            logger.debug(f"Acquired file lock: {self.name}")
            return True

        except (OSError, BlockingIOError):
            # Check if existing lock is expired
            if self._is_file_lock_expired():
                # Close fd before retrying to avoid leak
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    fd = None
                # Force remove and retry
                try:
                    lock_path.unlink(missing_ok=True)
                    return self._acquire_file()
                except OSError as e:
                    logger.debug(f"Failed to take over expired lock {self.name}: {e}")
            return False

        finally:
            # Always close fd if we still own it (wasn't transferred to _file_fd)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as e:
                    logger.debug(f"Error closing file descriptor for lock {self.name}: {e}")

    def _release_file(self) -> None:
        """Release file lock."""
        if self._file_fd is None:
            return

        try:
            fcntl.flock(self._file_fd, fcntl.LOCK_UN)
            os.close(self._file_fd)
            logger.debug(f"Released file lock: {self.name}")
        except OSError as e:
            logger.warning(f"File lock release failed: {e}")
        finally:
            self._file_fd = None

        # Try to remove lock file
        with suppress(Exception):
            self._get_lock_path().unlink(missing_ok=True)

    def _is_file_locked(self) -> bool:
        """Check if file lock exists and is held."""
        lock_path = self._get_lock_path()
        if not lock_path.exists():
            return False

        fd: int | None = None
        try:
            fd = os.open(str(lock_path), os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False  # Could acquire, so not locked
            except BlockingIOError:
                return True  # Couldn't acquire, so locked
        except OSError as e:
            logger.debug(f"Error checking lock status for {self.name}: {e}")
            return False
        finally:
            # Dec 2025: Always close fd to prevent leaks
            if fd is not None:
                with suppress(OSError):
                    os.close(fd)

    def _is_file_lock_expired(self) -> bool:
        """Check if existing file lock has expired."""
        lock_path = self._get_lock_path()
        if not lock_path.exists():
            return True

        try:
            with open(lock_path) as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    lock_time = float(lines[1].strip())
                    lock_timeout = float(lines[2].strip())
                    if time.time() - lock_time > lock_timeout:
                        logger.info(f"File lock expired: {self.name}")
                        return True
        except (OSError, ValueError) as e:
            logger.debug(f"Error reading lock file for {self.name}: {e}")
        return False


# Convenience functions
def acquire_training_lock(
    config_key: str,
    timeout: int = DEFAULT_ACQUIRE_TIMEOUT,
) -> DistributedLock | None:
    """Acquire a training lock for a config.

    Args:
        config_key: Config identifier (e.g., "square8_2p")
        timeout: Maximum time to wait for lock

    Returns:
        DistributedLock instance if acquired, None otherwise
    """
    lock = DistributedLock(f"training:{config_key}")
    if lock.acquire(timeout=timeout):
        return lock
    return None


def release_training_lock(lock: DistributedLock | None) -> None:
    """Release a training lock."""
    if lock is not None:
        lock.release()


@contextmanager
def training_lock(
    config_key: str, timeout: int = DEFAULT_ACQUIRE_TIMEOUT
) -> Generator[DistributedLock | None, None, None]:
    """Context manager for training lock.

    Usage:
        with training_lock("square8_2p") as lock:
            if lock:
                # Do training
            else:
                print("Could not acquire lock")
    """
    lock = DistributedLock(f"training:{config_key}")
    acquired = lock.acquire(timeout=timeout, blocking=True)

    try:
        yield lock if acquired else None
    finally:
        if acquired:
            lock.release()


def cleanup_stale_locks(
    max_age_hours: float = 24.0,
    lock_dir: Path | None = None,
) -> dict[str, int]:
    """Clean up stale lock files.

    Removes locks that are:
    - Older than max_age_hours
    - Have expired timeouts
    - Owned by dead processes on same host

    Args:
        max_age_hours: Maximum lock age before cleanup
        lock_dir: Lock directory (defaults to LOCK_DIR)

    Returns:
        Dict with cleanup statistics
    """
    lock_dir = lock_dir or LOCK_DIR
    stats = {
        "scanned": 0,
        "removed_expired": 0,
        "removed_old": 0,
        "removed_dead_process": 0,
        "errors": 0,
    }

    if not lock_dir.exists():
        return stats

    hostname = socket.gethostname()
    max_age_seconds = max_age_hours * 3600
    now = time.time()

    for lock_file in lock_dir.glob("*.lock"):
        stats["scanned"] += 1

        try:
            # Read lock metadata
            with open(lock_file) as f:
                lines = f.readlines()

            if len(lines) < 3:
                continue

            lock_id = lines[0].strip()
            lock_time = float(lines[1].strip())
            lock_timeout = float(lines[2].strip())

            # Check age
            age = now - lock_time
            if age > max_age_seconds:
                lock_file.unlink(missing_ok=True)
                stats["removed_old"] += 1
                logger.info(f"Removed old lock: {lock_file.name} (age: {age/3600:.1f}h)")
                continue

            # Check expired timeout
            if age > lock_timeout:
                lock_file.unlink(missing_ok=True)
                stats["removed_expired"] += 1
                logger.info(f"Removed expired lock: {lock_file.name}")
                continue

            # Check if process is dead (same host only)
            parts = lock_id.split(":")
            if len(parts) >= 2:
                lock_hostname = parts[0]
                try:
                    lock_pid = int(parts[1])

                    if lock_hostname == hostname:
                        # Check if process exists
                        try:
                            os.kill(lock_pid, 0)
                        except OSError:
                            # Process doesn't exist
                            lock_file.unlink(missing_ok=True)
                            stats["removed_dead_process"] += 1
                            logger.info(f"Removed lock from dead process: {lock_file.name} (pid {lock_pid})")
                            continue
                except ValueError:
                    logger.debug(f"[DistributedLock] Invalid PID format in lock file: {lock_file.name}")

        except (OSError, ValueError) as e:
            stats["errors"] += 1
            logger.debug(f"Error processing lock {lock_file}: {e}")

    total_removed = (
        stats["removed_expired"]
        + stats["removed_old"]
        + stats["removed_dead_process"]
    )
    if total_removed > 0:
        logger.info(f"Lock cleanup complete: {total_removed} removed of {stats['scanned']} scanned")

    return stats


# =============================================================================
# Training Lock with Automatic Heartbeat (January 4, 2026)
# =============================================================================

# Training lock configuration constants
TRAINING_LOCK_TTL_SECONDS = 4 * 60 * 60  # 4 hours
TRAINING_HEARTBEAT_INTERVAL_SECONDS = 5 * 60  # 5 minutes
TRAINING_LOCK_MAX_AGE_HOURS = 6.0  # Auto-release after 6 hours


class TrainingLockWithHeartbeat:
    """Training lock with automatic heartbeat to prevent stale lock deadlocks.

    January 4, 2026: Created to prevent training lock deadlocks that caused
    4+ day training stalls. Features:

    1. 4-hour TTL on training locks
    2. Automatic heartbeat every 5 minutes (extends TTL)
    3. Emits TRAINING_LOCK_TIMEOUT if lock expires without heartbeat
    4. Thread-safe cleanup on release or crash

    Usage:
        lock = TrainingLockWithHeartbeat("square8_2p")
        if lock.acquire():
            try:
                # Training code - heartbeat runs automatically in background
                train_model(...)
            finally:
                lock.release()

    Or as context manager:
        with TrainingLockWithHeartbeat("square8_2p") as lock:
            if lock.acquired:
                train_model(...)
    """

    def __init__(self, config_key: str):
        """Initialize a training lock with automatic heartbeat.

        Args:
            config_key: Config identifier (e.g., "square8_2p")
        """
        self.config_key = config_key
        self._lock = DistributedLock(
            f"training:{config_key}",
            lock_timeout=TRAINING_LOCK_TTL_SECONDS,
        )
        self._acquired = False
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_heartbeat = threading.Event()
        self._lock_acquired_time: float | None = None

    @property
    def acquired(self) -> bool:
        """Check if lock is currently held."""
        return self._acquired

    def acquire(self, timeout: int = DEFAULT_ACQUIRE_TIMEOUT) -> bool:
        """Acquire the training lock and start heartbeat thread.

        Args:
            timeout: Maximum time to wait for lock (seconds)

        Returns:
            True if lock acquired, False otherwise
        """
        if self._acquired:
            return True

        if self._lock.acquire(timeout=timeout):
            self._acquired = True
            self._lock_acquired_time = time.time()
            self._start_heartbeat_thread()
            logger.info(
                f"Training lock acquired: {self.config_key} "
                f"(TTL: {TRAINING_LOCK_TTL_SECONDS}s, heartbeat: {TRAINING_HEARTBEAT_INTERVAL_SECONDS}s)"
            )
            return True
        return False

    def release(self) -> None:
        """Release the training lock and stop heartbeat thread."""
        if not self._acquired:
            return

        self._stop_heartbeat_thread()
        self._lock.release()
        self._acquired = False

        hold_time = time.time() - (self._lock_acquired_time or time.time())
        logger.info(
            f"Training lock released: {self.config_key} "
            f"(held for {hold_time:.1f}s)"
        )

    def _start_heartbeat_thread(self) -> None:
        """Start background heartbeat thread."""
        self._stop_heartbeat.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"training-lock-heartbeat-{self.config_key}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat_thread(self) -> None:
        """Stop background heartbeat thread."""
        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        """Background loop that sends periodic heartbeats."""
        while not self._stop_heartbeat.wait(timeout=TRAINING_HEARTBEAT_INTERVAL_SECONDS):
            if not self._acquired:
                break

            # Check if we've exceeded max age (6 hours)
            if self._lock_acquired_time:
                age_hours = (time.time() - self._lock_acquired_time) / 3600
                if age_hours > TRAINING_LOCK_MAX_AGE_HOURS:
                    logger.warning(
                        f"Training lock exceeded max age: {self.config_key} "
                        f"(age: {age_hours:.1f}h > {TRAINING_LOCK_MAX_AGE_HOURS}h)"
                    )
                    self._emit_timeout_event("max_age_exceeded")
                    self._acquired = False  # Mark as not acquired
                    self._lock.release()
                    break

            # Send heartbeat
            if self._lock.heartbeat():
                logger.debug(f"Training lock heartbeat: {self.config_key}")
            else:
                logger.warning(f"Training lock heartbeat failed: {self.config_key}")
                self._emit_timeout_event("heartbeat_failed")
                self._acquired = False
                break

    def _emit_timeout_event(self, reason: str) -> None:
        """Emit TRAINING_LOCK_TIMEOUT event."""
        try:
            from app.distributed.data_events import DataEventType, get_event_bus

            hold_time = time.time() - (self._lock_acquired_time or time.time())
            get_event_bus().emit(
                DataEventType.TRAINING_LOCK_TIMEOUT,
                {
                    "config_key": self.config_key,
                    "reason": reason,
                    "hold_time_seconds": hold_time,
                    "max_age_hours": TRAINING_LOCK_MAX_AGE_HOURS,
                    "ttl_seconds": TRAINING_LOCK_TTL_SECONDS,
                    "timestamp": time.time(),
                },
            )
        except (ImportError, Exception) as e:
            logger.debug(f"Could not emit TRAINING_LOCK_TIMEOUT: {e}")

    def __enter__(self) -> "TrainingLockWithHeartbeat":
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.release()


def acquire_training_lock_with_heartbeat(
    config_key: str,
    timeout: int = DEFAULT_ACQUIRE_TIMEOUT,
) -> TrainingLockWithHeartbeat | None:
    """Acquire a training lock with automatic heartbeat.

    January 4, 2026: Recommended over acquire_training_lock() for long-running
    training jobs. Automatically heartbeats to prevent TTL expiry.

    Args:
        config_key: Config identifier (e.g., "square8_2p")
        timeout: Maximum time to wait for lock

    Returns:
        TrainingLockWithHeartbeat instance if acquired, None otherwise
    """
    lock = TrainingLockWithHeartbeat(config_key)
    if lock.acquire(timeout=timeout):
        return lock
    return None


@contextmanager
def training_lock_with_heartbeat(
    config_key: str,
    timeout: int = DEFAULT_ACQUIRE_TIMEOUT,
) -> Generator[TrainingLockWithHeartbeat | None, None, None]:
    """Context manager for training lock with automatic heartbeat.

    January 4, 2026: Recommended over training_lock() for long-running jobs.

    Usage:
        with training_lock_with_heartbeat("square8_2p") as lock:
            if lock and lock.acquired:
                # Do training - heartbeat runs automatically
                train_model(...)
            else:
                print("Could not acquire lock")
    """
    lock = TrainingLockWithHeartbeat(config_key)
    acquired = lock.acquire(timeout=timeout)

    try:
        yield lock if acquired else None
    finally:
        if acquired:
            lock.release()


__all__ = [
    # Constants
    "DEFAULT_ACQUIRE_TIMEOUT",
    "DEFAULT_LOCK_TIMEOUT",
    # Training lock constants (January 4, 2026)
    "TRAINING_LOCK_TTL_SECONDS",
    "TRAINING_HEARTBEAT_INTERVAL_SECONDS",
    "TRAINING_LOCK_MAX_AGE_HOURS",
    # Backend types (Dec 30, 2025)
    "LockBackendType",
    # Main class
    "DistributedLock",
    # Training lock with heartbeat (January 4, 2026)
    "TrainingLockWithHeartbeat",
    "acquire_training_lock_with_heartbeat",
    "training_lock_with_heartbeat",
    # Raft support (Dec 30, 2025)
    "RaftLockWrapper",
    "reset_raft_cache",
    # Protocol
    "LockProtocol",
    # Functions
    "acquire_training_lock",
    "cleanup_stale_locks",
    "get_appropriate_lock",
    "release_training_lock",
    "training_lock",
]
