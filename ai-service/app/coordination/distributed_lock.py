"""Distributed Locking for Training Coordination.

.. deprecated:: December 2025
    This module has been consolidated into ``app.coordination.core_utils``.
    Import from core_utils for new code. This module remains for backward
    compatibility and will be removed in Q2 2026.

    Migration:
        # Old import (deprecated)
        from app.coordination.distributed_lock import DistributedLock

        # New import (preferred)
        from app.coordination.core_utils import DistributedLock

Provides reliable distributed locks for coordinating training across
multiple nodes. Uses Redis when available with automatic fallback
to file-based locking.

Features:
- Redis-based distributed locks (preferred)
- Automatic lock expiry to prevent deadlocks
- File-based fallback for single-node or no-Redis scenarios
- Lock timeout and retry support
- Context manager interface

When to Use This (December 2025):
- **DistributedLock** (this module): Use for cross-node/cross-process
  coordination. Uses Redis or file locks for true distributed locking.
  Suitable for training locks, model registry, resource allocation.

- **LockHierarchy** (app.core.locking): Use for in-process async coordination
  with deadlock prevention through lock ordering. Suitable for coordinating
  async tasks within a single Python process.

Usage:
    from app.coordination.distributed_lock import DistributedLock

    lock = DistributedLock("training:square8_2p")

    # Context manager usage
    with lock:
        # Training code here
        pass

    # Or explicit acquire/release
    if lock.acquire(timeout=30):
        try:
            # Training code
        finally:
            lock.release()
"""

from __future__ import annotations

import fcntl
import logging
import os
import socket
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path

from app.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)


# =============================================================================
# Unified Lock Protocol (December 2025)
# =============================================================================

from typing import Generator, Protocol, runtime_checkable


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
    """Distributed lock with Redis + file-based fallback.

    Automatically selects the best available backend:
    1. Redis (if available and reachable)
    2. File-based locking (fallback)
    """

    def __init__(
        self,
        name: str,
        lock_timeout: int = DEFAULT_LOCK_TIMEOUT,
        use_redis: bool = True,
    ):
        """Initialize a distributed lock.

        Args:
            name: Unique lock name (e.g., "training:square8_2p")
            lock_timeout: Maximum time to hold lock (seconds)
            use_redis: Whether to try Redis (True) or force file-based (False)
        """
        self.name = name
        self.lock_timeout = lock_timeout
        self._lock_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._redis_client: redis.Redis | None = None
        self._file_fd: int | None = None
        self._acquired = False
        self._use_redis = use_redis and HAS_REDIS

        # Try to connect to Redis if available
        if self._use_redis:
            try:
                self._redis_client = redis.Redis.from_url(REDIS_URL, socket_timeout=5)
                self._redis_client.ping()
                logger.debug(f"Using Redis for lock: {name}")
            except Exception as e:
                logger.debug(f"Redis not available, using file lock: {e}")
                self._redis_client = None

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
            if self._redis_client is not None:
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
            if self._redis_client is not None:
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
        if self._redis_client is not None:
            return self._redis_client.exists(f"lock:{self.name}") > 0
        else:
            return self._is_file_locked()

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
        except Exception as e:
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
        except Exception as e:
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
                except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
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

        except Exception as e:
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


__all__ = [
    "DEFAULT_ACQUIRE_TIMEOUT",
    # Constants
    "DEFAULT_LOCK_TIMEOUT",
    # Main class
    "DistributedLock",
    # Protocol
    "LockProtocol",
    "acquire_training_lock",
    # Functions
    "cleanup_stale_locks",
    "get_appropriate_lock",
    "release_training_lock",
    "training_lock",
]
