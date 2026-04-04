#!/usr/bin/env python3
"""Sync Mutex for coordinating rsync and file transfer operations.

This module provides mutexes to prevent multiple processes from running
concurrent sync operations to/from the same host, which can cause:
- Network congestion and bandwidth contention
- File corruption on concurrent writes
- SSH connection limits being hit

Features:
- Per-host mutex (prevents concurrent syncs to same host)
- Global sync queue (limits total concurrent syncs)
- SQLite-backed for cross-process coordination
- Automatic cleanup of stale locks

Usage:
    from app.coordination.sync_mutex import (
        acquire_sync_lock,
        release_sync_lock,
        sync_lock,  # context manager
    )

    # Option 1: Context manager
    with sync_lock("host-1", "rsync"):
        subprocess.run(["rsync", "-avz", ...])

    # Option 2: Manual acquire/release
    if acquire_sync_lock("host-1", "rsync"):
        try:
            subprocess.run(["rsync", "-avz", ...])
        finally:
            release_sync_lock("host-1")
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import socket
import sqlite3
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_user_coordination_dir() -> Path:
    """Get user-specific coordination directory to avoid permission conflicts.

    This prevents conflicts when P2P orchestrator (root) and master_loop (ubuntu)
    both try to access the same coordination files.
    """
    # Allow override via environment
    custom_dir = os.environ.get("RINGRIFT_COORDINATION_DIR")
    if custom_dir:
        return Path(custom_dir)

    # Use XDG_RUNTIME_DIR if available (properly permissioned)
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        return Path(xdg_runtime) / "ringrift_coordination"

    # Fall back to user-specific /tmp directory
    try:
        uid = os.getuid()
    except AttributeError:
        uid = 0  # Windows

    if uid == 0:
        return Path("/tmp/ringrift_coordination")
    else:
        return Path(f"/tmp/ringrift_coordination_{uid}")


# Default database location - user-specific to avoid permission conflicts
DEFAULT_SYNC_DB = _get_user_coordination_dir() / "sync_mutex.db"

# Import centralized timeout thresholds
try:
    from app.config.thresholds import SQLITE_BUSY_TIMEOUT_MS, SQLITE_TIMEOUT
except ImportError:
    SQLITE_BUSY_TIMEOUT_MS = 10000
    SQLITE_TIMEOUT = 30

# Import centralized defaults (December 2025)
try:
    from app.config.coordination_defaults import HeartbeatDefaults, SyncDefaults
    LOCK_TIMEOUT_SECONDS = SyncDefaults.LOCK_TIMEOUT
    MAX_CONCURRENT_SYNCS_PER_HOST = SyncDefaults.MAX_CONCURRENT_PER_HOST
    MAX_GLOBAL_CONCURRENT_SYNCS = SyncDefaults.MAX_CONCURRENT_CLUSTER
    HEARTBEAT_INTERVAL = HeartbeatDefaults.INTERVAL
except ImportError:
    # Fallback for standalone use
    LOCK_TIMEOUT_SECONDS = 120
    MAX_CONCURRENT_SYNCS_PER_HOST = 1
    MAX_GLOBAL_CONCURRENT_SYNCS = 5
    HEARTBEAT_INTERVAL = 30

# Non-configurable constants
LOCK_POLL_INTERVAL = 0.5  # Polling interval when waiting for lock
CRASH_DETECTION_THRESHOLD = 60  # Consider process crashed if no heartbeat for this long
RELEASE_RETRY_ATTEMPTS = 5
RELEASE_RETRY_INTERVAL = 0.05


@dataclass
class SyncLockInfo:
    """Information about an active sync lock."""

    host: str
    operation: str
    holder_pid: int
    holder_hostname: str
    acquired_at: float
    timeout_at: float

    @property
    def age_seconds(self) -> float:
        return time.time() - self.acquired_at

    @property
    def is_expired(self) -> bool:
        return time.time() > self.timeout_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "operation": self.operation,
            "holder_pid": self.holder_pid,
            "holder_hostname": self.holder_hostname,
            "acquired_at": datetime.fromtimestamp(self.acquired_at).isoformat(),
            "age_seconds": round(self.age_seconds, 1),
            "is_expired": self.is_expired,
        }


class SyncMutex:
    """SQLite-backed mutex for synchronizing file transfer operations."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_SYNC_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Track all connections for cleanup on process exit (Dec 2025)
        self._all_connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=float(SQLITE_TIMEOUT))
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute(f'PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}')
            conn.execute('PRAGMA synchronous=NORMAL')
            self._local.conn = conn
            # Track connection for cleanup on process exit (Dec 2025)
            with self._connections_lock:
                self._all_connections.add(conn)
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript('''
            -- Sync locks table
            CREATE TABLE IF NOT EXISTS sync_locks (
                lock_id INTEGER PRIMARY KEY AUTOINCREMENT,
                host TEXT NOT NULL,
                operation TEXT NOT NULL,
                holder_pid INTEGER NOT NULL,
                holder_hostname TEXT NOT NULL,
                acquired_at REAL NOT NULL,
                timeout_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}',
                last_heartbeat REAL DEFAULT 0
            );

            -- Unique constraint: only one lock per host
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sync_locks_host
                ON sync_locks(host);

            -- Index for cleanup queries
            CREATE INDEX IF NOT EXISTS idx_sync_locks_timeout
                ON sync_locks(timeout_at);
        ''')

        # Add last_heartbeat column if it doesn't exist (migration)
        try:
            conn.execute("SELECT last_heartbeat FROM sync_locks LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE sync_locks ADD COLUMN last_heartbeat REAL DEFAULT 0")

        conn.commit()

    def acquire(
        self,
        host: str,
        operation: str = "rsync",
        timeout: float = LOCK_TIMEOUT_SECONDS,
        wait: bool = False,
        wait_timeout: float = 60.0,
    ) -> bool:
        """Acquire a sync lock for a host.

        Args:
            host: Target host for sync operation
            operation: Type of operation (rsync, scp, etc.)
            timeout: Lock expiration in seconds
            wait: If True, wait for lock to become available
            wait_timeout: Max time to wait for lock

        Returns:
            True if lock acquired, False otherwise
        """
        conn = self._get_connection()
        now = time.time()
        hostname = socket.gethostname()
        pid = os.getpid()

        # Clean up expired locks first
        self._cleanup_expired(conn)

        # Check global concurrent sync limit
        cursor = conn.execute('SELECT COUNT(*) FROM sync_locks')
        active_count = cursor.fetchone()[0]
        if active_count >= MAX_GLOBAL_CONCURRENT_SYNCS and not wait:
            return False
            # Will try to wait below

        start_time = time.time()
        while True:
            try:
                # Try to insert lock
                conn.execute(
                    '''INSERT INTO sync_locks
                       (host, operation, holder_pid, holder_hostname, acquired_at, timeout_at)
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    (host, operation, pid, hostname, now, now + timeout)
                )
                conn.commit()
                return True

            except sqlite3.IntegrityError:
                # Lock already held - check if we should wait
                if not wait:
                    return False

                # Check if wait timeout exceeded
                if time.time() - start_time > wait_timeout:
                    # P11-HIGH-1: Emit LOCK_TIMEOUT event for monitoring
                    wait_duration = time.time() - start_time
                    logger.warning(
                        f"[SyncMutex] Lock timeout for {host}/{operation}: "
                        f"waited {wait_duration:.1f}s (limit={wait_timeout}s)"
                    )
                    self._emit_lock_timeout(
                        host=host,
                        operation=operation,
                        wait_duration=wait_duration,
                        wait_timeout=wait_timeout,
                    )
                    return False

                # Wait and retry
                time.sleep(LOCK_POLL_INTERVAL)
                self._cleanup_expired(conn)

            except sqlite3.Error as e:
                logger.error(f"Database error acquiring lock for {host}: {e}")
                return False

    def release(self, host: str) -> bool:
        """Release a sync lock.

        Args:
            host: Host to release lock for

        Returns:
            True if lock was released
        """
        conn = self._get_connection()
        hostname = socket.gethostname()
        pid = os.getpid()
        original_busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        release_busy_timeout_ms = int(RELEASE_RETRY_INTERVAL * 1000)
        conn.execute(f"PRAGMA busy_timeout={release_busy_timeout_ms}")

        try:
            # Only release our own locks.
            # Concurrent thread/process activity can transiently lock the SQLite WAL;
            # retry a few times so callers don't surface thread exceptions on cleanup.
            for attempt in range(RELEASE_RETRY_ATTEMPTS):
                try:
                    cursor = conn.execute(
                        '''DELETE FROM sync_locks
                           WHERE host = ? AND holder_pid = ? AND holder_hostname = ?''',
                        (host, pid, hostname)
                    )
                    conn.commit()
                    return cursor.rowcount > 0
                except sqlite3.OperationalError as e:
                    if "locked" not in str(e).lower():
                        logger.error(f"Database error releasing lock for {host}: {e}")
                        return False

                    try:
                        conn.rollback()
                    except sqlite3.Error:
                        pass

                    if attempt == RELEASE_RETRY_ATTEMPTS - 1:
                        logger.warning(
                            f"Database remained locked releasing sync lock for {host} "
                            f"after {RELEASE_RETRY_ATTEMPTS} attempts"
                        )
                        return False

                    time.sleep(RELEASE_RETRY_INTERVAL)
        finally:
            conn.execute(f"PRAGMA busy_timeout={original_busy_timeout}")

        return False

    def release_all_for_process(self) -> int:
        """Release all locks held by this process.

        Returns:
            Number of locks released
        """
        conn = self._get_connection()
        hostname = socket.gethostname()
        pid = os.getpid()

        cursor = conn.execute(
            'DELETE FROM sync_locks WHERE holder_pid = ? AND holder_hostname = ?',
            (pid, hostname)
        )
        conn.commit()
        return cursor.rowcount

    def is_locked(self, host: str) -> bool:
        """Check if a host has an active sync lock."""
        conn = self._get_connection()
        self._cleanup_expired(conn)

        cursor = conn.execute(
            'SELECT 1 FROM sync_locks WHERE host = ?', (host,)
        )
        return cursor.fetchone() is not None

    def _emit_lock_timeout(
        self,
        host: str,
        operation: str,
        wait_duration: float,
        wait_timeout: float,
    ) -> None:
        """Emit a LOCK_TIMEOUT event for monitoring.

        P11-HIGH-1 (Dec 2025): This allows the feedback system to detect
        when sync operations are being blocked by contention, potentially
        indicating cluster bottlenecks or dead locks.
        """
        try:
            from app.coordination.event_router import publish_sync, DataEventType

            publish_sync(
                DataEventType.LOCK_TIMEOUT,
                payload={
                    "host": host,
                    "operation": operation,
                    "wait_duration": wait_duration,
                    "wait_timeout": wait_timeout,
                    "holder_hostname": socket.gethostname(),
                    "holder_pid": os.getpid(),
                },
                source="SyncMutex",
            )
        except Exception as e:
            logger.debug(f"Failed to emit LOCK_TIMEOUT event: {e}")

    def get_lock_info(self, host: str) -> SyncLockInfo | None:
        """Get information about a lock."""
        conn = self._get_connection()
        cursor = conn.execute(
            '''SELECT host, operation, holder_pid, holder_hostname, acquired_at, timeout_at
               FROM sync_locks WHERE host = ?''',
            (host,)
        )
        row = cursor.fetchone()
        if row:
            return SyncLockInfo(
                host=row["host"],
                operation=row["operation"],
                holder_pid=row["holder_pid"],
                holder_hostname=row["holder_hostname"],
                acquired_at=row["acquired_at"],
                timeout_at=row["timeout_at"],
            )
        return None

    def get_all_locks(self) -> list[SyncLockInfo]:
        """Get all active locks."""
        conn = self._get_connection()
        self._cleanup_expired(conn)

        cursor = conn.execute(
            '''SELECT host, operation, holder_pid, holder_hostname, acquired_at, timeout_at
               FROM sync_locks ORDER BY acquired_at'''
        )
        return [
            SyncLockInfo(
                host=row["host"],
                operation=row["operation"],
                holder_pid=row["holder_pid"],
                holder_hostname=row["holder_hostname"],
                acquired_at=row["acquired_at"],
                timeout_at=row["timeout_at"],
            )
            for row in cursor.fetchall()
        ]

    def _cleanup_expired(self, conn: sqlite3.Connection) -> int:
        """Remove expired locks and crashed process locks."""
        try:
            # Remove timed-out locks
            cursor = conn.execute(
                'DELETE FROM sync_locks WHERE timeout_at < ?', (time.time(),)
            )
            expired_count = cursor.rowcount

            # Also clean up crashed locks (does its own commit)
            crashed_count = self.cleanup_crashed_locks()

            if expired_count > 0:
                conn.commit()
            return expired_count + crashed_count
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                logger.debug("Skipping expired lock cleanup due to transient SQLite lock")
                return 0
            raise

    def force_release(self, host: str) -> bool:
        """Force release a lock (admin use only).

        Args:
            host: Host to release lock for

        Returns:
            True if lock was released
        """
        conn = self._get_connection()
        cursor = conn.execute(
            'DELETE FROM sync_locks WHERE host = ?', (host,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def heartbeat(self, host: str) -> bool:
        """Update heartbeat for a lock to indicate the process is alive.

        Should be called periodically (every HEARTBEAT_INTERVAL seconds)
        for long-running sync operations.

        Args:
            host: Host whose lock to update

        Returns:
            True if heartbeat was recorded
        """
        conn = self._get_connection()
        now = time.time()
        pid = os.getpid()
        hostname = socket.gethostname()

        # Only update if we own the lock
        cursor = conn.execute(
            '''UPDATE sync_locks
               SET last_heartbeat = ?
               WHERE host = ? AND holder_pid = ? AND holder_hostname = ?''',
            (now, host, pid, hostname)
        )
        conn.commit()
        return cursor.rowcount > 0

    def _is_process_alive(self, pid: int, hostname: str) -> bool:
        """Check if a process is still alive.

        Only works for processes on the same host. For remote hosts,
        relies on heartbeat mechanism.
        """
        current_hostname = socket.gethostname()

        # For remote hosts, we can't check PID - rely on heartbeat
        if hostname != current_hostname:
            return True  # Assume alive, let heartbeat timeout handle it

        # For local processes, check if PID exists
        try:
            os.kill(pid, 0)  # Signal 0 just checks if process exists
            return True
        except OSError:
            return False

    def cleanup_crashed_locks(self) -> int:
        """Clean up locks from crashed processes.

        Releases locks where:
        1. The process is dead (local hosts only)
        2. No heartbeat received for CRASH_DETECTION_THRESHOLD seconds

        Returns:
            Number of locks cleaned up
        """
        conn = self._get_connection()
        now = time.time()
        cleaned = 0

        try:
            # Get all active locks
            cursor = conn.execute(
                '''SELECT lock_id, host, holder_pid, holder_hostname,
                          last_heartbeat, acquired_at
                   FROM sync_locks'''
            )
            locks = cursor.fetchall()

            for lock in locks:
                should_release = False
                reason = ""

                lock_id = lock["lock_id"]
                host = lock["host"]
                pid = lock["holder_pid"]
                hostname = lock["holder_hostname"]
                last_heartbeat = lock["last_heartbeat"] or lock["acquired_at"]

                # Check if process is dead (local only)
                if not self._is_process_alive(pid, hostname):
                    should_release = True
                    reason = f"process {pid} on {hostname} is dead"

                # Check heartbeat timeout
                elif now - last_heartbeat > CRASH_DETECTION_THRESHOLD:
                    should_release = True
                    reason = f"no heartbeat for {now - last_heartbeat:.0f}s"

                if should_release:
                    conn.execute('DELETE FROM sync_locks WHERE lock_id = ?', (lock_id,))
                    cleaned += 1
                    logger.info(f"Released stale lock for {host}: {reason}")

            if cleaned > 0:
                conn.commit()

            return cleaned
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                logger.debug("Skipping crashed lock cleanup due to transient SQLite lock")
                return cleaned
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get sync mutex statistics."""
        conn = self._get_connection()
        self._cleanup_expired(conn)

        cursor = conn.execute('SELECT COUNT(*) FROM sync_locks')
        active_count = cursor.fetchone()[0]

        cursor = conn.execute(
            '''SELECT host, operation, holder_hostname,
                      (? - acquired_at) as age_seconds
               FROM sync_locks ORDER BY acquired_at''',
            (time.time(),)
        )
        locks = [
            {
                "host": row["host"],
                "operation": row["operation"],
                "holder": row["holder_hostname"],
                "age_seconds": round(row["age_seconds"], 1),
            }
            for row in cursor.fetchall()
        ]

        return {
            "active_locks": active_count,
            "max_per_host": MAX_CONCURRENT_SYNCS_PER_HOST,
            "max_global": MAX_GLOBAL_CONCURRENT_SYNCS,
            "locks": locks,
        }

    def close(self) -> None:
        """Close the current thread's database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            conn = self._local.conn
            with self._connections_lock:
                self._all_connections.discard(conn)
            conn.close()
            self._local.conn = None

    def close_all(self) -> int:
        """Close all tracked database connections (Dec 2025).

        Call this on process exit to ensure no connections are leaked.
        Returns the number of connections closed.
        """
        closed = 0
        with self._connections_lock:
            for conn in list(self._all_connections):
                try:
                    conn.close()
                    closed += 1
                except (sqlite3.Error, sqlite3.ProgrammingError):
                    # Connection may already be closed
                    pass
            self._all_connections.clear()
        # Also clear thread-local reference if present
        if hasattr(self._local, "conn"):
            self._local.conn = None
        return closed


# Global singleton instance
_sync_mutex: SyncMutex | None = None
_mutex_lock = threading.RLock()


def get_sync_mutex(db_path: Path | None = None) -> SyncMutex:
    """Get the global sync mutex singleton."""
    global _sync_mutex
    with _mutex_lock:
        if _sync_mutex is None:
            _sync_mutex = SyncMutex(db_path)
        return _sync_mutex


def reset_sync_mutex() -> None:
    """Reset the global sync mutex (for testing)."""
    global _sync_mutex
    with _mutex_lock:
        if _sync_mutex is not None:
            _sync_mutex.close_all()  # Close ALL connections, not just current thread
        _sync_mutex = None


def _cleanup_on_exit() -> None:
    """Cleanup handler for process exit (Dec 2025)."""
    global _sync_mutex
    if _sync_mutex is not None:
        closed = _sync_mutex.close_all()
        if closed > 0:
            logger.debug(f"[SyncMutex] Closed {closed} connection(s) on exit")


# Register cleanup handler
atexit.register(_cleanup_on_exit)


# Convenience functions


def acquire_sync_lock(
    host: str,
    operation: str = "rsync",
    timeout: float = LOCK_TIMEOUT_SECONDS,
    wait: bool = False,
    wait_timeout: float = 60.0,
) -> bool:
    """Acquire a sync lock for a host."""
    return get_sync_mutex().acquire(host, operation, timeout, wait, wait_timeout)


def release_sync_lock(host: str) -> bool:
    """Release a sync lock for a host."""
    return get_sync_mutex().release(host)


def is_sync_locked(host: str) -> bool:
    """Check if a host has an active sync lock."""
    return get_sync_mutex().is_locked(host)


def get_sync_stats() -> dict[str, Any]:
    """Get sync mutex statistics."""
    return get_sync_mutex().get_stats()


@contextmanager
def sync_lock(
    host: str,
    operation: str = "rsync",
    timeout: float = LOCK_TIMEOUT_SECONDS,
    wait: bool = True,
    wait_timeout: float = 120.0,
) -> Generator[bool]:
    """Context manager for sync operations.

    Usage:
        with sync_lock("host-1", "rsync") as acquired:
            if acquired:
                subprocess.run(["rsync", ...])
            else:
                print("Could not acquire lock")
    """
    mutex = get_sync_mutex()
    acquired = mutex.acquire(host, operation, timeout, wait, wait_timeout)
    try:
        yield acquired
    finally:
        if acquired:
            mutex.release(host)


@contextmanager
def sync_lock_required(
    host: str,
    operation: str = "rsync",
    timeout: float = LOCK_TIMEOUT_SECONDS,
    wait_timeout: float = 120.0,
) -> Generator[None]:
    """Context manager that raises if lock cannot be acquired.

    Usage:
        with sync_lock_required("host-1"):
            subprocess.run(["rsync", ...])
    """
    mutex = get_sync_mutex()
    if not mutex.acquire(host, operation, timeout, wait=True, wait_timeout=wait_timeout):
        raise TimeoutError(f"Could not acquire sync lock for {host} within {wait_timeout}s")
    try:
        yield
    finally:
        mutex.release(host)


def sync_heartbeat(host: str) -> bool:
    """Update heartbeat for a sync lock."""
    return get_sync_mutex().heartbeat(host)


def cleanup_crashed_sync_locks() -> int:
    """Clean up locks from crashed processes."""
    return get_sync_mutex().cleanup_crashed_locks()


# Command-line interface

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Sync mutex management")
    parser.add_argument("--status", action="store_true", help="Show mutex status")
    parser.add_argument("--acquire", type=str, help="Acquire lock for host")
    parser.add_argument("--release", type=str, help="Release lock for host")
    parser.add_argument("--force-release", type=str, help="Force release lock")
    parser.add_argument("--release-all", action="store_true", help="Release all locks for this process")
    parser.add_argument("--operation", type=str, default="rsync", help="Operation type")
    parser.add_argument("--timeout", type=int, default=300, help="Lock timeout in seconds")
    args = parser.parse_args()

    mutex = get_sync_mutex()

    if args.status:
        print(json.dumps(mutex.get_stats(), indent=2))

    elif args.acquire:
        if mutex.acquire(args.acquire, args.operation, args.timeout):
            print(f"Acquired lock for {args.acquire}")
            print("Press Ctrl+C to release and exit...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                mutex.release(args.acquire)
                print(f"\nReleased lock for {args.acquire}")
        else:
            print(f"Could not acquire lock for {args.acquire}")
            info = mutex.get_lock_info(args.acquire)
            if info:
                print(f"Lock held by: {info.holder_hostname}:{info.holder_pid}")

    elif args.release:
        if mutex.release(args.release):
            print(f"Released lock for {args.release}")
        else:
            print(f"No lock held for {args.release} (or not owned by this process)")

    elif args.force_release:
        if mutex.force_release(args.force_release):
            print(f"Force released lock for {args.force_release}")
        else:
            print(f"No lock found for {args.force_release}")

    elif args.release_all:
        count = mutex.release_all_for_process()
        print(f"Released {count} locks")

    else:
        parser.print_help()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Constants
    "LOCK_TIMEOUT_SECONDS",
    "MAX_CONCURRENT_SYNCS_PER_HOST",
    "MAX_GLOBAL_CONCURRENT_SYNCS",
    # Data classes
    "SyncLockInfo",
    # Main class
    "SyncMutex",
    "acquire_sync_lock",
    "cleanup_crashed_sync_locks",
    # Functions
    "get_sync_mutex",
    "get_sync_stats",
    "is_sync_locked",
    "release_sync_lock",
    "reset_sync_mutex",
    "sync_heartbeat",
    "sync_lock",
    "sync_lock_required",
]
