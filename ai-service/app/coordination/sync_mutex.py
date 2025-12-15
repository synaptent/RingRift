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
            release_sync_lock("host-1", "rsync")
"""

from __future__ import annotations

import os
import socket
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

# Default database location
DEFAULT_SYNC_DB = Path("/tmp/ringrift_coordination/sync_mutex.db")

# Lock configuration
LOCK_TIMEOUT_SECONDS = 300  # Locks auto-expire after 5 minutes
MAX_CONCURRENT_SYNCS_PER_HOST = 1  # Only 1 sync per host at a time
MAX_GLOBAL_CONCURRENT_SYNCS = 5  # Max total concurrent syncs across cluster
LOCK_POLL_INTERVAL = 0.5  # Polling interval when waiting for lock


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

    def to_dict(self) -> Dict[str, Any]:
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

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_SYNC_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute('PRAGMA journal_mode=WAL')
            self._local.conn.execute('PRAGMA busy_timeout=10000')
            self._local.conn.execute('PRAGMA synchronous=NORMAL')
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
                metadata TEXT DEFAULT '{}'
            );

            -- Unique constraint: only one lock per host
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sync_locks_host
                ON sync_locks(host);

            -- Index for cleanup queries
            CREATE INDEX IF NOT EXISTS idx_sync_locks_timeout
                ON sync_locks(timeout_at);
        ''')
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
        if active_count >= MAX_GLOBAL_CONCURRENT_SYNCS:
            if not wait:
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
                    return False

                # Wait and retry
                time.sleep(LOCK_POLL_INTERVAL)
                self._cleanup_expired(conn)

            except sqlite3.Error as e:
                print(f"[SyncMutex] Database error: {e}")
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

        # Only release our own locks
        cursor = conn.execute(
            '''DELETE FROM sync_locks
               WHERE host = ? AND holder_pid = ? AND holder_hostname = ?''',
            (host, pid, hostname)
        )
        conn.commit()
        return cursor.rowcount > 0

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

    def get_lock_info(self, host: str) -> Optional[SyncLockInfo]:
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

    def get_all_locks(self) -> List[SyncLockInfo]:
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
        """Remove expired locks."""
        cursor = conn.execute(
            'DELETE FROM sync_locks WHERE timeout_at < ?', (time.time(),)
        )
        if cursor.rowcount > 0:
            conn.commit()
        return cursor.rowcount

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

    def get_stats(self) -> Dict[str, Any]:
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
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Global singleton instance
_sync_mutex: Optional[SyncMutex] = None
_mutex_lock = threading.Lock()


def get_sync_mutex(db_path: Optional[Path] = None) -> SyncMutex:
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
            _sync_mutex.close()
        _sync_mutex = None


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


def get_sync_stats() -> Dict[str, Any]:
    """Get sync mutex statistics."""
    return get_sync_mutex().get_stats()


@contextmanager
def sync_lock(
    host: str,
    operation: str = "rsync",
    timeout: float = LOCK_TIMEOUT_SECONDS,
    wait: bool = True,
    wait_timeout: float = 120.0,
) -> Generator[bool, None, None]:
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
) -> Generator[None, None, None]:
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
