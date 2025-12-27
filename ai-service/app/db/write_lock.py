"""Database write lock mechanism for atomic game writes.

This module provides a simple file-based lock mechanism to indicate when a database
is being actively written to. Sync operations should check `is_database_safe_to_sync()`
before copying database files to ensure they don't capture incomplete game data.

The write lock uses a `.writing` file extension alongside the database file:
- `games.db` -> `games.db.writing` (lock file)

When the lock file exists, sync operations should skip the database and try again later.

Usage:
    from app.db.write_lock import DatabaseWriteLock, is_database_safe_to_sync

    # In selfplay runner (writer):
    with DatabaseWriteLock(db_path):
        writer.add_move(...)
        writer.add_move(...)
        writer.finalize(...)

    # In sync daemon (reader):
    if is_database_safe_to_sync(db_path):
        rsync_database(db_path, destination)
    else:
        logger.info(f"Skipping {db_path} - write in progress")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Type

logger = logging.getLogger(__name__)

# Default stale lock timeout (5 minutes)
STALE_LOCK_TIMEOUT_SECONDS = 300.0

# Lock file extension
LOCK_EXTENSION = ".writing"


class DatabaseWriteLock:
    """Context manager for indicating active database writes.

    Creates a lock file while the context is active, and removes it on exit.
    The lock file contains the timestamp when the lock was acquired.

    Args:
        db_path: Path to the database file (or directory for multi-DB locks)
        stale_timeout: Seconds after which a lock is considered stale (default: 300)

    Example:
        with DatabaseWriteLock(Path("data/games/selfplay.db")):
            # Database is marked as being written to
            writer.add_move(...)
            writer.finalize(...)
        # Lock file is removed
    """

    def __init__(
        self,
        db_path: Path | str,
        stale_timeout: float = STALE_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        self.db_path = Path(db_path)
        self.lock_path = self.db_path.parent / f"{self.db_path.name}{LOCK_EXTENSION}"
        self.stale_timeout = stale_timeout
        self._acquired = False

    def __enter__(self) -> "DatabaseWriteLock":
        """Acquire the write lock."""
        self._acquire()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """Release the write lock."""
        self._release()
        return False  # Don't suppress exceptions

    def _acquire(self) -> None:
        """Create the lock file with current timestamp."""
        try:
            # Clean up stale lock if exists
            if self.lock_path.exists():
                lock_age = time.time() - self.lock_path.stat().st_mtime
                if lock_age > self.stale_timeout:
                    logger.warning(
                        f"Removing stale write lock: {self.lock_path} "
                        f"(age: {lock_age:.0f}s, timeout: {self.stale_timeout}s)"
                    )
                    self.lock_path.unlink(missing_ok=True)

            # Create lock file with timestamp
            self.lock_path.write_text(str(time.time()))
            self._acquired = True
            logger.debug(f"Acquired write lock: {self.lock_path}")
        except OSError as e:
            logger.error(f"Failed to acquire write lock {self.lock_path}: {e}")
            # Don't raise - allow write to proceed without lock
            # Better to have potential sync issues than block writes

    def _release(self) -> None:
        """Remove the lock file."""
        if not self._acquired:
            return

        try:
            self.lock_path.unlink(missing_ok=True)
            self._acquired = False
            logger.debug(f"Released write lock: {self.lock_path}")
        except OSError as e:
            logger.error(f"Failed to release write lock {self.lock_path}: {e}")


def is_database_safe_to_sync(
    db_path: Path | str,
    stale_timeout: float = STALE_LOCK_TIMEOUT_SECONDS,
) -> bool:
    """Check if a database is safe to sync (no active writes).

    Args:
        db_path: Path to the database file
        stale_timeout: Seconds after which a lock is considered stale

    Returns:
        True if the database can be safely synced (no active lock or stale lock)
        False if there's an active write in progress
    """
    db_path = Path(db_path)
    lock_path = db_path.parent / f"{db_path.name}{LOCK_EXTENSION}"

    if not lock_path.exists():
        return True

    try:
        lock_age = time.time() - lock_path.stat().st_mtime
        if lock_age > stale_timeout:
            # Lock is stale - consider safe to sync
            logger.debug(
                f"Write lock is stale ({lock_age:.0f}s > {stale_timeout}s): {lock_path}"
            )
            return True

        # Active lock - not safe to sync
        logger.debug(
            f"Active write lock found ({lock_age:.0f}s old): {lock_path}"
        )
        return False
    except OSError as e:
        logger.warning(f"Error checking write lock {lock_path}: {e}")
        # If we can't check, assume safe
        return True


def get_active_write_locks(
    games_dir: Path | str,
    stale_timeout: float = STALE_LOCK_TIMEOUT_SECONDS,
) -> list[Path]:
    """Get all active (non-stale) write locks in a directory.

    Args:
        games_dir: Directory to search for lock files
        stale_timeout: Seconds after which a lock is considered stale

    Returns:
        List of paths to databases with active write locks
    """
    games_dir = Path(games_dir)
    active_locks = []

    for lock_file in games_dir.glob(f"*{LOCK_EXTENSION}"):
        try:
            lock_age = time.time() - lock_file.stat().st_mtime
            if lock_age <= stale_timeout:
                # Extract database path from lock path
                db_name = lock_file.name[: -len(LOCK_EXTENSION)]
                db_path = lock_file.parent / db_name
                active_locks.append(db_path)
        except OSError:
            continue

    return active_locks


def cleanup_stale_locks(
    games_dir: Path | str,
    stale_timeout: float = STALE_LOCK_TIMEOUT_SECONDS,
) -> int:
    """Remove stale lock files from a directory.

    Args:
        games_dir: Directory to search for lock files
        stale_timeout: Seconds after which a lock is considered stale

    Returns:
        Number of stale locks removed
    """
    games_dir = Path(games_dir)
    removed = 0

    for lock_file in games_dir.glob(f"*{LOCK_EXTENSION}"):
        try:
            lock_age = time.time() - lock_file.stat().st_mtime
            if lock_age > stale_timeout:
                lock_file.unlink(missing_ok=True)
                removed += 1
                logger.info(f"Removed stale write lock: {lock_file}")
        except OSError as e:
            logger.warning(f"Error removing stale lock {lock_file}: {e}")

    return removed
