"""Cross-process export coordination via SQLite advisory locks.

Limits concurrent NPZ exports across P2P and master_loop processes.
Both processes share only SQLite (events don't cross process boundaries),
so this uses a lock table in export_daemon_state.db to coordinate.

February 2026: Created to prevent I/O contention from 7 independent
export spawning paths competing for disk, blocking P2P event loop.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# Default max concurrent exports (configurable via env var)
_DEFAULT_MAX_CONCURRENT = 2


class ExportSlotUnavailable(Exception):
    """Raised when no export slots are available."""


class ExportCoordinator:
    """SQLite-backed coordinator for limiting concurrent exports across processes.

    Uses a lock table in the shared export_daemon_state.db. Stale locks from
    crashed processes are cleaned up by checking if the PID is still running.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        max_concurrent: int | None = None,
    ):
        if db_path is None:
            db_path = Path("data/export_daemon_state.db")
        self.db_path = Path(db_path)
        self.max_concurrent = max_concurrent or int(
            os.environ.get("RINGRIFT_MAX_CONCURRENT_EXPORTS", str(_DEFAULT_MAX_CONCURRENT))
        )
        self._initialized = False

    def _ensure_table(self, conn: sqlite3.Connection) -> None:
        """Create the export_locks table if it doesn't exist."""
        if self._initialized:
            return
        conn.execute("""
            CREATE TABLE IF NOT EXISTS export_locks (
                config_key TEXT NOT NULL,
                pid INTEGER NOT NULL,
                started_at REAL NOT NULL,
                hostname TEXT DEFAULT '',
                PRIMARY KEY (config_key, pid)
            )
        """)
        conn.commit()
        self._initialized = True

    def _clean_stale_locks(self, conn: sqlite3.Connection) -> int:
        """Remove locks held by processes that are no longer running.

        Returns the number of stale locks cleaned.
        """
        hostname = _get_hostname()
        rows = conn.execute(
            "SELECT config_key, pid, hostname FROM export_locks"
        ).fetchall()

        cleaned = 0
        for config_key, pid, lock_hostname in rows:
            # Only check PIDs on the same host
            if lock_hostname and lock_hostname != hostname:
                # Cross-host lock — check if it's very old (>1 hour stale)
                row = conn.execute(
                    "SELECT started_at FROM export_locks WHERE config_key=? AND pid=?",
                    (config_key, pid),
                ).fetchone()
                if row and (time.time() - row[0]) > 3600:
                    conn.execute(
                        "DELETE FROM export_locks WHERE config_key=? AND pid=?",
                        (config_key, pid),
                    )
                    cleaned += 1
                continue

            if not _is_pid_alive(pid):
                conn.execute(
                    "DELETE FROM export_locks WHERE config_key=? AND pid=?",
                    (config_key, pid),
                )
                cleaned += 1

        if cleaned:
            conn.commit()
            logger.info(f"[ExportCoordinator] Cleaned {cleaned} stale export lock(s)")
        return cleaned

    def try_acquire(self, config_key: str, pid: int | None = None) -> bool:
        """Try to acquire an export slot. Returns True if slot acquired."""
        if pid is None:
            pid = os.getpid()

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                self._ensure_table(conn)
                self._clean_stale_locks(conn)

                # Count active locks
                count = conn.execute(
                    "SELECT COUNT(*) FROM export_locks"
                ).fetchone()[0]

                if count >= self.max_concurrent:
                    logger.info(
                        f"[ExportCoordinator] Slot denied for {config_key} "
                        f"(pid={pid}): {count}/{self.max_concurrent} slots in use"
                    )
                    return False

                # Acquire slot
                conn.execute(
                    "INSERT OR REPLACE INTO export_locks (config_key, pid, started_at, hostname) "
                    "VALUES (?, ?, ?, ?)",
                    (config_key, pid, time.time(), _get_hostname()),
                )
                conn.commit()
                logger.info(
                    f"[ExportCoordinator] Slot acquired for {config_key} "
                    f"(pid={pid}): {count + 1}/{self.max_concurrent} slots in use"
                )
                return True
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as e:
            logger.warning(f"[ExportCoordinator] DB error in try_acquire: {e}")
            # Fail open — allow the export if the coordinator itself is broken
            return True

    def release(self, config_key: str, pid: int | None = None) -> None:
        """Release an export slot."""
        if pid is None:
            pid = os.getpid()

        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                self._ensure_table(conn)
                conn.execute(
                    "DELETE FROM export_locks WHERE config_key=? AND pid=?",
                    (config_key, pid),
                )
                conn.commit()
                logger.debug(
                    f"[ExportCoordinator] Slot released for {config_key} (pid={pid})"
                )
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as e:
            logger.warning(f"[ExportCoordinator] DB error in release: {e}")

    @contextmanager
    def export_slot(self, config_key: str) -> Generator[None, None, None]:
        """Context manager for acquiring/releasing export slots.

        Raises ExportSlotUnavailable if no slots are available.
        """
        pid = os.getpid()
        if not self.try_acquire(config_key, pid):
            raise ExportSlotUnavailable(
                f"Max {self.max_concurrent} concurrent exports reached"
            )
        try:
            yield
        finally:
            self.release(config_key, pid)


def _is_pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _get_hostname() -> str:
    """Get hostname for identifying which host holds a lock."""
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return ""


# Module-level singleton for convenience
_coordinator: ExportCoordinator | None = None


def get_export_coordinator() -> ExportCoordinator:
    """Get or create the module-level ExportCoordinator singleton."""
    global _coordinator
    if _coordinator is None:
        _coordinator = ExportCoordinator()
    return _coordinator
