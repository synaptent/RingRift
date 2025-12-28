"""Base registry class with shared database connection handling.

Provides common infrastructure for all registry implementations:
- Thread-safe database connection management
- Common query patterns
- Logging setup

December 2025 - ClusterManifest decomposition.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)


class BaseRegistry:
    """Base class for registry implementations.

    Provides shared database connection handling and common utilities.
    Can either manage its own connection or use an external one (for composition).
    """

    def __init__(
        self,
        db_path: Path | None = None,
        external_connection: sqlite3.Connection | None = None,
        external_lock: threading.RLock | None = None,
    ):
        """Initialize the registry.

        Args:
            db_path: Path to SQLite database (for standalone use)
            external_connection: External connection to use (for composition)
            external_lock: External lock to use (for composition)
        """
        self._db_path = db_path
        self._external_conn = external_connection
        self._own_conn: sqlite3.Connection | None = None
        self._lock = external_lock or threading.RLock()
        self._owns_connection = external_connection is None

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with thread safety.

        Yields:
            SQLite connection
        """
        with self._lock:
            if self._external_conn is not None:
                yield self._external_conn
            elif self._own_conn is not None:
                yield self._own_conn
            elif self._db_path is not None:
                if self._own_conn is None:
                    self._own_conn = sqlite3.connect(
                        self._db_path,
                        check_same_thread=False,
                        timeout=30.0,
                    )
                    self._own_conn.execute("PRAGMA journal_mode=WAL")
                    self._own_conn.execute("PRAGMA synchronous=NORMAL")
                yield self._own_conn
            else:
                raise RuntimeError("No database connection available")

    def set_external_connection(
        self,
        conn: sqlite3.Connection,
        lock: threading.RLock | None = None,
    ) -> None:
        """Set external connection for composition pattern.

        Args:
            conn: External SQLite connection
            lock: External lock (optional)
        """
        with self._lock:
            # Close own connection if we had one
            if self._own_conn is not None:
                self._own_conn.close()
                self._own_conn = None
            self._external_conn = conn
            self._owns_connection = False
        if lock is not None:
            self._lock = lock

    def close(self) -> None:
        """Close own database connection if we own it."""
        with self._lock:
            if self._owns_connection and self._own_conn is not None:
                self._own_conn.close()
                self._own_conn = None

    def _execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a query and return cursor.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Cursor with results
        """
        with self._connection() as conn:
            return conn.execute(query, params)

    def _execute_many(self, query: str, params_list: list[tuple]) -> int:
        """Execute query for multiple parameter sets.

        Args:
            query: SQL query
            params_list: List of parameter tuples

        Returns:
            Number of rows affected
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount

    def _fetch_one(self, query: str, params: tuple = ()) -> Any | None:
        """Fetch single row from query.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Row tuple or None
        """
        with self._connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchone()

    def _fetch_all(self, query: str, params: tuple = ()) -> list[Any]:
        """Fetch all rows from query.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            List of row tuples
        """
        with self._connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()

    def _commit(self) -> None:
        """Commit current transaction."""
        with self._connection() as conn:
            conn.commit()
