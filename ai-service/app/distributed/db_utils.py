#!/usr/bin/env python3
"""Database utilities for atomic operations and transaction management.

This module provides utilities for safe, atomic database operations across
the RingRift AI service. Use these utilities to prevent data corruption
from partial failures or concurrent access.

Usage:
    from app.distributed.db_utils import atomic_write, safe_transaction

    # Atomic file write
    with atomic_write("/path/to/file.json") as f:
        json.dump(data, f)

    # Safe SQLite transaction
    with safe_transaction(db_path) as conn:
        conn.execute("INSERT INTO ...")
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

from app.utils.file_utils import atomic_write


# Thread-local storage for database connections
_local = threading.local()


@contextmanager
def safe_transaction(
    db_path: Union[str, Path],
    timeout: float = 30.0,
    isolation_level: Optional[str] = None,
) -> Generator[sqlite3.Connection, None, None]:
    """Execute a SQLite transaction with automatic rollback on error.

    This ensures that database operations are either fully committed or
    fully rolled back, preventing partial state updates.

    Args:
        db_path: Path to SQLite database
        timeout: Lock acquisition timeout in seconds
        isolation_level: SQLite isolation level (None for autocommit off)

    Yields:
        SQLite connection with active transaction

    Example:
        with safe_transaction("data.db") as conn:
            conn.execute("INSERT INTO games ...")
            conn.execute("UPDATE stats SET ...")
    """
    conn = sqlite3.connect(
        str(db_path),
        timeout=timeout,
        isolation_level=isolation_level
    )
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def exclusive_db_lock(
    db_path: Union[str, Path],
    timeout: float = 60.0,
) -> Generator[sqlite3.Connection, None, None]:
    """Acquire an exclusive lock on a SQLite database.

    This prevents any other connections from reading or writing the database
    until the lock is released. Use for critical operations that require
    full isolation.

    Args:
        db_path: Path to SQLite database
        timeout: Lock acquisition timeout in seconds

    Yields:
        SQLite connection with EXCLUSIVE transaction
    """
    conn = sqlite3.connect(str(db_path), timeout=timeout)

    try:
        # Start exclusive transaction immediately
        conn.execute("BEGIN EXCLUSIVE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def atomic_json_update(
    path: Union[str, Path],
    update_fn: callable,
    default: Any = None,
) -> Any:
    """Atomically read, update, and write a JSON file.

    This is useful for updating configuration or state files without
    risk of corruption.

    Args:
        path: Path to JSON file
        update_fn: Function that takes current data and returns updated data
        default: Default value if file doesn't exist

    Returns:
        Updated data

    Example:
        def increment_counter(data):
            data = data or {"counter": 0}
            data["counter"] += 1
            return data

        atomic_json_update("state.json", increment_counter)
    """
    path = Path(path)

    # Read current data
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        data = default

    # Apply update
    updated = update_fn(data)

    # Write atomically
    with atomic_write(path) as f:
        json.dump(updated, f, indent=2)

    return updated


class TransactionManager:
    """Manages database transactions with automatic retry on lock conflicts.

    This class provides a higher-level interface for database operations
    with built-in retry logic for handling SQLite lock conflicts in
    concurrent environments.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        self.db_path = Path(db_path)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Execute a transaction with retry logic."""
        import time

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with safe_transaction(self.db_path) as conn:
                    yield conn
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < self.max_retries:
                    last_error = e
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

        if last_error:
            raise last_error

    def execute_with_retry(
        self,
        query: str,
        params: tuple = (),
    ) -> sqlite3.Cursor:
        """Execute a single query with retry logic."""
        with self.transaction() as conn:
            return conn.execute(query, params)


# Convenience function for common pattern
def save_state_atomically(
    state_path: Union[str, Path],
    state: Dict[str, Any],
) -> None:
    """Save state dictionary to JSON file atomically.

    Args:
        state_path: Path to state file
        state: State dictionary to save
    """
    with atomic_write(state_path) as f:
        json.dump(state, f, indent=2)


def load_state_safely(
    state_path: Union[str, Path],
    default: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load state dictionary from JSON file safely.

    Args:
        state_path: Path to state file
        default: Default value if file doesn't exist or is corrupted

    Returns:
        State dictionary
    """
    path = Path(state_path)
    if not path.exists():
        return default or {}

    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default or {}


# =============================================================================
# Database Registry
# =============================================================================


class DatabaseRegistry:
    """Centralized registry for database connections and paths.

    This singleton provides:
    - Connection pooling for SQLite databases
    - Well-known database path resolution
    - Schema version tracking
    - Automatic cleanup on shutdown

    Usage:
        from app.distributed.db_utils import get_database, get_db_path

        # Get a connection from the pool
        with get_database("elo") as conn:
            conn.execute("SELECT * FROM ratings")

        # Get database path for external tools
        path = get_db_path("selfplay")
    """

    # Singleton instance
    _instance: Optional["DatabaseRegistry"] = None
    _lock = threading.RLock()

    # Well-known database identifiers and their relative paths
    KNOWN_DATABASES: Dict[str, str] = {
        "elo": "data/unified_elo.db",
        "unified_elo": "data/unified_elo.db",
        "selfplay": "data/games/selfplay.db",
        "canonical": "data/games/canonical.db",
        "training_pool": "data/games/training_pool.db",
        "manifest": "data/data_manifest.db",
        "coordination": "data/coordination.db",
        "model_registry": "data/model_registry.db",
        "holdout": "data/holdouts/holdout.db",
    }

    def __new__(cls) -> "DatabaseRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._base_path: Optional[Path] = None
        self._connections: Dict[str, sqlite3.Connection] = {}
        self._pool_lock = threading.RLock()
        self._custom_paths: Dict[str, Path] = {}
        self._schema_versions: Dict[str, int] = {}

        # Try to auto-detect base path
        self._detect_base_path()

    def _detect_base_path(self) -> None:
        """Auto-detect the ai-service base path."""
        # Try common locations
        candidates = [
            Path(__file__).parent.parent.parent,  # ai-service/app/distributed -> ai-service
            Path.cwd() / "ai-service",
            Path.cwd(),
            Path(os.getenv("RINGRIFT_AI_SERVICE_PATH", "")),
        ]

        for candidate in candidates:
            if candidate and (candidate / "data").exists():
                self._base_path = candidate
                return

        # Default to current directory
        self._base_path = Path.cwd()

    def set_base_path(self, path: Union[str, Path]) -> None:
        """Set the base path for database resolution.

        Args:
            path: Base path (typically ai-service directory)
        """
        self._base_path = Path(path)

    def get_base_path(self) -> Path:
        """Get the current base path."""
        return self._base_path or Path.cwd()

    def register_database(
        self,
        identifier: str,
        relative_path: str,
    ) -> None:
        """Register a custom database path.

        Args:
            identifier: Database identifier
            relative_path: Path relative to base directory
        """
        self._custom_paths[identifier] = Path(relative_path)

    def get_path(
        self,
        identifier: str,
        create_dirs: bool = True,
    ) -> Path:
        """Get the full path for a database identifier.

        Args:
            identifier: Database identifier (e.g., "elo", "selfplay")
            create_dirs: Create parent directories if needed

        Returns:
            Full path to the database file

        Raises:
            ValueError: If identifier is unknown
        """
        # Check custom paths first
        if identifier in self._custom_paths:
            relative = self._custom_paths[identifier]
        elif identifier in self.KNOWN_DATABASES:
            relative = Path(self.KNOWN_DATABASES[identifier])
        else:
            # Treat as a literal path
            return Path(identifier)

        full_path = self.get_base_path() / relative

        if create_dirs:
            full_path.parent.mkdir(parents=True, exist_ok=True)

        return full_path

    def get_connection(
        self,
        identifier: str,
        timeout: float = 30.0,
        check_same_thread: bool = True,
    ) -> sqlite3.Connection:
        """Get a database connection (may be from pool).

        Args:
            identifier: Database identifier
            timeout: Lock acquisition timeout
            check_same_thread: SQLite thread safety check

        Returns:
            SQLite connection
        """
        path = self.get_path(identifier)
        key = str(path)

        with self._pool_lock:
            if key not in self._connections or not check_same_thread:
                conn = sqlite3.connect(
                    str(path),
                    timeout=timeout,
                    check_same_thread=check_same_thread,
                )
                conn.row_factory = sqlite3.Row

                if check_same_thread:
                    self._connections[key] = conn

                return conn

            return self._connections[key]

    @contextmanager
    def connection(
        self,
        identifier: str,
        timeout: float = 30.0,
        new_connection: bool = False,
    ) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection as a context manager.

        Args:
            identifier: Database identifier
            timeout: Lock acquisition timeout
            new_connection: Force a new connection (not from pool)

        Yields:
            SQLite connection with auto-commit/rollback
        """
        if new_connection:
            path = self.get_path(identifier)
            conn = sqlite3.connect(str(path), timeout=timeout)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            conn = self.get_connection(identifier, timeout)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def close_all(self) -> None:
        """Close all pooled connections."""
        with self._pool_lock:
            for conn in self._connections.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()

    def get_schema_version(self, identifier: str) -> int:
        """Get the schema version for a database.

        Args:
            identifier: Database identifier

        Returns:
            Schema version (0 if not set)
        """
        if identifier in self._schema_versions:
            return self._schema_versions[identifier]

        try:
            with self.connection(identifier, new_connection=True) as conn:
                result = conn.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                ).fetchone()
                version = result[0] if result else 0
                self._schema_versions[identifier] = version
                return version
        except sqlite3.OperationalError:
            return 0

    def set_schema_version(
        self,
        identifier: str,
        version: int,
    ) -> None:
        """Set the schema version for a database.

        Args:
            identifier: Database identifier
            version: Schema version number
        """
        with self.connection(identifier, new_connection=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (version,)
            )
            self._schema_versions[identifier] = version

    def list_databases(self) -> Dict[str, Path]:
        """List all known databases and their paths.

        Returns:
            Dict mapping identifiers to paths
        """
        result = {}
        for identifier in self.KNOWN_DATABASES:
            result[identifier] = self.get_path(identifier, create_dirs=False)
        for identifier in self._custom_paths:
            result[identifier] = self.get_path(identifier, create_dirs=False)
        return result

    def database_exists(self, identifier: str) -> bool:
        """Check if a database file exists.

        Args:
            identifier: Database identifier

        Returns:
            True if database file exists
        """
        path = self.get_path(identifier, create_dirs=False)
        return path.exists()


# Module-level singleton access
_registry: Optional[DatabaseRegistry] = None


def get_registry() -> DatabaseRegistry:
    """Get the singleton DatabaseRegistry instance."""
    global _registry
    if _registry is None:
        _registry = DatabaseRegistry()
    return _registry


def get_db_path(identifier: str, create_dirs: bool = True) -> Path:
    """Get the full path for a database identifier.

    Args:
        identifier: Database identifier (e.g., "elo", "selfplay")
        create_dirs: Create parent directories if needed

    Returns:
        Full path to the database file
    """
    return get_registry().get_path(identifier, create_dirs)


@contextmanager
def get_database(
    identifier: str,
    timeout: float = 30.0,
    new_connection: bool = False,
) -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection as a context manager.

    This is the recommended way to access databases throughout
    the RingRift AI service.

    Args:
        identifier: Database identifier (e.g., "elo", "selfplay")
                   or a direct file path
        timeout: Lock acquisition timeout
        new_connection: Force a new connection

    Yields:
        SQLite connection with auto-commit/rollback

    Example:
        with get_database("elo") as conn:
            ratings = conn.execute("SELECT * FROM ratings").fetchall()

        with get_database("data/custom.db") as conn:
            # Works with direct paths too
            pass
    """
    with get_registry().connection(identifier, timeout, new_connection) as conn:
        yield conn


def register_database(identifier: str, relative_path: str) -> None:
    """Register a custom database path.

    Args:
        identifier: Database identifier
        relative_path: Path relative to ai-service directory
    """
    get_registry().register_database(identifier, relative_path)


def close_all_connections() -> None:
    """Close all pooled database connections.

    Call this on shutdown to ensure clean cleanup.
    """
    if _registry is not None:
        _registry.close_all()


# Register cleanup on module unload
import atexit
atexit.register(close_all_connections)
