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
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union


# Thread-local storage for database connections
_local = threading.local()


@contextmanager
def atomic_write(
    path: Union[str, Path],
    mode: str = "w",
    encoding: str = "utf-8",
) -> Generator:
    """Write to a file atomically using a temporary file and rename.

    This ensures that the file is either fully written or not modified at all,
    preventing partial writes on crash or interrupt.

    Args:
        path: Target file path
        mode: File mode ("w" for text, "wb" for binary)
        encoding: Text encoding (ignored for binary mode)

    Yields:
        File object for writing

    Example:
        with atomic_write("config.json") as f:
            json.dump(data, f, indent=2)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory to ensure same filesystem
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )

    try:
        if "b" in mode:
            with os.fdopen(fd, mode) as f:
                yield f
        else:
            with os.fdopen(fd, mode, encoding=encoding) as f:
                yield f

        # Atomic rename (same filesystem guaranteed)
        os.replace(tmp_path, path)

    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


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
