#!/usr/bin/env python3
"""Database utilities for atomic operations and transaction management.

This module provides utilities for safe, atomic database operations across
the RingRift AI service. Use these utilities to prevent data corruption
from partial failures or concurrent access.

Usage:
    from app.distributed.db_utils import (
        get_db_connection,
        safe_transaction,
    )

    # Standard connection with centralized timeout
    conn = get_db_connection("/path/to/db.sqlite")

    # Quick check with short timeout
    conn = get_db_connection("/path/to/db.sqlite", quick=True)

    # Safe SQLite transaction
    with safe_transaction(db_path) as conn:
        conn.execute("INSERT INTO ...")
"""

from __future__ import annotations

import atexit
import json
import os
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from app.utils.file_utils import atomic_write

# Import centralized timeout constants (December 2025)
try:
    from app.config.thresholds import (
        SQLITE_BUSY_TIMEOUT_LONG_MS,
        SQLITE_BUSY_TIMEOUT_MS,
        SQLITE_BUSY_TIMEOUT_SHORT_MS,
        SQLITE_CACHE_SIZE_KB,
        SQLITE_JOURNAL_MODE,
        SQLITE_SHORT_TIMEOUT,
        SQLITE_SYNCHRONOUS,
        SQLITE_TIMEOUT,
        SQLITE_WAL_AUTOCHECKPOINT,
    )
except ImportError:
    SQLITE_TIMEOUT = 30
    SQLITE_SHORT_TIMEOUT = 10
    SQLITE_BUSY_TIMEOUT_MS = 10000
    SQLITE_BUSY_TIMEOUT_LONG_MS = 30000
    SQLITE_BUSY_TIMEOUT_SHORT_MS = 5000
    SQLITE_JOURNAL_MODE = "WAL"
    SQLITE_SYNCHRONOUS = "NORMAL"
    SQLITE_WAL_AUTOCHECKPOINT = 100
    SQLITE_CACHE_SIZE_KB = -2000

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Thread-local storage for database connections
_local = threading.local()


# =============================================================================
# SQLite Connection Limiter (January 2026 - Phase 6 P2P Stability)
# =============================================================================


@dataclass
class ConnectionSlot:
    """Tracks a single connection slot."""

    db_path: str
    acquired_at: float
    thread_id: int
    thread_name: str


class SQLiteConnectionLimiter:
    """Global connection limiter to prevent SQLite connection exhaustion.

    Jan 21, 2026: Phase 6 of P2P Stability Plan - prevents connection
    accumulation under high load conditions in 20+ node clusters.

    Features:
    - Global connection limit with backpressure (blocks when at limit)
    - Warning threshold at 80% capacity
    - Leak detection for connections open > threshold
    - Rollback via RINGRIFT_SQLITE_DISABLE_LIMIT=true

    Usage:
        limiter = get_connection_limiter()

        # Acquire slot before opening connection
        if limiter.acquire("path/to/db.sqlite", timeout=30.0):
            try:
                conn = sqlite3.connect("path/to/db.sqlite")
                # ... use connection ...
            finally:
                limiter.release()

        # Check for leaks periodically
        leaks = limiter.detect_leaks(threshold_seconds=300.0)
        for leak in leaks:
            logger.warning(f"Potential leak: {leak}")
    """

    # Configurable limits via environment
    MAX_CONNECTIONS = int(os.environ.get("RINGRIFT_SQLITE_MAX_CONNECTIONS", "100"))
    WARN_THRESHOLD = int(MAX_CONNECTIONS * 0.8)

    # Disable limiter for rollback
    _disabled = os.environ.get("RINGRIFT_SQLITE_DISABLE_LIMIT", "").lower() in (
        "true", "1", "yes"
    )

    # Singleton instance
    _instance: "SQLiteConnectionLimiter | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "SQLiteConnectionLimiter":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._active_slots: dict[int, ConnectionSlot] = {}  # slot_id -> ConnectionSlot
        self._next_slot_id = 0
        self._warned_at_threshold = False
        self._total_acquired = 0
        self._total_timeouts = 0
        self._total_leaks_detected = 0

        # Thread-local slot tracking for release()
        self._thread_local = threading.local()

        if self._disabled:
            logger.info("SQLiteConnectionLimiter disabled via RINGRIFT_SQLITE_DISABLE_LIMIT")

    def acquire(self, db_path: str, timeout: float = 30.0) -> bool:
        """Acquire a connection slot with backpressure.

        Args:
            db_path: Path to database (for tracking)
            timeout: Max time to wait for a slot (seconds)

        Returns:
            True if slot acquired, False if timeout

        Note:
            Call release() when done to free the slot.
        """
        if self._disabled:
            return True

        deadline = time.time() + timeout

        with self._condition:
            # Wait for available slot
            while len(self._active_slots) >= self.MAX_CONNECTIONS:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self._total_timeouts += 1
                    logger.warning(
                        f"SQLite connection limit timeout ({self.MAX_CONNECTIONS} connections, "
                        f"waited {timeout:.1f}s for {db_path})"
                    )
                    self._emit_backpressure_event(db_path, timeout)
                    return False

                self._condition.wait(timeout=remaining)

            # Acquire slot
            slot_id = self._next_slot_id
            self._next_slot_id += 1

            current_thread = threading.current_thread()
            slot = ConnectionSlot(
                db_path=db_path,
                acquired_at=time.time(),
                thread_id=current_thread.ident or 0,
                thread_name=current_thread.name,
            )
            self._active_slots[slot_id] = slot
            self._total_acquired += 1

            # Track slot for this thread (for release)
            if not hasattr(self._thread_local, "slot_stack"):
                self._thread_local.slot_stack = []
            self._thread_local.slot_stack.append(slot_id)

            # Warn at threshold
            current_count = len(self._active_slots)
            if current_count >= self.WARN_THRESHOLD and not self._warned_at_threshold:
                self._warned_at_threshold = True
                logger.warning(
                    f"SQLite connections at {current_count}/{self.MAX_CONNECTIONS} "
                    f"(warning threshold {self.WARN_THRESHOLD})"
                )
                self._emit_warning_event(current_count)

            return True

    def release(self) -> None:
        """Release the most recently acquired connection slot."""
        if self._disabled:
            return

        with self._condition:
            # Get slot from thread-local stack
            if not hasattr(self._thread_local, "slot_stack") or not self._thread_local.slot_stack:
                logger.debug("release() called without matching acquire()")
                return

            slot_id = self._thread_local.slot_stack.pop()

            if slot_id in self._active_slots:
                del self._active_slots[slot_id]

                # Reset threshold warning if we drop below
                if len(self._active_slots) < self.WARN_THRESHOLD:
                    self._warned_at_threshold = False

                # Notify waiters
                self._condition.notify()

    def detect_leaks(self, threshold_seconds: float = 300.0) -> list[dict]:
        """Find connections open longer than threshold.

        Args:
            threshold_seconds: Consider connections open longer than this as leaks

        Returns:
            List of dicts with leak details (db_path, thread_name, age_seconds)
        """
        if self._disabled:
            return []

        leaks = []
        now = time.time()

        with self._lock:
            for slot_id, slot in list(self._active_slots.items()):
                age = now - slot.acquired_at
                if age > threshold_seconds:
                    leaks.append({
                        "slot_id": slot_id,
                        "db_path": slot.db_path,
                        "thread_id": slot.thread_id,
                        "thread_name": slot.thread_name,
                        "age_seconds": round(age, 1),
                        "acquired_at": slot.acquired_at,
                    })

            if leaks:
                self._total_leaks_detected += len(leaks)

        return leaks

    def get_stats(self) -> dict:
        """Get connection limiter statistics."""
        with self._lock:
            return {
                "active_connections": len(self._active_slots),
                "max_connections": self.MAX_CONNECTIONS,
                "warn_threshold": self.WARN_THRESHOLD,
                "utilization_pct": round(len(self._active_slots) / self.MAX_CONNECTIONS * 100, 1),
                "total_acquired": self._total_acquired,
                "total_timeouts": self._total_timeouts,
                "total_leaks_detected": self._total_leaks_detected,
                "disabled": self._disabled,
            }

    def get_active_connections(self) -> list[dict]:
        """Get details of all active connections."""
        with self._lock:
            now = time.time()
            return [
                {
                    "slot_id": slot_id,
                    "db_path": slot.db_path,
                    "thread_name": slot.thread_name,
                    "age_seconds": round(now - slot.acquired_at, 1),
                }
                for slot_id, slot in self._active_slots.items()
            ]

    def _emit_backpressure_event(self, db_path: str, wait_time: float) -> None:
        """Emit event when connection limit causes backpressure."""
        try:
            from app.coordination.safe_event_emitter import safe_emit_event

            safe_emit_event(
                "sqlite_backpressure",  # DataEventType.SQLITE_BACKPRESSURE.value
                {
                    "db_path": db_path,
                    "wait_time": wait_time,
                    "active_connections": len(self._active_slots),
                    "max_connections": self.MAX_CONNECTIONS,
                },
                source="sqlite_connection_limiter",
            )
        except (ImportError, Exception):
            pass  # Event emission is optional

    def _emit_warning_event(self, current_count: int) -> None:
        """Emit event when approaching connection limit."""
        try:
            from app.coordination.safe_event_emitter import safe_emit_event

            safe_emit_event(
                "sqlite_connection_warning",  # DataEventType.SQLITE_CONNECTION_WARNING.value
                {
                    "active_connections": current_count,
                    "max_connections": self.MAX_CONNECTIONS,
                    "warn_threshold": self.WARN_THRESHOLD,
                },
                source="sqlite_connection_limiter",
            )
        except (ImportError, Exception):
            pass  # Event emission is optional


# Module-level singleton accessor
_limiter: SQLiteConnectionLimiter | None = None


def get_connection_limiter() -> SQLiteConnectionLimiter:
    """Get the singleton SQLiteConnectionLimiter instance."""
    global _limiter
    if _limiter is None:
        _limiter = SQLiteConnectionLimiter()
    return _limiter


@contextmanager
def limited_connection(
    db_path: str | Path,
    timeout: float | None = None,
    quick: bool = False,
    row_factory: bool = True,
) -> Generator[sqlite3.Connection]:
    """Get a SQLite connection with global limiting.

    This wraps get_db_connection() with the connection limiter to prevent
    connection exhaustion under high load.

    Args:
        db_path: Path to SQLite database
        timeout: Connection timeout (default uses SQLITE_TIMEOUT)
        quick: Use short timeout for quick checks
        row_factory: Set row_factory to sqlite3.Row

    Yields:
        SQLite connection

    Raises:
        TimeoutError: If connection slot cannot be acquired in time

    Example:
        with limited_connection("data.db") as conn:
            conn.execute("SELECT ...")

    Jan 21, 2026: Phase 6 P2P Stability Plan
    """
    limiter = get_connection_limiter()
    db_path_str = str(db_path)

    # Acquire slot with backpressure
    slot_timeout = timeout if timeout is not None else SQLITE_TIMEOUT
    if not limiter.acquire(db_path_str, timeout=slot_timeout):
        raise TimeoutError(
            f"Could not acquire SQLite connection slot for {db_path_str} "
            f"(limit: {limiter.MAX_CONNECTIONS})"
        )

    try:
        # Get actual connection
        conn = get_db_connection(db_path, quick=quick, timeout=timeout, row_factory=row_factory)
        try:
            yield conn
        finally:
            conn.close()
    finally:
        limiter.release()


# =============================================================================
# PRAGMA Configuration Profiles (December 2025)
# =============================================================================

class PragmaProfile:
    """PRAGMA configuration profiles for different use cases."""

    # Standard profile - used by most coordinators
    STANDARD = {
        "journal_mode": SQLITE_JOURNAL_MODE,
        "busy_timeout": SQLITE_BUSY_TIMEOUT_MS,
        "synchronous": SQLITE_SYNCHRONOUS,
    }

    # Extended profile - for cross-process events, needs more cache
    EXTENDED = {
        "journal_mode": SQLITE_JOURNAL_MODE,
        "busy_timeout": SQLITE_BUSY_TIMEOUT_LONG_MS,
        "synchronous": SQLITE_SYNCHRONOUS,
        "wal_autocheckpoint": SQLITE_WAL_AUTOCHECKPOINT,
        "cache_size": SQLITE_CACHE_SIZE_KB,
    }

    # Quick profile - for fast registry lookups
    QUICK = {
        "journal_mode": SQLITE_JOURNAL_MODE,
        "busy_timeout": SQLITE_BUSY_TIMEOUT_SHORT_MS,
        "synchronous": SQLITE_SYNCHRONOUS,
    }

    # Minimal profile - basic settings only
    MINIMAL = {
        "journal_mode": SQLITE_JOURNAL_MODE,
    }


def apply_pragmas(
    conn: sqlite3.Connection,
    profile: str = "standard",
    custom_pragmas: dict[str, Any] | None = None,
) -> None:
    """Apply PRAGMA settings to a SQLite connection.

    Args:
        conn: SQLite connection
        profile: Profile name ("standard", "extended", "quick", "minimal")
        custom_pragmas: Additional or override PRAGMA settings

    Example:
        conn = sqlite3.connect("db.sqlite")
        apply_pragmas(conn, profile="extended")

        # Or with custom settings
        apply_pragmas(conn, custom_pragmas={"busy_timeout": 60000})
    """
    # Whitelist of allowed PRAGMA names (security - prevents PRAGMA injection)
    ALLOWED_PRAGMAS = frozenset({
        "journal_mode", "busy_timeout", "synchronous", "wal_autocheckpoint",
        "cache_size", "temp_store", "mmap_size", "page_size", "auto_vacuum",
        "foreign_keys", "recursive_triggers", "secure_delete", "wal_checkpoint",
    })

    # Get base profile
    profiles = {
        "standard": PragmaProfile.STANDARD,
        "extended": PragmaProfile.EXTENDED,
        "quick": PragmaProfile.QUICK,
        "minimal": PragmaProfile.MINIMAL,
    }
    pragmas = profiles.get(profile, PragmaProfile.STANDARD).copy()

    # Apply custom overrides
    if custom_pragmas:
        pragmas.update(custom_pragmas)

    # Execute PRAGMAs with validation
    for pragma, value in pragmas.items():
        # Validate pragma name is in whitelist
        if pragma not in ALLOWED_PRAGMAS:
            logger.warning(f"Skipping disallowed PRAGMA: {pragma}")
            continue
        try:
            conn.execute(f"PRAGMA {pragma}={value}")
        except sqlite3.OperationalError as e:
            logger.debug(f"PRAGMA {pragma}={value} failed: {e}")


class ThreadLocalConnectionPool:
    """Thread-local SQLite connection pool with consistent PRAGMA configuration.

    This class replaces scattered thread-local connection patterns across
    coordinator modules. Use this to ensure consistent SQLite configuration.

    Usage:
        # In coordinator __init__:
        self._pool = ThreadLocalConnectionPool(
            db_path=self._db_path,
            profile="standard",
        )

        # To get connection:
        conn = self._pool.get_connection()

        # Instead of manually managing thread-local + PRAGMAs

    Replaces patterns in:
        - coordinator_base.py:398-404
        - sync_mutex.py:113-117
        - training_coordinator.py:185-193
        - queue_monitor.py:165-169
        - duration_scheduler.py:122-126
        - cross_process_events.py:112-123
        - orchestrator_registry.py:178-180
    """

    def __init__(
        self,
        db_path: str | Path,
        timeout: float | None = None,
        profile: str = "standard",
        row_factory: bool = True,
        custom_pragmas: dict[str, Any] | None = None,
    ):
        """Initialize the connection pool.

        Args:
            db_path: Path to SQLite database
            timeout: Connection timeout (defaults to SQLITE_TIMEOUT)
            profile: PRAGMA profile ("standard", "extended", "quick", "minimal")
            row_factory: Use sqlite3.Row for dict-like access
            custom_pragmas: Additional PRAGMA settings
        """
        self._db_path = Path(db_path)
        self._timeout = timeout if timeout is not None else SQLITE_TIMEOUT
        self._profile = profile
        self._row_factory = row_factory
        self._custom_pragmas = custom_pragmas
        self._local = threading.local()

    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection.

        Returns:
            SQLite connection configured with PRAGMAs

        Raises:
            sqlite3.OperationalError: If connection fails
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self._db_path),
                timeout=self._timeout,
            )
            if self._row_factory:
                self._local.conn.row_factory = sqlite3.Row

            # Apply PRAGMAs consistently
            apply_pragmas(
                self._local.conn,
                profile=self._profile,
                custom_pragmas=self._custom_pragmas,
            )

        return self._local.conn

    def close_connection(self) -> None:
        """Close the thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            with suppress(Exception):
                self._local.conn.close()
            self._local.conn = None

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a query on the thread-local connection.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Cursor with results
        """
        return self.get_connection().execute(query, params)

    def executemany(self, query: str, params_seq) -> sqlite3.Cursor:
        """Execute a query with multiple parameter sets.

        Args:
            query: SQL query
            params_seq: Sequence of parameter tuples

        Returns:
            Cursor
        """
        return self.get_connection().executemany(query, params_seq)

    @property
    def path(self) -> Path:
        """Get the database path."""
        return self._db_path


def create_coordinator_pool(
    db_path: str | Path,
    profile: str = "standard",
) -> ThreadLocalConnectionPool:
    """Create a connection pool for a coordinator module.

    This is the recommended factory for coordinator modules migrating
    from manual thread-local connection management.

    Args:
        db_path: Path to SQLite database
        profile: PRAGMA profile ("standard", "extended", "quick")

    Returns:
        Configured ThreadLocalConnectionPool

    Example:
        # In coordinator __init__:
        from app.distributed.db_utils import create_coordinator_pool

        self._pool = create_coordinator_pool(self._db_path, profile="standard")

        # In methods:
        conn = self._pool.get_connection()
        conn.execute("SELECT ...")
    """
    return ThreadLocalConnectionPool(
        db_path=db_path,
        profile=profile,
        row_factory=True,
    )


def get_db_connection(
    db_path: str | Path,
    quick: bool = False,
    timeout: float | None = None,
    row_factory: bool = True,
) -> sqlite3.Connection:
    """Get a SQLite connection with standardized timeout.

    Uses centralized timeout constants from app/config/thresholds.py.
    This is the preferred way to create SQLite connections across the codebase.

    Args:
        db_path: Path to SQLite database
        quick: If True, use short timeout for quick checks (10s default)
        timeout: Override timeout (use sparingly, prefer quick=True/False)
        row_factory: If True, set row_factory to sqlite3.Row for dict-like access

    Returns:
        SQLite connection with appropriate timeout

    Example:
        # Standard operation (30s timeout)
        conn = get_db_connection("data.db")

        # Quick check (10s timeout)
        conn = get_db_connection("data.db", quick=True)

    December 2025 - Standardization effort
    """
    if timeout is None:
        timeout = SQLITE_SHORT_TIMEOUT if quick else SQLITE_TIMEOUT

    conn = sqlite3.connect(str(db_path), timeout=timeout)
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def safe_transaction(
    db_path: str | Path,
    timeout: float | None = None,
    isolation_level: str | None = None,
) -> Generator[sqlite3.Connection]:
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
    if timeout is None:
        timeout = SQLITE_TIMEOUT

    conn = sqlite3.connect(
        str(db_path),
        timeout=timeout,
        isolation_level=isolation_level
    )
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def exclusive_db_lock(
    db_path: str | Path,
    timeout: float = 60.0,
) -> Generator[sqlite3.Connection]:
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
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.close()


def atomic_json_update(
    path: str | Path,
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
        db_path: str | Path,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        self.db_path = Path(db_path)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection]:
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
    state_path: str | Path,
    state: dict[str, Any],
) -> None:
    """Save state dictionary to JSON file atomically.

    Args:
        state_path: Path to state file
        state: State dictionary to save
    """
    with atomic_write(state_path) as f:
        json.dump(state, f, indent=2)


def load_state_safely(
    state_path: str | Path,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    except (OSError, json.JSONDecodeError):
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
    _instance: DatabaseRegistry | None = None
    _lock = threading.RLock()

    # Well-known database identifiers and their relative paths
    KNOWN_DATABASES: dict[str, str] = {
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

    def __new__(cls) -> DatabaseRegistry:
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
        self._base_path: Path | None = None
        self._connections: dict[str, sqlite3.Connection] = {}
        self._pool_lock = threading.RLock()
        self._custom_paths: dict[str, Path] = {}
        self._schema_versions: dict[str, int] = {}

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

    def set_base_path(self, path: str | Path) -> None:
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
    ) -> Generator[sqlite3.Connection]:
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
            except sqlite3.Error:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            conn = self.get_connection(identifier, timeout)
            try:
                yield conn
                conn.commit()
            except sqlite3.Error:
                conn.rollback()
                raise

    def close_all(self) -> None:
        """Close all pooled connections."""
        with self._pool_lock:
            for conn in self._connections.values():
                with suppress(Exception):
                    conn.close()
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

    def list_databases(self) -> dict[str, Path]:
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
_registry: DatabaseRegistry | None = None


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
) -> Generator[sqlite3.Connection]:
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
atexit.register(close_all_connections)


# Alias for backward compatibility (used by p2p_orchestrator.py)
safe_db_connection = safe_transaction
