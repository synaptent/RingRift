"""UnifiedRegistryBase - Common base class for registry implementations (December 2025).

This module provides a common base class for various registry implementations
across the codebase, standardizing common patterns like:
- SQLite-based persistence
- Thread-safe access
- Singleton pattern
- Event emission on changes
- Stats and health reporting

Registry implementations that can use this base:
- OrchestratorRegistry (app/coordination/orchestrator_registry.py)
- TaskRegistry (app/coordination/task_coordinator.py)
- HealthRegistry (app/distributed/health_registry.py)
- DynamicHostRegistry (app/distributed/dynamic_registry.py)
- DatabaseRegistry (app/distributed/db_utils.py)
- ModelRegistry (app/training/model_registry.py)
- AIAgentRegistry (app/tournament/agents.py)

Usage:
    from app.core.registry_base import UnifiedRegistryBase, RegistryConfig

    class MyRegistry(UnifiedRegistryBase):
        def __init__(self, db_path: Path):
            super().__init__(RegistryConfig(db_path=db_path, name="my_registry"))

        def _init_schema(self) -> None:
            # Create your tables
            pass

        def register(self, item: Any) -> bool:
            # Register an item
            pass
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RegistryConfig:
    """Configuration for a registry."""

    db_path: Path
    name: str
    auto_create: bool = True
    auto_vacuum: bool = True
    emit_events: bool = True
    health_check_interval: float = 60.0


@dataclass
class RegistryStats:
    """Statistics for a registry."""

    name: str
    item_count: int = 0
    last_modified: float = 0.0
    last_access: float = 0.0
    errors_count: int = 0
    db_size_bytes: int = 0
    is_healthy: bool = True


class UnifiedRegistryBase(ABC, Generic[T]):
    """Abstract base class for registry implementations.

    Provides common patterns for:
    - SQLite-based persistence with thread-safe access
    - Event emission on changes
    - Stats and health reporting
    - Singleton management

    Subclasses must implement:
    - _init_schema(): Initialize database schema
    - _get_item_count(): Return current item count
    """

    _instances: dict[str, UnifiedRegistryBase] = {}
    _instance_lock = threading.Lock()

    def __init__(self, config: RegistryConfig):
        """Initialize registry with configuration.

        Args:
            config: Registry configuration
        """
        self.config = config
        self.db_path = config.db_path
        self.name = config.name

        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._last_modified: float = 0.0
        self._last_access: float = 0.0
        self._errors_count: int = 0
        self._initialized: bool = False

        # Event callbacks
        self._on_change_callbacks: list[Callable[[str, Any], None]] = []

        # Initialize database
        if config.auto_create:
            self._ensure_db()

    @classmethod
    def get_instance(cls, config: RegistryConfig | None = None) -> UnifiedRegistryBase:
        """Get singleton instance of this registry.

        Args:
            config: Configuration (only used on first call)

        Returns:
            Registry singleton instance
        """
        with cls._instance_lock:
            if cls.__name__ not in cls._instances:
                if config is None:
                    raise ValueError(f"Config required for first instantiation of {cls.__name__}")
                cls._instances[cls.__name__] = cls(config)
            return cls._instances[cls.__name__]

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._instance_lock:
            if cls.__name__ in cls._instances:
                instance = cls._instances[cls.__name__]
                if hasattr(instance, "close"):
                    instance.close()
                del cls._instances[cls.__name__]

    def _ensure_db(self) -> None:
        """Ensure database and schema exist."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            try:
                # Ensure parent directory exists
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

                # Create connection and schema
                self._connection = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    timeout=30.0,
                )
                self._connection.row_factory = sqlite3.Row

                # Enable WAL mode for better concurrency
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=NORMAL")

                # Initialize schema
                self._init_schema()

                self._initialized = True
                logger.debug(f"[{self.name}] Database initialized at {self.db_path}")

            except Exception as e:
                self._errors_count += 1
                logger.error(f"[{self.name}] Failed to initialize database: {e}")
                raise

    @abstractmethod
    def _init_schema(self) -> None:
        """Initialize database schema.

        Subclasses must implement this to create their tables.
        """
        pass

    @abstractmethod
    def _get_item_count(self) -> int:
        """Get current item count in the registry.

        Returns:
            Number of items in the registry
        """
        pass

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection, initializing if needed.

        Returns:
            sqlite3.Connection
        """
        self._ensure_db()
        self._last_access = time.time()
        return self._connection

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a query with thread-safe locking.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            sqlite3.Cursor
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(query, params)
                return cursor
            except Exception as e:
                self._errors_count += 1
                logger.error(f"[{self.name}] Query failed: {query[:100]}... - {e}")
                raise

    def execute_many(self, query: str, params_list: list[tuple]) -> int:
        """Execute a query with multiple parameter sets.

        Args:
            query: SQL query
            params_list: List of parameter tuples

        Returns:
            Number of rows affected
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.executemany(query, params_list)
                conn.commit()
                self._last_modified = time.time()
                return cursor.rowcount
            except Exception as e:
                self._errors_count += 1
                logger.error(f"[{self.name}] Batch query failed: {e}")
                raise

    def commit(self) -> None:
        """Commit current transaction."""
        with self._lock:
            if self._connection:
                self._connection.commit()
                self._last_modified = time.time()

    def _emit_change(self, event_type: str, data: Any) -> None:
        """Emit a change event.

        Args:
            event_type: Type of change (e.g., "item_added", "item_removed")
            data: Event data
        """
        if not self.config.emit_events:
            return

        for callback in self._on_change_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.debug(f"[{self.name}] Change callback failed: {e}")

        # Also emit to event bus if available
        self._emit_to_event_bus(event_type, data)

    def _emit_to_event_bus(self, event_type: str, data: Any) -> None:
        """Emit change to the global event bus."""
        try:
            from app.distributed.data_events import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            # Map common registry events
            event_mapping = {
                "item_added": DataEventType.REGISTRY_UPDATED,
                "item_removed": DataEventType.REGISTRY_UPDATED,
                "item_updated": DataEventType.REGISTRY_UPDATED,
            }

            if event_type in event_mapping:
                event = DataEvent(
                    event_type=event_mapping[event_type],
                    payload={
                        "registry": self.name,
                        "event_type": event_type,
                        "data": data,
                        "timestamp": time.time(),
                    },
                    source=f"registry_{self.name}",
                )

                bus = get_event_bus()
                bus.publish_sync(event)

        except Exception:
            pass  # Event bus not available

    def on_change(self, callback: Callable[[str, Any], None]) -> None:
        """Register a callback for change events.

        Args:
            callback: Function to call on changes (event_type, data)
        """
        self._on_change_callbacks.append(callback)

    def get_stats(self) -> RegistryStats:
        """Get registry statistics.

        Returns:
            RegistryStats with current metrics
        """
        try:
            item_count = self._get_item_count()
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return RegistryStats(
                name=self.name,
                item_count=item_count,
                last_modified=self._last_modified,
                last_access=self._last_access,
                errors_count=self._errors_count,
                db_size_bytes=db_size,
                is_healthy=self._errors_count == 0,
            )
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to get stats: {e}")
            return RegistryStats(name=self.name, is_healthy=False)

    def is_healthy(self) -> bool:
        """Check if registry is healthy.

        Returns:
            True if registry is operational
        """
        try:
            with self._lock:
                conn = self._get_connection()
                conn.execute("SELECT 1")
                return True
        except Exception:
            return False

    def vacuum(self) -> bool:
        """Vacuum the database to reclaim space.

        Returns:
            True if successful
        """
        try:
            with self._lock:
                if self._connection:
                    self._connection.execute("VACUUM")
                    logger.debug(f"[{self.name}] Database vacuumed")
                    return True
        except Exception as e:
            logger.warning(f"[{self.name}] Vacuum failed: {e}")
        return False

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if self._connection:
                with contextlib.suppress(Exception):
                    self._connection.close()
                self._connection = None
                self._initialized = False

    def __enter__(self) -> UnifiedRegistryBase:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


# =============================================================================
# Convenience mixins for common registry patterns
# =============================================================================

class TimestampedRegistryMixin:
    """Mixin for registries that track creation/update timestamps."""

    def _add_timestamp_columns(self, table_name: str) -> str:
        """Get SQL to add timestamp columns to a table.

        Args:
            table_name: Name of the table

        Returns:
            SQL ALTER statements (may fail if columns exist)
        """
        return f"""
        ALTER TABLE {table_name} ADD COLUMN created_at REAL;
        ALTER TABLE {table_name} ADD COLUMN updated_at REAL;
        """


class TTLRegistryMixin:
    """Mixin for registries with TTL-based expiration."""

    def _cleanup_expired(self, table_name: str, ttl_seconds: float, timestamp_column: str = "updated_at") -> int:
        """Remove expired entries from a table.

        Args:
            table_name: Table to clean
            ttl_seconds: Time-to-live in seconds
            timestamp_column: Column with timestamp

        Returns:
            Number of entries removed
        """
        cutoff = time.time() - ttl_seconds
        cursor = self.execute(
            f"DELETE FROM {table_name} WHERE {timestamp_column} < ?",
            (cutoff,)
        )
        self.commit()
        return cursor.rowcount


__all__ = [
    "RegistryConfig",
    "RegistryStats",
    "TTLRegistryMixin",
    "TimestampedRegistryMixin",
    "UnifiedRegistryBase",
]
