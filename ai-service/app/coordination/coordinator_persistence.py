"""Coordinator Persistence Layer (December 2025).

This module provides comprehensive persistence capabilities for coordinators,
building on SQLitePersistenceMixin with:
- State serialization/deserialization
- Automatic state snapshots
- Recovery from persisted state
- Cross-coordinator state management

Features:
- JSON-based state serialization with versioning
- Automatic periodic snapshots
- State recovery on coordinator restart
- Checkpointing for crash recovery
- Cross-coordinator snapshot coordination

Usage:
    from app.coordination.coordinator_persistence import (
        StatePersistenceMixin,
        StateSnapshot,
        SnapshotCoordinator,
        get_snapshot_coordinator,
    )

    class MyCoordinator(CoordinatorBase, StatePersistenceMixin):
        def __init__(self, db_path: Path):
            super().__init__()
            self.init_persistence(db_path)

        def _get_state_for_persistence(self) -> Dict[str, Any]:
            return {"counter": self._counter, "mode": self._mode}

        def _restore_state_from_persistence(self, state: Dict[str, Any]) -> None:
            self._counter = state.get("counter", 0)
            self._mode = state.get("mode", "default")
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from app.coordination.coordinator_base import SQLitePersistenceMixin

if TYPE_CHECKING:
    from app.distributed.db_utils import ThreadLocalConnectionPool

logger = logging.getLogger(__name__)


# =============================================================================
# State Serialization
# =============================================================================


class StateSerializer:
    """Handles state serialization with versioning and compression.

    Provides consistent serialization across all coordinators with:
    - JSON encoding with datetime support
    - Optional gzip compression for large states
    - Version tracking for schema evolution
    """

    VERSION = 1
    COMPRESSION_THRESHOLD = 10000  # Compress states larger than 10KB

    @classmethod
    def serialize(
        cls,
        state: dict[str, Any],
        compress: bool = True,
    ) -> bytes:
        """Serialize state to bytes.

        Args:
            state: State dictionary to serialize
            compress: Whether to compress if large

        Returns:
            Serialized bytes (optionally compressed)
        """
        # Add version metadata
        wrapped = {
            "_version": cls.VERSION,
            "_timestamp": time.time(),
            "state": state,
        }

        json_bytes = json.dumps(
            wrapped,
            default=cls._json_encoder,
            separators=(",", ":"),  # Compact encoding
        ).encode("utf-8")

        # Compress if large and compression enabled
        if compress and len(json_bytes) > cls.COMPRESSION_THRESHOLD:
            compressed = gzip.compress(json_bytes)
            # Only use compressed if actually smaller
            if len(compressed) < len(json_bytes):
                return b"GZ:" + compressed

        return json_bytes

    @classmethod
    def deserialize(cls, data: bytes) -> dict[str, Any]:
        """Deserialize bytes to state dictionary.

        Args:
            data: Serialized bytes

        Returns:
            State dictionary

        Raises:
            ValueError: If deserialization fails
        """
        # Check for compression
        if data.startswith(b"GZ:"):
            data = gzip.decompress(data[3:])

        try:
            wrapped = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize state: {e}")

        # Handle version migration if needed
        version = wrapped.get("_version", 1)
        if version != cls.VERSION:
            wrapped = cls._migrate_state(wrapped, version)

        return wrapped.get("state", {})

    @classmethod
    def _json_encoder(cls, obj: Any) -> Any:
        """Custom JSON encoder for special types."""
        if isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        if isinstance(obj, timedelta):
            return {"__timedelta__": obj.total_seconds()}
        if isinstance(obj, set):
            return {"__set__": list(obj)}
        if isinstance(obj, bytes):
            return {"__bytes__": obj.hex()}
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    @classmethod
    def _migrate_state(
        cls,
        wrapped: dict[str, Any],
        from_version: int,
    ) -> dict[str, Any]:
        """Migrate state from old version to current.

        Args:
            wrapped: Wrapped state with version metadata
            from_version: Version the state was serialized with

        Returns:
            Migrated wrapped state
        """
        # Placeholder for future migrations
        # Add migration logic here as VERSION increases
        logger.warning(
            f"State migration from v{from_version} to v{cls.VERSION} "
            "(no migration needed)"
        )
        wrapped["_version"] = cls.VERSION
        return wrapped


# =============================================================================
# State Snapshot
# =============================================================================


@dataclass
class StateSnapshot:
    """Represents a point-in-time snapshot of coordinator state.

    Attributes:
        coordinator_name: Name of the coordinator
        timestamp: When the snapshot was taken
        state: The persisted state dictionary
        checksum: SHA256 checksum of the serialized state
        metadata: Additional snapshot metadata
    """

    coordinator_name: str
    timestamp: float
    state: dict[str, Any]
    checksum: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        coordinator_name: str,
        state: dict[str, Any],
        **metadata,
    ) -> StateSnapshot:
        """Create a new snapshot with checksum.

        Args:
            coordinator_name: Name of the coordinator
            state: State to snapshot
            **metadata: Additional metadata

        Returns:
            New StateSnapshot instance
        """
        serialized = StateSerializer.serialize(state, compress=False)
        checksum = hashlib.sha256(serialized).hexdigest()[:16]

        return cls(
            coordinator_name=coordinator_name,
            timestamp=time.time(),
            state=state,
            checksum=checksum,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "coordinator_name": self.coordinator_name,
            "timestamp": self.timestamp,
            "state": self.state,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateSnapshot:
        """Create from dictionary."""
        return cls(
            coordinator_name=data["coordinator_name"],
            timestamp=data["timestamp"],
            state=data["state"],
            checksum=data["checksum"],
            metadata=data.get("metadata", {}),
        )

    @property
    def age_seconds(self) -> float:
        """Age of the snapshot in seconds."""
        return time.time() - self.timestamp

    def verify_checksum(self) -> bool:
        """Verify the snapshot checksum."""
        serialized = StateSerializer.serialize(self.state, compress=False)
        expected = hashlib.sha256(serialized).hexdigest()[:16]
        return expected == self.checksum


# =============================================================================
# State Persistence Protocol
# =============================================================================


class StatePersistable(Protocol):
    """Protocol for persistable coordinators."""

    def _get_state_for_persistence(self) -> dict[str, Any]:
        """Get state dictionary to persist."""
        ...

    def _restore_state_from_persistence(self, state: dict[str, Any]) -> None:
        """Restore state from persisted dictionary."""
        ...


# =============================================================================
# State Persistence Mixin
# =============================================================================


class StatePersistenceMixin(SQLitePersistenceMixin):
    """Mixin for coordinator state persistence.

    Extends SQLitePersistenceMixin with:
    - Automatic state snapshots
    - State recovery on startup
    - Checkpoint management

    Usage:
        class MyCoordinator(CoordinatorBase, StatePersistenceMixin):
            def __init__(self, db_path: Path):
                super().__init__()
                self.init_persistence(db_path)

            def _get_state_for_persistence(self) -> Dict[str, Any]:
                return {"counter": self._counter}

            def _restore_state_from_persistence(self, state: Dict[str, Any]) -> None:
                self._counter = state.get("counter", 0)
    """

    _persistence_initialized: bool = False
    _auto_snapshot_interval: float = 300.0  # 5 minutes default
    _max_snapshots: int = 10
    _snapshot_task: asyncio.Task | None = None

    def init_persistence(
        self,
        db_path: Path,
        auto_snapshot: bool = True,
        snapshot_interval: float = 300.0,
        max_snapshots: int = 10,
        profile: str = "standard",
    ) -> None:
        """Initialize persistence with state management.

        Args:
            db_path: Path to SQLite database
            auto_snapshot: Whether to enable automatic snapshots
            snapshot_interval: Seconds between auto-snapshots
            max_snapshots: Maximum snapshots to retain
            profile: Database profile for PRAGMA settings
        """
        # Initialize base SQLite
        self.init_db(db_path, profile=profile)

        self._auto_snapshot_enabled = auto_snapshot
        self._auto_snapshot_interval = snapshot_interval
        self._max_snapshots = max_snapshots
        self._persistence_initialized = True

        logger.debug(
            f"[{getattr(self, '_name', 'Coordinator')}] "
            f"Persistence initialized (auto_snapshot={auto_snapshot})"
        )

    def _get_schema(self) -> str:
        """Get schema including state persistence tables."""
        base_schema = super()._get_schema() if hasattr(super(), "_get_schema") else ""

        persistence_schema = """
        -- State snapshots table
        CREATE TABLE IF NOT EXISTS coordinator_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coordinator_name TEXT NOT NULL,
            timestamp REAL NOT NULL,
            state_data BLOB NOT NULL,
            checksum TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(coordinator_name, timestamp)
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_coordinator
            ON coordinator_snapshots(coordinator_name, timestamp DESC);

        -- Checkpoint table for crash recovery
        CREATE TABLE IF NOT EXISTS coordinator_checkpoints (
            coordinator_name TEXT PRIMARY KEY,
            last_snapshot_id INTEGER,
            last_checkpoint_time REAL,
            recovery_info TEXT,
            FOREIGN KEY (last_snapshot_id) REFERENCES coordinator_snapshots(id)
        );
        """

        return base_schema + persistence_schema

    async def start_auto_snapshots(self) -> None:
        """Start automatic snapshot task."""
        if not self._auto_snapshot_enabled:
            return

        if self._snapshot_task and not self._snapshot_task.done():
            return

        self._snapshot_task = asyncio.create_task(self._auto_snapshot_loop())
        logger.debug(
            f"[{getattr(self, '_name', 'Coordinator')}] "
            f"Auto-snapshot task started"
        )

    async def stop_auto_snapshots(self) -> None:
        """Stop automatic snapshot task."""
        if self._snapshot_task:
            self._snapshot_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._snapshot_task
            self._snapshot_task = None

    async def _auto_snapshot_loop(self) -> None:
        """Background loop for automatic snapshots."""
        while True:
            try:
                await asyncio.sleep(self._auto_snapshot_interval)
                await self.save_snapshot()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(
                    f"[{getattr(self, '_name', 'Coordinator')}] "
                    f"Auto-snapshot failed: {e}"
                )

    def _save_snapshot_db_sync(
        self,
        snapshot: StateSnapshot,
        name: str,
    ) -> int:
        """Synchronous helper to persist snapshot to database.

        Args:
            snapshot: The snapshot to save
            name: Coordinator name

        Returns:
            Number of old snapshots deleted
        """
        conn = self._get_connection()
        serialized = StateSerializer.serialize(snapshot.state)

        conn.execute(
            """
            INSERT INTO coordinator_snapshots
            (coordinator_name, timestamp, state_data, checksum, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                snapshot.coordinator_name,
                snapshot.timestamp,
                serialized,
                snapshot.checksum,
                json.dumps(snapshot.metadata),
            ),
        )

        # Update checkpoint
        conn.execute(
            """
            INSERT OR REPLACE INTO coordinator_checkpoints
            (coordinator_name, last_snapshot_id, last_checkpoint_time)
            VALUES (?, last_insert_rowid(), ?)
            """,
            (name, time.time()),
        )

        conn.commit()

        # Cleanup old snapshots (sync)
        result = conn.execute(
            """
            DELETE FROM coordinator_snapshots
            WHERE coordinator_name = ?
            AND id NOT IN (
                SELECT id FROM coordinator_snapshots
                WHERE coordinator_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            )
            """,
            (name, name, self._max_snapshots),
        )

        deleted = result.rowcount
        if deleted > 0:
            conn.commit()

        return deleted

    async def save_snapshot(self, **metadata) -> StateSnapshot | None:
        """Save current state as a snapshot.

        Args:
            **metadata: Additional metadata to include

        Returns:
            The created snapshot, or None if no state
        """
        if not self._persistence_initialized:
            logger.warning("Persistence not initialized, cannot save snapshot")
            return None

        name = getattr(self, "_name", self.__class__.__name__)

        # Get state from implementing class
        if not hasattr(self, "_get_state_for_persistence"):
            logger.warning(f"[{name}] No _get_state_for_persistence method")
            return None

        try:
            state = self._get_state_for_persistence()
        except Exception as e:
            logger.error(f"[{name}] Failed to get state: {e}")
            return None

        if not state:
            return None

        # Create snapshot
        snapshot = StateSnapshot.create(name, state, **metadata)

        # Persist to database via thread pool (non-blocking)
        try:
            deleted = await asyncio.to_thread(
                self._save_snapshot_db_sync, snapshot, name
            )
            if deleted > 0:
                logger.debug(f"[{name}] Cleaned up {deleted} old snapshots")

            logger.debug(
                f"[{name}] Snapshot saved (checksum={snapshot.checksum})"
            )
            return snapshot

        except Exception as e:
            logger.error(f"[{name}] Failed to save snapshot: {e}")
            raise

    async def load_latest_snapshot(self) -> StateSnapshot | None:
        """Load the most recent snapshot.

        Returns:
            Latest snapshot or None if none exists
        """
        if not self._persistence_initialized:
            return None

        name = getattr(self, "_name", self.__class__.__name__)
        conn = self._get_connection()

        try:
            row = conn.execute(
                """
                SELECT coordinator_name, timestamp, state_data, checksum, metadata
                FROM coordinator_snapshots
                WHERE coordinator_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()

            if not row:
                return None

            state = StateSerializer.deserialize(row[2])
            metadata = json.loads(row[4]) if row[4] else {}

            return StateSnapshot(
                coordinator_name=row[0],
                timestamp=row[1],
                state=state,
                checksum=row[3],
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"[{name}] Failed to load snapshot: {e}")
            return None

    async def restore_from_snapshot(
        self,
        snapshot: StateSnapshot | None = None,
    ) -> bool:
        """Restore state from a snapshot.

        Args:
            snapshot: Snapshot to restore from (loads latest if None)

        Returns:
            True if restoration succeeded
        """
        name = getattr(self, "_name", self.__class__.__name__)

        if snapshot is None:
            snapshot = await self.load_latest_snapshot()

        if snapshot is None:
            logger.info(f"[{name}] No snapshot to restore from")
            return False

        # Verify checksum
        if not snapshot.verify_checksum():
            logger.warning(f"[{name}] Snapshot checksum verification failed")
            return False

        # Restore state
        if not hasattr(self, "_restore_state_from_persistence"):
            logger.warning(f"[{name}] No _restore_state_from_persistence method")
            return False

        try:
            self._restore_state_from_persistence(snapshot.state)
            logger.info(
                f"[{name}] Restored from snapshot "
                f"(age={snapshot.age_seconds:.0f}s, checksum={snapshot.checksum})"
            )
            return True
        except Exception as e:
            logger.error(f"[{name}] Failed to restore from snapshot: {e}")
            return False

    async def _cleanup_old_snapshots(
        self,
        conn: sqlite3.Connection,
        coordinator_name: str,
    ) -> int:
        """Remove old snapshots beyond max_snapshots.

        Args:
            conn: Database connection
            coordinator_name: Name of the coordinator

        Returns:
            Number of snapshots deleted
        """
        # Keep only the most recent max_snapshots
        result = conn.execute(
            """
            DELETE FROM coordinator_snapshots
            WHERE coordinator_name = ?
            AND id NOT IN (
                SELECT id FROM coordinator_snapshots
                WHERE coordinator_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            )
            """,
            (coordinator_name, coordinator_name, self._max_snapshots),
        )

        deleted = result.rowcount
        if deleted > 0:
            conn.commit()
            logger.debug(
                f"[{coordinator_name}] Cleaned up {deleted} old snapshots"
            )

        return deleted

    def get_snapshot_stats(self) -> dict[str, Any]:
        """Get snapshot statistics.

        Returns:
            Dict with snapshot stats
        """
        if not self._persistence_initialized:
            return {"initialized": False}

        name = getattr(self, "_name", self.__class__.__name__)
        conn = self._get_connection()

        try:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as count,
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest,
                    SUM(LENGTH(state_data)) as total_size
                FROM coordinator_snapshots
                WHERE coordinator_name = ?
                """,
                (name,),
            ).fetchone()

            return {
                "initialized": True,
                "snapshot_count": row[0],
                "oldest_snapshot_age": time.time() - row[1] if row[1] else None,
                "newest_snapshot_age": time.time() - row[2] if row[2] else None,
                "total_size_bytes": row[3] or 0,
                "auto_snapshot_enabled": self._auto_snapshot_enabled,
                "snapshot_interval": self._auto_snapshot_interval,
                "max_snapshots": self._max_snapshots,
            }
        except Exception as e:
            logger.warning(f"Failed to get snapshot stats: {e}")
            return {"initialized": True, "error": str(e)}

    # Abstract methods for subclasses to implement
    def _get_state_for_persistence(self) -> dict[str, Any]:
        """Get state dictionary to persist.

        Override in subclass to return the state that should be persisted.

        Returns:
            Dictionary of state to persist
        """
        return {}

    def _restore_state_from_persistence(self, state: dict[str, Any]) -> None:
        """Restore state from persisted dictionary.

        Override in subclass to restore internal state from the dictionary.

        Args:
            state: State dictionary from persistence
        """


# =============================================================================
# Cross-Coordinator Snapshot Coordinator
# =============================================================================


class SnapshotCoordinator:
    """Coordinates snapshots across multiple coordinators.

    Provides:
    - Synchronized multi-coordinator snapshots
    - Cross-coordinator state consistency
    - System-wide state recovery

    Usage:
        snapshot_coord = SnapshotCoordinator(db_path)

        # Take snapshot of all coordinators
        await snapshot_coord.snapshot_all()

        # Restore all from latest consistent snapshot
        await snapshot_coord.restore_all()
    """

    _instance: SnapshotCoordinator | None = None
    _lock = threading.Lock()

    def __init__(self, db_path: Path):
        """Initialize the snapshot coordinator.

        Args:
            db_path: Path to the snapshot database
        """
        self._db_path = db_path
        self._db_pool: ThreadLocalConnectionPool | None = None
        self._init_db()

        # Track registered coordinators
        self._coordinators: dict[str, StatePersistenceMixin] = {}

        logger.info("[SnapshotCoordinator] Initialized")

    def _init_db(self) -> None:
        """Initialize the snapshot database."""
        from app.distributed.db_utils import ThreadLocalConnectionPool

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_pool = ThreadLocalConnectionPool(
            db_path=self._db_path,
            profile="standard",
        )

        conn = self._db_pool.get_connection()
        conn.executescript("""
            -- System-wide snapshots
            CREATE TABLE IF NOT EXISTS system_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                description TEXT,
                coordinator_count INTEGER NOT NULL,
                checksum TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Links system snapshots to individual coordinator snapshots
            CREATE TABLE IF NOT EXISTS snapshot_members (
                system_snapshot_id INTEGER NOT NULL,
                coordinator_name TEXT NOT NULL,
                coordinator_snapshot_checksum TEXT NOT NULL,
                state_data BLOB NOT NULL,
                PRIMARY KEY (system_snapshot_id, coordinator_name),
                FOREIGN KEY (system_snapshot_id) REFERENCES system_snapshots(id)
            );

            CREATE INDEX IF NOT EXISTS idx_system_snapshots_time
                ON system_snapshots(timestamp DESC);
        """)
        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._db_pool is None:
            raise RuntimeError("Database not initialized")
        return self._db_pool.get_connection()

    @classmethod
    def get_instance(cls, db_path: Path | None = None) -> SnapshotCoordinator:
        """Get singleton instance.

        Args:
            db_path: Path to database (required on first call)

        Returns:
            SnapshotCoordinator singleton
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if db_path is None:
                        raise ValueError("db_path required for first initialization")
                    cls._instance = cls(db_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def register_coordinator(
        self,
        coordinator: StatePersistenceMixin,
    ) -> None:
        """Register a coordinator for snapshot management.

        Args:
            coordinator: Coordinator with StatePersistenceMixin
        """
        name = getattr(coordinator, "_name", coordinator.__class__.__name__)
        self._coordinators[name] = coordinator
        logger.debug(f"[SnapshotCoordinator] Registered: {name}")

    def unregister_coordinator(self, name: str) -> None:
        """Unregister a coordinator.

        Args:
            name: Name of coordinator to unregister
        """
        self._coordinators.pop(name, None)

    async def snapshot_all(
        self,
        description: str = "",
        **metadata,
    ) -> int | None:
        """Take synchronized snapshot of all registered coordinators.

        Args:
            description: Description of this snapshot
            **metadata: Additional metadata

        Returns:
            System snapshot ID or None if failed
        """
        if not self._coordinators:
            logger.warning("[SnapshotCoordinator] No coordinators registered")
            return None

        timestamp = time.time()
        snapshots: dict[str, StateSnapshot] = {}

        # Collect snapshots from all coordinators
        for name, coordinator in self._coordinators.items():
            try:
                if hasattr(coordinator, "_get_state_for_persistence"):
                    state = coordinator._get_state_for_persistence()
                    if state:
                        snapshots[name] = StateSnapshot.create(name, state)
            except Exception as e:
                logger.warning(
                    f"[SnapshotCoordinator] Failed to snapshot {name}: {e}"
                )

        if not snapshots:
            logger.warning("[SnapshotCoordinator] No coordinator states to snapshot")
            return None

        # Calculate system checksum
        combined = "".join(s.checksum for s in sorted(snapshots.values(), key=lambda x: x.coordinator_name))
        system_checksum = hashlib.sha256(combined.encode()).hexdigest()[:16]

        # Persist system snapshot
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO system_snapshots
                (timestamp, description, coordinator_count, checksum, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    description,
                    len(snapshots),
                    system_checksum,
                    json.dumps(metadata),
                ),
            )
            system_id = cursor.lastrowid

            # Persist individual coordinator snapshots
            for name, snapshot in snapshots.items():
                serialized = StateSerializer.serialize(snapshot.state)
                conn.execute(
                    """
                    INSERT INTO snapshot_members
                    (system_snapshot_id, coordinator_name, coordinator_snapshot_checksum, state_data)
                    VALUES (?, ?, ?, ?)
                    """,
                    (system_id, name, snapshot.checksum, serialized),
                )

            conn.commit()

            logger.info(
                f"[SnapshotCoordinator] System snapshot created "
                f"(id={system_id}, coordinators={len(snapshots)}, checksum={system_checksum})"
            )
            return system_id

        except Exception as e:
            conn.rollback()
            logger.error(f"[SnapshotCoordinator] Failed to save system snapshot: {e}")
            return None

    async def restore_all(
        self,
        snapshot_id: int | None = None,
    ) -> dict[str, bool]:
        """Restore all coordinators from a system snapshot.

        Args:
            snapshot_id: Specific snapshot to restore (latest if None)

        Returns:
            Dict mapping coordinator names to restore success
        """
        conn = self._get_connection()

        # Get snapshot ID if not specified
        if snapshot_id is None:
            row = conn.execute(
                "SELECT id FROM system_snapshots ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if not row:
                logger.warning("[SnapshotCoordinator] No system snapshots available")
                return {}
            snapshot_id = row[0]

        # Load snapshot members
        rows = conn.execute(
            """
            SELECT coordinator_name, coordinator_snapshot_checksum, state_data
            FROM snapshot_members
            WHERE system_snapshot_id = ?
            """,
            (snapshot_id,),
        ).fetchall()

        results: dict[str, bool] = {}

        for name, checksum, state_data in rows:
            coordinator = self._coordinators.get(name)
            if not coordinator:
                logger.warning(
                    f"[SnapshotCoordinator] Coordinator {name} not registered, skipping"
                )
                results[name] = False
                continue

            try:
                state = StateSerializer.deserialize(state_data)
                snapshot = StateSnapshot(
                    coordinator_name=name,
                    timestamp=0,  # Not used for restore
                    state=state,
                    checksum=checksum,
                )

                if not snapshot.verify_checksum():
                    logger.warning(f"[SnapshotCoordinator] Checksum mismatch for {name}")
                    results[name] = False
                    continue

                coordinator._restore_state_from_persistence(state)
                results[name] = True
                logger.info(f"[SnapshotCoordinator] Restored {name}")

            except Exception as e:
                logger.error(f"[SnapshotCoordinator] Failed to restore {name}: {e}")
                results[name] = False

        success_count = sum(results.values())
        logger.info(
            f"[SnapshotCoordinator] Restore complete: "
            f"{success_count}/{len(results)} coordinators restored"
        )

        return results

    def list_snapshots(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List available system snapshots.

        Args:
            limit: Maximum number of snapshots to return

        Returns:
            List of snapshot info dictionaries
        """
        conn = self._get_connection()

        rows = conn.execute(
            """
            SELECT id, timestamp, description, coordinator_count, checksum
            FROM system_snapshots
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "age_seconds": time.time() - row[1],
                "description": row[2],
                "coordinator_count": row[3],
                "checksum": row[4],
            }
            for row in rows
        ]

    def get_snapshot_detail(
        self,
        snapshot_id: int,
    ) -> dict[str, Any] | None:
        """Get detailed information about a specific snapshot.

        Args:
            snapshot_id: ID of the snapshot

        Returns:
            Snapshot details or None if not found
        """
        conn = self._get_connection()

        # Get snapshot header
        row = conn.execute(
            """
            SELECT id, timestamp, description, coordinator_count, checksum, metadata
            FROM system_snapshots
            WHERE id = ?
            """,
            (snapshot_id,),
        ).fetchone()

        if not row:
            return None

        # Get members
        members = conn.execute(
            """
            SELECT coordinator_name, coordinator_snapshot_checksum, LENGTH(state_data)
            FROM snapshot_members
            WHERE system_snapshot_id = ?
            """,
            (snapshot_id,),
        ).fetchall()

        return {
            "id": row[0],
            "timestamp": row[1],
            "age_seconds": time.time() - row[1],
            "description": row[2],
            "coordinator_count": row[3],
            "checksum": row[4],
            "metadata": json.loads(row[5]) if row[5] else {},
            "members": [
                {
                    "coordinator_name": m[0],
                    "checksum": m[1],
                    "size_bytes": m[2],
                }
                for m in members
            ],
        }

    async def cleanup_old_snapshots(
        self,
        max_age_seconds: float = 86400 * 7,  # 7 days
        max_count: int = 100,
    ) -> int:
        """Remove old system snapshots.

        Args:
            max_age_seconds: Maximum age of snapshots to keep
            max_count: Maximum number of snapshots to keep

        Returns:
            Number of snapshots deleted
        """
        conn = self._get_connection()
        cutoff = time.time() - max_age_seconds

        # Delete by age
        conn.execute(
            """
            DELETE FROM snapshot_members
            WHERE system_snapshot_id IN (
                SELECT id FROM system_snapshots WHERE timestamp < ?
            )
            """,
            (cutoff,),
        )

        result = conn.execute(
            "DELETE FROM system_snapshots WHERE timestamp < ?",
            (cutoff,),
        )
        deleted_by_age = result.rowcount

        # Delete by count - use NOT IN with LIMIT to keep the newest max_count
        # SQLite doesn't support OFFSET without LIMIT, so we invert the logic
        conn.execute(
            """
            DELETE FROM snapshot_members
            WHERE system_snapshot_id NOT IN (
                SELECT id FROM system_snapshots
                ORDER BY timestamp DESC
                LIMIT ?
            )
            """,
            (max_count,),
        )

        result = conn.execute(
            """
            DELETE FROM system_snapshots
            WHERE id NOT IN (
                SELECT id FROM system_snapshots
                ORDER BY timestamp DESC
                LIMIT ?
            )
            """,
            (max_count,),
        )
        deleted_by_count = result.rowcount

        conn.commit()

        total_deleted = deleted_by_age + deleted_by_count
        if total_deleted > 0:
            logger.info(
                f"[SnapshotCoordinator] Cleaned up {total_deleted} old snapshots"
            )

        return total_deleted

    def health_check(self) -> "HealthCheckResult":
        """Check health of the snapshot coordinator.

        Returns:
            HealthCheckResult indicating coordinator health status.

        December 2025: Added for DaemonManager health monitoring integration.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        try:
            # Check database connection health
            conn = self._get_connection()

            # Get snapshot count
            row = conn.execute("SELECT COUNT(*) FROM system_snapshots").fetchone()
            snapshot_count = row[0] if row else 0

            # Get latest snapshot age
            row = conn.execute(
                "SELECT timestamp FROM system_snapshots ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            latest_snapshot_age = (time.time() - row[0]) if row else float('inf')

            # Assess health
            warnings = []

            # No snapshots is a warning
            if snapshot_count == 0:
                warnings.append("No snapshots available")

            # Very old latest snapshot (> 24h) is a warning
            if snapshot_count > 0 and latest_snapshot_age > 86400:
                warnings.append(f"Latest snapshot is {latest_snapshot_age / 3600:.1f}h old")

            # Few coordinators registered is a warning
            coord_count = len(self._coordinators)
            if coord_count == 0:
                warnings.append("No coordinators registered")

            is_healthy = len(warnings) == 0
            status = CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.DEGRADED

            return HealthCheckResult(
                healthy=is_healthy,
                status=status,
                message="; ".join(warnings) if warnings else "SnapshotCoordinator healthy",
                details={
                    "snapshot_count": snapshot_count,
                    "coordinators_registered": coord_count,
                    "coordinator_names": list(self._coordinators.keys()),
                    "latest_snapshot_age_seconds": latest_snapshot_age if snapshot_count > 0 else None,
                    "db_path": str(self._db_path),
                },
            )
        except Exception as e:
            logger.warning(f"[SnapshotCoordinator] health_check error: {e}")
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check error: {e}",
                details={"error": str(e)},
            )


# =============================================================================
# Module-level convenience functions
# =============================================================================


_snapshot_coordinator: SnapshotCoordinator | None = None


def get_snapshot_coordinator(
    db_path: Path | None = None,
) -> SnapshotCoordinator:
    """Get the global snapshot coordinator.

    Args:
        db_path: Database path (required on first call)

    Returns:
        SnapshotCoordinator singleton
    """
    global _snapshot_coordinator
    if _snapshot_coordinator is None:
        if db_path is None:
            # Default path
            db_path = Path("data/coordination/snapshots.db")
        _snapshot_coordinator = SnapshotCoordinator(db_path)
    return _snapshot_coordinator


def reset_snapshot_coordinator() -> None:
    """Reset the global snapshot coordinator (for testing)."""
    global _snapshot_coordinator
    _snapshot_coordinator = None


__all__ = [
    "SnapshotCoordinator",
    "StatePersistable",
    "StatePersistenceMixin",
    "StateSerializer",
    "StateSnapshot",
    "get_snapshot_coordinator",
    "reset_snapshot_coordinator",
]
