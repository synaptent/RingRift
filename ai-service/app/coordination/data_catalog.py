"""Data Catalog - Central registry of all cluster data locations.

This module provides the DataCatalog class which maintains a real-time view
of what data exists on which nodes in the cluster. It serves as the single
source of truth for data location queries.

Part of the Unified Data Plane Daemon architecture (December 2025).

Usage:
    from app.coordination.data_catalog import DataCatalog, get_data_catalog

    catalog = get_data_catalog()

    # Register data entry
    catalog.register(DataEntry(
        path="games/canonical_hex8_2p.db",
        data_type=DataType.GAMES,
        config_key="hex8_2p",
        size_bytes=1024000,
        checksum="sha256:abc123...",
        mtime=time.time(),
        locations={"node-1", "node-2"},
        primary_location="node-1",
    ))

    # Query missing data on a node
    missing = catalog.get_missing_on_node("training-node", DataType.NPZ)

    # Check replication factor
    factor = catalog.get_replication_factor("games/canonical_hex8_2p.db")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "DataType",
    "DataEntry",
    "DataCatalog",
    "DataCatalogConfig",
    "get_data_catalog",
    "reset_data_catalog",
]


# December 2025: Import from canonical source (renamed to CatalogDataType)
# Backward-compatible alias DataType retained for existing code
from app.coordination.enums import CatalogDataType
from app.coordination.event_handler_utils import extract_config_from_path
from app.utils.sqlite_utils import connect_safe

# Backward-compatible alias (deprecated, remove Q2 2026)
DataType = CatalogDataType


@dataclass
class DataEntry:
    """Entry in the data catalog representing a piece of data."""

    path: str  # Relative path (e.g., "games/canonical_hex8_2p.db")
    data_type: DataType
    config_key: str  # e.g., "hex8_2p" or "" if not config-specific
    size_bytes: int
    checksum: str  # SHA256 or empty if not computed
    mtime: float  # Modification time (unix timestamp)
    locations: set[str]  # Node IDs where this data exists
    primary_location: str  # Authoritative source node

    # Optional metadata
    game_count: int = 0  # For GAMES type
    sample_count: int = 0  # For NPZ type
    model_version: str = ""  # For MODELS type
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """Ensure locations is a set."""
        if isinstance(self.locations, (list, tuple)):
            self.locations = set(self.locations)
        if isinstance(self.data_type, str):
            self.data_type = DataType(self.data_type)

    @property
    def filename(self) -> str:
        """Get just the filename from path."""
        return Path(self.path).name

    @property
    def replication_factor(self) -> int:
        """Get number of nodes that have this data."""
        return len(self.locations)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "data_type": self.data_type.value,
            "config_key": self.config_key,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "mtime": self.mtime,
            "locations": list(self.locations),
            "primary_location": self.primary_location,
            "game_count": self.game_count,
            "sample_count": self.sample_count,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataEntry:
        """Create from dictionary."""
        return cls(
            path=data["path"],
            data_type=DataType(data["data_type"]),
            config_key=data.get("config_key", ""),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum", ""),
            mtime=data.get("mtime", 0.0),
            locations=set(data.get("locations", [])),
            primary_location=data.get("primary_location", ""),
            game_count=data.get("game_count", 0),
            sample_count=data.get("sample_count", 0),
            model_version=data.get("model_version", ""),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


@dataclass
class DataCatalogConfig:
    """Configuration for DataCatalog."""

    db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("RINGRIFT_CATALOG_DB", "data/coordination/data_catalog.db")
        )
    )
    auto_persist: bool = True  # Auto-save changes to SQLite
    manifest_refresh_interval: float = 60.0  # Seconds between manifest refreshes
    stale_entry_threshold: float = 86400.0  # 24 hours - entries older marked stale
    min_replication_factor: int = 3  # Minimum copies for data safety


class DataCatalog:
    """Central registry of all cluster data locations.

    Maintains a real-time view of what data exists on which nodes.
    Provides efficient queries for sync planning and replication management.

    Thread-safe for concurrent access.
    """

    SCHEMA_VERSION = 1

    def __init__(self, config: DataCatalogConfig | None = None):
        """Initialize the data catalog.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or DataCatalogConfig()
        self._lock = threading.RLock()

        # In-memory indices for fast lookups
        self._entries: dict[str, DataEntry] = {}  # path -> entry
        self._by_node: dict[str, set[str]] = {}  # node_id -> set of paths
        self._by_type: dict[DataType, set[str]] = {}  # type -> set of paths
        self._by_config: dict[str, set[str]] = {}  # config_key -> set of paths

        # Stats tracking
        self._stats = {
            "total_entries": 0,
            "total_bytes": 0,
            "registrations": 0,
            "queries": 0,
            "last_refresh": 0.0,
        }

        # Initialize database
        self._db_initialized = False
        if self.config.auto_persist:
            self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for persistence."""
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            # Create tables
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS data_entries (
                    path TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    config_key TEXT DEFAULT '',
                    size_bytes INTEGER DEFAULT 0,
                    checksum TEXT DEFAULT '',
                    mtime REAL DEFAULT 0,
                    locations TEXT DEFAULT '[]',
                    primary_location TEXT DEFAULT '',
                    game_count INTEGER DEFAULT 0,
                    sample_count INTEGER DEFAULT 0,
                    model_version TEXT DEFAULT '',
                    created_at REAL DEFAULT 0,
                    updated_at REAL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_entries_type ON data_entries(data_type);
                CREATE INDEX IF NOT EXISTS idx_entries_config ON data_entries(config_key);
                CREATE INDEX IF NOT EXISTS idx_entries_updated ON data_entries(updated_at);

                CREATE TABLE IF NOT EXISTS node_manifests (
                    node_id TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL,
                    collected_at REAL NOT NULL,
                    file_count INTEGER DEFAULT 0
                );
                """
            )

            # Check/set schema version
            cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (self.SCHEMA_VERSION,),
                )
            conn.commit()

        self._db_initialized = True
        self._load_from_db()
        logger.info(f"DataCatalog initialized with {len(self._entries)} entries")

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with proper cleanup."""
        conn = connect_safe(self.config.db_path, timeout=30.0)
        try:
            yield conn
        finally:
            conn.close()

    def _load_from_db(self) -> None:
        """Load entries from database into memory."""
        if not self._db_initialized:
            return

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM data_entries")
                for row in cursor:
                    entry = DataEntry(
                        path=row["path"],
                        data_type=DataType(row["data_type"]),
                        config_key=row["config_key"],
                        size_bytes=row["size_bytes"],
                        checksum=row["checksum"],
                        mtime=row["mtime"],
                        locations=set(json.loads(row["locations"])),
                        primary_location=row["primary_location"],
                        game_count=row["game_count"],
                        sample_count=row["sample_count"],
                        model_version=row["model_version"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    self._index_entry(entry)

            self._update_stats()

    def _index_entry(self, entry: DataEntry) -> None:
        """Add entry to in-memory indices (caller must hold lock)."""
        self._entries[entry.path] = entry

        # Index by node
        for node_id in entry.locations:
            if node_id not in self._by_node:
                self._by_node[node_id] = set()
            self._by_node[node_id].add(entry.path)

        # Index by type
        if entry.data_type not in self._by_type:
            self._by_type[entry.data_type] = set()
        self._by_type[entry.data_type].add(entry.path)

        # Index by config
        if entry.config_key:
            if entry.config_key not in self._by_config:
                self._by_config[entry.config_key] = set()
            self._by_config[entry.config_key].add(entry.path)

    def _remove_from_indices(self, entry: DataEntry) -> None:
        """Remove entry from in-memory indices (caller must hold lock)."""
        # Remove from node indices
        for node_id in entry.locations:
            if node_id in self._by_node:
                self._by_node[node_id].discard(entry.path)

        # Remove from type index
        if entry.data_type in self._by_type:
            self._by_type[entry.data_type].discard(entry.path)

        # Remove from config index
        if entry.config_key and entry.config_key in self._by_config:
            self._by_config[entry.config_key].discard(entry.path)

    def _update_stats(self) -> None:
        """Update internal statistics (caller must hold lock)."""
        self._stats["total_entries"] = len(self._entries)
        self._stats["total_bytes"] = sum(e.size_bytes for e in self._entries.values())

    def _persist_entry(self, entry: DataEntry) -> None:
        """Persist entry to database (caller must hold lock)."""
        if not self.config.auto_persist or not self._db_initialized:
            return

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO data_entries
                (path, data_type, config_key, size_bytes, checksum, mtime,
                 locations, primary_location, game_count, sample_count,
                 model_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.path,
                    entry.data_type.value,
                    entry.config_key,
                    entry.size_bytes,
                    entry.checksum,
                    entry.mtime,
                    json.dumps(list(entry.locations)),
                    entry.primary_location,
                    entry.game_count,
                    entry.sample_count,
                    entry.model_version,
                    entry.created_at,
                    entry.updated_at,
                ),
            )
            conn.commit()

    # =========================================================================
    # Public API - Registration
    # =========================================================================

    def register(self, entry: DataEntry) -> None:
        """Register or update a data entry.

        If an entry with the same path exists, it will be updated.
        Locations are merged (union) with existing locations.

        Args:
            entry: The data entry to register.
        """
        with self._lock:
            entry.updated_at = time.time()

            existing = self._entries.get(entry.path)
            if existing:
                # Merge locations
                entry.locations = existing.locations | entry.locations
                # Preserve created_at
                entry.created_at = existing.created_at
                # Remove old indices
                self._remove_from_indices(existing)

            self._index_entry(entry)
            self._persist_entry(entry)
            self._stats["registrations"] += 1
            self._update_stats()

            logger.debug(
                f"Registered: {entry.path} ({entry.data_type.value}) "
                f"on {len(entry.locations)} nodes"
            )

    def register_from_manifest(
        self,
        node_id: str,
        manifest: dict[str, dict[str, Any]],
    ) -> int:
        """Register entries from a node manifest.

        A manifest is a dict mapping relative paths to file info:
        {
            "games/canonical_hex8_2p.db": {
                "size": 1024000,
                "mtime": 1703750000.0,
                "type": "database",
            },
            ...
        }

        Args:
            node_id: The node ID this manifest is from.
            manifest: Dict of path -> file info.

        Returns:
            Number of entries registered.
        """
        count = 0
        for path, info in manifest.items():
            # Infer data type from path or info
            data_type = DataType.from_path(path)
            if info.get("type") == "database":
                data_type = DataType.GAMES
            elif info.get("type") == "model":
                data_type = DataType.MODELS
            elif info.get("type") == "npz":
                data_type = DataType.NPZ

            # Extract config_key from path
            config_key = self._extract_config_key(path)

            entry = DataEntry(
                path=path,
                data_type=data_type,
                config_key=config_key,
                size_bytes=info.get("size", 0),
                checksum=info.get("sha256", info.get("checksum", "")),
                mtime=info.get("mtime", 0.0),
                locations={node_id},
                primary_location=node_id,
                game_count=info.get("game_count", 0),
                sample_count=info.get("sample_count", 0),
            )
            self.register(entry)
            count += 1

        # Store manifest for reference
        if self.config.auto_persist and self._db_initialized:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO node_manifests
                    (node_id, manifest_json, collected_at, file_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (node_id, json.dumps(manifest), time.time(), len(manifest)),
                )
                conn.commit()

        logger.info(f"Registered {count} entries from {node_id} manifest")
        return count

    def _extract_config_key(self, path: str) -> str:
        """Extract config key (e.g., 'hex8_2p') from path."""
        # December 30, 2025: Use consolidated utility for config extraction
        # Common patterns:
        # canonical_hex8_2p.db -> hex8_2p
        # hex8_2p_selfplay.db -> hex8_2p
        # selfplay_square8_4p.db -> square8_4p
        return extract_config_from_path(path) or ""

    def mark_synced(self, path: str, node_id: str) -> None:
        """Mark data as synced to a node.

        Args:
            path: The data path.
            node_id: The node that now has this data.
        """
        with self._lock:
            entry = self._entries.get(path)
            if entry:
                if node_id not in entry.locations:
                    entry.locations.add(node_id)
                    entry.updated_at = time.time()

                    # Update node index
                    if node_id not in self._by_node:
                        self._by_node[node_id] = set()
                    self._by_node[node_id].add(path)

                    self._persist_entry(entry)
                    logger.debug(f"Marked {path} as synced to {node_id}")

    def mark_removed(self, path: str, node_id: str) -> None:
        """Mark data as removed from a node.

        Args:
            path: The data path.
            node_id: The node that no longer has this data.
        """
        with self._lock:
            entry = self._entries.get(path)
            if entry and node_id in entry.locations:
                entry.locations.discard(node_id)
                entry.updated_at = time.time()

                # Update node index
                if node_id in self._by_node:
                    self._by_node[node_id].discard(path)

                self._persist_entry(entry)
                logger.debug(f"Marked {path} as removed from {node_id}")

    # =========================================================================
    # Public API - Queries
    # =========================================================================

    def get(self, path: str) -> DataEntry | None:
        """Get entry by path.

        Args:
            path: The data path to look up.

        Returns:
            DataEntry if found, None otherwise.
        """
        with self._lock:
            self._stats["queries"] += 1
            return self._entries.get(path)

    def get_all(
        self,
        data_type: DataType | None = None,
        config_key: str | None = None,
    ) -> list[DataEntry]:
        """Get all entries, optionally filtered.

        Args:
            data_type: Filter by data type.
            config_key: Filter by config key.

        Returns:
            List of matching entries.
        """
        with self._lock:
            self._stats["queries"] += 1

            # Use indices for faster lookups
            if data_type and config_key:
                paths = self._by_type.get(data_type, set()) & self._by_config.get(
                    config_key, set()
                )
            elif data_type:
                paths = self._by_type.get(data_type, set())
            elif config_key:
                paths = self._by_config.get(config_key, set())
            else:
                paths = set(self._entries.keys())

            return [self._entries[p] for p in paths if p in self._entries]

    def get_on_node(
        self,
        node_id: str,
        data_type: DataType | None = None,
    ) -> list[DataEntry]:
        """Get entries that exist on a specific node.

        Args:
            node_id: The node to query.
            data_type: Optional filter by data type.

        Returns:
            List of entries on the node.
        """
        with self._lock:
            self._stats["queries"] += 1

            paths = self._by_node.get(node_id, set())

            if data_type:
                type_paths = self._by_type.get(data_type, set())
                paths = paths & type_paths

            return [self._entries[p] for p in paths if p in self._entries]

    def get_missing_on_node(
        self,
        node_id: str,
        data_type: DataType | None = None,
        config_key: str | None = None,
    ) -> list[DataEntry]:
        """Get entries that should exist on node but don't.

        This returns entries that exist somewhere in the cluster but
        are not present on the specified node.

        Args:
            node_id: The node to check.
            data_type: Optional filter by data type.
            config_key: Optional filter by config key.

        Returns:
            List of entries missing on the node.
        """
        with self._lock:
            self._stats["queries"] += 1

            # Get all entries matching filters
            all_entries = self.get_all(data_type=data_type, config_key=config_key)

            # Filter to those missing on this node
            return [e for e in all_entries if node_id not in e.locations]

    def get_under_replicated(self, min_factor: int | None = None) -> list[DataEntry]:
        """Get entries with replication factor below minimum.

        Args:
            min_factor: Minimum replication factor. Defaults to config value.

        Returns:
            List of under-replicated entries.
        """
        min_factor = min_factor or self.config.min_replication_factor

        with self._lock:
            self._stats["queries"] += 1
            return [
                e for e in self._entries.values() if e.replication_factor < min_factor
            ]

    def get_replication_factor(self, path: str) -> int:
        """Get number of nodes that have this data.

        Args:
            path: The data path.

        Returns:
            Number of nodes, 0 if not found.
        """
        entry = self.get(path)
        return entry.replication_factor if entry else 0

    def get_nodes_with_data(self, path: str) -> set[str]:
        """Get set of nodes that have this data.

        Args:
            path: The data path.

        Returns:
            Set of node IDs.
        """
        entry = self.get(path)
        return entry.locations.copy() if entry else set()

    def get_primary_location(self, path: str) -> str | None:
        """Get the primary (authoritative) location for data.

        Args:
            path: The data path.

        Returns:
            Node ID of primary location, or None if not found.
        """
        entry = self.get(path)
        return entry.primary_location if entry else None

    def get_stale_entries(self, threshold: float | None = None) -> list[DataEntry]:
        """Get entries that haven't been updated recently.

        Args:
            threshold: Max age in seconds. Defaults to config value.

        Returns:
            List of stale entries.
        """
        threshold = threshold or self.config.stale_entry_threshold
        cutoff = time.time() - threshold

        with self._lock:
            self._stats["queries"] += 1
            return [e for e in self._entries.values() if e.updated_at < cutoff]

    # =========================================================================
    # Public API - Statistics & Health
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get catalog statistics.

        Returns:
            Dict with stats including entry counts, bytes, queries, etc.
        """
        with self._lock:
            return {
                **self._stats,
                "entries_by_type": {
                    t.value: len(paths) for t, paths in self._by_type.items()
                },
                "nodes_tracked": len(self._by_node),
                "configs_tracked": len(self._by_config),
            }

    def health_check(self) -> "HealthCheckResult":
        """Check catalog health.

        Returns:
            HealthCheckResult with status and details.
        """
        # Import from contracts (zero dependencies)
        from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

        with self._lock:
            stats = self.get_stats()

            # Check for issues
            issues = []
            if stats["total_entries"] == 0:
                issues.append("No entries registered")

            under_rep = len(self.get_under_replicated())
            if under_rep > 0:
                issues.append(f"{under_rep} entries under-replicated")

            stale = len(self.get_stale_entries())
            if stale > stats["total_entries"] * 0.5 and stats["total_entries"] > 0:
                issues.append(f"{stale} stale entries (>50%)")

            healthy = len(issues) == 0
            status = CoordinatorStatus.RUNNING if healthy else CoordinatorStatus.DEGRADED

            return HealthCheckResult(
                healthy=healthy,
                status=status,
                message="; ".join(issues) if issues else "Catalog healthy",
                details={
                    **stats,
                    "under_replicated": under_rep,
                    "stale_entries": stale,
                },
            )

    def clear(self) -> None:
        """Clear all entries (for testing)."""
        with self._lock:
            self._entries.clear()
            self._by_node.clear()
            self._by_type.clear()
            self._by_config.clear()
            self._update_stats()

            if self.config.auto_persist and self._db_initialized:
                with self._get_connection() as conn:
                    conn.execute("DELETE FROM data_entries")
                    conn.commit()

            logger.info("DataCatalog cleared")


# =============================================================================
# Module-level singleton
# =============================================================================

_catalog_singleton: DataCatalog | None = None
_catalog_lock = threading.Lock()


def get_data_catalog(config: DataCatalogConfig | None = None) -> DataCatalog:
    """Get the global DataCatalog singleton.

    Args:
        config: Configuration for catalog. Only used on first call.

    Returns:
        The global DataCatalog instance.
    """
    global _catalog_singleton

    with _catalog_lock:
        if _catalog_singleton is None:
            _catalog_singleton = DataCatalog(config)
        return _catalog_singleton


def reset_data_catalog() -> None:
    """Reset the catalog singleton (for testing)."""
    global _catalog_singleton

    with _catalog_lock:
        _catalog_singleton = None
