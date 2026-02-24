"""Cluster Manifest - Central Registry for Data Locations Across the Cluster.

This module provides a unified registry that tracks where games, models, and NPZ
files exist across all cluster nodes. Unlike DataManifest (which tracks what
has been synced TO a node), ClusterManifest tracks WHERE data exists.

Key features:
1. Game location registry (game_id -> [node_id, db_path] mappings)
2. Model location registry (model_path -> [node_id] mappings)
3. NPZ file registry (npz_path -> [node_id] mappings)
4. Disk usage awareness (max 70% disk utilization)
5. Node exclusion rules (dev machines, local Macs except external drives)
6. Replication target selection based on capacity and role
7. P2P gossip for manifest propagation

Usage:
    from app.distributed.cluster_manifest import (
        ClusterManifest,
        GameLocation,
        NodeInventory,
        get_cluster_manifest,
    )

    manifest = get_cluster_manifest()

    # Register a game location
    manifest.register_game("game-123", "gh200-a", "/data/games/selfplay.db")

    # Find all locations for a game
    locations = manifest.find_game("game-123")

    # Get replication targets for a game
    targets = manifest.get_replication_targets("game-123", min_copies=2)

    # Check node inventory
    inventory = manifest.get_node_inventory("gh200-a")
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generator

from app.config.cluster_config import load_cluster_config
from app.distributed.cluster_config_manager import (
    ClusterConfigManager,
    NodeSyncPolicy as ConfigNodeSyncPolicy,
)
from app.distributed.data_location_registry import DataLocationRegistry
from app.distributed.node_capacity_manager import (
    NodeCapacityManager,
    NodeCapacity,
    NodeInventory,
)
from app.distributed.sync_target_selector import (
    SyncTargetSelector,
    SyncCandidateNode as SelectorSyncCandidateNode,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Enums
    "DataType",
    "NodeRole",
    "DataSource",  # January 2026: S3/OWC unified sync
    # Data classes
    "GameLocation",
    "ModelLocation",
    "NPZLocation",
    "CheckpointLocation",
    "SyncReceipt",
    "TorrentMetadata",
    "ExternalStorageLocation",  # January 2026: S3/OWC tracking
    "NodeCapacity",
    "NodeInventory",
    "SyncCandidateNode",
    # Backwards compatibility alias (historical name)
    "SyncTarget",
    "DiskCleanupPolicy",
    "DiskCleanupResult",
    "CleanupCandidate",
    "NodeSyncPolicy",
    # Main class
    "ClusterManifest",
    # Singleton accessors
    "get_cluster_manifest",
    "reset_cluster_manifest",
    # Constants
    "MANIFEST_DB_NAME",
    "SCHEMA_VERSION",
    "MAX_DISK_USAGE_PERCENT",
    "MIN_FREE_DISK_PERCENT",
    "REPLICATION_TARGET_COUNT",
]

# Constants
MANIFEST_DB_NAME = "cluster_manifest.db"
SCHEMA_VERSION = "1.0"
try:
    from app.config.thresholds import DISK_SYNC_TARGET_PERCENT
    MAX_DISK_USAGE_PERCENT = DISK_SYNC_TARGET_PERCENT  # Don't sync to nodes above this usage
except ImportError:
    MAX_DISK_USAGE_PERCENT = 70  # Don't sync to nodes above this usage
MIN_FREE_DISK_PERCENT = 30  # Ensure at least this much free space
REPLICATION_TARGET_COUNT = 2  # Default replication count


# =============================================================================
# Data Classes
# =============================================================================


class DataType(str, Enum):
    """Types of data tracked in the manifest."""
    GAME = "game"
    MODEL = "model"
    NPZ = "npz"
    CHECKPOINT = "checkpoint"


class NodeRole(str, Enum):
    """Node roles for sync targeting."""
    TRAINING = "training"
    SELFPLAY = "selfplay"
    COORDINATOR = "coordinator"
    STORAGE = "storage"
    EXCLUDED = "excluded"


class DataSource(str, Enum):
    """Source of data in the cluster.

    January 2026: Added for unified data sync across all storage locations.
    """
    LOCAL = "local"  # Local filesystem on a cluster node
    P2P = "p2p"  # P2P network (other cluster nodes)
    S3 = "s3"  # Amazon S3 bucket
    OWC = "owc"  # OWC external drive (mac-studio)


@dataclass
class GameLocation:
    """Location of a game in the cluster.

    December 2025: Added is_consolidated and consolidated_at fields to track
    whether games have been merged into canonical databases.
    """
    game_id: str
    node_id: str
    db_path: str
    board_type: str | None = None
    num_players: int | None = None
    engine_mode: str | None = None
    registered_at: float = 0.0
    last_seen: float = 0.0
    # December 2025: Track consolidation status for training pipeline
    is_consolidated: bool = False  # Whether game has been merged into canonical DB
    consolidated_at: float = 0.0  # When consolidation happened (Unix timestamp)
    canonical_db: str | None = None  # Path to canonical DB if consolidated


@dataclass
class ModelLocation:
    """Location of a model in the cluster."""
    model_path: str  # Relative path (e.g., "models/canonical_hex8_2p.pth")
    node_id: str
    board_type: str | None = None
    num_players: int | None = None
    model_version: str | None = None
    file_size: int = 0
    registered_at: float = 0.0
    last_seen: float = 0.0


@dataclass
class NPZLocation:
    """Location of an NPZ training file in the cluster."""
    npz_path: str  # Relative path (e.g., "data/training/hex8_2p.npz")
    node_id: str
    board_type: str | None = None
    num_players: int | None = None
    sample_count: int = 0
    file_size: int = 0
    registered_at: float = 0.0
    last_seen: float = 0.0


@dataclass
class CheckpointLocation:
    """Location of a training checkpoint in the cluster.

    Checkpoints include optimizer state, scheduler state, and training metadata
    needed to resume training from a specific point.
    """
    checkpoint_path: str  # Relative path (e.g., "checkpoints/hex8_2p/epoch_50.pth")
    node_id: str
    config_key: str | None = None  # e.g., "hex8_2p"
    board_type: str | None = None
    num_players: int | None = None
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    file_size: int = 0
    registered_at: float = 0.0
    last_seen: float = 0.0
    is_best: bool = False  # Whether this is the best checkpoint for this config


@dataclass
class SyncReceipt:
    """Receipt confirming a file has been synced to a destination node.

    December 2025: Added for push-based sync with verified cleanup.
    GPU nodes push data to coordinator, receive receipts confirming sync.
    Files are only deleted locally after N verified receipts exist.

    The checksum is SHA256 of the file contents, used to verify integrity.
    """
    file_path: str  # Relative path (e.g., "data/games/selfplay_hex8.db")
    file_checksum: str  # SHA256 hash of file contents
    synced_to: str  # Destination node_id
    synced_at: float  # Unix timestamp when sync completed
    verified: bool = False  # Whether checksum was verified at destination
    file_size: int = 0  # Size in bytes (for reporting)
    source_node: str = ""  # Node that initiated the push


@dataclass
class TorrentMetadata:
    """Metadata for a BitTorrent swarm tracking a file.

    Enables resilient P2P file sync across cluster nodes by tracking:
    - Which nodes are seeding each file (swarm membership)
    - Web seed URLs for hybrid HTTP+BitTorrent downloads
    - Torrent file location for aria2 downloads
    """
    info_hash: str  # SHA1 hash of torrent info dict (40 hex chars)
    file_path: str  # Relative path to the data file
    torrent_path: str  # Path to the .torrent file
    file_size: int = 0
    piece_size: int = 262144  # Default 256KB pieces
    piece_count: int = 0
    seeders: list[str] = field(default_factory=list)  # Node IDs currently seeding
    web_seeds: list[str] = field(default_factory=list)  # HTTP fallback URLs
    created_at: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "info_hash": self.info_hash,
            "file_path": self.file_path,
            "torrent_path": self.torrent_path,
            "file_size": self.file_size,
            "piece_size": self.piece_size,
            "piece_count": self.piece_count,
            "seeders": self.seeders,
            "web_seeds": self.web_seeds,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TorrentMetadata":
        """Create from dictionary."""
        return cls(
            info_hash=data["info_hash"],
            file_path=data["file_path"],
            torrent_path=data.get("torrent_path", ""),
            file_size=data.get("file_size", 0),
            piece_size=data.get("piece_size", 262144),
            piece_count=data.get("piece_count", 0),
            seeders=data.get("seeders", []),
            web_seeds=data.get("web_seeds", []),
            created_at=data.get("created_at", 0.0),
            last_seen=data.get("last_seen", 0.0),
        )


@dataclass
class ExternalStorageLocation:
    """Location of data in external storage (S3 or OWC).

    January 2026: Added for unified data sync tracking.
    Represents a database or file stored in S3 or OWC external drive.
    """
    config_key: str  # e.g., "hex8_2p"
    source: DataSource  # S3 or OWC
    path: str  # S3 key or OWC path
    game_count: int = 0
    file_size: int = 0
    board_type: str | None = None
    num_players: int | None = None
    registered_at: float = 0.0
    last_verified: float = 0.0
    # S3-specific fields
    s3_bucket: str | None = None
    # OWC-specific fields
    owc_host: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_key": self.config_key,
            "source": self.source.value,
            "path": self.path,
            "game_count": self.game_count,
            "file_size": self.file_size,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "registered_at": self.registered_at,
            "last_verified": self.last_verified,
            "s3_bucket": self.s3_bucket,
            "owc_host": self.owc_host,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExternalStorageLocation":
        """Create from dictionary."""
        return cls(
            config_key=data["config_key"],
            source=DataSource(data["source"]),
            path=data["path"],
            game_count=data.get("game_count", 0),
            file_size=data.get("file_size", 0),
            board_type=data.get("board_type"),
            num_players=data.get("num_players"),
            registered_at=data.get("registered_at", 0.0),
            last_verified=data.get("last_verified", 0.0),
            s3_bucket=data.get("s3_bucket"),
            owc_host=data.get("owc_host"),
        )


# NodeCapacity and NodeInventory imported from node_capacity_manager.py
# (December 2025 extraction for modularity)


@dataclass
class SyncCandidateNode:
    """A potential node for syncing data.

    Note: Renamed from SyncTarget (Dec 2025) to avoid collision with
    app.coordination.sync_constants.SyncTarget which is for SSH connection specs.
    """
    node_id: str
    priority: int = 0  # Higher = sync first
    reason: str = ""
    capacity: NodeCapacity | None = None


# Backwards compatibility: older code/tests import SyncTarget from this module.
# Keep this alias to avoid breaking runtime imports.
SyncTarget = SyncCandidateNode


@dataclass
class DiskCleanupPolicy:
    """Policy for disk cleanup when usage exceeds threshold.

    Cleanup prioritizes removing data that is:
    1. Old and not recently accessed
    2. Uses outdated schemas (older than current)
    3. Low quality (poor training value)
    4. Already replicated elsewhere
    """
    trigger_usage_percent: float = MAX_DISK_USAGE_PERCENT  # Trigger at 70%
    target_usage_percent: float = 60.0  # Clean down to 60%
    min_age_days: int = 7  # Don't delete data newer than this
    # Phase 9 (Dec 2025): Increased from 2 to 3 for better data safety
    min_replicas_before_delete: int = 3  # Only delete if replicated to 3+ places
    prefer_low_quality: bool = True  # Prefer deleting low-quality games
    prefer_old_schema: bool = True  # Prefer deleting games with old schema
    preserve_canonical: bool = True  # Never delete canonical databases
    dry_run: bool = False  # If True, report what would be deleted without deleting


@dataclass
class DiskCleanupResult:
    """Result of a disk cleanup operation."""
    triggered: bool = False  # Whether cleanup was triggered
    initial_usage_percent: float = 0.0
    final_usage_percent: float = 0.0
    bytes_freed: int = 0
    games_deleted: int = 0
    databases_deleted: int = 0
    npz_deleted: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def success(self) -> bool:
        """Check if cleanup succeeded (reached target or made progress)."""
        return len(self.errors) == 0 and (
            not self.triggered or self.bytes_freed > 0
        )


@dataclass
class CleanupCandidate:
    """A candidate for disk cleanup."""
    path: Path
    data_type: DataType
    size_bytes: int
    age_days: float
    quality_score: float = 0.5
    schema_version: int = 0
    replication_count: int = 0
    is_canonical: bool = False
    board_type: str = ""
    num_players: int = 0
    game_count: int = 0

    @property
    def cleanup_priority(self) -> float:
        """Compute cleanup priority (higher = delete first).

        Priority factors:
        - Age: older data has higher priority
        - Quality: low-quality data has higher priority
        - Schema: older schema has higher priority
        - Replication: more replicas = safer to delete
        - Canonical: never delete
        """
        if self.is_canonical:
            return -1000.0  # Never delete canonical

        priority = 0.0

        # Age factor: older is higher priority
        priority += min(self.age_days / 30.0, 5.0) * 10  # Max 50 points

        # Quality factor: lower quality is higher priority
        priority += (1.0 - self.quality_score) * 30  # Max 30 points

        # Schema factor: older schema is higher priority
        if self.schema_version > 0:
            # Assume current schema is ~12, older versions get higher priority
            schema_age = max(0, 12 - self.schema_version)
            priority += schema_age * 5  # 5 points per version behind

        # Replication factor: more replicas = safer to delete
        priority += min(self.replication_count, 5) * 5  # Max 25 points

        return priority


@dataclass
class NodeSyncPolicy:
    """Sync policy for a node."""
    node_id: str
    receive_games: bool = True
    receive_models: bool = True
    receive_npz: bool = True
    max_disk_usage_percent: float = MAX_DISK_USAGE_PERCENT
    excluded: bool = False
    exclusion_reason: str = ""


# =============================================================================
# Helper Functions
# =============================================================================


def _cleanup_orphaned_wal_files(db_path: Path) -> bool:
    """Remove orphaned WAL/SHM files if main database doesn't exist.

    SQLite cannot open WAL files without the main database.
    This can happen when the database is deleted but WAL/SHM files remain.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        True if any orphaned files were removed, False otherwise.
    """
    wal_path = db_path.with_suffix(db_path.suffix + "-wal")
    shm_path = db_path.with_suffix(db_path.suffix + "-shm")

    cleaned = False
    if not db_path.exists():
        for path in [wal_path, shm_path]:
            if path.exists():
                try:
                    path.unlink()
                    logger.warning(f"[ClusterManifest] Removed orphaned: {path}")
                    cleaned = True
                except OSError as e:
                    logger.error(f"[ClusterManifest] Failed to remove orphaned file {path}: {e}")
    return cleaned


# =============================================================================
# ClusterManifest Class
# =============================================================================


class ClusterManifest:
    """Central registry tracking data locations across the cluster.

    Provides:
    - Game ID to node location mappings
    - Model path to node location mappings
    - NPZ file to node location mappings
    - Disk capacity tracking per node
    - Replication target selection
    - Node exclusion rules
    """

    def __init__(self, db_path: Path | None = None, config_path: Path | None = None):
        """Initialize the cluster manifest.

        Args:
            db_path: Path to SQLite database (default: data/cluster_manifest.db)
            config_path: Path to distributed_hosts.yaml
        """
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            db_path = base_dir / "data" / MANIFEST_DB_NAME

        self.db_path = db_path
        self.node_id = socket.gethostname()
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()  # RLock allows reentrant locking (e.g., get_node_inventory -> get_node_capacity)

        # Load host configuration via ClusterConfigManager
        self._config_manager = ClusterConfigManager(config_path)

        # Initialize database
        self._init_db()

        # Initialize data location registry (delegated component)
        self._registry = DataLocationRegistry(
            db_path=self.db_path,
            connection_factory=self._connection,
            node_id=self.node_id,
        )

        # Initialize node capacity manager (delegated component)
        self._capacity_manager = NodeCapacityManager(
            db_path=self.db_path,
            connection_factory=self._connection,
            node_id=self.node_id,
            hosts_config=self._hosts_config,
        )

        # Initialize sync target selector (delegated component)
        self._sync_selector = SyncTargetSelector(
            capacity_manager=self._capacity_manager,
            config_manager=self._config_manager,
            registry=self._registry,
            connection_factory=self._connection,
            hosts_config=self._hosts_config,
        )

        logger.info(f"ClusterManifest initialized: node={self.node_id}, db={db_path}")

    # -------------------------------------------------------------------------
    # Config accessors (delegated to ClusterConfigManager)
    # -------------------------------------------------------------------------

    @property
    def _hosts_config(self) -> dict[str, Any]:
        """Get hosts configuration (delegated to ClusterConfigManager)."""
        return self._config_manager.hosts_config

    @property
    def _exclusion_rules(self) -> dict[str, ConfigNodeSyncPolicy]:
        """Get exclusion rules (delegated to ClusterConfigManager)."""
        return self._config_manager.get_all_policies()

    @property
    def _max_disk_usage(self) -> float:
        """Get max disk usage percent (delegated to ClusterConfigManager)."""
        return self._config_manager.max_disk_usage_percent

    @property
    def _priority_hosts(self) -> set[str]:
        """Get priority hosts (delegated to ClusterConfigManager)."""
        return self._config_manager.priority_hosts

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with thread safety."""
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    timeout=30.0,
                )
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            yield self._conn

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _init_db(self) -> None:
        """Initialize the manifest database schema."""
        # Clean up orphaned WAL/SHM files before opening DB
        _cleanup_orphaned_wal_files(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Phase 1: Create tables (without indexes that depend on new columns)
        cursor.executescript("""
            -- Game locations table
            CREATE TABLE IF NOT EXISTS game_locations (
                game_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                db_path TEXT NOT NULL,
                board_type TEXT,
                num_players INTEGER,
                engine_mode TEXT,
                registered_at REAL NOT NULL,
                last_seen REAL NOT NULL,
                -- December 2025: Consolidation tracking for training pipeline
                is_consolidated INTEGER DEFAULT 0,
                consolidated_at REAL DEFAULT 0,
                canonical_db TEXT,
                PRIMARY KEY (game_id, node_id)
            );

            -- Model locations table
            CREATE TABLE IF NOT EXISTS model_locations (
                model_path TEXT NOT NULL,
                node_id TEXT NOT NULL,
                board_type TEXT,
                num_players INTEGER,
                model_version TEXT,
                file_size INTEGER DEFAULT 0,
                registered_at REAL NOT NULL,
                last_seen REAL NOT NULL,
                PRIMARY KEY (model_path, node_id)
            );

            -- NPZ locations table
            CREATE TABLE IF NOT EXISTS npz_locations (
                npz_path TEXT NOT NULL,
                node_id TEXT NOT NULL,
                board_type TEXT,
                num_players INTEGER,
                sample_count INTEGER DEFAULT 0,
                file_size INTEGER DEFAULT 0,
                registered_at REAL NOT NULL,
                last_seen REAL NOT NULL,
                PRIMARY KEY (npz_path, node_id)
            );

            -- Database locations table (Phase 4A.3 - December 2025)
            -- Tracks database files for immediate visibility (no 5-min orphan scan delay)
            CREATE TABLE IF NOT EXISTS database_locations (
                db_path TEXT NOT NULL,
                node_id TEXT NOT NULL,
                board_type TEXT,
                num_players INTEGER,
                config_key TEXT,
                game_count INTEGER DEFAULT 0,
                file_size INTEGER DEFAULT 0,
                engine_mode TEXT,
                registered_at REAL NOT NULL,
                last_seen REAL NOT NULL,
                PRIMARY KEY (db_path, node_id)
            );

            -- Checkpoint locations table (December 2025)
            -- Tracks training checkpoints for distributed training resume/failover
            CREATE TABLE IF NOT EXISTS checkpoint_locations (
                checkpoint_path TEXT NOT NULL,
                node_id TEXT NOT NULL,
                config_key TEXT,
                board_type TEXT,
                num_players INTEGER,
                epoch INTEGER DEFAULT 0,
                step INTEGER DEFAULT 0,
                loss REAL DEFAULT 0.0,
                file_size INTEGER DEFAULT 0,
                is_best INTEGER DEFAULT 0,
                registered_at REAL NOT NULL,
                last_seen REAL NOT NULL,
                PRIMARY KEY (checkpoint_path, node_id)
            );

            -- Node capacity table
            CREATE TABLE IF NOT EXISTS node_capacity (
                node_id TEXT PRIMARY KEY,
                total_bytes INTEGER DEFAULT 0,
                used_bytes INTEGER DEFAULT 0,
                free_bytes INTEGER DEFAULT 0,
                usage_percent REAL DEFAULT 0.0,
                last_updated REAL NOT NULL
            );

            -- Torrent metadata table (December 2025)
            -- Tracks BitTorrent swarms for resilient P2P file sync
            CREATE TABLE IF NOT EXISTS torrent_metadata (
                info_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                torrent_path TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                piece_size INTEGER DEFAULT 262144,
                piece_count INTEGER DEFAULT 0,
                seeders TEXT DEFAULT '[]',  -- JSON array of node IDs
                web_seeds TEXT DEFAULT '[]',  -- JSON array of HTTP URLs
                created_at REAL NOT NULL,
                last_seen REAL NOT NULL
            );

            -- Sync receipts table (December 2025)
            -- Tracks verified syncs for safe cleanup on GPU nodes
            -- Files are only deleted after N verified receipts exist
            CREATE TABLE IF NOT EXISTS sync_receipts (
                file_path TEXT NOT NULL,
                file_checksum TEXT NOT NULL,  -- SHA256 hash
                synced_to TEXT NOT NULL,      -- Destination node_id
                synced_at REAL NOT NULL,      -- Timestamp
                verified INTEGER DEFAULT 0,   -- Checksum verified at destination
                file_size INTEGER DEFAULT 0,  -- For reporting
                source_node TEXT DEFAULT '',  -- Node that pushed
                PRIMARY KEY (file_path, synced_to)
            );

            -- External storage locations table (January 2026)
            -- Tracks data in S3 and OWC external drive for unified visibility
            CREATE TABLE IF NOT EXISTS external_storage_locations (
                config_key TEXT NOT NULL,       -- e.g., "hex8_2p"
                source TEXT NOT NULL,           -- "s3" or "owc"
                path TEXT NOT NULL,             -- S3 key or OWC path
                game_count INTEGER DEFAULT 0,
                file_size INTEGER DEFAULT 0,
                board_type TEXT,
                num_players INTEGER,
                registered_at REAL NOT NULL,
                last_verified REAL DEFAULT 0,
                s3_bucket TEXT,                 -- For S3 entries
                owc_host TEXT,                  -- For OWC entries
                PRIMARY KEY (config_key, source, path)
            );

            -- Metadata table
            CREATE TABLE IF NOT EXISTS manifest_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        conn.commit()

        # Phase 2: Migrate existing databases to add new columns
        # This handles databases created before is_consolidated was added
        # Must run BEFORE creating indexes that depend on these columns
        self._migrate_schema(conn)

        # Phase 3: Create indexes (including ones that depend on migrated columns)
        cursor.executescript(f"""
            -- Indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_game_locations_node
                ON game_locations(node_id);
            CREATE INDEX IF NOT EXISTS idx_game_locations_board
                ON game_locations(board_type, num_players);
            -- December 2025: Index for unconsolidated games (training pipeline)
            CREATE INDEX IF NOT EXISTS idx_game_locations_consolidated
                ON game_locations(is_consolidated, board_type, num_players);
            CREATE INDEX IF NOT EXISTS idx_model_locations_node
                ON model_locations(node_id);
            CREATE INDEX IF NOT EXISTS idx_model_locations_board
                ON model_locations(board_type, num_players);
            CREATE INDEX IF NOT EXISTS idx_npz_locations_node
                ON npz_locations(node_id);
            CREATE INDEX IF NOT EXISTS idx_npz_locations_board
                ON npz_locations(board_type, num_players);
            CREATE INDEX IF NOT EXISTS idx_database_locations_node
                ON database_locations(node_id);
            CREATE INDEX IF NOT EXISTS idx_database_locations_config
                ON database_locations(config_key);
            CREATE INDEX IF NOT EXISTS idx_database_locations_board
                ON database_locations(board_type, num_players);
            CREATE INDEX IF NOT EXISTS idx_checkpoint_locations_node
                ON checkpoint_locations(node_id);
            CREATE INDEX IF NOT EXISTS idx_checkpoint_locations_config
                ON checkpoint_locations(config_key);
            CREATE INDEX IF NOT EXISTS idx_checkpoint_locations_best
                ON checkpoint_locations(config_key, is_best);
            CREATE INDEX IF NOT EXISTS idx_torrent_metadata_file
                ON torrent_metadata(file_path);
            CREATE INDEX IF NOT EXISTS idx_sync_receipts_file
                ON sync_receipts(file_path);
            CREATE INDEX IF NOT EXISTS idx_sync_receipts_verified
                ON sync_receipts(file_path, verified);
            -- January 2026: External storage indexes for unified data discovery
            CREATE INDEX IF NOT EXISTS idx_external_storage_config
                ON external_storage_locations(config_key);
            CREATE INDEX IF NOT EXISTS idx_external_storage_source
                ON external_storage_locations(source);

            -- Initialize metadata
            INSERT OR IGNORE INTO manifest_metadata (key, value, updated_at)
            VALUES
                ('schema_version', '{SCHEMA_VERSION}', {time.time()}),
                ('created_at', '{time.time()}', {time.time()});
        """)
        conn.commit()
        conn.close()

        logger.debug(f"Initialized cluster manifest at {self.db_path}")

        # January 2026: Real-time event subscriptions for immediate manifest updates
        self._subscribed = False
        self._subscription_ids: list[str] = []

    # =========================================================================
    # Real-Time Event Subscriptions (January 2026)
    # =========================================================================

    def subscribe_to_events(self) -> bool:
        """Subscribe to data events for real-time manifest updates.

        Subscribes to:
        - NEW_GAMES_AVAILABLE: New games generated locally
        - GAMES_UPLOADED_TO_S3: Games uploaded to S3
        - GAMES_UPLOADED_TO_OWC: Games uploaded to OWC drive

        This reduces manifest lag from 5 minutes (gossip interval) to near-real-time
        for locally-generated data.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router is None:
                logger.debug("[ClusterManifest] Event router not available")
                return False

            # Subscribe to NEW_GAMES_AVAILABLE for local game registrations
            try:
                from app.distributed.data_events import DataEventType
                router.subscribe(
                    DataEventType.NEW_GAMES_AVAILABLE,
                    self._on_new_games_available,
                )
                logger.debug("[ClusterManifest] Subscribed to NEW_GAMES_AVAILABLE")
            except ImportError:
                logger.debug("[ClusterManifest] DataEventType not available")

            # Subscribe to upload completion events (custom string events)
            router.subscribe("GAMES_UPLOADED_TO_S3", self._on_s3_upload)
            router.subscribe("GAMES_UPLOADED_TO_OWC", self._on_owc_upload)

            # January 2026: Subscribe to BACKUP_COMPLETED from UnifiedBackupDaemon
            # This provides per-database details for S3/OWC registration
            router.subscribe("BACKUP_COMPLETED", self._on_backup_completed)

            self._subscribed = True
            logger.info(
                "[ClusterManifest] Subscribed to real-time events "
                "(NEW_GAMES_AVAILABLE, GAMES_UPLOADED_TO_S3, GAMES_UPLOADED_TO_OWC, "
                "BACKUP_COMPLETED)"
            )
            return True

        except ImportError as e:
            logger.debug(f"[ClusterManifest] Event router not importable: {e}")
            return False
        except Exception as e:
            logger.warning(f"[ClusterManifest] Failed to subscribe to events: {e}")
            return False

    async def _on_new_games_available(self, event: Any) -> None:
        """Handle NEW_GAMES_AVAILABLE event - register games immediately.

        This bypasses the 5-minute gossip interval for locally-generated games,
        making them visible in the manifest immediately.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            db_path = payload.get("db_path", "")
            game_count = payload.get("game_count", 0)
            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            config_key = payload.get("config_key", "")

            if not db_path:
                return

            # Register the database immediately
            self.register_database(
                db_path=db_path,
                node_id=self.node_id,
                board_type=board_type,
                num_players=num_players,
                config_key=config_key,
                game_count=game_count,
            )

            logger.debug(
                f"[ClusterManifest] Immediate registration: {db_path} "
                f"({game_count} games, {config_key or 'unknown config'})"
            )

        except Exception as e:
            logger.debug(f"[ClusterManifest] Error handling NEW_GAMES_AVAILABLE: {e}")

    async def _on_s3_upload(self, event: Any) -> None:
        """Handle GAMES_UPLOADED_TO_S3 event - track S3 upload completion.

        Updates manifest to note that games are now also available in S3.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            db_path = payload.get("db_path", "")
            s3_key = payload.get("s3_key", "")
            game_count = payload.get("game_count", 0)
            config_key = payload.get("config_key", "")

            if not db_path or not s3_key:
                return

            # Register the S3 location as a virtual node
            # Use "s3" as node_id to distinguish from cluster nodes
            self.register_database(
                db_path=s3_key,
                node_id="s3",
                config_key=config_key,
                game_count=game_count,
            )

            logger.debug(
                f"[ClusterManifest] Registered S3 upload: {s3_key} "
                f"({game_count} games)"
            )

        except Exception as e:
            logger.debug(f"[ClusterManifest] Error handling GAMES_UPLOADED_TO_S3: {e}")

    async def _on_owc_upload(self, event: Any) -> None:
        """Handle GAMES_UPLOADED_TO_OWC event - track OWC upload completion.

        Updates manifest to note that games are now also available on OWC drive.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            db_path = payload.get("db_path", "")
            dest_path = payload.get("dest_path", "")
            game_count = payload.get("game_count", 0)
            config_key = payload.get("config_key", "")

            if not db_path or not dest_path:
                return

            # Register the OWC location as a virtual node
            # Use "owc" as node_id to distinguish from cluster nodes
            self.register_database(
                db_path=dest_path,
                node_id="owc",
                config_key=config_key,
                game_count=game_count,
            )

            logger.debug(
                f"[ClusterManifest] Registered OWC upload: {dest_path} "
                f"({game_count} games)"
            )

        except Exception as e:
            logger.debug(f"[ClusterManifest] Error handling GAMES_UPLOADED_TO_OWC: {e}")

    async def _on_backup_completed(self, event: Any) -> None:
        """Handle BACKUP_COMPLETED event from UnifiedBackupDaemon.

        Registers S3 and OWC locations for each database that was backed up.
        This provides unified visibility of data across all storage locations.

        Expected payload structure (from unified_backup_daemon.py):
        {
            "source": "unified_backup_daemon",
            "success": True,
            "backup_details": [
                {
                    "config_key": "hex8_2p",
                    "game_count": 1234,
                    "db_path": "/path/to/local.db",
                    "owc_path": "/Volumes/RingRift-Data/...",  # if backed up
                    "owc_host": "mac-studio",                  # if backed up
                    "s3_key": "databases/...",                 # if backed up
                    "s3_bucket": "ringrift-models-..."         # if backed up
                },
                ...
            ]
        }
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event

            # Skip failed backups
            if not payload.get("success", False):
                return

            backup_details = payload.get("backup_details", [])
            if not backup_details:
                return

            s3_registered = 0
            owc_registered = 0

            for detail in backup_details:
                config_key = detail.get("config_key", "")
                game_count = detail.get("game_count", 0)

                if not config_key:
                    continue

                # Register S3 location if backed up to S3
                s3_key = detail.get("s3_key")
                s3_bucket = detail.get("s3_bucket")
                if s3_key and s3_bucket:
                    self.register_s3_location(
                        s3_key=s3_key,
                        config_key=config_key,
                        game_count=game_count,
                        s3_bucket=s3_bucket,
                    )
                    s3_registered += 1

                # Register OWC location if backed up to OWC
                owc_path = detail.get("owc_path")
                owc_host = detail.get("owc_host")
                if owc_path and owc_host:
                    self.register_owc_location(
                        owc_path=owc_path,
                        config_key=config_key,
                        game_count=game_count,
                        owc_host=owc_host,
                    )
                    owc_registered += 1

            if s3_registered > 0 or owc_registered > 0:
                logger.info(
                    f"[ClusterManifest] BACKUP_COMPLETED: registered "
                    f"{s3_registered} S3 + {owc_registered} OWC locations"
                )

        except Exception as e:
            logger.warning(f"[ClusterManifest] Error handling BACKUP_COMPLETED: {e}")

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Apply schema migrations for existing databases.

        This handles databases created before new columns were added.
        Uses safe ALTER TABLE IF NOT EXISTS pattern.
        """
        cursor = conn.cursor()

        # Check existing columns in game_locations
        cursor.execute("PRAGMA table_info(game_locations)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Add is_consolidated column if missing (December 2025)
        if 'is_consolidated' not in existing_columns:
            try:
                cursor.execute("""
                    ALTER TABLE game_locations
                    ADD COLUMN is_consolidated INTEGER DEFAULT 0
                """)
                logger.info("Migrated game_locations: added is_consolidated column")
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Add consolidated_at column if missing (December 2025)
        if 'consolidated_at' not in existing_columns:
            try:
                cursor.execute("""
                    ALTER TABLE game_locations
                    ADD COLUMN consolidated_at REAL DEFAULT 0
                """)
                logger.info("Migrated game_locations: added consolidated_at column")
            except sqlite3.OperationalError:
                pass

        # Add canonical_db column if missing (December 2025)
        if 'canonical_db' not in existing_columns:
            try:
                cursor.execute("""
                    ALTER TABLE game_locations
                    ADD COLUMN canonical_db TEXT
                """)
                logger.info("Migrated game_locations: added canonical_db column")
            except sqlite3.OperationalError:
                pass

        conn.commit()

    # =========================================================================
    # Game Location Registry (delegated to DataLocationRegistry)
    # =========================================================================

    def register_game(
        self,
        game_id: str,
        node_id: str,
        db_path: str,
        board_type: str | None = None,
        num_players: int | None = None,
        engine_mode: str | None = None,
    ) -> None:
        """Register a game location in the manifest."""
        return self._registry.register_game(
            game_id, node_id, db_path, board_type, num_players, engine_mode
        )

    def register_games_batch(
        self,
        games: list[tuple[str, str, str]],
        board_type: str | None = None,
        num_players: int | None = None,
        engine_mode: str | None = None,
    ) -> int:
        """Register multiple game locations efficiently."""
        return self._registry.register_games_batch(
            games, board_type, num_players, engine_mode
        )

    def mark_games_consolidated(
        self,
        game_ids: list[str],
        canonical_db: str,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> int:
        """Mark games as consolidated into a canonical database."""
        return self._registry.mark_games_consolidated(
            game_ids, canonical_db, board_type, num_players
        )

    def get_unconsolidated_games(
        self,
        board_type: str,
        num_players: int,
        limit: int = 10000,
    ) -> list[GameLocation]:
        """Get games that haven't been consolidated yet."""
        return self._registry.get_unconsolidated_games(board_type, num_players, limit)

    def find_game(self, game_id: str) -> list[GameLocation]:
        """Find all locations where a game exists."""
        return self._registry.find_game(game_id)

    def get_game_replication_count(self, game_id: str) -> int:
        """Get number of nodes where a game is replicated."""
        return self._registry.get_game_replication_count(game_id)

    def get_under_replicated_games(
        self,
        min_copies: int = REPLICATION_TARGET_COUNT,
        board_type: str | None = None,
        num_players: int | None = None,
        limit: int = 1000,
    ) -> list[tuple[str, int]]:
        """Find games that exist on fewer than min_copies nodes."""
        return self._registry.get_under_replicated_games(
            min_copies, board_type, num_players, limit
        )

    def get_game_locations(self) -> dict[str, Any]:
        """Get game locations grouped by game_id."""
        return self._registry.get_game_locations()

    # =========================================================================
    # Model Location Registry (delegated to DataLocationRegistry)
    # =========================================================================

    def register_model(
        self,
        model_path: str,
        node_id: str,
        board_type: str | None = None,
        num_players: int | None = None,
        model_version: str | None = None,
        file_size: int = 0,
    ) -> None:
        """Register a model location in the manifest."""
        return self._registry.register_model(
            model_path, node_id, board_type, num_players, model_version, file_size
        )

    def find_model(self, model_path: str) -> list[ModelLocation]:
        """Find all locations where a model exists."""
        return self._registry.find_model(model_path)

    def find_models_for_config(
        self,
        board_type: str,
        num_players: int,
    ) -> list[ModelLocation]:
        """Find all models for a specific board configuration."""
        return self._registry.find_models_for_config(board_type, num_players)

    def get_model_availability_score(self, model_path: str) -> float:
        """Calculate availability score for a model across the cluster."""
        return self._registry.get_model_availability_score(model_path)

    def count_gpu_nodes(self) -> int:
        """Count total GPU-capable nodes known to the manifest."""
        return self._registry.count_gpu_nodes()

    def sync_model_locations_from_peers(
        self,
        peer_locations: list[dict],
        max_age_seconds: float = 3600.0,
    ) -> int:
        """Sync model locations from peer node manifests."""
        return self._registry.sync_model_locations_from_peers(
            peer_locations, max_age_seconds
        )

    def get_all_model_locations(self) -> list[dict]:
        """Get all model locations as dicts for sync/export."""
        return self._registry.get_all_model_locations()

    # =========================================================================
    # NPZ Location Registry (delegated to DataLocationRegistry)
    # =========================================================================

    def register_npz(
        self,
        npz_path: str,
        node_id: str,
        board_type: str | None = None,
        num_players: int | None = None,
        sample_count: int = 0,
        file_size: int = 0,
    ) -> None:
        """Register an NPZ file location in the manifest."""
        return self._registry.register_npz(
            npz_path, node_id, board_type, num_players, sample_count, file_size
        )

    def find_npz_for_config(
        self,
        board_type: str,
        num_players: int,
    ) -> list[NPZLocation]:
        """Find all NPZ files for a specific board configuration."""
        return self._registry.find_npz_for_config(board_type, num_players)

    # =========================================================================
    # Checkpoint Location Registry (delegated to DataLocationRegistry)
    # =========================================================================

    def register_checkpoint(
        self,
        checkpoint_path: str,
        node_id: str,
        config_key: str | None = None,
        board_type: str | None = None,
        num_players: int | None = None,
        epoch: int = 0,
        step: int = 0,
        loss: float = 0.0,
        file_size: int = 0,
        is_best: bool = False,
    ) -> None:
        """Register a training checkpoint location in the manifest."""
        return self._registry.register_checkpoint(
            checkpoint_path, node_id, config_key, board_type, num_players,
            epoch, step, loss, file_size, is_best
        )

    def find_checkpoint(self, checkpoint_path: str) -> list[CheckpointLocation]:
        """Find all locations where a checkpoint exists."""
        return self._registry.find_checkpoint(checkpoint_path)

    def find_checkpoints_for_config(
        self,
        config_key: str,
        only_best: bool = False,
    ) -> list[CheckpointLocation]:
        """Find all checkpoints for a specific configuration."""
        return self._registry.find_checkpoints_for_config(config_key, only_best)

    def get_latest_checkpoint_for_config(
        self,
        config_key: str,
        prefer_best: bool = True,
    ) -> CheckpointLocation | None:
        """Get the latest checkpoint for a configuration."""
        return self._registry.get_latest_checkpoint_for_config(config_key, prefer_best)

    def mark_checkpoint_as_best(
        self,
        config_key: str,
        checkpoint_path: str,
    ) -> None:
        """Mark a checkpoint as the best for its configuration."""
        return self._registry.mark_checkpoint_as_best(config_key, checkpoint_path)

    # =========================================================================
    # Torrent Registry (December 2025 - BitTorrent P2P Sync)
    # =========================================================================

    def register_torrent(
        self,
        info_hash: str,
        file_path: str,
        torrent_path: str,
        file_size: int = 0,
        piece_size: int = 262144,
        piece_count: int = 0,
        web_seeds: list[str] | None = None,
    ) -> None:
        """Register a torrent in the manifest.

        Creates a new torrent entry or updates an existing one.
        The local node is automatically added as a seeder.

        Args:
            info_hash: SHA1 hash of the torrent info dict (40 hex chars)
            file_path: Relative path to the data file
            torrent_path: Path to the .torrent file
            file_size: Size of the data file in bytes
            piece_size: Size of each piece in bytes
            piece_count: Number of pieces
            web_seeds: Optional HTTP fallback URLs
        """
        now = time.time()
        web_seeds = web_seeds or []

        with self._connection() as conn:
            cursor = conn.cursor()

            # Check if torrent exists
            cursor.execute(
                "SELECT seeders FROM torrent_metadata WHERE info_hash = ?",
                (info_hash,)
            )
            row = cursor.fetchone()

            if row:
                # Update existing - merge seeders
                existing_seeders = json.loads(row[0]) if row[0] else []
                if self.node_id not in existing_seeders:
                    existing_seeders.append(self.node_id)

                cursor.execute("""
                    UPDATE torrent_metadata
                    SET file_path = ?, torrent_path = ?, file_size = ?,
                        piece_size = ?, piece_count = ?, seeders = ?,
                        web_seeds = ?, last_seen = ?
                    WHERE info_hash = ?
                """, (file_path, torrent_path, file_size, piece_size, piece_count,
                      json.dumps(existing_seeders), json.dumps(web_seeds), now, info_hash))
            else:
                # Create new
                cursor.execute("""
                    INSERT INTO torrent_metadata
                    (info_hash, file_path, torrent_path, file_size, piece_size,
                     piece_count, seeders, web_seeds, created_at, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (info_hash, file_path, torrent_path, file_size, piece_size,
                      piece_count, json.dumps([self.node_id]), json.dumps(web_seeds),
                      now, now))

            conn.commit()

        logger.debug(f"Registered torrent: {info_hash[:16]}... for {file_path}")

    def get_torrent(self, info_hash: str) -> TorrentMetadata | None:
        """Get torrent metadata by info_hash.

        Args:
            info_hash: SHA1 hash of the torrent info dict

        Returns:
            TorrentMetadata or None if not found
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT info_hash, file_path, torrent_path, file_size, piece_size,
                       piece_count, seeders, web_seeds, created_at, last_seen
                FROM torrent_metadata
                WHERE info_hash = ?
            """, (info_hash,))

            row = cursor.fetchone()
            if row:
                return TorrentMetadata(
                    info_hash=row[0],
                    file_path=row[1],
                    torrent_path=row[2],
                    file_size=row[3],
                    piece_size=row[4],
                    piece_count=row[5],
                    seeders=json.loads(row[6]) if row[6] else [],
                    web_seeds=json.loads(row[7]) if row[7] else [],
                    created_at=row[8],
                    last_seen=row[9],
                )
            return None

    def get_torrent_for_file(self, file_path: str) -> TorrentMetadata | None:
        """Get torrent metadata by file path.

        Args:
            file_path: Path to the data file (can be relative or contain filename)

        Returns:
            TorrentMetadata or None if not found
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            # Match on exact path or filename
            cursor.execute("""
                SELECT info_hash, file_path, torrent_path, file_size, piece_size,
                       piece_count, seeders, web_seeds, created_at, last_seen
                FROM torrent_metadata
                WHERE file_path = ? OR file_path LIKE ?
                ORDER BY last_seen DESC
                LIMIT 1
            """, (file_path, f"%/{file_path.split('/')[-1]}"))

            row = cursor.fetchone()
            if row:
                return TorrentMetadata(
                    info_hash=row[0],
                    file_path=row[1],
                    torrent_path=row[2],
                    file_size=row[3],
                    piece_size=row[4],
                    piece_count=row[5],
                    seeders=json.loads(row[6]) if row[6] else [],
                    web_seeds=json.loads(row[7]) if row[7] else [],
                    created_at=row[8],
                    last_seen=row[9],
                )
            return None

    def add_seeder(self, info_hash: str, node_id: str) -> bool:
        """Add a seeder to a torrent.

        Args:
            info_hash: Torrent info hash
            node_id: Node ID to add as seeder

        Returns:
            True if seeder was added, False if torrent not found or already seeding
        """
        now = time.time()
        with self._connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT seeders FROM torrent_metadata WHERE info_hash = ?",
                (info_hash,)
            )
            row = cursor.fetchone()

            if not row:
                return False

            seeders = json.loads(row[0]) if row[0] else []
            if node_id in seeders:
                return False  # Already seeding

            seeders.append(node_id)
            cursor.execute("""
                UPDATE torrent_metadata
                SET seeders = ?, last_seen = ?
                WHERE info_hash = ?
            """, (json.dumps(seeders), now, info_hash))
            conn.commit()

            logger.debug(f"Added seeder {node_id} to torrent {info_hash[:16]}...")
            return True

    def remove_seeder(self, info_hash: str, node_id: str) -> bool:
        """Remove a seeder from a torrent.

        Args:
            info_hash: Torrent info hash
            node_id: Node ID to remove as seeder

        Returns:
            True if seeder was removed, False if not found
        """
        now = time.time()
        with self._connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT seeders FROM torrent_metadata WHERE info_hash = ?",
                (info_hash,)
            )
            row = cursor.fetchone()

            if not row:
                return False

            seeders = json.loads(row[0]) if row[0] else []
            if node_id not in seeders:
                return False

            seeders.remove(node_id)
            cursor.execute("""
                UPDATE torrent_metadata
                SET seeders = ?, last_seen = ?
                WHERE info_hash = ?
            """, (json.dumps(seeders), now, info_hash))
            conn.commit()

            logger.debug(f"Removed seeder {node_id} from torrent {info_hash[:16]}...")
            return True

    def get_torrent_seeders(self, info_hash: str) -> list[str]:
        """Get list of seeders for a torrent.

        Args:
            info_hash: Torrent info hash

        Returns:
            List of node IDs currently seeding
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT seeders FROM torrent_metadata WHERE info_hash = ?",
                (info_hash,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return []

    def list_all_torrents(self) -> list[TorrentMetadata]:
        """List all registered torrents.

        Returns:
            List of all TorrentMetadata entries
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT info_hash, file_path, torrent_path, file_size, piece_size,
                       piece_count, seeders, web_seeds, created_at, last_seen
                FROM torrent_metadata
                ORDER BY last_seen DESC
            """)

            torrents = []
            for row in cursor.fetchall():
                torrents.append(TorrentMetadata(
                    info_hash=row[0],
                    file_path=row[1],
                    torrent_path=row[2],
                    file_size=row[3],
                    piece_size=row[4],
                    piece_count=row[5],
                    seeders=json.loads(row[6]) if row[6] else [],
                    web_seeds=json.loads(row[7]) if row[7] else [],
                    created_at=row[8],
                    last_seen=row[9],
                ))
            return torrents

    def get_well_seeded_torrents(self, min_seeders: int = 2) -> list[TorrentMetadata]:
        """Get torrents with at least min_seeders.

        Useful for finding files that are safe to download via BitTorrent.

        Args:
            min_seeders: Minimum number of seeders required

        Returns:
            List of well-seeded TorrentMetadata entries
        """
        all_torrents = self.list_all_torrents()
        return [t for t in all_torrents if len(t.seeders) >= min_seeders]

    # =========================================================================
    # External Storage Registry (January 2026 - S3/OWC Unified Sync)
    # =========================================================================

    def register_s3_location(
        self,
        s3_key: str,
        config_key: str,
        game_count: int,
        file_size: int = 0,
        s3_bucket: str | None = None,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> None:
        """Register a database uploaded to S3.

        Args:
            s3_key: S3 key (path within bucket)
            config_key: Configuration key (e.g., "hex8_2p")
            game_count: Number of games in the database
            file_size: File size in bytes
            s3_bucket: S3 bucket name (default from env)
            board_type: Board type
            num_players: Number of players
        """
        now = time.time()
        bucket = s3_bucket or os.getenv("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO external_storage_locations
                (config_key, source, path, game_count, file_size, board_type,
                 num_players, registered_at, last_verified, s3_bucket, owc_host)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """, (config_key, DataSource.S3.value, s3_key, game_count, file_size,
                  board_type, num_players, now, now, bucket))
            conn.commit()

        logger.debug(f"Registered S3 location: {s3_key} ({game_count} games)")

    def register_owc_location(
        self,
        owc_path: str,
        config_key: str,
        game_count: int,
        file_size: int = 0,
        owc_host: str | None = None,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> None:
        """Register a database synced to OWC drive.

        Args:
            owc_path: Path on OWC drive
            config_key: Configuration key (e.g., "hex8_2p")
            game_count: Number of games in the database
            file_size: File size in bytes
            owc_host: OWC host (default from env)
            board_type: Board type
            num_players: Number of players
        """
        now = time.time()
        host = owc_host or os.getenv("OWC_HOST", "mac-studio")

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO external_storage_locations
                (config_key, source, path, game_count, file_size, board_type,
                 num_players, registered_at, last_verified, s3_bucket, owc_host)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """, (config_key, DataSource.OWC.value, owc_path, game_count, file_size,
                  board_type, num_players, now, now, host))
            conn.commit()

        logger.debug(f"Registered OWC location: {owc_path} ({game_count} games)")

    def find_external_storage_for_config(
        self,
        config_key: str,
    ) -> list[ExternalStorageLocation]:
        """Find all external storage locations for a config.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            List of ExternalStorageLocation entries
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT config_key, source, path, game_count, file_size, board_type,
                       num_players, registered_at, last_verified, s3_bucket, owc_host
                FROM external_storage_locations
                WHERE config_key = ?
                ORDER BY game_count DESC
            """, (config_key,))

            return [
                ExternalStorageLocation(
                    config_key=row[0],
                    source=DataSource(row[1]),
                    path=row[2],
                    game_count=row[3],
                    file_size=row[4],
                    board_type=row[5],
                    num_players=row[6],
                    registered_at=row[7],
                    last_verified=row[8],
                    s3_bucket=row[9],
                    owc_host=row[10],
                )
                for row in cursor.fetchall()
            ]

    def find_across_all_sources(
        self,
        config_key: str,
    ) -> dict[DataSource, list[dict[str, Any]]]:
        """Find data for a config across LOCAL, P2P, S3, OWC.

        This is the primary method for unified data visibility.
        Training can use this to find the best data source.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            Dictionary mapping DataSource to list of locations:
            {
                DataSource.LOCAL: [{db_path, game_count, ...}],
                DataSource.P2P: [{node_id, db_path, game_count, ...}],
                DataSource.S3: [{s3_key, game_count, s3_bucket, ...}],
                DataSource.OWC: [{owc_path, game_count, owc_host, ...}],
            }
        """
        result: dict[DataSource, list[dict[str, Any]]] = {
            DataSource.LOCAL: [],
            DataSource.P2P: [],
            DataSource.S3: [],
            DataSource.OWC: [],
        }

        # Parse config key
        parts = config_key.split("_")
        board_type = parts[0] if parts else None
        num_players = int(parts[1].rstrip("p")) if len(parts) > 1 else None

        # Find LOCAL and P2P databases
        db_locations = self.find_databases_for_config(
            config_key=config_key,
            board_type=board_type,
            num_players=num_players,
        )

        for loc in db_locations:
            node_id = loc.get("node_id", "")
            if node_id == self.node_id:
                result[DataSource.LOCAL].append(loc)
            else:
                result[DataSource.P2P].append(loc)

        # Find S3 and OWC external storage
        external = self.find_external_storage_for_config(config_key)
        for ext in external:
            if ext.source == DataSource.S3:
                result[DataSource.S3].append({
                    "s3_key": ext.path,
                    "game_count": ext.game_count,
                    "file_size": ext.file_size,
                    "s3_bucket": ext.s3_bucket,
                    "registered_at": ext.registered_at,
                })
            elif ext.source == DataSource.OWC:
                result[DataSource.OWC].append({
                    "owc_path": ext.path,
                    "game_count": ext.game_count,
                    "file_size": ext.file_size,
                    "owc_host": ext.owc_host,
                    "registered_at": ext.registered_at,
                })

        return result

    def get_total_games_across_sources(self, config_key: str) -> dict[str, int]:
        """Get total game count per source for a config.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            Dictionary mapping source name to game count
        """
        sources = self.find_across_all_sources(config_key)
        return {
            source.value: sum(loc.get("game_count", 0) for loc in locations)
            for source, locations in sources.items()
        }

    async def refresh_external_storage_counts(
        self,
        configs: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Refresh game counts from external storage (S3 and OWC).

        January 2026: Added as part of unified data discovery infrastructure.
        This method actively queries external storage and updates the manifest,
        rather than relying on event-driven updates.

        Args:
            configs: List of config keys to refresh, or None for all 12 configs

        Returns:
            Dictionary mapping config_key -> {source -> count}
        """
        try:
            from app.utils.unified_game_aggregator import (
                GameSourceConfig,
                UnifiedGameAggregator,
            )

            # Configure to only query external sources (skip local/cluster)
            config = GameSourceConfig(
                include_local=False,
                include_remote=False,
                include_s3=True,
                include_owc=True,
            )
            aggregator = UnifiedGameAggregator(config)

            # Default to all 12 configs
            if configs is None:
                configs = [
                    f"{board}_{players}p"
                    for board in ["hex8", "square8", "square19", "hexagonal"]
                    for players in [2, 3, 4]
                ]

            results: dict[str, dict[str, int]] = {}

            for config_key in configs:
                try:
                    parts = config_key.split("_")
                    board_type = "_".join(parts[:-1])
                    num_players = int(parts[-1].rstrip("p"))

                    counts = await aggregator.get_total_games(board_type, num_players)

                    results[config_key] = {
                        "s3": counts.s3_count,
                        "owc": counts.owc_count,
                    }

                    # Update manifest with discovered external storage
                    if counts.s3_count > 0:
                        # Get S3 details from source_details
                        for detail in counts.source_details:
                            if detail.source_name == "s3" and detail.details:
                                s3_files = detail.details.get("files", 0)
                                if s3_files > 0:
                                    self.register_s3_location(
                                        config_key=config_key,
                                        s3_key=f"games/{config_key}/",
                                        game_count=counts.s3_count,
                                        file_size=detail.details.get("total_bytes", 0),
                                        board_type=board_type,
                                        num_players=num_players,
                                    )

                    if counts.owc_count > 0:
                        # Get OWC details from source_details
                        for detail in counts.source_details:
                            if detail.source_name == "owc" and detail.details:
                                owc_dbs = detail.details.get("databases", {})
                                for owc_path, db_count in owc_dbs.items():
                                    self.register_owc_location(
                                        config_key=config_key,
                                        owc_path=owc_path,
                                        game_count=db_count,
                                        board_type=board_type,
                                        num_players=num_players,
                                    )

                except (ValueError, AttributeError) as e:
                    logger.debug(f"[ClusterManifest] Failed to refresh {config_key}: {e}")
                    results[config_key] = {"s3": 0, "owc": 0, "error": str(e)}

            logger.info(
                f"[ClusterManifest] Refreshed external storage counts for "
                f"{len(results)} configs"
            )
            return results

        except ImportError as e:
            logger.warning(f"[ClusterManifest] UnifiedGameAggregator not available: {e}")
            return {}

    # =========================================================================
    # Database Location Registry (delegated to DataLocationRegistry)
    # =========================================================================

    def register_database(
        self,
        db_path: str,
        node_id: str,
        board_type: str | None = None,
        num_players: int | None = None,
        config_key: str | None = None,
        game_count: int = 0,
        file_size: int = 0,
        engine_mode: str | None = None,
    ) -> None:
        """Register a database file location in the manifest."""
        return self._registry.register_database(
            db_path, node_id, board_type, num_players, config_key,
            game_count, file_size, engine_mode
        )

    def update_database_game_count(
        self,
        db_path: str,
        node_id: str,
        game_count: int,
        file_size: int | None = None,
    ) -> None:
        """Update game count for a registered database."""
        return self._registry.update_database_game_count(
            db_path, node_id, game_count, file_size
        )

    def find_databases_for_config(
        self,
        config_key: str | None = None,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> list[dict[str, Any]]:
        """Find all database files for a specific configuration."""
        return self._registry.find_databases_for_config(
            config_key, board_type, num_players
        )

    def get_all_database_locations(self) -> list[dict[str, Any]]:
        """Get all registered database locations."""
        return self._registry.get_all_database_locations()

    # =========================================================================
    # Node Capacity & Inventory (delegated to NodeCapacityManager)
    # =========================================================================

    def update_node_capacity(
        self,
        node_id: str,
        total_bytes: int,
        used_bytes: int,
        free_bytes: int,
    ) -> None:
        """Update disk capacity information for a node."""
        return self._capacity_manager.update_node_capacity(
            node_id, total_bytes, used_bytes, free_bytes
        )

    def update_local_capacity(self) -> NodeCapacity:
        """Update capacity for the local node and return it."""
        return self._capacity_manager.update_local_capacity()

    def get_node_capacity(self, node_id: str) -> NodeCapacity | None:
        """Get capacity information for a node."""
        return self._capacity_manager.get_node_capacity(node_id)

    def refresh_capacity_data(self, nodes: list[str] | None = None) -> int:
        """Refresh capacity data for all or specified nodes.

        Called when cluster membership changes (CLUSTER_CAPACITY_CHANGED event)
        to ensure routing decisions use current state rather than stale cache.

        Args:
            nodes: List of node IDs to refresh. If None, refreshes all active nodes.

        Returns:
            Number of nodes refreshed.

        Note:
            This method is called by SyncRouter._on_cluster_capacity_changed()
            to invalidate cached capacity after topology changes.
        """
        from app.config.cluster_config import get_active_nodes

        try:
            nodes_to_refresh = nodes or [n.name for n in get_active_nodes()]
        except Exception as e:
            logger.debug(f"[ClusterManifest] Could not get active nodes: {e}")
            nodes_to_refresh = []

        refreshed = 0
        for node_id in nodes_to_refresh:
            try:
                # Clear cached capacity for this node to force re-fetch
                if hasattr(self._capacity_manager, "clear_cache"):
                    self._capacity_manager.clear_cache(node_id)
                elif hasattr(self._capacity_manager, "_node_capacity"):
                    # Direct cache invalidation fallback
                    self._capacity_manager._node_capacity.pop(node_id, None)
                refreshed += 1
            except Exception as e:
                logger.debug(f"Could not refresh capacity for {node_id}: {e}")

        if refreshed > 0:
            logger.debug(f"[ClusterManifest] Refreshed capacity data for {refreshed} nodes")

        return refreshed

    def get_node_inventory(self, node_id: str) -> NodeInventory:
        """Get full inventory for a node."""
        return self._capacity_manager.get_node_inventory(node_id)

    def get_all_db_paths(self, node_id: str | None = None) -> set[str]:
        """Get all tracked database paths."""
        return self._capacity_manager.get_all_db_paths(node_id)

    def get_all_npz_paths(self, node_id: str | None = None) -> set[str]:
        """Get all tracked NPZ file paths."""
        return self._capacity_manager.get_all_npz_paths(node_id)

    def get_all_model_paths(self, node_id: str | None = None) -> set[str]:
        """Get all tracked model file paths."""
        return self._capacity_manager.get_all_model_paths(node_id)

    # =========================================================================
    # Sync Target Selection
    # =========================================================================

    def get_all_nodes(self) -> dict[str, dict[str, Any]]:
        """Get all known nodes with their properties."""
        return self._capacity_manager.get_all_nodes()

    def get_sync_policy(self, node_id: str) -> NodeSyncPolicy:
        """Get sync policy for a node."""
        # Delegate to sync selector and convert to local NodeSyncPolicy type
        selector_policy = self._sync_selector.get_sync_policy(node_id)
        return NodeSyncPolicy(
            node_id=selector_policy.node_id,
            receive_games=selector_policy.receive_games,
            receive_models=selector_policy.receive_models,
            receive_npz=selector_policy.receive_npz,
            max_disk_usage_percent=selector_policy.max_disk_usage_percent,
            excluded=selector_policy.excluded,
            exclusion_reason=selector_policy.exclusion_reason,
        )

    def can_receive_data(self, node_id: str, data_type: DataType) -> bool:
        """Check if a node can receive a specific type of data."""
        # Import sync_target_selector DataType for conversion
        from app.distributed.sync_target_selector import DataType as SelectorDataType

        # Convert local DataType to selector's DataType
        selector_data_type = SelectorDataType(data_type.value)
        return self._sync_selector.can_receive_data(node_id, selector_data_type)

    def get_replication_targets(
        self,
        game_id: str,
        min_copies: int = REPLICATION_TARGET_COUNT,
        exclude_nodes: list[str] | None = None,
    ) -> list[SyncCandidateNode]:
        """Get candidate nodes for replicating a game."""
        # Get targets from selector
        selector_targets = self._sync_selector.get_replication_targets(
            game_id, min_copies, exclude_nodes
        )

        # Convert to local SyncCandidateNode type
        return [
            SyncCandidateNode(
                node_id=t.node_id,
                priority=t.priority,
                reason=t.reason,
                capacity=t.capacity,
            )
            for t in selector_targets
        ]

    # =========================================================================
    # Manifest Propagation (for P2P gossip)
    # =========================================================================

    def export_local_state(self) -> dict[str, Any]:
        """Export local manifest state for P2P propagation.

        Returns:
            Dict with local game/model/npz registrations
        """
        with self._connection() as conn:
            cursor = conn.cursor()

            # Export games registered from local node
            cursor.execute("""
                SELECT game_id, db_path, board_type, num_players, engine_mode, last_seen
                FROM game_locations
                WHERE node_id = ?
            """, (self.node_id,))

            games = []
            for row in cursor.fetchall():
                games.append({
                    "game_id": row[0],
                    "db_path": row[1],
                    "board_type": row[2],
                    "num_players": row[3],
                    "engine_mode": row[4],
                    "last_seen": row[5],
                })

            # Export models
            cursor.execute("""
                SELECT model_path, board_type, num_players, model_version,
                       file_size, last_seen
                FROM model_locations
                WHERE node_id = ?
            """, (self.node_id,))

            models = []
            for row in cursor.fetchall():
                models.append({
                    "model_path": row[0],
                    "board_type": row[1],
                    "num_players": row[2],
                    "model_version": row[3],
                    "file_size": row[4],
                    "last_seen": row[5],
                })

            # Export NPZ files
            cursor.execute("""
                SELECT npz_path, board_type, num_players, sample_count,
                       file_size, last_seen
                FROM npz_locations
                WHERE node_id = ?
            """, (self.node_id,))

            npz_files = []
            for row in cursor.fetchall():
                npz_files.append({
                    "npz_path": row[0],
                    "board_type": row[1],
                    "num_players": row[2],
                    "sample_count": row[3],
                    "file_size": row[4],
                    "last_seen": row[5],
                })

            # Export checkpoints (December 2025)
            cursor.execute("""
                SELECT checkpoint_path, config_key, board_type, num_players,
                       epoch, step, loss, file_size, is_best, last_seen
                FROM checkpoint_locations
                WHERE node_id = ?
            """, (self.node_id,))

            checkpoints = []
            for row in cursor.fetchall():
                checkpoints.append({
                    "checkpoint_path": row[0],
                    "config_key": row[1],
                    "board_type": row[2],
                    "num_players": row[3],
                    "epoch": row[4],
                    "step": row[5],
                    "loss": row[6],
                    "file_size": row[7],
                    "is_best": bool(row[8]),
                    "last_seen": row[9],
                })

            # Export capacity
            capacity = self.get_node_capacity(self.node_id)

            return {
                "node_id": self.node_id,
                "timestamp": time.time(),
                "games": games,
                "models": models,
                "npz_files": npz_files,
                "checkpoints": checkpoints,
                "capacity": {
                    "total_bytes": capacity.total_bytes if capacity else 0,
                    "used_bytes": capacity.used_bytes if capacity else 0,
                    "free_bytes": capacity.free_bytes if capacity else 0,
                    "usage_percent": capacity.usage_percent if capacity else 0,
                } if capacity else None,
            }

    def import_remote_state(self, state: dict[str, Any]) -> int:
        """Import manifest state from a remote node.

        Args:
            state: State dict from export_local_state()

        Returns:
            Number of entries imported
        """
        node_id = state.get("node_id")
        if not node_id:
            return 0

        imported = 0

        # Import games
        for game in state.get("games", []):
            self.register_game(
                game_id=game["game_id"],
                node_id=node_id,
                db_path=game["db_path"],
                board_type=game.get("board_type"),
                num_players=game.get("num_players"),
                engine_mode=game.get("engine_mode"),
            )
            imported += 1

        # Import models
        for model in state.get("models", []):
            self.register_model(
                model_path=model["model_path"],
                node_id=node_id,
                board_type=model.get("board_type"),
                num_players=model.get("num_players"),
                model_version=model.get("model_version"),
                file_size=model.get("file_size", 0),
            )
            imported += 1

        # Import NPZ files
        for npz in state.get("npz_files", []):
            self.register_npz(
                npz_path=npz["npz_path"],
                node_id=node_id,
                board_type=npz.get("board_type"),
                num_players=npz.get("num_players"),
                sample_count=npz.get("sample_count", 0),
                file_size=npz.get("file_size", 0),
            )
            imported += 1

        # Import checkpoints (December 2025)
        for checkpoint in state.get("checkpoints", []):
            self.register_checkpoint(
                checkpoint_path=checkpoint["checkpoint_path"],
                node_id=node_id,
                config_key=checkpoint.get("config_key"),
                board_type=checkpoint.get("board_type"),
                num_players=checkpoint.get("num_players"),
                epoch=checkpoint.get("epoch", 0),
                step=checkpoint.get("step", 0),
                loss=checkpoint.get("loss", 0.0),
                file_size=checkpoint.get("file_size", 0),
                is_best=checkpoint.get("is_best", False),
            )
            imported += 1

        # Import capacity
        capacity = state.get("capacity")
        if capacity:
            self.update_node_capacity(
                node_id=node_id,
                total_bytes=capacity.get("total_bytes", 0),
                used_bytes=capacity.get("used_bytes", 0),
                free_bytes=capacity.get("free_bytes", 0),
            )

        return imported

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_cluster_stats(self) -> dict[str, Any]:
        """Get cluster-wide statistics."""
        with self._connection() as conn:
            cursor = conn.cursor()

            # Total games
            cursor.execute("SELECT COUNT(DISTINCT game_id) FROM game_locations")
            total_games = cursor.fetchone()[0]

            # Total models
            cursor.execute("SELECT COUNT(DISTINCT model_path) FROM model_locations")
            total_models = cursor.fetchone()[0]

            # Total NPZ files
            cursor.execute("SELECT COUNT(DISTINCT npz_path) FROM npz_locations")
            total_npz = cursor.fetchone()[0]

            # Total checkpoints (December 2025)
            cursor.execute("SELECT COUNT(DISTINCT checkpoint_path) FROM checkpoint_locations")
            total_checkpoints = cursor.fetchone()[0]

            # Best checkpoints by config
            cursor.execute("""
                SELECT config_key, COUNT(DISTINCT checkpoint_path)
                FROM checkpoint_locations
                WHERE is_best = 1 AND config_key IS NOT NULL
                GROUP BY config_key
            """)
            best_checkpoints_by_config = {row[0]: row[1] for row in cursor.fetchall()}

            # Games by node
            cursor.execute("""
                SELECT node_id, COUNT(*) FROM game_locations GROUP BY node_id
            """)
            games_by_node = {row[0]: row[1] for row in cursor.fetchall()}

            # Games by config
            cursor.execute("""
                SELECT board_type || '_' || num_players || 'p', COUNT(DISTINCT game_id)
                FROM game_locations
                WHERE board_type IS NOT NULL
                GROUP BY board_type, num_players
            """)
            games_by_config = {row[0]: row[1] for row in cursor.fetchall()}

            # Under-replicated games
            cursor.execute("""
                SELECT COUNT(*) FROM (
                    SELECT game_id FROM game_locations
                    GROUP BY game_id
                    HAVING COUNT(DISTINCT node_id) < ?
                )
            """, (REPLICATION_TARGET_COUNT,))
            under_replicated = cursor.fetchone()[0]

            return {
                "total_games": total_games,
                "total_models": total_models,
                "total_npz_files": total_npz,
                "total_checkpoints": total_checkpoints,
                "best_checkpoints_by_config": best_checkpoints_by_config,
                "games_by_node": games_by_node,
                "games_by_config": games_by_config,
                "under_replicated_games": under_replicated,
                "replication_target": REPLICATION_TARGET_COUNT,
            }

    def get_games_count_by_config(self) -> dict[str, int]:
        """Get game counts grouped by configuration.

        Dec 30, 2025: Added for ClusterAwareDataCatalog integration.

        Returns:
            Dict mapping config_key (e.g., 'hex8_2p') to game count
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT board_type || '_' || num_players || 'p' as config,
                       COUNT(DISTINCT game_id)
                FROM game_locations
                WHERE board_type IS NOT NULL AND num_players IS NOT NULL
                GROUP BY board_type, num_players
            """)
            return {row[0]: row[1] for row in cursor.fetchall() if row[0]}

    def get_games_by_node_and_config(self, config_key: str) -> dict[str, int]:
        """Get game counts per node for a specific configuration.

        Dec 30, 2025: Added for ClusterAwareDataCatalog integration.

        Args:
            config_key: Configuration key (e.g., 'hex8_2p')

        Returns:
            Dict mapping node_id to game count for this config
        """
        # Parse config key
        parts = config_key.replace("_", " ").split()
        if len(parts) < 2:
            return {}
        board_type = parts[0]
        try:
            num_players = int(parts[1].rstrip("p"))
        except ValueError:
            return {}

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT node_id, COUNT(DISTINCT game_id)
                FROM game_locations
                WHERE board_type = ? AND num_players = ?
                GROUP BY node_id
            """, (board_type, num_players))
            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_unconsolidated_count_by_config(self) -> dict[str, int]:
        """Get count of unconsolidated games per configuration.

        Dec 30, 2025: Added for data pipeline tracking.

        Returns:
            Dict mapping config_key to count of games not yet consolidated
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT board_type || '_' || num_players || 'p' as config,
                       COUNT(DISTINCT game_id)
                FROM game_locations
                WHERE board_type IS NOT NULL
                  AND num_players IS NOT NULL
                  AND (is_consolidated = 0 OR is_consolidated IS NULL)
                GROUP BY board_type, num_players
            """)
            return {row[0]: row[1] for row in cursor.fetchall() if row[0]}

    # =========================================================================
    # Disk Cleanup
    # =========================================================================

    def check_disk_cleanup_needed(
        self,
        policy: DiskCleanupPolicy | None = None,
    ) -> bool:
        """Check if disk cleanup is needed based on current usage.

        Args:
            policy: Cleanup policy (uses defaults if None)

        Returns:
            True if cleanup should be triggered
        """
        if policy is None:
            policy = DiskCleanupPolicy()

        capacity = self.update_local_capacity()
        return capacity.usage_percent >= policy.trigger_usage_percent

    def get_cleanup_candidates(
        self,
        policy: DiskCleanupPolicy | None = None,
        data_dir: Path | None = None,
    ) -> list[CleanupCandidate]:
        """Get list of candidates for cleanup, sorted by priority.

        Args:
            policy: Cleanup policy
            data_dir: Data directory to scan

        Returns:
            List of CleanupCandidate sorted by cleanup_priority (highest first)
        """
        if policy is None:
            policy = DiskCleanupPolicy()

        if data_dir is None:
            data_dir = self.db_path.parent.parent / "games"

        candidates: list[CleanupCandidate] = []
        now = time.time()
        min_age_seconds = policy.min_age_days * 86400

        # Scan game databases
        if data_dir.exists():
            for db_path in data_dir.glob("*.db"):
                if db_path.name.startswith("."):
                    continue

                candidate = self._analyze_database_for_cleanup(
                    db_path, now, min_age_seconds, policy
                )
                if candidate:
                    candidates.append(candidate)

        # Scan NPZ files
        npz_dir = self.db_path.parent.parent / "training"
        if npz_dir.exists():
            for npz_path in npz_dir.glob("*.npz"):
                if npz_path.name.startswith("."):
                    continue

                candidate = self._analyze_npz_for_cleanup(
                    npz_path, now, min_age_seconds, policy
                )
                if candidate:
                    candidates.append(candidate)

        # Sort by cleanup priority (highest first)
        candidates.sort(key=lambda c: c.cleanup_priority, reverse=True)

        return candidates

    def _analyze_database_for_cleanup(
        self,
        db_path: Path,
        now: float,
        min_age_seconds: float,
        policy: DiskCleanupPolicy,
    ) -> CleanupCandidate | None:
        """Analyze a database file for cleanup potential.

        Args:
            db_path: Path to database
            now: Current timestamp
            min_age_seconds: Minimum age in seconds
            policy: Cleanup policy

        Returns:
            CleanupCandidate or None if not a candidate
        """
        try:
            stat = db_path.stat()
            age_seconds = now - stat.st_mtime
            age_days = age_seconds / 86400

            # Skip if too new
            if age_seconds < min_age_seconds:
                return None

            # Check if canonical
            is_canonical = "canonical" in db_path.name.lower()
            if is_canonical and policy.preserve_canonical:
                return None

            # Get database metadata
            schema_version = 0
            quality_score = 0.5
            game_count = 0
            board_type = ""
            num_players = 0

            conn = None
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Get schema version
                try:
                    cursor.execute(
                        "SELECT value FROM schema_info WHERE key = 'version'"
                    )
                    row = cursor.fetchone()
                    if row:
                        schema_version = int(row[0])
                except sqlite3.Error:
                    # Try alternative schema location
                    try:
                        cursor.execute("PRAGMA user_version")
                        schema_version = cursor.fetchone()[0]
                    except sqlite3.Error:
                        pass

                # Get game count
                try:
                    cursor.execute("SELECT COUNT(*) FROM games")
                    game_count = cursor.fetchone()[0]
                except sqlite3.Error:
                    pass

                # Get board type and num_players from first game
                try:
                    cursor.execute(
                        "SELECT board_type, num_players FROM games LIMIT 1"
                    )
                    row = cursor.fetchone()
                    if row:
                        board_type = row[0] or ""
                        num_players = row[1] or 0
                except sqlite3.Error:
                    pass

                # Estimate quality from metadata if available
                try:
                    cursor.execute("""
                        SELECT AVG(
                            CASE
                                WHEN json_extract(metadata_json, '$.quality_score') IS NOT NULL
                                THEN json_extract(metadata_json, '$.quality_score')
                                ELSE 0.5
                            END
                        )
                        FROM games
                        WHERE metadata_json IS NOT NULL
                        LIMIT 100
                    """)
                    row = cursor.fetchone()
                    if row and row[0]:
                        quality_score = float(row[0])
                except sqlite3.Error:
                    pass

            except sqlite3.Error as e:
                logger.debug(f"Error reading {db_path}: {e}")
            finally:
                if conn is not None:
                    conn.close()

            # Get replication count
            replication_count = self._get_db_replication_count(db_path)

            # Skip if under-replicated and policy requires replication
            if replication_count < policy.min_replicas_before_delete:
                return None

            return CleanupCandidate(
                path=db_path,
                data_type=DataType.GAME,
                size_bytes=stat.st_size,
                age_days=age_days,
                quality_score=quality_score,
                schema_version=schema_version,
                replication_count=replication_count,
                is_canonical=is_canonical,
                board_type=board_type,
                num_players=num_players,
                game_count=game_count,
            )

        except Exception as e:
            logger.debug(f"Error analyzing {db_path}: {e}")
            return None

    def _analyze_npz_for_cleanup(
        self,
        npz_path: Path,
        now: float,
        min_age_seconds: float,
        policy: DiskCleanupPolicy,
    ) -> CleanupCandidate | None:
        """Analyze an NPZ file for cleanup potential.

        Args:
            npz_path: Path to NPZ file
            now: Current timestamp
            min_age_seconds: Minimum age in seconds
            policy: Cleanup policy

        Returns:
            CleanupCandidate or None if not a candidate
        """
        try:
            stat = npz_path.stat()
            age_seconds = now - stat.st_mtime
            age_days = age_seconds / 86400

            # Skip if too new
            if age_seconds < min_age_seconds:
                return None

            # Parse board type and players from filename
            board_type = ""
            num_players = 0
            name = npz_path.stem

            # Common patterns: "hex8_2p", "square8_4p_v2"
            for bt in ["hex8", "hexagonal", "square8", "square19"]:
                if bt in name:
                    board_type = bt
                    break

            for np in ["2p", "3p", "4p"]:
                if np in name:
                    num_players = int(np[0])
                    break

            # Get replication count
            replication_count = self._get_npz_replication_count(npz_path)

            # Skip if under-replicated
            if replication_count < policy.min_replicas_before_delete:
                return None

            # Check if current (not a candidate) or old
            # Files containing "old", "backup", "archive" have higher cleanup priority
            quality_score = 0.5
            if any(x in name.lower() for x in ["old", "backup", "archive", "_v1"]):
                quality_score = 0.2

            return CleanupCandidate(
                path=npz_path,
                data_type=DataType.NPZ,
                size_bytes=stat.st_size,
                age_days=age_days,
                quality_score=quality_score,
                schema_version=0,
                replication_count=replication_count,
                is_canonical=False,
                board_type=board_type,
                num_players=num_players,
            )

        except Exception as e:
            logger.debug(f"Error analyzing {npz_path}: {e}")
            return None

    def _get_db_replication_count(self, db_path: Path) -> int:
        """Get replication count for a database by checking manifest."""
        db_name = db_path.name

        with self._connection() as conn:
            cursor = conn.cursor()
            # Check how many nodes have games from this DB
            cursor.execute("""
                SELECT COUNT(DISTINCT node_id)
                FROM game_locations
                WHERE db_path LIKE ?
            """, (f"%{db_name}",))
            return cursor.fetchone()[0]

    def _get_npz_replication_count(self, npz_path: Path) -> int:
        """Get replication count for an NPZ file by checking manifest."""
        npz_name = npz_path.name

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(DISTINCT node_id)
                FROM npz_locations
                WHERE npz_path LIKE ?
            """, (f"%{npz_name}",))
            return cursor.fetchone()[0]

    def run_disk_cleanup(
        self,
        policy: DiskCleanupPolicy | None = None,
        data_dir: Path | None = None,
    ) -> DiskCleanupResult:
        """Run disk cleanup to free space.

        Args:
            policy: Cleanup policy (uses defaults if None)
            data_dir: Data directory (uses default if None)

        Returns:
            DiskCleanupResult with cleanup statistics
        """
        if policy is None:
            policy = DiskCleanupPolicy()

        result = DiskCleanupResult(dry_run=policy.dry_run)

        # Check initial capacity
        capacity = self.update_local_capacity()
        result.initial_usage_percent = capacity.usage_percent

        # Check if cleanup needed
        if capacity.usage_percent < policy.trigger_usage_percent:
            result.triggered = False
            result.final_usage_percent = capacity.usage_percent
            return result

        result.triggered = True
        logger.info(
            f"Disk cleanup triggered: {capacity.usage_percent:.1f}% usage "
            f"(threshold: {policy.trigger_usage_percent}%)"
        )

        # Get cleanup candidates
        candidates = self.get_cleanup_candidates(policy, data_dir)

        if not candidates:
            logger.warning("No cleanup candidates found")
            result.final_usage_percent = capacity.usage_percent
            return result

        # Calculate how much space we need to free
        target_free_bytes = int(
            capacity.total_bytes * (100 - policy.target_usage_percent) / 100
        )
        bytes_to_free = target_free_bytes - capacity.free_bytes

        if bytes_to_free <= 0:
            result.final_usage_percent = capacity.usage_percent
            return result

        logger.info(
            f"Need to free {bytes_to_free / 1024 / 1024:.1f} MB "
            f"(target: {policy.target_usage_percent}% usage)"
        )

        # Delete candidates until we reach target or run out
        for candidate in candidates:
            if result.bytes_freed >= bytes_to_free:
                break

            if policy.dry_run:
                logger.info(
                    f"[DRY RUN] Would delete {candidate.path} "
                    f"({candidate.size_bytes / 1024 / 1024:.1f} MB, "
                    f"priority={candidate.cleanup_priority:.1f})"
                )
                result.bytes_freed += candidate.size_bytes
                if candidate.data_type == DataType.GAME:
                    result.databases_deleted += 1
                    result.games_deleted += candidate.game_count
                elif candidate.data_type == DataType.NPZ:
                    result.npz_deleted += 1
                continue

            try:
                # Delete the file
                candidate.path.unlink()

                # Also delete associated WAL/SHM files for databases
                if candidate.data_type == DataType.GAME:
                    wal_path = candidate.path.with_suffix(".db-wal")
                    shm_path = candidate.path.with_suffix(".db-shm")
                    if wal_path.exists():
                        wal_path.unlink()
                    if shm_path.exists():
                        shm_path.unlink()

                result.bytes_freed += candidate.size_bytes
                if candidate.data_type == DataType.GAME:
                    result.databases_deleted += 1
                    result.games_deleted += candidate.game_count
                elif candidate.data_type == DataType.NPZ:
                    result.npz_deleted += 1

                logger.info(
                    f"Deleted {candidate.path} "
                    f"({candidate.size_bytes / 1024 / 1024:.1f} MB, "
                    f"{result.bytes_freed / 1024 / 1024:.1f} MB freed total)"
                )

            except Exception as e:
                error_msg = f"Failed to delete {candidate.path}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Update final capacity
        if not policy.dry_run:
            capacity = self.update_local_capacity()

        result.final_usage_percent = (
            result.initial_usage_percent - (result.bytes_freed / capacity.total_bytes * 100)
            if policy.dry_run else capacity.usage_percent
        )

        logger.info(
            f"Disk cleanup complete: freed {result.bytes_freed / 1024 / 1024:.1f} MB, "
            f"deleted {result.databases_deleted} DBs + {result.npz_deleted} NPZ files, "
            f"usage {result.initial_usage_percent:.1f}% -> {result.final_usage_percent:.1f}%"
        )

        return result

    # =========================================================================
    # Sync Receipt Registry (December 2025 - Push-Based Sync with Safe Cleanup)
    # =========================================================================

    def register_sync_receipt(self, receipt: SyncReceipt) -> None:
        """Record that a file was successfully synced to a destination.

        Called by GPU nodes after pushing data to coordinator and receiving
        confirmation. Used by safe cleanup to ensure files aren't deleted
        until they have N verified copies.

        Args:
            receipt: SyncReceipt with file path, checksum, and destination
        """
        now = time.time()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sync_receipts
                (file_path, file_checksum, synced_to, synced_at, verified,
                 file_size, source_node)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                receipt.file_path,
                receipt.file_checksum,
                receipt.synced_to,
                receipt.synced_at or now,
                1 if receipt.verified else 0,
                receipt.file_size,
                receipt.source_node,
            ))
            conn.commit()

        logger.debug(
            f"Registered sync receipt: {receipt.file_path} -> {receipt.synced_to} "
            f"(verified={receipt.verified})"
        )

    def get_sync_receipts(self, file_path: str) -> list[SyncReceipt]:
        """Get all sync receipts for a file.

        Args:
            file_path: Path to the file (relative or absolute)

        Returns:
            List of SyncReceipt objects for all destinations
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT file_path, file_checksum, synced_to, synced_at,
                       verified, file_size, source_node
                FROM sync_receipts
                WHERE file_path = ?
                ORDER BY synced_at DESC
            """, (file_path,))

            receipts = []
            for row in cursor.fetchall():
                receipts.append(SyncReceipt(
                    file_path=row[0],
                    file_checksum=row[1],
                    synced_to=row[2],
                    synced_at=row[3],
                    verified=bool(row[4]),
                    file_size=row[5] or 0,
                    source_node=row[6] or "",
                ))
            return receipts

    def get_verified_replication_count(self, file_path: str) -> int:
        """Return number of verified copies of this file across cluster.

        Args:
            file_path: Path to the file

        Returns:
            Count of verified sync receipts (copies on other nodes)
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(DISTINCT synced_to)
                FROM sync_receipts
                WHERE file_path = ? AND verified = 1
            """, (file_path,))
            return cursor.fetchone()[0]

    def is_safe_to_delete(
        self,
        file_path: str,
        min_copies: int = 2,
    ) -> bool:
        """Check if file has been synced to at least N verified locations.

        This is the gate for safe cleanup - files are only deleted locally
        after this returns True.

        Args:
            file_path: Path to the file to check
            min_copies: Minimum number of verified copies required

        Returns:
            True if file has min_copies verified replicas, False otherwise
        """
        verified_count = self.get_verified_replication_count(file_path)
        return verified_count >= min_copies

    def get_pending_sync_files(
        self,
        max_age_hours: float = 24.0,
        data_dir: Path | None = None,
    ) -> list[Path]:
        """Get files that haven't been synced anywhere yet.

        Scans the local data directory and returns files that have no
        sync receipts. Used by SyncPushDaemon to find files to push.

        Args:
            max_age_hours: Only return files older than this (to avoid
                          syncing files still being written)
            data_dir: Directory to scan (defaults to games directory)

        Returns:
            List of Path objects for files needing sync
        """
        if data_dir is None:
            data_dir = self.db_path.parent.parent / "games"

        pending: list[Path] = []
        now = time.time()
        min_age_seconds = max_age_hours * 3600

        # Get all known synced files
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT file_path FROM sync_receipts")
            synced_files = {row[0] for row in cursor.fetchall()}

        # Scan for database files
        if data_dir.exists():
            for db_path in data_dir.glob("*.db"):
                if db_path.name.startswith("."):
                    continue

                # Check age
                try:
                    stat = db_path.stat()
                    if (now - stat.st_mtime) < min_age_seconds:
                        continue  # Too new, might still be written to
                except OSError:
                    continue

                # Check if already synced
                rel_path = str(db_path)
                if rel_path not in synced_files:
                    pending.append(db_path)

        return pending

    def mark_receipt_verified(
        self,
        file_path: str,
        synced_to: str,
        checksum: str,
    ) -> bool:
        """Mark a sync receipt as verified after checksum confirmation.

        Called when coordinator confirms file exists with matching checksum.

        Args:
            file_path: Path to the file
            synced_to: Destination node that was synced to
            checksum: Expected checksum (must match existing receipt)

        Returns:
            True if receipt was found and verified, False otherwise
        """
        with self._connection() as conn:
            cursor = conn.cursor()

            # Check if receipt exists with matching checksum
            cursor.execute("""
                SELECT file_checksum FROM sync_receipts
                WHERE file_path = ? AND synced_to = ?
            """, (file_path, synced_to))
            row = cursor.fetchone()

            if not row:
                logger.warning(
                    f"No sync receipt found for {file_path} -> {synced_to}"
                )
                return False

            if row[0] != checksum:
                logger.warning(
                    f"Checksum mismatch for {file_path} -> {synced_to}: "
                    f"expected {row[0][:16]}..., got {checksum[:16]}..."
                )
                return False

            # Mark as verified
            cursor.execute("""
                UPDATE sync_receipts
                SET verified = 1, synced_at = ?
                WHERE file_path = ? AND synced_to = ?
            """, (time.time(), file_path, synced_to))
            conn.commit()

            logger.debug(f"Verified sync receipt: {file_path} -> {synced_to}")
            return True

    def delete_sync_receipts(self, file_path: str) -> int:
        """Delete all sync receipts for a file.

        Called after file is deleted locally to clean up receipts.

        Args:
            file_path: Path to the file

        Returns:
            Number of receipts deleted
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM sync_receipts WHERE file_path = ?",
                (file_path,)
            )
            deleted = cursor.rowcount
            conn.commit()

        if deleted > 0:
            logger.debug(f"Deleted {deleted} sync receipts for {file_path}")
        return deleted

    def get_sync_stats(self) -> dict[str, Any]:
        """Get statistics about sync receipts.

        Returns:
            Dict with sync stats: total_receipts, verified_receipts,
            unique_files, files_with_N_copies, etc.
        """
        with self._connection() as conn:
            cursor = conn.cursor()

            # Total receipts
            cursor.execute("SELECT COUNT(*) FROM sync_receipts")
            total = cursor.fetchone()[0]

            # Verified receipts
            cursor.execute(
                "SELECT COUNT(*) FROM sync_receipts WHERE verified = 1"
            )
            verified = cursor.fetchone()[0]

            # Unique files
            cursor.execute(
                "SELECT COUNT(DISTINCT file_path) FROM sync_receipts"
            )
            unique_files = cursor.fetchone()[0]

            # Files with 2+ verified copies (safe to delete)
            cursor.execute("""
                SELECT COUNT(*) FROM (
                    SELECT file_path
                    FROM sync_receipts
                    WHERE verified = 1
                    GROUP BY file_path
                    HAVING COUNT(DISTINCT synced_to) >= 2
                )
            """)
            safe_to_delete = cursor.fetchone()[0]

            # Files with only unverified receipts
            cursor.execute("""
                SELECT COUNT(DISTINCT file_path) FROM sync_receipts
                WHERE file_path NOT IN (
                    SELECT file_path FROM sync_receipts WHERE verified = 1
                )
            """)
            unverified_only = cursor.fetchone()[0]

            return {
                "total_receipts": total,
                "verified_receipts": verified,
                "unique_files": unique_files,
                "safe_to_delete_count": safe_to_delete,
                "unverified_only_count": unverified_only,
                "verification_rate": verified / total if total > 0 else 0.0,
            }


# =============================================================================
# Module-level utilities
# =============================================================================


_cluster_manifest: ClusterManifest | None = None
_cluster_manifest_lock = threading.Lock()


def get_cluster_manifest(auto_subscribe: bool = True) -> ClusterManifest:
    """Get the singleton ClusterManifest instance.

    Thread-safe with double-checked locking pattern.

    Args:
        auto_subscribe: If True (default), automatically subscribe to
            real-time events on first creation. Set to False for testing
            or when event router is not available.

    Returns:
        The singleton ClusterManifest instance.
    """
    global _cluster_manifest

    # Fast path: already initialized
    if _cluster_manifest is not None:
        return _cluster_manifest

    # Slow path: need to create (with lock)
    with _cluster_manifest_lock:
        # Double-check after acquiring lock
        if _cluster_manifest is None:
            _cluster_manifest = ClusterManifest()
            if auto_subscribe:
                try:
                    # Subscribe to real-time events for immediate manifest updates
                    # This reduces lag from 5 minutes (gossip) to near-real-time
                    _cluster_manifest.subscribe_to_events()
                except Exception as e:
                    logger.warning(f"[ClusterManifest] Event subscription deferred: {e}")

    return _cluster_manifest


def reset_cluster_manifest() -> None:
    """Reset the singleton (for testing)."""
    global _cluster_manifest
    if _cluster_manifest is not None:
        _cluster_manifest.close()
    _cluster_manifest = None
