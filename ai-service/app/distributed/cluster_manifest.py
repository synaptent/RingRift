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
import shutil
import socket
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generator

import yaml

logger = logging.getLogger(__name__)

__all__ = [
    # Enums
    "DataType",
    "NodeRole",
    # Data classes
    "GameLocation",
    "ModelLocation",
    "NPZLocation",
    "CheckpointLocation",
    "NodeCapacity",
    "NodeInventory",
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


@dataclass
class GameLocation:
    """Location of a game in the cluster."""
    game_id: str
    node_id: str
    db_path: str
    board_type: str | None = None
    num_players: int | None = None
    engine_mode: str | None = None
    registered_at: float = 0.0
    last_seen: float = 0.0


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
class NodeCapacity:
    """Disk capacity information for a node."""
    node_id: str
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    usage_percent: float = 0.0
    last_updated: float = 0.0

    @property
    def can_receive_sync(self) -> bool:
        """Check if node can receive more data."""
        return self.usage_percent < MAX_DISK_USAGE_PERCENT

    @property
    def free_percent(self) -> float:
        """Get percentage of free space."""
        return 100.0 - self.usage_percent


@dataclass
class NodeInventory:
    """Inventory of data on a node."""
    node_id: str
    game_count: int = 0
    model_count: int = 0
    npz_count: int = 0
    total_games_size: int = 0
    total_models_size: int = 0
    total_npz_size: int = 0
    capacity: NodeCapacity | None = None
    databases: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    npz_files: list[str] = field(default_factory=list)


@dataclass
class SyncTarget:
    """A potential target for syncing data."""
    node_id: str
    priority: int = 0  # Higher = sync first
    reason: str = ""
    capacity: NodeCapacity | None = None


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
    min_replicas_before_delete: int = 2  # Only delete if replicated elsewhere
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

        # Load host configuration
        self._hosts_config: dict[str, Any] = {}
        self._exclusion_rules: dict[str, NodeSyncPolicy] = {}
        self._max_disk_usage = MAX_DISK_USAGE_PERCENT
        self._priority_hosts: set[str] = set()
        self._load_config(config_path)

        # Initialize database
        self._init_db()

        logger.info(f"ClusterManifest initialized: node={self.node_id}, db={db_path}")

    def _load_config(self, config_path: Path | None = None) -> None:
        """Load host configuration and exclusion rules."""
        if config_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            config_path = base_dir / "config" / "distributed_hosts.yaml"

        if not config_path.exists():
            logger.warning(f"No config found at {config_path}")
            return

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            self._hosts_config = config.get("hosts", {})

            # Build exclusion rules from config
            self._build_exclusion_rules(config)

        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def _build_exclusion_rules(self, config: dict[str, Any]) -> None:
        """Build node exclusion rules from configuration."""
        hosts = config.get("hosts", {})

        # Get sync routing configuration
        sync_routing = config.get("sync_routing", {})

        # Read max disk usage from config
        self._max_disk_usage = sync_routing.get(
            "max_disk_usage_percent", MAX_DISK_USAGE_PERCENT
        )

        # Auto-sync exclusion from auto_sync section
        auto_sync = config.get("auto_sync", {})
        exclude_hosts = set(auto_sync.get("exclude_hosts", []))

        # Process sync_routing.excluded_hosts with detailed policies
        excluded_host_policies: dict[str, dict] = {}
        for entry in sync_routing.get("excluded_hosts", []):
            if isinstance(entry, dict):
                name = entry.get("name", "")
                if name:
                    exclude_hosts.add(name)
                    excluded_host_policies[name] = entry
            else:
                exclude_hosts.add(entry)

        # Process allowed_external_storage overrides
        external_storage_overrides: dict[str, dict] = {}
        for entry in sync_routing.get("allowed_external_storage", []):
            if isinstance(entry, dict):
                host = entry.get("host", "")
                if host:
                    external_storage_overrides[host] = entry

        # Priority hosts for training data
        self._priority_hosts = set(sync_routing.get("priority_hosts", []))

        for host_name, host_config in hosts.items():
            role = host_config.get("role", "selfplay")

            # Default policy
            policy = NodeSyncPolicy(
                node_id=host_name,
                max_disk_usage_percent=self._max_disk_usage,
            )

            # Check if coordinator (typically dev machines)
            if role == "coordinator":
                policy.receive_games = False
                policy.receive_npz = False
                policy.receive_models = True  # Still receive models
                policy.exclusion_reason = "coordinator node"

            # Check if explicitly excluded with detailed policy
            if host_name in excluded_host_policies:
                entry = excluded_host_policies[host_name]
                policy.receive_games = entry.get("receive_games", False)
                policy.receive_npz = entry.get("receive_npz", False)
                policy.receive_models = entry.get("receive_models", True)
                policy.exclusion_reason = entry.get("reason", "explicitly excluded")
            elif host_name in exclude_hosts:
                policy.receive_games = False
                policy.receive_npz = False
                policy.receive_models = True
                policy.exclusion_reason = "explicitly excluded"

            # Check selfplay_enabled/training_enabled flags
            if not host_config.get("selfplay_enabled", True) and \
               not host_config.get("training_enabled", True):
                policy.receive_games = False
                policy.receive_npz = False
                policy.exclusion_reason = "selfplay and training disabled"

            # Mac machines - special handling
            if self._is_local_mac(host_name, host_config):
                # Exclude local Macs by default
                policy.receive_games = False
                policy.receive_npz = False
                policy.receive_models = True
                policy.exclusion_reason = "local Mac machine"

                # Check for external storage override (e.g., OWC drive)
                if host_name in external_storage_overrides:
                    override = external_storage_overrides[host_name]
                    # Only apply override if the external path exists
                    ext_path = override.get("path", "")
                    if ext_path and Path(ext_path).exists():
                        policy.receive_games = override.get("receive_games", True)
                        policy.receive_npz = override.get("receive_npz", True)
                        policy.receive_models = override.get("receive_models", True)
                        policy.exclusion_reason = ""
                        logger.info(
                            f"External storage override for {host_name}: {ext_path}"
                        )
                elif self._has_owc_external_drive(host_name, host_config):
                    # Fallback to legacy detection
                    policy.receive_games = True
                    policy.receive_npz = True
                    policy.exclusion_reason = ""

            self._exclusion_rules[host_name] = policy

    def _is_local_mac(self, host_name: str, host_config: dict) -> bool:
        """Check if this is a local Mac machine."""
        # Check hostname patterns
        if "mac" in host_name.lower() or "mbp" in host_name.lower():
            return True

        # Check GPU field for MPS
        gpu = host_config.get("gpu", "")
        if "MPS" in gpu or "M1" in gpu or "M2" in gpu or "M3" in gpu:
            return True

        return False

    def _has_owc_external_drive(self, host_name: str, host_config: dict) -> bool:
        """Check if this Mac has an OWC external drive for sync."""
        # Mac Studio with external storage
        if "mac-studio" in host_name.lower():
            # Check for configured external drive path
            ringrift_path = host_config.get("ringrift_path", "")
            if "/Volumes/OWC" in ringrift_path or "/Volumes/External" in ringrift_path:
                return True

            # Check for sync_storage_path configuration
            sync_path = host_config.get("sync_storage_path", "")
            if "/Volumes/OWC" in sync_path:
                return True

        return False

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
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript(f"""
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

            -- Metadata table
            CREATE TABLE IF NOT EXISTS manifest_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            -- Indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_game_locations_node
                ON game_locations(node_id);
            CREATE INDEX IF NOT EXISTS idx_game_locations_board
                ON game_locations(board_type, num_players);
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

            -- Initialize metadata
            INSERT OR IGNORE INTO manifest_metadata (key, value, updated_at)
            VALUES
                ('schema_version', '{SCHEMA_VERSION}', {time.time()}),
                ('created_at', '{time.time()}', {time.time()});
        """)
        conn.commit()
        conn.close()

        logger.debug(f"Initialized cluster manifest at {self.db_path}")

    # =========================================================================
    # Game Location Registry
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
        """Register a game location in the manifest.

        Args:
            game_id: Unique game identifier
            node_id: Node where the game exists
            db_path: Path to database containing the game
            board_type: Board configuration (e.g., "hex8", "square8")
            num_players: Number of players
            engine_mode: Engine mode used (e.g., "gumbel-mcts")
        """
        now = time.time()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO game_locations
                (game_id, node_id, db_path, board_type, num_players,
                 engine_mode, registered_at, last_seen)
                VALUES (?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT registered_at FROM game_locations
                              WHERE game_id = ? AND node_id = ?), ?),
                    ?)
            """, (game_id, node_id, db_path, board_type, num_players,
                  engine_mode, game_id, node_id, now, now))
            conn.commit()

    def register_games_batch(
        self,
        games: list[tuple[str, str, str]],
        board_type: str | None = None,
        num_players: int | None = None,
        engine_mode: str | None = None,
    ) -> int:
        """Register multiple game locations efficiently.

        Args:
            games: List of (game_id, node_id, db_path) tuples
            board_type: Board configuration
            num_players: Number of players
            engine_mode: Engine mode

        Returns:
            Number of games registered
        """
        if not games:
            return 0

        now = time.time()
        registered = 0

        with self._connection() as conn:
            cursor = conn.cursor()
            for game_id, node_id, db_path in games:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO game_locations
                        (game_id, node_id, db_path, board_type, num_players,
                         engine_mode, registered_at, last_seen)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (game_id, node_id, db_path, board_type, num_players,
                          engine_mode, now, now))
                    registered += 1
                except sqlite3.Error as e:
                    logger.warning(f"Failed to register game {game_id}: {e}")
            conn.commit()

        return registered

    def find_game(self, game_id: str) -> list[GameLocation]:
        """Find all locations where a game exists.

        Args:
            game_id: Game identifier

        Returns:
            List of GameLocation objects
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT game_id, node_id, db_path, board_type, num_players,
                       engine_mode, registered_at, last_seen
                FROM game_locations
                WHERE game_id = ?
            """, (game_id,))

            locations = []
            for row in cursor.fetchall():
                locations.append(GameLocation(
                    game_id=row[0],
                    node_id=row[1],
                    db_path=row[2],
                    board_type=row[3],
                    num_players=row[4],
                    engine_mode=row[5],
                    registered_at=row[6],
                    last_seen=row[7],
                ))

            return locations

    def get_game_replication_count(self, game_id: str) -> int:
        """Get number of nodes where a game is replicated.

        Args:
            game_id: Game identifier

        Returns:
            Number of nodes with the game
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(DISTINCT node_id) FROM game_locations WHERE game_id = ?",
                (game_id,)
            )
            return cursor.fetchone()[0]

    def get_under_replicated_games(
        self,
        min_copies: int = REPLICATION_TARGET_COUNT,
        board_type: str | None = None,
        num_players: int | None = None,
        limit: int = 1000,
    ) -> list[tuple[str, int]]:
        """Find games that exist on fewer than min_copies nodes.

        Args:
            min_copies: Minimum required copies
            board_type: Optional filter
            num_players: Optional filter
            limit: Maximum results

        Returns:
            List of (game_id, current_copies) tuples
        """
        with self._connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT game_id, COUNT(DISTINCT node_id) as copies
                FROM game_locations
            """
            params: list[Any] = []

            where_clauses = []
            if board_type:
                where_clauses.append("board_type = ?")
                params.append(board_type)
            if num_players:
                where_clauses.append("num_players = ?")
                params.append(num_players)

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            query += " GROUP BY game_id HAVING copies < ? LIMIT ?"
            params.extend([min_copies, limit])

            cursor.execute(query, params)
            return [(row[0], row[1]) for row in cursor.fetchall()]

    # =========================================================================
    # Model Location Registry
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
        """Register a model location in the manifest.

        Args:
            model_path: Relative path to model file
            node_id: Node where the model exists
            board_type: Board configuration
            num_players: Number of players
            model_version: Model version string
            file_size: File size in bytes
        """
        now = time.time()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO model_locations
                (model_path, node_id, board_type, num_players, model_version,
                 file_size, registered_at, last_seen)
                VALUES (?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT registered_at FROM model_locations
                              WHERE model_path = ? AND node_id = ?), ?),
                    ?)
            """, (model_path, node_id, board_type, num_players, model_version,
                  file_size, model_path, node_id, now, now))
            conn.commit()

    def find_model(self, model_path: str) -> list[ModelLocation]:
        """Find all locations where a model exists.

        Args:
            model_path: Model file path

        Returns:
            List of ModelLocation objects
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_path, node_id, board_type, num_players,
                       model_version, file_size, registered_at, last_seen
                FROM model_locations
                WHERE model_path = ?
            """, (model_path,))

            locations = []
            for row in cursor.fetchall():
                locations.append(ModelLocation(
                    model_path=row[0],
                    node_id=row[1],
                    board_type=row[2],
                    num_players=row[3],
                    model_version=row[4],
                    file_size=row[5],
                    registered_at=row[6],
                    last_seen=row[7],
                ))

            return locations

    def find_models_for_config(
        self,
        board_type: str,
        num_players: int,
    ) -> list[ModelLocation]:
        """Find all models for a specific board configuration.

        Args:
            board_type: Board configuration
            num_players: Number of players

        Returns:
            List of ModelLocation objects
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_path, node_id, board_type, num_players,
                       model_version, file_size, registered_at, last_seen
                FROM model_locations
                WHERE board_type = ? AND num_players = ?
                ORDER BY last_seen DESC
            """, (board_type, num_players))

            locations = []
            for row in cursor.fetchall():
                locations.append(ModelLocation(
                    model_path=row[0],
                    node_id=row[1],
                    board_type=row[2],
                    num_players=row[3],
                    model_version=row[4],
                    file_size=row[5],
                    registered_at=row[6],
                    last_seen=row[7],
                ))

            return locations

    # =========================================================================
    # NPZ Location Registry
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
        """Register an NPZ file location in the manifest.

        Args:
            npz_path: Relative path to NPZ file
            node_id: Node where the file exists
            board_type: Board configuration
            num_players: Number of players
            sample_count: Number of training samples
            file_size: File size in bytes
        """
        now = time.time()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO npz_locations
                (npz_path, node_id, board_type, num_players, sample_count,
                 file_size, registered_at, last_seen)
                VALUES (?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT registered_at FROM npz_locations
                              WHERE npz_path = ? AND node_id = ?), ?),
                    ?)
            """, (npz_path, node_id, board_type, num_players, sample_count,
                  file_size, npz_path, node_id, now, now))
            conn.commit()

    def find_npz_for_config(
        self,
        board_type: str,
        num_players: int,
    ) -> list[NPZLocation]:
        """Find all NPZ files for a specific board configuration.

        Args:
            board_type: Board configuration
            num_players: Number of players

        Returns:
            List of NPZLocation objects
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT npz_path, node_id, board_type, num_players,
                       sample_count, file_size, registered_at, last_seen
                FROM npz_locations
                WHERE board_type = ? AND num_players = ?
                ORDER BY sample_count DESC, last_seen DESC
            """, (board_type, num_players))

            locations = []
            for row in cursor.fetchall():
                locations.append(NPZLocation(
                    npz_path=row[0],
                    node_id=row[1],
                    board_type=row[2],
                    num_players=row[3],
                    sample_count=row[4],
                    file_size=row[5],
                    registered_at=row[6],
                    last_seen=row[7],
                ))

            return locations

    # =========================================================================
    # Checkpoint Location Registry (December 2025)
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
        """Register a training checkpoint location in the manifest.

        This enables distributed training resume and failover by tracking
        where checkpoints exist across the cluster.

        Args:
            checkpoint_path: Relative path to checkpoint file
            node_id: Node where the checkpoint exists
            config_key: Configuration key (e.g., "hex8_2p")
            board_type: Board configuration
            num_players: Number of players
            epoch: Training epoch number
            step: Training step number
            loss: Training loss at this checkpoint
            file_size: File size in bytes
            is_best: Whether this is the best checkpoint for this config
        """
        now = time.time()

        # Derive config_key if not provided
        if config_key is None and board_type and num_players:
            config_key = f"{board_type}_{num_players}p"

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO checkpoint_locations
                (checkpoint_path, node_id, config_key, board_type, num_players,
                 epoch, step, loss, file_size, is_best, registered_at, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT registered_at FROM checkpoint_locations
                              WHERE checkpoint_path = ? AND node_id = ?), ?),
                    ?)
            """, (checkpoint_path, node_id, config_key, board_type, num_players,
                  epoch, step, loss, file_size, 1 if is_best else 0,
                  checkpoint_path, node_id, now, now))
            conn.commit()

        logger.debug(f"Registered checkpoint: {checkpoint_path} on {node_id} (epoch={epoch})")

    def find_checkpoint(self, checkpoint_path: str) -> list[CheckpointLocation]:
        """Find all locations where a checkpoint exists.

        Args:
            checkpoint_path: Checkpoint file path

        Returns:
            List of CheckpointLocation objects
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT checkpoint_path, node_id, config_key, board_type, num_players,
                       epoch, step, loss, file_size, is_best, registered_at, last_seen
                FROM checkpoint_locations
                WHERE checkpoint_path = ?
            """, (checkpoint_path,))

            locations = []
            for row in cursor.fetchall():
                locations.append(CheckpointLocation(
                    checkpoint_path=row[0],
                    node_id=row[1],
                    config_key=row[2],
                    board_type=row[3],
                    num_players=row[4],
                    epoch=row[5],
                    step=row[6],
                    loss=row[7],
                    file_size=row[8],
                    is_best=bool(row[9]),
                    registered_at=row[10],
                    last_seen=row[11],
                ))

            return locations

    def find_checkpoints_for_config(
        self,
        config_key: str,
        only_best: bool = False,
    ) -> list[CheckpointLocation]:
        """Find all checkpoints for a specific configuration.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            only_best: If True, only return best checkpoints

        Returns:
            List of CheckpointLocation objects, sorted by epoch descending
        """
        with self._connection() as conn:
            cursor = conn.cursor()

            if only_best:
                cursor.execute("""
                    SELECT checkpoint_path, node_id, config_key, board_type, num_players,
                           epoch, step, loss, file_size, is_best, registered_at, last_seen
                    FROM checkpoint_locations
                    WHERE config_key = ? AND is_best = 1
                    ORDER BY epoch DESC, last_seen DESC
                """, (config_key,))
            else:
                cursor.execute("""
                    SELECT checkpoint_path, node_id, config_key, board_type, num_players,
                           epoch, step, loss, file_size, is_best, registered_at, last_seen
                    FROM checkpoint_locations
                    WHERE config_key = ?
                    ORDER BY epoch DESC, last_seen DESC
                """, (config_key,))

            locations = []
            for row in cursor.fetchall():
                locations.append(CheckpointLocation(
                    checkpoint_path=row[0],
                    node_id=row[1],
                    config_key=row[2],
                    board_type=row[3],
                    num_players=row[4],
                    epoch=row[5],
                    step=row[6],
                    loss=row[7],
                    file_size=row[8],
                    is_best=bool(row[9]),
                    registered_at=row[10],
                    last_seen=row[11],
                ))

            return locations

    def get_latest_checkpoint_for_config(
        self,
        config_key: str,
        prefer_best: bool = True,
    ) -> CheckpointLocation | None:
        """Get the latest checkpoint for a configuration.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            prefer_best: If True, prefer best checkpoint over latest epoch

        Returns:
            CheckpointLocation or None if not found
        """
        with self._connection() as conn:
            cursor = conn.cursor()

            if prefer_best:
                # First try to find best checkpoint
                cursor.execute("""
                    SELECT checkpoint_path, node_id, config_key, board_type, num_players,
                           epoch, step, loss, file_size, is_best, registered_at, last_seen
                    FROM checkpoint_locations
                    WHERE config_key = ? AND is_best = 1
                    ORDER BY epoch DESC, last_seen DESC
                    LIMIT 1
                """, (config_key,))
                row = cursor.fetchone()
                if row:
                    return CheckpointLocation(
                        checkpoint_path=row[0],
                        node_id=row[1],
                        config_key=row[2],
                        board_type=row[3],
                        num_players=row[4],
                        epoch=row[5],
                        step=row[6],
                        loss=row[7],
                        file_size=row[8],
                        is_best=bool(row[9]),
                        registered_at=row[10],
                        last_seen=row[11],
                    )

            # Fall back to latest by epoch
            cursor.execute("""
                SELECT checkpoint_path, node_id, config_key, board_type, num_players,
                       epoch, step, loss, file_size, is_best, registered_at, last_seen
                FROM checkpoint_locations
                WHERE config_key = ?
                ORDER BY epoch DESC, last_seen DESC
                LIMIT 1
            """, (config_key,))
            row = cursor.fetchone()
            if row:
                return CheckpointLocation(
                    checkpoint_path=row[0],
                    node_id=row[1],
                    config_key=row[2],
                    board_type=row[3],
                    num_players=row[4],
                    epoch=row[5],
                    step=row[6],
                    loss=row[7],
                    file_size=row[8],
                    is_best=bool(row[9]),
                    registered_at=row[10],
                    last_seen=row[11],
                )

            return None

    def mark_checkpoint_as_best(
        self,
        config_key: str,
        checkpoint_path: str,
    ) -> None:
        """Mark a checkpoint as the best for its configuration.

        This also clears is_best from any other checkpoints for this config.

        Args:
            config_key: Configuration key
            checkpoint_path: Path to the checkpoint to mark as best
        """
        now = time.time()
        with self._connection() as conn:
            cursor = conn.cursor()

            # Clear existing best for this config
            cursor.execute("""
                UPDATE checkpoint_locations
                SET is_best = 0, last_seen = ?
                WHERE config_key = ? AND is_best = 1
            """, (now, config_key))

            # Set new best
            cursor.execute("""
                UPDATE checkpoint_locations
                SET is_best = 1, last_seen = ?
                WHERE config_key = ? AND checkpoint_path = ?
            """, (now, config_key, checkpoint_path))

            conn.commit()

        logger.info(f"Marked {checkpoint_path} as best for {config_key}")

    # =========================================================================
    # Database Location Registry (Phase 4A.3 - December 2025)
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
        """Register a database file location in the manifest.

        This enables immediate visibility of new databases without waiting
        for the 5-minute orphan scan. Called by selfplay when creating DBs.

        Args:
            db_path: Path to database file (absolute or relative)
            node_id: Node where the database exists
            board_type: Board configuration (e.g., "hex8", "square8")
            num_players: Number of players
            config_key: Configuration key (e.g., "hex8_2p")
            game_count: Initial game count (usually 0)
            file_size: File size in bytes
            engine_mode: Engine mode used (e.g., "gumbel-mcts")
        """
        now = time.time()

        # Derive config_key if not provided
        if config_key is None and board_type and num_players:
            config_key = f"{board_type}_{num_players}p"

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO database_locations
                (db_path, node_id, board_type, num_players, config_key,
                 game_count, file_size, engine_mode, registered_at, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?,
                    ?,
                    COALESCE((SELECT registered_at FROM database_locations
                              WHERE db_path = ? AND node_id = ?), ?),
                    ?)
            """, (db_path, node_id, board_type, num_players, config_key,
                  game_count, file_size, engine_mode, db_path, node_id, now, now))
            conn.commit()

        logger.debug(f"Registered database: {db_path} on {node_id}")

    def update_database_game_count(
        self,
        db_path: str,
        node_id: str,
        game_count: int,
        file_size: int | None = None,
    ) -> None:
        """Update game count for a registered database.

        Args:
            db_path: Path to database file
            node_id: Node where the database exists
            game_count: Current game count
            file_size: Optional updated file size
        """
        now = time.time()
        with self._connection() as conn:
            cursor = conn.cursor()
            if file_size is not None:
                cursor.execute("""
                    UPDATE database_locations
                    SET game_count = ?, file_size = ?, last_seen = ?
                    WHERE db_path = ? AND node_id = ?
                """, (game_count, file_size, now, db_path, node_id))
            else:
                cursor.execute("""
                    UPDATE database_locations
                    SET game_count = ?, last_seen = ?
                    WHERE db_path = ? AND node_id = ?
                """, (game_count, now, db_path, node_id))
            conn.commit()

    def find_databases_for_config(
        self,
        config_key: str | None = None,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> list[dict[str, Any]]:
        """Find all database files for a specific configuration.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            board_type: Board configuration (alternative to config_key)
            num_players: Number of players (alternative to config_key)

        Returns:
            List of database location dictionaries with:
            - db_path, node_id, board_type, num_players, config_key
            - game_count, file_size, engine_mode
            - registered_at, last_seen
        """
        with self._connection() as conn:
            cursor = conn.cursor()

            if config_key:
                cursor.execute("""
                    SELECT db_path, node_id, board_type, num_players, config_key,
                           game_count, file_size, engine_mode, registered_at, last_seen
                    FROM database_locations
                    WHERE config_key = ?
                    ORDER BY game_count DESC, last_seen DESC
                """, (config_key,))
            elif board_type and num_players:
                cursor.execute("""
                    SELECT db_path, node_id, board_type, num_players, config_key,
                           game_count, file_size, engine_mode, registered_at, last_seen
                    FROM database_locations
                    WHERE board_type = ? AND num_players = ?
                    ORDER BY game_count DESC, last_seen DESC
                """, (board_type, num_players))
            else:
                # Return all databases
                cursor.execute("""
                    SELECT db_path, node_id, board_type, num_players, config_key,
                           game_count, file_size, engine_mode, registered_at, last_seen
                    FROM database_locations
                    ORDER BY game_count DESC, last_seen DESC
                """)

            results = []
            for row in cursor.fetchall():
                results.append({
                    "db_path": row[0],
                    "node_id": row[1],
                    "board_type": row[2],
                    "num_players": row[3],
                    "config_key": row[4],
                    "game_count": row[5],
                    "file_size": row[6],
                    "engine_mode": row[7],
                    "registered_at": row[8],
                    "last_seen": row[9],
                })

            return results

    def get_all_database_locations(self) -> list[dict[str, Any]]:
        """Get all registered database locations.

        Returns:
            List of all database location dictionaries
        """
        return self.find_databases_for_config()

    # =========================================================================
    # Node Capacity & Inventory
    # =========================================================================

    def update_node_capacity(
        self,
        node_id: str,
        total_bytes: int,
        used_bytes: int,
        free_bytes: int,
    ) -> None:
        """Update disk capacity information for a node.

        Args:
            node_id: Node identifier
            total_bytes: Total disk space
            used_bytes: Used disk space
            free_bytes: Free disk space
        """
        usage_percent = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0
        now = time.time()

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO node_capacity
                (node_id, total_bytes, used_bytes, free_bytes,
                 usage_percent, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (node_id, total_bytes, used_bytes, free_bytes,
                  usage_percent, now))
            conn.commit()

    def update_local_capacity(self) -> NodeCapacity:
        """Update capacity for the local node and return it."""
        try:
            stat = shutil.disk_usage(self.db_path.parent)
            self.update_node_capacity(
                self.node_id,
                stat.total,
                stat.used,
                stat.free,
            )
            return NodeCapacity(
                node_id=self.node_id,
                total_bytes=stat.total,
                used_bytes=stat.used,
                free_bytes=stat.free,
                usage_percent=(stat.used / stat.total * 100) if stat.total > 0 else 0,
                last_updated=time.time(),
            )
        except Exception as e:
            logger.error(f"Failed to update local capacity: {e}")
            return NodeCapacity(node_id=self.node_id)

    def get_node_capacity(self, node_id: str) -> NodeCapacity | None:
        """Get capacity information for a node.

        Args:
            node_id: Node identifier

        Returns:
            NodeCapacity or None if not found
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT node_id, total_bytes, used_bytes, free_bytes,
                       usage_percent, last_updated
                FROM node_capacity
                WHERE node_id = ?
            """, (node_id,))

            row = cursor.fetchone()
            if row:
                return NodeCapacity(
                    node_id=row[0],
                    total_bytes=row[1],
                    used_bytes=row[2],
                    free_bytes=row[3],
                    usage_percent=row[4],
                    last_updated=row[5],
                )
            return None

    def get_node_inventory(self, node_id: str) -> NodeInventory:
        """Get full inventory for a node.

        Args:
            node_id: Node identifier

        Returns:
            NodeInventory with counts and lists
        """
        with self._connection() as conn:
            cursor = conn.cursor()

            # Game counts
            cursor.execute(
                "SELECT COUNT(*), COALESCE(SUM(1), 0) FROM game_locations WHERE node_id = ?",
                (node_id,)
            )
            game_count = cursor.fetchone()[0]

            # Model counts
            cursor.execute(
                "SELECT COUNT(*), COALESCE(SUM(file_size), 0) FROM model_locations WHERE node_id = ?",
                (node_id,)
            )
            row = cursor.fetchone()
            model_count, model_size = row[0], row[1] or 0

            # NPZ counts
            cursor.execute(
                "SELECT COUNT(*), COALESCE(SUM(file_size), 0) FROM npz_locations WHERE node_id = ?",
                (node_id,)
            )
            row = cursor.fetchone()
            npz_count, npz_size = row[0], row[1] or 0

            # Get lists
            cursor.execute(
                "SELECT DISTINCT db_path FROM game_locations WHERE node_id = ?",
                (node_id,)
            )
            databases = [row[0] for row in cursor.fetchall()]

            cursor.execute(
                "SELECT model_path FROM model_locations WHERE node_id = ?",
                (node_id,)
            )
            models = [row[0] for row in cursor.fetchall()]

            cursor.execute(
                "SELECT npz_path FROM npz_locations WHERE node_id = ?",
                (node_id,)
            )
            npz_files = [row[0] for row in cursor.fetchall()]

            # Get capacity
            capacity = self.get_node_capacity(node_id)

            return NodeInventory(
                node_id=node_id,
                game_count=game_count,
                model_count=model_count,
                npz_count=npz_count,
                total_models_size=model_size,
                total_npz_size=npz_size,
                capacity=capacity,
                databases=databases,
                models=models,
                npz_files=npz_files,
            )

    # =========================================================================
    # Sync Target Selection
    # =========================================================================

    def get_sync_policy(self, node_id: str) -> NodeSyncPolicy:
        """Get sync policy for a node.

        Args:
            node_id: Node identifier

        Returns:
            NodeSyncPolicy for the node
        """
        return self._exclusion_rules.get(
            node_id,
            NodeSyncPolicy(node_id=node_id)
        )

    def can_receive_data(self, node_id: str, data_type: DataType) -> bool:
        """Check if a node can receive a specific type of data.

        Args:
            node_id: Node identifier
            data_type: Type of data to sync

        Returns:
            True if node can receive this data type
        """
        policy = self.get_sync_policy(node_id)

        if policy.excluded:
            return False

        # Check capacity
        capacity = self.get_node_capacity(node_id)
        if capacity and capacity.usage_percent >= policy.max_disk_usage_percent:
            return False

        # Check data type permissions
        if data_type == DataType.GAME:
            return policy.receive_games
        elif data_type == DataType.MODEL:
            return policy.receive_models
        elif data_type == DataType.NPZ:
            return policy.receive_npz
        elif data_type == DataType.CHECKPOINT:
            # Checkpoints follow model policy - training nodes need them
            return policy.receive_models

        return True

    def get_replication_targets(
        self,
        game_id: str,
        min_copies: int = REPLICATION_TARGET_COUNT,
        exclude_nodes: list[str] | None = None,
    ) -> list[SyncTarget]:
        """Get candidate nodes for replicating a game.

        Args:
            game_id: Game to replicate
            min_copies: Desired minimum copies
            exclude_nodes: Nodes to exclude

        Returns:
            List of SyncTarget sorted by priority
        """
        exclude_nodes = set(exclude_nodes or [])

        # Get current locations
        current_locations = self.find_game(game_id)
        current_nodes = {loc.node_id for loc in current_locations}
        exclude_nodes.update(current_nodes)

        # Need more copies?
        copies_needed = min_copies - len(current_nodes)
        if copies_needed <= 0:
            return []

        targets: list[SyncTarget] = []

        # Find candidate nodes
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT node_id FROM node_capacity")
            all_nodes = [row[0] for row in cursor.fetchall()]

        for node_id in all_nodes:
            if node_id in exclude_nodes:
                continue

            # Check if can receive games
            if not self.can_receive_data(node_id, DataType.GAME):
                continue

            # Get capacity and compute priority
            capacity = self.get_node_capacity(node_id)
            priority = self._compute_sync_priority(node_id, capacity)

            reason = self._get_sync_reason(node_id, capacity)

            targets.append(SyncTarget(
                node_id=node_id,
                priority=priority,
                reason=reason,
                capacity=capacity,
            ))

        # Sort by priority (highest first)
        targets.sort(key=lambda t: t.priority, reverse=True)

        return targets[:copies_needed]

    def _compute_sync_priority(
        self,
        node_id: str,
        capacity: NodeCapacity | None,
    ) -> int:
        """Compute sync priority for a node.

        Higher priority = sync first.

        Factors:
        - Training nodes get higher priority
        - Nodes with more free space get higher priority
        - Ephemeral nodes get lower priority (data may be lost)
        """
        priority = 50  # Base priority

        # Adjust based on role
        host_config = self._hosts_config.get(node_id, {})
        role = host_config.get("role", "selfplay")

        if "training" in role:
            priority += 20
        elif role == "coordinator":
            priority -= 30

        # Adjust based on capacity
        if capacity:
            # More free space = higher priority
            if capacity.free_percent > 50:
                priority += 10
            elif capacity.free_percent < 20:
                priority -= 20

        # Ephemeral hosts get lower priority (we should sync FROM them, not TO)
        if host_config.get("storage_type") == "ephemeral":
            priority -= 10

        return priority

    def _get_sync_reason(
        self,
        node_id: str,
        capacity: NodeCapacity | None,
    ) -> str:
        """Get human-readable reason for sync target selection."""
        host_config = self._hosts_config.get(node_id, {})
        role = host_config.get("role", "selfplay")

        reasons = []

        if "training" in role:
            reasons.append("training node")

        if capacity:
            reasons.append(f"{capacity.free_percent:.1f}% free")

        return ", ".join(reasons) if reasons else "available"

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

                conn.close()

            except sqlite3.Error as e:
                logger.debug(f"Error reading {db_path}: {e}")

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


# =============================================================================
# Module-level utilities
# =============================================================================


_cluster_manifest: ClusterManifest | None = None


def get_cluster_manifest() -> ClusterManifest:
    """Get the singleton ClusterManifest instance."""
    global _cluster_manifest
    if _cluster_manifest is None:
        _cluster_manifest = ClusterManifest()
    return _cluster_manifest


def reset_cluster_manifest() -> None:
    """Reset the singleton (for testing)."""
    global _cluster_manifest
    if _cluster_manifest is not None:
        _cluster_manifest.close()
    _cluster_manifest = None
