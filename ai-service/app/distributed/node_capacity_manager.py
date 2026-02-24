"""Node capacity and inventory management for the cluster.

Extracted from cluster_manifest.py (December 2025) to improve modularity
and testability. Tracks disk capacity and data inventory across cluster nodes.

Usage:
    from app.distributed.node_capacity_manager import NodeCapacityManager

    manager = NodeCapacityManager(
        db_path=Path("cluster_manifest.db"),
        connection_factory=manifest._connection,
        node_id="my-node",
        hosts_config={"node-1": {"role": "training"}},
    )

    # Update local capacity
    capacity = manager.update_local_capacity()

    # Get inventory for a node
    inventory = manager.get_node_inventory("node-1")
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants
try:
    from app.config.thresholds import DISK_SYNC_TARGET_PERCENT
    MAX_DISK_USAGE_PERCENT = DISK_SYNC_TARGET_PERCENT  # Don't sync to nodes above this usage
except ImportError:
    MAX_DISK_USAGE_PERCENT = 70  # Don't sync to nodes above this usage


# ============================================================================
# Data Classes
# ============================================================================


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
        """Percentage of disk that is free."""
        return 100.0 - self.usage_percent


@dataclass
class NodeInventory:
    """Full inventory of data on a node."""

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


# ============================================================================
# NodeCapacityManager
# ============================================================================


class NodeCapacityManager:
    """Manages disk capacity and inventory tracking across cluster nodes.

    This class handles:
    - Node disk capacity tracking (total, used, free bytes)
    - Node data inventory (games, models, NPZ files)
    - Path enumeration for all data types
    - Node discovery from capacity database and config

    Thread-safe via the connection factory pattern.
    """

    def __init__(
        self,
        db_path: Path,
        connection_factory: Callable[[], Generator[sqlite3.Connection, None, None]],
        node_id: str,
        hosts_config: dict[str, dict[str, Any]] | None = None,
    ):
        """Initialize the capacity manager.

        Args:
            db_path: Path to the manifest database
            connection_factory: Context manager factory for database connections
            node_id: This node's identifier
            hosts_config: Optional hosts configuration dict
        """
        self.db_path = db_path
        self._connection = connection_factory
        self.node_id = node_id
        self._hosts_config = hosts_config or {}

    # =========================================================================
    # Capacity Management
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
            cursor.execute(
                """
                INSERT OR REPLACE INTO node_capacity
                (node_id, total_bytes, used_bytes, free_bytes,
                 usage_percent, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (node_id, total_bytes, used_bytes, free_bytes, usage_percent, now),
            )
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
            cursor.execute(
                """
                SELECT node_id, total_bytes, used_bytes, free_bytes,
                       usage_percent, last_updated
                FROM node_capacity
                WHERE node_id = ?
            """,
                (node_id,),
            )

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

    # =========================================================================
    # Inventory Management
    # =========================================================================

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
                (node_id,),
            )
            game_count = cursor.fetchone()[0]

            # Model counts
            cursor.execute(
                "SELECT COUNT(*), COALESCE(SUM(file_size), 0) FROM model_locations WHERE node_id = ?",
                (node_id,),
            )
            row = cursor.fetchone()
            model_count, model_size = row[0], row[1] or 0

            # NPZ counts
            cursor.execute(
                "SELECT COUNT(*), COALESCE(SUM(file_size), 0) FROM npz_locations WHERE node_id = ?",
                (node_id,),
            )
            row = cursor.fetchone()
            npz_count, npz_size = row[0], row[1] or 0

            # Get lists
            cursor.execute(
                "SELECT DISTINCT db_path FROM game_locations WHERE node_id = ?",
                (node_id,),
            )
            databases = [row[0] for row in cursor.fetchall()]

            cursor.execute(
                "SELECT model_path FROM model_locations WHERE node_id = ?",
                (node_id,),
            )
            models = [row[0] for row in cursor.fetchall()]

            cursor.execute(
                "SELECT npz_path FROM npz_locations WHERE node_id = ?",
                (node_id,),
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
    # Path Enumeration
    # =========================================================================

    def get_all_db_paths(self, node_id: str | None = None) -> set[str]:
        """Get all tracked database paths.

        Args:
            node_id: If specified, only return paths for this node.
                    If None, return all paths across the cluster.

        Returns:
            Set of database paths (e.g., "/data/games/selfplay_hex8_2p.db")
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            if node_id:
                cursor.execute(
                    "SELECT DISTINCT db_path FROM game_locations WHERE node_id = ?",
                    (node_id,),
                )
            else:
                cursor.execute("SELECT DISTINCT db_path FROM game_locations")
            return {row[0] for row in cursor.fetchall()}

    def get_all_npz_paths(self, node_id: str | None = None) -> set[str]:
        """Get all tracked NPZ file paths.

        Args:
            node_id: If specified, only return paths for this node.
                    If None, return all paths across the cluster.

        Returns:
            Set of NPZ file paths (e.g., "data/training/hex8_2p.npz")
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            if node_id:
                cursor.execute(
                    "SELECT DISTINCT npz_path FROM npz_locations WHERE node_id = ?",
                    (node_id,),
                )
            else:
                cursor.execute("SELECT DISTINCT npz_path FROM npz_locations")
            return {row[0] for row in cursor.fetchall()}

    def get_all_model_paths(self, node_id: str | None = None) -> set[str]:
        """Get all tracked model file paths.

        Args:
            node_id: If specified, only return paths for this node.
                    If None, return all paths across the cluster.

        Returns:
            Set of model file paths (e.g., "models/canonical_hex8_2p.pth")
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            if node_id:
                cursor.execute(
                    "SELECT DISTINCT model_path FROM model_locations WHERE node_id = ?",
                    (node_id,),
                )
            else:
                cursor.execute("SELECT DISTINCT model_path FROM model_locations")
            return {row[0] for row in cursor.fetchall()}

    # =========================================================================
    # Node Discovery
    # =========================================================================

    def get_all_nodes(self) -> dict[str, dict[str, Any]]:
        """Get all known nodes with their properties.

        Returns:
            Dict mapping node_id to node properties dict with keys:
            - disk_usage_percent: Current disk usage (0-100)
            - is_storage_node: True if node has large storage capacity
            - is_ephemeral: True if node is ephemeral (Vast.ai, spot instances)
            - free_bytes: Available disk space in bytes
            - role: Node role from config (selfplay, training, coordinator)

        December 2025: Added for AutoSyncDaemon push-to-neighbors support.
        """
        result: dict[str, dict[str, Any]] = {}

        # Get nodes from capacity database
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT node_id, total_bytes, used_bytes, free_bytes
                FROM node_capacity
                WHERE last_updated > ?
            """,
                (time.time() - 3600,),
            )
            capacity_rows = cursor.fetchall()

        for row in capacity_rows:
            node_id, total_bytes, used_bytes, free_bytes = row
            usage_percent = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0

            # Get node config from hosts
            host_config = self._hosts_config.get(node_id, {})
            role = host_config.get("role", "selfplay")

            # Determine if storage node (>500GB free or >1TB total)
            is_storage_node = (
                free_bytes > 500 * 1024 * 1024 * 1024  # >500GB free
                or total_bytes > 1024 * 1024 * 1024 * 1024  # >1TB total
            )

            # Determine if ephemeral (Vast.ai, spot instances)
            is_ephemeral = (
                node_id.startswith("vast-")
                or host_config.get("ephemeral", False)
                or host_config.get("provider", "").lower() in ("vast", "spot")
            )

            result[node_id] = {
                "disk_usage_percent": usage_percent,
                "is_storage_node": is_storage_node,
                "is_ephemeral": is_ephemeral,
                "free_bytes": free_bytes,
                "role": role,
            }

        # Also add nodes from config that might not have capacity info yet
        for node_id, host_config in self._hosts_config.items():
            if node_id not in result:
                result[node_id] = {
                    "disk_usage_percent": 0,  # Unknown
                    "is_storage_node": False,
                    "is_ephemeral": (
                        node_id.startswith("vast-")
                        or host_config.get("ephemeral", False)
                        or host_config.get("provider", "").lower() in ("vast", "spot")
                    ),
                    "free_bytes": 0,
                    "role": host_config.get("role", "selfplay"),
                }

        return result

    def set_hosts_config(self, hosts_config: dict[str, dict[str, Any]]) -> None:
        """Update the hosts configuration.

        Args:
            hosts_config: New hosts configuration dict
        """
        self._hosts_config = hosts_config
