"""Node Inventory Manager - tracks node capacity and inventory.

Extracted from ClusterManifest to improve maintainability.

December 2025 - ClusterManifest decomposition.
"""

from __future__ import annotations

import logging
import shutil
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.distributed.registries.base import BaseRegistry

logger = logging.getLogger(__name__)

# Constants
MAX_DISK_USAGE_PERCENT = 70


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


class NodeInventoryManager(BaseRegistry):
    """Manages node capacity and inventory information.

    Tracks disk usage and data inventory for all cluster nodes.
    """

    def __init__(self, data_dir: Path | None = None, **kwargs):
        """Initialize the node inventory manager.

        Args:
            data_dir: Data directory for local disk usage
            **kwargs: Arguments passed to BaseRegistry
        """
        super().__init__(**kwargs)
        self._data_dir = data_dir
        self._node_id = socket.gethostname()

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
        """Update capacity for the local node and return it.

        Returns:
            NodeCapacity for local node
        """
        try:
            check_dir = self._data_dir
            if check_dir is None:
                check_dir = Path.cwd()

            stat = shutil.disk_usage(check_dir)
            self.update_node_capacity(
                self._node_id,
                stat.total,
                stat.used,
                stat.free,
            )
            return NodeCapacity(
                node_id=self._node_id,
                total_bytes=stat.total,
                used_bytes=stat.used,
                free_bytes=stat.free,
                usage_percent=(stat.used / stat.total * 100) if stat.total > 0 else 0,
                last_updated=time.time(),
            )
        except Exception as e:
            logger.error(f"Failed to update local capacity: {e}")
            return NodeCapacity(node_id=self._node_id)

    def get_node_capacity(self, node_id: str) -> NodeCapacity | None:
        """Get capacity information for a node.

        Args:
            node_id: Node identifier

        Returns:
            NodeCapacity or None if not found
        """
        row = self._fetch_one("""
            SELECT node_id, total_bytes, used_bytes, free_bytes,
                   usage_percent, last_updated
            FROM node_capacity
            WHERE node_id = ?
        """, (node_id,))

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

    def get_node_inventory(
        self,
        node_id: str,
        game_registry=None,
        model_registry=None,
        npz_registry=None,
    ) -> NodeInventory:
        """Get full inventory for a node.

        Args:
            node_id: Node identifier
            game_registry: Optional GameLocationRegistry for game data
            model_registry: Optional ModelRegistry for model data
            npz_registry: Optional NPZRegistry for NPZ data

        Returns:
            NodeInventory with counts and lists
        """
        inventory = NodeInventory(node_id=node_id)

        # Use provided registries or query directly
        if game_registry is not None:
            games = game_registry.get_games_by_node(node_id)
            inventory.game_count = len(games)
            inventory.databases = list({g.db_path for g in games})
        else:
            # Direct query
            row = self._fetch_one(
                "SELECT COUNT(*) FROM game_locations WHERE node_id = ?",
                (node_id,)
            )
            inventory.game_count = row[0] if row else 0

            rows = self._fetch_all(
                "SELECT DISTINCT db_path FROM game_locations WHERE node_id = ?",
                (node_id,)
            )
            inventory.databases = [r[0] for r in rows]

        if model_registry is not None:
            models = model_registry.get_models_by_node(node_id)
            inventory.model_count = len(models)
            inventory.total_models_size = sum(m.file_size for m in models)
            inventory.models = [m.model_path for m in models]
        else:
            row = self._fetch_one(
                "SELECT COUNT(*), COALESCE(SUM(file_size), 0) FROM model_locations WHERE node_id = ?",
                (node_id,)
            )
            if row:
                inventory.model_count = row[0]
                inventory.total_models_size = row[1] or 0

            rows = self._fetch_all(
                "SELECT model_path FROM model_locations WHERE node_id = ?",
                (node_id,)
            )
            inventory.models = [r[0] for r in rows]

        if npz_registry is not None:
            npz_files = npz_registry.get_npz_by_node(node_id)
            inventory.npz_count = len(npz_files)
            inventory.total_npz_size = sum(n.file_size for n in npz_files)
            inventory.npz_files = [n.npz_path for n in npz_files]
        else:
            row = self._fetch_one(
                "SELECT COUNT(*), COALESCE(SUM(file_size), 0) FROM npz_locations WHERE node_id = ?",
                (node_id,)
            )
            if row:
                inventory.npz_count = row[0]
                inventory.total_npz_size = row[1] or 0

            rows = self._fetch_all(
                "SELECT npz_path FROM npz_locations WHERE node_id = ?",
                (node_id,)
            )
            inventory.npz_files = [r[0] for r in rows]

        # Get capacity
        inventory.capacity = self.get_node_capacity(node_id)

        return inventory

    def get_all_node_ids(self) -> list[str]:
        """Get all known node IDs.

        Returns:
            List of node identifiers
        """
        rows = self._fetch_all("SELECT DISTINCT node_id FROM node_capacity")
        return [row[0] for row in rows]

    def get_nodes_with_free_space(
        self,
        min_free_percent: float = 30.0,
    ) -> list[NodeCapacity]:
        """Get nodes with sufficient free space.

        Args:
            min_free_percent: Minimum free space percentage

        Returns:
            List of NodeCapacity objects
        """
        max_usage = 100.0 - min_free_percent
        rows = self._fetch_all("""
            SELECT node_id, total_bytes, used_bytes, free_bytes,
                   usage_percent, last_updated
            FROM node_capacity
            WHERE usage_percent < ?
            ORDER BY usage_percent ASC
        """, (max_usage,))

        return [
            NodeCapacity(
                node_id=row[0],
                total_bytes=row[1],
                used_bytes=row[2],
                free_bytes=row[3],
                usage_percent=row[4],
                last_updated=row[5],
            )
            for row in rows
        ]

    def get_cluster_capacity_stats(self) -> dict[str, Any]:
        """Get aggregate capacity statistics.

        Returns:
            Dict with cluster-wide capacity info
        """
        rows = self._fetch_all("""
            SELECT total_bytes, used_bytes, free_bytes, usage_percent
            FROM node_capacity
        """)

        if not rows:
            return {
                "total_nodes": 0,
                "total_bytes": 0,
                "used_bytes": 0,
                "free_bytes": 0,
                "avg_usage_percent": 0.0,
                "nodes_over_threshold": 0,
            }

        total_bytes = sum(r[0] for r in rows)
        used_bytes = sum(r[1] for r in rows)
        free_bytes = sum(r[2] for r in rows)
        avg_usage = sum(r[3] for r in rows) / len(rows)
        over_threshold = sum(1 for r in rows if r[3] >= MAX_DISK_USAGE_PERCENT)

        return {
            "total_nodes": len(rows),
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
            "free_bytes": free_bytes,
            "avg_usage_percent": avg_usage,
            "nodes_over_threshold": over_threshold,
        }

    def export_capacity(self, node_id: str) -> dict[str, Any] | None:
        """Export capacity for P2P gossip.

        Args:
            node_id: Node identifier

        Returns:
            Capacity dict or None
        """
        capacity = self.get_node_capacity(node_id)
        if capacity:
            return {
                "total_bytes": capacity.total_bytes,
                "used_bytes": capacity.used_bytes,
                "free_bytes": capacity.free_bytes,
                "usage_percent": capacity.usage_percent,
            }
        return None

    def import_capacity(
        self,
        node_id: str,
        capacity: dict[str, Any],
    ) -> None:
        """Import capacity from P2P gossip.

        Args:
            node_id: Node identifier
            capacity: Capacity dict
        """
        self.update_node_capacity(
            node_id=node_id,
            total_bytes=capacity.get("total_bytes", 0),
            used_bytes=capacity.get("used_bytes", 0),
            free_bytes=capacity.get("free_bytes", 0),
        )
