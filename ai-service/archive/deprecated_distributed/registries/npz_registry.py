"""NPZ Registry - tracks NPZ training file locations across the cluster.

Extracted from ClusterManifest to improve maintainability.

December 2025 - ClusterManifest decomposition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.distributed.registries.base import BaseRegistry

logger = logging.getLogger(__name__)


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


class NPZRegistry(BaseRegistry):
    """Registry for NPZ training file locations across the cluster.

    Tracks which NPZ files exist on which nodes.
    """

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

    def find_npz(self, npz_path: str) -> list[NPZLocation]:
        """Find all locations where an NPZ file exists.

        Args:
            npz_path: NPZ file path

        Returns:
            List of NPZLocation objects
        """
        rows = self._fetch_all("""
            SELECT npz_path, node_id, board_type, num_players,
                   sample_count, file_size, registered_at, last_seen
            FROM npz_locations
            WHERE npz_path = ?
        """, (npz_path,))

        return [
            NPZLocation(
                npz_path=row[0],
                node_id=row[1],
                board_type=row[2],
                num_players=row[3],
                sample_count=row[4],
                file_size=row[5],
                registered_at=row[6],
                last_seen=row[7],
            )
            for row in rows
        ]

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
        rows = self._fetch_all("""
            SELECT npz_path, node_id, board_type, num_players,
                   sample_count, file_size, registered_at, last_seen
            FROM npz_locations
            WHERE board_type = ? AND num_players = ?
            ORDER BY sample_count DESC, last_seen DESC
        """, (board_type, num_players))

        return [
            NPZLocation(
                npz_path=row[0],
                node_id=row[1],
                board_type=row[2],
                num_players=row[3],
                sample_count=row[4],
                file_size=row[5],
                registered_at=row[6],
                last_seen=row[7],
            )
            for row in rows
        ]

    def get_npz_by_node(self, node_id: str) -> list[NPZLocation]:
        """Get all NPZ files on a specific node.

        Args:
            node_id: Node identifier

        Returns:
            List of NPZLocation objects
        """
        rows = self._fetch_all("""
            SELECT npz_path, node_id, board_type, num_players,
                   sample_count, file_size, registered_at, last_seen
            FROM npz_locations
            WHERE node_id = ?
            ORDER BY last_seen DESC
        """, (node_id,))

        return [
            NPZLocation(
                npz_path=row[0],
                node_id=row[1],
                board_type=row[2],
                num_players=row[3],
                sample_count=row[4],
                file_size=row[5],
                registered_at=row[6],
                last_seen=row[7],
            )
            for row in rows
        ]

    def get_total_unique_npz(self) -> int:
        """Get total unique NPZ file count.

        Returns:
            Number of unique NPZ files
        """
        row = self._fetch_one("SELECT COUNT(DISTINCT npz_path) FROM npz_locations")
        return row[0] if row else 0

    def get_total_samples_for_config(
        self,
        board_type: str,
        num_players: int,
    ) -> int:
        """Get total sample count for a configuration.

        Args:
            board_type: Board configuration
            num_players: Number of players

        Returns:
            Total samples
        """
        row = self._fetch_one("""
            SELECT COALESCE(SUM(sample_count), 0)
            FROM npz_locations
            WHERE board_type = ? AND num_players = ?
        """, (board_type, num_players))
        return row[0] if row else 0

    def get_total_size_by_node(self, node_id: str) -> int:
        """Get total NPZ size for a node.

        Args:
            node_id: Node identifier

        Returns:
            Total bytes
        """
        row = self._fetch_one(
            "SELECT COALESCE(SUM(file_size), 0) FROM npz_locations WHERE node_id = ?",
            (node_id,)
        )
        return row[0] if row else 0

    def get_distinct_npz_paths(self, node_id: str | None = None) -> set[str]:
        """Get all distinct NPZ paths.

        Args:
            node_id: Optional filter by node

        Returns:
            Set of NPZ paths
        """
        if node_id:
            rows = self._fetch_all(
                "SELECT DISTINCT npz_path FROM npz_locations WHERE node_id = ?",
                (node_id,)
            )
        else:
            rows = self._fetch_all("SELECT DISTINCT npz_path FROM npz_locations")
        return {row[0] for row in rows}

    def remove_npz(self, npz_path: str, node_id: str | None = None) -> int:
        """Remove NPZ location(s).

        Args:
            npz_path: NPZ path
            node_id: Optional node to remove from (all nodes if None)

        Returns:
            Number of rows removed
        """
        with self._connection() as conn:
            if node_id:
                cursor = conn.execute(
                    "DELETE FROM npz_locations WHERE npz_path = ? AND node_id = ?",
                    (npz_path, node_id)
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM npz_locations WHERE npz_path = ?",
                    (npz_path,)
                )
            conn.commit()
            return cursor.rowcount

    def get_replication_count(self, npz_path: str) -> int:
        """Get replication count for an NPZ file.

        Args:
            npz_path: NPZ file path

        Returns:
            Number of nodes with the file
        """
        row = self._fetch_one(
            "SELECT COUNT(DISTINCT node_id) FROM npz_locations WHERE npz_path LIKE ?",
            (f"%{npz_path}",)
        )
        return row[0] if row else 0

    def export_for_node(self, node_id: str) -> list[dict[str, Any]]:
        """Export NPZ registrations for a node (for P2P gossip).

        Args:
            node_id: Node identifier

        Returns:
            List of NPZ dicts
        """
        rows = self._fetch_all("""
            SELECT npz_path, board_type, num_players, sample_count,
                   file_size, last_seen
            FROM npz_locations
            WHERE node_id = ?
        """, (node_id,))

        return [
            {
                "npz_path": row[0],
                "board_type": row[1],
                "num_players": row[2],
                "sample_count": row[3],
                "file_size": row[4],
                "last_seen": row[5],
            }
            for row in rows
        ]

    def import_from_remote(
        self,
        node_id: str,
        npz_files: list[dict[str, Any]],
    ) -> int:
        """Import NPZ registrations from remote node.

        Args:
            node_id: Source node identifier
            npz_files: List of NPZ dicts

        Returns:
            Number of files imported
        """
        imported = 0
        for npz in npz_files:
            self.register_npz(
                npz_path=npz["npz_path"],
                node_id=node_id,
                board_type=npz.get("board_type"),
                num_players=npz.get("num_players"),
                sample_count=npz.get("sample_count", 0),
                file_size=npz.get("file_size", 0),
            )
            imported += 1
        return imported
