"""Model Registry - tracks model locations across the cluster.

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


class ModelRegistry(BaseRegistry):
    """Registry for model locations across the cluster.

    Tracks which models exist on which nodes.
    """

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
        rows = self._fetch_all("""
            SELECT model_path, node_id, board_type, num_players,
                   model_version, file_size, registered_at, last_seen
            FROM model_locations
            WHERE model_path = ?
        """, (model_path,))

        return [
            ModelLocation(
                model_path=row[0],
                node_id=row[1],
                board_type=row[2],
                num_players=row[3],
                model_version=row[4],
                file_size=row[5],
                registered_at=row[6],
                last_seen=row[7],
            )
            for row in rows
        ]

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
        rows = self._fetch_all("""
            SELECT model_path, node_id, board_type, num_players,
                   model_version, file_size, registered_at, last_seen
            FROM model_locations
            WHERE board_type = ? AND num_players = ?
            ORDER BY last_seen DESC
        """, (board_type, num_players))

        return [
            ModelLocation(
                model_path=row[0],
                node_id=row[1],
                board_type=row[2],
                num_players=row[3],
                model_version=row[4],
                file_size=row[5],
                registered_at=row[6],
                last_seen=row[7],
            )
            for row in rows
        ]

    def get_models_by_node(self, node_id: str) -> list[ModelLocation]:
        """Get all models on a specific node.

        Args:
            node_id: Node identifier

        Returns:
            List of ModelLocation objects
        """
        rows = self._fetch_all("""
            SELECT model_path, node_id, board_type, num_players,
                   model_version, file_size, registered_at, last_seen
            FROM model_locations
            WHERE node_id = ?
            ORDER BY last_seen DESC
        """, (node_id,))

        return [
            ModelLocation(
                model_path=row[0],
                node_id=row[1],
                board_type=row[2],
                num_players=row[3],
                model_version=row[4],
                file_size=row[5],
                registered_at=row[6],
                last_seen=row[7],
            )
            for row in rows
        ]

    def get_total_unique_models(self) -> int:
        """Get total unique model count.

        Returns:
            Number of unique models
        """
        row = self._fetch_one("SELECT COUNT(DISTINCT model_path) FROM model_locations")
        return row[0] if row else 0

    def get_total_size_by_node(self, node_id: str) -> int:
        """Get total model size for a node.

        Args:
            node_id: Node identifier

        Returns:
            Total bytes
        """
        row = self._fetch_one(
            "SELECT COALESCE(SUM(file_size), 0) FROM model_locations WHERE node_id = ?",
            (node_id,)
        )
        return row[0] if row else 0

    def get_distinct_model_paths(self, node_id: str | None = None) -> set[str]:
        """Get all distinct model paths.

        Args:
            node_id: Optional filter by node

        Returns:
            Set of model paths
        """
        if node_id:
            rows = self._fetch_all(
                "SELECT DISTINCT model_path FROM model_locations WHERE node_id = ?",
                (node_id,)
            )
        else:
            rows = self._fetch_all("SELECT DISTINCT model_path FROM model_locations")
        return {row[0] for row in rows}

    def remove_model(self, model_path: str, node_id: str | None = None) -> int:
        """Remove model location(s).

        Args:
            model_path: Model path
            node_id: Optional node to remove from (all nodes if None)

        Returns:
            Number of rows removed
        """
        with self._connection() as conn:
            if node_id:
                cursor = conn.execute(
                    "DELETE FROM model_locations WHERE model_path = ? AND node_id = ?",
                    (model_path, node_id)
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM model_locations WHERE model_path = ?",
                    (model_path,)
                )
            conn.commit()
            return cursor.rowcount

    def export_for_node(self, node_id: str) -> list[dict[str, Any]]:
        """Export model registrations for a node (for P2P gossip).

        Args:
            node_id: Node identifier

        Returns:
            List of model dicts
        """
        rows = self._fetch_all("""
            SELECT model_path, board_type, num_players, model_version,
                   file_size, last_seen
            FROM model_locations
            WHERE node_id = ?
        """, (node_id,))

        return [
            {
                "model_path": row[0],
                "board_type": row[1],
                "num_players": row[2],
                "model_version": row[3],
                "file_size": row[4],
                "last_seen": row[5],
            }
            for row in rows
        ]

    def import_from_remote(
        self,
        node_id: str,
        models: list[dict[str, Any]],
    ) -> int:
        """Import model registrations from remote node.

        Args:
            node_id: Source node identifier
            models: List of model dicts

        Returns:
            Number of models imported
        """
        imported = 0
        for model in models:
            self.register_model(
                model_path=model["model_path"],
                node_id=node_id,
                board_type=model.get("board_type"),
                num_players=model.get("num_players"),
                model_version=model.get("model_version"),
                file_size=model.get("file_size", 0),
            )
            imported += 1
        return imported
