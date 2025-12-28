"""Checkpoint Registry - tracks training checkpoint locations across the cluster.

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
class CheckpointLocation:
    """Location of a training checkpoint in the cluster."""
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


class CheckpointRegistry(BaseRegistry):
    """Registry for training checkpoint locations across the cluster.

    Tracks checkpoints for distributed training resume and failover.
    """

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
        """Register a training checkpoint location.

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
        rows = self._fetch_all("""
            SELECT checkpoint_path, node_id, config_key, board_type, num_players,
                   epoch, step, loss, file_size, is_best, registered_at, last_seen
            FROM checkpoint_locations
            WHERE checkpoint_path = ?
        """, (checkpoint_path,))

        return [
            CheckpointLocation(
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
            for row in rows
        ]

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
        if only_best:
            rows = self._fetch_all("""
                SELECT checkpoint_path, node_id, config_key, board_type, num_players,
                       epoch, step, loss, file_size, is_best, registered_at, last_seen
                FROM checkpoint_locations
                WHERE config_key = ? AND is_best = 1
                ORDER BY epoch DESC, last_seen DESC
            """, (config_key,))
        else:
            rows = self._fetch_all("""
                SELECT checkpoint_path, node_id, config_key, board_type, num_players,
                       epoch, step, loss, file_size, is_best, registered_at, last_seen
                FROM checkpoint_locations
                WHERE config_key = ?
                ORDER BY epoch DESC, last_seen DESC
            """, (config_key,))

        return [
            CheckpointLocation(
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
            for row in rows
        ]

    def get_latest_checkpoint(
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
        if prefer_best:
            # First try to find best checkpoint
            row = self._fetch_one("""
                SELECT checkpoint_path, node_id, config_key, board_type, num_players,
                       epoch, step, loss, file_size, is_best, registered_at, last_seen
                FROM checkpoint_locations
                WHERE config_key = ? AND is_best = 1
                ORDER BY epoch DESC, last_seen DESC
                LIMIT 1
            """, (config_key,))

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
        row = self._fetch_one("""
            SELECT checkpoint_path, node_id, config_key, board_type, num_players,
                   epoch, step, loss, file_size, is_best, registered_at, last_seen
            FROM checkpoint_locations
            WHERE config_key = ?
            ORDER BY epoch DESC, last_seen DESC
            LIMIT 1
        """, (config_key,))

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

    def mark_as_best(
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

    def get_total_unique_checkpoints(self) -> int:
        """Get total unique checkpoint count.

        Returns:
            Number of unique checkpoints
        """
        row = self._fetch_one(
            "SELECT COUNT(DISTINCT checkpoint_path) FROM checkpoint_locations"
        )
        return row[0] if row else 0

    def get_best_checkpoints_by_config(self) -> dict[str, int]:
        """Get best checkpoint counts by config.

        Returns:
            Dict mapping config_key to count of best checkpoints
        """
        rows = self._fetch_all("""
            SELECT config_key, COUNT(DISTINCT checkpoint_path)
            FROM checkpoint_locations
            WHERE is_best = 1 AND config_key IS NOT NULL
            GROUP BY config_key
        """)
        return {row[0]: row[1] for row in rows}

    def export_for_node(self, node_id: str) -> list[dict[str, Any]]:
        """Export checkpoint registrations for a node (for P2P gossip).

        Args:
            node_id: Node identifier

        Returns:
            List of checkpoint dicts
        """
        rows = self._fetch_all("""
            SELECT checkpoint_path, config_key, board_type, num_players,
                   epoch, step, loss, file_size, is_best, last_seen
            FROM checkpoint_locations
            WHERE node_id = ?
        """, (node_id,))

        return [
            {
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
            }
            for row in rows
        ]

    def import_from_remote(
        self,
        node_id: str,
        checkpoints: list[dict[str, Any]],
    ) -> int:
        """Import checkpoint registrations from remote node.

        Args:
            node_id: Source node identifier
            checkpoints: List of checkpoint dicts

        Returns:
            Number of checkpoints imported
        """
        imported = 0
        for checkpoint in checkpoints:
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
        return imported
