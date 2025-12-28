"""Game Location Registry - tracks game locations across the cluster.

Extracted from ClusterManifest to improve maintainability.

December 2025 - ClusterManifest decomposition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.distributed.registries.base import BaseRegistry

logger = logging.getLogger(__name__)


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


class GameLocationRegistry(BaseRegistry):
    """Registry for game locations across the cluster.

    Tracks which games exist on which nodes and in which databases.
    """

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
                except Exception as e:
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
        rows = self._fetch_all("""
            SELECT game_id, node_id, db_path, board_type, num_players,
                   engine_mode, registered_at, last_seen
            FROM game_locations
            WHERE game_id = ?
        """, (game_id,))

        return [
            GameLocation(
                game_id=row[0],
                node_id=row[1],
                db_path=row[2],
                board_type=row[3],
                num_players=row[4],
                engine_mode=row[5],
                registered_at=row[6],
                last_seen=row[7],
            )
            for row in rows
        ]

    def get_replication_count(self, game_id: str) -> int:
        """Get number of nodes where a game is replicated.

        Args:
            game_id: Game identifier

        Returns:
            Number of nodes with the game
        """
        row = self._fetch_one(
            "SELECT COUNT(DISTINCT node_id) FROM game_locations WHERE game_id = ?",
            (game_id,)
        )
        return row[0] if row else 0

    def get_under_replicated_games(
        self,
        min_copies: int = 2,
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

        rows = self._fetch_all(query, tuple(params))
        return [(row[0], row[1]) for row in rows]

    def get_games_by_node(self, node_id: str) -> list[GameLocation]:
        """Get all games on a specific node.

        Args:
            node_id: Node identifier

        Returns:
            List of GameLocation objects
        """
        rows = self._fetch_all("""
            SELECT game_id, node_id, db_path, board_type, num_players,
                   engine_mode, registered_at, last_seen
            FROM game_locations
            WHERE node_id = ?
            ORDER BY last_seen DESC
        """, (node_id,))

        return [
            GameLocation(
                game_id=row[0],
                node_id=row[1],
                db_path=row[2],
                board_type=row[3],
                num_players=row[4],
                engine_mode=row[5],
                registered_at=row[6],
                last_seen=row[7],
            )
            for row in rows
        ]

    def get_games_for_config(
        self,
        board_type: str,
        num_players: int,
        limit: int = 10000,
    ) -> list[GameLocation]:
        """Get all games for a specific configuration.

        Args:
            board_type: Board type
            num_players: Number of players
            limit: Maximum results

        Returns:
            List of GameLocation objects
        """
        rows = self._fetch_all("""
            SELECT game_id, node_id, db_path, board_type, num_players,
                   engine_mode, registered_at, last_seen
            FROM game_locations
            WHERE board_type = ? AND num_players = ?
            ORDER BY last_seen DESC
            LIMIT ?
        """, (board_type, num_players, limit))

        return [
            GameLocation(
                game_id=row[0],
                node_id=row[1],
                db_path=row[2],
                board_type=row[3],
                num_players=row[4],
                engine_mode=row[5],
                registered_at=row[6],
                last_seen=row[7],
            )
            for row in rows
        ]

    def count_games_by_node(self) -> dict[str, int]:
        """Get game count for each node.

        Returns:
            Dict mapping node_id to game count
        """
        rows = self._fetch_all("""
            SELECT node_id, COUNT(*) FROM game_locations GROUP BY node_id
        """)
        return {row[0]: row[1] for row in rows}

    def count_games_by_config(self) -> dict[str, int]:
        """Get game count for each configuration.

        Returns:
            Dict mapping config key (e.g., "hex8_2p") to game count
        """
        rows = self._fetch_all("""
            SELECT board_type || '_' || num_players || 'p', COUNT(DISTINCT game_id)
            FROM game_locations
            WHERE board_type IS NOT NULL
            GROUP BY board_type, num_players
        """)
        return {row[0]: row[1] for row in rows}

    def get_total_unique_games(self) -> int:
        """Get total unique game count.

        Returns:
            Number of unique games
        """
        row = self._fetch_one("SELECT COUNT(DISTINCT game_id) FROM game_locations")
        return row[0] if row else 0

    def get_distinct_db_paths(self, node_id: str | None = None) -> set[str]:
        """Get all distinct database paths.

        Args:
            node_id: Optional filter by node

        Returns:
            Set of database paths
        """
        if node_id:
            rows = self._fetch_all(
                "SELECT DISTINCT db_path FROM game_locations WHERE node_id = ?",
                (node_id,)
            )
        else:
            rows = self._fetch_all("SELECT DISTINCT db_path FROM game_locations")
        return {row[0] for row in rows}

    def remove_game(self, game_id: str, node_id: str | None = None) -> int:
        """Remove game location(s).

        Args:
            game_id: Game identifier
            node_id: Optional node to remove from (all nodes if None)

        Returns:
            Number of rows removed
        """
        with self._connection() as conn:
            if node_id:
                cursor = conn.execute(
                    "DELETE FROM game_locations WHERE game_id = ? AND node_id = ?",
                    (game_id, node_id)
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM game_locations WHERE game_id = ?",
                    (game_id,)
                )
            conn.commit()
            return cursor.rowcount

    def export_for_node(self, node_id: str) -> list[dict[str, Any]]:
        """Export game registrations for a node (for P2P gossip).

        Args:
            node_id: Node identifier

        Returns:
            List of game dicts
        """
        rows = self._fetch_all("""
            SELECT game_id, db_path, board_type, num_players, engine_mode, last_seen
            FROM game_locations
            WHERE node_id = ?
        """, (node_id,))

        return [
            {
                "game_id": row[0],
                "db_path": row[1],
                "board_type": row[2],
                "num_players": row[3],
                "engine_mode": row[4],
                "last_seen": row[5],
            }
            for row in rows
        ]

    def import_from_remote(
        self,
        node_id: str,
        games: list[dict[str, Any]],
    ) -> int:
        """Import game registrations from remote node.

        Args:
            node_id: Source node identifier
            games: List of game dicts

        Returns:
            Number of games imported
        """
        imported = 0
        for game in games:
            self.register_game(
                game_id=game["game_id"],
                node_id=node_id,
                db_path=game["db_path"],
                board_type=game.get("board_type"),
                num_players=game.get("num_players"),
                engine_mode=game.get("engine_mode"),
            )
            imported += 1
        return imported
