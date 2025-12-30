"""Data location registry for ClusterManifest.

This module provides the DataLocationRegistry class which handles registration
and lookup for all data types tracked in the cluster manifest:
- Game locations
- Model locations
- NPZ training file locations
- Checkpoint locations
- Database file locations

Extracted from ClusterManifest for improved testability and separation
of concerns (December 2025).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator

if TYPE_CHECKING:
    from app.distributed.cluster_manifest import (
        CheckpointLocation,
        GameLocation,
        ModelLocation,
        NPZLocation,
    )

logger = logging.getLogger(__name__)

# December 29, 2025: Cache configuration
# 5-minute TTL for manifest cache by default (configurable via env var)
DEFAULT_CACHE_TTL_SECONDS = int(os.environ.get("RINGRIFT_MANIFEST_CACHE_TTL", "300"))

__all__ = [
    "DataLocationRegistry",
    "CacheEntry",
    "RegistryCache",
]


@dataclass
class CacheEntry:
    """A cached value with TTL tracking.

    December 29, 2025: Added to reduce database queries for frequently
    accessed manifest data.
    """

    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS

    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - self.timestamp < self.ttl_seconds


class RegistryCache:
    """TTL-based cache for registry queries.

    December 29, 2025: Implements a simple in-memory cache with:
    - 5-minute TTL (configurable)
    - Automatic invalidation on write operations
    - Per-key storage for efficient lookups

    Usage:
        cache = RegistryCache(ttl_seconds=300)
        cache.set("game:123", locations)
        result = cache.get("game:123")  # Returns cached or None
        cache.invalidate_prefix("game:")  # Invalidate all game entries
    """

    def __init__(self, ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS):
        """Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default 5 min)
        """
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, CacheEntry] = {}
        self._hit_count = 0
        self._miss_count = 0

    def get(self, key: str) -> Any | None:
        """Get a cached value if valid.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        entry = self._cache.get(key)
        if entry is not None and entry.is_valid():
            self._hit_count += 1
            return entry.value
        if entry is not None:
            # Entry expired, remove it
            del self._cache[key]
        self._miss_count += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Store a value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = CacheEntry(
            value=value,
            ttl_seconds=self.ttl_seconds,
        )

    def invalidate(self, key: str) -> None:
        """Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate
        """
        self._cache.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> int:
        """Invalidate all cache entries with given prefix.

        Args:
            prefix: Key prefix to match (e.g., "game:", "model:")

        Returns:
            Number of entries invalidated
        """
        keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._cache[key]
        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hit/miss counts, entry count, hit ratio
        """
        total = self._hit_count + self._miss_count
        return {
            "entries": len(self._cache),
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_ratio": self._hit_count / max(1, total),
            "ttl_seconds": self.ttl_seconds,
        }


class DataLocationRegistry:
    """Manages location tracking for games, models, NPZ, and checkpoints.

    This class handles all data registration and lookup operations for the
    cluster manifest. It uses a shared database connection factory to ensure
    thread-safe access to the underlying SQLite database.

    Usage
    -----
    ```python
    from app.distributed.data_location_registry import DataLocationRegistry

    # Typically instantiated by ClusterManifest
    registry = DataLocationRegistry(
        db_path=Path("data/cluster_manifest.db"),
        connection_factory=manifest._connection,
        node_id="my-node",
    )

    # Register a game location
    registry.register_game("game-123", "node-1", "/data/games/selfplay.db")

    # Find all locations for a game
    locations = registry.find_game("game-123")
    ```
    """

    def __init__(
        self,
        db_path: Path,
        connection_factory: Callable[[], Generator[sqlite3.Connection, None, None]],
        node_id: str,
    ):
        """Initialize the data location registry.

        Parameters
        ----------
        db_path:
            Path to the SQLite database file.
        connection_factory:
            A callable that returns a context manager yielding a database connection.
            This is typically `ClusterManifest._connection`.
        node_id:
            The identifier for the local node.
        """
        self.db_path = db_path
        self._connection = connection_factory
        self.node_id = node_id

        # December 29, 2025: Add caching for frequently queried data
        # Cache TTL defaults to 5 minutes (300 seconds)
        self._cache = RegistryCache(ttl_seconds=DEFAULT_CACHE_TTL_SECONDS)
        logger.debug(
            f"[DataLocationRegistry] Cache enabled with TTL={DEFAULT_CACHE_TTL_SECONDS}s"
        )

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache hit/miss counts and ratio
        """
        return self._cache.get_stats()

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        logger.debug("[DataLocationRegistry] Cache cleared")

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

        Parameters
        ----------
        game_id:
            Unique game identifier.
        node_id:
            Node where the game exists.
        db_path:
            Path to database containing the game.
        board_type:
            Board configuration (e.g., "hex8", "square8").
        num_players:
            Number of players.
        engine_mode:
            Engine mode used (e.g., "gumbel-mcts").
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
        # Invalidate cache for this game
        self._cache.invalidate(f"game:{game_id}")

    def register_games_batch(
        self,
        games: list[tuple[str, str, str]],
        board_type: str | None = None,
        num_players: int | None = None,
        engine_mode: str | None = None,
    ) -> int:
        """Register multiple game locations efficiently.

        Parameters
        ----------
        games:
            List of (game_id, node_id, db_path) tuples.
        board_type:
            Board configuration.
        num_players:
            Number of players.
        engine_mode:
            Engine mode.

        Returns
        -------
        int
            Number of games registered.
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

    def mark_games_consolidated(
        self,
        game_ids: list[str],
        canonical_db: str,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> int:
        """Mark games as consolidated into a canonical database.

        Parameters
        ----------
        game_ids:
            List of game IDs that were consolidated.
        canonical_db:
            Path to canonical database they were merged into.
        board_type:
            Optional board type filter.
        num_players:
            Optional player count filter.

        Returns
        -------
        int
            Number of games marked as consolidated.
        """
        if not game_ids:
            return 0

        now = time.time()
        marked = 0

        with self._connection() as conn:
            cursor = conn.cursor()
            for game_id in game_ids:
                try:
                    query = """
                        UPDATE game_locations
                        SET is_consolidated = 1, consolidated_at = ?, canonical_db = ?
                        WHERE game_id = ?
                    """
                    params: list[Any] = [now, canonical_db, game_id]

                    if board_type:
                        query += " AND board_type = ?"
                        params.append(board_type)
                    if num_players:
                        query += " AND num_players = ?"
                        params.append(num_players)

                    cursor.execute(query, params)
                    marked += cursor.rowcount
                except sqlite3.Error as e:
                    logger.warning(f"Failed to mark game {game_id} as consolidated: {e}")
            conn.commit()

        logger.info(f"Marked {marked} games as consolidated into {canonical_db}")
        return marked

    def get_unconsolidated_games(
        self,
        board_type: str,
        num_players: int,
        limit: int = 10000,
    ) -> list["GameLocation"]:
        """Get games that haven't been consolidated yet.

        Parameters
        ----------
        board_type:
            Board configuration.
        num_players:
            Number of players.
        limit:
            Maximum number of games to return.

        Returns
        -------
        list[GameLocation]
            List of unconsolidated GameLocation objects.
        """
        # Import here to avoid circular dependency
        from app.distributed.cluster_manifest import GameLocation

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT game_id, node_id, db_path, board_type, num_players,
                       engine_mode, registered_at, last_seen,
                       is_consolidated, consolidated_at, canonical_db
                FROM game_locations
                WHERE board_type = ? AND num_players = ?
                  AND (is_consolidated = 0 OR is_consolidated IS NULL)
                ORDER BY registered_at DESC
                LIMIT ?
            """, (board_type, num_players, limit))

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
                    is_consolidated=bool(row[8]) if row[8] is not None else False,
                    consolidated_at=row[9] or 0.0,
                    canonical_db=row[10],
                ))

            return locations

    def find_game(self, game_id: str) -> list["GameLocation"]:
        """Find all locations where a game exists.

        Parameters
        ----------
        game_id:
            Game identifier.

        Returns
        -------
        list[GameLocation]
            List of GameLocation objects.

        Note
        ----
        December 29, 2025: Results are cached with 5-minute TTL.
        """
        # Check cache first
        cache_key = f"game:{game_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        from app.distributed.cluster_manifest import GameLocation

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT game_id, node_id, db_path, board_type, num_players,
                       engine_mode, registered_at, last_seen,
                       is_consolidated, consolidated_at, canonical_db
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
                    is_consolidated=bool(row[8]) if row[8] is not None else False,
                    consolidated_at=row[9] or 0.0,
                    canonical_db=row[10],
                ))

            # Cache the result
            self._cache.set(cache_key, locations)
            return locations

    def get_game_replication_count(self, game_id: str) -> int:
        """Get number of nodes where a game is replicated.

        Parameters
        ----------
        game_id:
            Game identifier.

        Returns
        -------
        int
            Number of nodes with the game.
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
        min_copies: int = 2,
        board_type: str | None = None,
        num_players: int | None = None,
        limit: int = 1000,
    ) -> list[tuple[str, int]]:
        """Find games that exist on fewer than min_copies nodes.

        Parameters
        ----------
        min_copies:
            Minimum required copies.
        board_type:
            Optional filter.
        num_players:
            Optional filter.
        limit:
            Maximum results.

        Returns
        -------
        list[tuple[str, int]]
            List of (game_id, current_copies) tuples.
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

    def get_game_locations(self) -> dict[str, Any]:
        """Get game locations grouped by game_id.

        Returns
        -------
        dict[str, Any]
            Dict mapping game_id to location info.
        """
        result: dict[str, Any] = {}

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT game_id, node_id, db_path, board_type, num_players
                FROM game_locations
                ORDER BY game_id
            """)

            for row in cursor:
                game_id, node_id, db_path, board_type, num_players = row
                if game_id not in result:
                    result[game_id] = {
                        "locations": [],
                        "db_paths": [],
                        "board_type": board_type,
                        "num_players": num_players,
                    }
                result[game_id]["locations"].append(node_id)
                result[game_id]["db_paths"].append(db_path)

        return result

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

        Parameters
        ----------
        model_path:
            Relative path to model file.
        node_id:
            Node where the model exists.
        board_type:
            Board configuration.
        num_players:
            Number of players.
        model_version:
            Model version string.
        file_size:
            File size in bytes.
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
        # Invalidate cache for this model and config
        self._cache.invalidate(f"model:{model_path}")
        if board_type and num_players:
            self._cache.invalidate(f"models_config:{board_type}_{num_players}p")

    def find_model(self, model_path: str) -> list["ModelLocation"]:
        """Find all locations where a model exists.

        Parameters
        ----------
        model_path:
            Model file path.

        Returns
        -------
        list[ModelLocation]
            List of ModelLocation objects.

        Note
        ----
        December 29, 2025: Results are cached with 5-minute TTL.
        """
        # Check cache first
        cache_key = f"model:{model_path}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        from app.distributed.cluster_manifest import ModelLocation

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

            # Cache the result
            self._cache.set(cache_key, locations)
            return locations

    def find_models_for_config(
        self,
        board_type: str,
        num_players: int,
    ) -> list["ModelLocation"]:
        """Find all models for a specific board configuration.

        Parameters
        ----------
        board_type:
            Board configuration.
        num_players:
            Number of players.

        Returns
        -------
        list[ModelLocation]
            List of ModelLocation objects.

        Note
        ----
        December 29, 2025: Results are cached with 5-minute TTL.
        """
        # Check cache first
        cache_key = f"models_config:{board_type}_{num_players}p"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        from app.distributed.cluster_manifest import ModelLocation

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

            # Cache the result
            self._cache.set(cache_key, locations)
            return locations

    def get_model_availability_score(self, model_path: str) -> float:
        """Calculate availability score for a model across the cluster.

        Parameters
        ----------
        model_path:
            Path to the model file.

        Returns
        -------
        float
            Score from 0.0 to 1.0 where 1.0 means available on all GPU nodes.
        """
        locations = self.find_model(model_path)
        if not locations:
            return 0.0

        total_gpu_nodes = self.count_gpu_nodes()
        if total_gpu_nodes == 0:
            return 0.0

        return min(1.0, len(locations) / total_gpu_nodes)

    def count_gpu_nodes(self) -> int:
        """Count total GPU-capable nodes known to the manifest.

        Returns
        -------
        int
            Count of GPU-capable nodes (minimum 1 to avoid division by zero).
        """
        with self._connection() as conn:
            cursor = conn.cursor()

            # First try: count nodes with capacity data (GPU nodes report this)
            cursor.execute("SELECT COUNT(DISTINCT node_id) FROM node_capacity")
            count = cursor.fetchone()[0]

            if count > 0:
                return count

            # Fallback: count unique nodes with model locations
            cursor.execute("SELECT COUNT(DISTINCT node_id) FROM model_locations")
            count = cursor.fetchone()[0]

            return max(1, count)

    def sync_model_locations_from_peers(
        self,
        peer_locations: list[dict],
        max_age_seconds: float = 3600.0,
    ) -> int:
        """Sync model locations from peer node manifests.

        Parameters
        ----------
        peer_locations:
            List of dicts with model location data.
        max_age_seconds:
            Skip entries older than this (default: 1 hour).

        Returns
        -------
        int
            Number of locations inserted/updated.
        """
        if not peer_locations:
            return 0

        now = time.time()
        min_timestamp = now - max_age_seconds
        count = 0

        with self._connection() as conn:
            cursor = conn.cursor()

            for loc in peer_locations:
                last_seen = loc.get("last_seen", 0)
                if last_seen < min_timestamp:
                    continue

                cursor.execute("""
                    INSERT OR REPLACE INTO model_locations
                    (model_path, node_id, board_type, num_players, model_version,
                     file_size, registered_at, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?,
                        COALESCE((SELECT registered_at FROM model_locations
                                  WHERE model_path = ? AND node_id = ?), ?),
                        ?)
                """, (
                    loc.get("model_path"),
                    loc.get("node_id"),
                    loc.get("board_type"),
                    loc.get("num_players"),
                    loc.get("model_version"),
                    loc.get("file_size", 0),
                    loc.get("model_path"),
                    loc.get("node_id"),
                    loc.get("registered_at", now),
                    max(last_seen, loc.get("registered_at", now)),
                ))
                count += 1

            conn.commit()

        return count

    def get_all_model_locations(self) -> list[dict]:
        """Get all model locations as dicts for sync/export.

        Returns
        -------
        list[dict]
            List of dicts with model location data.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_path, node_id, board_type, num_players,
                       model_version, file_size, registered_at, last_seen
                FROM model_locations
                ORDER BY last_seen DESC
            """)

            return [
                {
                    "model_path": row[0],
                    "node_id": row[1],
                    "board_type": row[2],
                    "num_players": row[3],
                    "model_version": row[4],
                    "file_size": row[5],
                    "registered_at": row[6],
                    "last_seen": row[7],
                }
                for row in cursor.fetchall()
            ]

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

        Parameters
        ----------
        npz_path:
            Relative path to NPZ file.
        node_id:
            Node where the file exists.
        board_type:
            Board configuration.
        num_players:
            Number of players.
        sample_count:
            Number of training samples.
        file_size:
            File size in bytes.
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
        # Invalidate cache for this config
        if board_type and num_players:
            self._cache.invalidate(f"npz_config:{board_type}_{num_players}p")

    def find_npz_for_config(
        self,
        board_type: str,
        num_players: int,
    ) -> list["NPZLocation"]:
        """Find all NPZ files for a specific board configuration.

        Parameters
        ----------
        board_type:
            Board configuration.
        num_players:
            Number of players.

        Returns
        -------
        list[NPZLocation]
            List of NPZLocation objects.

        Note
        ----
        December 29, 2025: Results are cached with 5-minute TTL.
        """
        # Check cache first
        cache_key = f"npz_config:{board_type}_{num_players}p"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        from app.distributed.cluster_manifest import NPZLocation

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

            # Cache the result
            self._cache.set(cache_key, locations)
            return locations

    # =========================================================================
    # Checkpoint Location Registry
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

        Parameters
        ----------
        checkpoint_path:
            Relative path to checkpoint file.
        node_id:
            Node where the checkpoint exists.
        config_key:
            Configuration key (e.g., "hex8_2p").
        board_type:
            Board configuration.
        num_players:
            Number of players.
        epoch:
            Training epoch number.
        step:
            Training step number.
        loss:
            Training loss at this checkpoint.
        file_size:
            File size in bytes.
        is_best:
            Whether this is the best checkpoint for this config.
        """
        now = time.time()

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

        # Invalidate cache for this checkpoint and config
        self._cache.invalidate(f"checkpoint:{checkpoint_path}")
        if config_key:
            self._cache.invalidate(f"checkpoints_config:{config_key}")
            self._cache.invalidate(f"checkpoints_config:{config_key}_best")
            self._cache.invalidate(f"latest_checkpoint:{config_key}")

        logger.debug(f"Registered checkpoint: {checkpoint_path} on {node_id} (epoch={epoch})")

    def find_checkpoint(self, checkpoint_path: str) -> list["CheckpointLocation"]:
        """Find all locations where a checkpoint exists.

        Parameters
        ----------
        checkpoint_path:
            Checkpoint file path.

        Returns
        -------
        list[CheckpointLocation]
            List of CheckpointLocation objects.

        Note
        ----
        December 29, 2025: Results are cached with 5-minute TTL.
        """
        # Check cache first
        cache_key = f"checkpoint:{checkpoint_path}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        from app.distributed.cluster_manifest import CheckpointLocation

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

            # Cache the result
            self._cache.set(cache_key, locations)
            return locations

    def find_checkpoints_for_config(
        self,
        config_key: str,
        only_best: bool = False,
    ) -> list["CheckpointLocation"]:
        """Find all checkpoints for a specific configuration.

        Parameters
        ----------
        config_key:
            Configuration key (e.g., "hex8_2p").
        only_best:
            If True, only return best checkpoints.

        Returns
        -------
        list[CheckpointLocation]
            List of CheckpointLocation objects, sorted by epoch descending.

        Note
        ----
        December 29, 2025: Results are cached with 5-minute TTL.
        """
        # Check cache first
        cache_key = f"checkpoints_config:{config_key}{'_best' if only_best else ''}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        from app.distributed.cluster_manifest import CheckpointLocation

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

            # Cache the result
            self._cache.set(cache_key, locations)
            return locations

    def get_latest_checkpoint_for_config(
        self,
        config_key: str,
        prefer_best: bool = True,
    ) -> "CheckpointLocation | None":
        """Get the latest checkpoint for a configuration.

        Parameters
        ----------
        config_key:
            Configuration key (e.g., "hex8_2p").
        prefer_best:
            If True, prefer best checkpoint over latest epoch.

        Returns
        -------
        CheckpointLocation | None
            CheckpointLocation or None if not found.
        """
        from app.distributed.cluster_manifest import CheckpointLocation

        with self._connection() as conn:
            cursor = conn.cursor()

            if prefer_best:
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

        Parameters
        ----------
        config_key:
            Configuration key.
        checkpoint_path:
            Path to the checkpoint to mark as best.
        """
        now = time.time()
        with self._connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE checkpoint_locations
                SET is_best = 0, last_seen = ?
                WHERE config_key = ? AND is_best = 1
            """, (now, config_key))

            cursor.execute("""
                UPDATE checkpoint_locations
                SET is_best = 1, last_seen = ?
                WHERE config_key = ? AND checkpoint_path = ?
            """, (now, config_key, checkpoint_path))

            conn.commit()

        logger.info(f"Marked {checkpoint_path} as best for {config_key}")

    # =========================================================================
    # Database Location Registry
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

        Parameters
        ----------
        db_path:
            Path to database file (absolute or relative).
        node_id:
            Node where the database exists.
        board_type:
            Board configuration.
        num_players:
            Number of players.
        config_key:
            Configuration key (e.g., "hex8_2p").
        game_count:
            Initial game count (usually 0).
        file_size:
            File size in bytes.
        engine_mode:
            Engine mode used.
        """
        now = time.time()

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

        Parameters
        ----------
        db_path:
            Path to database file.
        node_id:
            Node where the database exists.
        game_count:
            Current game count.
        file_size:
            Optional updated file size.
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

        Parameters
        ----------
        config_key:
            Configuration key (e.g., "hex8_2p").
        board_type:
            Board configuration (alternative to config_key).
        num_players:
            Number of players (alternative to config_key).

        Returns
        -------
        list[dict[str, Any]]
            List of database location dictionaries.
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

        Returns
        -------
        list[dict[str, Any]]
            List of all database location dictionaries.
        """
        return self.find_databases_for_config()
