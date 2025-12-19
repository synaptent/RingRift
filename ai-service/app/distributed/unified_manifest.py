"""Unified Data Manifest for RingRift AI Service.

This module provides a single DataManifest implementation for tracking synced games,
consolidating functionality from:
- scripts/streaming_data_collector.py:DataManifest
- app/distributed/unified_data_sync.py:DataManifest

Key features:
1. Game deduplication by ID and content hash
2. Host sync state tracking
3. Sync history logging
4. Dead letter queue for failed syncs
5. Content-based deduplication

Usage:
    from app.distributed.unified_manifest import DataManifest, HostSyncState

    manifest = DataManifest(db_path=Path("data/data_manifest.db"))

    # Check if game is synced
    if not manifest.is_game_synced(game_id):
        # Sync the game
        manifest.mark_games_synced([game_id], source_host, source_db)

    # Track host state
    manifest.save_host_state(host_state)
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import event helpers for quality events (December 2025)
try:
    from app.distributed.event_helpers import emit_sync, has_event_bus
    HAS_EVENT_HELPERS = True
except ImportError:
    HAS_EVENT_HELPERS = False

    def emit_sync(*args, **kwargs):
        return False

    def has_event_bus():
        return False


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class HostSyncState:
    """Sync state for a remote host."""
    name: str
    last_sync_time: float = 0.0
    last_game_count: int = 0
    total_games_synced: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    last_error_time: float = 0.0
    # Extended fields for ephemeral host support
    is_ephemeral: bool = False
    storage_type: str = "persistent"
    poll_interval: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "last_sync_time": self.last_sync_time,
            "last_game_count": self.last_game_count,
            "total_games_synced": self.total_games_synced,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time,
            "is_ephemeral": self.is_ephemeral,
            "storage_type": self.storage_type,
            "poll_interval": self.poll_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HostSyncState":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            last_sync_time=data.get("last_sync_time", 0.0),
            last_game_count=data.get("last_game_count", 0),
            total_games_synced=data.get("total_games_synced", 0),
            consecutive_failures=data.get("consecutive_failures", 0),
            last_error=data.get("last_error", ""),
            last_error_time=data.get("last_error_time", 0.0),
            is_ephemeral=data.get("is_ephemeral", False),
            storage_type=data.get("storage_type", "persistent"),
            poll_interval=data.get("poll_interval", 60),
        )


@dataclass
class SyncHistoryEntry:
    """A single sync history entry."""
    id: int
    host_name: str
    sync_time: float
    games_synced: int
    duration_seconds: float
    success: bool
    sync_method: str = ""
    error_message: str = ""


@dataclass
class DeadLetterEntry:
    """A dead letter queue entry for failed sync."""
    id: int
    game_id: str
    source_host: str
    source_db: str
    error_message: str
    error_type: str
    added_at: float
    retry_count: int = 0
    last_retry_at: Optional[float] = None
    resolved: bool = False


@dataclass
class ManifestStats:
    """Statistics for the manifest."""
    total_games: int = 0
    games_by_host: Dict[str, int] = field(default_factory=dict)
    games_by_board_type: Dict[str, int] = field(default_factory=dict)
    recent_sync_count: int = 0  # Last 24 hours
    dead_letter_count: int = 0


@dataclass
class GameQualityMetadata:
    """Quality metadata for a game used in priority sync and training selection.

    Quality score is computed as a weighted combination of:
    - avg_player_elo: Higher Elo = better quality training data
    - game_length: Longer games typically have more learning signal
    - is_decisive: Decisive wins are preferred over draws
    - model_version: Recent model versions are preferred

    The quality_score is normalized to [0, 1] for easy comparison.
    """
    game_id: str
    avg_player_elo: float = 1500.0
    min_player_elo: float = 1500.0
    max_player_elo: float = 1500.0
    elo_difference: float = 0.0
    game_length: int = 0
    is_decisive: bool = False
    termination_reason: str = ""
    model_version: str = ""
    quality_score: float = 0.0
    created_at: float = 0.0

    @classmethod
    def compute_quality_score(
        cls,
        avg_elo: float = 1500.0,
        game_length: int = 0,
        is_decisive: bool = False,
        elo_weight: float = 0.4,
        length_weight: float = 0.3,
        decisive_weight: float = 0.3,
        min_elo: float = 1200.0,
        max_elo: float = 2400.0,
        min_length: int = 10,
        max_length: int = 200,
    ) -> float:
        """Compute a normalized quality score for a game.

        Args:
            avg_elo: Average Elo of players in the game
            game_length: Number of moves in the game
            is_decisive: Whether the game had a clear winner
            elo_weight: Weight for Elo component (0-1)
            length_weight: Weight for game length component (0-1)
            decisive_weight: Weight for decisive outcome component (0-1)
            min_elo: Minimum Elo for normalization
            max_elo: Maximum Elo for normalization
            min_length: Minimum game length for normalization
            max_length: Maximum game length for normalization

        Returns:
            Normalized quality score in [0, 1]
        """
        # Normalize Elo to [0, 1]
        elo_normalized = max(0.0, min(1.0, (avg_elo - min_elo) / (max_elo - min_elo)))

        # Normalize game length to [0, 1] with diminishing returns above max
        if game_length < min_length:
            length_normalized = 0.0
        elif game_length > max_length:
            length_normalized = 1.0
        else:
            length_normalized = (game_length - min_length) / (max_length - min_length)

        # Decisive outcome is binary
        decisive_normalized = 1.0 if is_decisive else 0.3  # Still give some credit to draws

        # Weighted combination
        score = (
            elo_weight * elo_normalized +
            length_weight * length_normalized +
            decisive_weight * decisive_normalized
        )

        return min(1.0, max(0.0, score))


@dataclass
class PriorityQueueEntry:
    """Entry in the sync priority queue."""
    id: int
    game_id: str
    source_host: str
    source_db: str
    priority_score: float
    avg_player_elo: Optional[float] = None
    game_length: Optional[int] = None
    is_decisive: bool = False
    queued_at: float = 0.0
    synced: bool = False
    synced_at: Optional[float] = None


# =============================================================================
# Unified DataManifest
# =============================================================================


class DataManifest:
    """Unified manifest for tracking synced games and host states.

    Provides:
    - Game ID deduplication
    - Content hash deduplication
    - Host sync state persistence
    - Sync history logging
    - Dead letter queue for failed syncs
    """

    SCHEMA_VERSION = "3.0"  # Added quality scores

    def __init__(self, db_path: Path):
        """Initialize the manifest.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the manifest database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript(f"""
            -- Synced games table with content hash and quality score support
            CREATE TABLE IF NOT EXISTS synced_games (
                game_id TEXT PRIMARY KEY,
                source_host TEXT NOT NULL,
                source_db TEXT NOT NULL,
                synced_at REAL NOT NULL,
                board_type TEXT,
                num_players INTEGER,
                content_hash TEXT,
                game_length INTEGER,
                winner TEXT,
                -- Quality score fields (v3.0)
                avg_player_elo REAL,
                min_player_elo REAL,
                max_player_elo REAL,
                elo_difference REAL,
                is_decisive INTEGER DEFAULT 0,
                termination_reason TEXT,
                quality_score REAL DEFAULT 0.0,
                model_version TEXT,
                created_at REAL
            );

            -- Priority sync queue for high-quality games
            CREATE TABLE IF NOT EXISTS sync_priority_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL UNIQUE,
                source_host TEXT NOT NULL,
                source_db TEXT NOT NULL,
                priority_score REAL NOT NULL,
                avg_player_elo REAL,
                game_length INTEGER,
                is_decisive INTEGER DEFAULT 0,
                queued_at REAL NOT NULL,
                synced INTEGER DEFAULT 0,
                synced_at REAL
            );

            -- Host sync state with ephemeral support
            CREATE TABLE IF NOT EXISTS host_states (
                host_name TEXT PRIMARY KEY,
                last_sync_time REAL,
                last_game_count INTEGER,
                total_games_synced INTEGER,
                consecutive_failures INTEGER,
                last_error TEXT,
                last_error_time REAL,
                is_ephemeral INTEGER DEFAULT 0,
                storage_type TEXT DEFAULT 'persistent',
                poll_interval INTEGER DEFAULT 60
            );

            -- Sync history for analytics
            CREATE TABLE IF NOT EXISTS sync_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                host_name TEXT NOT NULL,
                sync_time REAL NOT NULL,
                games_synced INTEGER NOT NULL,
                duration_seconds REAL,
                success INTEGER NOT NULL,
                sync_method TEXT,
                error_message TEXT
            );

            -- Dead letter queue for failed syncs
            CREATE TABLE IF NOT EXISTS dead_letter_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                source_host TEXT NOT NULL,
                source_db TEXT NOT NULL,
                error_message TEXT NOT NULL,
                error_type TEXT NOT NULL,
                added_at REAL NOT NULL,
                retry_count INTEGER DEFAULT 0,
                last_retry_at REAL,
                resolved INTEGER DEFAULT 0
            );

            -- Metadata table
            CREATE TABLE IF NOT EXISTS manifest_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            -- Initialize metadata
            INSERT OR IGNORE INTO manifest_metadata (key, value, updated_at)
            VALUES
                ('schema_version', '{self.SCHEMA_VERSION}', strftime('%s', 'now')),
                ('created_at', strftime('%s', 'now'), strftime('%s', 'now'));
        """)
        self._ensure_schema(conn)
        conn.commit()
        conn.close()
        logger.debug(f"Initialized unified manifest at {self.db_path}")

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Ensure newer schema additions exist on older databases."""
        cursor = conn.cursor()
        existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(synced_games)")}
        base_columns = {
            "board_type": "TEXT",
            "num_players": "INTEGER",
            "content_hash": "TEXT",
            "game_length": "INTEGER",
            "winner": "TEXT",
        }
        quality_columns = {
            "avg_player_elo": "REAL",
            "min_player_elo": "REAL",
            "max_player_elo": "REAL",
            "elo_difference": "REAL",
            "is_decisive": "INTEGER DEFAULT 0",
            "termination_reason": "TEXT",
            "quality_score": "REAL DEFAULT 0.0",
            "model_version": "TEXT",
            "created_at": "REAL",
        }
        for col, col_type in {**base_columns, **quality_columns}.items():
            if col not in existing_cols:
                cursor.execute(f"ALTER TABLE synced_games ADD COLUMN {col} {col_type}")

        # Ensure indexes exist (safe on older DBs).
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_synced_games_host "
            "ON synced_games(source_host)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_synced_games_time "
            "ON synced_games(synced_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_synced_games_content "
            "ON synced_games(content_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_synced_games_config "
            "ON synced_games(board_type, num_players)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_synced_games_quality "
            "ON synced_games(quality_score DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_synced_games_elo "
            "ON synced_games(avg_player_elo DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_synced_games_decisive "
            "ON synced_games(is_decisive, quality_score DESC)"
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_priority_queue_priority "
            "ON sync_priority_queue(synced, priority_score DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_priority_queue_host "
            "ON sync_priority_queue(source_host, synced)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sync_history_time "
            "ON sync_history(sync_time)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sync_history_host "
            "ON sync_history(host_name, sync_time)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_dead_letter_unresolved "
            "ON dead_letter_queue(resolved, added_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_dead_letter_host "
            "ON dead_letter_queue(source_host, resolved)"
        )

        cursor.execute(
            "INSERT OR IGNORE INTO manifest_metadata (key, value, updated_at) "
            "VALUES ('schema_version', ?, strftime('%s', 'now'))",
            (self.SCHEMA_VERSION,),
        )
        cursor.execute(
            "UPDATE manifest_metadata SET value = ?, updated_at = strftime('%s', 'now') "
            "WHERE key = 'schema_version'",
            (self.SCHEMA_VERSION,),
        )

    # =========================================================================
    # Game Deduplication
    # =========================================================================

    def is_game_synced(self, game_id: str) -> bool:
        """Check if a game has already been synced.

        Args:
            game_id: Game identifier to check

        Returns:
            True if game is already synced
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM synced_games WHERE game_id = ?", (game_id,))
        result = cursor.fetchone() is not None
        conn.close()
        return result

    def is_content_synced(self, content_hash: str) -> bool:
        """Check if content with this hash has been synced.

        Args:
            content_hash: SHA256 hash of game content

        Returns:
            True if content is already synced
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM synced_games WHERE content_hash = ?", (content_hash,))
        result = cursor.fetchone() is not None
        conn.close()
        return result

    def get_unsynced_game_ids(self, game_ids: List[str]) -> List[str]:
        """Filter list to only unsynced game IDs.

        Args:
            game_ids: List of game IDs to check

        Returns:
            List of game IDs that haven't been synced
        """
        if not game_ids:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Batch check
        placeholders = ",".join("?" * len(game_ids))
        cursor.execute(f"SELECT game_id FROM synced_games WHERE game_id IN ({placeholders})", game_ids)
        synced = {row[0] for row in cursor.fetchall()}
        conn.close()

        return [gid for gid in game_ids if gid not in synced]

    def mark_games_synced(
        self,
        game_ids: List[str],
        source_host: str,
        source_db: str,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
        content_hashes: Optional[List[str]] = None,
    ) -> int:
        """Mark games as synced.

        Args:
            game_ids: List of game IDs to mark
            source_host: Source host name
            source_db: Source database path
            board_type: Board type (square8, hexagonal, etc.)
            num_players: Number of players
            content_hashes: Optional content hashes for deduplication

        Returns:
            Number of games marked (excludes duplicates)
        """
        if not game_ids:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()
        marked = 0

        for i, game_id in enumerate(game_ids):
            content_hash = content_hashes[i] if content_hashes and i < len(content_hashes) else None
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO synced_games
                    (game_id, source_host, source_db, synced_at, board_type, num_players, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (game_id, source_host, source_db, now, board_type, num_players, content_hash))
                if cursor.rowcount > 0:
                    marked += 1
            except sqlite3.Error as e:
                logger.warning(f"Failed to mark game {game_id}: {e}")

        conn.commit()
        conn.close()
        return marked

    def mark_games_synced_with_quality(
        self,
        games: List[GameQualityMetadata],
        source_host: str,
        source_db: str,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ) -> int:
        """Mark games as synced with quality metadata.

        Args:
            games: List of GameQualityMetadata objects
            source_host: Source host name
            source_db: Source database path
            board_type: Board type (square8, hexagonal, etc.)
            num_players: Number of players

        Returns:
            Number of games marked (excludes duplicates)
        """
        if not games:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()
        marked = 0

        for game in games:
            try:
                cursor.execute(
                    "SELECT 1 FROM synced_games WHERE game_id = ?",
                    (game.game_id,),
                )
                existed = cursor.fetchone() is not None

                cursor.execute("""
                    INSERT OR IGNORE INTO synced_games
                    (game_id, source_host, source_db, synced_at, board_type, num_players,
                     avg_player_elo, min_player_elo, max_player_elo, elo_difference,
                     game_length, is_decisive, termination_reason, quality_score,
                     model_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game.game_id, source_host, source_db, now, board_type, num_players,
                    game.avg_player_elo, game.min_player_elo, game.max_player_elo,
                    game.elo_difference, game.game_length, 1 if game.is_decisive else 0,
                    game.termination_reason, game.quality_score, game.model_version,
                    game.created_at,
                ))

                cursor.execute("""
                    UPDATE synced_games SET
                        source_host = ?,
                        source_db = ?,
                        synced_at = ?,
                        board_type = ?,
                        num_players = ?,
                        avg_player_elo = ?,
                        min_player_elo = ?,
                        max_player_elo = ?,
                        elo_difference = ?,
                        game_length = ?,
                        is_decisive = ?,
                        termination_reason = ?,
                        quality_score = ?,
                        model_version = ?,
                        created_at = ?
                    WHERE game_id = ?
                """, (
                    source_host,
                    source_db,
                    now,
                    board_type,
                    num_players,
                    game.avg_player_elo,
                    game.min_player_elo,
                    game.max_player_elo,
                    game.elo_difference,
                    game.game_length,
                    1 if game.is_decisive else 0,
                    game.termination_reason,
                    game.quality_score,
                    game.model_version,
                    game.created_at,
                    game.game_id,
                ))

                if not existed:
                    marked += 1
            except sqlite3.Error as e:
                logger.warning(f"Failed to mark game {game.game_id} with quality: {e}")

        conn.commit()
        conn.close()

        # Emit quality events after batch sync (December 2025)
        if marked > 0:
            config = f"{board_type}_{num_players}p" if board_type and num_players else ""
            self._emit_quality_events(config, board_type, num_players)

        return marked

    def update_game_quality(self, game_id: str, quality: GameQualityMetadata) -> bool:
        """Update quality metadata for an existing synced game.

        Args:
            game_id: Game ID to update
            quality: Quality metadata to apply

        Returns:
            True if update succeeded
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                UPDATE synced_games SET
                    avg_player_elo = ?,
                    min_player_elo = ?,
                    max_player_elo = ?,
                    elo_difference = ?,
                    game_length = ?,
                    is_decisive = ?,
                    termination_reason = ?,
                    quality_score = ?,
                    model_version = ?
                WHERE game_id = ?
            """, (
                quality.avg_player_elo, quality.min_player_elo, quality.max_player_elo,
                quality.elo_difference, quality.game_length, 1 if quality.is_decisive else 0,
                quality.termination_reason, quality.quality_score, quality.model_version,
                game_id
            ))
            conn.commit()
            success = cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.warning(f"Failed to update game quality for {game_id}: {e}")
            success = False
        finally:
            conn.close()
        return success

    def get_high_quality_games(
        self,
        min_quality_score: float = 0.5,
        limit: int = 1000,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ) -> List[GameQualityMetadata]:
        """Get high-quality games for training.

        Args:
            min_quality_score: Minimum quality score threshold
            limit: Maximum number of games to return
            board_type: Optional filter by board type
            num_players: Optional filter by player count

        Returns:
            List of GameQualityMetadata sorted by quality score descending
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT game_id, avg_player_elo, min_player_elo, max_player_elo,
                   elo_difference, game_length, is_decisive, termination_reason,
                   quality_score, model_version, created_at
            FROM synced_games
            WHERE quality_score >= ?
        """
        params: List[Any] = [min_quality_score]

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)

        query += " ORDER BY quality_score DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        games = []
        for row in cursor.fetchall():
            games.append(GameQualityMetadata(
                game_id=row[0],
                avg_player_elo=row[1] or 1500.0,
                min_player_elo=row[2] or 1500.0,
                max_player_elo=row[3] or 1500.0,
                elo_difference=row[4] or 0.0,
                game_length=row[5] or 0,
                is_decisive=bool(row[6]),
                termination_reason=row[7] or "",
                quality_score=row[8] or 0.0,
                model_version=row[9] or "",
                created_at=row[10] or 0.0,
            ))
        conn.close()
        return games

    def get_quality_distribution(
        self,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get distribution statistics for quality scores.

        Args:
            board_type: Optional filter by board type
            num_players: Optional filter by player count

        Returns:
            Dict with quality distribution statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        where_clause = "WHERE quality_score IS NOT NULL"
        params: List[Any] = []

        if board_type:
            where_clause += " AND board_type = ?"
            params.append(board_type)
        if num_players:
            where_clause += " AND num_players = ?"
            params.append(num_players)

        cursor.execute(f"""
            SELECT
                COUNT(*) as total,
                AVG(quality_score) as avg_quality,
                MIN(quality_score) as min_quality,
                MAX(quality_score) as max_quality,
                AVG(avg_player_elo) as avg_elo,
                AVG(game_length) as avg_length,
                SUM(CASE WHEN is_decisive = 1 THEN 1 ELSE 0 END) as decisive_count
            FROM synced_games {where_clause}
        """, params)

        row = cursor.fetchone()
        conn.close()

        return {
            "total_games": row[0] or 0,
            "avg_quality_score": row[1] or 0.0,
            "min_quality_score": row[2] or 0.0,
            "max_quality_score": row[3] or 0.0,
            "avg_player_elo": row[4] or 1500.0,
            "avg_game_length": row[5] or 0.0,
            "decisive_count": row[6] or 0,
            "decisive_rate": (row[6] or 0) / max(1, row[0] or 1),
        }

    def _emit_quality_events(
        self,
        config: str,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
        high_quality_threshold: float = 0.7,
        low_quality_threshold: float = 0.3,
    ) -> None:
        """Check quality distribution and emit events if significant.

        Args:
            config: Configuration key for events
            board_type: Board type filter
            num_players: Player count filter
            high_quality_threshold: Threshold for high-quality classification
            low_quality_threshold: Threshold for low-quality classification
        """
        if not HAS_EVENT_HELPERS or not has_event_bus():
            return

        try:
            dist = self.get_quality_distribution(board_type, num_players)
            total = dist.get("total_games", 0)
            if total == 0:
                return

            avg_quality = dist.get("avg_quality_score", 0.0)

            # Get detailed counts
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            where_clause = "WHERE quality_score IS NOT NULL"
            params: List[Any] = []
            if board_type:
                where_clause += " AND board_type = ?"
                params.append(board_type)
            if num_players:
                where_clause += " AND num_players = ?"
                params.append(num_players)

            cursor.execute(f"""
                SELECT
                    SUM(CASE WHEN quality_score >= ? THEN 1 ELSE 0 END) as high_count,
                    SUM(CASE WHEN quality_score < ? THEN 1 ELSE 0 END) as low_count
                FROM synced_games {where_clause}
            """, [high_quality_threshold, low_quality_threshold] + params)

            row = cursor.fetchone()
            conn.close()

            high_count = row[0] or 0
            low_count = row[1] or 0
            high_ratio = high_count / total if total > 0 else 0
            low_ratio = low_count / total if total > 0 else 0

            # Emit quality distribution event
            emit_sync(
                "QUALITY_DISTRIBUTION_CHANGED",
                {
                    "config": config,
                    "avg_quality": avg_quality,
                    "high_quality_ratio": high_ratio,
                    "low_quality_ratio": low_ratio,
                    "total_games": total,
                },
                "DataManifest"
            )

            # Emit specific events based on thresholds
            if high_count >= 100 and avg_quality >= high_quality_threshold:
                emit_sync(
                    "HIGH_QUALITY_DATA_AVAILABLE",
                    {
                        "config": config,
                        "high_quality_count": high_count,
                        "avg_quality": avg_quality,
                    },
                    "DataManifest"
                )

            if low_ratio > 0.3:  # More than 30% low quality
                emit_sync(
                    "LOW_QUALITY_DATA_WARNING",
                    {
                        "config": config,
                        "low_quality_count": low_count,
                        "low_quality_ratio": low_ratio,
                        "avg_quality": avg_quality,
                    },
                    "DataManifest"
                )

        except Exception as e:
            logger.debug(f"Failed to emit quality events: {e}")

    # =========================================================================
    # Priority Sync Queue
    # =========================================================================

    def add_to_priority_queue(
        self,
        game_id: str,
        source_host: str,
        source_db: str,
        priority_score: float,
        avg_player_elo: Optional[float] = None,
        game_length: Optional[int] = None,
        is_decisive: bool = False,
    ) -> int:
        """Add a game to the priority sync queue.

        Args:
            game_id: Game ID to queue
            source_host: Source host for sync
            source_db: Source database path
            priority_score: Computed priority score (higher = sync first)
            avg_player_elo: Average player Elo
            game_length: Game length in moves
            is_decisive: Whether the game had a decisive outcome

        Returns:
            Queue entry ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()

        cursor.execute("""
            INSERT OR REPLACE INTO sync_priority_queue
            (game_id, source_host, source_db, priority_score, avg_player_elo,
             game_length, is_decisive, queued_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id, source_host, source_db, priority_score,
            avg_player_elo, game_length, 1 if is_decisive else 0, now
        ))
        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return entry_id

    def get_priority_queue_batch(
        self,
        limit: int = 100,
        source_host: Optional[str] = None,
        min_priority: Optional[float] = None,
    ) -> List[PriorityQueueEntry]:
        """Get a batch of high-priority games to sync.

        Args:
            limit: Maximum number of entries to return
            source_host: Optional filter by source host
            min_priority: Optional minimum priority score threshold

        Returns:
            List of PriorityQueueEntry sorted by priority descending
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT id, game_id, source_host, source_db, priority_score,
                   avg_player_elo, game_length, is_decisive, queued_at, synced, synced_at
            FROM sync_priority_queue
            WHERE synced = 0
        """
        params: List[Any] = []

        if source_host:
            query += " AND source_host = ?"
            params.append(source_host)

        if min_priority is not None:
            query += " AND priority_score >= ?"
            params.append(min_priority)

        query += " ORDER BY priority_score DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        entries = []
        for row in cursor.fetchall():
            entries.append(PriorityQueueEntry(
                id=row[0],
                game_id=row[1],
                source_host=row[2],
                source_db=row[3],
                priority_score=row[4],
                avg_player_elo=row[5],
                game_length=row[6],
                is_decisive=bool(row[7]),
                queued_at=row[8],
                synced=bool(row[9]),
                synced_at=row[10],
            ))
        conn.close()
        return entries

    def mark_queue_entries_synced(self, entry_ids: List[int]) -> int:
        """Mark priority queue entries as synced.

        Args:
            entry_ids: List of queue entry IDs

        Returns:
            Number of entries marked
        """
        if not entry_ids:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()

        placeholders = ",".join("?" * len(entry_ids))
        cursor.execute(f"""
            UPDATE sync_priority_queue
            SET synced = 1, synced_at = ?
            WHERE id IN ({placeholders})
        """, [now] + entry_ids)

        marked = cursor.rowcount
        conn.commit()
        conn.close()
        return marked

    def get_priority_queue_stats(self) -> Dict[str, Any]:
        """Get priority queue statistics.

        Returns:
            Dict with queue statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN synced = 0 THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN synced = 1 THEN 1 ELSE 0 END) as synced,
                AVG(priority_score) as avg_priority,
                MAX(priority_score) as max_priority
            FROM sync_priority_queue
        """)
        row = cursor.fetchone()

        # Get by-host breakdown
        cursor.execute("""
            SELECT source_host, COUNT(*), SUM(CASE WHEN synced = 0 THEN 1 ELSE 0 END)
            FROM sync_priority_queue
            GROUP BY source_host
        """)
        by_host = {r[0]: {"total": r[1], "pending": r[2]} for r in cursor.fetchall()}

        conn.close()

        return {
            "total_entries": row[0] or 0,
            "pending": row[1] or 0,
            "synced": row[2] or 0,
            "avg_priority_score": row[3] or 0.0,
            "max_priority_score": row[4] or 0.0,
            "by_host": by_host,
        }

    def cleanup_old_queue_entries(self, days: int = 7) -> int:
        """Remove old synced entries from the priority queue.

        Args:
            days: Remove synced entries older than this

        Returns:
            Number of entries removed
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff = time.time() - (days * 86400)

        cursor.execute("""
            DELETE FROM sync_priority_queue
            WHERE synced = 1 AND synced_at < ?
        """, (cutoff,))

        removed = cursor.rowcount
        conn.commit()
        conn.close()
        return removed

    def get_synced_count(self) -> int:
        """Get total number of synced games."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM synced_games")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_synced_count_by_host(self) -> Dict[str, int]:
        """Get synced game counts by host."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT source_host, COUNT(*) FROM synced_games
            GROUP BY source_host
        """)
        result = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return result

    def get_synced_count_by_config(self) -> Dict[str, int]:
        """Get synced game counts by board config."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT board_type, num_players, COUNT(*) FROM synced_games
            WHERE board_type IS NOT NULL
            GROUP BY board_type, num_players
        """)
        result = {}
        for row in cursor.fetchall():
            key = f"{row[0]}_{row[1]}p" if row[0] and row[1] else "unknown"
            result[key] = row[2]
        conn.close()
        return result

    # =========================================================================
    # Host Sync State
    # =========================================================================

    def save_host_state(self, state: HostSyncState) -> None:
        """Save host sync state.

        Args:
            state: Host sync state to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO host_states
            (host_name, last_sync_time, last_game_count, total_games_synced,
             consecutive_failures, last_error, last_error_time,
             is_ephemeral, storage_type, poll_interval)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.name, state.last_sync_time, state.last_game_count,
            state.total_games_synced, state.consecutive_failures,
            state.last_error, state.last_error_time,
            1 if state.is_ephemeral else 0, state.storage_type, state.poll_interval
        ))
        conn.commit()
        conn.close()

    def load_host_state(self, host_name: str) -> Optional[HostSyncState]:
        """Load host sync state.

        Args:
            host_name: Host name to load

        Returns:
            HostSyncState if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT host_name, last_sync_time, last_game_count, total_games_synced,
                   consecutive_failures, last_error, last_error_time,
                   is_ephemeral, storage_type, poll_interval
            FROM host_states WHERE host_name = ?
        """, (host_name,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return HostSyncState(
                name=row[0],
                last_sync_time=row[1] or 0.0,
                last_game_count=row[2] or 0,
                total_games_synced=row[3] or 0,
                consecutive_failures=row[4] or 0,
                last_error=row[5] or "",
                last_error_time=row[6] or 0.0,
                is_ephemeral=bool(row[7]) if len(row) > 7 else False,
                storage_type=row[8] if len(row) > 8 else "persistent",
                poll_interval=row[9] if len(row) > 9 else 60,
            )
        return None

    def load_all_host_states(self) -> List[HostSyncState]:
        """Load all host sync states."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT host_name, last_sync_time, last_game_count, total_games_synced,
                   consecutive_failures, last_error, last_error_time,
                   is_ephemeral, storage_type, poll_interval
            FROM host_states
        """)
        states = []
        for row in cursor.fetchall():
            states.append(HostSyncState(
                name=row[0],
                last_sync_time=row[1] or 0.0,
                last_game_count=row[2] or 0,
                total_games_synced=row[3] or 0,
                consecutive_failures=row[4] or 0,
                last_error=row[5] or "",
                last_error_time=row[6] or 0.0,
                is_ephemeral=bool(row[7]) if len(row) > 7 else False,
                storage_type=row[8] if len(row) > 8 else "persistent",
                poll_interval=row[9] if len(row) > 9 else 60,
            ))
        conn.close()
        return states

    def get_ephemeral_hosts(self) -> List[str]:
        """Get list of ephemeral host names."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT host_name FROM host_states WHERE is_ephemeral = 1")
        hosts = [row[0] for row in cursor.fetchall()]
        conn.close()
        return hosts

    # =========================================================================
    # Sync History
    # =========================================================================

    def log_sync(
        self,
        host_name: str,
        games_synced: int,
        duration_seconds: float,
        success: bool,
        sync_method: str = "",
        error_message: str = "",
    ) -> int:
        """Log a sync operation.

        Args:
            host_name: Host that was synced
            games_synced: Number of games synced
            duration_seconds: Time taken
            success: Whether sync succeeded
            sync_method: Method used (rsync, scp, etc.)
            error_message: Error message if failed

        Returns:
            ID of the log entry
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sync_history
            (host_name, sync_time, games_synced, duration_seconds, success, sync_method, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (host_name, time.time(), games_synced, duration_seconds, 1 if success else 0, sync_method, error_message))
        log_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return log_id

    # Alias for backwards compatibility with tests
    def record_sync(
        self,
        host_name: str,
        games_synced: int,
        duration_seconds: float,
        success: bool,
        sync_method: str = "",
        error_message: str = "",
    ) -> int:
        """Alias for log_sync (backwards compatibility)."""
        return self.log_sync(
            host_name=host_name,
            games_synced=games_synced,
            duration_seconds=duration_seconds,
            success=success,
            sync_method=sync_method,
            error_message=error_message,
        )

    def get_recent_syncs(self, hours: int = 24, host_name: Optional[str] = None) -> List[SyncHistoryEntry]:
        """Get recent sync history.

        Args:
            hours: Lookback hours
            host_name: Optional filter by host

        Returns:
            List of sync history entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff = time.time() - (hours * 3600)

        if host_name:
            cursor.execute("""
                SELECT id, host_name, sync_time, games_synced, duration_seconds, success, sync_method, error_message
                FROM sync_history
                WHERE sync_time > ? AND host_name = ?
                ORDER BY sync_time DESC
            """, (cutoff, host_name))
        else:
            cursor.execute("""
                SELECT id, host_name, sync_time, games_synced, duration_seconds, success, sync_method, error_message
                FROM sync_history
                WHERE sync_time > ?
                ORDER BY sync_time DESC
            """, (cutoff,))

        entries = []
        for row in cursor.fetchall():
            entries.append(SyncHistoryEntry(
                id=row[0],
                host_name=row[1],
                sync_time=row[2],
                games_synced=row[3],
                duration_seconds=row[4] or 0.0,
                success=bool(row[5]),
                sync_method=row[6] or "",
                error_message=row[7] or "",
            ))
        conn.close()
        return entries

    # =========================================================================
    # Dead Letter Queue
    # =========================================================================

    def add_to_dead_letter(
        self,
        game_id: str,
        source_host: str,
        source_db: str,
        error_message: str,
        error_type: str = "sync_error",
    ) -> int:
        """Add a failed sync to dead letter queue.

        Args:
            game_id: Game ID that failed
            source_host: Source host
            source_db: Source database
            error_message: Error description
            error_type: Error category

        Returns:
            ID of dead letter entry
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dead_letter_queue
            (game_id, source_host, source_db, error_message, error_type, added_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (game_id, source_host, source_db, error_message, error_type, time.time()))
        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return entry_id

    def get_dead_letter_entries(self, limit: int = 100, include_resolved: bool = False) -> List[DeadLetterEntry]:
        """Get dead letter queue entries.

        Args:
            limit: Maximum entries to return
            include_resolved: Include resolved entries

        Returns:
            List of dead letter entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if include_resolved:
            cursor.execute("""
                SELECT id, game_id, source_host, source_db, error_message, error_type,
                       added_at, retry_count, last_retry_at, resolved
                FROM dead_letter_queue
                ORDER BY added_at DESC
                LIMIT ?
            """, (limit,))
        else:
            cursor.execute("""
                SELECT id, game_id, source_host, source_db, error_message, error_type,
                       added_at, retry_count, last_retry_at, resolved
                FROM dead_letter_queue
                WHERE resolved = 0
                ORDER BY added_at DESC
                LIMIT ?
            """, (limit,))

        entries = []
        for row in cursor.fetchall():
            entries.append(DeadLetterEntry(
                id=row[0],
                game_id=row[1],
                source_host=row[2],
                source_db=row[3],
                error_message=row[4],
                error_type=row[5],
                added_at=row[6],
                retry_count=row[7],
                last_retry_at=row[8],
                resolved=bool(row[9]),
            ))
        conn.close()
        return entries

    def mark_dead_letter_resolved(self, entry_ids: List[int]) -> int:
        """Mark dead letter entries as resolved.

        Args:
            entry_ids: IDs to mark resolved

        Returns:
            Number of entries marked
        """
        if not entry_ids:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(entry_ids))
        cursor.execute(f"""
            UPDATE dead_letter_queue SET resolved = 1
            WHERE id IN ({placeholders})
        """, entry_ids)
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        return updated

    def increment_dead_letter_retry(self, entry_id: int) -> None:
        """Increment retry count for a dead letter entry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE dead_letter_queue
            SET retry_count = retry_count + 1, last_retry_at = ?
            WHERE id = ?
        """, (time.time(), entry_id))
        conn.commit()
        conn.close()

    def cleanup_old_dead_letters(self, days: int = 7) -> int:
        """Remove old resolved dead letter entries.

        Args:
            days: Remove entries older than this

        Returns:
            Number of entries removed
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff = time.time() - (days * 86400)
        cursor.execute("""
            DELETE FROM dead_letter_queue
            WHERE resolved = 1 AND added_at < ?
        """, (cutoff,))
        removed = cursor.rowcount
        conn.commit()
        conn.close()
        return removed

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> ManifestStats:
        """Get manifest statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total games
        cursor.execute("SELECT COUNT(*) FROM synced_games")
        total = cursor.fetchone()[0]

        # By host
        cursor.execute("SELECT source_host, COUNT(*) FROM synced_games GROUP BY source_host")
        by_host = {row[0]: row[1] for row in cursor.fetchall()}

        # By board type
        cursor.execute("""
            SELECT board_type || '_' || num_players || 'p', COUNT(*)
            FROM synced_games
            WHERE board_type IS NOT NULL
            GROUP BY board_type, num_players
        """)
        by_config = {row[0]: row[1] for row in cursor.fetchall()}

        # Recent syncs (24h)
        cutoff = time.time() - 86400
        cursor.execute("SELECT COUNT(*) FROM synced_games WHERE synced_at > ?", (cutoff,))
        recent = cursor.fetchone()[0]

        # Dead letter count
        cursor.execute("SELECT COUNT(*) FROM dead_letter_queue WHERE resolved = 0")
        dead_letters = cursor.fetchone()[0]

        conn.close()

        return ManifestStats(
            total_games=total,
            games_by_host=by_host,
            games_by_board_type=by_config,
            recent_sync_count=recent,
            dead_letter_count=dead_letters,
        )

    def cleanup_old_history(self, days: int = 30) -> int:
        """Clean up old sync history.

        Args:
            days: Remove entries older than this

        Returns:
            Number of entries removed
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff = time.time() - (days * 86400)
        cursor.execute("DELETE FROM sync_history WHERE sync_time < ?", (cutoff,))
        removed = cursor.rowcount
        conn.commit()
        conn.close()
        return removed


# =============================================================================
# Module-level utilities
# =============================================================================


def create_manifest(data_dir: Path) -> DataManifest:
    """Factory function to create a DataManifest.

    Args:
        data_dir: Base data directory

    Returns:
        Configured DataManifest instance
    """
    return DataManifest(data_dir / "data_manifest.db")


# Backward compatibility exports
__all__ = [
    "DataManifest",
    "HostSyncState",
    "SyncHistoryEntry",
    "DeadLetterEntry",
    "ManifestStats",
    "GameQualityMetadata",
    "PriorityQueueEntry",
    "create_manifest",
]
