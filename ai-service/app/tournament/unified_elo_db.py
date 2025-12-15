"""Unified Elo database for all tournament types.

Provides a single, shared Elo rating system across all tournament scripts.
Uses composite primary key (participant_id, board_type, num_players) to track
ratings per game configuration.

Usage:
    from app.tournament import get_elo_database, EloDatabase

    # Get singleton database instance
    db = get_elo_database()

    # Record a match and update ratings
    db.record_match_and_update(
        participant_ids=["model_a", "model_b"],
        rankings=[0, 1],  # model_a won (rank 0 = 1st place)
        board_type="square8",
        num_players=2,
        tournament_id="my_tournament_123",
    )

    # Query leaderboard
    leaders = db.get_leaderboard(board_type="square8", num_players=2)
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .elo import EloCalculator

logger = logging.getLogger(__name__)

# Database location - canonical Elo database for all trained models
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "unified_elo.db"

# Global singleton
_elo_db_instance: Optional["EloDatabase"] = None
_elo_db_lock = threading.Lock()


@dataclass
class UnifiedEloRating:
    """Elo rating for a participant in a specific game configuration."""

    participant_id: str
    board_type: str
    num_players: int
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rating_deviation: float = 350.0  # Initial RD (high uncertainty)
    last_update: Optional[float] = None

    # Glicko-style constants
    INITIAL_RD: float = 350.0
    MIN_RD: float = 50.0
    RD_DECAY_GAMES: int = 100

    @property
    def calculated_rd(self) -> float:
        """Calculate rating deviation based on games played."""
        import math

        if self.games_played == 0:
            return self.INITIAL_RD
        decay_factor = math.exp(-self.games_played / self.RD_DECAY_GAMES)
        return self.MIN_RD + (self.INITIAL_RD - self.MIN_RD) * decay_factor

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "rating": round(self.rating, 1),
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": round(self.win_rate, 3),
            "rating_deviation": round(self.calculated_rd, 1),
        }


@dataclass
class MatchRecord:
    """Record of a completed match."""

    match_id: str
    participant_ids: List[str]
    rankings: List[int]  # Position in final standings (0=1st, 1=2nd, etc.)
    winner_id: Optional[str]
    board_type: str
    num_players: int
    game_length: int
    duration_sec: float
    timestamp: str
    tournament_id: str
    worker: Optional[str] = None


class EloDatabase:
    """SQLite database for unified Elo tracking across all tournaments.

    Schema uses composite primary key (participant_id, board_type, num_players)
    to track separate ratings for each game configuration.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
        logger.info(f"EloDatabase initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @property
    def id_column(self) -> str:
        """Get the correct ID column name for elo_ratings table.

        Returns 'model_id' for elo_leaderboard.db, 'participant_id' for unified_elo.db.
        """
        return "model_id" if getattr(self, "_uses_model_id_schema", False) else "participant_id"

    def _init_db(self):
        """Initialize database schema.

        This schema is backwards-compatible with existing unified_elo.db while
        supporting new features. The schema will be migrated if needed.

        Also supports elo_leaderboard.db which uses model_id instead of participant_id.
        """
        conn = self._get_connection()

        # Check if we need to migrate an existing database
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='participants'")
        participants_exists = cursor.fetchone() is not None

        # Check if elo_ratings exists with different schema (model_id vs participant_id)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='elo_ratings'")
        elo_ratings_exists = cursor.fetchone() is not None
        self._uses_model_id_schema = False

        if elo_ratings_exists:
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            elo_columns = {row[1] for row in cursor.fetchall()}
            # elo_leaderboard.db uses model_id, unified_elo.db uses participant_id
            self._uses_model_id_schema = "model_id" in elo_columns and "participant_id" not in elo_columns
            if self._uses_model_id_schema:
                # Don't modify elo_leaderboard.db schema - it's a different format
                self._old_participant_schema = False
                self._old_match_schema = False
                return  # Skip schema creation for model_id databases

        if participants_exists:
            # Check if it's old schema (participant_id as PK) vs new (id as PK)
            cursor = conn.execute("PRAGMA table_info(participants)")
            columns = {row[1] for row in cursor.fetchall()}
            self._old_participant_schema = "participant_id" in columns and "id" not in columns

            # Check match_history schema
            cursor = conn.execute("PRAGMA table_info(match_history)")
            match_columns = {row[1] for row in cursor.fetchall()}
            self._old_match_schema = "participant_a" in match_columns
        else:
            self._old_participant_schema = False
            self._old_match_schema = False

        # Create tables with backwards-compatible schema
        conn.executescript("""
            -- Participants table: all known participants (models, AI types, etc.)
            -- Uses participant_id as PK for backwards compatibility
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                participant_type TEXT NOT NULL DEFAULT 'model',
                ai_type TEXT,
                difficulty INTEGER,
                use_neural_net INTEGER,
                model_path TEXT,
                model_version TEXT,
                metadata TEXT,
                created_at REAL,
                last_seen REAL
            );

            -- Elo ratings: per-configuration ratings
            CREATE TABLE IF NOT EXISTS elo_ratings (
                participant_id TEXT NOT NULL,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                rating REAL DEFAULT 1500.0,
                games_played INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                rating_deviation REAL DEFAULT 350.0,
                last_update REAL,
                PRIMARY KEY (participant_id, board_type, num_players)
            );

            -- Match history: supports both old and new formats
            -- Old format: participant_a, participant_b columns
            -- New format: participant_ids JSON array, rankings JSON array
            CREATE TABLE IF NOT EXISTS match_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_a TEXT,
                participant_b TEXT,
                participant_ids TEXT,   -- JSON array (new format)
                rankings TEXT,          -- JSON array of final positions (new format)
                winner TEXT,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                game_length INTEGER,
                duration_sec REAL,
                timestamp REAL NOT NULL,
                tournament_id TEXT,
                game_id TEXT,
                worker TEXT,
                metadata TEXT
            );

            -- Rating history: track Elo changes over time
            CREATE TABLE IF NOT EXISTS rating_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id TEXT NOT NULL,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                rating REAL NOT NULL,
                rating_change REAL,
                games_played INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                match_id TEXT,
                tournament_id TEXT
            );

            -- Indices for common queries
            CREATE INDEX IF NOT EXISTS idx_elo_config
            ON elo_ratings(board_type, num_players, rating DESC);

            CREATE INDEX IF NOT EXISTS idx_elo_participant
            ON elo_ratings(participant_id);

            CREATE INDEX IF NOT EXISTS idx_match_timestamp
            ON match_history(timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_match_tournament
            ON match_history(tournament_id);

            CREATE INDEX IF NOT EXISTS idx_rating_history_participant
            ON rating_history(participant_id, board_type, num_players, timestamp DESC);
        """)
        conn.commit()

        # Add missing columns if upgrading schema
        self._upgrade_schema_if_needed(conn)

    def _upgrade_schema_if_needed(self, conn: sqlite3.Connection):
        """Add missing columns to existing tables for schema upgrades."""
        # Get current columns for each table
        def get_columns(table: str) -> set:
            cursor = conn.execute(f"PRAGMA table_info({table})")
            return {row[1] for row in cursor.fetchall()}

        # Upgrade participants table
        participant_cols = get_columns("participants")
        if "ai_type" not in participant_cols:
            conn.execute("ALTER TABLE participants ADD COLUMN ai_type TEXT")
        if "difficulty" not in participant_cols:
            conn.execute("ALTER TABLE participants ADD COLUMN difficulty INTEGER")
        if "use_neural_net" not in participant_cols:
            conn.execute("ALTER TABLE participants ADD COLUMN use_neural_net INTEGER")
        if "model_version" not in participant_cols:
            conn.execute("ALTER TABLE participants ADD COLUMN model_version TEXT")

        # Upgrade match_history table
        match_cols = get_columns("match_history")
        if "participant_ids" not in match_cols:
            conn.execute("ALTER TABLE match_history ADD COLUMN participant_ids TEXT")
        if "rankings" not in match_cols:
            conn.execute("ALTER TABLE match_history ADD COLUMN rankings TEXT")
        if "worker" not in match_cols:
            conn.execute("ALTER TABLE match_history ADD COLUMN worker TEXT")

        # Upgrade rating_history table
        rating_cols = get_columns("rating_history")
        if "rating_change" not in rating_cols:
            conn.execute("ALTER TABLE rating_history ADD COLUMN rating_change REAL")
        if "match_id" not in rating_cols:
            conn.execute("ALTER TABLE rating_history ADD COLUMN match_id TEXT")

        conn.commit()

    def close(self):
        """Close thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # =========================================================================
    # Participant Management
    # =========================================================================

    def register_participant(
        self,
        participant_id: str,
        name: Optional[str] = None,
        participant_type: str = "model",
        ai_type: Optional[str] = None,
        difficulty: Optional[int] = None,
        use_neural_net: bool = False,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        model_version: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Register a participant (model, baseline, or AI type)."""
        conn = self._get_connection()
        now = time.time()
        conn.execute("""
            INSERT INTO participants
            (participant_id, participant_type, ai_type, difficulty, use_neural_net,
             model_path, model_version, metadata, created_at, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(participant_id) DO UPDATE SET
                participant_type = COALESCE(excluded.participant_type, participant_type),
                ai_type = COALESCE(excluded.ai_type, ai_type),
                difficulty = COALESCE(excluded.difficulty, difficulty),
                use_neural_net = COALESCE(excluded.use_neural_net, use_neural_net),
                model_path = COALESCE(excluded.model_path, model_path),
                model_version = COALESCE(excluded.model_version, model_version),
                metadata = COALESCE(excluded.metadata, metadata),
                last_seen = excluded.last_seen
        """, (
            participant_id,
            participant_type,
            ai_type,
            difficulty,
            int(use_neural_net) if use_neural_net else None,
            model_path,
            model_version,
            json.dumps(metadata) if metadata else None,
            now,
            now,
        ))
        conn.commit()

    def get_participant(self, participant_id: str) -> Optional[Dict[str, Any]]:
        """Get participant info."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM participants WHERE participant_id = ?",
            (participant_id,)
        ).fetchone()
        if row:
            result = dict(row)
            if result.get("metadata"):
                result["metadata"] = json.loads(result["metadata"])
            return result
        return None

    def ensure_participant(self, participant_id: str, **kwargs) -> None:
        """Ensure participant exists, creating with defaults if not."""
        if not self.get_participant(participant_id):
            self.register_participant(participant_id, **kwargs)

    # =========================================================================
    # Rating Management
    # =========================================================================

    def get_rating(
        self,
        participant_id: str,
        board_type: str,
        num_players: int,
    ) -> UnifiedEloRating:
        """Get rating for a participant in a specific configuration."""
        conn = self._get_connection()
        row = conn.execute("""
            SELECT rating, games_played, wins, losses, draws, rating_deviation, last_update
            FROM elo_ratings
            WHERE participant_id = ? AND board_type = ? AND num_players = ?
        """, (participant_id, board_type, num_players)).fetchone()

        if row:
            return UnifiedEloRating(
                participant_id=participant_id,
                board_type=board_type,
                num_players=num_players,
                rating=row["rating"],
                games_played=row["games_played"],
                wins=row["wins"],
                losses=row["losses"],
                draws=row["draws"],
                rating_deviation=row["rating_deviation"] or 350.0,
                last_update=row["last_update"],
            )
        return UnifiedEloRating(participant_id, board_type, num_players)

    def get_ratings_batch(
        self,
        participant_ids: List[str],
        board_type: str,
        num_players: int,
    ) -> Dict[str, UnifiedEloRating]:
        """Get ratings for multiple participants at once."""
        return {
            pid: self.get_rating(pid, board_type, num_players)
            for pid in participant_ids
        }

    def update_rating(self, rating: UnifiedEloRating) -> None:
        """Update a single rating."""
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO elo_ratings
            (participant_id, board_type, num_players, rating, games_played,
             wins, losses, draws, rating_deviation, last_update)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(participant_id, board_type, num_players) DO UPDATE SET
                rating = excluded.rating,
                games_played = excluded.games_played,
                wins = excluded.wins,
                losses = excluded.losses,
                draws = excluded.draws,
                rating_deviation = excluded.rating_deviation,
                last_update = excluded.last_update
        """, (
            rating.participant_id,
            rating.board_type,
            rating.num_players,
            rating.rating,
            rating.games_played,
            rating.wins,
            rating.losses,
            rating.draws,
            rating.calculated_rd,
            time.time(),
        ))
        conn.commit()

    def update_ratings_batch(self, ratings: List[UnifiedEloRating]) -> None:
        """Update multiple ratings in a single transaction."""
        conn = self._get_connection()
        now = time.time()
        for r in ratings:
            conn.execute("""
                INSERT INTO elo_ratings
                (participant_id, board_type, num_players, rating, games_played,
                 wins, losses, draws, rating_deviation, last_update)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(participant_id, board_type, num_players) DO UPDATE SET
                    rating = excluded.rating,
                    games_played = excluded.games_played,
                    wins = excluded.wins,
                    losses = excluded.losses,
                    draws = excluded.draws,
                    rating_deviation = excluded.rating_deviation,
                    last_update = excluded.last_update
            """, (
                r.participant_id, r.board_type, r.num_players, r.rating,
                r.games_played, r.wins, r.losses, r.draws, r.calculated_rd, now,
            ))
        conn.commit()

    # =========================================================================
    # Match Recording and Elo Updates
    # =========================================================================

    def record_match(
        self,
        participant_ids: List[str],
        rankings: List[int],
        board_type: str,
        num_players: int,
        tournament_id: str,
        game_length: int = 0,
        duration_sec: float = 0.0,
        worker: Optional[str] = None,
        metadata: Optional[Dict] = None,
        game_id: Optional[str] = None,
    ) -> int:
        """Record a match without updating Elo ratings.

        Args:
            participant_ids: List of participant IDs in the match
            rankings: Final positions (0=1st, 1=2nd, etc.) for each participant
            board_type: Board type (square8, square19, hexagonal)
            num_players: Number of players (2, 3, or 4)
            tournament_id: ID of the tournament
            game_length: Number of moves in the game
            duration_sec: Duration of the game in seconds
            worker: Worker that ran the game (optional)
            metadata: Additional match metadata (optional)
            game_id: Optional game UUID for tracking

        Returns:
            Generated match ID (integer)
        """
        timestamp = time.time()

        # Determine winner (participant with ranking 0)
        winner_id = None
        for pid, rank in zip(participant_ids, rankings):
            if rank == 0:
                winner_id = pid
                break

        # For backwards compatibility, also populate participant_a/participant_b
        participant_a = participant_ids[0] if len(participant_ids) > 0 else None
        participant_b = participant_ids[1] if len(participant_ids) > 1 else None

        conn = self._get_connection()
        cursor = conn.execute("""
            INSERT INTO match_history
            (participant_a, participant_b, participant_ids, rankings, winner,
             board_type, num_players, game_length, duration_sec, timestamp,
             tournament_id, game_id, worker, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            participant_a,
            participant_b,
            json.dumps(participant_ids),
            json.dumps(rankings),
            winner_id,
            board_type,
            num_players,
            game_length,
            duration_sec,
            timestamp,
            tournament_id,
            game_id,
            worker,
            json.dumps(metadata) if metadata else None,
        ))
        match_id = cursor.lastrowid
        conn.commit()
        return match_id

    def record_match_and_update(
        self,
        participant_ids: List[str],
        rankings: List[int],
        board_type: str,
        num_players: int,
        tournament_id: str,
        game_length: int = 0,
        duration_sec: float = 0.0,
        worker: Optional[str] = None,
        metadata: Optional[Dict] = None,
        k_factor: float = 32.0,
        game_id: Optional[str] = None,
    ) -> Tuple[int, Dict[str, float]]:
        """Record a match and update Elo ratings.

        Args:
            participant_ids: List of participant IDs in the match
            rankings: Final positions (0=1st, 1=2nd, etc.) for each participant
            board_type: Board type (square8, square19, hexagonal)
            num_players: Number of players (2, 3, or 4)
            tournament_id: ID of the tournament
            game_length: Number of moves in the game
            duration_sec: Duration of the game in seconds
            worker: Worker that ran the game (optional)
            metadata: Additional match metadata (optional)
            k_factor: K-factor for Elo calculation
            game_id: Optional game UUID for tracking

        Returns:
            Tuple of (match_id, dict of participant_id -> new_rating)
        """
        # Ensure all participants are registered
        for pid in participant_ids:
            self.ensure_participant(pid)

        # Record the match
        match_id = self.record_match(
            participant_ids=participant_ids,
            rankings=rankings,
            board_type=board_type,
            num_players=num_players,
            tournament_id=tournament_id,
            game_length=game_length,
            duration_sec=duration_sec,
            worker=worker,
            metadata=metadata,
            game_id=game_id,
        )

        # Get current ratings
        ratings = self.get_ratings_batch(participant_ids, board_type, num_players)

        # Calculate new ratings using EloCalculator
        calculator = EloCalculator(k_factor=k_factor)

        # Load current ratings into calculator
        for pid, rating in ratings.items():
            elo_rating = calculator.get_rating(pid)
            elo_rating.rating = rating.rating
            elo_rating.games_played = rating.games_played
            elo_rating.wins = rating.wins
            elo_rating.losses = rating.losses
            elo_rating.draws = rating.draws

        # Sort participants by ranking for the multiplayer update
        sorted_participants = sorted(
            zip(participant_ids, rankings),
            key=lambda x: x[1]
        )
        ordered_ids = [pid for pid, _ in sorted_participants]

        # Update ratings
        if len(participant_ids) == 2:
            # Two-player match
            result = 1.0 if rankings[0] < rankings[1] else (0.5 if rankings[0] == rankings[1] else 0.0)
            calculator.update_ratings(participant_ids[0], participant_ids[1], result)
        else:
            # Multiplayer match
            calculator.update_multiplayer_ratings(ordered_ids)

        # Update database with new ratings
        new_ratings = {}
        updated_ratings = []
        conn = self._get_connection()
        now = time.time()

        for pid in participant_ids:
            old_rating = ratings[pid].rating
            elo_rating = calculator.get_rating(pid)
            new_rating = elo_rating.rating

            updated = UnifiedEloRating(
                participant_id=pid,
                board_type=board_type,
                num_players=num_players,
                rating=new_rating,
                games_played=elo_rating.games_played,
                wins=elo_rating.wins,
                losses=elo_rating.losses,
                draws=elo_rating.draws,
            )
            updated_ratings.append(updated)
            new_ratings[pid] = new_rating

            # Record rating history
            conn.execute("""
                INSERT INTO rating_history
                (participant_id, board_type, num_players, rating, rating_change,
                 games_played, timestamp, match_id, tournament_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pid, board_type, num_players, new_rating, new_rating - old_rating,
                elo_rating.games_played, now, match_id, tournament_id,
            ))

        conn.commit()
        self.update_ratings_batch(updated_ratings)

        return match_id, new_ratings

    def record_two_player_result(
        self,
        winner_id: str,
        loser_id: str,
        board_type: str,
        num_players: int,
        tournament_id: str,
        is_draw: bool = False,
        game_length: int = 0,
        duration_sec: float = 0.0,
        worker: Optional[str] = None,
    ) -> Tuple[int, Dict[str, float]]:
        """Convenience method for recording a two-player match result.

        Args:
            winner_id: ID of the winner (or first player if draw)
            loser_id: ID of the loser (or second player if draw)
            board_type: Board type
            num_players: Number of players (usually 2)
            tournament_id: Tournament ID
            is_draw: Whether the game was a draw
            game_length: Number of moves
            duration_sec: Duration in seconds
            worker: Worker that ran the game

        Returns:
            Tuple of (match_id, dict of participant_id -> new_rating)
        """
        if is_draw:
            rankings = [0, 0]  # Tie
        else:
            rankings = [0, 1]  # Winner first, loser second

        return self.record_match_and_update(
            participant_ids=[winner_id, loser_id],
            rankings=rankings,
            board_type=board_type,
            num_players=num_players,
            tournament_id=tournament_id,
            game_length=game_length,
            duration_sec=duration_sec,
            worker=worker,
        )

    # =========================================================================
    # Queries
    # =========================================================================

    def get_leaderboard(
        self,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
        min_games: int = 1,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get leaderboard, optionally filtered by configuration."""
        conn = self._get_connection()

        # Check if participants table exists and get the id column name
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='participants'")
        has_participants = cursor.fetchone() is not None
        id_col = self.id_column  # model_id or participant_id

        if has_participants and not self._uses_model_id_schema:
            # Full schema with participants table
            query = f"""
                SELECT e.*, p.participant_type, p.ai_type, p.difficulty,
                       p.use_neural_net, p.model_path, p.model_version
                FROM elo_ratings e
                LEFT JOIN participants p ON e.{id_col} = p.participant_id
                WHERE e.games_played >= ?
            """
        else:
            # Simple schema (elo_leaderboard.db) - no participants table
            query = f"""
                SELECT {id_col} as participant_id, board_type, num_players,
                       rating, games_played, wins, losses, draws, last_update
                FROM elo_ratings
                WHERE games_played >= ?
            """
        params: List[Any] = [min_games]

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)

        query += " ORDER BY rating DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_match_history(
        self,
        participant_id: Optional[str] = None,
        tournament_id: Optional[str] = None,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get match history with optional filters."""
        conn = self._get_connection()

        query = "SELECT * FROM match_history WHERE 1=1"
        params: List[Any] = []

        if participant_id:
            query += " AND participant_ids LIKE ?"
            params.append(f'%"{participant_id}"%')
        if tournament_id:
            query += " AND tournament_id = ?"
            params.append(tournament_id)
        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["participant_ids"] = json.loads(r["participant_ids"])
            r["rankings"] = json.loads(r["rankings"]) if r["rankings"] else []
            if r.get("metadata"):
                r["metadata"] = json.loads(r["metadata"])
            results.append(r)
        return results

    def get_rating_history(
        self,
        participant_id: str,
        board_type: str,
        num_players: int,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get rating history for a participant in a specific config."""
        conn = self._get_connection()
        rows = conn.execute("""
            SELECT * FROM rating_history
            WHERE participant_id = ? AND board_type = ? AND num_players = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (participant_id, board_type, num_players, limit)).fetchall()
        return [dict(row) for row in rows]

    def get_head_to_head(
        self,
        participant_a: str,
        participant_b: str,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get head-to-head stats between two participants."""
        conn = self._get_connection()

        # Build query to find matches with both participants
        query = """
            SELECT * FROM match_history
            WHERE participant_ids LIKE ? AND participant_ids LIKE ?
        """
        params: List[Any] = [f'%"{participant_a}"%', f'%"{participant_b}"%']

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)

        rows = conn.execute(query, params).fetchall()

        a_wins = 0
        b_wins = 0
        draws = 0

        for row in rows:
            winner = row["winner"]
            if winner == participant_a:
                a_wins += 1
            elif winner == participant_b:
                b_wins += 1
            else:
                draws += 1

        total = a_wins + b_wins + draws
        return {
            "participant_a": participant_a,
            "participant_b": participant_b,
            "total_games": total,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "draws": draws,
            "a_win_rate": a_wins / total if total > 0 else 0.0,
            "b_win_rate": b_wins / total if total > 0 else 0.0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        conn = self._get_connection()

        participant_count = conn.execute(
            "SELECT COUNT(*) FROM participants"
        ).fetchone()[0]

        rating_count = conn.execute(
            "SELECT COUNT(*) FROM elo_ratings WHERE games_played > 0"
        ).fetchone()[0]

        match_count = conn.execute(
            "SELECT COUNT(*) FROM match_history"
        ).fetchone()[0]

        configs = conn.execute("""
            SELECT board_type, num_players, COUNT(*) as count
            FROM elo_ratings
            WHERE games_played > 0
            GROUP BY board_type, num_players
        """).fetchall()

        return {
            "total_participants": participant_count,
            "rated_participants": rating_count,
            "total_matches": match_count,
            "configurations": [dict(c) for c in configs],
        }


# =============================================================================
# Singleton Access
# =============================================================================

def get_elo_database(db_path: Optional[Path] = None) -> EloDatabase:
    """Get or create the singleton EloDatabase instance.

    Args:
        db_path: Optional custom database path. If provided on first call,
                 will be used for the singleton. Subsequent calls ignore this.

    Returns:
        The singleton EloDatabase instance.
    """
    global _elo_db_instance

    with _elo_db_lock:
        if _elo_db_instance is None:
            _elo_db_instance = EloDatabase(db_path)
        return _elo_db_instance


def reset_elo_database() -> None:
    """Reset the singleton (for testing)."""
    global _elo_db_instance

    with _elo_db_lock:
        if _elo_db_instance is not None:
            _elo_db_instance.close()
            _elo_db_instance = None
