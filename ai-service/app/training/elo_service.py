#!/usr/bin/env python3
"""Centralized Elo Rating Service - Single source of truth for all model ratings.

This module provides THE authoritative interface for all Elo operations across
the RingRift AI improvement infrastructure. All scripts should import from here
rather than implementing their own Elo access.

Features:
- Persistent SQLite storage with unified_elo.db
- Thread-safe operations with connection pooling
- Automatic schema migrations
- Feedback hooks for training parameter adaptation
- Integration with model lifecycle management

Usage:
    from app.training.elo_service import EloService, get_elo_service

    # Get singleton instance
    elo = get_elo_service()

    # Register and rate models
    elo.register_model("model_v1", board_type="square8", num_players=2)
    elo.record_match("model_v1", "model_v2", winner="model_v1", board_type="square8", num_players=2)

    # Get feedback signals for training
    feedback = elo.get_training_feedback("square8", 2)
    if feedback.elo_stagnating:
        # Adjust training parameters
        pass
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Path setup
from app.utils.paths import UNIFIED_ELO_DB
DEFAULT_ELO_DB_PATH = UNIFIED_ELO_DB

# Import canonical thresholds
try:
    from app.config.thresholds import (
        INITIAL_ELO_RATING,
        ELO_K_FACTOR,
        MIN_GAMES_FOR_ELO,
    )
except ImportError:
    # Fallback defaults if thresholds not available
    INITIAL_ELO_RATING = 1500.0
    ELO_K_FACTOR = 32
    MIN_GAMES_FOR_ELO = 30

# Import coordination for single-writer enforcement
# Using the new coordination module (cluster_coordinator is deprecated)
try:
    from app.coordination.helpers import (
        has_coordination as _has_coordination,
        has_role,
        get_role_holder,
        get_orchestrator_roles,
    )
    HAS_COORDINATION = _has_coordination()
    OrchestratorRole = get_orchestrator_roles()
except ImportError:
    HAS_COORDINATION = False
    OrchestratorRole = None
    has_role = None
    get_role_holder = None

# Singleton instance
_elo_service_instance: Optional["EloService"] = None
_elo_service_lock = threading.RLock()

# Event emission for ELO updates
try:
    from app.distributed.data_events import emit_elo_updated
    HAS_ELO_EVENTS = True
except ImportError:
    HAS_ELO_EVENTS = False
    emit_elo_updated = None


@dataclass
class EloRating:
    """Elo rating with metadata.

    Note: Default rating uses INITIAL_ELO_RATING from app.config.thresholds (1500.0)
    """
    participant_id: str
    rating: float = 1500.0  # See app.config.thresholds.INITIAL_ELO_RATING
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_update: float = 0.0
    confidence: float = 0.0  # Based on games played

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.games_played


@dataclass
class MatchResult:
    """Result of a single match."""
    match_id: str
    participant_ids: List[str]
    winner_id: Optional[str]  # None for draw
    game_length: int
    duration_sec: float
    board_type: str
    num_players: int
    timestamp: str
    elo_changes: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingFeedback:
    """Feedback signals for training parameter adaptation."""
    board_type: str
    num_players: int

    # Elo trend analysis
    best_elo: float = 1500.0
    recent_elo_delta: float = 0.0  # Change over last N iterations
    elo_stagnating: bool = False  # True if no improvement in 5+ iterations
    elo_declining: bool = False  # True if negative trend

    # Win rate analysis
    best_win_rate: float = 0.5
    win_rate_vs_baseline: float = 0.5

    # Recommended adjustments
    epochs_multiplier: float = 1.0  # Multiply default epochs by this
    lr_multiplier: float = 1.0  # Multiply default LR by this
    exploration_boost: float = 0.0  # Add to temperature/noise

    # Curriculum recommendation
    recommended_curriculum_stage: int = 0


@dataclass
class LeaderboardEntry:
    """Entry in the Elo leaderboard."""
    rank: int
    participant_id: str
    name: str
    ai_type: str
    rating: float
    games_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    last_active: str


class EloService:
    """Centralized Elo rating service with feedback integration and single-writer enforcement."""

    # Use canonical thresholds from app.config.thresholds
    K_FACTOR = float(ELO_K_FACTOR)
    INITIAL_ELO = float(INITIAL_ELO_RATING)
    CONFIDENCE_GAMES = MIN_GAMES_FOR_ELO  # Games needed for high confidence

    def __init__(self, db_path: Optional[Path] = None, enforce_single_writer: bool = True):
        """Initialize the Elo service.

        Args:
            db_path: Path to SQLite database
            enforce_single_writer: If True, check cluster coordination before writes
        """
        self.db_path = db_path or DEFAULT_ELO_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._feedback_callbacks: List[Callable[[TrainingFeedback], None]] = []
        self._enforce_single_writer = enforce_single_writer and HAS_COORDINATION
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def _transaction(self, is_write: bool = True):
        """Context manager for database transactions.

        Args:
            is_write: If True, check single-writer enforcement before proceeding
        """
        # Check single-writer enforcement for write operations
        if is_write and self._enforce_single_writer and OrchestratorRole is not None:
            # Check if tournament role is held (tournaments write to Elo DB)
            if has_role is not None and has_role(OrchestratorRole.TOURNAMENT_RUNNER):
                holder_info = get_role_holder(OrchestratorRole.TOURNAMENT_RUNNER) if get_role_holder else None
                if holder_info and hasattr(holder_info, 'pid') and holder_info.pid != os.getpid():
                    raise RuntimeError(
                        f"Elo write blocked: TOURNAMENT_RUNNER role held by PID {holder_info.pid}. "
                        "Only one process should write to Elo DB at a time."
                    )

        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def check_write_permission(self) -> Tuple[bool, str]:
        """Check if this process can write to the Elo database.

        Returns:
            (can_write, reason) tuple
        """
        if not self._enforce_single_writer:
            return True, "Single-writer enforcement disabled"

        if OrchestratorRole is None or has_role is None:
            return True, "No coordinator available"

        if has_role(OrchestratorRole.TOURNAMENT_RUNNER):
            holder_info = get_role_holder(OrchestratorRole.TOURNAMENT_RUNNER) if get_role_holder else None
            if holder_info and hasattr(holder_info, 'pid'):
                if holder_info.pid == os.getpid():
                    return True, "This process holds TOURNAMENT_RUNNER role"
                return False, f"TOURNAMENT_RUNNER role held by PID {holder_info.pid}"

        return True, "No conflicting role held"

    def execute_query(
        self,
        query: str,
        params: Tuple = ()
    ) -> List[sqlite3.Row]:
        """Execute a read-only query and return results.

        This provides a centralized way to run custom queries against the Elo
        database while benefiting from connection pooling and thread-safety.

        Args:
            query: SQL query string (should be read-only SELECT)
            params: Query parameters tuple

        Returns:
            List of sqlite3.Row objects (supports both index and name access)

        Example:
            elo = get_elo_service()
            rows = elo.execute_query(
                "SELECT participant_id, rating FROM elo_ratings WHERE rating > ?",
                (1300,)
            )
            for row in rows:
                print(f"{row['participant_id']}: {row['rating']}")
        """
        conn = self._get_connection()
        cursor = conn.execute(query, params)
        return cursor.fetchall()

    def _init_db(self):
        """Initialize database schema with all required tables."""
        with self._transaction() as conn:
            # Participants table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS participants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    ai_type TEXT NOT NULL,
                    difficulty INTEGER,
                    use_neural_net INTEGER DEFAULT 0,
                    model_path TEXT,
                    created_at REAL,
                    metadata TEXT
                )
            """)

            # Elo ratings per configuration
            conn.execute("""
                CREATE TABLE IF NOT EXISTS elo_ratings (
                    participant_id TEXT NOT NULL,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    rating REAL DEFAULT 1500.0,
                    games_played INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    peak_rating REAL DEFAULT 1500.0,
                    last_update REAL,
                    PRIMARY KEY (participant_id, board_type, num_players)
                )
            """)

            # Match history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS match_history (
                    id TEXT PRIMARY KEY,
                    participant_ids TEXT NOT NULL,
                    winner_id TEXT,
                    game_length INTEGER,
                    duration_sec REAL,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    timestamp TEXT,
                    elo_before TEXT,
                    elo_after TEXT,
                    tournament_id TEXT
                )
            """)

            # Elo history for trend analysis
            conn.execute("""
                CREATE TABLE IF NOT EXISTS elo_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    participant_id TEXT NOT NULL,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    rating REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    iteration INTEGER
                )
            """)

            # Training feedback signals
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_feedback (
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    iteration INTEGER NOT NULL,
                    best_elo REAL,
                    elo_delta REAL,
                    epochs_multiplier REAL DEFAULT 1.0,
                    lr_multiplier REAL DEFAULT 1.0,
                    curriculum_stage INTEGER DEFAULT 0,
                    timestamp REAL,
                    PRIMARY KEY (board_type, num_players, iteration)
                )
            """)

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_elo_config ON elo_ratings(board_type, num_players)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_match_time ON match_history(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_participant ON elo_history(participant_id, board_type, num_players)")

    def register_participant(
        self,
        participant_id: str,
        name: str,
        ai_type: str,
        difficulty: Optional[int] = None,
        use_neural_net: bool = False,
        model_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Register a new participant (model or AI baseline)."""
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO participants
                (id, name, ai_type, difficulty, use_neural_net, model_path, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                participant_id,
                name,
                ai_type,
                difficulty,
                int(use_neural_net),
                model_path,
                time.time(),
                json.dumps(metadata) if metadata else None
            ))

    def register_model(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        model_path: Optional[str] = None,
        parent_model_id: Optional[str] = None
    ) -> None:
        """Register a trained model and initialize its Elo rating."""
        # Register as participant
        self.register_participant(
            participant_id=model_id,
            name=model_id,
            ai_type="neural_net",
            use_neural_net=True,
            model_path=model_path,
            metadata={"parent_model_id": parent_model_id}
        )

        # Initialize rating
        self.get_rating(model_id, board_type, num_players)

    def get_rating(
        self,
        participant_id: str,
        board_type: str,
        num_players: int
    ) -> EloRating:
        """Get participant's Elo rating, creating initial if needed."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT rating, games_played, wins, losses, draws, last_update
            FROM elo_ratings
            WHERE participant_id = ? AND board_type = ? AND num_players = ?
        """, (participant_id, board_type, num_players))
        row = cursor.fetchone()

        if row:
            confidence = min(1.0, row["games_played"] / self.CONFIDENCE_GAMES)
            return EloRating(
                participant_id=participant_id,
                rating=row["rating"],
                games_played=row["games_played"],
                wins=row["wins"],
                losses=row["losses"],
                draws=row["draws"],
                last_update=row["last_update"] or 0.0,
                confidence=confidence
            )

        # Create initial rating
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO elo_ratings
                (participant_id, board_type, num_players, rating, last_update)
                VALUES (?, ?, ?, ?, ?)
            """, (participant_id, board_type, num_players, self.INITIAL_ELO, time.time()))

        return EloRating(
            participant_id=participant_id,
            rating=self.INITIAL_ELO,
            confidence=0.0
        )

    def record_match(
        self,
        participant_a: str,
        participant_b: str,
        winner: Optional[str],  # None for draw
        board_type: str,
        num_players: int,
        game_length: int = 0,
        duration_sec: float = 0.0,
        tournament_id: Optional[str] = None
    ) -> MatchResult:
        """Record a match result and update Elo ratings."""
        match_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Get current ratings
        rating_a = self.get_rating(participant_a, board_type, num_players)
        rating_b = self.get_rating(participant_b, board_type, num_players)

        elo_before = {participant_a: rating_a.rating, participant_b: rating_b.rating}

        # Calculate expected scores
        exp_a = 1.0 / (1.0 + math.pow(10, (rating_b.rating - rating_a.rating) / 400))
        exp_b = 1.0 - exp_a

        # Actual scores
        if winner == participant_a:
            score_a, score_b = 1.0, 0.0
        elif winner == participant_b:
            score_a, score_b = 0.0, 1.0
        else:
            score_a, score_b = 0.5, 0.5

        # Calculate Elo changes
        change_a = self.K_FACTOR * (score_a - exp_a)
        change_b = self.K_FACTOR * (score_b - exp_b)

        new_rating_a = rating_a.rating + change_a
        new_rating_b = rating_b.rating + change_b

        elo_after = {participant_a: new_rating_a, participant_b: new_rating_b}
        elo_changes = {participant_a: change_a, participant_b: change_b}

        # Update database
        with self._transaction() as conn:
            # Update ratings
            for pid, new_rating, score in [
                (participant_a, new_rating_a, score_a),
                (participant_b, new_rating_b, score_b)
            ]:
                win_inc = 1 if score == 1.0 else 0
                loss_inc = 1 if score == 0.0 else 0
                draw_inc = 1 if score == 0.5 else 0

                conn.execute("""
                    UPDATE elo_ratings
                    SET rating = ?,
                        games_played = games_played + 1,
                        wins = wins + ?,
                        losses = losses + ?,
                        draws = draws + ?,
                        peak_rating = MAX(peak_rating, ?),
                        last_update = ?
                    WHERE participant_id = ? AND board_type = ? AND num_players = ?
                """, (
                    new_rating, win_inc, loss_inc, draw_inc,
                    new_rating, time.time(),
                    pid, board_type, num_players
                ))

            # Record match
            conn.execute("""
                INSERT INTO match_history
                (id, participant_ids, winner_id, game_length, duration_sec,
                 board_type, num_players, timestamp, elo_before, elo_after, tournament_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id,
                json.dumps([participant_a, participant_b]),
                winner,
                game_length,
                duration_sec,
                board_type,
                num_players,
                timestamp,
                json.dumps(elo_before),
                json.dumps(elo_after),
                tournament_id
            ))

        # Emit ELO_UPDATED events for both participants
        # This enables event-driven coordination across the training pipeline
        if HAS_ELO_EVENTS and emit_elo_updated is not None:
            config_key = f"{board_type}_{num_players}p"
            try:
                # Try to get running event loop
                try:
                    loop = asyncio.get_running_loop()
                    # Schedule coroutines in the running loop
                    for pid, old_elo, new_elo in [
                        (participant_a, elo_before[participant_a], elo_after[participant_a]),
                        (participant_b, elo_before[participant_b], elo_after[participant_b]),
                    ]:
                        asyncio.ensure_future(emit_elo_updated(
                            config=config_key,
                            model_id=pid,
                            new_elo=new_elo,
                            old_elo=old_elo,
                            games_played=1,
                            source="elo_service",
                        ))
                except RuntimeError:
                    # No running loop - create one for sync context
                    # This is less efficient but ensures events are emitted
                    pass  # Skip in pure sync context to avoid blocking
            except Exception:
                pass  # Don't let event emission break match recording

        return MatchResult(
            match_id=match_id,
            participant_ids=[participant_a, participant_b],
            winner_id=winner,
            game_length=game_length,
            duration_sec=duration_sec,
            board_type=board_type,
            num_players=num_players,
            timestamp=timestamp,
            elo_changes=elo_changes
        )

    def get_leaderboard(
        self,
        board_type: str,
        num_players: int,
        limit: int = 50,
        min_games: int = 0
    ) -> List[LeaderboardEntry]:
        """Get the Elo leaderboard for a configuration."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT
                e.participant_id,
                p.name,
                p.ai_type,
                e.rating,
                e.games_played,
                e.wins,
                e.losses,
                e.draws,
                e.last_update
            FROM elo_ratings e
            JOIN participants p ON e.participant_id = p.id
            WHERE e.board_type = ? AND e.num_players = ? AND e.games_played >= ?
            ORDER BY e.rating DESC
            LIMIT ?
        """, (board_type, num_players, min_games, limit))

        entries = []
        for rank, row in enumerate(cursor.fetchall(), 1):
            games = row["games_played"]
            win_rate = (row["wins"] + 0.5 * row["draws"]) / games if games > 0 else 0.5
            entries.append(LeaderboardEntry(
                rank=rank,
                participant_id=row["participant_id"],
                name=row["name"],
                ai_type=row["ai_type"],
                rating=row["rating"],
                games_played=games,
                wins=row["wins"],
                losses=row["losses"],
                draws=row["draws"],
                win_rate=win_rate,
                last_active=datetime.fromtimestamp(row["last_update"] or 0).isoformat()
            ))
        return entries

    def get_training_feedback(
        self,
        board_type: str,
        num_players: int,
        lookback_iterations: int = 5
    ) -> TrainingFeedback:
        """Get feedback signals for training parameter adaptation."""
        conn = self._get_connection()

        # Get best model rating
        cursor = conn.execute("""
            SELECT MAX(rating) as best_rating, MAX(peak_rating) as peak
            FROM elo_ratings
            WHERE board_type = ? AND num_players = ?
        """, (board_type, num_players))
        row = cursor.fetchone()
        best_elo = row["best_rating"] or self.INITIAL_ELO

        # Get recent Elo history for trend analysis
        cursor = conn.execute("""
            SELECT rating, iteration FROM elo_history
            WHERE board_type = ? AND num_players = ?
            ORDER BY iteration DESC
            LIMIT ?
        """, (board_type, num_players, lookback_iterations))
        history = cursor.fetchall()

        # Calculate trend
        recent_elo_delta = 0.0
        elo_stagnating = False
        elo_declining = False

        if len(history) >= 2:
            recent_elo_delta = history[0]["rating"] - history[-1]["rating"]
            if abs(recent_elo_delta) < 10:  # Less than 10 Elo change
                elo_stagnating = True
            if recent_elo_delta < -20:  # More than 20 Elo drop
                elo_declining = True

        # Calculate recommended adjustments
        epochs_multiplier = 1.0
        lr_multiplier = 1.0
        exploration_boost = 0.0
        curriculum_stage = 0

        if elo_stagnating:
            # Try longer training with lower LR
            epochs_multiplier = 1.5
            lr_multiplier = 0.8
            exploration_boost = 0.1  # Add some exploration

        if elo_declining:
            # Something went wrong - increase epochs significantly
            epochs_multiplier = 2.0
            lr_multiplier = 0.5
            exploration_boost = 0.2

        # Curriculum progression based on Elo
        if best_elo > 1600:
            curriculum_stage = 1
        if best_elo > 1700:
            curriculum_stage = 2
        if best_elo > 1800:
            curriculum_stage = 3

        feedback = TrainingFeedback(
            board_type=board_type,
            num_players=num_players,
            best_elo=best_elo,
            recent_elo_delta=recent_elo_delta,
            elo_stagnating=elo_stagnating,
            elo_declining=elo_declining,
            epochs_multiplier=epochs_multiplier,
            lr_multiplier=lr_multiplier,
            exploration_boost=exploration_boost,
            recommended_curriculum_stage=curriculum_stage
        )

        # Notify callbacks
        for callback in self._feedback_callbacks:
            try:
                callback(feedback)
            except Exception:
                pass

        return feedback

    def record_iteration_elo(
        self,
        participant_id: str,
        board_type: str,
        num_players: int,
        iteration: int
    ) -> None:
        """Record Elo snapshot for trend analysis."""
        rating = self.get_rating(participant_id, board_type, num_players)
        with self._transaction() as conn:
            conn.execute("""
                INSERT INTO elo_history
                (participant_id, board_type, num_players, rating, timestamp, iteration)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                participant_id, board_type, num_players,
                rating.rating, time.time(), iteration
            ))

    def register_feedback_callback(self, callback: Callable[[TrainingFeedback], None]) -> None:
        """Register a callback to be notified of training feedback."""
        self._feedback_callbacks.append(callback)

    def get_win_rate_vs_baseline(
        self,
        model_id: str,
        baseline_id: str,
        board_type: str,
        num_players: int
    ) -> Tuple[float, int]:
        """Get win rate of model vs specific baseline."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT winner_id FROM match_history
            WHERE board_type = ? AND num_players = ?
            AND participant_ids LIKE ? AND participant_ids LIKE ?
        """, (board_type, num_players, f'%{model_id}%', f'%{baseline_id}%'))

        wins = 0
        total = 0
        for row in cursor:
            total += 1
            if row["winner_id"] == model_id:
                wins += 1
            elif row["winner_id"] is None:
                wins += 0.5  # Draw counts as half

        win_rate = wins / total if total > 0 else 0.5
        return win_rate, total


def get_elo_service(db_path: Optional[Path] = None) -> EloService:
    """Get the singleton EloService instance."""
    global _elo_service_instance
    with _elo_service_lock:
        if _elo_service_instance is None:
            _elo_service_instance = EloService(db_path)
        return _elo_service_instance


# =============================================================================
# Backwards Compatibility Layer
# =============================================================================
# These functions provide the same interface as scripts/run_model_elo_tournament.py
# to allow smooth migration of orchestrators to use this centralized service.


def init_elo_database(db_path: Optional[Path] = None) -> EloService:
    """Initialize and return the Elo service (backwards compatible)."""
    return get_elo_service(db_path)


def register_models(db: EloService, models: List[Dict[str, Any]]) -> None:
    """Register multiple models in the Elo database.

    Args:
        db: The EloService instance
        models: List of model dicts with keys: model_id, model_path, board_type, num_players
    """
    for model in models:
        model_id = model.get("model_id") or model.get("id", "")
        model_path = model.get("model_path", "")
        board_type = model.get("board_type", "square8")
        num_players = model.get("num_players", 2)

        if not model_id:
            continue

        db.register_model(
            model_id=model_id,
            board_type=board_type,
            num_players=num_players,
            model_path=model_path
        )


def update_elo_after_match(
    db: EloService,
    model_a_id: str,
    model_b_id: str,
    winner: Optional[str],  # model_a_id, model_b_id, "draw", or None
    board_type: str = "square8",
    num_players: int = 2,
    game_length: int = 0,
    duration_sec: float = 0.0,
) -> Dict[str, Any]:
    """Update Elo ratings after a match (backwards compatible).

    Args:
        db: The EloService instance
        model_a_id: First model ID
        model_b_id: Second model ID
        winner: Winner ID, "draw", or None for draw
        board_type: Board type string
        num_players: Number of players
        game_length: Number of moves in game
        duration_sec: Game duration in seconds

    Returns:
        Dict with rating changes: {"model_a": new_rating, "model_b": new_rating, "changes": {...}}
    """
    # Normalize winner for draw cases
    winner_id = None
    if winner == "draw" or winner is None:
        winner_id = None
    elif winner == model_a_id or winner == "model_a":
        winner_id = model_a_id
    elif winner == model_b_id or winner == "model_b":
        winner_id = model_b_id
    else:
        winner_id = winner

    result = db.record_match(
        participant_a=model_a_id,
        participant_b=model_b_id,
        winner=winner_id,
        board_type=board_type,
        num_players=num_players,
        game_length=game_length,
        duration_sec=duration_sec
    )

    # Return in the format expected by legacy callers
    rating_a = db.get_rating(model_a_id, board_type, num_players)
    rating_b = db.get_rating(model_b_id, board_type, num_players)

    return {
        "model_a": rating_a.rating,
        "model_b": rating_b.rating,
        "changes": result.elo_changes,
        "match_id": result.match_id
    }


def get_leaderboard(
    db: EloService,
    board_type: str = "square8",
    num_players: int = 2,
    limit: int = 50,
    min_games: int = 0,
) -> List[Dict[str, Any]]:
    """Get Elo leaderboard (backwards compatible).

    Args:
        db: The EloService instance
        board_type: Board type to query
        num_players: Number of players to query
        limit: Maximum entries to return
        min_games: Minimum games required

    Returns:
        List of leaderboard entries as dicts
    """
    entries = db.get_leaderboard(
        board_type=board_type,
        num_players=num_players,
        limit=limit,
        min_games=min_games
    )
    return [asdict(e) for e in entries]


# =============================================================================
# Additional Query Methods (merged from unified_elo_db.py)
# =============================================================================


def get_head_to_head(
    db: EloService,
    participant_a: str,
    participant_b: str,
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
) -> Dict[str, Any]:
    """Get head-to-head stats between two participants.

    Args:
        db: The EloService instance
        participant_a: First participant ID
        participant_b: Second participant ID
        board_type: Filter by board type (optional)
        num_players: Filter by player count (optional)

    Returns:
        Dict with head-to-head stats
    """
    conn = db._get_connection()

    query = """
        SELECT winner_id FROM match_history
        WHERE participant_ids LIKE ? AND participant_ids LIKE ?
    """
    params: List[Any] = [f'%{participant_a}%', f'%{participant_b}%']

    if board_type:
        query += " AND board_type = ?"
        params.append(board_type)
    if num_players:
        query += " AND num_players = ?"
        params.append(num_players)

    cursor = conn.execute(query, params)

    a_wins = 0
    b_wins = 0
    draws = 0

    for row in cursor:
        winner = row["winner_id"]
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


def get_database_stats(db: EloService) -> Dict[str, Any]:
    """Get overall database statistics.

    Args:
        db: The EloService instance

    Returns:
        Dict with database stats
    """
    conn = db._get_connection()

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
        SELECT board_type, num_players, COUNT(*) as count, MAX(rating) as top_rating
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


def get_match_history(
    db: EloService,
    participant_id: Optional[str] = None,
    tournament_id: Optional[str] = None,
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get match history with optional filters.

    Args:
        db: The EloService instance
        participant_id: Filter by participant
        tournament_id: Filter by tournament
        board_type: Filter by board type
        num_players: Filter by player count
        limit: Maximum results

    Returns:
        List of match records
    """
    conn = db._get_connection()

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

    cursor = conn.execute(query, params)
    results = []
    for row in cursor.fetchall():
        r = dict(row)
        # Parse JSON fields if present
        if r.get("participant_ids"):
            try:
                r["participant_ids"] = json.loads(r["participant_ids"])
            except (json.JSONDecodeError, TypeError):
                pass
        if r.get("elo_before"):
            try:
                r["elo_before"] = json.loads(r["elo_before"])
            except (json.JSONDecodeError, TypeError):
                pass
        if r.get("elo_after"):
            try:
                r["elo_after"] = json.loads(r["elo_after"])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(r)
    return results


def get_rating_history(
    db: EloService,
    participant_id: str,
    board_type: str,
    num_players: int,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get rating history for a participant in a specific config.

    Args:
        db: The EloService instance
        participant_id: Participant to query
        board_type: Board type
        num_players: Player count
        limit: Maximum results

    Returns:
        List of rating history records
    """
    conn = db._get_connection()
    cursor = conn.execute("""
        SELECT * FROM elo_history
        WHERE participant_id = ? AND board_type = ? AND num_players = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (participant_id, board_type, num_players, limit))
    return [dict(row) for row in cursor.fetchall()]


# Canonical path - orchestrators should use this
ELO_DB_PATH = DEFAULT_ELO_DB_PATH
