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
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path setup
from app.utils.paths import UNIFIED_ELO_DB
from app.utils.torch_utils import safe_load_checkpoint

DEFAULT_ELO_DB_PATH = UNIFIED_ELO_DB

# Import canonical thresholds
try:
    from app.config.thresholds import (
        BASELINE_ELO_RANDOM,
        ELO_K_FACTOR,
        INITIAL_ELO_RATING,
        MIN_GAMES_FOR_ELO,
    )
except ImportError:
    # Fallback defaults if thresholds not available
    INITIAL_ELO_RATING = 1500.0
    ELO_K_FACTOR = 32
    MIN_GAMES_FOR_ELO = 30
    BASELINE_ELO_RANDOM = 400


def _is_random_participant(participant_id: str) -> bool:
    """Check if participant is a random baseline that should be anchored at 400 Elo.

    Random players serve as the anchor point for the entire rating system.
    Their rating is fixed at BASELINE_ELO_RANDOM (400) to prevent rating inflation.
    """
    pid_lower = participant_id.lower()
    # Composite format: none:random:d1
    if pid_lower.startswith("none:random"):
        return True
    # Legacy formats
    if pid_lower in ("random", "baseline_random", "tier1_random"):
        return True
    # Robust check: "random" in name but not "heuristic" or other algorithms
    if "random" in pid_lower and not any(
        x in pid_lower for x in ("heuristic", "minimax", "mcts", "descent", "neural")
    ):
        return True
    return False

# Import coordination for single-writer enforcement
# Using the new coordination module (cluster_coordinator is deprecated)
try:
    from app.coordination.helpers import (
        get_orchestrator_roles,
        get_role_holder,
        has_coordination as _has_coordination,
        has_role,
    )
    HAS_COORDINATION = _has_coordination()
    OrchestratorRole = get_orchestrator_roles()
except ImportError:
    HAS_COORDINATION = False
    OrchestratorRole = None
    has_role = None
    get_role_holder = None

# Singleton instance
_elo_service_instance: EloService | None = None
_elo_service_lock = threading.RLock()

# Event emission for ELO updates
try:
    from app.coordination.event_router import emit_elo_updated
    HAS_ELO_EVENTS = True
except ImportError:
    HAS_ELO_EVENTS = False
    emit_elo_updated = None

# Event emission for Elo velocity/significant changes (December 2025 - training pipeline fix)
try:
    from app.distributed.data_events import emit_data_event, DataEventType
    HAS_ELO_VELOCITY_EVENTS = True
except ImportError:
    HAS_ELO_VELOCITY_EVENTS = False
    emit_data_event = None  # type: ignore
    DataEventType = None  # type: ignore

# Elo event thresholds (December 2025)
# ELO_SIGNIFICANT_CHANGE: Threshold for single-match Elo change to be "significant"
ELO_SIGNIFICANT_CHANGE_THRESHOLD = 25.0  # More than 25 Elo change in a single match

# ELO_VELOCITY_CHANGE: Minimum Elo change per hour to trigger velocity event
ELO_VELOCITY_THRESHOLD_PER_HOUR = 50.0  # 50+ Elo/hour indicates significant improvement rate

# Composite event emission (Sprint 5)
try:
    from app.training.event_integration import publish_composite_elo_updated_sync
    from app.training.composite_participant import (
        is_composite_id,
        parse_composite_participant_id,
    )
    HAS_COMPOSITE_EVENTS = True
except ImportError:
    HAS_COMPOSITE_EVENTS = False
    publish_composite_elo_updated_sync = None
    is_composite_id = None
    parse_composite_participant_id = None


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
    participant_ids: list[str]
    winner_id: str | None  # None for draw
    game_length: int
    duration_sec: float
    board_type: str
    num_players: int
    timestamp: str
    elo_changes: dict[str, float] = field(default_factory=dict)


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

    def __init__(self, db_path: Path | None = None, enforce_single_writer: bool = True):
        """Initialize the Elo service.

        Args:
            db_path: Path to SQLite database
            enforce_single_writer: If True, check cluster coordination before writes
        """
        self.db_path = db_path or DEFAULT_ELO_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._feedback_callbacks: list[Callable[[TrainingFeedback], None]] = []
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
        if (is_write and self._enforce_single_writer and OrchestratorRole is not None
                and has_role is not None and has_role(OrchestratorRole.TOURNAMENT_RUNNER)):
            # Check if tournament role is held (tournaments write to Elo DB)
            holder_info = get_role_holder(OrchestratorRole.TOURNAMENT_RUNNER) if get_role_holder is not None else None
            if holder_info and hasattr(holder_info, 'pid') and holder_info.pid != os.getpid():
                raise RuntimeError(
                    f"Elo write blocked: TOURNAMENT_RUNNER role held by PID {holder_info.pid}. "
                    "Only one process should write to Elo DB at a time."
                )

        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise

    def check_write_permission(self) -> tuple[bool, str]:
        """Check if this process can write to the Elo database.

        Returns:
            (can_write, reason) tuple
        """
        if not self._enforce_single_writer:
            return True, "Single-writer enforcement disabled"

        if OrchestratorRole is None or has_role is None:
            return True, "No coordinator available"

        if has_role(OrchestratorRole.TOURNAMENT_RUNNER):
            holder_info = get_role_holder(OrchestratorRole.TOURNAMENT_RUNNER) if get_role_holder is not None else None
            if holder_info and hasattr(holder_info, 'pid'):
                if holder_info.pid == os.getpid():
                    return True, "This process holds TOURNAMENT_RUNNER role"
                return False, f"TOURNAMENT_RUNNER role held by PID {holder_info.pid}"

        return True, "No conflicting role held"

    def execute_query(
        self,
        query: str,
        params: tuple = ()
    ) -> list[sqlite3.Row]:
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
                    participant_id TEXT PRIMARY KEY,
                    participant_type TEXT NOT NULL DEFAULT 'model',
                    ai_type TEXT,
                    difficulty INTEGER,
                    use_neural_net INTEGER DEFAULT 0,
                    model_path TEXT,
                    model_version TEXT,
                    metadata TEXT,
                    created_at REAL,
                    last_seen REAL,
                    nn_model_id TEXT,
                    nn_model_path TEXT,
                    ai_algorithm TEXT,
                    algorithm_config TEXT,
                    is_composite INTEGER DEFAULT 0
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
                    tournament_id TEXT,
                    metadata TEXT
                )
            """)

            # Migration: add metadata column if not exists (for existing DBs)
            try:
                conn.execute("ALTER TABLE match_history ADD COLUMN metadata TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Migration: add winner_id column if not exists (for existing DBs)
            try:
                conn.execute("ALTER TABLE match_history ADD COLUMN winner_id TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

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

            # Schema migrations for existing databases (December 2025)
            # Add peak_rating column if missing (older databases don't have it)
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            columns = {row[1] for row in cursor.fetchall()}
            if "peak_rating" not in columns:
                conn.execute("ALTER TABLE elo_ratings ADD COLUMN peak_rating REAL DEFAULT 1500.0")
                logger.info("Migrated elo_ratings: added peak_rating column")

    def register_participant(
        self,
        participant_id: str,
        name: str | None = None,  # Deprecated: not stored in DB, use participant_id
        ai_type: str = "unknown",
        difficulty: int | None = None,
        use_neural_net: bool = False,
        model_path: str | None = None,
        metadata: dict | None = None
    ) -> None:
        """Register a new participant (model or AI baseline).

        Note: The `name` parameter is deprecated and ignored. The participant_id
        serves as the display name.
        """
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO participants
                (participant_id, ai_type, difficulty, use_neural_net, model_path, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                participant_id,
                ai_type,
                difficulty,
                int(use_neural_net),
                model_path,
                time.time(),
                json.dumps(metadata) if metadata else None
            ))

    def _validate_model_player_count(
        self, model_path: str, expected_num_players: int
    ) -> tuple[bool, int | None]:
        """Validate that a model's player count matches expected value.

        Args:
            model_path: Path to model checkpoint file
            expected_num_players: Expected number of players (2, 3, or 4)

        Returns:
            Tuple of (is_valid, actual_num_players). actual_num_players is None
            if it couldn't be determined from the checkpoint.
        """
        try:
            import torch
            path = Path(model_path)
            if not path.exists():
                # Check common directories
                for model_dir in [Path("models"), Path("models_essential")]:
                    candidate = model_dir / path
                    if candidate.exists():
                        path = candidate
                        break
                    candidate = model_dir / path.name
                    if candidate.exists():
                        path = candidate
                        break

            if not path.exists():
                return True, None  # Can't validate, assume OK

            checkpoint = safe_load_checkpoint(path, map_location="cpu", warn_on_unsafe=False)

            # Try to get num_players from checkpoint metadata
            actual_num_players = checkpoint.get("num_players")

            # If not in metadata, infer from value head shape
            if actual_num_players is None:
                state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
                if isinstance(state_dict, dict):
                    # Check value_fc2 shape - first dim is num_players
                    for key in ["value_fc2.weight", "value_head.fc2.weight"]:
                        if key in state_dict:
                            actual_num_players = state_dict[key].shape[0]
                            break

            if actual_num_players is None:
                return True, None  # Can't determine, assume OK

            return actual_num_players == expected_num_players, actual_num_players

        except Exception as e:
            logger.debug(f"Could not validate player count for {model_path}: {e}")
            return True, None  # On error, don't block registration

    def _validate_model_path(self, model_path: str | None) -> bool:
        """Validate that a model file exists.

        Args:
            model_path: Path to model file (absolute or relative)

        Returns:
            True if file exists or path is None, False otherwise
        """
        if not model_path:
            return True  # None is valid for baselines

        path = Path(model_path)

        # Check absolute path
        if path.is_absolute():
            return path.exists()

        # Check relative to common model directories
        model_dirs = [
            Path("models"),
            Path("models_essential"),
            Path("."),
        ]
        for model_dir in model_dirs:
            candidate = model_dir / path
            if candidate.exists():
                return True
            # Try just the filename
            candidate = model_dir / path.name
            if candidate.exists():
                return True

        return path.exists()

    def register_model(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        model_path: str | None = None,
        parent_model_id: str | None = None,
        validate_file: bool = True
    ) -> None:
        """Register a trained model and initialize its Elo rating.

        Args:
            model_id: Unique identifier for the model
            board_type: Board type (e.g., 'square8', 'hex8')
            num_players: Number of players (2, 3, or 4)
            model_path: Optional path to model file
            parent_model_id: Optional ID of parent model (for lineage tracking)
            validate_file: If True, verify model file exists before registering
        """
        # Validate model file exists (prevent phantom entries)
        if validate_file and model_path and not self._validate_model_path(model_path):
            logger.warning(
                f"Model file not found, skipping registration: {model_path} "
                f"(model_id={model_id}). Use validate_file=False to override."
            )
            return

        # Validate player count matches (prevent player count mismatch - Dec 2025)
        if validate_file and model_path:
            is_valid, actual_players = self._validate_model_player_count(
                model_path, num_players
            )
            if not is_valid:
                logger.error(
                    f"Player count mismatch! Model {model_path} has {actual_players} "
                    f"players but trying to register for {num_players}-player config. "
                    f"Skipping registration to prevent invalid Elo entries."
                )
                return

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
        """Get participant's Elo rating, creating initial if needed.

        Note: Random players are anchored at BASELINE_ELO_RANDOM (400) to serve
        as the fixed reference point and prevent rating inflation.
        """
        # Anchor random participants at fixed Elo to prevent rating drift
        if _is_random_participant(participant_id):
            # Still fetch games_played from DB for stats, but rating is fixed
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT games_played, wins, losses, draws, last_update
                FROM elo_ratings
                WHERE participant_id = ? AND board_type = ? AND num_players = ?
            """, (participant_id, board_type, num_players))
            row = cursor.fetchone()
            if row:
                return EloRating(
                    participant_id=participant_id,
                    rating=float(BASELINE_ELO_RANDOM),  # ANCHORED
                    games_played=row["games_played"],
                    wins=row["wins"],
                    losses=row["losses"],
                    draws=row["draws"],
                    last_update=row["last_update"] or 0.0,
                    confidence=1.0  # Random is always reliable as anchor
                )
            # Create entry for random participant at anchored rating
            with self._transaction() as txn_conn:
                txn_conn.execute("""
                    INSERT OR IGNORE INTO elo_ratings
                    (participant_id, board_type, num_players, rating, last_update)
                    VALUES (?, ?, ?, ?, ?)
                """, (participant_id, board_type, num_players, float(BASELINE_ELO_RANDOM), time.time()))
            return EloRating(
                participant_id=participant_id,
                rating=float(BASELINE_ELO_RANDOM),  # ANCHORED
                confidence=1.0
            )

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

    def _get_adaptive_k_factor(
        self,
        games_a: int,
        games_b: int,
        base_k: float,
    ) -> float:
        """Calculate adaptive K-factor based on games played.

        Adaptive K-factor improves rating convergence:
        - New models (< 30 games): K = 1.5x base (faster convergence)
        - Developing models (30-100 games): K = 1.25x base
        - Established models (100-300 games): K = 1.0x base
        - Mature models (300+ games): K = 0.75x base (stability)

        The K-factor is determined by the LESS confident participant,
        so matches involving new models are more impactful.

        Args:
            games_a: Games played by participant A
            games_b: Games played by participant B
            base_k: Base K-factor (after multiplayer scaling)

        Returns:
            Adaptive K-factor scaled by confidence

        December 2025: Added to improve rating accuracy for new models
        while maintaining stability for established models.
        """
        # Use the minimum games (less confident participant drives K)
        min_games = min(games_a, games_b)

        # Tiered K-factor multipliers
        if min_games < 30:
            # Provisional: High K for rapid convergence
            k_multiplier = 1.5
        elif min_games < 100:
            # Developing: Moderately high K
            k_multiplier = 1.25
        elif min_games < 300:
            # Established: Standard K
            k_multiplier = 1.0
        else:
            # Mature: Lower K for stability
            k_multiplier = 0.75

        return base_k * k_multiplier

    def record_match(
        self,
        participant_a: str,
        participant_b: str,
        winner: str | None,  # None for draw
        board_type: str,
        num_players: int,
        game_length: int = 0,
        duration_sec: float = 0.0,
        tournament_id: str | None = None,
        metadata: dict | None = None,
    ) -> MatchResult:
        """Record a match result and update Elo ratings.

        Args:
            metadata: Optional dict with match metadata. Useful keys:
                - weight_profile_a: Heuristic weight profile ID for participant A
                - weight_profile_b: Heuristic weight profile ID for participant B
                - source: Origin of the match (e.g., "tournament", "selfplay")
        """
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

        # Scale K-factor for multiplayer games
        # In N-player games, each pairwise matchup is 1/(N-1) of the rating info
        # This ensures consistent rating change magnitude across player counts
        base_k = self.K_FACTOR / (num_players - 1) if num_players > 2 else self.K_FACTOR

        # Apply adaptive K-factor based on confidence (December 2025)
        # New models (low games) get higher K for faster convergence
        # Established models (high games) get lower K for stability
        k = self._get_adaptive_k_factor(
            games_a=rating_a.games_played,
            games_b=rating_b.games_played,
            base_k=base_k,
        )

        # Calculate Elo changes
        change_a = k * (score_a - exp_a)
        change_b = k * (score_b - exp_b)

        new_rating_a = rating_a.rating + change_a
        new_rating_b = rating_b.rating + change_b

        # ANCHOR: Force random players back to fixed Elo (prevents rating inflation)
        # Random serves as the anchor point for the entire rating system
        if _is_random_participant(participant_a):
            new_rating_a = float(BASELINE_ELO_RANDOM)
            change_a = 0.0  # No effective change for anchored player
        if _is_random_participant(participant_b):
            new_rating_b = float(BASELINE_ELO_RANDOM)
            change_b = 0.0  # No effective change for anchored player

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

            # Record match with optional metadata (e.g., weight profiles used)
            conn.execute("""
                INSERT INTO match_history
                (id, participant_ids, winner_id, game_length, duration_sec,
                 board_type, num_players, timestamp, elo_before, elo_after,
                 tournament_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                tournament_id,
                json.dumps(metadata) if metadata else None
            ))

        # Emit ELO_UPDATED events for both participants
        # This enables event-driven coordination across the training pipeline
        if HAS_ELO_EVENTS and emit_elo_updated is not None:
            config_key = f"{board_type}_{num_players}p"
            try:
                # Try to get running event loop
                try:
                    asyncio.get_running_loop()
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
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass  # Don't let event emission break match recording

        # Emit composite ELO events for composite participants (Sprint 5)
        if HAS_COMPOSITE_EVENTS and publish_composite_elo_updated_sync is not None:
            for pid, old_elo, new_elo in [
                (participant_a, elo_before[participant_a], elo_after[participant_a]),
                (participant_b, elo_before[participant_b], elo_after[participant_b]),
            ]:
                if is_composite_id and is_composite_id(pid):
                    try:
                        parsed = parse_composite_participant_id(pid)
                        if parsed:
                            nn_id, ai_type, config_hash = parsed
                            publish_composite_elo_updated_sync(
                                nn_id=nn_id,
                                ai_type=ai_type,
                                config_hash=config_hash,
                                participant_id=pid,
                                old_elo=old_elo,
                                new_elo=new_elo,
                                games_played=1,
                                board_type=board_type,
                                num_players=num_players,
                            )
                    except (RuntimeError, AttributeError, TypeError, ValueError, KeyError):
                        pass  # Don't let event emission break match recording

        # December 2025: Emit ELO_SIGNIFICANT_CHANGE and ELO_VELOCITY_CHANGED events
        # These events drive training prioritization in SelfplayScheduler and
        # curriculum rebalancing in CurriculumIntegration
        if HAS_ELO_VELOCITY_EVENTS and emit_data_event is not None:
            config_key = f"{board_type}_{num_players}p"
            try:
                # Check for significant single-match Elo changes
                for pid, old_elo, new_elo in [
                    (participant_a, elo_before[participant_a], elo_after[participant_a]),
                    (participant_b, elo_before[participant_b], elo_after[participant_b]),
                ]:
                    elo_delta = new_elo - old_elo
                    if abs(elo_delta) > ELO_SIGNIFICANT_CHANGE_THRESHOLD:
                        try:
                            asyncio.get_running_loop()
                            asyncio.ensure_future(emit_data_event(
                                event_type=DataEventType.ELO_SIGNIFICANT_CHANGE,
                                payload={
                                    "config_key": config_key,
                                    "board_type": board_type,
                                    "num_players": num_players,
                                    "participant_id": pid,
                                    "old_elo": old_elo,
                                    "new_elo": new_elo,
                                    "delta": elo_delta,
                                },
                            ))
                        except RuntimeError:
                            pass  # No event loop - skip in sync context

                # Calculate velocity (Elo per hour) if we have duration
                if duration_sec and duration_sec > 0:
                    hours = duration_sec / 3600.0
                    for pid, old_elo, new_elo in [
                        (participant_a, elo_before[participant_a], elo_after[participant_a]),
                        (participant_b, elo_before[participant_b], elo_after[participant_b]),
                    ]:
                        elo_delta = new_elo - old_elo
                        elo_per_hour = elo_delta / hours if hours > 0 else 0.0

                        if abs(elo_per_hour) > ELO_VELOCITY_THRESHOLD_PER_HOUR:
                            try:
                                asyncio.get_running_loop()
                                asyncio.ensure_future(emit_data_event(
                                    event_type=DataEventType.ELO_VELOCITY_CHANGED,
                                    payload={
                                        "config_key": config_key,
                                        "board_type": board_type,
                                        "num_players": num_players,
                                        "participant_id": pid,
                                        "velocity": elo_per_hour,
                                        "trend": "improving" if elo_per_hour > 0 else "declining",
                                    },
                                ))
                            except RuntimeError:
                                pass  # No event loop - skip in sync context

            except (RuntimeError, AttributeError, TypeError, ValueError):
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
    ) -> list[LeaderboardEntry]:
        """Get the Elo leaderboard for a configuration."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT
                e.participant_id,
                p.participant_id AS name,
                p.ai_type,
                e.rating,
                e.games_played,
                e.wins,
                e.losses,
                e.draws,
                e.last_update
            FROM elo_ratings e
            JOIN participants p ON e.participant_id = p.participant_id
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
            with suppress(Exception):
                callback(feedback)

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
    ) -> tuple[float, int]:
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

    # =========================================================================
    # Composite Participant Methods (Sprint 1)
    # =========================================================================

    def register_composite_participant(
        self,
        nn_id: str | None,
        ai_type: str,
        config: dict[str, Any] | None = None,
        board_type: str = "square8",
        num_players: int = 2,
        nn_model_path: str | None = None,
    ) -> str:
        """Register a composite (NN, Algorithm) participant.

        Creates a composite participant ID and registers it with full metadata.

        Args:
            nn_id: Neural network identifier, or None for non-NN participants
            ai_type: Search algorithm type (e.g., "gumbel_mcts", "mcts")
            config: Algorithm configuration (uses defaults if None)
            board_type: Board type for rating
            num_players: Number of players
            nn_model_path: Path to NN model file

        Returns:
            Composite participant ID
        """
        from app.training.composite_participant import (
            encode_config_hash,
            get_standard_config,
            make_composite_participant_id,
        )

        # Create composite ID
        actual_config = config or get_standard_config(ai_type)
        participant_id = make_composite_participant_id(nn_id, ai_type, actual_config)
        config_hash = encode_config_hash(actual_config, ai_type)

        # Register as participant with extended metadata
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO participants
                (participant_id, participant_type, ai_type, use_neural_net, model_path,
                 nn_model_id, nn_model_path, ai_algorithm, algorithm_config, is_composite,
                 created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                participant_id,
                "composite",
                ai_type,
                int(nn_id is not None),
                nn_model_path,
                nn_id,
                nn_model_path,
                ai_type,
                json.dumps(actual_config),
                1,  # is_composite = True
                time.time(),
            ))

        # Initialize rating for this config
        self.get_rating(participant_id, board_type, num_players)

        return participant_id

    def get_nn_performance_summary(
        self,
        nn_model_id: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any] | None:
        """Get aggregated performance summary for an NN across algorithms.

        Args:
            nn_model_id: Neural network identifier
            board_type: Board type
            num_players: Number of players

        Returns:
            Dict with best_algorithm, best_elo, avg_elo, algorithms_tested
            or None if no data
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT best_algorithm, best_elo, avg_elo, algorithms_tested, last_updated
            FROM nn_performance_summary
            WHERE nn_model_id = ? AND board_type = ? AND num_players = ?
        """, (nn_model_id, board_type, num_players))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "nn_model_id": nn_model_id,
            "board_type": board_type,
            "num_players": num_players,
            "best_algorithm": row["best_algorithm"],
            "best_elo": row["best_elo"],
            "avg_elo": row["avg_elo"],
            "algorithms_tested": row["algorithms_tested"],
            "last_updated": row["last_updated"],
        }

    def get_algorithm_baseline(
        self,
        ai_algorithm: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any] | None:
        """Get baseline rating for an algorithm.

        Args:
            ai_algorithm: Algorithm type
            board_type: Board type
            num_players: Number of players

        Returns:
            Dict with baseline_elo, games_played or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT baseline_elo, games_played, last_updated
            FROM algorithm_baselines
            WHERE ai_algorithm = ? AND board_type = ? AND num_players = ?
        """, (ai_algorithm, board_type, num_players))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "ai_algorithm": ai_algorithm,
            "board_type": board_type,
            "num_players": num_players,
            "baseline_elo": row["baseline_elo"],
            "games_played": row["games_played"],
            "last_updated": row["last_updated"],
        }

    def update_algorithm_baseline(
        self,
        ai_algorithm: str,
        board_type: str,
        num_players: int,
        baseline_elo: float,
        games_played: int = 0,
    ) -> None:
        """Update baseline rating for an algorithm.

        Args:
            ai_algorithm: Algorithm type
            board_type: Board type
            num_players: Number of players
            baseline_elo: New baseline Elo rating
            games_played: Number of games played
        """
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO algorithm_baselines
                (ai_algorithm, board_type, num_players, baseline_elo, games_played, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ai_algorithm, board_type, num_players, baseline_elo, games_played, time.time()))

    def get_composite_leaderboard(
        self,
        board_type: str,
        num_players: int,
        ai_algorithm: str | None = None,
        nn_model_id: str | None = None,
        limit: int = 50,
        min_games: int = 0,
    ) -> list[dict[str, Any]]:
        """Get leaderboard filtered by algorithm or NN.

        Args:
            board_type: Board type
            num_players: Number of players
            ai_algorithm: Filter by algorithm (optional)
            nn_model_id: Filter by NN model (optional)
            limit: Maximum entries
            min_games: Minimum games required

        Returns:
            List of leaderboard entries
        """
        conn = self._get_connection()

        query = """
            SELECT
                e.participant_id,
                p.nn_model_id,
                p.ai_algorithm,
                e.rating,
                e.games_played,
                e.wins,
                e.losses,
                e.draws,
                e.last_update
            FROM elo_ratings e
            JOIN participants p ON e.participant_id = p.participant_id
            WHERE e.board_type = ? AND e.num_players = ? AND e.games_played >= ?
        """
        params: list[Any] = [board_type, num_players, min_games]

        if ai_algorithm:
            query += " AND p.ai_algorithm = ?"
            params.append(ai_algorithm)

        if nn_model_id:
            query += " AND p.nn_model_id = ?"
            params.append(nn_model_id)

        query += " ORDER BY e.rating DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)

        entries = []
        for rank, row in enumerate(cursor.fetchall(), 1):
            games = row["games_played"]
            win_rate = (row["wins"] + 0.5 * row["draws"]) / games if games > 0 else 0.5
            entries.append({
                "rank": rank,
                "participant_id": row["participant_id"],
                "nn_model_id": row["nn_model_id"],
                "ai_algorithm": row["ai_algorithm"],
                "rating": row["rating"],
                "games_played": games,
                "wins": row["wins"],
                "losses": row["losses"],
                "draws": row["draws"],
                "win_rate": win_rate,
                "last_update": row["last_update"],
            })

        return entries

    def get_algorithm_rankings(
        self,
        board_type: str,
        num_players: int,
        min_games: int = 10,
    ) -> list[dict[str, Any]]:
        """Get algorithm rankings based on average performance.

        Aggregates ratings across all NNs for each algorithm.

        Args:
            board_type: Board type
            num_players: Number of players
            min_games: Minimum games for inclusion

        Returns:
            List of algorithm rankings with avg_elo, best_elo, nn_count
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT
                p.ai_algorithm,
                AVG(e.rating) as avg_elo,
                MAX(e.rating) as best_elo,
                MIN(e.rating) as worst_elo,
                COUNT(DISTINCT p.nn_model_id) as nn_count,
                SUM(e.games_played) as total_games
            FROM elo_ratings e
            JOIN participants p ON e.participant_id = p.participant_id
            WHERE e.board_type = ? AND e.num_players = ?
                AND e.games_played >= ?
                AND p.ai_algorithm IS NOT NULL
            GROUP BY p.ai_algorithm
            ORDER BY avg_elo DESC
        """, (board_type, num_players, min_games))

        rankings = []
        for rank, row in enumerate(cursor.fetchall(), 1):
            rankings.append({
                "rank": rank,
                "ai_algorithm": row["ai_algorithm"],
                "avg_elo": row["avg_elo"],
                "best_elo": row["best_elo"],
                "worst_elo": row["worst_elo"],
                "nn_count": row["nn_count"],
                "total_games": row["total_games"],
                "elo_spread": row["best_elo"] - row["worst_elo"],
            })

        return rankings

    def get_nn_rankings(
        self,
        board_type: str,
        num_players: int,
        min_games: int = 10,
    ) -> list[dict[str, Any]]:
        """Get NN rankings based on best performance across algorithms.

        Args:
            board_type: Board type
            num_players: Number of players
            min_games: Minimum games for inclusion

        Returns:
            List of NN rankings with best_elo, best_algorithm, algorithm_count
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT
                p.nn_model_id,
                MAX(e.rating) as best_elo,
                AVG(e.rating) as avg_elo,
                COUNT(DISTINCT p.ai_algorithm) as algorithm_count,
                SUM(e.games_played) as total_games
            FROM elo_ratings e
            JOIN participants p ON e.participant_id = p.participant_id
            WHERE e.board_type = ? AND e.num_players = ?
                AND e.games_played >= ?
                AND p.nn_model_id IS NOT NULL
                AND p.nn_model_id != 'none'
            GROUP BY p.nn_model_id
            ORDER BY best_elo DESC
        """, (board_type, num_players, min_games))

        rankings = []
        for rank, row in enumerate(cursor.fetchall(), 1):
            # Get best algorithm for this NN
            best_algo_cursor = conn.execute("""
                SELECT p.ai_algorithm
                FROM elo_ratings e
                JOIN participants p ON e.participant_id = p.participant_id
                WHERE p.nn_model_id = ? AND e.board_type = ? AND e.num_players = ?
                ORDER BY e.rating DESC
                LIMIT 1
            """, (row["nn_model_id"], board_type, num_players))
            best_algo_row = best_algo_cursor.fetchone()
            best_algorithm = best_algo_row["ai_algorithm"] if best_algo_row else None

            rankings.append({
                "rank": rank,
                "nn_model_id": row["nn_model_id"],
                "best_elo": row["best_elo"],
                "avg_elo": row["avg_elo"],
                "best_algorithm": best_algorithm,
                "algorithm_count": row["algorithm_count"],
                "total_games": row["total_games"],
            })

        return rankings


def get_elo_service(db_path: Path | None = None) -> EloService:
    """Get the singleton EloService instance."""
    global _elo_service_instance
    with _elo_service_lock:
        if _elo_service_instance is None:
            _elo_service_instance = EloService(db_path)
        return _elo_service_instance


def reset_elo_service() -> None:
    """Reset the singleton EloService instance (for testing).

    Dec 29, 2025: Added to fix test class leak issue in tournament tests.
    Ensures test isolation by clearing the cached service instance.
    Thread-local connections will be garbage collected.

    Usage in tests:
        @pytest.fixture(autouse=True)
        def cleanup_elo():
            yield
            reset_elo_service()
    """
    global _elo_service_instance
    with _elo_service_lock:
        if _elo_service_instance is not None:
            # Try to close thread-local connection if accessible
            try:
                if hasattr(_elo_service_instance, '_local'):
                    local = _elo_service_instance._local
                    if hasattr(local, 'connection') and local.connection is not None:
                        local.connection.close()
                        local.connection = None
            except (sqlite3.Error, AttributeError):
                pass  # Ignore close errors, connection will be GC'd
            _elo_service_instance = None


# =============================================================================
# Backwards Compatibility Layer
# =============================================================================
# These functions provide the same interface as scripts/run_model_elo_tournament.py
# to allow smooth migration of orchestrators to use this centralized service.


def init_elo_database(db_path: Path | None = None) -> EloService:
    """Initialize and return the Elo service (backwards compatible)."""
    return get_elo_service(db_path)


def register_models(db: EloService, models: list[dict[str, Any]]) -> None:
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
    winner: str | None,  # model_a_id, model_b_id, "draw", or None
    board_type: str = "square8",
    num_players: int = 2,
    game_length: int = 0,
    duration_sec: float = 0.0,
) -> dict[str, Any]:
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
) -> list[dict[str, Any]]:
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
    board_type: str | None = None,
    num_players: int | None = None,
) -> dict[str, Any]:
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
    params: list[Any] = [f'%{participant_a}%', f'%{participant_b}%']

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


def get_database_stats(db: EloService) -> dict[str, Any]:
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
    participant_id: str | None = None,
    tournament_id: str | None = None,
    board_type: str | None = None,
    num_players: int | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
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
    params: list[Any] = []

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
            with suppress(json.JSONDecodeError, TypeError):
                r["participant_ids"] = json.loads(r["participant_ids"])
        if r.get("elo_before"):
            with suppress(json.JSONDecodeError, TypeError):
                r["elo_before"] = json.loads(r["elo_before"])
        if r.get("elo_after"):
            with suppress(json.JSONDecodeError, TypeError):
                r["elo_after"] = json.loads(r["elo_after"])
        results.append(r)
    return results


def get_rating_history(
    db: EloService,
    participant_id: str,
    board_type: str,
    num_players: int,
    limit: int = 100,
) -> list[dict[str, Any]]:
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


def get_elo_trend(
    db: EloService,
    participant_id: str,
    board_type: str,
    num_players: int,
    hours: int = 48,
    min_samples: int = 3,
) -> dict[str, Any]:
    """Get Elo trend (slope) over the last N hours.

    December 2025: Added for Phase 4 weak config detection.
    This enables detecting permanently stuck configs that should have
    their selfplay resource allocation reduced.

    Args:
        db: The EloService instance
        participant_id: Participant to query (e.g., "canonical_model")
        board_type: Board type (e.g., "square8")
        num_players: Player count
        hours: Time window in hours (default 48h)
        min_samples: Minimum data points required

    Returns:
        Dict with trend analysis:
        - slope: Elo change per hour (positive = improving)
        - start_elo: Rating at start of window
        - end_elo: Current/latest rating
        - total_change: end_elo - start_elo
        - duration_hours: Actual time span covered
        - sample_count: Number of data points
        - is_plateau: True if slope < 1.0 Elo/hour and 5+ samples
        - is_declining: True if slope < -1.0 Elo/hour
        - confidence: 0.0-1.0 based on sample count

    Example:
        trend = get_elo_trend(db, "canonical", "square8", 2, hours=48)
        if trend["is_plateau"] and trend["duration_hours"] >= 48:
            print(f"Config stuck for {trend['duration_hours']:.1f}h")
    """
    cutoff_time = time.time() - (hours * 3600)

    conn = db._get_connection()
    cursor = conn.execute("""
        SELECT rating, timestamp FROM elo_history
        WHERE participant_id = ? AND board_type = ? AND num_players = ?
          AND timestamp >= ?
        ORDER BY timestamp ASC
    """, (participant_id, board_type, num_players, cutoff_time))

    rows = cursor.fetchall()

    # Default return for insufficient data
    result = {
        "slope": 0.0,
        "start_elo": 0.0,
        "end_elo": 0.0,
        "total_change": 0.0,
        "duration_hours": 0.0,
        "sample_count": len(rows),
        "is_plateau": False,
        "is_declining": False,
        "confidence": 0.0,
    }

    if len(rows) < min_samples:
        return result

    # Extract ratings and timestamps
    ratings = [row["rating"] for row in rows]
    timestamps = [row["timestamp"] for row in rows]

    start_elo = ratings[0]
    end_elo = ratings[-1]
    duration_seconds = timestamps[-1] - timestamps[0]
    duration_hours = duration_seconds / 3600.0

    if duration_hours < 0.1:  # Less than 6 minutes
        return result

    # Calculate slope using simple linear regression
    total_change = end_elo - start_elo
    slope = total_change / duration_hours if duration_hours > 0 else 0.0

    # Calculate R² for confidence (optional, simple variance-based)
    mean_rating = sum(ratings) / len(ratings)
    ss_tot = sum((r - mean_rating) ** 2 for r in ratings)
    if ss_tot > 0 and len(ratings) >= 3:
        # Simple linear fit residuals
        predicted = [
            start_elo + slope * ((t - timestamps[0]) / 3600.0)
            for t in timestamps
        ]
        ss_res = sum((r - p) ** 2 for r, p in zip(ratings, predicted))
        r_squared = 1.0 - (ss_res / ss_tot)
        r_squared = max(0.0, min(1.0, r_squared))
    else:
        r_squared = 0.5  # Default moderate confidence

    # Confidence based on sample count and R²
    sample_confidence = min(1.0, len(rows) / 20.0)  # Max confidence at 20 samples
    confidence = (sample_confidence + r_squared) / 2.0

    # Plateau detection: less than 1 Elo/hour change with enough samples
    is_plateau = abs(slope) < 1.0 and len(rows) >= 5

    # Decline detection: losing more than 1 Elo/hour
    is_declining = slope < -1.0

    result.update({
        "slope": round(slope, 3),
        "start_elo": round(start_elo, 1),
        "end_elo": round(end_elo, 1),
        "total_change": round(total_change, 1),
        "duration_hours": round(duration_hours, 2),
        "sample_count": len(rows),
        "is_plateau": is_plateau,
        "is_declining": is_declining,
        "confidence": round(confidence, 3),
    })

    return result


def get_elo_trend_for_config(
    config_key: str,
    hours: int = 48,
    participant_id: str = "canonical",
) -> dict[str, Any]:
    """Convenience function to get Elo trend for a config key.

    December 2025: Added for Phase 4 weak config detection.

    Args:
        config_key: Config identifier like "square8_2p"
        hours: Time window in hours
        participant_id: Participant ID to query

    Returns:
        Trend analysis dict (see get_elo_trend)
    """
    # Parse config key
    parts = config_key.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].endswith("p"):
        return {
            "slope": 0.0, "is_plateau": False, "confidence": 0.0,
            "error": f"Invalid config_key format: {config_key}"
        }

    board_type = parts[0]
    try:
        num_players = int(parts[1][:-1])
    except ValueError:
        return {
            "slope": 0.0, "is_plateau": False, "confidence": 0.0,
            "error": f"Invalid player count in config_key: {config_key}"
        }

    db = get_elo_service()
    return get_elo_trend(db, participant_id, board_type, num_players, hours)


# Canonical path - orchestrators should use this
ELO_DB_PATH = DEFAULT_ELO_DB_PATH
