"""Tournament CLI helper functions.

This module provides shared utilities for tournament management scripts:
- CLI argument parsing helpers
- Archive management (low-Elo model archiving/unarchiving)
- Elo-based matchmaking

SSoT for tournament management operations.
"""
from __future__ import annotations

import sqlite3
import time
from typing import TYPE_CHECKING, Any, Iterable

from app.config.thresholds import ARCHIVE_ELO_THRESHOLD

if TYPE_CHECKING:
    from app.elo.database import EloDatabase


# ============================================================================
# CLI Argument Parsing
# ============================================================================

_GLOBAL_FLAGS = {"-v", "--verbose"}
_GLOBAL_KV = {"--config", "--output-dir", "--seed"}


def split_global_args(argv: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split run_tournament global args from subcommand args."""
    global_args: list[str] = []
    subcommand_args: list[str] = []
    args = list(argv)
    idx = 0

    while idx < len(args):
        arg = args[idx]
        if arg in _GLOBAL_FLAGS:
            global_args.append(arg)
            idx += 1
            continue
        if arg in _GLOBAL_KV:
            if idx + 1 < len(args):
                global_args.extend([arg, args[idx + 1]])
                idx += 2
                continue
        subcommand_args.append(arg)
        idx += 1

    return global_args, subcommand_args


# ============================================================================
# Archive Management
# ============================================================================

def ensure_archived_models_table(db: EloDatabase) -> None:
    """Ensure the archived_models table exists in the database."""
    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS archived_models (
            model_id TEXT,
            board_type TEXT,
            num_players INTEGER,
            final_rating REAL,
            games_played INTEGER,
            archived_at REAL,
            PRIMARY KEY (model_id, board_type, num_players)
        )
    """)
    conn.commit()


# Protected algorithm patterns - never archive these
PROTECTED_ALGORITHM_PATTERNS = [
    "baseline_",      # baseline_random, baseline_heuristic, baseline_mcts
    "random",         # Random player
    "heuristic",      # Heuristic player
    "minimax",        # Minimax variants
    "mcts",           # Pure MCTS (non-neural)
    "nnue",           # NNUE models
]


def is_protected_algorithm(model_id: str) -> bool:
    """Check if a model ID represents a protected non-NN algorithm.

    Protected algorithms (random, heuristic, minimax, NNUE, etc.) should never
    be archived as they serve as calibration baselines for ELO ratings.

    Args:
        model_id: The participant/model identifier

    Returns:
        True if the model should be protected from archiving
    """
    model_id_lower = model_id.lower()
    for pattern in PROTECTED_ALGORITHM_PATTERNS:
        if pattern in model_id_lower:
            return True
    return False


def archive_low_elo_models(
    db: EloDatabase,
    board_type: str,
    num_players: int,
    elo_threshold: int = ARCHIVE_ELO_THRESHOLD,
    min_games: int = 50,
) -> list[str]:
    """Archive models with low Elo after sufficient games.

    Archived models are marked in the database and excluded from future tournaments.
    Non-NN algorithms (random, heuristic, minimax, NNUE, baselines) are never archived
    as they serve as calibration baselines for the ELO system.

    Args:
        db: EloDatabase instance
        board_type: Board type (e.g., "square8")
        num_players: Number of players (2, 3, or 4)
        elo_threshold: Archive models below this rating (default: ARCHIVE_ELO_THRESHOLD)
        min_games: Minimum games played before archiving (default: 50)

    Returns:
        List of archived model IDs.
    """
    conn = db._get_connection()
    cursor = conn.cursor()

    # Find models to archive (using unified schema with participant_id)
    # Excludes baseline and non-NN algorithms which should never be archived
    cursor.execute("""
        SELECT participant_id, rating, games_played
        FROM elo_ratings
        WHERE board_type = ? AND num_players = ?
          AND rating < ? AND games_played >= ?
          AND participant_id NOT LIKE 'baseline_%'
          AND participant_id NOT LIKE '%random%'
          AND participant_id NOT LIKE '%heuristic%'
          AND participant_id NOT LIKE '%minimax%'
          AND participant_id NOT LIKE '%mcts%'
          AND participant_id NOT LIKE '%nnue%'
    """, (board_type, num_players, elo_threshold, min_games))

    to_archive = []
    for row in cursor.fetchall():
        model_id, rating, games = row
        # Double-check protection (safety net in case SQL patterns miss something)
        if is_protected_algorithm(model_id):
            continue
        to_archive.append({
            "model_id": model_id,
            "rating": rating,
            "games_played": games,
        })

    if not to_archive:
        return []

    # Ensure archived_models table exists
    ensure_archived_models_table(db)

    # Archive the models
    archived = []
    now = time.time()
    for model in to_archive:
        cursor.execute("""
            INSERT OR REPLACE INTO archived_models
            (model_id, board_type, num_players, final_rating, games_played, archived_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (model["model_id"], board_type, num_players, model["rating"], model["games_played"], now))
        archived.append(model["model_id"])

    conn.commit()
    return archived


def is_model_archived(db: EloDatabase, model_id: str, board_type: str, num_players: int) -> bool:
    """Check if a model has been archived.

    Args:
        db: EloDatabase instance
        model_id: Model identifier
        board_type: Board type (e.g., "square8")
        num_players: Number of players (2, 3, or 4)

    Returns:
        True if the model is archived, False otherwise.
    """
    conn = db._get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 1 FROM archived_models
            WHERE model_id = ? AND board_type = ? AND num_players = ?
        """, (model_id, board_type, num_players))
        return cursor.fetchone() is not None
    except sqlite3.OperationalError:
        # Table doesn't exist yet - no models archived
        return False


def unarchive_model(db: EloDatabase, model_id: str, board_type: str, num_players: int) -> bool:
    """Remove a model from the archived_models table.

    Args:
        db: EloDatabase instance
        model_id: Model identifier
        board_type: Board type (e.g., "square8")
        num_players: Number of players (2, 3, or 4)

    Returns:
        True if the model was unarchived, False if it wasn't archived.
    """
    conn = db._get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM archived_models
            WHERE model_id = ? AND board_type = ? AND num_players = ?
        """, (model_id, board_type, num_players))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.OperationalError:
        # Table doesn't exist - nothing to unarchive
        return False


def unarchive_discovered_models(
    db: EloDatabase,
    models: list[dict],
    board_type: str,
    num_players: int,
    verbose: bool = True,
) -> int:
    """Unarchive any discovered models that are marked as archived in the database.

    This handles the case where model files were restored from archived/ directory
    to models/ directory but the database still has them marked as archived.

    Args:
        db: EloDatabase instance
        models: List of model dicts with "model_id" key
        board_type: Board type (e.g., "square8")
        num_players: Number of players (2, 3, or 4)
        verbose: If True, print unarchived model names

    Returns:
        The count of models unarchived.
    """
    unarchived_count = 0
    for m in models:
        model_id = m.get("model_id", "")
        if is_model_archived(db, model_id, board_type, num_players):
            if unarchive_model(db, model_id, board_type, num_players):
                if verbose:
                    print(f"  Unarchived: {model_id} (model file exists)")
                unarchived_count += 1
    return unarchived_count


def get_archived_models(
    db: EloDatabase,
    board_type: str | None = None,
    num_players: int | None = None,
) -> list[dict]:
    """Get list of archived models.

    Args:
        db: EloDatabase instance
        board_type: Optional filter by board type
        num_players: Optional filter by number of players

    Returns:
        List of archived model records.
    """
    conn = db._get_connection()
    cursor = conn.cursor()
    try:
        if board_type and num_players:
            cursor.execute("""
                SELECT model_id, board_type, num_players, final_rating, games_played, archived_at
                FROM archived_models
                WHERE board_type = ? AND num_players = ?
                ORDER BY archived_at DESC
            """, (board_type, num_players))
        elif board_type:
            cursor.execute("""
                SELECT model_id, board_type, num_players, final_rating, games_played, archived_at
                FROM archived_models
                WHERE board_type = ?
                ORDER BY archived_at DESC
            """, (board_type,))
        elif num_players:
            cursor.execute("""
                SELECT model_id, board_type, num_players, final_rating, games_played, archived_at
                FROM archived_models
                WHERE num_players = ?
                ORDER BY archived_at DESC
            """, (num_players,))
        else:
            cursor.execute("""
                SELECT model_id, board_type, num_players, final_rating, games_played, archived_at
                FROM archived_models
                ORDER BY archived_at DESC
            """)

        return [
            {
                "model_id": row[0],
                "board_type": row[1],
                "num_players": row[2],
                "final_rating": row[3],
                "games_played": row[4],
                "archived_at": row[5],
            }
            for row in cursor.fetchall()
        ]
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return []


# ============================================================================
# Elo-Based Matchmaking
# ============================================================================

def generate_elo_based_matchups(
    models: list[dict[str, Any]],
    db: EloDatabase,
    board_type: str,
    num_players: int,
    max_elo_diff: int = 200,
    default_rating: float = 1500.0,
) -> list[tuple[dict, dict]]:
    """Generate matchups between models with similar Elo ratings.

    This produces more informative games than random matchups, as close
    games provide more Elo information than one-sided blowouts.

    Args:
        models: List of model dicts with "model_id" key
        db: EloDatabase instance
        board_type: Board type (e.g., "square8")
        num_players: Number of players (2, 3, or 4)
        max_elo_diff: Maximum Elo difference for pairing (default: 200)
        default_rating: Default rating for unrated models (default: 1500.0)

    Returns:
        List of (model1, model2) tuples for matchups.
    """
    # Get current Elo ratings for all models
    model_elos = {}
    for model in models:
        rating = db.get_rating(model["model_id"], board_type, num_players)
        model_elos[model["model_id"]] = rating.rating

    # Sort models by Elo
    sorted_models = sorted(
        models,
        key=lambda m: model_elos.get(m["model_id"], default_rating),
        reverse=True
    )

    matchups = []
    used: set[str] = set()

    # Pair adjacent models in Elo ranking (closest ratings play each other)
    for _i, m1 in enumerate(sorted_models):
        if m1["model_id"] in used:
            continue

        # Find best opponent (closest Elo within range, not already paired)
        best_opponent = None
        best_diff = float("inf")

        for m2 in sorted_models:
            if m2["model_id"] == m1["model_id"] or m2["model_id"] in used:
                continue

            elo_diff = abs(model_elos[m1["model_id"]] - model_elos[m2["model_id"]])
            if elo_diff <= max_elo_diff and elo_diff < best_diff:
                best_diff = elo_diff
                best_opponent = m2

        if best_opponent:
            matchups.append((m1, best_opponent))
            used.add(m1["model_id"])
            used.add(best_opponent["model_id"])

    # Add remaining unmatched models paired with closest available
    unmatched = [m for m in sorted_models if m["model_id"] not in used]
    for i in range(0, len(unmatched) - 1, 2):
        matchups.append((unmatched[i], unmatched[i + 1]))

    return matchups


def filter_archived_models(
    models: list[dict],
    db: EloDatabase,
    board_type: str,
    num_players: int,
) -> list[dict]:
    """Filter out archived models from a list.

    Args:
        models: List of model dicts with "model_id" key
        db: EloDatabase instance
        board_type: Board type (e.g., "square8")
        num_players: Number of players (2, 3, or 4)

    Returns:
        List of models that are not archived.
    """
    return [
        m for m in models
        if not is_model_archived(db, m.get("model_id", ""), board_type, num_players)
    ]
