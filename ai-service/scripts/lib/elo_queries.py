"""Unified ELO database query library.

This module provides a Single Source of Truth (SSoT) for all ELO database queries
used across dashboard, leaderboard, alerts, and promotion scripts. All scripts
should import query functions from here rather than writing inline SQL.

Usage:
    from scripts.lib.elo_queries import (
        get_production_candidates,
        get_top_models,
        get_games_by_config,
        get_model_stats,
    )

    # Use with default database
    candidates = get_production_candidates()

    # Use with custom database path
    candidates = get_production_candidates(db_path=Path("/custom/path.db"))
"""
from __future__ import annotations


import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Import canonical thresholds from SSoT
import sys

_SCRIPT_DIR = Path(__file__).parent.parent
_AI_SERVICE_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_AI_SERVICE_ROOT))

from app.config.thresholds import (
    ELO_TIER_ADVANCED,
    ELO_TIER_EXPERT,
    ELO_TIER_GRANDMASTER,
    ELO_TIER_INTERMEDIATE,
    ELO_TIER_MASTER,
    ELO_TIER_NOVICE,
    INITIAL_ELO_RATING,
    PRODUCTION_ELO_THRESHOLD,
    PRODUCTION_MIN_GAMES,
)

DEFAULT_DB = _AI_SERVICE_ROOT / "data" / "unified_elo.db"

# Tier thresholds in descending order for tier lookup
TIER_THRESHOLDS = [
    (ELO_TIER_GRANDMASTER, "Grandmaster", "GM"),
    (ELO_TIER_MASTER, "Master", "M"),
    (ELO_TIER_EXPERT, "Expert", "E"),
    (ELO_TIER_ADVANCED, "Advanced", "A"),
    (ELO_TIER_INTERMEDIATE, "Intermediate", "I"),
    (ELO_TIER_NOVICE, "Novice", "N"),
    (0, "Beginner", "B"),
]


@dataclass
class ModelRating:
    """A model's ELO rating data."""

    participant_id: str
    rating: float
    games_played: int
    board_type: str | None = None
    num_players: int | None = None

    @property
    def tier(self) -> str:
        """Get tier name for this rating."""
        return get_tier_name(self.rating)

    @property
    def tier_abbr(self) -> str:
        """Get tier abbreviation for this rating."""
        return get_tier_abbr(self.rating)

    @property
    def is_production_ready(self) -> bool:
        """Check if this model meets production criteria."""
        return (
            self.rating >= PRODUCTION_ELO_THRESHOLD
            and self.games_played >= PRODUCTION_MIN_GAMES
        )


@dataclass
class ModelStats:
    """Summary statistics for the ELO database."""

    total_models: int
    total_games: int
    production_ready: int
    best_model: str | None
    best_rating: float


def get_tier_name(rating: float) -> str:
    """Get tier name for an ELO rating."""
    for threshold, name, _abbr in TIER_THRESHOLDS:
        if rating >= threshold:
            return name
    return "Beginner"


def get_tier_abbr(rating: float) -> str:
    """Get tier abbreviation for an ELO rating."""
    for threshold, _name, abbr in TIER_THRESHOLDS:
        if rating >= threshold:
            return abbr
    return "B"


def _get_connection(db_path: Path | None = None) -> sqlite3.Connection | None:
    """Get database connection, returning None if DB doesn't exist."""
    path = db_path or DEFAULT_DB
    if not path.exists():
        return None
    return sqlite3.connect(str(path))


def get_production_candidates(
    db_path: Path | None = None,
    include_baselines: bool = False,
) -> list[ModelRating]:
    """Get models that meet production criteria.

    Args:
        db_path: Path to ELO database (defaults to unified_elo.db)
        include_baselines: Whether to include baseline models

    Returns:
        List of ModelRating objects for production-ready models, sorted by rating DESC
    """
    conn = _get_connection(db_path)
    if not conn:
        return []

    query = """
        SELECT participant_id, rating, games_played, board_type, num_players
        FROM elo_ratings
        WHERE rating >= ? AND games_played >= ?
    """
    params: list[Any] = [PRODUCTION_ELO_THRESHOLD, PRODUCTION_MIN_GAMES]

    if not include_baselines:
        query += " AND participant_id NOT LIKE 'baseline_%'"

    query += " ORDER BY rating DESC"

    cursor = conn.execute(query, params)
    results = [
        ModelRating(
            participant_id=row[0],
            rating=row[1],
            games_played=row[2],
            board_type=row[3],
            num_players=row[4],
        )
        for row in cursor.fetchall()
    ]
    conn.close()
    return results


def get_top_models(
    db_path: Path | None = None,
    limit: int = 10,
    include_baselines: bool = False,
    config: str | None = None,
    tier: str | None = None,
) -> list[ModelRating]:
    """Get top N models by rating.

    Args:
        db_path: Path to ELO database
        limit: Maximum number of models to return
        include_baselines: Whether to include baseline models
        config: Optional config filter (e.g., "square8_2p")
        tier: Optional tier filter (e.g., "expert", "master")

    Returns:
        List of ModelRating objects sorted by rating DESC
    """
    conn = _get_connection(db_path)
    if not conn:
        return []

    query = """
        SELECT participant_id, rating, games_played, board_type, num_players
        FROM elo_ratings
        WHERE 1=1
    """
    params: list[Any] = []

    if not include_baselines:
        query += " AND participant_id NOT LIKE 'baseline_%'"

    if config:
        # Parse config like "square8_2p" -> board_type="square8", num_players=2
        parts = config.rsplit("_", 1)
        if len(parts) == 2:
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))
            query += " AND board_type = ? AND num_players = ?"
            params.extend([board_type, num_players])

    if tier:
        tier_lower = tier.lower()
        for threshold, name, _abbr in TIER_THRESHOLDS:
            if name.lower() == tier_lower:
                # Find next tier threshold for upper bound
                next_threshold = 10000
                for i, (t, n, _) in enumerate(TIER_THRESHOLDS):
                    if n.lower() == tier_lower and i > 0:
                        next_threshold = TIER_THRESHOLDS[i - 1][0]
                        break
                query += " AND rating >= ? AND rating < ?"
                params.extend([threshold, next_threshold])
                break

    query += " ORDER BY rating DESC LIMIT ?"
    params.append(limit)

    cursor = conn.execute(query, params)
    results = [
        ModelRating(
            participant_id=row[0],
            rating=row[1],
            games_played=row[2],
            board_type=row[3],
            num_players=row[4],
        )
        for row in cursor.fetchall()
    ]
    conn.close()
    return results


def get_all_ratings(
    db_path: Path | None = None,
    include_baselines: bool = False,
) -> dict[str, float]:
    """Get all participant ratings as a dictionary.

    Args:
        db_path: Path to ELO database
        include_baselines: Whether to include baseline models

    Returns:
        Dictionary mapping participant_id to rating
    """
    conn = _get_connection(db_path)
    if not conn:
        return {}

    query = "SELECT participant_id, rating FROM elo_ratings"
    if not include_baselines:
        query += " WHERE participant_id NOT LIKE 'baseline_%'"
    query += " ORDER BY rating DESC"

    cursor = conn.execute(query)
    ratings = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return ratings


def get_model_stats(db_path: Path | None = None) -> ModelStats | None:
    """Get summary statistics for the database.

    Args:
        db_path: Path to ELO database

    Returns:
        ModelStats object with summary data, or None if DB doesn't exist
    """
    conn = _get_connection(db_path)
    if not conn:
        return None

    # Total models and games (excluding baselines)
    cursor = conn.execute("""
        SELECT COUNT(*), COALESCE(SUM(games_played), 0)
        FROM elo_ratings
        WHERE participant_id NOT LIKE 'baseline_%'
    """)
    total_models, total_games = cursor.fetchone()

    # Production ready count
    cursor = conn.execute(
        """
        SELECT COUNT(*)
        FROM elo_ratings
        WHERE rating >= ? AND games_played >= ?
          AND participant_id NOT LIKE 'baseline_%'
    """,
        (PRODUCTION_ELO_THRESHOLD, PRODUCTION_MIN_GAMES),
    )
    production_ready = cursor.fetchone()[0]

    # Best model
    cursor = conn.execute("""
        SELECT participant_id, rating
        FROM elo_ratings
        WHERE participant_id NOT LIKE 'baseline_%'
        ORDER BY rating DESC
        LIMIT 1
    """)
    best = cursor.fetchone()

    conn.close()

    return ModelStats(
        total_models=total_models or 0,
        total_games=total_games or 0,
        production_ready=production_ready,
        best_model=best[0] if best else None,
        best_rating=best[1] if best else 0.0,
    )


def get_games_by_config(db_path: Path | None = None) -> dict[str, int]:
    """Get game count by config (board_type + num_players).

    Args:
        db_path: Path to ELO database

    Returns:
        Dictionary mapping config key (e.g., "square8_2p") to game count
    """
    conn = _get_connection(db_path)
    if not conn:
        return {}

    cursor = conn.execute("""
        SELECT board_type, num_players, COUNT(*) as games
        FROM match_history
        GROUP BY board_type, num_players
    """)

    coverage = {f"{row[0]}_{row[1]}p": row[2] for row in cursor.fetchall()}
    conn.close()
    return coverage


def get_games_last_n_hours(
    db_path: Path | None = None,
    hours: int = 1,
) -> int:
    """Get count of games played in the last N hours.

    Args:
        db_path: Path to ELO database
        hours: Number of hours to look back

    Returns:
        Count of games in the time window
    """
    conn = _get_connection(db_path)
    if not conn:
        return 0

    import time

    cutoff = time.time() - (hours * 3600)

    cursor = conn.execute(
        """
        SELECT COUNT(*)
        FROM match_history
        WHERE timestamp > ?
    """,
        (cutoff,),
    )

    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_last_game_timestamp(db_path: Path | None = None) -> float | None:
    """Get timestamp of the most recent game.

    Args:
        db_path: Path to ELO database

    Returns:
        Unix timestamp of last game, or None if no games exist
    """
    conn = _get_connection(db_path)
    if not conn:
        return None

    cursor = conn.execute("SELECT MAX(timestamp) FROM match_history")
    result = cursor.fetchone()
    conn.close()

    if result and result[0]:
        return float(result[0])
    return None


def get_near_production(
    db_path: Path | None = None,
    min_games: int = 50,
    min_elo: float = 1500,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get models close to but not yet meeting production criteria.

    Args:
        db_path: Path to ELO database
        min_games: Minimum games to be considered "near"
        min_elo: Minimum ELO to be considered "near"
        limit: Maximum number of results

    Returns:
        List of dicts with model info and what's needed for production
    """
    conn = _get_connection(db_path)
    if not conn:
        return []

    cursor = conn.execute(
        """
        SELECT participant_id, rating, games_played, board_type, num_players
        FROM elo_ratings
        WHERE games_played >= ?
          AND rating >= ?
          AND rating < ?
          AND participant_id NOT LIKE 'baseline_%'
        ORDER BY rating DESC
        LIMIT ?
    """,
        (min_games, min_elo, PRODUCTION_ELO_THRESHOLD, limit),
    )

    results = []
    for row in cursor.fetchall():
        model_id, rating, games, board_type, num_players = row
        elo_needed = PRODUCTION_ELO_THRESHOLD - rating
        games_needed = max(0, PRODUCTION_MIN_GAMES - games)

        results.append(
            {
                "model_id": model_id,
                "rating": rating,
                "games": games,
                "board_type": board_type or "unknown",
                "num_players": num_players or 2,
                "tier": get_tier_name(rating),
                "elo_needed": elo_needed,
                "games_needed": games_needed,
            }
        )

    conn.close()
    return results


def get_models_by_tier(
    db_path: Path | None = None,
    include_baselines: bool = False,
) -> dict[str, int]:
    """Get count of models in each tier.

    Args:
        db_path: Path to ELO database
        include_baselines: Whether to include baseline models

    Returns:
        Dictionary mapping tier name to model count
    """
    conn = _get_connection(db_path)
    if not conn:
        return {}

    baseline_filter = (
        "" if include_baselines else " WHERE participant_id NOT LIKE 'baseline_%'"
    )

    tier_counts = {}
    for threshold, name, _abbr in TIER_THRESHOLDS:
        query = f"""
            SELECT COUNT(*)
            FROM elo_ratings
            {baseline_filter}
            {"AND" if baseline_filter else "WHERE"} rating >= ?
        """
        cursor = conn.execute(query, (threshold,))
        tier_counts[name] = cursor.fetchone()[0]

    conn.close()
    return tier_counts


# =============================================================================
# Harness-based Queries (January 2026)
# =============================================================================


def get_ratings_by_harness(
    db_path: Path | None = None,
    harness_type: str = "gumbel_mcts",
    board_type: str | None = None,
    num_players: int | None = None,
    include_baselines: bool = False,
) -> list[ModelRating]:
    """Get all ratings for a specific harness type.

    Args:
        db_path: Path to ELO database (defaults to unified_elo.db)
        harness_type: Harness type to filter by (e.g., "gumbel_mcts", "minimax")
        board_type: Optional board type filter
        num_players: Optional player count filter
        include_baselines: Whether to include baseline models

    Returns:
        List of ModelRating objects for models with the specified harness type
    """
    conn = _get_connection(db_path)
    if not conn:
        return []

    query = """
        SELECT participant_id, rating, games_played, board_type, num_players
        FROM elo_ratings
        WHERE harness_type = ?
    """
    params: list[Any] = [harness_type]

    if not include_baselines:
        query += " AND participant_id NOT LIKE 'baseline_%'"

    if board_type:
        query += " AND board_type = ?"
        params.append(board_type)

    if num_players:
        query += " AND num_players = ?"
        params.append(num_players)

    query += " ORDER BY rating DESC"

    cursor = conn.execute(query, params)
    results = [
        ModelRating(
            participant_id=row[0],
            rating=row[1],
            games_played=row[2],
            board_type=row[3],
            num_players=row[4],
        )
        for row in cursor.fetchall()
    ]
    conn.close()
    return results


def get_top_models_by_harness(
    db_path: Path | None = None,
    harness_type: str = "gumbel_mcts",
    board_type: str = "hex8",
    num_players: int = 2,
    limit: int = 10,
    include_baselines: bool = False,
) -> list[ModelRating]:
    """Get top-rated models for a specific harness type and config.

    Args:
        db_path: Path to ELO database
        harness_type: Harness type to filter by
        board_type: Board type to filter by
        num_players: Player count to filter by
        limit: Maximum number of models to return
        include_baselines: Whether to include baseline models

    Returns:
        List of ModelRating objects sorted by rating DESC
    """
    conn = _get_connection(db_path)
    if not conn:
        return []

    query = """
        SELECT participant_id, rating, games_played, board_type, num_players
        FROM elo_ratings
        WHERE harness_type = ?
          AND board_type = ?
          AND num_players = ?
    """
    params: list[Any] = [harness_type, board_type, num_players]

    if not include_baselines:
        query += " AND participant_id NOT LIKE 'baseline_%'"

    query += " ORDER BY rating DESC LIMIT ?"
    params.append(limit)

    cursor = conn.execute(query, params)
    results = [
        ModelRating(
            participant_id=row[0],
            rating=row[1],
            games_played=row[2],
            board_type=row[3],
            num_players=row[4],
        )
        for row in cursor.fetchall()
    ]
    conn.close()
    return results


def get_harness_performance_comparison(
    db_path: Path | None = None,
    model_id: str = "",
) -> dict[str, dict[str, Any]]:
    """Compare a model's performance across different harness types.

    Args:
        db_path: Path to ELO database
        model_id: Model identifier (e.g., "canonical_hex8_2p")

    Returns:
        Dict mapping harness_type to {rating, games_played, board_type, num_players}

    Example:
        >>> comparison = get_harness_performance_comparison(model_id="canonical_hex8_2p")
        >>> comparison["gumbel_mcts"]
        {'rating': 1750.0, 'games_played': 50, 'board_type': 'hex8', 'num_players': 2}
        >>> comparison["minimax"]
        {'rating': 1680.0, 'games_played': 30, 'board_type': 'hex8', 'num_players': 2}
    """
    conn = _get_connection(db_path)
    if not conn:
        return {}

    query = """
        SELECT harness_type, rating, games_played, board_type, num_players
        FROM elo_ratings
        WHERE participant_id = ?
          AND harness_type IS NOT NULL
        ORDER BY rating DESC
    """

    cursor = conn.execute(query, (model_id,))
    results = {}
    for row in cursor.fetchall():
        harness = row[0] or "unknown"
        results[harness] = {
            "rating": row[1],
            "games_played": row[2],
            "board_type": row[3],
            "num_players": row[4],
        }

    conn.close()
    return results


def get_harness_distribution(
    db_path: Path | None = None,
    board_type: str | None = None,
    num_players: int | None = None,
) -> dict[str, int]:
    """Get count of rated models by harness type.

    Args:
        db_path: Path to ELO database
        board_type: Optional board type filter
        num_players: Optional player count filter

    Returns:
        Dict mapping harness_type to count of models

    Example:
        >>> dist = get_harness_distribution(board_type="hex8", num_players=2)
        >>> dist
        {'gumbel_mcts': 45, 'minimax': 12, 'policy_only': 8, 'heuristic': 5}
    """
    conn = _get_connection(db_path)
    if not conn:
        return {}

    query = """
        SELECT harness_type, COUNT(*) as count
        FROM elo_ratings
        WHERE harness_type IS NOT NULL
    """
    params: list[Any] = []

    if board_type:
        query += " AND board_type = ?"
        params.append(board_type)

    if num_players:
        query += " AND num_players = ?"
        params.append(num_players)

    query += " GROUP BY harness_type ORDER BY count DESC"

    cursor = conn.execute(query, params)
    results = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()
    return results


# Re-export thresholds for convenience
__all__ = [
    # Query functions
    "get_production_candidates",
    "get_top_models",
    "get_all_ratings",
    "get_model_stats",
    "get_games_by_config",
    "get_games_last_n_hours",
    "get_last_game_timestamp",
    "get_near_production",
    "get_models_by_tier",
    # Harness-based queries (January 2026)
    "get_ratings_by_harness",
    "get_top_models_by_harness",
    "get_harness_performance_comparison",
    "get_harness_distribution",
    # Helper functions
    "get_tier_name",
    "get_tier_abbr",
    # Data classes
    "ModelRating",
    "ModelStats",
    # Thresholds (re-exported for convenience)
    "INITIAL_ELO_RATING",
    "PRODUCTION_ELO_THRESHOLD",
    "PRODUCTION_MIN_GAMES",
    "TIER_THRESHOLDS",
    "DEFAULT_DB",
]
