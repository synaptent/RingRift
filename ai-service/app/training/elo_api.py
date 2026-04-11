"""Legacy wrapper functions around the centralized Elo service."""

from __future__ import annotations

import json
import time
from contextlib import suppress
from dataclasses import asdict
from typing import Any


def _resolve_db(db: Any | None) -> Any:
    if db is not None:
        return db

    from app.training.elo_service import get_elo_service

    return get_elo_service()


def register_models(db: Any, models: list[dict[str, Any]]) -> None:
    """Register multiple models using the historical compatibility format."""
    service = _resolve_db(db)
    for model in models:
        model_id = model.get("participant_id") or model.get("model_id") or model.get("id")
        if not model_id:
            continue

        service.register_model(
            model_id=model_id,
            board_type=model.get("board_type", "square8"),
            num_players=model.get("num_players", 2),
            model_path=model.get("model_path"),
        )


def update_elo_after_match(
    db: Any,
    model_a_id: str,
    model_b_id: str,
    winner: str | None,
    board_type: str = "square8",
    num_players: int = 2,
    game_length: int = 0,
    duration_sec: float = 0.0,
    tournament_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    harness_type: str = "gumbel_mcts",
    is_multi_harness: bool = False,
) -> dict[str, Any]:
    """Backwards-compatible match recording wrapper."""
    service = _resolve_db(db)

    winner_id = None
    if winner not in (None, "draw"):
        if winner in (model_a_id, "model_a"):
            winner_id = model_a_id
        elif winner in (model_b_id, "model_b"):
            winner_id = model_b_id
        else:
            winner_id = winner

    result = service.record_match(
        participant_a=model_a_id,
        participant_b=model_b_id,
        winner=winner_id,
        board_type=board_type,
        num_players=num_players,
        game_length=game_length,
        duration_sec=duration_sec,
        tournament_id=tournament_id,
        metadata=metadata,
        harness_type=harness_type,
        is_multi_harness=is_multi_harness,
    )

    rating_a = service.get_rating(model_a_id, board_type, num_players)
    rating_b = service.get_rating(model_b_id, board_type, num_players)

    return {
        "model_a": rating_a.rating,
        "model_b": rating_b.rating,
        "changes": result.elo_changes,
        "match_id": result.match_id,
    }


def get_leaderboard(
    db: Any,
    board_type: str = "square8",
    num_players: int = 2,
    limit: int = 50,
    min_games: int = 0,
) -> list[dict[str, Any]]:
    """Return leaderboard rows as dictionaries for older callers."""
    service = _resolve_db(db)
    entries = service.get_leaderboard(
        board_type=board_type,
        num_players=num_players,
        limit=limit,
        min_games=min_games,
    )
    return [asdict(entry) for entry in entries]


def get_head_to_head(
    db: Any,
    participant_a: str,
    participant_b: str,
    board_type: str | None = None,
    num_players: int | None = None,
) -> dict[str, Any]:
    """Get head-to-head stats between two participants."""
    service = _resolve_db(db)
    conn = service._get_connection()

    query = """
        SELECT winner_id FROM match_history
        WHERE participant_ids LIKE ? AND participant_ids LIKE ?
    """
    params: list[Any] = [f"%{participant_a}%", f"%{participant_b}%"]

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


def get_database_stats(db: Any) -> dict[str, Any]:
    """Return the historical summary shape used by scripts and tests."""
    service = _resolve_db(db)
    conn = service._get_connection()

    participant_count = conn.execute("SELECT COUNT(*) FROM participants").fetchone()[0]
    rating_count = conn.execute(
        "SELECT COUNT(*) FROM elo_ratings WHERE games_played > 0"
    ).fetchone()[0]
    match_count = conn.execute("SELECT COUNT(*) FROM match_history").fetchone()[0]
    configs = conn.execute(
        """
        SELECT board_type, num_players, COUNT(*) as count, MAX(rating) as top_rating
        FROM elo_ratings
        WHERE games_played > 0
        GROUP BY board_type, num_players
        """
    ).fetchall()

    return {
        "total_participants": participant_count,
        "rated_participants": rating_count,
        "total_matches": match_count,
        "configurations": [dict(config) for config in configs],
    }


def get_match_history(
    db: Any,
    participant_id: str | None = None,
    tournament_id: str | None = None,
    board_type: str | None = None,
    num_players: int | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get match history with optional filters."""
    service = _resolve_db(db)
    conn = service._get_connection()

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
    results: list[dict[str, Any]] = []
    for row in cursor.fetchall():
        result = dict(row)
        if result.get("participant_ids"):
            with suppress(json.JSONDecodeError, TypeError):
                result["participant_ids"] = json.loads(result["participant_ids"])
        if result.get("elo_before"):
            with suppress(json.JSONDecodeError, TypeError):
                result["elo_before"] = json.loads(result["elo_before"])
        if result.get("elo_after"):
            with suppress(json.JSONDecodeError, TypeError):
                result["elo_after"] = json.loads(result["elo_after"])
        results.append(result)
    return results


def get_rating_history(
    db: Any,
    participant_id: str,
    board_type: str,
    num_players: int,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get rating history for a participant in a specific config."""
    service = _resolve_db(db)
    conn = service._get_connection()
    cursor = conn.execute(
        """
        SELECT * FROM elo_history
        WHERE participant_id = ? AND board_type = ? AND num_players = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (participant_id, board_type, num_players, limit),
    )
    return [dict(row) for row in cursor.fetchall()]


def get_elo_trend(
    db: Any,
    participant_id: str,
    board_type: str,
    num_players: int,
    hours: int = 48,
    min_samples: int = 3,
) -> dict[str, Any]:
    """Get Elo trend over the last N hours."""
    service = _resolve_db(db)
    cutoff_time = time.time() - (hours * 3600)
    conn = service._get_connection()
    cursor = conn.execute(
        """
        SELECT rating, timestamp FROM elo_history
        WHERE participant_id = ? AND board_type = ? AND num_players = ?
          AND timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (participant_id, board_type, num_players, cutoff_time),
    )
    rows = cursor.fetchall()

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

    ratings = [row["rating"] for row in rows]
    timestamps = [row["timestamp"] for row in rows]
    start_elo = ratings[0]
    end_elo = ratings[-1]
    duration_seconds = timestamps[-1] - timestamps[0]
    duration_hours = duration_seconds / 3600.0
    if duration_hours < 0.1:
        return result

    total_change = end_elo - start_elo
    slope = total_change / duration_hours if duration_hours > 0 else 0.0
    mean_rating = sum(ratings) / len(ratings)
    ss_tot = sum((rating - mean_rating) ** 2 for rating in ratings)
    if ss_tot > 0 and len(ratings) >= 3:
        predicted = [
            start_elo + slope * ((timestamp - timestamps[0]) / 3600.0)
            for timestamp in timestamps
        ]
        ss_res = sum((rating - pred) ** 2 for rating, pred in zip(ratings, predicted))
        r_squared = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))
    else:
        r_squared = 0.5

    sample_confidence = min(1.0, len(rows) / 20.0)
    confidence = (sample_confidence + r_squared) / 2.0

    result.update(
        {
            "slope": round(slope, 3),
            "start_elo": round(start_elo, 1),
            "end_elo": round(end_elo, 1),
            "total_change": round(total_change, 1),
            "duration_hours": round(duration_hours, 2),
            "sample_count": len(rows),
            "is_plateau": abs(slope) < 1.0 and len(rows) >= 5,
            "is_declining": slope < -1.0,
            "confidence": round(confidence, 3),
        }
    )
    return result


def get_elo_trend_for_config(
    config_key: str,
    hours: int = 48,
    participant_id: str = "canonical",
) -> dict[str, Any]:
    """Convenience wrapper matching the historical config-key API."""
    parts = config_key.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].endswith("p"):
        return {
            "slope": 0.0,
            "is_plateau": False,
            "confidence": 0.0,
            "error": f"Invalid config_key format: {config_key}",
        }

    board_type = parts[0]
    try:
        num_players = int(parts[1][:-1])
    except ValueError:
        return {
            "slope": 0.0,
            "is_plateau": False,
            "confidence": 0.0,
            "error": f"Invalid player count in config_key: {config_key}",
        }

    return get_elo_trend(
        None,
        participant_id=participant_id,
        board_type=board_type,
        num_players=num_players,
        hours=hours,
    )


__all__ = [
    "get_database_stats",
    "get_elo_trend",
    "get_elo_trend_for_config",
    "get_head_to_head",
    "get_leaderboard",
    "get_match_history",
    "get_rating_history",
    "register_models",
    "update_elo_after_match",
]
