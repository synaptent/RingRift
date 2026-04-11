"""Batch and legacy replay query helpers for GameReplayDB."""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from pydantic import ValidationError

from app.models import BoardType, GameState, Move

from .replay_serialization import (
    _decompress_json,
    _deserialize_move,
    _deserialize_state,
    _move_from_columns,
)

logger = logging.getLogger("app.db.game_replay")


class ReplayBatchQueriesMixin:
    """Batch-access helpers shared by GameReplayDB."""

    def get_initial_states_batch(
        self,
        game_ids: list[str],
    ) -> dict[str, GameState | None]:
        """Get initial states for multiple games in a single query."""
        if not game_ids:
            return {}

        results: dict[str, GameState | None] = dict.fromkeys(game_ids)
        with self._get_conn() as conn:
            has_table = (
                conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='game_initial_state'"
                ).fetchone()
                is not None
            )

            if has_table:
                placeholders = ",".join("?" * len(game_ids))
                cols = [
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(game_initial_state)"
                    ).fetchall()
                ]
                has_compressed = "compressed" in cols
                json_col = (
                    "initial_state_json"
                    if "initial_state_json" in cols
                    else "state_json"
                )

                if has_compressed:
                    rows = conn.execute(
                        f"SELECT game_id, {json_col}, compressed FROM game_initial_state WHERE game_id IN ({placeholders})",
                        game_ids,
                    ).fetchall()
                    for row in rows:
                        json_str = row[json_col]
                        if row["compressed"]:
                            json_str = _decompress_json(json_str)
                        results[row["game_id"]] = _deserialize_state(json_str)
                else:
                    rows = conn.execute(
                        f"SELECT game_id, {json_col} FROM game_initial_state WHERE game_id IN ({placeholders})",
                        game_ids,
                    ).fetchall()
                    for row in rows:
                        results[row["game_id"]] = _deserialize_state(row[json_col])

            missing = [game_id for game_id, state in results.items() if state is None]
            if missing:
                placeholders = ",".join("?" * len(missing))
                meta_rows = conn.execute(
                    f"""
                    SELECT game_id, board_type, num_players
                    FROM games
                    WHERE game_id IN ({placeholders})
                    """,
                    missing,
                ).fetchall()

                from app.training.initial_state import create_initial_state

                for row in meta_rows:
                    board_type_str = row["board_type"]
                    board_type = (
                        BoardType(board_type_str)
                        if board_type_str
                        else BoardType.SQUARE8
                    )
                    results[row["game_id"]] = create_initial_state(
                        board_type=board_type,
                        num_players=row["num_players"],
                    )

        return results

    def get_moves_batch(self, game_ids: list[str]) -> dict[str, list[Move]]:
        """Get moves for multiple games in a single query."""
        if not game_ids:
            return {}

        results: dict[str, list[Move]] = {game_id: [] for game_id in game_ids}
        with self._get_conn() as conn:
            placeholders = ",".join("?" * len(game_ids))
            if self._has_move_json_column:
                rows = conn.execute(
                    f"""
                    SELECT game_id, move_json
                    FROM game_moves
                    WHERE game_id IN ({placeholders})
                    ORDER BY game_id, move_number
                    """,
                    game_ids,
                ).fetchall()
                for row in rows:
                    results[row["game_id"]].append(_deserialize_move(row["move_json"]))
            elif self._has_column_schema:
                rows = conn.execute(
                    f"""
                    SELECT game_id, move_number, player, position_q, position_r, move_type
                    FROM game_moves
                    WHERE game_id IN ({placeholders})
                    ORDER BY game_id, move_number
                    """,
                    game_ids,
                ).fetchall()

                failed_games: set[str] = set()
                for row in rows:
                    game_id = row["game_id"]
                    if game_id in failed_games:
                        continue
                    move = _move_from_columns(row)
                    if move is not None:
                        results[game_id].append(move)
                    else:
                        results[game_id] = []
                        failed_games.add(game_id)

            games_without_moves = [game_id for game_id in game_ids if not results[game_id]]
            if games_without_moves:
                self._get_moves_batch_from_inline_json(conn, games_without_moves, results)

        return results

    def _get_moves_batch_from_inline_json(
        self,
        conn: sqlite3.Connection,
        game_ids: list[str],
        results: dict[str, list[Move]],
    ) -> None:
        """Fallback: get moves from inline JSON for multiple games."""
        if "moves" not in (getattr(self, "_schema_columns", None) or set()):
            return

        placeholders = ",".join("?" * len(game_ids))
        rows = conn.execute(
            f"""
            SELECT game_id, moves
            FROM games
            WHERE game_id IN ({placeholders})
              AND moves IS NOT NULL
              AND LENGTH(moves) > 10
            """,
            game_ids,
        ).fetchall()

        for row in rows:
            game_id = row["game_id"]
            try:
                moves_data = json.loads(row["moves"])
                if not isinstance(moves_data, list):
                    continue
                for move_dict in moves_data:
                    if isinstance(move_dict, str):
                        results[game_id].append(_deserialize_move(move_dict))
                    elif isinstance(move_dict, dict):
                        results[game_id].append(Move.model_validate(move_dict))
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.debug("Failed to parse inline moves for game %s: %s", game_id, exc)

    def get_move_probs_batch(
        self,
        game_ids: list[str],
    ) -> dict[str, dict[int, dict[str, float]]]:
        """Get soft policy targets for multiple games in a single query."""
        if not game_ids:
            return {}

        results: dict[str, dict[int, dict[str, float]]] = {}
        with self._get_conn() as conn:
            placeholders = ",".join("?" * len(game_ids))
            rows = conn.execute(
                f"""
                SELECT game_id, move_number, move_probs
                FROM game_moves
                WHERE game_id IN ({placeholders})
                  AND move_probs IS NOT NULL
                ORDER BY game_id, move_number
                """,
                game_ids,
            ).fetchall()
            for row in rows:
                move_probs_json = row["move_probs"]
                if not move_probs_json:
                    continue
                try:
                    move_probs = json.loads(move_probs_json)
                    results.setdefault(row["game_id"], {})[row["move_number"]] = move_probs
                except (json.JSONDecodeError, TypeError):
                    pass

        return results

    def get_move_records(
        self,
        game_id: str,
        start: int = 0,
        end: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get move records with full metadata including v2 fields."""
        with self._get_conn() as conn:
            if end is None:
                rows = conn.execute(
                    """
                    SELECT move_number, turn_number, player, phase, move_type, move_json,
                           timestamp, think_time_ms, time_remaining_ms, engine_eval,
                           engine_eval_type, engine_depth, engine_nodes, engine_pv,
                           engine_time_ms
                    FROM game_moves
                    WHERE game_id = ? AND move_number >= ?
                    ORDER BY move_number
                    """,
                    (game_id, start),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT move_number, turn_number, player, phase, move_type, move_json,
                           timestamp, think_time_ms, time_remaining_ms, engine_eval,
                           engine_eval_type, engine_depth, engine_nodes, engine_pv,
                           engine_time_ms
                    FROM game_moves
                    WHERE game_id = ? AND move_number >= ? AND move_number < ?
                    ORDER BY move_number
                    """,
                    (game_id, start, end),
                ).fetchall()

        result = []
        for row in rows:
            result.append(
                {
                    "moveNumber": row["move_number"],
                    "turnNumber": row["turn_number"],
                    "player": row["player"],
                    "phase": row["phase"],
                    "moveType": row["move_type"],
                    "move": json.loads(row["move_json"]),
                    "timestamp": row["timestamp"],
                    "thinkTimeMs": row["think_time_ms"],
                    "timeRemainingMs": row["time_remaining_ms"],
                    "engineEval": row["engine_eval"],
                    "engineEvalType": row["engine_eval_type"],
                    "engineDepth": row["engine_depth"],
                    "engineNodes": row["engine_nodes"],
                    "enginePV": json.loads(row["engine_pv"]) if row["engine_pv"] else None,
                    "engineTimeMs": row["engine_time_ms"],
                }
            )
        return result
