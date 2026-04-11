"""Query, reporting, and event-emission methods for the Elo service."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config.thresholds import get_pinned_baseline_rating
from app.training.elo_types import LeaderboardEntry, TrainingFeedback

logger = logging.getLogger(__name__)

try:
    from app.distributed.data_events import emit_elo_updated, emit_elo_updated_sync

    HAS_ELO_EVENTS = True
except ImportError:
    HAS_ELO_EVENTS = False
    emit_elo_updated = None
    emit_elo_updated_sync = None

try:
    from app.distributed.data_events import DataEventType, emit_data_event

    HAS_ELO_VELOCITY_EVENTS = True
except ImportError:
    HAS_ELO_VELOCITY_EVENTS = False
    emit_data_event = None  # type: ignore
    DataEventType = None  # type: ignore

try:
    from app.training.composite_participant import (
        is_composite_id,
        parse_composite_participant_id,
    )
    from app.training.event_integration import publish_composite_elo_updated_sync

    HAS_COMPOSITE_EVENTS = True
except ImportError:
    HAS_COMPOSITE_EVENTS = False
    publish_composite_elo_updated_sync = None
    is_composite_id = None
    parse_composite_participant_id = None

ELO_SIGNIFICANT_CHANGE_THRESHOLD = 25.0
ELO_VELOCITY_THRESHOLD_PER_HOUR = 50.0


class EloReportingMixin:
    """Read-heavy and event-heavy Elo service helpers."""

    def _emit_elo_events(
        self,
        participant_a: str,
        participant_b: str,
        elo_before: dict[str, float],
        elo_after: dict[str, float],
        elo_changes: dict[str, float],
        board_type: str,
        num_players: int,
        duration_sec: float,
    ) -> None:
        config_key = f"{board_type}_{num_players}p"

        if HAS_ELO_EVENTS:
            for pid, old_elo, new_elo in [
                (participant_a, elo_before[participant_a], elo_after[participant_a]),
                (participant_b, elo_before[participant_b], elo_after[participant_b]),
            ]:
                try:
                    asyncio.get_running_loop()
                    if emit_elo_updated is not None:
                        asyncio.ensure_future(
                            emit_elo_updated(
                                config=config_key,
                                model_id=pid,
                                new_elo=new_elo,
                                old_elo=old_elo,
                                games_played=1,
                                source="elo_service",
                            )
                        )
                except RuntimeError:
                    if emit_elo_updated_sync is not None:
                        emit_elo_updated_sync(
                            config=config_key,
                            model_id=pid,
                            new_elo=new_elo,
                            old_elo=old_elo,
                            games_played=1,
                            source="elo_service",
                        )
                except (AttributeError, TypeError, ValueError):
                    pass

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
                        pass

        if HAS_ELO_VELOCITY_EVENTS and emit_data_event is not None:
            try:
                for pid, old_elo, new_elo in [
                    (participant_a, elo_before[participant_a], elo_after[participant_a]),
                    (participant_b, elo_before[participant_b], elo_after[participant_b]),
                ]:
                    elo_delta = new_elo - old_elo
                    if abs(elo_delta) > ELO_SIGNIFICANT_CHANGE_THRESHOLD:
                        try:
                            asyncio.get_running_loop()
                            asyncio.ensure_future(
                                emit_data_event(
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
                                )
                            )
                        except RuntimeError:
                            pass

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
                                asyncio.ensure_future(
                                    emit_data_event(
                                        event_type=DataEventType.ELO_VELOCITY_CHANGED,
                                        payload={
                                            "config_key": config_key,
                                            "board_type": board_type,
                                            "num_players": num_players,
                                            "participant_id": pid,
                                            "velocity": elo_per_hour,
                                            "trend": "improving" if elo_per_hour > 0 else "declining",
                                        },
                                    )
                                )
                            except RuntimeError:
                                pass
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass

    def get_leaderboard(
        self,
        board_type: str,
        num_players: int,
        limit: int = 50,
        min_games: int = 0,
    ) -> list[LeaderboardEntry]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT
                e.participant_id,
                COALESCE(p.participant_id, e.participant_id) AS name,
                COALESCE(p.ai_type, 'unknown') AS ai_type,
                e.rating,
                e.games_played,
                e.wins,
                e.losses,
                e.draws,
                e.last_update
            FROM elo_ratings e
            LEFT JOIN participants p ON e.participant_id = p.participant_id
            WHERE e.board_type = ? AND e.num_players = ? AND e.games_played >= ?
            ORDER BY e.rating DESC
            LIMIT ?
            """,
            (board_type, num_players, min_games, limit),
        )

        entries: list[LeaderboardEntry] = []
        for rank, row in enumerate(cursor.fetchall(), 1):
            games = row["games_played"]
            win_rate = (row["wins"] + 0.5 * row["draws"]) / games if games > 0 else 0.5
            entries.append(
                LeaderboardEntry(
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
                    last_active=datetime.fromtimestamp(row["last_update"] or 0).isoformat(),
                )
            )
        return entries

    def log_leaderboard(self, board_type: str, num_players: int, top_n: int = 5) -> list[LeaderboardEntry]:
        config_key = f"{board_type}_{num_players}p"
        entries = self.get_leaderboard(board_type, num_players, limit=top_n, min_games=1)
        if not entries:
            logger.info("[Elo] Leaderboard for %s: (no entries with games)", config_key)
            return entries
        logger.info("[Elo] Leaderboard for %s:", config_key)
        for entry in entries:
            model_name = entry.name[:30] if len(entry.name) > 30 else entry.name
            logger.info(
                "  #%s: %s - Elo %.0f (%sW/%sL, %.0f%%)",
                entry.rank,
                model_name,
                entry.rating,
                entry.wins,
                entry.losses,
                entry.win_rate * 100,
            )
        return entries

    def get_training_feedback(
        self,
        board_type: str,
        num_players: int,
        lookback_iterations: int = 5,
    ) -> TrainingFeedback:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT MAX(rating) as best_rating, MAX(peak_rating) as peak
            FROM elo_ratings
            WHERE board_type = ? AND num_players = ?
            """,
            (board_type, num_players),
        )
        row = cursor.fetchone()
        best_elo = row["best_rating"] or self.INITIAL_ELO

        cursor = conn.execute(
            """
            SELECT rating, iteration FROM elo_history
            WHERE board_type = ? AND num_players = ?
            ORDER BY iteration DESC
            LIMIT ?
            """,
            (board_type, num_players, lookback_iterations),
        )
        history = cursor.fetchall()

        recent_elo_delta = 0.0
        elo_stagnating = False
        elo_declining = False
        if len(history) >= 2:
            recent_elo_delta = history[0]["rating"] - history[-1]["rating"]
            elo_stagnating = abs(recent_elo_delta) < 10
            elo_declining = recent_elo_delta < -20

        epochs_multiplier = 1.0
        lr_multiplier = 1.0
        exploration_boost = 0.0
        curriculum_stage = 0
        if elo_stagnating:
            epochs_multiplier = 1.5
            lr_multiplier = 0.8
            exploration_boost = 0.1
        if elo_declining:
            epochs_multiplier = 2.0
            lr_multiplier = 0.5
            exploration_boost = 0.2
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
            recommended_curriculum_stage=curriculum_stage,
        )
        for callback in self._feedback_callbacks:
            with suppress(Exception):
                callback(feedback)
        return feedback

    def record_training_feedback(
        self,
        board_type: str,
        num_players: int,
        iteration: int,
        best_elo: float,
        elo_delta: float,
        epochs_multiplier: float = 1.0,
        lr_multiplier: float = 1.0,
        curriculum_stage: int = 0,
    ) -> bool:
        try:
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO training_feedback
                    (board_type, num_players, iteration, best_elo, elo_delta,
                     epochs_multiplier, lr_multiplier, curriculum_stage, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        board_type,
                        num_players,
                        iteration,
                        best_elo,
                        elo_delta,
                        epochs_multiplier,
                        lr_multiplier,
                        curriculum_stage,
                        time.time(),
                    ),
                )
            logger.debug(
                "Recorded training feedback: %s_%sp iter=%s elo=%.0f delta=%+.0f",
                board_type,
                num_players,
                iteration,
                best_elo,
                elo_delta,
            )
            return True
        except sqlite3.Error as exc:
            logger.error("Failed to record training feedback: %s", exc)
            return False

    def record_iteration_elo(self, participant_id: str, board_type: str, num_players: int, iteration: int) -> None:
        rating = self.get_rating(participant_id, board_type, num_players)
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO elo_history
                (participant_id, board_type, num_players, rating, timestamp, iteration)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (participant_id, board_type, num_players, rating.rating, time.time(), iteration),
            )

    def _record_elo_snapshot(
        self,
        participant_id: str,
        board_type: str,
        num_players: int,
        min_interval_seconds: float = 300.0,
    ) -> None:
        if get_pinned_baseline_rating(participant_id) is not None:
            return
        current_rating = self.get_rating(participant_id, board_type, num_players)
        if not current_rating or current_rating.games_played == 0:
            return

        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT MAX(timestamp) FROM elo_history
            WHERE participant_id = ? AND board_type = ? AND num_players = ?
            """,
            (participant_id, board_type, num_players),
        )
        row = cursor.fetchone()
        last_snapshot = row[0] if row and row[0] else 0.0
        if time.time() - last_snapshot < min_interval_seconds:
            return
        with self._transaction() as txn_conn:
            txn_conn.execute(
                """
                INSERT INTO elo_history
                (participant_id, board_type, num_players, rating, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (participant_id, board_type, num_players, current_rating.rating, time.time()),
            )

    def register_feedback_callback(self, callback: Callable[[TrainingFeedback], None]) -> None:
        self._feedback_callbacks.append(callback)

    def get_win_rate_vs_baseline(
        self,
        model_id: str,
        baseline_id: str,
        board_type: str,
        num_players: int,
    ) -> tuple[float, int]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT winner_id FROM match_history
            WHERE board_type = ? AND num_players = ?
            AND participant_ids LIKE ? AND participant_ids LIKE ?
            """,
            (board_type, num_players, f"%{model_id}%", f"%{baseline_id}%"),
        )
        wins = 0
        total = 0
        for row in cursor:
            total += 1
            if row["winner_id"] == model_id:
                wins += 1
            elif row["winner_id"] is None:
                wins += 0.5
        return (wins / total if total > 0 else 0.5), total

    def get_nn_performance_summary(self, nn_model_id: str, board_type: str, num_players: int) -> dict[str, Any] | None:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT best_algorithm, best_elo, avg_elo, algorithms_tested, last_updated
            FROM nn_performance_summary
            WHERE nn_model_id = ? AND board_type = ? AND num_players = ?
            """,
            (nn_model_id, board_type, num_players),
        )
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

    def get_algorithm_baseline(self, ai_algorithm: str, board_type: str, num_players: int) -> dict[str, Any] | None:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT baseline_elo, games_played, last_updated
            FROM algorithm_baselines
            WHERE ai_algorithm = ? AND board_type = ? AND num_players = ?
            """,
            (ai_algorithm, board_type, num_players),
        )
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
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO algorithm_baselines
                (ai_algorithm, board_type, num_players, baseline_elo, games_played, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ai_algorithm, board_type, num_players, baseline_elo, games_played, time.time()),
            )

    def get_composite_leaderboard(
        self,
        board_type: str,
        num_players: int,
        ai_algorithm: str | None = None,
        nn_model_id: str | None = None,
        limit: int = 50,
        min_games: int = 0,
    ) -> list[dict[str, Any]]:
        conn = self._get_connection()
        query = """
            SELECT
                e.participant_id,
                COALESCE(p.nn_model_id, e.participant_id) AS nn_model_id,
                COALESCE(p.ai_algorithm, 'unknown') AS ai_algorithm,
                e.rating,
                e.games_played,
                e.wins,
                e.losses,
                e.draws,
                e.last_update
            FROM elo_ratings e
            LEFT JOIN participants p ON e.participant_id = p.participant_id
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
            entries.append(
                {
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
                }
            )
        return entries

    def get_algorithm_rankings(self, board_type: str, num_players: int, min_games: int = 10) -> list[dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
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
            """,
            (board_type, num_players, min_games),
        )
        rankings = []
        for rank, row in enumerate(cursor.fetchall(), 1):
            rankings.append(
                {
                    "rank": rank,
                    "ai_algorithm": row["ai_algorithm"],
                    "avg_elo": row["avg_elo"],
                    "best_elo": row["best_elo"],
                    "worst_elo": row["worst_elo"],
                    "nn_count": row["nn_count"],
                    "total_games": row["total_games"],
                    "elo_spread": row["best_elo"] - row["worst_elo"],
                }
            )
        return rankings

    def get_nn_rankings(self, board_type: str, num_players: int, min_games: int = 10) -> list[dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
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
            """,
            (board_type, num_players, min_games),
        )
        rankings = []
        for rank, row in enumerate(cursor.fetchall(), 1):
            best_algo_cursor = conn.execute(
                """
                SELECT p.ai_algorithm
                FROM elo_ratings e
                JOIN participants p ON e.participant_id = p.participant_id
                WHERE p.nn_model_id = ? AND e.board_type = ? AND e.num_players = ?
                ORDER BY e.rating DESC
                LIMIT 1
                """,
                (row["nn_model_id"], board_type, num_players),
            )
            best_algo_row = best_algo_cursor.fetchone()
            rankings.append(
                {
                    "rank": rank,
                    "nn_model_id": row["nn_model_id"],
                    "best_elo": row["best_elo"],
                    "avg_elo": row["avg_elo"],
                    "best_algorithm": best_algo_row["ai_algorithm"] if best_algo_row else None,
                    "algorithm_count": row["algorithm_count"],
                    "total_games": row["total_games"],
                }
            )
        return rankings

    def get_unevaluated_models(
        self,
        models_directory: Path | str | None = None,
        include_subdirs: bool = True,
    ) -> list[dict[str, Any]]:
        if models_directory is None:
            from app.utils.paths import MODELS_DIR

            models_directory = MODELS_DIR

        models_dir = Path(models_directory)
        if not models_dir.exists():
            logger.warning("Models directory not found: %s", models_dir)
            return []

        model_files = list(models_dir.rglob("*.pth")) if include_subdirs else list(models_dir.glob("*.pth"))
        if not model_files:
            logger.debug("No .pth files found in %s", models_dir)
            return []

        board_types = {"hex8", "square8", "square19", "hexagonal"}
        unevaluated = []
        for model_path in model_files:
            model_name = model_path.stem.lower()
            board_type = next((bt for bt in board_types if bt in model_name), None)
            import re

            player_match = re.search(r"(\\d)p", model_name)
            num_players = int(player_match.group(1)) if player_match else None
            if not board_type or not num_players:
                logger.debug("Could not parse config from filename: %s", model_path.name)
                continue

            model_path_str = str(model_path)
            conn = self._get_connection()
            cursor = conn.execute(
                """
                SELECT p.participant_id, e.rating, e.games_played
                FROM participants p
                LEFT JOIN elo_ratings e ON p.participant_id = e.participant_id
                    AND e.board_type = ? AND e.num_players = ?
                WHERE p.model_path = ?
                """,
                (board_type, num_players, model_path_str),
            )
            row = cursor.fetchone()
            needs_evaluation = row is None or row["rating"] is None or (
                row["games_played"] is not None and row["games_played"] == 0
            )
            if needs_evaluation:
                stat = model_path.stat()
                unevaluated.append(
                    {
                        "model_path": model_path_str,
                        "board_type": board_type,
                        "num_players": num_players,
                        "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified_at": stat.st_mtime,
                    }
                )

        unevaluated.sort(key=lambda item: item["modified_at"], reverse=True)
        logger.info(
            "Found %s unevaluated models out of %s total",
            len(unevaluated),
            len(model_files),
        )
        return unevaluated


__all__ = ["EloReportingMixin"]
