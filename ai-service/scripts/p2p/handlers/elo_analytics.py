"""Elo Analytics HTTP handlers for P2P orchestrator.

January 2026 - P2P Modularization Phase 4a

This mixin provides HTTP handlers for Elo rating analytics,
leaderboards, and historical rating visualization.

Must be mixed into a class that provides:
- self._get_ai_service_path() -> str
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Import safe_db_connection for connection leak prevention
try:
    from app.db.safe_connection import safe_db_connection
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def safe_db_connection(db_path):
        """Fallback context manager for SQLite connections."""
        conn = sqlite3.connect(str(db_path))
        try:
            yield conn
        finally:
            conn.close()


class EloAnalyticsHandlersMixin:
    """Mixin providing Elo analytics HTTP handlers.

    Endpoints:
    - GET /api/elo-leaderboard - Get Elo leaderboard for all board types
    - GET /elo/history - Historical Elo ratings for time series visualization
    """

    async def handle_api_elo_leaderboard(self, request: web.Request) -> web.Response:
        """Get Elo leaderboard for all board types from persistent database.

        Query params:
            board_type: Filter by board type (optional)
            num_players: Filter by number of players (optional)
            limit: Max results per config (default 20)
        """
        try:
            # Try to import Elo database functions
            try:
                from scripts.run_model_elo_tournament import (
                    ELO_DB_PATH,
                    get_leaderboard,
                    init_elo_database,
                )
            except ImportError:
                return web.json_response({
                    "success": False,
                    "error": "Elo database module not available",
                }, status=500)

            # Check if database exists
            if not ELO_DB_PATH or not ELO_DB_PATH.exists():
                return web.json_response({
                    "success": True,
                    "leaderboards": {},
                    "message": "No Elo database found yet. Run cross-model tournament to populate.",
                })

            board_type = request.query.get("board_type")
            num_players_str = request.query.get("num_players")
            num_players = int(num_players_str) if num_players_str else None
            limit = int(request.query.get("limit", "20"))

            db = init_elo_database()

            # If specific filter requested, return just that
            if board_type and num_players:
                leaderboard = get_leaderboard(db, board_type, num_players, limit=limit)
                db.close()
                return web.json_response({
                    "success": True,
                    "leaderboards": {f"{board_type}_{num_players}p": leaderboard},
                    "total_models": len(leaderboard),
                    "timestamp": time.time(),
                })

            # Otherwise return all board/player combinations
            # Query unique board_type/num_players combinations
            conn = db._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT board_type, num_players
                FROM elo_ratings
                WHERE board_type IS NOT NULL AND num_players IS NOT NULL
                ORDER BY board_type, num_players
            """)
            configs = cursor.fetchall()

            leaderboards = {}
            total_models = 0
            total_games = 0

            for bt, np in configs:
                key = f"{bt}_{np}p"
                lb = get_leaderboard(db, bt, np, limit=limit)
                if lb:
                    leaderboards[key] = lb
                    total_models += len(lb)
                    total_games += sum(entry.get("games_played", 0) for entry in lb)

            # Get match history stats
            cursor.execute("SELECT COUNT(*) FROM match_history")
            match_count = cursor.fetchone()[0]

            db.close()

            return web.json_response({
                "success": True,
                "leaderboards": leaderboards,
                "total_models": total_models,
                "total_matches": match_count,
                "total_games_recorded": total_games,
                "configs": [f"{bt}_{np}p" for bt, np in configs],
                "timestamp": time.time(),
            })

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_elo_history(self, request: web.Request) -> web.Response:
        """GET /elo/history - Historical Elo ratings for time series visualization.

        Query params:
            - config: Filter by config (e.g., square8_2p)
            - model: Filter by model/participant_id (supports partial match)
            - nn_only: If "true", filter to NN models only
            - hours: Hours of history (default 168 = 1 week)
            - limit: Max entries to return (default 5000)
        """
        try:
            config_filter = request.query.get("config")
            model_filter = request.query.get("model")
            nn_only = request.query.get("nn_only", "").lower() == "true"
            hours = int(request.query.get("hours", "168"))
            limit = int(request.query.get("limit", "5000"))

            ai_root = Path(self._get_ai_service_path())

            # Canonical Elo database for trained models
            db_paths = [
                ai_root / "data" / "unified_elo.db",
            ]

            data = []
            cutoff = time.time() - (hours * 3600)

            for db_path in db_paths:
                if not db_path.exists():
                    continue

                try:
                    # Phase 3.4 Dec 29, 2025: Use context manager to prevent connection leaks
                    with safe_db_connection(db_path) as conn:
                        cursor = conn.cursor()

                        # Check if this DB has data
                        cursor.execute("SELECT COUNT(*) FROM rating_history WHERE timestamp > ?", (cutoff,))
                        count = cursor.fetchone()[0]
                        if count == 0:
                            continue

                        # Build query - unified_elo.db has different schema (no board_type/num_players)
                        cursor.execute("PRAGMA table_info(rating_history)")
                        columns = {col[1] for col in cursor.fetchall()}

                        if "board_type" in columns:
                            # unified_elo.db schema
                            query = """
                                SELECT participant_id, board_type, num_players, rating, games_played, timestamp
                                FROM rating_history
                                WHERE timestamp > ?
                            """
                            params = [cutoff]

                            if config_filter:
                                parts = config_filter.replace("_", " ").split()
                                if len(parts) >= 2:
                                    board_type = parts[0]
                                    num_players = int(parts[1].replace("p", ""))
                                    query += " AND board_type = ? AND num_players = ?"
                                    params.extend([board_type, num_players])
                        else:
                            # unified_elo.db schema (model_id instead of participant_id)
                            query = """
                                SELECT model_id, rating, games_played, timestamp
                                FROM rating_history
                                WHERE timestamp > ?
                            """
                            params = [cutoff]

                        if model_filter:
                            col = "participant_id" if "participant_id" in columns else "model_id"
                            query += f" AND {col} LIKE ?"
                            params.append(f"%{model_filter}%")

                        if nn_only:
                            col = "participant_id" if "participant_id" in columns else "model_id"
                            query += f" AND ({col} LIKE '%nn%' OR {col} LIKE '%NN%')"

                        query += f" ORDER BY timestamp DESC LIMIT {limit}"

                        cursor.execute(query, params)
                        rows = cursor.fetchall()

                    # Format for Grafana time series
                    for row in rows:
                        if "board_type" in columns:
                            participant_id, board_type, num_players, rating, games_played, ts = row
                            config = f"{board_type}_{num_players}p"
                        else:
                            model_id, rating, games_played, ts = row
                            participant_id = model_id
                            # Extract config from model name (e.g., sq8_2p_nn_baseline -> square8_2p)
                            if "sq8" in model_id.lower() or "square8" in model_id.lower():
                                config = "square8_2p"
                            elif "sq19" in model_id.lower() or "square19" in model_id.lower():
                                config = "square19_2p"
                            else:
                                config = "unknown"

                        data.append({
                            "time": int(ts * 1000),  # Grafana expects ms
                            "model": participant_id,
                            "config": config,
                            "elo": round(rating, 1),
                            "games": games_played,
                        })

                    # If we got data from this DB, don't check others
                    if data:
                        break

                except sqlite3.Error:
                    continue

            # Sort by time ascending for time series
            data.sort(key=lambda x: x["time"])

            return web.json_response(data)

        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])
