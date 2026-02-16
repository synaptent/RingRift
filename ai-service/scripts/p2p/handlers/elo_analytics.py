"""Elo Analytics HTTP handlers for P2P orchestrator.

January 2026 - P2P Modularization Phase 4a

This mixin provides HTTP handlers for Elo rating analytics,
leaderboards, and historical rating visualization.

Must be mixed into a class that provides:
- self._get_ai_service_path() -> str
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

from scripts.p2p.db_helpers import p2p_db_connection as safe_db_connection

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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

            # Feb 2026: Run all synchronous SQLite queries in a thread to
            # prevent blocking the P2P event loop (was causing 54-137s blocks).
            def _query_leaderboard(bt_filter, np_filter, lim):
                db = init_elo_database()
                try:
                    if bt_filter and np_filter:
                        lb = get_leaderboard(db, bt_filter, np_filter, limit=lim)
                        return {
                            "leaderboards": {f"{bt_filter}_{np_filter}p": lb},
                            "total_models": len(lb),
                            "total_matches": 0,
                            "total_games_recorded": 0,
                            "configs": [f"{bt_filter}_{np_filter}p"],
                        }

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
                        lb = get_leaderboard(db, bt, np, limit=lim)
                        if lb:
                            leaderboards[key] = lb
                            total_models += len(lb)
                            total_games += sum(entry.get("games_played", 0) for entry in lb)

                    cursor.execute("SELECT COUNT(*) FROM match_history")
                    match_count = cursor.fetchone()[0]

                    return {
                        "leaderboards": leaderboards,
                        "total_models": total_models,
                        "total_matches": match_count,
                        "total_games_recorded": total_games,
                        "configs": [f"{bt}_{np}p" for bt, np in configs],
                    }
                finally:
                    db.close()

            result = await asyncio.to_thread(_query_leaderboard, board_type, num_players, limit)
            result["success"] = True
            result["timestamp"] = time.time()

            return web.json_response(result)

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

            # Feb 2026: Run all synchronous SQLite queries in a thread to
            # prevent blocking the P2P event loop (was causing 54-137s blocks).
            def _query_history(ai_root_str, cfg_filter, mdl_filter, nn_flag, hrs, lim):
                db_paths = [
                    Path(ai_root_str) / "data" / "unified_elo.db",
                ]
                data = []
                cutoff_ts = time.time() - (hrs * 3600)

                for db_path in db_paths:
                    if not db_path.exists():
                        continue
                    try:
                        with safe_db_connection(db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT COUNT(*) FROM rating_history WHERE timestamp > ?", (cutoff_ts,))
                            count = cursor.fetchone()[0]
                            if count == 0:
                                continue

                            cursor.execute("PRAGMA table_info(rating_history)")
                            columns = {col[1] for col in cursor.fetchall()}

                            if "board_type" in columns:
                                query = """
                                    SELECT participant_id, board_type, num_players, rating, games_played, timestamp
                                    FROM rating_history
                                    WHERE timestamp > ?
                                """
                                params = [cutoff_ts]
                                if cfg_filter:
                                    parts = cfg_filter.replace("_", " ").split()
                                    if len(parts) >= 2:
                                        bt = parts[0]
                                        npl = int(parts[1].replace("p", ""))
                                        query += " AND board_type = ? AND num_players = ?"
                                        params.extend([bt, npl])
                            else:
                                query = """
                                    SELECT model_id, rating, games_played, timestamp
                                    FROM rating_history
                                    WHERE timestamp > ?
                                """
                                params = [cutoff_ts]

                            if mdl_filter:
                                col = "participant_id" if "participant_id" in columns else "model_id"
                                query += f" AND {col} LIKE ?"
                                params.append(f"%{mdl_filter}%")
                            if nn_flag:
                                col = "participant_id" if "participant_id" in columns else "model_id"
                                query += f" AND ({col} LIKE '%nn%' OR {col} LIKE '%NN%')"

                            query += f" ORDER BY timestamp DESC LIMIT {lim}"
                            cursor.execute(query, params)
                            rows = cursor.fetchall()

                        for row in rows:
                            if "board_type" in columns:
                                participant_id, bt, npl, rating, games_played, ts = row
                                cfg = f"{bt}_{npl}p"
                            else:
                                model_id, rating, games_played, ts = row
                                participant_id = model_id
                                if "sq8" in model_id.lower() or "square8" in model_id.lower():
                                    cfg = "square8_2p"
                                elif "sq19" in model_id.lower() or "square19" in model_id.lower():
                                    cfg = "square19_2p"
                                else:
                                    cfg = "unknown"
                            data.append({
                                "time": int(ts * 1000),
                                "model": participant_id,
                                "config": cfg,
                                "elo": round(rating, 1),
                                "games": games_played,
                            })
                        if data:
                            break
                    except sqlite3.Error:
                        continue

                data.sort(key=lambda x: x["time"])
                return data

            data = await asyncio.to_thread(
                _query_history, str(ai_root), config_filter, model_filter, nn_only, hours, limit
            )

            return web.json_response(data)

        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])
