"""Table Handlers Mixin for P2P Orchestrator.

December 2025 - Phase 8 decomposition.

Provides HTTP handlers for table-format endpoints used by Grafana Infinity.
These endpoints return flat JSON arrays suitable for table visualization.

Table Endpoints:
- GET /elo/table - Elo leaderboard
- GET /nodes/table - Node status table
- GET /holdout/table - Holdout validation metrics
- GET /mcts/table - MCTS search statistics
- GET /matchups/table - Head-to-head matchups
- GET /models/lineage/table - Model ancestry
- GET /data/quality/table - Data quality metrics
- GET /training/efficiency/table - Training efficiency
- GET /trends/table - Performance trends
- GET /abtest/table - A/B test status
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.db_helpers import p2p_db_connection
from .base import BaseP2PHandler
from .timeout_decorator import handler_timeout, HANDLER_TIMEOUT_GOSSIP

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TableHandlersMixin(BaseP2PHandler):
    """Mixin providing table-format HTTP handlers for Grafana visualization.

    All table handlers:
    - Return JSON arrays suitable for Grafana Infinity data source
    - Include error handling with error object in array
    - Use cached data fetch methods where available
    """

    # =========================================================================
    # Elo Table
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_elo_table(self, request: web.Request) -> web.Response:
        """GET /elo/table - Elo leaderboard in flat table format for Grafana Infinity.

        Query params:
            - source: "tournament" (default) or "trained" (actual trained NN models)
            - limit: Max entries (default 50)
            - board_type: Filter by board type
            - num_players: Filter by player count
            - nn_only: If "true", filter to NN models only (for tournament source)

        Returns a simple JSON array of model entries with rank, suitable for table display.
        """
        try:
            source = request.query.get("source", "tournament")
            limit = int(request.query.get("limit", "50"))
            board_type_filter = request.query.get("board_type")
            num_players_filter = request.query.get("num_players")
            nn_only = request.query.get("nn_only", "").lower() == "true"

            ai_root = Path(self.ringrift_path) / "ai-service"

            if source == "trained":
                return await self._handle_elo_table_trained(
                    ai_root, limit, nn_only
                )
            else:
                return await self._handle_elo_table_tournament(
                    limit, board_type_filter, num_players_filter, nn_only
                )

        except ImportError:
            return web.json_response([{"error": "Elo database module not available"}])
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    async def _handle_elo_table_trained(
        self,
        ai_root: Path,
        limit: int,
        nn_only: bool,
    ) -> web.Response:
        """Handle trained source for Elo table."""
        db_path = ai_root / "data" / "unified_elo.db"
        if not db_path.exists():
            return web.json_response([])

        def _fetch_elo_data() -> list[tuple[Any, ...]]:
            """Blocking SQLite query - runs in thread pool."""
            with p2p_db_connection(db_path) as conn:
                cursor = conn.cursor()

                query = """
                    SELECT model_id, rating, games_played, wins, losses
                    FROM elo_ratings
                    WHERE games_played >= 10
                """
                params: list[Any] = []

                if nn_only:
                    query += " AND (model_id LIKE '%nn%' OR model_id LIKE '%NN%' OR model_id LIKE '%baseline%')"

                query += " ORDER BY rating DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                return cursor.fetchall()

        # Run blocking SQLite in thread pool to avoid blocking event loop
        rows = await asyncio.to_thread(_fetch_elo_data)

        table_data = []
        for rank, row in enumerate(rows, 1):
            model_id, rating, games, wins, losses = row

            # Extract config from model name
            if "sq8" in model_id.lower() or "square8" in model_id.lower():
                config = "square8_2p"
            elif "sq19" in model_id.lower() or "square19" in model_id.lower():
                config = "square19_2p"
            elif "hex" in model_id.lower():
                config = "hexagonal_2p"
            else:
                config = "unknown"

            # Calculate win rate
            total_decided = wins + losses
            win_rate = wins / total_decided if total_decided > 0 else 0.5

            table_data.append({
                "Rank": rank,
                "Model": model_id,
                "Elo": round(rating, 1),
                "WinRate": round(win_rate * 100, 1),
                "Games": games,
                "Wins": wins,
                "Losses": losses,
                "Draws": 0,
                "Config": config,
            })

        return web.json_response(table_data)

    async def _handle_elo_table_tournament(
        self,
        limit: int,
        board_type_filter: str | None,
        num_players_filter: str | None,
        nn_only: bool,
    ) -> web.Response:
        """Handle tournament source for Elo table."""
        from scripts.run_model_elo_tournament import (
            ELO_DB_PATH,
            init_elo_database,
        )

        if not ELO_DB_PATH or not ELO_DB_PATH.exists():
            return web.json_response([])

        def _fetch_tournament_data() -> list[tuple[Any, ...]]:
            """Blocking SQLite query - runs in thread pool."""
            db = init_elo_database()
            conn = db._get_connection()
            cursor = conn.cursor()

            id_col = db.id_column

            query = f"""
                SELECT
                    {id_col},
                    board_type,
                    num_players,
                    rating,
                    games_played,
                    wins,
                    losses,
                    draws,
                    last_update
                FROM elo_ratings
                WHERE games_played >= 5
            """
            params: list[Any] = []

            if board_type_filter:
                query += " AND board_type = ?"
                params.append(board_type_filter)

            if num_players_filter:
                query += " AND num_players = ?"
                params.append(int(num_players_filter))

            if nn_only:
                query += f" AND ({id_col} LIKE '%NN%' OR {id_col} LIKE '%nn%')"

            query += " ORDER BY rating DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            db.close()
            return rows

        # Run blocking SQLite in thread pool to avoid blocking event loop
        rows = await asyncio.to_thread(_fetch_tournament_data)

        table_data = []
        for rank, row in enumerate(rows, 1):
            participant_id, board_type, num_players, rating, games, wins, losses, draws, _last_update = row

            # Extract model name from participant_id
            model_name = participant_id
            if participant_id.startswith("nn:"):
                model_name = Path(participant_id[3:]).stem

            # Calculate win rate
            total_decided = wins + losses
            win_rate = wins / total_decided if total_decided > 0 else 0.5

            # Format config
            config = f"{board_type}_{num_players}p"

            table_data.append({
                "Rank": rank,
                "Model": model_name,
                "Elo": round(rating, 1),
                "WinRate": round(win_rate * 100, 1),
                "Games": games,
                "Wins": wins,
                "Losses": losses,
                "Draws": draws,
                "Config": config,
            })

        return web.json_response(table_data)

    # =========================================================================
    # Holdout Table
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_holdout_table(self, request: web.Request) -> web.Response:
        """GET /holdout/table - Holdout validation data in table format for Grafana Infinity.

        Returns holdout metrics as flat table rows.
        """
        try:
            metrics = await self.analytics_cache_manager.get_holdout_metrics_cached()

            table_data = []
            for config, data in metrics.get("configs", {}).items():
                row = {
                    "Config": config,
                    "HoldoutGames": data.get("holdout_games", 0),
                    "HoldoutPositions": data.get("holdout_positions", 0),
                    "HoldoutLoss": round(data.get("holdout_loss", 0), 4) if data.get("holdout_loss") else None,
                    "HoldoutAccuracy": round(data.get("holdout_accuracy", 0) * 100, 1) if data.get("holdout_accuracy") else None,
                    "OverfitGap": round(data.get("overfit_gap", 0), 4) if data.get("overfit_gap") else None,
                    "Status": "OK" if (data.get("overfit_gap") or 0) < 0.15 else "OVERFITTING",
                }
                table_data.append(row)

            return web.json_response(table_data)

        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # MCTS Table
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_mcts_table(self, request: web.Request) -> web.Response:
        """GET /mcts/table - MCTS stats in table format for Grafana Infinity.

        Returns MCTS statistics as flat table rows.
        """
        try:
            stats = await self.analytics_cache_manager.get_mcts_stats_cached()

            table_data = []
            # Add summary row
            summary = stats.get("summary", {})
            if summary:
                table_data.append({
                    "Config": "CLUSTER AVERAGE",
                    "AvgNodes": round(summary.get("avg_nodes_per_move", 0), 0),
                    "MaxNodes": summary.get("max_nodes_per_move", 0),
                    "AvgDepth": round(summary.get("avg_search_depth", 0), 1),
                    "MaxDepth": summary.get("max_search_depth", 0),
                    "AvgTime": round(summary.get("avg_time_per_move", 0), 3) if summary.get("avg_time_per_move") else None,
                })

            # Add per-config rows
            for config, data in stats.get("configs", {}).items():
                table_data.append({
                    "Config": config,
                    "AvgNodes": round(data.get("avg_nodes", 0), 0) if data.get("avg_nodes") else None,
                    "MaxNodes": None,
                    "AvgDepth": round(data.get("avg_depth", 0), 1) if data.get("avg_depth") else None,
                    "MaxDepth": None,
                    "AvgTime": None,
                })

            return web.json_response(table_data)

        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Matchup Table
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_matchup_table(self, request: web.Request) -> web.Response:
        """GET /matchups/table - Matchups in table format for Grafana Infinity."""
        try:
            matrix = await self.analytics_cache_manager.get_matchup_matrix_cached()
            table_data = []
            for matchup in matrix.get("matchups", []):
                table_data.append({
                    "ModelA": matchup["model_a"],
                    "ModelB": matchup["model_b"],
                    "AWins": matchup["a_wins"],
                    "BWins": matchup["b_wins"],
                    "Draws": matchup["draws"],
                    "Total": matchup["total"],
                    "AWinRate": round(matchup["a_win_rate"] * 100, 1),
                })
            return web.json_response(table_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Model Lineage Table
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_model_lineage_table(self, request: web.Request) -> web.Response:
        """GET /models/lineage/table - Model lineage in table format for Grafana Infinity."""
        try:
            lineage = await self._get_model_lineage_cached()
            table_data = []
            for model in lineage.get("models", []):
                table_data.append({
                    "Name": model["name"],
                    "Config": model["config"],
                    "Generation": model["generation"],
                    "SizeMB": model["size_mb"],
                    "AgeHours": model["age_hours"],
                })
            return web.json_response(sorted(table_data, key=lambda x: (-x["Generation"], x["Config"])))
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Data Quality Table
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_data_quality_table(self, request: web.Request) -> web.Response:
        """GET /data/quality/table - Data quality in table format for Grafana Infinity."""
        try:
            quality = await self._get_data_quality_cached()
            table_data = []
            for config, metrics in quality.get("configs", {}).items():
                status = "OK"
                for issue in quality.get("issues", []):
                    if issue["config"] == config and issue["severity"] == "warning":
                        status = "WARNING"
                        break
                table_data.append({
                    "Config": config,
                    "Games": metrics["total_games"],
                    "AvgLength": metrics["avg_length"],
                    "ShortRate": metrics["short_game_rate"],
                    "StalemateRate": metrics["stalemate_rate"],
                    "OpeningDiv": metrics["opening_diversity"],
                    "Status": status,
                })
            return web.json_response(table_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Training Efficiency Table
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_training_efficiency_table(self, request: web.Request) -> web.Response:
        """GET /training/efficiency/table - Efficiency in table format for Grafana Infinity."""
        try:
            efficiency = await self._get_training_efficiency_cached()
            table_data = []
            for config, metrics in efficiency.get("configs", {}).items():
                table_data.append({
                    "Config": config,
                    "GPUHours": metrics["gpu_hours"],
                    "EloGain": metrics["elo_gain"],
                    "EloPerHour": metrics["elo_per_gpu_hour"],
                    "CostUSD": metrics["estimated_cost_usd"],
                    "CostPerElo": metrics["cost_per_elo_point"],
                })
            return web.json_response(table_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # A/B Test Table
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_abtest_table(self, request: web.Request) -> web.Response:
        """GET /abtest/table - A/B tests in table format for Grafana Infinity.

        Query params:
            status: Filter by status (optional)
        """
        try:
            status_filter = request.query.get("status")

            def _fetch_abtest_rows(
                db_path: str, status: str | None
            ) -> list[tuple[Any, ...]]:
                """Fetch A/B test rows - runs in thread pool."""
                with p2p_db_connection(db_path) as conn:
                    cursor = conn.cursor()
                    if status:
                        cursor.execute(
                            "SELECT test_id, name, board_type, num_players, model_a, model_b, status, winner, created_at "
                            "FROM ab_tests WHERE status = ? ORDER BY created_at DESC LIMIT 100",
                            (status,),
                        )
                    else:
                        cursor.execute(
                            "SELECT test_id, name, board_type, num_players, model_a, model_b, status, winner, created_at "
                            "FROM ab_tests ORDER BY created_at DESC LIMIT 100"
                        )
                    return cursor.fetchall()

            rows = await asyncio.to_thread(_fetch_abtest_rows, str(self.db_path), status_filter)

            table_data = []
            for row in rows:
                test_id = row[0]
                stats = self._calculate_ab_test_stats(test_id)
                from datetime import datetime
                created = datetime.fromtimestamp(row[8]).strftime("%Y-%m-%d %H:%M") if row[8] else ""

                table_data.append({
                    "Test ID": test_id[:8],
                    "Name": row[1],
                    "Config": f"{row[2]}_{row[3]}p",
                    "Model A": row[4].split("/")[-1] if "/" in row[4] else row[4],
                    "Model B": row[5].split("/")[-1] if "/" in row[5] else row[5],
                    "Games": stats.get("games_played", 0),
                    "A Win%": f"{stats.get('model_a_winrate', 0):.1%}",
                    "B Win%": f"{stats.get('model_b_winrate', 0):.1%}",
                    "Confidence": f"{stats.get('confidence', 0):.1%}",
                    "Status": row[6],
                    "Winner": row[7] or "-",
                    "Created": created,
                })

            return web.json_response(table_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Autoscale Recommendations Table
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_autoscale_recommendations(self, request: web.Request) -> web.Response:
        """GET /autoscale/recommendations - Autoscaling recommendations table."""
        try:
            metrics = await self._get_autoscaling_metrics()
            table_data = []
            for rec in metrics.get("recommendations", []):
                table_data.append({
                    "Action": rec["action"].upper(),
                    "Reason": rec["reason"],
                    "SuggestedWorkers": rec["suggested_workers"],
                })
            if not table_data:
                table_data.append({
                    "Action": "NONE",
                    "Reason": "Cluster is properly sized",
                    "SuggestedWorkers": metrics.get("current_workers", 0),
                })
            return web.json_response(table_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Rollback Candidates Table
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_rollback_candidates(self, request: web.Request) -> web.Response:
        """GET /rollback/candidates - Rollback candidates in table format."""
        try:
            status = await self._check_rollback_conditions()
            table_data = []
            for candidate in status.get("candidates", []):
                table_data.append({
                    "Config": candidate["config"],
                    "Reasons": ", ".join(candidate["reasons"]),
                    "Recommended": "YES" if candidate["rollback_recommended"] else "NO",
                })
            return web.json_response(table_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Data Quality Issues Table
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_data_quality_issues(self, request: web.Request) -> web.Response:
        """GET /data/quality/issues - Data quality issues in table format."""
        try:
            quality = await self._get_data_quality_cached()
            return web.json_response(quality.get("issues", []))
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Nodes Table (Phase 8 - Dec 28, 2025)
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_nodes_table(self, request: web.Request) -> web.Response:
        """GET /nodes/table - Node status in flat table format for Grafana Infinity.

        Returns current status of all cluster nodes in table format.
        """
        try:
            # Import JobType dynamically to avoid circular imports
            from scripts.p2p_orchestrator import JobType, NodeRole

            nodes = []

            # Add self
            node_name = self.node_id or "unknown"
            role = "Leader" if self.role == NodeRole.LEADER else "Worker"
            cpu = getattr(self.self_info, 'cpu_percent', 0)
            mem = getattr(self.self_info, 'memory_percent', 0)
            gpu = getattr(self.self_info, 'gpu_percent', 0) if self.self_info.has_gpu else 0
            gpu_mem = getattr(self.self_info, 'gpu_memory_percent', 0) if self.self_info.has_gpu else 0

            with self.jobs_lock:
                selfplay_jobs = len([j for j in self.local_jobs.values()
                                    if j.job_type in (JobType.SELFPLAY, JobType.GPU_SELFPLAY, JobType.HYBRID_SELFPLAY, JobType.CPU_SELFPLAY, JobType.GUMBEL_SELFPLAY)
                                    and j.status == "running"])

            nodes.append({
                "Node": node_name,
                "Role": role,
                "Status": "Online",
                "CPU": round(cpu, 1),
                "Memory": round(mem, 1),
                "GPU": round(gpu, 1),
                "GPUMem": round(gpu_mem, 1),
                "Jobs": selfplay_jobs,
                "HasGPU": "Yes" if self.self_info.has_gpu else "No",
            })

            # Add peers
            # Mar 2026: Use lock-free snapshot
            for peer_id, peer in self.get_peers_ro().items():
                peer_name = peer_id or "unknown"
                is_alive = peer.is_alive()
                status = "Online" if is_alive else "Offline"

                peer_cpu = getattr(peer, 'cpu_percent', 0) or 0
                peer_mem = getattr(peer, 'memory_percent', 0) or 0
                peer_gpu = getattr(peer, 'gpu_percent', 0) or 0
                peer_gpu_mem = getattr(peer, 'gpu_memory_percent', 0) or 0
                peer_jobs = getattr(peer, 'selfplay_jobs', 0) or 0
                has_gpu = getattr(peer, 'has_gpu', False)

                nodes.append({
                    "Node": peer_name,
                    "Role": "Worker",
                    "Status": status,
                    "CPU": round(peer_cpu, 1),
                    "Memory": round(peer_mem, 1),
                    "GPU": round(peer_gpu, 1),
                    "GPUMem": round(peer_gpu_mem, 1),
                    "Jobs": peer_jobs,
                    "HasGPU": "Yes" if has_gpu else "No",
                })

            # Sort by role (leader first) then by name
            nodes.sort(key=lambda n: (0 if n["Role"] == "Leader" else 1, n["Node"]))

            return web.json_response(nodes)

        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Victory Table (Phase 8 - Dec 28, 2025)
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_victory_table(self, request: web.Request) -> web.Response:
        """GET /victory/table - Victory type breakdown for Grafana Infinity.

        Returns victory type counts by board config in table format.
        Supports optional query params:
            - board_type: filter by board type
            - num_players: filter by player count
        """
        from collections import defaultdict

        try:
            board_type_filter = request.query.get("board_type")
            num_players_filter = request.query.get("num_players")
            if num_players_filter:
                try:
                    num_players_filter = int(num_players_filter)
                except ValueError:
                    num_players_filter = None

            stats = await self.analytics_cache_manager.get_victory_type_stats()

            # Group by config for table display
            config_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for (board_type, num_players, victory_type), count in stats.items():
                # Apply filters
                if board_type_filter and board_type != board_type_filter:
                    continue
                if num_players_filter and num_players != num_players_filter:
                    continue
                config = f"{board_type}_{num_players}p"
                config_stats[config][victory_type] = count

            # Build table rows
            table_data = []
            for config in sorted(config_stats.keys()):
                vt_counts = config_stats[config]
                total = sum(vt_counts.values())
                row = {
                    "Config": config,
                    "Total": total,
                    "Territory": vt_counts.get("territory", 0),
                    "LPS": vt_counts.get("lps", 0),
                    "Elimination": vt_counts.get("elimination", 0),
                    "RingElim": vt_counts.get("ring_elimination", 0),
                    "Stalemate": vt_counts.get("stalemate", 0),
                }
                # Add percentages
                if total > 0:
                    row["Territory%"] = round(100 * vt_counts.get("territory", 0) / total, 1)
                    row["LPS%"] = round(100 * vt_counts.get("lps", 0) / total, 1)
                    row["Elimination%"] = round(100 * vt_counts.get("elimination", 0) / total, 1)
                    row["RingElim%"] = round(100 * vt_counts.get("ring_elimination", 0) / total, 1)
                    row["Stalemate%"] = round(100 * vt_counts.get("stalemate", 0) / total, 1)
                else:
                    row["Territory%"] = row["LPS%"] = row["Elimination%"] = row["RingElim%"] = row["Stalemate%"] = 0
                table_data.append(row)

            return web.json_response(table_data)

        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Trends Table (Phase 8 - Dec 28, 2025)
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_trends_table(self, request: web.Request) -> web.Response:
        """GET /trends/table - Historical trends in table format for Grafana Infinity.

        Query params:
            metric: Metric type (required)
            hours: Time period (default: 168 = 7 days)
        """
        try:
            metric_type = request.query.get("metric")
            if not metric_type:
                return web.json_response([{"error": "Missing metric parameter"}])

            hours = float(request.query.get("hours", "168"))
            history = self.get_metrics_history(metric_type=metric_type, hours=hours, limit=500)

            table_data = []
            for record in history:
                from datetime import datetime
                ts = datetime.fromtimestamp(record["timestamp"]).strftime("%Y-%m-%d %H:%M")
                config = f"{record.get('board_type', '')}_{record.get('num_players', '')}p" if record.get('board_type') else "global"
                table_data.append({
                    "Timestamp": ts,
                    "Config": config,
                    "Value": round(record["value"], 3),
                    "Metric": metric_type,
                })

            return web.json_response(table_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Model Inventory (Jan 9, 2026)
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_model_inventory(self, request: web.Request) -> web.Response:
        """GET /models/inventory - Return local model inventory for cluster-wide discovery.

        Returns all NN and NNUE models available on this node with metadata.
        Used by ClusterModelEnumerator to build a unified model catalog.

        Response format:
        {
            "node_id": "node-name",
            "models": [
                {
                    "path": "models/canonical_hex8_2p.pth",
                    "type": "nn",
                    "board_type": "hex8",
                    "num_players": 2,
                    "size_bytes": 12345678,
                    "modified": 1704825600.0,
                    "architecture": "v2",
                    "hash": "abc123..."
                },
                ...
            ]
        }
        """
        import hashlib
        import os
        import re

        try:
            node_id = getattr(self, "node_id", None) or os.environ.get(
                "RINGRIFT_NODE_ID", "unknown"
            )
            models_dir = Path(
                getattr(self, "ringrift_path", ".")
            ) / "ai-service" / "models"

            models = []

            if not models_dir.exists():
                # Try alternate path (when running from ai-service directly)
                models_dir = Path("models")

            if models_dir.exists():
                for pth_file in models_dir.rglob("*.pth"):
                    try:
                        model_info = await asyncio.to_thread(
                            self._extract_model_info, pth_file
                        )
                        if model_info:
                            models.append(model_info)
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"Error scanning {pth_file}: {e}")

            return web.json_response({
                "node_id": node_id,
                "models": models,
                "count": len(models),
            })

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error getting model inventory: {e}")
            return web.json_response({"error": str(e)}, status=500)

    def _extract_model_info(self, pth_file: Path) -> dict[str, Any] | None:
        """Extract model metadata from a .pth file.

        Args:
            pth_file: Path to the model file

        Returns:
            Dict with model metadata or None if extraction failed
        """
        import hashlib
        import re

        try:
            name = pth_file.name.lower()
            path_str = str(pth_file)

            # Extract board type
            board_match = re.search(
                r"(hex8|square8|square19|hexagonal)", path_str.lower()
            )
            board_type = board_match.group(1) if board_match else None

            # Extract num_players
            players_match = re.search(r"_(\d)p", path_str.lower())
            num_players = int(players_match.group(1)) if players_match else None

            if not board_type or not num_players:
                return None

            # Detect model type
            if "nnue_mp" in name:
                model_type = "nnue_mp"
            elif "nnue" in name:
                model_type = "nnue"
            else:
                model_type = "nn"

            # Detect architecture from filename
            architecture = "v2"  # Default
            for arch in ["v5-heavy-xl", "v5-heavy-large", "v5-heavy", "v5", "v4", "v3", "v2"]:
                if arch.replace("-", "_") in name or arch in name:
                    architecture = arch
                    break

            # Compute hash from filename for identification
            file_hash = hashlib.sha256(pth_file.name.encode()).hexdigest()[:16]

            stat = pth_file.stat()

            return {
                "path": str(pth_file),
                "type": model_type,
                "board_type": board_type,
                "num_players": num_players,
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
                "architecture": architecture,
                "hash": file_hash,
            }

        except Exception as e:
            logger.debug(f"Failed to extract model info from {pth_file}: {e}")
            return None
