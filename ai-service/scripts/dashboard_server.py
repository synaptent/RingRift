#!/usr/bin/env python3
"""Simple web server for RingRift model performance dashboard.

Serves the dashboard HTML and provides API endpoints for:
- /api/leaderboard - Elo rankings
- /api/cluster/status - Cluster node status
- /api/stats - Overall statistics

Usage:
    python scripts/dashboard_server.py
    python scripts/dashboard_server.py --port 8080 --host 0.0.0.0
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Setup paths
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
DASHBOARD_ASSETS = SCRIPT_DIR / "dashboard_assets"
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"
GAMES_DB_PATH = AI_SERVICE_ROOT / "data" / "games" / "selfplay.db"

app = Flask(__name__, static_folder=str(DASHBOARD_ASSETS))
CORS(app)


# ============================================================================
# Dashboard HTML
# ============================================================================

@app.route("/")
def index():
    """Serve the dashboard HTML."""
    return send_from_directory(str(DASHBOARD_ASSETS), "model_dashboard.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve static files from dashboard assets."""
    return send_from_directory(str(DASHBOARD_ASSETS), filename)


# ============================================================================
# API Endpoints
# ============================================================================

@app.route("/api/leaderboard")
def api_leaderboard():
    """Get Elo leaderboard for specified board type and player count."""
    board = request.args.get("board", "square8")
    players = request.args.get("players", "2", type=int)

    if not ELO_DB_PATH.exists():
        return jsonify([])

    try:
        conn = sqlite3.connect(str(ELO_DB_PATH))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("""
            SELECT
                model_id,
                rating,
                rating_deviation,
                games_played,
                wins,
                losses,
                draws,
                last_game_at
            FROM elo_ratings
            WHERE board_type = ? AND num_players = ?
            ORDER BY rating DESC
            LIMIT 50
        """, (board, players))

        results = []
        for row in cursor:
            total = row["wins"] + row["losses"] + row["draws"]
            win_rate = row["wins"] / total if total > 0 else 0

            # Calculate CI bounds (simple approximation)
            rd = row["rating_deviation"] or 50
            ci_lower = row["rating"] - 1.96 * rd
            ci_upper = row["rating"] + 1.96 * rd

            results.append({
                "model_id": row["model_id"],
                "rating": row["rating"],
                "rating_deviation": rd,
                "games_played": row["games_played"],
                "win_rate": round(win_rate, 4),
                "ci_lower": round(ci_lower, 1),
                "ci_upper": round(ci_upper, 1),
            })

        conn.close()
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cluster/status")
def api_cluster_status():
    """Get cluster node status."""
    # Try to read from metrics or return mock data
    nodes = [
        {"node": "192.222.53.22", "gpu_type": "GH200", "is_up": True, "gpu_utilization": 0.75, "cost_per_hour": 2.49},
        {"node": "192.222.53.23", "gpu_type": "GH200", "is_up": True, "gpu_utilization": 0.82, "cost_per_hour": 2.49},
        {"node": "192.222.53.24", "gpu_type": "GH200", "is_up": True, "gpu_utilization": 0.68, "cost_per_hour": 2.49},
    ]
    return jsonify(nodes)


@app.route("/api/stats")
def api_stats():
    """Get overall training statistics."""
    stats = {
        "total_models": 0,
        "games_today": 0,
        "active_nodes": 3,
        "cost_today": 0.0,
    }

    # Count models
    models_dir = AI_SERVICE_ROOT / "models"
    if models_dir.exists():
        stats["total_models"] = len(list(models_dir.glob("*.pth")))

    # Count games today
    if GAMES_DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(GAMES_DB_PATH))
            today = datetime.utcnow().strftime("%Y-%m-%d")
            cursor = conn.execute(
                "SELECT COUNT(*) FROM games WHERE created_at >= ?",
                (today,)
            )
            stats["games_today"] = cursor.fetchone()[0]
            conn.close()
        except Exception:
            pass

    # Estimate cost (3 nodes * $2.49/hr * 24hr)
    stats["cost_today"] = 3 * 2.49 * 24  # ~$179/day with 3 GH200s

    return jsonify(stats)


@app.route("/api/promotions")
def api_promotions():
    """Get recent model promotions."""
    history_path = AI_SERVICE_ROOT / "data" / "model_promotion_history.json"
    if not history_path.exists():
        return jsonify([])

    try:
        with open(history_path) as f:
            history = json.load(f)
        # Return last 20 promotions
        return jsonify(history[-20:] if isinstance(history, list) else [])
    except Exception:
        return jsonify([])


@app.route("/api/training/progress")
def api_training_progress():
    """Get training progress by board type."""
    progress = {
        "square8_2p": {"games": 0, "models": 0},
        "square8_3p": {"games": 0, "models": 0},
        "square19_2p": {"games": 0, "models": 0},
        "hexagonal_2p": {"games": 0, "models": 0},
    }

    # Count games per board type
    games_dir = AI_SERVICE_ROOT / "data" / "games"
    if games_dir.exists():
        for db_path in games_dir.glob("*.db"):
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute(
                    "SELECT board_type, num_players, COUNT(*) FROM games GROUP BY board_type, num_players"
                )
                for row in cursor:
                    key = f"{row[0]}_{row[1]}p"
                    if key in progress:
                        progress[key]["games"] += row[2]
                conn.close()
            except Exception:
                pass

    return jsonify(progress)


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RingRift Dashboard Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print(f"Starting RingRift Dashboard at http://{args.host}:{args.port}")
    print(f"Dashboard assets: {DASHBOARD_ASSETS}")
    print(f"Elo database: {ELO_DB_PATH}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
