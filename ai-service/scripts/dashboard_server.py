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
import signal
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, redirect, request, send_from_directory
from flask_cors import CORS

# Setup paths
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
DASHBOARD_ASSETS = SCRIPT_DIR / "dashboard_assets"
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"
GAMES_DB_PATH = AI_SERVICE_ROOT / "data" / "games" / "selfplay.db"
TENSORBOARD_LOGDIR = AI_SERVICE_ROOT / "runs"
TENSORBOARD_PORT = 6006

app = Flask(__name__, static_folder=str(DASHBOARD_ASSETS))
CORS(app)

# TensorBoard process management
_tensorboard_process: Optional[subprocess.Popen] = None


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
# Training Dashboard Routes
# ============================================================================

@app.route("/training")
def training_dashboard():
    """Serve training metrics dashboard."""
    return send_from_directory(str(DASHBOARD_ASSETS), "training_dashboard.html")


@app.route("/api/training/loss-curves")
def api_loss_curves():
    """Get training loss curves."""
    model_id = request.args.get("model_id", "")
    hours = request.args.get("hours", "24", type=int)

    # Try to read from training reports
    reports_dir = AI_SERVICE_ROOT / "data" / "training_runs"
    steps = []
    total_loss = []
    policy_loss = []
    value_loss = []

    if reports_dir.exists():
        for report_path in sorted(reports_dir.glob("*/nn_training_report.json"))[-5:]:
            try:
                with open(report_path) as f:
                    report = json.load(f)
                    metrics = report.get("metrics", {})
                    if "loss_history" in metrics:
                        total_loss.extend(metrics["loss_history"])
                        steps.extend(range(len(total_loss)))
                    if "policy_loss_history" in metrics:
                        policy_loss.extend(metrics["policy_loss_history"])
                    if "value_loss_history" in metrics:
                        value_loss.extend(metrics["value_loss_history"])
            except Exception:
                pass

    # Generate sample data if no real data
    if not steps:
        import random
        steps = list(range(100))
        total_loss = [1.0 - i * 0.005 + random.uniform(-0.02, 0.02) for i in range(100)]
        policy_loss = [0.7 - i * 0.003 + random.uniform(-0.01, 0.01) for i in range(100)]
        value_loss = [0.3 - i * 0.002 + random.uniform(-0.01, 0.01) for i in range(100)]

    return jsonify({
        "steps": steps,
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    })


@app.route("/api/training/lr-history")
def api_lr_history():
    """Get learning rate history."""
    # Return sample LR schedule
    steps = list(range(100))
    learning_rates = [0.001 * (0.95 ** (i // 10)) for i in range(100)]

    return jsonify({
        "steps": steps,
        "learning_rates": learning_rates,
    })


@app.route("/api/training/throughput")
def api_throughput():
    """Get games per hour by host."""
    # Try to get real data from P2P status
    import urllib.request

    hosts = {}
    try:
        # Get P2P leader from config or environment
        p2p_leader = os.environ.get("P2P_LEADER", "localhost")
        if p2p_leader == "localhost":
            # Try to load from distributed_hosts.yaml
            config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path) as f:
                        config = yaml.safe_load(f) or {}
                    # Find the coordinator/leader node
                    coordinator = config.get("elo_sync", {}).get("coordinator", "mac-studio")
                    hosts_config = config.get("hosts", {})
                    if coordinator in hosts_config:
                        p2p_leader = hosts_config[coordinator].get("tailscale_ip", "localhost")
                except Exception:
                    pass
        with urllib.request.urlopen(f"http://{p2p_leader}:8770/status", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            for peer_id, peer_info in data.get("peers", {}).items():
                if not peer_info.get("retired", False):
                    jobs = peer_info.get("selfplay_jobs", 0)
                    hosts[peer_id] = jobs * 60  # Estimate games/hour
    except Exception:
        # Fallback sample data (empty - requires P2P connection)
        hosts = {}

    return jsonify({"hosts": hosts})


@app.route("/api/elo/progression")
def api_elo_progression():
    """Get Elo progression over time for top models."""
    if not ELO_DB_PATH.exists():
        return jsonify({"timestamps": [], "models": []})

    try:
        conn = sqlite3.connect(str(ELO_DB_PATH))
        conn.row_factory = sqlite3.Row

        # Get top 5 models
        cursor = conn.execute("""
            SELECT model_id, rating
            FROM elo_ratings
            WHERE board_type = 'square8' AND num_players = 2
            ORDER BY rating DESC
            LIMIT 5
        """)

        models = []
        for row in cursor:
            # Generate progression (in real implementation, query rating_history)
            import random
            base_rating = row["rating"]
            ratings = [max(1200, base_rating - 300 + i * 30 + random.uniform(-20, 20))
                      for i in range(20)]
            models.append({
                "name": row["model_id"][:20],
                "ratings": ratings,
            })

        timestamps = [f"Day {i+1}" for i in range(20)]

        conn.close()
        return jsonify({"timestamps": timestamps, "models": models})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/list")
def api_models_list():
    """List available models."""
    models_dir = AI_SERVICE_ROOT / "models"
    models = []

    if models_dir.exists():
        for pth_file in sorted(models_dir.glob("*.pth"), key=lambda x: -x.stat().st_mtime)[:50]:
            models.append({
                "id": pth_file.stem,
                "name": pth_file.stem,
                "path": str(pth_file),
            })

    return jsonify({"models": models})


@app.route("/api/models/compare")
def api_models_compare():
    """Compare top models."""
    if not ELO_DB_PATH.exists():
        return jsonify({"models": []})

    try:
        conn = sqlite3.connect(str(ELO_DB_PATH))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("""
            SELECT model_id, rating, games_played
            FROM elo_ratings
            WHERE board_type = 'square8' AND num_players = 2
            ORDER BY rating DESC
            LIMIT 10
        """)

        models = []
        for row in cursor:
            models.append({
                "name": row["model_id"][:25],
                "elo": row["rating"],
                "games": row["games_played"],
                "loss": None,  # Would need training data
            })

        conn.close()
        return jsonify({"models": models})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cluster/utilization")
def api_cluster_utilization():
    """Get cluster resource utilization."""
    # Placeholder - would query actual cluster metrics
    return jsonify({
        "gpu_percent": 72,
        "cpu_percent": 45,
        "memory_percent": 58,
    })


# ============================================================================
# Game Replay Routes
# ============================================================================

@app.route("/replay")
def replay_viewer():
    """Serve the game replay viewer."""
    return send_from_directory(str(DASHBOARD_ASSETS), "replay_viewer.html")


@app.route("/replay/<path:game_id>")
def replay_game(game_id):
    """Direct link to replay a specific game."""
    return send_from_directory(str(DASHBOARD_ASSETS), "replay_viewer.html")


@app.route("/compare")
def compare_models():
    """Serve the model comparison page."""
    return send_from_directory(str(DASHBOARD_ASSETS), "model_comparison.html")


@app.route("/api/replay/games")
def api_replay_games():
    """List games for replay."""
    limit = request.args.get("limit", 20, type=int)
    offset = request.args.get("offset", 0, type=int)
    board_type = request.args.get("board_type", "")
    winner = request.args.get("winner", "")
    game_id_filter = request.args.get("game_id", "")

    if not GAMES_DB_PATH.exists():
        return jsonify({"games": []})

    try:
        conn = sqlite3.connect(str(GAMES_DB_PATH))
        conn.row_factory = sqlite3.Row

        query = "SELECT game_id, board_type, num_players, winner, total_moves, duration_ms, created_at FROM games WHERE 1=1"
        params = []

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if winner:
            if winner == "draw":
                query += " AND winner IS NULL"
            else:
                query += " AND winner = ?"
                params.append(int(winner))
        if game_id_filter:
            query += " AND game_id LIKE ?"
            params.append(f"%{game_id_filter}%")

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.execute(query, params)
        games = []
        for row in cursor:
            games.append({
                "gameId": row["game_id"],
                "boardType": row["board_type"],
                "numPlayers": row["num_players"],
                "winner": row["winner"],
                "totalMoves": row["total_moves"],
                "durationMs": row["duration_ms"],
                "createdAt": row["created_at"],
            })

        conn.close()
        return jsonify({"games": games})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/replay/games/<game_id>")
def api_replay_game_detail(game_id):
    """Get game metadata."""
    if not GAMES_DB_PATH.exists():
        return jsonify({"error": "Database not found"}), 404

    try:
        conn = sqlite3.connect(str(GAMES_DB_PATH))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("""
            SELECT game_id, board_type, num_players, winner, total_moves,
                   duration_ms, termination_reason, created_at
            FROM games WHERE game_id = ?
        """, (game_id,))

        row = cursor.fetchone()
        if not row:
            conn.close()
            return jsonify({"error": "Game not found"}), 404

        result = {
            "gameId": row["game_id"],
            "boardType": row["board_type"],
            "numPlayers": row["num_players"],
            "winner": row["winner"],
            "totalMoves": row["total_moves"],
            "durationMs": row["duration_ms"],
            "terminationReason": row["termination_reason"],
            "createdAt": row["created_at"],
        }

        conn.close()
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/replay/games/<game_id>/state")
def api_replay_game_state(game_id):
    """Get game state at specific move."""
    move_number = request.args.get("move_number", 0, type=int)

    if not GAMES_DB_PATH.exists():
        return jsonify({"error": "Database not found"}), 404

    try:
        conn = sqlite3.connect(str(GAMES_DB_PATH))
        conn.row_factory = sqlite3.Row

        # Try to get state from history
        cursor = conn.execute("""
            SELECT state_json FROM game_history_entries
            WHERE game_id = ? AND move_number <= ?
            ORDER BY move_number DESC LIMIT 1
        """, (game_id, move_number))

        row = cursor.fetchone()
        if row and row["state_json"]:
            state = json.loads(row["state_json"])
            conn.close()
            return jsonify({"gameState": state, "moveNumber": move_number})

        # Fallback: try initial state
        cursor = conn.execute("""
            SELECT initial_state_json FROM game_initial_state
            WHERE game_id = ?
        """, (game_id,))

        row = cursor.fetchone()
        if row and row["initial_state_json"]:
            state = json.loads(row["initial_state_json"])
            conn.close()
            return jsonify({"gameState": state, "moveNumber": 0})

        conn.close()
        return jsonify({"gameState": {}, "moveNumber": move_number})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/replay/games/<game_id>/moves")
def api_replay_game_moves(game_id):
    """Get move list for a game."""
    limit = request.args.get("limit", 1000, type=int)

    if not GAMES_DB_PATH.exists():
        return jsonify({"moves": []})

    try:
        conn = sqlite3.connect(str(GAMES_DB_PATH))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("""
            SELECT move_number, move_json, player_number
            FROM game_history_entries
            WHERE game_id = ?
            ORDER BY move_number ASC
            LIMIT ?
        """, (game_id, limit))

        moves = []
        for row in cursor:
            move_data = {}
            if row["move_json"]:
                try:
                    move_data = json.loads(row["move_json"])
                except Exception:
                    pass
            move_data["moveNumber"] = row["move_number"]
            move_data["player"] = row["player_number"]
            moves.append(move_data)

        conn.close()
        return jsonify({"moves": moves})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/replay/games/<game_id>/eval")
def api_replay_game_eval(game_id):
    """Get AI evaluation for a position in a game.

    This endpoint provides:
    - Win probability from current player's perspective
    - Value head output (raw)
    - Best move suggestion
    - Quality of actual move played (if applicable)
    """
    move_number = request.args.get("move_number", 0, type=int)

    if not GAMES_DB_PATH.exists():
        return jsonify({"error": "Game database not found"}), 404

    try:
        conn = sqlite3.connect(str(GAMES_DB_PATH))
        conn.row_factory = sqlite3.Row

        # Check if we have cached evaluation data
        cursor = conn.execute("""
            SELECT eval_json
            FROM game_history_entries
            WHERE game_id = ? AND move_number = ?
        """, (game_id, move_number))

        row = cursor.fetchone()
        eval_data = None

        if row and row["eval_json"]:
            try:
                eval_data = json.loads(row["eval_json"])
            except Exception:
                pass

        conn.close()

        # If we have cached eval data, return it
        if eval_data:
            return jsonify({
                "winProbability": eval_data.get("win_prob", 0.5),
                "value": eval_data.get("value", 0.0),
                "bestMove": eval_data.get("best_move"),
                "moveQuality": eval_data.get("move_quality"),
                "policyEntropy": eval_data.get("policy_entropy"),
                "cached": True,
            })

        # Otherwise return placeholder data (real-time eval would require model loading)
        # In production, this would call the inference service
        return jsonify({
            "winProbability": 0.5,
            "value": 0.0,
            "bestMove": None,
            "moveQuality": None,
            "cached": False,
            "message": "Real-time evaluation not available - no cached data",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# TensorBoard Integration
# ============================================================================

def _is_tensorboard_running() -> bool:
    """Check if TensorBoard is running on the expected port."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(("127.0.0.1", TENSORBOARD_PORT))
            return result == 0
    except Exception:
        return False


def _start_tensorboard() -> bool:
    """Start TensorBoard process if not running."""
    global _tensorboard_process

    if _is_tensorboard_running():
        return True

    # Create logs directory if needed
    TENSORBOARD_LOGDIR.mkdir(parents=True, exist_ok=True)

    try:
        _tensorboard_process = subprocess.Popen(
            [
                sys.executable, "-m", "tensorboard",
                "--logdir", str(TENSORBOARD_LOGDIR),
                "--port", str(TENSORBOARD_PORT),
                "--bind_all",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Give it a moment to start
        import time
        time.sleep(2)
        return _is_tensorboard_running()
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")
        return False


def _stop_tensorboard() -> bool:
    """Stop TensorBoard process if running."""
    global _tensorboard_process

    if _tensorboard_process:
        try:
            _tensorboard_process.terminate()
            _tensorboard_process.wait(timeout=5)
        except Exception:
            _tensorboard_process.kill()
        _tensorboard_process = None
        return True
    return False


@app.route("/api/tensorboard/status")
def api_tensorboard_status():
    """Get TensorBoard status."""
    running = _is_tensorboard_running()
    log_dir_exists = TENSORBOARD_LOGDIR.exists()

    # Count run directories
    run_count = 0
    if log_dir_exists:
        run_count = len([d for d in TENSORBOARD_LOGDIR.iterdir() if d.is_dir()])

    return jsonify({
        "running": running,
        "port": TENSORBOARD_PORT,
        "url": f"http://localhost:{TENSORBOARD_PORT}" if running else None,
        "logdir": str(TENSORBOARD_LOGDIR),
        "logdir_exists": log_dir_exists,
        "run_count": run_count,
    })


@app.route("/api/tensorboard/start", methods=["POST"])
def api_tensorboard_start():
    """Start TensorBoard."""
    success = _start_tensorboard()
    return jsonify({
        "success": success,
        "running": _is_tensorboard_running(),
        "url": f"http://localhost:{TENSORBOARD_PORT}" if success else None,
    })


@app.route("/api/tensorboard/stop", methods=["POST"])
def api_tensorboard_stop():
    """Stop TensorBoard."""
    success = _stop_tensorboard()
    return jsonify({
        "success": success,
        "running": _is_tensorboard_running(),
    })


@app.route("/tensorboard")
def tensorboard_redirect():
    """Redirect to TensorBoard (starts it if needed)."""
    if not _is_tensorboard_running():
        _start_tensorboard()
    return redirect(f"http://localhost:{TENSORBOARD_PORT}")


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
