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
import contextlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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
_tensorboard_process: subprocess.Popen | None = None


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

    except (sqlite3.Error, OSError) as e:
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
        except sqlite3.Error:
            pass

    # Estimate cost (3 nodes * $2.49/hr * 24hr)
    stats["cost_today"] = 3 * 2.49 * 24  # ~$179/day with 3 GH200s

    return jsonify(stats)


@app.route("/api/progress")
def api_progress():
    """Get Elo progress report for demonstrating iterative improvement.

    January 16, 2026: Added for progress dashboard.

    Query parameters:
        config: Optional config filter (e.g., "hex8_2p")
        days: Lookback period in days (default: 30)

    Returns JSON with:
        - configs: Per-config progress data
        - overall: Summary stats
        - generated_at: Timestamp
    """
    config_filter = request.args.get("config")
    try:
        days = float(request.args.get("days", "30"))
    except ValueError:
        days = 30.0

    try:
        # Import and use progress report module
        from scripts.elo_progress_report import get_full_report
        from dataclasses import asdict

        report = get_full_report(days=days, config_filter=config_filter)

        # Convert to JSON-serializable dict
        data = {
            "configs": {k: asdict(v) for k, v in report.configs.items()},
            "overall": asdict(report.overall),
            "generated_at": report.generated_at,
        }

        return jsonify(data)

    except ImportError as e:
        return jsonify({
            "error": "progress_report_unavailable",
            "detail": str(e),
        }), 500
    except Exception as e:
        return jsonify({
            "error": "internal_error",
            "detail": str(e),
        }), 500


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
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return jsonify([])


@app.route("/api/promotion_rejections")
def api_promotion_rejections():
    """Get promotion rejection summary for pipeline health dashboard.

    Feb 23, 2026: Reads PROMOTION_REJECTED events persisted by AutoPromotionDaemon
    to a JSONL file. Returns per-config rejection summaries with consecutive counts,
    last gate, and timestamps.

    Query params:
        hours: Lookback period in hours (default: 168 = 7 days)

    Returns:
        {
            configs: {
                "hex8_2p": {
                    rejections: [{gate, reason, timestamp, actual, threshold}, ...],
                    consecutive_by_gate: {"quality": 3, "elo_improvement": 1},
                    last_rejection_time: 1708700000.0,
                    last_rejection_gate: "quality",
                    total_rejections: 4,
                    status: "stalled" | "warning" | "ok"
                },
                ...
            },
            generated_at: "2026-02-23T12:00:00"
        }
    """
    import time as _time
    from collections import defaultdict

    try:
        hours = float(request.args.get("hours", "168"))
    except ValueError:
        hours = 168.0

    cutoff = _time.time() - (hours * 3600)
    rejections_path = AI_SERVICE_ROOT / "data" / "promotion_rejections.jsonl"

    # Read all rejection events from JSONL file
    raw_rejections: list[dict] = []
    if rejections_path.exists():
        try:
            with open(rejections_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        ts = entry.get("timestamp", 0)
                        if ts >= cutoff:
                            raw_rejections.append(entry)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    # Read recent promotions to determine "last promoted" time per config
    last_promoted: dict[str, float] = {}
    history_path = AI_SERVICE_ROOT / "data" / "model_promotion_history.json"
    if history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
            if isinstance(history, list):
                for entry in history:
                    config = entry.get("config_key", "")
                    ts = entry.get("timestamp", 0)
                    if config and ts > last_promoted.get(config, 0):
                        last_promoted[config] = ts
        except (json.JSONDecodeError, OSError):
            pass

    # Group rejections by config and compute summaries
    configs_data: dict[str, dict] = {}
    rejections_by_config: dict[str, list[dict]] = defaultdict(list)

    for entry in raw_rejections:
        config_key = entry.get("config_key", "unknown")
        rejections_by_config[config_key].append(entry)

    for config_key, rejections in rejections_by_config.items():
        # Sort by timestamp ascending
        rejections.sort(key=lambda e: e.get("timestamp", 0))

        # Count consecutive rejections by gate (from most recent backwards)
        # Only count rejections AFTER the last successful promotion
        config_last_promoted = last_promoted.get(config_key, 0)
        recent_rejections = [
            r for r in rejections if r.get("timestamp", 0) > config_last_promoted
        ]

        consecutive_by_gate: dict[str, int] = defaultdict(int)
        for r in recent_rejections:
            gate = r.get("gate", "unknown")
            consecutive_by_gate[gate] += 1

        total_consecutive = len(recent_rejections)
        last_rejection = rejections[-1] if rejections else {}
        last_rejection_time = last_rejection.get("timestamp", 0)
        last_rejection_gate = last_rejection.get("gate", "")

        # Determine status: red (5+), yellow (1-4), green (0 or recently promoted)
        if total_consecutive >= 5:
            status = "stalled"
        elif total_consecutive >= 1:
            status = "warning"
        else:
            status = "ok"

        # Return last 20 rejections (newest first) for display
        display_rejections = list(reversed(rejections[-20:]))

        configs_data[config_key] = {
            "rejections": display_rejections,
            "consecutive_by_gate": dict(consecutive_by_gate),
            "last_rejection_time": last_rejection_time,
            "last_rejection_gate": last_rejection_gate,
            "total_rejections": len(rejections),
            "total_consecutive": total_consecutive,
            "last_promoted_time": config_last_promoted,
            "status": status,
        }

    return jsonify({
        "configs": configs_data,
        "generated_at": datetime.utcnow().isoformat(),
    })


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
            except sqlite3.Error:
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
    request.args.get("model_id", "")
    request.args.get("hours", "24", type=int)

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
            except (FileNotFoundError, json.JSONDecodeError, OSError):
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
                except (FileNotFoundError, OSError, KeyError, ValueError):
                    pass
        with urllib.request.urlopen(f"http://{p2p_leader}:8770/status", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            for peer_id, peer_info in data.get("peers", {}).items():
                if not peer_info.get("retired", False):
                    jobs = peer_info.get("selfplay_jobs", 0)
                    hosts[peer_id] = jobs * 60  # Estimate games/hour
    except (ConnectionError, TimeoutError, json.JSONDecodeError, OSError):
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

    except (sqlite3.Error, OSError) as e:
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

    except (sqlite3.Error, OSError) as e:
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

    except (sqlite3.Error, OSError, ValueError) as e:
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

    except (sqlite3.Error, OSError) as e:
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

    except (sqlite3.Error, json.JSONDecodeError, OSError) as e:
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
                with contextlib.suppress(Exception):
                    move_data = json.loads(row["move_json"])
            move_data["moveNumber"] = row["move_number"]
            move_data["player"] = row["player_number"]
            moves.append(move_data)

        conn.close()
        return jsonify({"moves": moves})

    except (sqlite3.Error, json.JSONDecodeError, OSError) as e:
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
            with contextlib.suppress(Exception):
                eval_data = json.loads(row["eval_json"])

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

    except (sqlite3.Error, json.JSONDecodeError, OSError) as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Elo Dashboard Routes
# ============================================================================

@app.route("/elo")
def elo_dashboard():
    """Serve the Elo observability dashboard."""
    return send_from_directory(str(DASHBOARD_ASSETS), "elo_dashboard.html")


@app.route("/api/elo/history")
def api_elo_history():
    """Get Elo history for time-series chart.

    Query params:
        board: Board type (hex8, square8, square19, hexagonal)
        players: Number of players (2, 3, 4)
        hours: Hours of history to fetch (default 168 = 7 days)

    Returns:
        {timestamps: [], models: [{name, ratings: []}]}
    """
    board = request.args.get("board", "hex8")
    players = request.args.get("players", 2, type=int)
    hours = request.args.get("hours", 168, type=int)

    if not ELO_DB_PATH.exists():
        return jsonify({"timestamps": [], "models": []})

    try:
        conn = sqlite3.connect(str(ELO_DB_PATH))
        conn.row_factory = sqlite3.Row

        # Calculate cutoff timestamp
        import time
        cutoff = time.time() - (hours * 3600)

        # Get rating history for top models in this config
        cursor = conn.execute("""
            SELECT
                participant_id,
                rating,
                timestamp
            FROM rating_history
            WHERE board_type = ? AND num_players = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        """, (board, players, cutoff))

        # Group by model
        model_data: dict[str, list[tuple[float, float]]] = {}
        for row in cursor:
            model_id = row["participant_id"]
            if model_id not in model_data:
                model_data[model_id] = []
            model_data[model_id].append((row["timestamp"], row["rating"]))

        # Get current top 10 models by rating
        cursor = conn.execute("""
            SELECT participant_id, rating
            FROM elo_ratings
            WHERE board_type = ? AND num_players = ?
            ORDER BY rating DESC
            LIMIT 10
        """, (board, players))
        top_models = {row["participant_id"] for row in cursor}

        # Filter to only include top models with history
        models = []
        all_timestamps: set[float] = set()

        for model_id in top_models:
            if model_id in model_data:
                for ts, _ in model_data[model_id]:
                    all_timestamps.add(ts)

        # Sort timestamps and create time labels
        sorted_ts = sorted(all_timestamps)
        timestamps = [
            datetime.fromtimestamp(ts).strftime("%m/%d %H:%M")
            for ts in sorted_ts
        ]

        # Build model rating arrays aligned to timestamps
        for model_id in top_models:
            if model_id not in model_data:
                continue

            # Create a map of timestamp -> rating
            ts_to_rating = {ts: rating for ts, rating in model_data[model_id]}

            # Interpolate ratings for each timestamp
            ratings = []
            last_rating: Optional[float] = None
            for ts in sorted_ts:
                if ts in ts_to_rating:
                    last_rating = ts_to_rating[ts]
                ratings.append(last_rating if last_rating else None)

            # Only include if we have some data
            if any(r is not None for r in ratings):
                # Truncate model name for display
                display_name = model_id[:25] if len(model_id) > 25 else model_id
                models.append({
                    "name": display_name,
                    "ratings": ratings,
                })

        conn.close()
        return jsonify({"timestamps": timestamps, "models": models})

    except (sqlite3.Error, OSError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/elo/velocity")
def api_elo_velocity():
    """Get Elo velocity metrics per config.

    Returns Elo change per hour for each board/player configuration.

    Returns:
        {configs: [{config_key, current_elo, velocity_per_hour, trend, games_24h}]}
    """
    if not ELO_DB_PATH.exists():
        return jsonify({"configs": []})

    try:
        conn = sqlite3.connect(str(ELO_DB_PATH))
        conn.row_factory = sqlite3.Row

        import time
        now = time.time()
        cutoff_24h = now - (24 * 3600)
        cutoff_7d = now - (7 * 24 * 3600)

        configs = []
        all_configs = [
            ("hex8", 2), ("hex8", 3), ("hex8", 4),
            ("square8", 2), ("square8", 3), ("square8", 4),
            ("square19", 2), ("square19", 3), ("square19", 4),
            ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
        ]

        for board, players in all_configs:
            config_key = f"{board}_{players}p"

            # Get current best model rating
            cursor = conn.execute("""
                SELECT participant_id, rating, games_played
                FROM elo_ratings
                WHERE board_type = ? AND num_players = ?
                  AND participant_id NOT LIKE '%random%'
                  AND participant_id NOT LIKE '%heuristic%'
                ORDER BY rating DESC
                LIMIT 1
            """, (board, players))
            row = cursor.fetchone()

            if not row:
                configs.append({
                    "config_key": config_key,
                    "current_elo": 1500,
                    "velocity_per_hour": 0.0,
                    "trend": "stable",
                    "games_24h": 0,
                    "best_model": None,
                })
                continue

            current_elo = row["rating"]
            best_model = row["participant_id"]

            # Get rating 24 hours ago
            cursor = conn.execute("""
                SELECT rating
                FROM rating_history
                WHERE board_type = ? AND num_players = ?
                  AND participant_id = ?
                  AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (board, players, best_model, cutoff_24h))
            row_24h = cursor.fetchone()
            elo_24h_ago = row_24h["rating"] if row_24h else current_elo

            # Get rating 7 days ago for trend
            cursor = conn.execute("""
                SELECT rating
                FROM rating_history
                WHERE board_type = ? AND num_players = ?
                  AND participant_id = ?
                  AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (board, players, best_model, cutoff_7d))
            row_7d = cursor.fetchone()
            elo_7d_ago = row_7d["rating"] if row_7d else current_elo

            # Calculate velocity (Elo per hour over last 24h)
            velocity = (current_elo - elo_24h_ago) / 24.0

            # Determine trend based on 7-day movement
            elo_change_7d = current_elo - elo_7d_ago
            if elo_change_7d > 50:
                trend = "rising"
            elif elo_change_7d < -50:
                trend = "falling"
            else:
                trend = "stable"

            # Count games in last 24h
            cursor = conn.execute("""
                SELECT COUNT(*) as cnt
                FROM rating_history
                WHERE board_type = ? AND num_players = ? AND timestamp >= ?
            """, (board, players, cutoff_24h))
            games_24h = cursor.fetchone()["cnt"]

            configs.append({
                "config_key": config_key,
                "current_elo": round(current_elo, 1),
                "velocity_per_hour": round(velocity, 2),
                "trend": trend,
                "games_24h": games_24h,
                "best_model": best_model[:30] if best_model else None,
            })

        conn.close()
        return jsonify({"configs": configs})

    except (sqlite3.Error, OSError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/elo/targets")
def api_elo_targets():
    """Get Elo targets and progress toward 2000 Elo goal.

    Returns:
        {configs: [{config_key, current_elo, target_elo, games_played, percent_complete}]}
    """
    target_elo = 2000  # Hardcoded target

    if not ELO_DB_PATH.exists():
        return jsonify({"configs": [], "target_elo": target_elo})

    try:
        conn = sqlite3.connect(str(ELO_DB_PATH))
        conn.row_factory = sqlite3.Row

        configs = []
        all_configs = [
            ("hex8", 2), ("hex8", 3), ("hex8", 4),
            ("square8", 2), ("square8", 3), ("square8", 4),
            ("square19", 2), ("square19", 3), ("square19", 4),
            ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
        ]

        for board, players in all_configs:
            config_key = f"{board}_{players}p"

            # Get current best model rating (excluding baselines)
            cursor = conn.execute("""
                SELECT MAX(rating) as best_rating
                FROM elo_ratings
                WHERE board_type = ? AND num_players = ?
                  AND participant_id NOT LIKE '%random%'
                  AND participant_id NOT LIKE '%heuristic%'
            """, (board, players))
            row = cursor.fetchone()
            current_elo = row["best_rating"] if row and row["best_rating"] else 1500

            # Get total games played for this config
            cursor = conn.execute("""
                SELECT SUM(games_played) as total_games
                FROM elo_ratings
                WHERE board_type = ? AND num_players = ?
            """, (board, players))
            row = cursor.fetchone()
            games_played = row["total_games"] if row and row["total_games"] else 0

            # Calculate progress (assuming baseline of 1200)
            baseline = 1200
            progress_range = target_elo - baseline  # 800 points
            current_progress = max(0, current_elo - baseline)
            percent_complete = min(100, round((current_progress / progress_range) * 100, 1))

            configs.append({
                "config_key": config_key,
                "current_elo": round(current_elo, 1),
                "target_elo": target_elo,
                "games_played": games_played,
                "percent_complete": percent_complete,
            })

        conn.close()
        return jsonify({"configs": configs, "target_elo": target_elo})

    except (sqlite3.Error, OSError) as e:
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
    except (OSError, socket.error):
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
        except (OSError, subprocess.TimeoutExpired):
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
