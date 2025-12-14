#!/usr/bin/env python3
"""Elo Metrics Exporter for Prometheus/Grafana.

Exposes NN model Elo ratings and improvement trends as Prometheus metrics.
Designed to run as a lightweight service that Grafana can scrape.

Usage:
    python scripts/elo_metrics_exporter.py --port 9092

    # Or run in background
    nohup python scripts/elo_metrics_exporter.py --port 9092 &

Metrics exported:
    - ringrift_model_elo_current: Current Elo rating per model/config
    - ringrift_model_elo_best: Best model Elo per config
    - ringrift_model_elo_improvement_rate: Elo points gained per hour
    - ringrift_model_games_played: Total games played per model
    - ringrift_model_win_rate: Win rate (0-1) per model
    - ringrift_training_iterations: Total training iterations completed
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Prometheus client
try:
    from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    print("Warning: prometheus_client not installed. Run: pip install prometheus_client")

# Paths
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"
MODELS_DIR = AI_SERVICE_ROOT / "models"

# Metrics
if HAS_PROMETHEUS:
    MODEL_ELO_CURRENT = Gauge(
        'ringrift_model_elo_current',
        'Current Elo rating for a model',
        ['config', 'model', 'model_type']
    )

    MODEL_ELO_BEST = Gauge(
        'ringrift_model_elo_best',
        'Best model Elo rating per configuration',
        ['config']
    )

    MODEL_ELO_IMPROVEMENT_RATE = Gauge(
        'ringrift_model_elo_improvement_rate',
        'Elo improvement rate (points per hour) over last 24h',
        ['config']
    )

    MODEL_ELO_IMPROVEMENT_24H = Gauge(
        'ringrift_model_elo_improvement_24h',
        'Total Elo improvement in last 24 hours',
        ['config']
    )

    MODEL_GAMES_PLAYED = Gauge(
        'ringrift_model_games_played',
        'Total games played by a model',
        ['config', 'model']
    )

    MODEL_WIN_RATE = Gauge(
        'ringrift_model_win_rate',
        'Win rate (0-1) for a model',
        ['config', 'model']
    )

    TRAINING_ITERATIONS = Gauge(
        'ringrift_training_iterations_total',
        'Total training iterations completed',
        ['config']
    )

    MODELS_TOTAL = Gauge(
        'ringrift_models_total',
        'Total number of models per configuration',
        ['config']
    )

    LAST_TRAINING_TIME = Gauge(
        'ringrift_last_training_timestamp',
        'Unix timestamp of last training completion',
        ['config']
    )

    DATA_SYNC_AGE = Gauge(
        'ringrift_data_sync_age_seconds',
        'Age of most recent synced data in seconds',
        []
    )


def get_current_elo_ratings() -> List[Tuple]:
    """Get current Elo ratings from database."""
    if not ELO_DB_PATH.exists():
        return []

    conn = sqlite3.connect(ELO_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            e.participant_id,
            e.board_type,
            e.num_players,
            e.rating,
            e.games_played,
            e.wins,
            e.losses,
            e.draws,
            e.last_update,
            p.participant_type,
            p.model_version
        FROM elo_ratings e
        LEFT JOIN participants p ON e.participant_id = p.participant_id
        WHERE e.games_played >= 5
        ORDER BY e.board_type, e.num_players, e.rating DESC
    """)

    results = cursor.fetchall()
    conn.close()
    return results


def get_elo_history(hours: int = 24) -> Dict[str, List[Tuple]]:
    """Get Elo rating history for the last N hours, grouped by config."""
    if not ELO_DB_PATH.exists():
        return {}

    conn = sqlite3.connect(ELO_DB_PATH)
    cursor = conn.cursor()

    cutoff = time.time() - (hours * 3600)

    cursor.execute("""
        SELECT
            participant_id,
            board_type,
            num_players,
            rating,
            games_played,
            timestamp
        FROM rating_history
        WHERE timestamp > ?
        AND participant_id LIKE '%nn%' OR participant_id LIKE 'ringrift_%'
        ORDER BY board_type, num_players, timestamp ASC
    """, (cutoff,))

    results = cursor.fetchall()
    conn.close()

    # Group by config
    history: Dict[str, List[Tuple]] = {}
    for row in results:
        config = f"{row[1]}_{row[2]}p"
        if config not in history:
            history[config] = []
        history[config].append(row)

    return history


def calculate_improvement_rate(config: str, history: List[Tuple]) -> Tuple[float, float]:
    """Calculate Elo improvement rate for a config.

    Returns: (improvement_per_hour, total_improvement_24h)
    """
    if not history or len(history) < 2:
        return 0.0, 0.0

    # Get best rating at each timestamp
    best_by_time: Dict[float, float] = {}
    for row in history:
        ts = row[5]
        rating = row[3]
        if ts not in best_by_time or rating > best_by_time[ts]:
            best_by_time[ts] = rating

    if len(best_by_time) < 2:
        return 0.0, 0.0

    sorted_times = sorted(best_by_time.keys())
    earliest_rating = best_by_time[sorted_times[0]]
    latest_rating = best_by_time[sorted_times[-1]]

    time_span_hours = (sorted_times[-1] - sorted_times[0]) / 3600
    if time_span_hours < 0.1:  # Less than 6 minutes
        return 0.0, 0.0

    total_improvement = latest_rating - earliest_rating
    rate_per_hour = total_improvement / time_span_hours

    return rate_per_hour, total_improvement


def count_training_iterations(config: str) -> int:
    """Count training iterations from model files."""
    iterations = 0
    board_type, players = config.rsplit('_', 1)
    players = players.replace('p', '')

    pattern = f"{board_type}_{players}p_*.pth"
    for model_file in MODELS_DIR.glob(pattern):
        iterations += 1

    # Also check for versioned models
    pattern2 = f"*{board_type}*{players}p*.pth"
    for model_file in MODELS_DIR.glob(pattern2):
        if model_file.name not in [m.name for m in MODELS_DIR.glob(pattern)]:
            iterations += 1

    return iterations


def update_metrics():
    """Update all Prometheus metrics from database."""
    if not HAS_PROMETHEUS:
        return

    # Get current ratings
    ratings = get_current_elo_ratings()

    # Track best per config
    best_per_config: Dict[str, float] = {}
    models_per_config: Dict[str, int] = {}

    for row in ratings:
        participant_id, board_type, num_players, rating, games, wins, losses, draws, last_update, ptype, version = row

        config = f"{board_type}_{num_players}p"

        # Determine model type
        if 'nn' in participant_id.lower() or 'ringrift' in participant_id.lower():
            model_type = 'neural'
        elif 'mcts' in participant_id.lower():
            model_type = 'mcts'
        elif 'heuristic' in participant_id.lower():
            model_type = 'heuristic'
        elif 'random' in participant_id.lower():
            model_type = 'random'
        else:
            model_type = 'other'

        # Short model name for labels
        model_short = participant_id.split('/')[-1][:50]

        # Update current Elo
        MODEL_ELO_CURRENT.labels(config=config, model=model_short, model_type=model_type).set(rating)

        # Update games played
        MODEL_GAMES_PLAYED.labels(config=config, model=model_short).set(games)

        # Calculate and update win rate
        total = wins + losses + draws
        if total > 0:
            win_rate = wins / total
            MODEL_WIN_RATE.labels(config=config, model=model_short).set(win_rate)

        # Track best per config
        if config not in best_per_config or rating > best_per_config[config]:
            best_per_config[config] = rating

        # Count models per config
        models_per_config[config] = models_per_config.get(config, 0) + 1

    # Update best Elo per config
    for config, best_elo in best_per_config.items():
        MODEL_ELO_BEST.labels(config=config).set(best_elo)
        MODELS_TOTAL.labels(config=config).set(models_per_config.get(config, 0))

    # Get history and calculate improvement rates
    history = get_elo_history(hours=24)

    for config, config_history in history.items():
        rate, total_improvement = calculate_improvement_rate(config, config_history)
        MODEL_ELO_IMPROVEMENT_RATE.labels(config=config).set(rate)
        MODEL_ELO_IMPROVEMENT_24H.labels(config=config).set(total_improvement)

    # Count training iterations
    for config in best_per_config.keys():
        iterations = count_training_iterations(config)
        TRAINING_ITERATIONS.labels(config=config).set(iterations)

    # Update data sync age (time since last rating update)
    if ratings:
        latest_update = max(r[8] for r in ratings if r[8])
        if latest_update:
            age = time.time() - latest_update
            DATA_SYNC_AGE.set(age)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""

    def do_GET(self):
        if self.path == '/metrics':
            # Update metrics before serving
            update_metrics()

            # Generate response
            output = generate_latest()
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.send_header('Content-Length', len(output))
            self.end_headers()
            self.wfile.write(output)
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def main():
    parser = argparse.ArgumentParser(description='Elo Metrics Exporter')
    parser.add_argument('--port', type=int, default=9092, help='Port to listen on')
    parser.add_argument('--once', action='store_true', help='Print metrics once and exit')
    args = parser.parse_args()

    if not HAS_PROMETHEUS:
        print("ERROR: prometheus_client not installed")
        return 1

    if args.once:
        update_metrics()
        print(generate_latest().decode())
        return 0

    # Start HTTP server
    server = HTTPServer(('0.0.0.0', args.port), MetricsHandler)
    print(f"Elo Metrics Exporter started on http://0.0.0.0:{args.port}/metrics")
    print(f"Database: {ELO_DB_PATH}")
    print(f"Models dir: {MODELS_DIR}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

    return 0


if __name__ == '__main__':
    exit(main())
