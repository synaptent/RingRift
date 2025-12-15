#!/usr/bin/env python3
"""Data Quality Monitor for selfplay training data.

Monitors training data for quality issues and exports Prometheus metrics.
Designed to catch contamination issues like timeout games before they affect training.

Usage:
    python scripts/data_quality_monitor.py --port 9093

    # One-shot audit mode
    python scripts/data_quality_monitor.py --audit-only

Metrics exported:
    - ringrift_data_games_total: Total games in database
    - ringrift_data_games_with_winner: Games that ended with a winner
    - ringrift_data_games_draws: Games that ended in draw
    - ringrift_data_games_high_moves: Games with moves > 1000 (potential timeout)
    - ringrift_data_games_max_moves: Games hitting exact move limits (500, 10000)
    - ringrift_data_draw_rate: Current draw rate (0-1)
    - ringrift_data_avg_moves: Average moves per game
    - ringrift_data_quality_score: Overall quality score (0-100)

Alerts:
    - High draw rate (>20%)
    - Games hitting move limits
    - Sudden changes in game statistics
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Prometheus client
try:
    from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

# Paths
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
DATA_DIR = AI_SERVICE_ROOT / "data"

# Alert thresholds
DRAW_RATE_THRESHOLD = 0.20  # Alert if draw rate > 20%
HIGH_MOVES_THRESHOLD = 1000  # Flag games with > 1000 moves
MOVE_LIMIT_VALUES = [500, 10000]  # Known move limits to watch for


class DataQualityMetrics:
    """Collect and expose data quality metrics."""

    def __init__(self):
        if HAS_PROMETHEUS:
            self.games_total = Gauge(
                'ringrift_data_games_total',
                'Total games in database',
                ['db_name', 'board_type']
            )
            self.games_with_winner = Gauge(
                'ringrift_data_games_with_winner',
                'Games that ended with a winner',
                ['db_name', 'board_type']
            )
            self.games_draws = Gauge(
                'ringrift_data_games_draws',
                'Games that ended in draw',
                ['db_name', 'board_type']
            )
            self.games_high_moves = Gauge(
                'ringrift_data_games_high_moves',
                'Games with suspiciously high move counts',
                ['db_name', 'threshold']
            )
            self.games_at_limit = Gauge(
                'ringrift_data_games_at_limit',
                'Games hitting exact move limits',
                ['db_name', 'limit']
            )
            self.draw_rate = Gauge(
                'ringrift_data_draw_rate',
                'Current draw rate',
                ['db_name']
            )
            self.avg_moves = Gauge(
                'ringrift_data_avg_moves',
                'Average moves per game',
                ['db_name', 'board_type']
            )
            self.quality_score = Gauge(
                'ringrift_data_quality_score',
                'Overall data quality score (0-100)',
                ['db_name']
            )
            self.last_check = Gauge(
                'ringrift_data_last_check_timestamp',
                'Timestamp of last quality check',
                ['db_name']
            )

    def analyze_database(self, db_path: Path) -> Dict[str, Any]:
        """Analyze a single database for quality metrics."""
        db_name = db_path.stem
        results = {
            'db_name': db_name,
            'db_path': str(db_path),
            'timestamp': datetime.now().isoformat(),
            'board_types': {},
            'alerts': [],
            'quality_score': 100,
        }

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Check if games table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
            if not cursor.fetchone():
                results['error'] = 'No games table found'
                return results

            # Get column info
            cursor.execute("PRAGMA table_info(games)")
            columns = {row['name'] for row in cursor.fetchall()}

            has_winner = 'winner' in columns
            has_total_moves = 'total_moves' in columns
            has_board_type = 'board_type' in columns

            # Total games
            cursor.execute("SELECT COUNT(*) as cnt FROM games")
            total = cursor.fetchone()['cnt']
            results['total_games'] = total

            if total == 0:
                results['quality_score'] = 0
                return results

            # Per board type stats
            if has_board_type:
                cursor.execute("SELECT DISTINCT board_type FROM games WHERE board_type IS NOT NULL")
                board_types = [row['board_type'] for row in cursor.fetchall()]
            else:
                board_types = ['unknown']

            for board_type in board_types:
                bt_filter = f"WHERE board_type = '{board_type}'" if has_board_type and board_type != 'unknown' else ""

                # Games with winner vs draws
                if has_winner:
                    cursor.execute(f"SELECT COUNT(*) as cnt FROM games {bt_filter} {'AND' if bt_filter else 'WHERE'} winner IS NOT NULL")
                    with_winner = cursor.fetchone()['cnt']
                    cursor.execute(f"SELECT COUNT(*) as cnt FROM games {bt_filter} {'AND' if bt_filter else 'WHERE'} winner IS NULL")
                    draws = cursor.fetchone()['cnt']
                else:
                    with_winner = total
                    draws = 0

                # Average moves
                if has_total_moves:
                    cursor.execute(f"SELECT AVG(total_moves) as avg, MAX(total_moves) as max FROM games {bt_filter}")
                    row = cursor.fetchone()
                    avg_moves = row['avg'] or 0
                    max_moves = row['max'] or 0
                else:
                    avg_moves = 0
                    max_moves = 0

                bt_total = with_winner + draws
                draw_rate = draws / bt_total if bt_total > 0 else 0

                results['board_types'][board_type] = {
                    'total': bt_total,
                    'with_winner': with_winner,
                    'draws': draws,
                    'draw_rate': draw_rate,
                    'avg_moves': avg_moves,
                    'max_moves': max_moves,
                }

                # Export metrics
                if HAS_PROMETHEUS:
                    self.games_total.labels(db_name=db_name, board_type=board_type).set(bt_total)
                    self.games_with_winner.labels(db_name=db_name, board_type=board_type).set(with_winner)
                    self.games_draws.labels(db_name=db_name, board_type=board_type).set(draws)
                    self.avg_moves.labels(db_name=db_name, board_type=board_type).set(avg_moves)

                # Check for alerts
                if draw_rate > DRAW_RATE_THRESHOLD:
                    results['alerts'].append({
                        'severity': 'warning',
                        'type': 'high_draw_rate',
                        'message': f"High draw rate {draw_rate:.1%} for {board_type}",
                        'board_type': board_type,
                        'value': draw_rate,
                    })
                    results['quality_score'] -= 20

            # Check for move limit hits
            if has_total_moves and has_winner:
                for limit in MOVE_LIMIT_VALUES:
                    cursor.execute(f"SELECT COUNT(*) as cnt FROM games WHERE total_moves = {limit} AND winner IS NULL")
                    at_limit = cursor.fetchone()['cnt']
                    if at_limit > 0:
                        results['alerts'].append({
                            'severity': 'critical',
                            'type': 'games_at_move_limit',
                            'message': f"{at_limit} games hit {limit} move limit (timeout contamination)",
                            'limit': limit,
                            'count': at_limit,
                        })
                        # Severe penalty for timeout contamination
                        results['quality_score'] -= min(50, at_limit // 100)

                    if HAS_PROMETHEUS:
                        self.games_at_limit.labels(db_name=db_name, limit=str(limit)).set(at_limit)

                # Check for high move games
                cursor.execute(f"SELECT COUNT(*) as cnt FROM games WHERE total_moves > {HIGH_MOVES_THRESHOLD}")
                high_moves = cursor.fetchone()['cnt']
                if HAS_PROMETHEUS:
                    self.games_high_moves.labels(db_name=db_name, threshold=str(HIGH_MOVES_THRESHOLD)).set(high_moves)

            # Overall draw rate and quality score
            if has_winner:
                cursor.execute("SELECT COUNT(*) as cnt FROM games WHERE winner IS NULL")
                total_draws = cursor.fetchone()['cnt']
                overall_draw_rate = total_draws / total if total > 0 else 0
            else:
                overall_draw_rate = 0

            if HAS_PROMETHEUS:
                self.draw_rate.labels(db_name=db_name).set(overall_draw_rate)
                self.quality_score.labels(db_name=db_name).set(max(0, results['quality_score']))
                self.last_check.labels(db_name=db_name).set(time.time())

            results['overall_draw_rate'] = overall_draw_rate
            results['quality_score'] = max(0, results['quality_score'])

            conn.close()

        except Exception as e:
            results['error'] = str(e)
            results['quality_score'] = 0

        return results


def find_databases(data_dir: Path) -> List[Path]:
    """Find all game databases to analyze."""
    dbs = []

    # Check common locations
    for pattern in ['*.db', 'games/*.db', 'selfplay/*.db', 'canonical/*.db']:
        dbs.extend(data_dir.glob(pattern))

    # Deduplicate
    return list(set(dbs))


def run_audit(data_dir: Path, verbose: bool = True) -> Dict[str, Any]:
    """Run a one-shot audit of all databases."""
    metrics = DataQualityMetrics()
    databases = find_databases(data_dir)

    if verbose:
        print(f"Found {len(databases)} databases to analyze")
        print("=" * 60)

    all_results = []
    total_games = 0
    total_draws = 0
    total_alerts = 0

    for db_path in sorted(databases):
        result = metrics.analyze_database(db_path)
        all_results.append(result)

        if 'error' in result:
            if verbose:
                print(f"  {db_path.name}: ERROR - {result['error']}")
            continue

        total_games += result.get('total_games', 0)
        total_alerts += len(result.get('alerts', []))

        for bt_data in result.get('board_types', {}).values():
            total_draws += bt_data.get('draws', 0)

        if verbose:
            print(f"\n{db_path.name}:")
            print(f"  Total games: {result.get('total_games', 0):,}")
            print(f"  Quality score: {result.get('quality_score', 0)}/100")

            for bt, data in result.get('board_types', {}).items():
                print(f"  {bt}: {data['total']:,} games, {data['draw_rate']:.1%} draws, avg {data['avg_moves']:.0f} moves")

            for alert in result.get('alerts', []):
                severity = alert['severity'].upper()
                print(f"  [{severity}] {alert['message']}")

    summary = {
        'databases_checked': len(databases),
        'total_games': total_games,
        'total_draws': total_draws,
        'overall_draw_rate': total_draws / total_games if total_games > 0 else 0,
        'total_alerts': total_alerts,
        'results': all_results,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Databases checked: {len(databases)}")
        print(f"Total games: {total_games:,}")
        print(f"Total draws: {total_draws:,} ({summary['overall_draw_rate']:.1%})")
        print(f"Total alerts: {total_alerts}")

    return summary


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics."""

    metrics: DataQualityMetrics
    data_dir: Path
    last_refresh: float = 0
    refresh_interval: int = 60

    def do_GET(self):
        if self.path == '/metrics':
            # Refresh metrics if needed
            now = time.time()
            if now - MetricsHandler.last_refresh > MetricsHandler.refresh_interval:
                databases = find_databases(MetricsHandler.data_dir)
                for db_path in databases:
                    MetricsHandler.metrics.analyze_database(db_path)
                MetricsHandler.last_refresh = now

            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(generate_latest())
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logging


def run_server(port: int, data_dir: Path, refresh_interval: int = 60):
    """Run the metrics server."""
    if not HAS_PROMETHEUS:
        print("ERROR: prometheus_client not installed. Run: pip install prometheus_client")
        return

    MetricsHandler.metrics = DataQualityMetrics()
    MetricsHandler.data_dir = data_dir
    MetricsHandler.refresh_interval = refresh_interval

    # Initial scan
    databases = find_databases(data_dir)
    print(f"Monitoring {len(databases)} databases")
    for db_path in databases:
        result = MetricsHandler.metrics.analyze_database(db_path)
        print(f"  {db_path.name}: {result.get('total_games', 0):,} games, score={result.get('quality_score', 0)}")
    MetricsHandler.last_refresh = time.time()

    server = HTTPServer(('0.0.0.0', port), MetricsHandler)
    print(f"\nData quality metrics server running on http://0.0.0.0:{port}/metrics")
    print(f"Refresh interval: {refresh_interval}s")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Data Quality Monitor")
    parser.add_argument('--port', type=int, default=9093, help='Port for metrics server')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR), help='Data directory to monitor')
    parser.add_argument('--audit-only', action='store_true', help='Run one-shot audit and exit')
    parser.add_argument('--refresh-interval', type=int, default=60, help='Metrics refresh interval in seconds')
    parser.add_argument('--output', type=str, help='Output JSON file for audit results')

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if args.audit_only:
        results = run_audit(data_dir, verbose=True)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        run_server(args.port, data_dir, args.refresh_interval)


if __name__ == '__main__':
    main()
