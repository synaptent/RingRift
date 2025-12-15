#!/usr/bin/env python3
"""Track Elo improvement over time with detailed analytics.

Usage:
    python scripts/track_elo_improvement.py [--db PATH] [--hours N] [--export CSV]
"""

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import sys

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Get database connection with row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


def show_current_standings(conn: sqlite3.Connection, limit: int = 20):
    """Show current top Elo standings by config."""
    print_section("CURRENT TOP ELO BY CONFIG")

    query = """
    SELECT board_type, num_players, model_id, rating, games_played, wins, losses,
           datetime(last_update, 'unixepoch', 'localtime') as last_seen
    FROM elo_ratings
    WHERE games_played >= 5
    ORDER BY rating DESC
    LIMIT ?
    """

    print(f"{'Rating':>8} | {'Config':15} | {'W/L':>10} | {'Games':>5} | Model")
    print("-" * 80)

    for r in conn.execute(query, (limit,)):
        config = f"{r['board_type'][:8]} {r['num_players']}p"
        wl = f"{r['wins']}/{r['losses']}"
        model = r['model_id'][:35] if r['model_id'] else 'N/A'
        print(f"{r['rating']:8.1f} | {config:15} | {wl:>10} | {r['games_played']:>5} | {model}")


def show_peak_elo_by_config(conn: sqlite3.Connection):
    """Show all-time peak Elo for each config."""
    print_section("ALL-TIME PEAK ELO BY CONFIG")

    query = """
    SELECT e.board_type, e.num_players,
           MAX(e.rating) as peak_rating,
           (SELECT model_id FROM elo_ratings e2
            WHERE e2.board_type = e.board_type AND e2.num_players = e.num_players
            ORDER BY rating DESC LIMIT 1) as best_model
    FROM elo_ratings e
    WHERE e.games_played >= 5
    GROUP BY e.board_type, e.num_players
    ORDER BY peak_rating DESC
    """

    print(f"{'Config':15} | {'Peak Elo':>10} | Best Model")
    print("-" * 70)

    for r in conn.execute(query):
        config = f"{r['board_type']} {r['num_players']}p"
        model = r['best_model'][:40] if r['best_model'] else 'N/A'
        print(f"{config:15} | {r['peak_rating']:>10.1f} | {model}")


def show_elo_progression(conn: sqlite3.Connection, hours: int = 24):
    """Show Elo progression over time."""
    print_section(f"ELO PROGRESSION (Last {hours} hours)")

    cutoff = datetime.now().timestamp() - (hours * 3600)

    query = """
    SELECT datetime(CAST(timestamp/3600 AS INTEGER)*3600, 'unixepoch', 'localtime') as hour,
           MAX(rating) as peak_elo,
           AVG(rating) as avg_elo,
           COUNT(DISTINCT model_id) as models_active,
           COUNT(*) as updates
    FROM rating_history
    WHERE timestamp >= ?
    GROUP BY CAST(timestamp/3600 AS INTEGER)
    ORDER BY timestamp DESC
    """

    print(f"{'Time':20} | {'Peak':>8} | {'Avg':>8} | {'Models':>6} | {'Updates':>7}")
    print("-" * 60)

    prev_peak = None
    for r in conn.execute(query, (cutoff,)):
        delta = ""
        if prev_peak is not None:
            d = r['peak_elo'] - prev_peak
            delta = f" ({d:+.0f})" if abs(d) > 0.5 else ""
        prev_peak = r['peak_elo']

        print(f"{r['hour']:20} | {r['peak_elo']:>8.1f}{delta:6} | {r['avg_elo']:>8.1f} | {r['models_active']:>6} | {r['updates']:>7}")


def show_model_improvement(conn: sqlite3.Connection, model_prefix: str = "nn"):
    """Show improvement trajectory for neural net models."""
    print_section(f"NEURAL NET MODEL IMPROVEMENT")

    query = """
    SELECT model_id,
           MIN(rating) as start_rating,
           MAX(rating) as peak_rating,
           (SELECT rating FROM rating_history r2
            WHERE r2.model_id = r.model_id
            ORDER BY timestamp DESC LIMIT 1) as current_rating,
           COUNT(*) as rating_updates,
           MIN(timestamp) as first_seen,
           MAX(timestamp) as last_seen
    FROM rating_history r
    WHERE model_id LIKE '%nn%' OR model_id LIKE '%ringrift%'
    GROUP BY model_id
    HAVING COUNT(*) >= 5
    ORDER BY peak_rating DESC
    LIMIT 15
    """

    print(f"{'Model':40} | {'Start':>7} | {'Peak':>7} | {'Now':>7} | {'Gain':>7}")
    print("-" * 80)

    for r in conn.execute(query):
        model = r['model_id'][:40]
        gain = r['peak_rating'] - r['start_rating']
        print(f"{model:40} | {r['start_rating']:>7.1f} | {r['peak_rating']:>7.1f} | {r['current_rating']:>7.1f} | {gain:>+7.1f}")


def show_training_impact(conn: sqlite3.Connection):
    """Analyze impact of training iterations."""
    print_section("TRAINING ITERATION ANALYSIS")

    # Extract iteration info from model names
    query = """
    SELECT model_id, rating, games_played
    FROM elo_ratings
    WHERE model_id LIKE '%baseline%' AND games_played >= 10
    ORDER BY rating DESC
    """

    iterations = {}
    for r in conn.execute(query):
        model_id = r['model_id']
        # Extract date/iteration from model name
        parts = model_id.split('_')
        if len(parts) >= 4:
            base_key = '_'.join(parts[:4])  # e.g., sq8_2p_nn_baseline
            date_parts = [p for p in parts if p.startswith('2025')]
            if date_parts:
                if base_key not in iterations:
                    iterations[base_key] = []
                iterations[base_key].append({
                    'model': model_id,
                    'rating': r['rating'],
                    'games': r['games_played'],
                    'iteration': len(date_parts)
                })

    for base_key, models in iterations.items():
        if len(models) > 1:
            print(f"\n{base_key}:")
            for m in sorted(models, key=lambda x: x['rating'], reverse=True)[:5]:
                print(f"  {m['rating']:7.1f} | {m['games']:4d} games | iter {m['iteration']} | {m['model'][-30:]}")


def show_summary_stats(conn: sqlite3.Connection):
    """Show summary statistics."""
    print_section("SUMMARY STATISTICS")

    stats = {}

    # Total participants
    stats['total_models'] = conn.execute(
        "SELECT COUNT(DISTINCT model_id) FROM elo_ratings"
    ).fetchone()[0]

    # Total games
    stats['total_games'] = conn.execute(
        "SELECT SUM(games_played)/2 FROM elo_ratings"
    ).fetchone()[0] or 0

    # Rating history entries
    stats['history_entries'] = conn.execute(
        "SELECT COUNT(*) FROM rating_history"
    ).fetchone()[0]

    # Peak Elo ever
    peak = conn.execute(
        "SELECT MAX(rating), model_id FROM elo_ratings WHERE games_played >= 10"
    ).fetchone()
    stats['peak_elo'] = peak[0] if peak else 0
    stats['peak_model'] = peak[1] if peak else 'N/A'

    # Elo gain (best model start to peak)
    gain = conn.execute("""
        SELECT MAX(rating) - MIN(rating) as gain, model_id
        FROM rating_history
        GROUP BY model_id
        HAVING COUNT(*) >= 10
        ORDER BY gain DESC
        LIMIT 1
    """).fetchone()
    stats['max_gain'] = gain[0] if gain else 0
    stats['gain_model'] = gain[1] if gain else 'N/A'

    print(f"Total Models:      {stats['total_models']}")
    print(f"Total Games:       {stats['total_games']:.0f}")
    print(f"History Entries:   {stats['history_entries']}")
    print(f"Peak Elo:          {stats['peak_elo']:.1f} ({stats['peak_model'][:30]})")
    print(f"Max Elo Gain:      {stats['max_gain']:.1f} ({stats['gain_model'][:30]})")


def export_to_csv(conn: sqlite3.Connection, output_path: str):
    """Export rating history to CSV for external analysis."""
    import csv

    query = """
    SELECT r.model_id, r.rating, r.games_played,
           datetime(r.timestamp, 'unixepoch') as time,
           e.board_type, e.num_players
    FROM rating_history r
    LEFT JOIN elo_ratings e ON r.model_id = e.model_id
    ORDER BY r.timestamp
    """

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_id', 'rating', 'games_played', 'timestamp', 'board_type', 'num_players'])
        for r in conn.execute(query):
            writer.writerow(r)

    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Track Elo improvement over time")
    parser.add_argument("--db", type=str, default="data/unified_elo.db",
                        help="Path to Elo database")
    parser.add_argument("--hours", type=int, default=24,
                        help="Hours of history to show")
    parser.add_argument("--export", type=str, default=None,
                        help="Export history to CSV file")
    args = parser.parse_args()

    db_path = AI_SERVICE_ROOT / args.db
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    conn = get_db_connection(str(db_path))

    if args.export:
        export_to_csv(conn, args.export)
        return

    show_summary_stats(conn)
    show_current_standings(conn)
    show_peak_elo_by_config(conn)
    show_elo_progression(conn, args.hours)
    show_model_improvement(conn)
    show_training_impact(conn)

    conn.close()


if __name__ == "__main__":
    main()
