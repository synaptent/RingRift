#!/usr/bin/env python3
"""
Elo Progression Monitor for RingRift AI Training

Usage:
    python scripts/monitor_elo.py              # One-time status
    python scripts/monitor_elo.py --watch      # Continuous monitoring
    python scripts/monitor_elo.py --watch -i 60  # Update every 60s
"""

import argparse
import sqlite3
import time
import sys
from datetime import datetime
from pathlib import Path

from app.config.thresholds import PRODUCTION_ELO_THRESHOLD

# Target Elo ratings from improvement plan
TARGETS = {
    "square8_2p": 1900,
    "hex8_2p": 1750,
    "square19_2p": 1800,
    "hex8_3p": 1700,
    "square8_3p": 1700,
    "square19_3p": 1700,
    "hexagonal_2p": 1700,
    "hexagonal_3p": 1700,
    "hex8_4p": PRODUCTION_ELO_THRESHOLD,
    "square8_4p": PRODUCTION_ELO_THRESHOLD,
    "square19_4p": PRODUCTION_ELO_THRESHOLD,
    "hexagonal_4p": 1600,
}

BASELINE = 1500


def get_elo_data(db_path: Path) -> dict:
    """Get current Elo ratings from database."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get best rating per config (canonical models only)
    cursor.execute("""
        SELECT
            board_type || '_' || num_players || 'p' as config,
            MAX(rating) as best_rating,
            SUM(games_played) as total_games
        FROM elo_ratings
        WHERE participant_id LIKE 'canonical%' OR participant_id LIKE 'ringrift_best%'
        GROUP BY board_type, num_players
        ORDER BY best_rating DESC
    """)

    results = {}
    for row in cursor.fetchall():
        config, rating, games = row
        results[config] = {"rating": rating or BASELINE, "games": games or 0}

    conn.close()
    return results


def get_game_counts(db_path: Path) -> dict:
    """Get game counts from selfplay databases."""
    counts = {}
    data_dir = db_path.parent.parent / "selfplay"

    if not data_dir.exists():
        return counts

    # Search for game databases
    for db_file in data_dir.rglob("*.db"):
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM games WHERE outcome IS NOT NULL")
            count = cursor.fetchone()[0]

            # Extract config from path
            parts = db_file.parts
            for part in parts:
                if any(b in part for b in ["hex8", "square8", "square19", "hexagonal"]):
                    if part not in counts:
                        counts[part] = 0
                    counts[part] += count
                    break
            conn.close()
        except (OSError, sqlite3.Error):
            continue

    return counts


def print_status(elo_data: dict, game_counts: dict, clear: bool = False):
    """Print current status."""
    if clear:
        print("\033[2J\033[H", end="")  # Clear screen

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{'='*70}")
    print(f"RingRift AI Training Monitor - {now}")
    print(f"{'='*70}")
    print()

    # Header
    print(f"{'Config':<15} {'Current':>8} {'Target':>8} {'Gap':>8} {'Games':>10} {'Status':<12}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*12}")

    # Sort by gap (worst first)
    configs = []
    for config, target in TARGETS.items():
        current = elo_data.get(config, {}).get("rating", BASELINE)
        games = elo_data.get(config, {}).get("games", 0)
        gap = current - target
        configs.append((config, current, target, gap, games))

    configs.sort(key=lambda x: x[3])  # Sort by gap ascending (most negative first)

    total_gap = 0
    competitive_count = 0

    for config, current, target, gap, games in configs:
        total_gap += gap

        # Status indicators
        if current >= target:
            status = "TARGET MET"
            competitive_count += 1
        elif current >= target - 100:
            status = "CLOSE"
        elif current > BASELINE:
            status = "IMPROVING"
        elif games < 100:
            status = "CRITICAL"
        else:
            status = "UNDERTRAINED"

        # Color coding (ANSI)
        if status == "TARGET MET":
            color = "\033[92m"  # Green
        elif status == "CLOSE":
            color = "\033[93m"  # Yellow
        elif status == "CRITICAL":
            color = "\033[91m"  # Red
        else:
            color = "\033[0m"   # Default

        reset = "\033[0m"

        print(f"{config:<15} {current:>8.0f} {target:>8} {gap:>+8.0f} {games:>10} {color}{status:<12}{reset}")

    print()
    print(f"{'='*70}")
    print(f"Summary: {competitive_count}/12 configs at target | Avg gap: {total_gap/12:+.0f} Elo")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Monitor RingRift AI Elo progression")
    parser.add_argument("--watch", "-w", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Update interval (seconds)")
    parser.add_argument("--db", type=str, default="data/unified_elo.db", help="Elo database path")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = Path(__file__).parent.parent / db_path

    if args.watch:
        print("Starting continuous monitoring (Ctrl+C to stop)...")
        try:
            while True:
                elo_data = get_elo_data(db_path)
                game_counts = get_game_counts(db_path)
                print_status(elo_data, game_counts, clear=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        elo_data = get_elo_data(db_path)
        game_counts = get_game_counts(db_path)
        print_status(elo_data, game_counts)


if __name__ == "__main__":
    main()
