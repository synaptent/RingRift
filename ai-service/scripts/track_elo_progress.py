#!/usr/bin/env python3
"""Track Elo progress over time for research documentation.

Creates a CSV file tracking Elo ratings per config at each snapshot.
Run daily via cron to build improvement timeline.

Usage:
    python scripts/track_elo_progress.py              # Add snapshot
    python scripts/track_elo_progress.py --report     # Show progress report
    python scripts/track_elo_progress.py --plot       # Generate plot (requires matplotlib)
"""

import argparse
import csv
import sqlite3
from datetime import datetime
from pathlib import Path

TRACKING_FILE = Path("data/elo_progress.csv")
ELO_DB = Path("data/unified_elo.db")

CONFIGS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]


def get_current_elos() -> dict:
    """Get current Elo ratings for all configs."""
    if not ELO_DB.exists():
        return {}
    
    conn = sqlite3.connect(ELO_DB)
    elos = {}
    
    for config in CONFIGS:
        board, players = config.rsplit("_", 1)
        result = conn.execute("""
            SELECT rating, games_played FROM elo_ratings 
            WHERE participant_id LIKE ? AND archived_at IS NULL
            ORDER BY games_played DESC LIMIT 1
        """, (f"%{board}%{players}%",)).fetchone()
        
        if result:
            elos[config] = {"elo": result[0], "games": result[1]}
        else:
            elos[config] = {"elo": None, "games": 0}
    
    conn.close()
    return elos


def add_snapshot():
    """Add current Elo snapshot to tracking file."""
    elos = get_current_elos()
    timestamp = datetime.now().isoformat()
    
    # Create file with header if doesn't exist
    if not TRACKING_FILE.exists():
        TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRACKING_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + CONFIGS)
    
    # Add snapshot
    with open(TRACKING_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        row = [timestamp] + [elos.get(c, {}).get("elo", "") for c in CONFIGS]
        writer.writerow(row)
    
    print(f"Snapshot added at {timestamp}")
    for config in CONFIGS:
        elo = elos.get(config, {}).get("elo")
        games = elos.get(config, {}).get("games", 0)
        if elo:
            print(f"  {config}: {elo:.0f} Elo ({games} games)")
        else:
            print(f"  {config}: NO DATA")


def show_report():
    """Show progress report from tracking file."""
    if not TRACKING_FILE.exists():
        print("No tracking data yet. Run without --report to add first snapshot.")
        return
    
    with open(TRACKING_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if len(rows) < 2:
        print("Need at least 2 snapshots to show progress.")
        print(f"Current snapshots: {len(rows)}")
        return
    
    first = rows[0]
    last = rows[-1]
    
    print(f"=== Elo Progress Report ===")
    print(f"Period: {first['timestamp'][:10]} to {last['timestamp'][:10]}")
    print(f"Snapshots: {len(rows)}")
    print()
    print(f"{'Config':<15} {'Start':>8} {'Current':>8} {'Change':>8}")
    print("-" * 45)
    
    total_change = 0
    configs_with_data = 0
    
    for config in CONFIGS:
        start = first.get(config, "")
        end = last.get(config, "")
        
        if start and end:
            start_val = float(start)
            end_val = float(end)
            change = end_val - start_val
            total_change += change
            configs_with_data += 1
            
            change_str = f"+{change:.0f}" if change >= 0 else f"{change:.0f}"
            print(f"{config:<15} {start_val:>8.0f} {end_val:>8.0f} {change_str:>8}")
        else:
            print(f"{config:<15} {'--':>8} {end or '--':>8} {'--':>8}")
    
    if configs_with_data > 0:
        avg_change = total_change / configs_with_data
        print("-" * 45)
        print(f"{'Average':<15} {'':>8} {'':>8} {'+' if avg_change >= 0 else ''}{avg_change:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Track Elo progress over time")
    parser.add_argument("--report", action="store_true", help="Show progress report")
    parser.add_argument("--plot", action="store_true", help="Generate progress plot")
    args = parser.parse_args()
    
    if args.report:
        show_report()
    elif args.plot:
        print("Plot generation requires matplotlib. Install with: pip install matplotlib")
        # TODO: Add plot generation
    else:
        add_snapshot()


if __name__ == "__main__":
    main()
