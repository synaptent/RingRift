#!/usr/bin/env python3
"""ELO Monitoring Dashboard.

.. deprecated:: 2025-12
    Use the unified dashboard instead::

        python scripts/dashboard.py elo

Real-time terminal dashboard showing:
- Top models and production progress
- Tournament activity and throughput
- Node health status
- Config coverage

Usage:
    python scripts/elo_dashboard.py           # One-time view
    python scripts/elo_dashboard.py --live    # Live updates
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.config.thresholds import (
    PRODUCTION_ELO_THRESHOLD,
    PRODUCTION_MIN_GAMES,
)

DEFAULT_DB = AI_SERVICE_ROOT / "data" / "unified_elo.db"

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')


def progress_bar(current: float, target: float, width: int = 20) -> str:
    """Create a progress bar."""
    pct = min(1.0, current / target) if target > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    color = GREEN if pct >= 1 else YELLOW if pct >= 0.8 else RESET
    return f"{color}{bar}{RESET} {pct*100:5.1f}%"


def get_dashboard_data(db_path: Path) -> dict:
    """Gather all dashboard data."""
    if not db_path.exists():
        return {"error": "Database not found"}
    
    conn = sqlite3.connect(str(db_path))
    data = {}
    
    # Top models
    cursor = conn.execute("""
        SELECT participant_id, rating, games_played
        FROM elo_ratings
        WHERE participant_id NOT LIKE 'baseline_%'
        ORDER BY rating DESC
        LIMIT 10
    """)
    data["top_models"] = [
        {"id": r[0], "elo": r[1], "games": r[2]}
        for r in cursor.fetchall()
    ]
    
    # Total stats
    cursor = conn.execute("""
        SELECT COUNT(*), SUM(games_played)
        FROM elo_ratings
        WHERE participant_id NOT LIKE 'baseline_%'
    """)
    row = cursor.fetchone()
    data["total_models"] = row[0] or 0
    data["total_games"] = row[1] or 0
    
    # Production ready
    cursor = conn.execute("""
        SELECT COUNT(*)
        FROM elo_ratings
        WHERE rating >= ? AND games_played >= ?
          AND participant_id NOT LIKE 'baseline_%'
    """, (PRODUCTION_ELO_THRESHOLD, PRODUCTION_MIN_GAMES))
    data["production_ready"] = cursor.fetchone()[0]
    
    # Recent games (last hour)
    cursor = conn.execute("""
        SELECT COUNT(*)
        FROM match_history
        WHERE timestamp > ?
    """, (time.time() - 3600,))
    data["games_last_hour"] = cursor.fetchone()[0]
    
    # Games by config
    cursor = conn.execute("""
        SELECT board_type, num_players, COUNT(*)
        FROM match_history
        GROUP BY board_type, num_players
    """)
    data["config_games"] = {
        f"{r[0]}_{r[1]}p": r[2]
        for r in cursor.fetchall()
    }
    
    conn.close()
    return data


def render_dashboard(data: dict) -> str:
    """Render dashboard as string."""
    lines = []
    width = 80
    
    # Header
    lines.append(f"\n{BOLD}{'═' * width}{RESET}")
    lines.append(f"{BOLD}{CYAN}{'ELO MONITORING DASHBOARD':^{width}}{RESET}")
    lines.append(f"{BOLD}{'═' * width}{RESET}")
    lines.append(f"  Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    if "error" in data:
        lines.append(f"  {RED}Error: {data['error']}{RESET}")
        return "\n".join(lines)
    
    # Summary stats
    lines.append(f"{BOLD}  SUMMARY{RESET}")
    lines.append(f"  {'─' * 40}")
    lines.append(f"  Total Models: {data['total_models']:<10} Total Games: {data['total_games']}")
    lines.append(f"  Production Ready: {data['production_ready']:<7} Games/Hour: {data['games_last_hour']}")
    lines.append("")
    
    # Top models with progress
    lines.append(f"{BOLD}  TOP MODELS (Production: ELO≥{PRODUCTION_ELO_THRESHOLD}, Games≥{PRODUCTION_MIN_GAMES}){RESET}")
    lines.append(f"  {'─' * 70}")
    lines.append(f"  {'#':<3} {'Model':<35} {'ELO':>7} {'Games':>6}  {'Progress'}")
    
    for i, model in enumerate(data["top_models"][:7], 1):
        model_id = model["id"][:35]
        elo = model["elo"]
        games = model["games"]
        
        # Progress based on both ELO and games
        elo_pct = min(1, elo / PRODUCTION_ELO_THRESHOLD)
        games_pct = min(1, games / PRODUCTION_MIN_GAMES)
        overall_pct = min(elo_pct, games_pct)
        
        if elo >= PRODUCTION_ELO_THRESHOLD and games >= PRODUCTION_MIN_GAMES:
            status = f"{GREEN}✓ PRODUCTION{RESET}"
        elif elo >= PRODUCTION_ELO_THRESHOLD:
            status = f"{YELLOW}+{PRODUCTION_MIN_GAMES - games} games{RESET}"
        elif games >= PRODUCTION_MIN_GAMES:
            status = f"{YELLOW}+{PRODUCTION_ELO_THRESHOLD - elo:.1f} ELO{RESET}"
        else:
            status = progress_bar(overall_pct, 1, 15)
        
        lines.append(f"  {i:<3} {model_id:<35} {elo:>7.1f} {games:>6}  {status}")
    
    lines.append("")
    
    # Config coverage
    lines.append(f"{BOLD}  CONFIG COVERAGE{RESET}")
    lines.append(f"  {'─' * 40}")
    
    configs = [
        ("square8_2p", "Sq8 2P"), ("square8_3p", "Sq8 3P"), ("square8_4p", "Sq8 4P"),
        ("square19_2p", "Sq19 2P"), ("square19_3p", "Sq19 3P"), ("square19_4p", "Sq19 4P"),
        ("hex8_2p", "Hex8 2P"), ("hex8_3p", "Hex8 3P"), ("hex8_4p", "Hex8 4P"),
        ("hexagonal_2p", "Hexnl 2P"), ("hexagonal_3p", "Hexnl 3P"), ("hexagonal_4p", "Hexnl 4P"),
    ]
    
    row1 = "  "
    row2 = "  "
    for config_key, label in configs[:6]:
        games = data["config_games"].get(config_key, 0)
        color = GREEN if games >= 500 else YELLOW if games >= 100 else RED
        row1 += f"{label:<10}"
        row2 += f"{color}{games:<10}{RESET}"
    
    lines.append(row1)
    lines.append(row2)
    
    row1 = "  "
    row2 = "  "
    for config_key, label in configs[6:]:
        games = data["config_games"].get(config_key, 0)
        color = GREEN if games >= 500 else YELLOW if games >= 100 else RED
        row1 += f"{label:<10}"
        row2 += f"{color}{games:<10}{RESET}"
    
    lines.append(row1)
    lines.append(row2)
    lines.append("")
    
    lines.append(f"{'═' * width}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ELO Monitoring Dashboard")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--live", action="store_true", help="Live updates")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval")
    
    args = parser.parse_args()
    
    if args.live:
        try:
            while True:
                clear_screen()
                data = get_dashboard_data(args.db)
                print(render_dashboard(data))
                print(f"\n  Refreshing every {args.interval}s (Ctrl+C to exit)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        data = get_dashboard_data(args.db)
        print(render_dashboard(data))


if __name__ == "__main__":
    main()
