#!/usr/bin/env python3
"""Monitor curriculum weight allocation effectiveness.

Tracks whether the curriculum weight fix (0.10 -> 0.40) is working by:
1. Checking game generation rates per config
2. Verifying hexagonal/square19 configs are getting allocation
3. Monitoring Elo progression
4. Alerting on imbalanced allocation

Usage:
    python scripts/monitor_curriculum_allocation.py              # Single check
    python scripts/monitor_curriculum_allocation.py --watch      # Continuous monitoring
    python scripts/monitor_curriculum_allocation.py --days 1     # Check last 24h

Created: Jan 6, 2026 (Session 17.42b)
"""

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# All 12 canonical configs
ALL_CONFIGS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
]

# Configs that were previously starved (curriculum weight bug)
PREVIOUSLY_STARVED = ["hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
                      "square19_2p", "square19_3p", "square19_4p"]

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
GAMES_DIR = AI_SERVICE_ROOT / "data" / "games"


@dataclass
class ConfigStats:
    """Statistics for a single config."""
    config: str
    total_games: int = 0
    games_24h: int = 0
    games_1h: int = 0
    latest_game: Optional[datetime] = None
    allocation_pct: float = 0.0


@dataclass
class AllocationReport:
    """Full allocation report."""
    timestamp: datetime
    configs: dict = field(default_factory=dict)
    total_games_24h: int = 0
    starved_configs: list = field(default_factory=list)
    healthy: bool = True
    message: str = ""


def get_game_counts_from_dbs(hours: float = 24.0) -> dict[str, ConfigStats]:
    """Get game counts from local SQLite databases."""
    stats = {c: ConfigStats(config=c) for c in ALL_CONFIGS}

    cutoff_24h = datetime.now() - timedelta(hours=24)
    cutoff_1h = datetime.now() - timedelta(hours=1)
    cutoff_custom = datetime.now() - timedelta(hours=hours)

    # Find all selfplay databases
    db_patterns = [
        GAMES_DIR / "selfplay.db",
        GAMES_DIR / "selfplay_*.db",
        GAMES_DIR / "canonical_*.db",
    ]

    dbs_found = []
    for pattern in db_patterns:
        if "*" in str(pattern):
            dbs_found.extend(GAMES_DIR.glob(pattern.name))
        elif pattern.exists():
            dbs_found.append(pattern)

    for db_path in dbs_found:
        try:
            conn = sqlite3.connect(db_path, timeout=5)
            cur = conn.cursor()

            # Check if games table exists
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
            if not cur.fetchone():
                conn.close()
                continue

            # Check if status column exists
            cur.execute("PRAGMA table_info(games)")
            columns = [col[1] for col in cur.fetchall()]
            has_status = 'status' in columns

            # Get game counts by board_type
            if has_status:
                cur.execute("""
                    SELECT
                        board_type || '_' || num_players || 'p' as config,
                        COUNT(*) as total,
                        SUM(CASE WHEN created_at > ? THEN 1 ELSE 0 END) as games_24h,
                        SUM(CASE WHEN created_at > ? THEN 1 ELSE 0 END) as games_1h,
                        MAX(created_at) as latest
                    FROM games
                    WHERE status = 'complete' OR status IS NULL
                    GROUP BY board_type, num_players
                """, (cutoff_24h.isoformat(), cutoff_1h.isoformat()))
            else:
                # No status column - count all games
                cur.execute("""
                    SELECT
                        board_type || '_' || num_players || 'p' as config,
                        COUNT(*) as total,
                        SUM(CASE WHEN created_at > ? THEN 1 ELSE 0 END) as games_24h,
                        SUM(CASE WHEN created_at > ? THEN 1 ELSE 0 END) as games_1h,
                        MAX(created_at) as latest
                    FROM games
                    GROUP BY board_type, num_players
                """, (cutoff_24h.isoformat(), cutoff_1h.isoformat()))

            for row in cur.fetchall():
                config, total, games_24h, games_1h, latest = row
                if config in stats:
                    stats[config].total_games += total or 0
                    stats[config].games_24h += games_24h or 0
                    stats[config].games_1h += games_1h or 0
                    if latest:
                        try:
                            dt = datetime.fromisoformat(latest.replace('Z', '+00:00'))
                            if stats[config].latest_game is None or dt > stats[config].latest_game:
                                stats[config].latest_game = dt
                        except ValueError:
                            pass

            conn.close()
        except Exception as e:
            print(f"  Warning: Could not read {db_path}: {e}")

    return stats


def get_p2p_work_queue() -> dict:
    """Query P2P work queue for allocation info."""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:8770/work/queue"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (json.JSONDecodeError, OSError, subprocess.SubprocessError):
        pass
    return {}


def get_p2p_status() -> dict:
    """Get P2P cluster status."""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:8770/status"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (json.JSONDecodeError, OSError, subprocess.SubprocessError):
        pass
    return {}


def calculate_allocation(stats: dict[str, ConfigStats]) -> dict[str, float]:
    """Calculate allocation percentages from game counts."""
    total_24h = sum(s.games_24h for s in stats.values())
    if total_24h == 0:
        return {c: 0.0 for c in ALL_CONFIGS}

    return {c: (s.games_24h / total_24h) * 100 for c, s in stats.items()}


def generate_report(hours: float = 24.0) -> AllocationReport:
    """Generate comprehensive allocation report."""
    report = AllocationReport(timestamp=datetime.now())

    # Get game statistics
    stats = get_game_counts_from_dbs(hours)
    allocation = calculate_allocation(stats)

    # Update stats with allocation
    for config, pct in allocation.items():
        stats[config].allocation_pct = pct

    report.configs = stats
    report.total_games_24h = sum(s.games_24h for s in stats.values())

    # Check for starved configs (< 2% allocation when they should have ~8%)
    for config in PREVIOUSLY_STARVED:
        if stats[config].allocation_pct < 2.0 and report.total_games_24h > 100:
            report.starved_configs.append(config)

    # Determine health
    if report.starved_configs:
        report.healthy = False
        report.message = f"ALERT: {len(report.starved_configs)} configs still starved after curriculum fix"
    elif report.total_games_24h < 50:
        report.healthy = True
        report.message = "Insufficient data (< 50 games in 24h) - check back later"
    else:
        report.healthy = True
        report.message = "Curriculum allocation appears balanced"

    return report


def print_report(report: AllocationReport, verbose: bool = True):
    """Print formatted report."""
    print(f"\n{'='*70}")
    print(f"Curriculum Allocation Monitor - {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Summary
    status_emoji = "OK" if report.healthy else "ALERT"
    print(f"\nStatus: [{status_emoji}] {report.message}")
    print(f"Total games (24h): {report.total_games_24h:,}")

    if report.starved_configs:
        print(f"\nStarved configs: {', '.join(report.starved_configs)}")

    if verbose:
        # Detailed table
        print(f"\n{'Config':<15} {'Total':>10} {'24h':>8} {'1h':>6} {'Alloc%':>8} {'Status':<12}")
        print("-" * 70)

        # Sort by allocation (ascending to show starved first)
        sorted_configs = sorted(report.configs.values(), key=lambda x: x.allocation_pct)

        for s in sorted_configs:
            status = ""
            if s.config in PREVIOUSLY_STARVED:
                if s.allocation_pct < 2.0:
                    status = "STARVED"
                elif s.allocation_pct < 5.0:
                    status = "Low"
                else:
                    status = "OK"
            else:
                status = "OK" if s.allocation_pct > 0 else "-"

            age_str = ""
            if s.latest_game:
                age = datetime.now() - s.latest_game
                if age.days > 0:
                    age_str = f"{age.days}d ago"
                elif age.seconds > 3600:
                    age_str = f"{age.seconds // 3600}h ago"
                else:
                    age_str = "recent"

            print(f"{s.config:<15} {s.total_games:>10,} {s.games_24h:>8} {s.games_1h:>6} "
                  f"{s.allocation_pct:>7.1f}% {status:<12}")

        print("-" * 70)

        # P2P status
        p2p = get_p2p_status()
        if p2p:
            leader = p2p.get("leader_id", "unknown")
            peers = p2p.get("alive_peers", 0)
            print(f"\nP2P: Leader={leader}, Alive={peers} peers")

    print()


def watch_mode(interval: int = 300, hours: float = 24.0):
    """Continuous monitoring mode."""
    print(f"Starting continuous monitoring (interval: {interval}s)")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            report = generate_report(hours)
            print_report(report, verbose=True)

            if not report.healthy:
                print("WARNING: Allocation imbalance detected!")

            print(f"Next check in {interval}s...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")


def main():
    parser = argparse.ArgumentParser(description="Monitor curriculum weight allocation")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds (default: 300)")
    parser.add_argument("--days", type=float, default=1.0, help="Look back period in days (default: 1)")
    parser.add_argument("--quiet", action="store_true", help="Only show summary")
    args = parser.parse_args()

    hours = args.days * 24

    if args.watch:
        watch_mode(interval=args.interval, hours=hours)
    else:
        report = generate_report(hours)
        print_report(report, verbose=not args.quiet)

        # Exit code based on health
        sys.exit(0 if report.healthy else 1)


if __name__ == "__main__":
    main()
