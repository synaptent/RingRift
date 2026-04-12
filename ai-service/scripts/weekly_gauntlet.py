#!/usr/bin/env python3
"""
Weekly Gauntlet Evaluation for RingRift AI Models

Runs gauntlet evaluation for all canonical models and generates
a progress report toward Elo targets.

Usage:
    python scripts/weekly_gauntlet.py              # Run full gauntlet
    python scripts/weekly_gauntlet.py --report     # Generate report only
    python scripts/weekly_gauntlet.py --config hex8_2p  # Single config
"""

import argparse
import asyncio
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from app.config.thresholds import PRODUCTION_ELO_THRESHOLD

# Targets from improvement plan
ELO_TARGETS = {
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

CANONICAL_MODELS = {
    "hex8_2p": "models/canonical_hex8_2p.pth",
    "hex8_3p": "models/canonical_hex8_3p.pth",
    "hex8_4p": "models/canonical_hex8_4p.pth",
    "square8_2p": "models/canonical_square8_2p.pth",
    "square8_3p": "models/canonical_square8_3p.pth",
    "square8_4p": "models/canonical_square8_4p.pth",
    "square19_2p": "models/canonical_square19_2p.pth",
    "square19_3p": "models/canonical_square19_3p.pth",
    "square19_4p": "models/canonical_square19_4p.pth",
    "hexagonal_2p": "models/canonical_hexagonal_2p.pth",
    "hexagonal_3p": "models/canonical_hexagonal_3p.pth",
    "hexagonal_4p": "models/canonical_hexagonal_4p.pth",
}


def get_elo_data(db_path: Path) -> dict:
    """Get current Elo ratings from database."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            board_type || '_' || num_players || 'p' as config,
            MAX(rating) as best_rating,
            SUM(games_played) as total_games
        FROM elo_ratings
        WHERE participant_id LIKE 'canonical%' OR participant_id LIKE 'ringrift_best%'
        GROUP BY board_type, num_players
    """)

    results = {}
    for row in cursor.fetchall():
        config, rating, games = row
        results[config] = {"rating": rating or 1500, "games": games or 0}

    conn.close()
    return results


def get_elo_history(db_path: Path, days: int = 7) -> dict:
    """Get Elo ratings from N days ago for comparison."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days)).timestamp()

    cursor.execute("""
        SELECT
            board_type || '_' || num_players || 'p' as config,
            rating,
            games_played,
            last_update
        FROM elo_ratings
        WHERE (participant_id LIKE 'canonical%' OR participant_id LIKE 'ringrift_best%')
          AND last_update < ?
        ORDER BY last_update DESC
    """, (cutoff,))

    # Get earliest rating per config within window
    results = {}
    for row in cursor.fetchall():
        config, rating, games, updated = row
        if config not in results:
            results[config] = {"rating": rating or 1500, "games": games or 0}

    conn.close()
    return results


def run_gauntlet(config: str, model_path: str, games: int = 30) -> Optional[dict]:
    """Run gauntlet evaluation for a single config."""
    parts = config.replace("_", " ").split()
    board_type = parts[0]
    num_players = int(parts[1].replace("p", ""))

    cmd = [
        sys.executable, "-m", "app.gauntlet.runner",
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--model-path", model_path,
        "--games", str(games),
        "--json-output"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            # Parse JSON output
            for line in result.stdout.strip().split("\n"):
                if line.startswith("{"):
                    return json.loads(line)
        else:
            print(f"  Gauntlet failed for {config}: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  Gauntlet timed out for {config}")
    except Exception as e:
        print(f"  Gauntlet error for {config}: {e}")

    return None


def generate_report(elo_data: dict, elo_history: dict) -> str:
    """Generate weekly progress report."""
    report = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report.append("=" * 70)
    report.append(f"RingRift AI Weekly Progress Report - {now}")
    report.append("=" * 70)
    report.append("")

    # Summary stats
    at_target = sum(1 for c in ELO_TARGETS if elo_data.get(c, {}).get("rating", 1500) >= ELO_TARGETS[c])
    avg_gap = sum(elo_data.get(c, {}).get("rating", 1500) - ELO_TARGETS[c] for c in ELO_TARGETS) / len(ELO_TARGETS)

    report.append(f"Summary: {at_target}/12 configs at target | Avg gap: {avg_gap:+.0f} Elo")
    report.append("")

    # Detailed table
    report.append(f"{'Config':<15} {'Current':>8} {'Target':>8} {'Gap':>8} {'7d Change':>10} {'Games':>8}")
    report.append("-" * 70)

    configs = []
    for config, target in ELO_TARGETS.items():
        current = elo_data.get(config, {}).get("rating", 1500)
        games = elo_data.get(config, {}).get("games", 0)
        prev = elo_history.get(config, {}).get("rating", 1500)
        change = current - prev
        gap = current - target
        configs.append((config, current, target, gap, change, games))

    # Sort by gap
    configs.sort(key=lambda x: x[3])

    for config, current, target, gap, change, games in configs:
        change_str = f"{change:+.0f}" if change != 0 else "-"
        status = "" if current >= target else ""
        report.append(f"{config:<15} {current:>8.0f} {target:>8} {gap:>+8.0f} {change_str:>10} {games:>8}")

    report.append("")
    report.append("=" * 70)

    # Recommendations
    report.append("Recommendations:")
    critical = [c for c, _, _, g, _, gm in configs if g < -150 and gm < 500]
    if critical:
        report.append(f"  - CRITICAL: {', '.join(critical)} need more selfplay")

    slow = [c for c, _, _, _, ch, _ in configs if ch < 10]
    if slow:
        report.append(f"  - SLOW PROGRESS: {', '.join(slow[:3])} - consider architecture changes")

    report.append("")
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Weekly gauntlet evaluation")
    parser.add_argument("--report", action="store_true", help="Generate report only")
    parser.add_argument("--config", type=str, help="Run single config")
    parser.add_argument("--games", type=int, default=30, help="Games per gauntlet")
    parser.add_argument("--db", type=str, default="data/unified_elo.db", help="Elo database")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = Path(__file__).parent.parent / db_path

    elo_data = get_elo_data(db_path)
    elo_history = get_elo_history(db_path, days=7)

    if args.report:
        print(generate_report(elo_data, elo_history))
        return

    configs = [args.config] if args.config else list(CANONICAL_MODELS.keys())

    print(f"Running gauntlet for {len(configs)} configs...")
    print()

    for config in configs:
        model_path = CANONICAL_MODELS.get(config)
        if not model_path or not Path(model_path).exists():
            print(f"[SKIP] {config}: Model not found at {model_path}")
            continue

        print(f"[EVAL] {config}...")
        result = run_gauntlet(config, model_path, args.games)
        if result:
            print(f"  vs Random: {result.get('vs_random', 'N/A')}%")
            print(f"  vs Heuristic: {result.get('vs_heuristic', 'N/A')}%")
            print(f"  Estimated Elo: {result.get('estimated_elo', 'N/A')}")
        print()

    # Generate final report
    elo_data = get_elo_data(db_path)  # Refresh after gauntlet
    print(generate_report(elo_data, elo_history))


if __name__ == "__main__":
    main()
