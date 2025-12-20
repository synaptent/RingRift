#!/usr/bin/env python3
"""Training Metrics Dashboard - Visual overview of AI training progress.

Generates a comprehensive dashboard showing:
1. Per-config status (models, games, ELO)
2. Trend charts (ELO over time)
3. System health metrics
4. Active processes status

Usage:
    python scripts/training_dashboard.py           # Terminal dashboard
    python scripts/training_dashboard.py --html    # Generate HTML dashboard
    python scripts/training_dashboard.py --json    # JSON output
    python scripts/training_dashboard.py --watch   # Auto-refresh
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Paths
MODELS_DIR = AI_SERVICE_ROOT / "models"
DATA_DIR = AI_SERVICE_ROOT / "data"
UNIFIED_ELO_DB = DATA_DIR / "unified_elo.db"
HOLDOUT_DB = DATA_DIR / "holdouts" / "holdout_validation.db"
PROMOTION_FILE = AI_SERVICE_ROOT / "runs" / "promotion" / "promoted_models.json"

# All 9 configs
CONFIGS = [
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


def get_config_stats() -> list[dict[str, Any]]:
    """Get comprehensive stats for all configs."""
    stats = []

    for board_type, num_players in CONFIGS:
        key = f"{board_type}_{num_players}p"

        # Count models
        model_count = 0
        board_short = board_type[:3] if board_type != "hexagonal" else "hex"
        patterns = [f"{board_short}_{num_players}p", f"{board_type}_{num_players}p"]
        for model_file in MODELS_DIR.glob("*.pth"):
            name = model_file.stem.lower()
            if any(p in name for p in patterns):
                model_count += 1

        # Get ELO
        elo_best, elo_avg, elo_games = 0.0, 0.0, 0
        if UNIFIED_ELO_DB.exists():
            try:
                conn = sqlite3.connect(UNIFIED_ELO_DB)
                row = conn.execute("""
                    SELECT MAX(rating), AVG(rating), SUM(games_played)
                    FROM elo_ratings WHERE board_type = ? AND num_players = ?
                """, (board_type, num_players)).fetchone()
                if row:
                    elo_best = row[0] or 0.0
                    elo_avg = row[1] or 0.0
                    elo_games = int(row[2] or 0)
                conn.close()
            except Exception:
                pass

        # Get holdout count
        holdout_count = 0
        if HOLDOUT_DB.exists():
            try:
                conn = sqlite3.connect(HOLDOUT_DB)
                holdout_count = conn.execute(
                    "SELECT COUNT(*) FROM holdout_games WHERE board_type = ? AND num_players = ?",
                    (board_type, num_players)
                ).fetchone()[0]
                conn.close()
            except Exception:
                pass

        # Get promoted model
        promoted = None
        if PROMOTION_FILE.exists():
            try:
                with open(PROMOTION_FILE) as f:
                    data = json.load(f)
                if key in data.get("models", {}):
                    promoted = data["models"][key].get("model_id", "")[:25]
            except Exception:
                pass

        stats.append({
            "config": key,
            "board_type": board_type,
            "num_players": num_players,
            "models": model_count,
            "elo_best": round(elo_best, 1),
            "elo_avg": round(elo_avg, 1),
            "elo_games": elo_games,
            "holdout": holdout_count,
            "promoted": promoted or "-",
        })

    return stats


def get_elo_history(board_type: str, num_players: int, days: int = 7) -> list[tuple[str, float]]:
    """Get ELO history for a config."""
    if not UNIFIED_ELO_DB.exists():
        return []

    try:
        conn = sqlite3.connect(UNIFIED_ELO_DB)
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        rows = conn.execute("""
            SELECT created_at, rating FROM elo_ratings
            WHERE board_type = ? AND num_players = ? AND created_at > ?
            ORDER BY created_at
        """, (board_type, num_players, cutoff)).fetchall()
        conn.close()
        return [(r[0], r[1]) for r in rows]
    except Exception:
        return []


def sparkline(values: list[float], width: int = 20) -> str:
    """Create ASCII sparkline from values."""
    if not values:
        return "─" * width

    chars = "▁▂▃▄▅▆▇█"
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return "▄" * min(len(values), width)

    # Sample if too many values
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]

    result = ""
    for v in values:
        idx = int((v - min_val) / (max_val - min_val) * (len(chars) - 1))
        result += chars[idx]

    return result


def print_dashboard(stats: list[dict[str, Any]]):
    """Print terminal dashboard."""
    print("\n" + "═" * 80)
    print("  🎮 RingRift AI Training Dashboard")
    print("  " + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
    print("═" * 80)

    # Summary
    total_models = sum(s["models"] for s in stats)
    total_holdout = sum(s["holdout"] for s in stats)
    avg_elo = sum(s["elo_best"] for s in stats if s["elo_best"] > 0) / max(1, sum(1 for s in stats if s["elo_best"] > 0))

    print(f"\n  📊 Summary: {total_models} models | {total_holdout:,} holdout games | Avg Best ELO: {avg_elo:.0f}")

    # Per-config table
    print("\n  ┌─────────────┬────────┬──────────┬──────────┬─────────┬──────────────────────────┐")
    print("  │ Config      │ Models │ ELO Best │ ELO Avg  │ Holdout │ Promoted                 │")
    print("  ├─────────────┼────────┼──────────┼──────────┼─────────┼──────────────────────────┤")

    for s in stats:
        config = s["config"][:11].ljust(11)
        models = str(s["models"]).rjust(6)
        elo_best = f"{s['elo_best']:.0f}".rjust(8) if s["elo_best"] > 0 else "    -   "
        elo_avg = f"{s['elo_avg']:.0f}".rjust(8) if s["elo_avg"] > 0 else "    -   "
        holdout = str(s["holdout"]).rjust(7)
        promoted = s["promoted"][:24].ljust(24)

        print(f"  │ {config} │ {models} │ {elo_best} │ {elo_avg} │ {holdout} │ {promoted} │")

    print("  └─────────────┴────────┴──────────┴──────────┴─────────┴──────────────────────────┘")

    # ELO Trends
    print("\n  📈 ELO Trends (7 days):")
    for s in stats:
        if s["elo_best"] > 0:
            history = get_elo_history(s["board_type"], s["num_players"])
            if history:
                values = [h[1] for h in history]
                trend = sparkline(values, 20)
                config = s["config"][:11].ljust(11)
                print(f"    {config}: {trend} ({values[0]:.0f} → {values[-1]:.0f})")

    print("\n" + "═" * 80)


def generate_html(stats: list[dict[str, Any]]) -> str:
    """Generate HTML dashboard."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>RingRift AI Training Dashboard</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="60">
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #00d4ff; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #333; }
        th { background: #16213e; color: #00d4ff; }
        tr:hover { background: #1f2f4f; }
        .metric { font-size: 24px; font-weight: bold; color: #00d4ff; }
        .card { background: #16213e; padding: 20px; border-radius: 8px; margin: 10px 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .highlight { color: #00ff88; }
        .warning { color: #ffaa00; }
    </style>
</head>
<body>
    <h1>🎮 RingRift AI Training Dashboard</h1>
    <p>Last updated: """ + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC") + """</p>
    
    <div class="grid">
        <div class="card">
            <div class="metric">""" + str(sum(s["models"] for s in stats)) + """</div>
            <div>Total Models</div>
        </div>
        <div class="card">
            <div class="metric">""" + f"{sum(s['holdout'] for s in stats):,}" + """</div>
            <div>Holdout Games</div>
        </div>
        <div class="card">
            <div class="metric">""" + f"{max(s['elo_best'] for s in stats):.0f}" + """</div>
            <div>Best ELO</div>
        </div>
    </div>
    
    <h2>Per-Config Status</h2>
    <table>
        <tr><th>Config</th><th>Models</th><th>ELO Best</th><th>ELO Avg</th><th>ELO Games</th><th>Holdout</th><th>Promoted</th></tr>
"""
    for s in stats:
        elo_class = "highlight" if s["elo_best"] > 1600 else ("warning" if s["elo_best"] < 1500 and s["elo_best"] > 0 else "")
        elo_best_str = f"{s['elo_best']:.0f}" if s['elo_best'] > 0 else "-"
        elo_avg_str = f"{s['elo_avg']:.0f}" if s['elo_avg'] > 0 else "-"
        html += f"""        <tr>
            <td>{s['config']}</td>
            <td>{s['models']}</td>
            <td class="{elo_class}">{elo_best_str}</td>
            <td>{elo_avg_str}</td>
            <td>{s['elo_games']}</td>
            <td>{s['holdout']:,}</td>
            <td>{s['promoted']}</td>
        </tr>
"""

    html += """    </table>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Training metrics dashboard")
    parser.add_argument("--html", action="store_true", help="Generate HTML dashboard")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--watch", action="store_true", help="Auto-refresh mode")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval")

    args = parser.parse_args()

    if args.watch:
        import time
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            stats = get_config_stats()
            print_dashboard(stats)
            time.sleep(args.interval)
    else:
        stats = get_config_stats()

        if args.json:
            output = json.dumps(stats, indent=2)
        elif args.html:
            output = generate_html(stats)
        else:
            print_dashboard(stats)
            return

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Dashboard saved to {args.output}")
        else:
            print(output)


if __name__ == "__main__":
    main()
