#!/usr/bin/env python3
"""Pipeline Dashboard - Quick status overview of the training pipeline.

Usage:
    python scripts/pipeline_dashboard.py
    watch -n 60 python scripts/pipeline_dashboard.py  # Auto-refresh
"""

import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def get_game_counts():
    """Get game counts by config."""
    db_path = AI_SERVICE_ROOT / "data" / "games" / "selfplay.db"
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("""
        SELECT board_type, num_players, COUNT(*)
        FROM games GROUP BY board_type, num_players
    """).fetchall()
    conn.close()
    return {f"{b}_{p}p": c for b, p, c in rows}


def get_process_counts():
    """Count running processes."""
    try:
        result = subprocess.run(
            ["pgrep", "-fa", "selfplay|training|p2p|tournament"],
            capture_output=True, text=True, timeout=5
        )
        lines = [line for line in result.stdout.strip().split("\n") if line]
        counts = {
            "selfplay": sum(1 for line in lines if "selfplay" in line.lower()),
            "training": sum(1 for line in lines if "training" in line.lower()),
            "p2p": sum(1 for line in lines if "p2p" in line.lower()),
            "tournament": sum(1 for line in lines if "tournament" in line.lower()),
        }
        return counts
    except Exception:
        return {}


def get_recent_models():
    """Get recently created models."""
    models_dir = AI_SERVICE_ROOT / "models"
    if not models_dir.exists():
        return []
    models = list(models_dir.glob("*.pt*"))
    models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [(m.name, datetime.fromtimestamp(m.stat().st_mtime)) for m in models[:5]]


def get_top_elo():
    """Get top models by Elo."""
    db_path = AI_SERVICE_ROOT / "data" / "unified_elo.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("""
            SELECT p.model_path, e.board_type, e.num_players, e.rating, e.games_played
            FROM elo_ratings e
            JOIN participants p ON e.participant_id = p.participant_id
            WHERE e.archived_at IS NULL AND p.model_path IS NOT NULL
            AND p.model_path NOT LIKE '%BASELINE%'
            ORDER BY e.rating DESC LIMIT 10
        """).fetchall()
        conn.close()
        return [(Path(p).name if p else "?", f"{b}_{n}p", r, g) for p, b, n, r, g in rows]
    except Exception:
        conn.close()
        return []


def main():
    print("=" * 60)
    print(f"  RINGRIFT PIPELINE DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Game counts
    print("\n[GAMES]")
    counts = get_game_counts()
    total = sum(counts.values())
    for config, count in sorted(counts.items()):
        print(f"  {config:<15} {count:>6}")
    print(f"  {'TOTAL':<15} {total:>6}")

    # Processes
    print("\n[PROCESSES]")
    procs = get_process_counts()
    for name, count in procs.items():
        status = "OK" if count > 0 else "NONE"
        print(f"  {name:<15} {count:>3} [{status}]")

    # Recent models
    print("\n[RECENT MODELS]")
    models = get_recent_models()
    for name, mtime in models:
        age = (datetime.now() - mtime).total_seconds() / 3600
        print(f"  {name[:40]:<40} {age:.1f}h ago")

    # Top Elo
    print("\n[TOP ELO]")
    elo = get_top_elo()
    for name, config, rating, games in elo[:5]:
        print(f"  {name[:30]:<30} {config:<12} Elo={rating:.0f} ({games}g)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
