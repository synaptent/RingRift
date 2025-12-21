#!/usr/bin/env python3
"""Unified Dashboard - Single entry point for all monitoring views.

Consolidates multiple dashboard scripts:
- elo_dashboard.py -> dashboard.py elo
- training_dashboard.py -> dashboard.py training
- pipeline_dashboard.py -> dashboard.py pipeline
- composite_elo_dashboard.py -> dashboard.py composite

Usage:
    python scripts/dashboard.py                # Full overview (default)
    python scripts/dashboard.py training       # Training metrics
    python scripts/dashboard.py elo            # ELO rankings
    python scripts/dashboard.py pipeline       # Pipeline status
    python scripts/dashboard.py composite      # Composite ELO
    python scripts/dashboard.py cluster        # Cluster health
    python scripts/dashboard.py --live         # Auto-refresh mode
    python scripts/dashboard.py --json         # JSON output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Setup paths
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))


def print_header(title: str, char: str = "=", width: int = 70) -> None:
    """Print a section header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n  {title}")
    print(f"  {'-' * (len(title) + 2)}")


# =============================================================================
# Data Collection
# =============================================================================


def get_db_connection(db_name: str) -> sqlite3.Connection | None:
    """Get database connection with error handling."""
    db_paths = {
        "elo": AI_SERVICE_ROOT / "data" / "elo_database.db",
        "unified_elo": AI_SERVICE_ROOT / "data" / "unified_elo.db",
        "games": AI_SERVICE_ROOT / "data" / "games" / "selfplay.db",
        "manifest": AI_SERVICE_ROOT / "data" / "data_manifest.db",
    }
    db_path = db_paths.get(db_name)
    if db_path and db_path.exists():
        return sqlite3.connect(str(db_path))
    return None


def get_game_counts() -> dict[str, int]:
    """Get game counts by config."""
    conn = get_db_connection("games")
    if not conn:
        return {}
    try:
        rows = conn.execute("""
            SELECT board_type, num_players, COUNT(*)
            FROM games GROUP BY board_type, num_players
        """).fetchall()
        return {f"{b}_{p}p": c for b, p, c in rows}
    except Exception:
        return {}
    finally:
        conn.close()


def get_elo_top_models(limit: int = 10) -> list[dict[str, Any]]:
    """Get top ELO-rated models."""
    conn = get_db_connection("unified_elo")
    if not conn:
        return []
    try:
        rows = conn.execute("""
            SELECT participant_id, elo_rating, games_played,
                   wins, losses, draws, last_played
            FROM participants
            ORDER BY elo_rating DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [
            {
                "id": row[0],
                "elo": row[1],
                "games": row[2],
                "wins": row[3],
                "losses": row[4],
                "draws": row[5],
                "last_played": row[6],
            }
            for row in rows
        ]
    except Exception:
        return []
    finally:
        conn.close()


def get_training_status() -> dict[str, Any]:
    """Get training status from checkpoint files."""
    models_dir = AI_SERVICE_ROOT / "models"
    if not models_dir.exists():
        return {}

    status = {
        "checkpoints": [],
        "latest_mtime": None,
    }

    for ckpt in models_dir.glob("*.pt"):
        mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
        status["checkpoints"].append({
            "name": ckpt.name,
            "size_mb": ckpt.stat().st_size / (1024 * 1024),
            "modified": mtime.isoformat(),
        })
        if status["latest_mtime"] is None or mtime > status["latest_mtime"]:
            status["latest_mtime"] = mtime

    status["checkpoints"].sort(key=lambda x: x["modified"], reverse=True)
    return status


def get_cluster_status() -> dict[str, Any]:
    """Get cluster status using UnifiedClusterMonitor."""
    try:
        from app.monitoring.unified_cluster_monitor import UnifiedClusterMonitor
        monitor = UnifiedClusterMonitor()
        # Run async method synchronously
        return asyncio.run(monitor.get_full_status())
    except Exception as e:
        return {"error": str(e)}


def get_quality_stats() -> dict[str, Any]:
    """Get data quality statistics."""
    conn = get_db_connection("manifest")
    if not conn:
        return {}
    try:
        # Get quality distribution
        rows = conn.execute("""
            SELECT
                COUNT(*) as total,
                AVG(quality_score) as avg_quality,
                SUM(CASE WHEN quality_score >= 0.7 THEN 1 ELSE 0 END) as high_quality,
                SUM(CASE WHEN quality_score < 0.3 THEN 1 ELSE 0 END) as low_quality
            FROM game_metadata
            WHERE quality_score IS NOT NULL
        """).fetchone()

        return {
            "total_games": rows[0] or 0,
            "avg_quality": rows[1] or 0.0,
            "high_quality_count": rows[2] or 0,
            "low_quality_count": rows[3] or 0,
        }
    except Exception:
        return {}
    finally:
        conn.close()


# =============================================================================
# Dashboard Views
# =============================================================================


def render_overview(args: argparse.Namespace) -> dict[str, Any]:
    """Render full system overview."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "games": get_game_counts(),
        "top_models": get_elo_top_models(5),
        "training": get_training_status(),
        "quality": get_quality_stats(),
    }

    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return data

    print_header("RingRift AI - System Overview")
    print(f"  Timestamp: {data['timestamp']}")

    # Games summary
    print_subheader("Game Counts by Config")
    if data["games"]:
        for config, count in sorted(data["games"].items()):
            print(f"    {config}: {count:,}")
    else:
        print("    No games found")

    # Top models
    print_subheader("Top Models (by ELO)")
    if data["top_models"]:
        print(f"    {'ID':<35} {'ELO':>8} {'Games':>7}")
        for model in data["top_models"]:
            print(f"    {model['id'][:35]:<35} {model['elo']:>8.1f} {model['games']:>7}")
    else:
        print("    No models found")

    # Training status
    print_subheader("Training Status")
    if data["training"].get("checkpoints"):
        latest = data["training"]["checkpoints"][0]
        print(f"    Latest checkpoint: {latest['name']}")
        print(f"    Modified: {latest['modified']}")
        print(f"    Size: {latest['size_mb']:.1f} MB")
    else:
        print("    No checkpoints found")

    # Quality stats
    print_subheader("Data Quality")
    if data["quality"]:
        print(f"    Total games with quality: {data['quality'].get('total_games', 0):,}")
        print(f"    Average quality: {data['quality'].get('avg_quality', 0):.3f}")
        print(f"    High quality (>=0.7): {data['quality'].get('high_quality_count', 0):,}")
    else:
        print("    No quality data available")

    print()
    return data


def render_training(args: argparse.Namespace) -> dict[str, Any]:
    """Render training-focused dashboard."""
    # Delegate to existing training_dashboard if more detail needed
    from scripts.training_dashboard import main as training_main
    sys.argv = ["training_dashboard.py"]
    if args.json:
        sys.argv.append("--json")
    training_main()
    return {}


def render_elo(args: argparse.Namespace) -> dict[str, Any]:
    """Render ELO-focused dashboard."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "top_models": get_elo_top_models(20),
    }

    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return data

    print_header("ELO Rankings")
    print(f"  Timestamp: {data['timestamp']}")

    if data["top_models"]:
        print(f"\n  {'Rank':<5} {'ID':<40} {'ELO':>8} {'W':>5} {'L':>5} {'D':>5} {'Games':>7}")
        print("  " + "-" * 75)
        for i, model in enumerate(data["top_models"], 1):
            print(f"  {i:<5} {model['id'][:40]:<40} {model['elo']:>8.1f} "
                  f"{model['wins']:>5} {model['losses']:>5} {model['draws']:>5} {model['games']:>7}")
    else:
        print("\n  No models found")

    print()
    return data


def render_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Render pipeline status dashboard."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "games": get_game_counts(),
        "quality": get_quality_stats(),
    }

    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return data

    print_header("Pipeline Status")
    print(f"  Timestamp: {data['timestamp']}")

    print_subheader("Game Counts")
    total = 0
    for config, count in sorted(data["games"].items()):
        print(f"    {config}: {count:,}")
        total += count
    print(f"    {'Total'}: {total:,}")

    print_subheader("Quality Distribution")
    if data["quality"]:
        q = data["quality"]
        print(f"    Total with quality: {q.get('total_games', 0):,}")
        print(f"    Average quality: {q.get('avg_quality', 0):.3f}")
        hq = q.get('high_quality_count', 0)
        lq = q.get('low_quality_count', 0)
        total_q = q.get('total_games', 1)
        print(f"    High quality: {hq:,} ({100*hq/max(total_q,1):.1f}%)")
        print(f"    Low quality: {lq:,} ({100*lq/max(total_q,1):.1f}%)")

    print()
    return data


def render_composite(args: argparse.Namespace) -> dict[str, Any]:
    """Render composite ELO dashboard."""
    # Delegate to existing composite_elo_dashboard
    from scripts.composite_elo_dashboard import main as composite_main
    sys.argv = ["composite_elo_dashboard.py"]
    if hasattr(args, "board") and args.board:
        sys.argv.extend(["--board", args.board])
    composite_main()
    return {}


def render_cluster(args: argparse.Namespace) -> dict[str, Any]:
    """Render cluster health dashboard."""
    data = get_cluster_status()

    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return data

    print_header("Cluster Health")

    if "error" in data:
        print(f"  Error: {data['error']}")
        return data

    # Use the unified cluster monitor's print function
    try:
        from app.monitoring.unified_cluster_monitor import print_cluster_status
        print_cluster_status(data)
    except Exception as e:
        print(f"  Status data collected but display failed: {e}")
        print(f"  Raw data: {json.dumps(data, indent=2, default=str)}")

    return data


# =============================================================================
# Main Entry Point
# =============================================================================


VIEW_HANDLERS = {
    "overview": render_overview,
    "training": render_training,
    "elo": render_elo,
    "pipeline": render_pipeline,
    "composite": render_composite,
    "cluster": render_cluster,
}


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Dashboard - Single entry point for all monitoring views",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "view",
        nargs="?",
        default="overview",
        choices=list(VIEW_HANDLERS.keys()),
        help="Dashboard view to display (default: overview)",
    )
    parser.add_argument(
        "--live", "-l",
        action="store_true",
        help="Enable live auto-refresh mode",
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--board", "-b",
        type=str,
        help="Board type filter (for composite view)",
    )

    args = parser.parse_args()
    handler = VIEW_HANDLERS.get(args.view, render_overview)

    try:
        if args.live:
            while True:
                if not args.json:
                    print("\033[2J\033[H", end="")  # Clear screen
                handler(args)
                if not args.json:
                    print(f"\n[Refreshing in {args.interval}s... Ctrl+C to exit]")
                time.sleep(args.interval)
        else:
            handler(args)
        return 0
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
