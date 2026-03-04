#!/usr/bin/env python3
"""
Pipeline health diagnostic for the RingRift training pipeline.

Checks generation tracking, Elo progress, NPZ freshness, and model freshness
across all 12 canonical configurations, identifying bottlenecks.

Usage:
    python scripts/pipeline_health.py
    python scripts/pipeline_health.py --db-dir data/ --verbose
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

BOARD_TYPES = ["hex8", "square8", "square19", "hexagonal"]
PLAYER_COUNTS = [2, 3, 4]

ALL_CONFIGS = [
    f"{board}_{n}p" for board in BOARD_TYPES for n in PLAYER_COUNTS
]

# Thresholds (hours)
STALE_NPZ_HOURS = 48
STALE_MODEL_HOURS = 168
NO_EVAL_HOURS = 48
NO_DATA_RATIO = 0.90


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline health diagnostic for the RingRift training pipeline."
    )
    parser.add_argument(
        "--db-dir",
        default="data/",
        help="Directory containing generation_tracking.db and elo_progress.db (default: data/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show additional detail per config",
    )
    return parser.parse_args()


def hours_since(timestamp_str_or_epoch):
    """Return hours elapsed since a timestamp (ISO string or epoch float)."""
    now = time.time()
    if isinstance(timestamp_str_or_epoch, (int, float)):
        return (now - timestamp_str_or_epoch) / 3600.0
    # Try ISO format parsing (manual, no dateutil dependency)
    ts = str(timestamp_str_or_epoch)
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            import datetime
            dt = datetime.datetime.strptime(ts.split("+")[0].split("Z")[0], fmt)
            epoch = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
            return (now - epoch) / 3600.0
        except ValueError:
            continue
    return float("inf")


def file_age_hours(path):
    """Return hours since a file was last modified, or None if missing."""
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    return (time.time() - mtime) / 3600.0


def query_generation_stats(db_path, config_key):
    """Query generation_tracking.db for a config's statistics."""
    board_type, num_players_str = config_key.rsplit("_", 1)
    num_players = int(num_players_str.replace("p", ""))

    result = {
        "total_generations": 0,
        "generations_with_data": 0,
        "last_generation_age_hours": None,
    }

    if not Path(db_path).exists():
        return result

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Total generations
        cur.execute(
            "SELECT COUNT(*) FROM model_generations WHERE board_type = ? AND num_players = ?",
            (board_type, num_players),
        )
        result["total_generations"] = cur.fetchone()[0]

        # Generations with data (training_samples > 0)
        cur.execute(
            "SELECT COUNT(*) FROM model_generations WHERE board_type = ? AND num_players = ? AND training_samples > 0",
            (board_type, num_players),
        )
        result["generations_with_data"] = cur.fetchone()[0]

        # Last generation created_at
        cur.execute(
            "SELECT created_at FROM model_generations WHERE board_type = ? AND num_players = ? ORDER BY created_at DESC LIMIT 1",
            (board_type, num_players),
        )
        row = cur.fetchone()
        if row and row[0]:
            result["last_generation_age_hours"] = hours_since(row[0])

        conn.close()
    except sqlite3.Error as e:
        print(f"  WARNING: generation_tracking.db error for {config_key}: {e}", file=sys.stderr)

    return result


def query_elo_stats(db_path, config_key):
    """Query elo_progress.db for a config's latest Elo information."""
    result = {
        "current_elo": None,
        "games_played": None,
        "vs_random_wr": None,
        "vs_heuristic_wr": None,
        "last_elo_age_hours": None,
        "best_model_id": None,
    }

    if not Path(db_path).exists():
        return result

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM elo_progress WHERE config_key = ? ORDER BY timestamp DESC LIMIT 1",
            (config_key,),
        )
        row = cur.fetchone()
        if row:
            result["current_elo"] = row["best_elo"]
            result["games_played"] = row["games_played"]
            result["vs_random_wr"] = row["vs_random_win_rate"]
            result["vs_heuristic_wr"] = row["vs_heuristic_win_rate"]
            result["best_model_id"] = row["best_model_id"]
            if row["timestamp"]:
                result["last_elo_age_hours"] = hours_since(row["timestamp"])

        conn.close()
    except sqlite3.Error as e:
        print(f"  WARNING: elo_progress.db error for {config_key}: {e}", file=sys.stderr)

    return result


def identify_bottleneck(gen_stats, elo_stats, npz_age, model_age):
    """Identify the primary bottleneck for a config."""
    total = gen_stats["total_generations"]
    with_data = gen_stats["generations_with_data"]

    # NO_DATA: >90% of generations have 0 samples
    if total > 0 and with_data / total < (1.0 - NO_DATA_RATIO):
        return "NO_DATA"

    # STALE_NPZ: NPZ older than 48h
    if npz_age is not None and npz_age > STALE_NPZ_HOURS:
        return "STALE_NPZ"

    # STALE_MODEL: model older than 168h (7 days)
    if model_age is not None and model_age > STALE_MODEL_HOURS:
        return "STALE_MODEL"

    # NO_EVALUATION: no Elo entry in 48h
    elo_age = elo_stats.get("last_elo_age_hours")
    if elo_age is None or elo_age > NO_EVAL_HOURS:
        return "NO_EVALUATION"

    return "HEALTHY"


def fmt_hours(h):
    """Format hours as a readable string."""
    if h is None:
        return "-"
    if h < 1:
        return f"{h * 60:.0f}m"
    if h < 24:
        return f"{h:.1f}h"
    days = h / 24
    return f"{days:.1f}d"


def fmt_elo(elo):
    if elo is None:
        return "-"
    return f"{elo:.0f}"


def fmt_ratio(num, denom):
    if denom == 0:
        return "0/0"
    return f"{num}/{denom}"


def fmt_pct(val):
    if val is None:
        return "-"
    return f"{val * 100:.0f}%"


def main():
    args = parse_args()

    # Resolve paths relative to CWD (expected: ai-service/)
    db_dir = Path(args.db_dir)
    gen_db = db_dir / "generation_tracking.db"
    elo_db = db_dir / "elo_progress.db"
    training_dir = db_dir / "training"
    models_dir = Path("models")

    # Check existence
    missing = []
    if not gen_db.exists():
        missing.append(str(gen_db))
    if not elo_db.exists():
        missing.append(str(elo_db))
    if missing:
        print(f"WARNING: Missing databases: {', '.join(missing)}", file=sys.stderr)
        print("Run from ai-service/ directory or use --db-dir to point to the data directory.\n", file=sys.stderr)

    # Collect data for all configs
    rows = []
    for config_key in ALL_CONFIGS:
        board_type, np_str = config_key.rsplit("_", 1)

        gen_stats = query_generation_stats(str(gen_db), config_key)
        elo_stats = query_elo_stats(str(elo_db), config_key)

        npz_path = training_dir / f"{config_key}.npz"
        npz_age = file_age_hours(npz_path)

        model_path = models_dir / f"canonical_{config_key}.pth"
        model_age = file_age_hours(model_path)

        bottleneck = identify_bottleneck(gen_stats, elo_stats, npz_age, model_age)

        rows.append({
            "config": config_key,
            "elo": elo_stats["current_elo"],
            "games": elo_stats["games_played"],
            "gens_total": gen_stats["total_generations"],
            "gens_data": gen_stats["generations_with_data"],
            "last_gen_age": gen_stats["last_generation_age_hours"],
            "npz_age": npz_age,
            "model_age": model_age,
            "bottleneck": bottleneck,
            "vs_random": elo_stats["vs_random_wr"],
            "vs_heuristic": elo_stats["vs_heuristic_wr"],
            "best_model": elo_stats["best_model_id"],
        })

    # Print table
    header = f"{'Config':<16} {'Elo':>6} {'Gens':>10} {'LastGen':>8} {'NPZ':>8} {'Model':>8} {'Bottleneck':<15}"
    sep = "-" * len(header)

    print()
    print("RingRift Pipeline Health Report")
    print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print()
    print(header)
    print(sep)

    status_counts = {"HEALTHY": 0, "NO_DATA": 0, "STALE_NPZ": 0, "STALE_MODEL": 0, "NO_EVALUATION": 0}

    for r in rows:
        gens_str = fmt_ratio(r["gens_data"], r["gens_total"])
        line = (
            f"{r['config']:<16} "
            f"{fmt_elo(r['elo']):>6} "
            f"{gens_str:>10} "
            f"{fmt_hours(r['last_gen_age']):>8} "
            f"{fmt_hours(r['npz_age']):>8} "
            f"{fmt_hours(r['model_age']):>8} "
            f"{r['bottleneck']:<15}"
        )
        print(line)
        status_counts[r["bottleneck"]] = status_counts.get(r["bottleneck"], 0) + 1

    print(sep)
    print()

    # Summary
    healthy = status_counts.get("HEALTHY", 0)
    total = len(rows)
    print(f"Summary: {healthy}/{total} configs healthy")
    for status, count in sorted(status_counts.items()):
        if count > 0 and status != "HEALTHY":
            print(f"  {status}: {count} config(s)")
    print()

    # Column legend
    print("Columns:")
    print("  Elo       - Current best Elo rating (from elo_progress.db)")
    print("  Gens      - Generations with data / total generations")
    print("  LastGen   - Time since last generation was created")
    print("  NPZ       - Time since training NPZ was last updated")
    print("  Model     - Time since canonical model was last updated")
    print()
    print("Bottleneck thresholds:")
    print(f"  NO_DATA       - >{NO_DATA_RATIO * 100:.0f}% of generations have 0 training samples")
    print(f"  STALE_NPZ     - Training NPZ older than {STALE_NPZ_HOURS}h")
    print(f"  STALE_MODEL   - Canonical model older than {STALE_MODEL_HOURS}h ({STALE_MODEL_HOURS // 24}d)")
    print(f"  NO_EVALUATION - No Elo progress entry in {NO_EVAL_HOURS}h")
    print(f"  HEALTHY       - None of the above")
    print()

    # Verbose mode
    if args.verbose:
        print("=" * 60)
        print("Detailed Config Information")
        print("=" * 60)
        for r in rows:
            print(f"\n--- {r['config']} ---")
            print(f"  Elo:             {fmt_elo(r['elo'])}")
            print(f"  Games played:    {r['games'] if r['games'] is not None else '-'}")
            print(f"  vs Random WR:    {fmt_pct(r['vs_random'])}")
            print(f"  vs Heuristic WR: {fmt_pct(r['vs_heuristic'])}")
            print(f"  Best model:      {r['best_model'] or '-'}")
            print(f"  Generations:     {r['gens_data']}/{r['gens_total']} with data")
            print(f"  Last gen age:    {fmt_hours(r['last_gen_age'])}")
            print(f"  NPZ age:         {fmt_hours(r['npz_age'])}")
            print(f"  Model age:       {fmt_hours(r['model_age'])}")
            print(f"  Bottleneck:      {r['bottleneck']}")

    # Exit code: non-zero if any config is unhealthy
    if healthy < total:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
