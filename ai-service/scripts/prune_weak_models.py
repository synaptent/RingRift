#!/usr/bin/env python3
"""Prune weak models based on Elo ratings from unified database.

Removes models that have Elo below a threshold and sufficient games played,
keeping only competitive models to reduce disk usage and speed up evaluation.

Usage:
    python scripts/prune_weak_models.py --dry-run           # Preview deletions
    python scripts/prune_weak_models.py --prune             # Actually delete
    python scripts/prune_weak_models.py --elo-threshold 1200  # Custom Elo threshold
    python scripts/prune_weak_models.py --min-games 50      # Require more games for pruning
"""

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).parent.parent
MODELS_DIR = AI_SERVICE_ROOT / "models"
ELO_DB = AI_SERVICE_ROOT / "data" / "unified_elo.db"
BACKUP_DIR = AI_SERVICE_ROOT / "models" / "_pruned_backup"
PRUNE_LOG = AI_SERVICE_ROOT / "data" / "prune_log.json"

# Models to never delete (best models, baselines)
PROTECTED_PATTERNS = [
    "ringrift_best_",
    "nnue_policy_",
    "nn_baseline",
    "_baseline",
    "baseline_",
]

# Default thresholds
DEFAULT_ELO_THRESHOLD = 1200  # Elo below this = weak
DEFAULT_MIN_GAMES = 30        # Need enough games for reliable rating


def load_elo_ratings() -> dict[str, dict]:
    """Load model Elo ratings from unified database."""
    ratings = {}

    if not ELO_DB.exists():
        return ratings

    conn = sqlite3.connect(str(ELO_DB))
    try:
        cur = conn.cursor()
        # Get the best rating for each model (across board types)
        cur.execute("""
            SELECT participant_id, MAX(rating), SUM(games_played)
            FROM elo_ratings
            GROUP BY participant_id
        """)
        for row in cur.fetchall():
            ratings[row[0]] = {
                "elo": row[1],
                "games": row[2],
            }
    finally:
        conn.close()

    return ratings


def is_protected(model_name: str) -> bool:
    """Check if a model is protected from pruning."""
    return any(pattern in model_name for pattern in PROTECTED_PATTERNS)


def find_models_to_prune(
    ratings: dict[str, dict],
    elo_threshold: float,
    min_games: int
) -> list[tuple[Path, dict]]:
    """Find model files that should be pruned based on Elo ratings."""
    to_prune = []

    for model_file in MODELS_DIR.glob("*.pth"):
        name = model_file.stem

        # Skip protected models
        if is_protected(name):
            continue

        # Check if we have ratings for this model
        info = ratings.get(name)

        # Only prune if: has rating, enough games, and below threshold
        if info and info["games"] >= min_games and info["elo"] < elo_threshold:
            to_prune.append((model_file, info))

    return to_prune


def backup_model(model_file: Path):
    """Move model to backup directory instead of deleting."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dest = BACKUP_DIR / model_file.name
    shutil.move(str(model_file), str(dest))


def prune_models(models: list[tuple[Path, dict]], backup: bool = True) -> int:
    """Prune the specified models."""
    pruned = 0
    log_entries = []

    for model_file, info in models:
        try:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            if backup:
                backup_model(model_file)
                action = "backed up"
            else:
                model_file.unlink()
                action = "deleted"

            pruned += 1
            log_entries.append({
                "name": model_file.name,
                "action": action,
                "size_mb": round(size_mb, 2),
                "elo": info["elo"],
                "games": info["games"],
                "timestamp": datetime.now().isoformat(),
            })
            print(f"  {action}: {model_file.name} ({size_mb:.1f}MB)")
        except Exception as e:
            print(f"  Error pruning {model_file.name}: {e}", file=sys.stderr)

    # Save log
    if log_entries:
        existing_log = []
        if PRUNE_LOG.exists():
            with open(PRUNE_LOG) as f:
                existing_log = json.load(f)

        existing_log.extend(log_entries)
        with open(PRUNE_LOG, "w") as f:
            json.dump(existing_log, f, indent=2)

    return pruned


def calculate_savings(models: list[tuple[Path, dict]]) -> float:
    """Calculate total disk space that would be freed."""
    return sum(m.stat().st_size for m, _ in models) / (1024 * 1024 * 1024)  # GB


def main():
    parser = argparse.ArgumentParser(description="Prune weak models based on Elo ratings")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("--prune", action="store_true", help="Actually prune models")
    parser.add_argument("--elo-threshold", type=float, default=DEFAULT_ELO_THRESHOLD,
                        help=f"Elo threshold - prune below this (default: {DEFAULT_ELO_THRESHOLD})")
    parser.add_argument("--min-games", type=int, default=DEFAULT_MIN_GAMES,
                        help=f"Min games required for reliable rating (default: {DEFAULT_MIN_GAMES})")
    parser.add_argument("--no-backup", action="store_true", help="Delete instead of backup")
    args = parser.parse_args()

    if not args.dry_run and not args.prune:
        print("Specify --dry-run to preview or --prune to delete")
        return 1

    # Load Elo ratings
    ratings = load_elo_ratings()
    print(f"Loaded Elo ratings for {len(ratings)} models")

    if not ratings:
        print("No Elo ratings found. Run tournaments first to generate ratings.")
        return 1

    # Find models to prune
    to_prune = find_models_to_prune(ratings, args.elo_threshold, args.min_games)
    savings_gb = calculate_savings(to_prune)

    print(f"\nModels below Elo {args.elo_threshold} (with >= {args.min_games} games):")
    print(f"  Count: {len(to_prune)}")
    print(f"  Space: {savings_gb:.2f} GB")

    if not to_prune:
        print("\nNo models to prune!")
        return 0

    if args.dry_run:
        print("\n[DRY RUN] Would prune:")
        for m, info in sorted(to_prune, key=lambda x: x[1]["elo"]):
            print(f"  {m.name}: Elo={info['elo']:.0f} ({info['games']} games)")
        print("\nRun with --prune to actually delete these models")
    else:
        print(f"\nPruning {len(to_prune)} models...")
        pruned = prune_models(to_prune, backup=not args.no_backup)
        print(f"\nPruned {pruned} models, freed {savings_gb:.2f} GB")
        if not args.no_backup:
            print(f"Backups saved to: {BACKUP_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
