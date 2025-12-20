#!/usr/bin/env python3
"""Prune weak models based on gauntlet results.

Removes models that score below a threshold in the baseline gauntlet,
keeping only competitive models to reduce disk usage and speed up evaluation.

Usage:
    python scripts/prune_weak_models.py --dry-run           # Preview deletions
    python scripts/prune_weak_models.py --prune             # Actually delete
    python scripts/prune_weak_models.py --threshold 0.6     # Custom threshold
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

AI_SERVICE_ROOT = Path(__file__).parent.parent
MODELS_DIR = AI_SERVICE_ROOT / "models"
RESULTS_FILE = AI_SERVICE_ROOT / "data" / "aggregated_gauntlet_results.json"
BACKUP_DIR = AI_SERVICE_ROOT / "models" / "_pruned_backup"
PRUNE_LOG = AI_SERVICE_ROOT / "data" / "prune_log.json"

# Models to never delete (best models, baselines)
PROTECTED_PATTERNS = [
    "ringrift_best_",
    "nnue_policy_",
    "nn_baseline",
    "_baseline",
]

# Default score threshold (models below this get pruned)
DEFAULT_THRESHOLD = 0.5


def load_gauntlet_results() -> dict[str, float]:
    """Load model scores from aggregated results."""
    scores = {}

    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            data = json.load(f)
            for result in data.get("results", []):
                name = result.get("name", "")
                score = result.get("avg_score", 0)
                scores[name] = score

    return scores


def is_protected(model_name: str) -> bool:
    """Check if a model is protected from pruning."""
    for pattern in PROTECTED_PATTERNS:
        if pattern in model_name:
            return True
    return False


def find_models_to_prune(scores: dict[str, float], threshold: float) -> list[Path]:
    """Find model files that should be pruned."""
    to_prune = []

    for model_file in MODELS_DIR.glob("*.pth"):
        name = model_file.stem

        # Skip protected models
        if is_protected(name):
            continue

        # Check if we have a score for this model
        score = scores.get(name, None)

        if score is not None and score < threshold:
            to_prune.append(model_file)

    return to_prune


def backup_model(model_file: Path):
    """Move model to backup directory instead of deleting."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dest = BACKUP_DIR / model_file.name
    shutil.move(str(model_file), str(dest))


def prune_models(models: list[Path], backup: bool = True) -> int:
    """Prune the specified models."""
    pruned = 0
    log_entries = []

    for model_file in models:
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


def calculate_savings(models: list[Path]) -> float:
    """Calculate total disk space that would be freed."""
    return sum(m.stat().st_size for m in models) / (1024 * 1024 * 1024)  # GB


def main():
    parser = argparse.ArgumentParser(description="Prune weak models")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("--prune", action="store_true", help="Actually prune models")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Score threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--no-backup", action="store_true", help="Delete instead of backup")
    args = parser.parse_args()

    if not args.dry_run and not args.prune:
        print("Specify --dry-run to preview or --prune to delete")
        return 1

    # Load scores
    scores = load_gauntlet_results()
    print(f"Loaded scores for {len(scores)} models")

    if not scores:
        print("No gauntlet results found. Run aggregate_elo_results.py --collect first.")
        return 1

    # Find models to prune
    to_prune = find_models_to_prune(scores, args.threshold)
    savings_gb = calculate_savings(to_prune)

    print(f"\nModels below threshold ({args.threshold}):")
    print(f"  Count: {len(to_prune)}")
    print(f"  Space: {savings_gb:.2f} GB")

    if not to_prune:
        print("\nNo models to prune!")
        return 0

    if args.dry_run:
        print("\n[DRY RUN] Would prune:")
        for m in sorted(to_prune, key=lambda x: scores.get(x.stem, 0)):
            score = scores.get(m.stem, 0)
            print(f"  {m.name}: score={score:.2f}")
        print(f"\nRun with --prune to actually delete these models")
    else:
        print(f"\nPruning {len(to_prune)} models...")
        pruned = prune_models(to_prune, backup=not args.no_backup)
        print(f"\nPruned {pruned} models, freed {savings_gb:.2f} GB")
        if not args.no_backup:
            print(f"Backups saved to: {BACKUP_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
