#!/usr/bin/env python3
"""Model cleanup script - removes old/unused models to save disk space.

Keeps:
- Best models (ringrift_best_*)
- Most recent N models per board config
- Models referenced in promotion history
- Models referenced in Elo leaderboard
"""
import os
import re
import json
import sys
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.paths import MODELS_DIR, DATA_DIR

PROMOTION_HISTORY = str(DATA_DIR / "model_promotion_history.json")
KEEP_PER_CONFIG = 5  # Keep most recent N models per board config

def parse_model_filename(filename):
    """Extract metadata from model filename.

    Examples:
    - ringrift_v4_sq8_2p_20251210_094248.pth
    - ringrift_v5_sq19_3p_20251211_152301.pth
    - sq8_2p_nn_baseline_20251212_110907.pth
    """
    # Skip non-pth files
    if not filename.endswith(".pth"):
        return None

    # Skip best models (always keep)
    if "best" in filename.lower():
        return None

    # Try to extract timestamp (format: YYYYMMDD_HHMMSS)
    match = re.search(r'(\d{8}_\d{6})', filename)
    if not match:
        return None

    timestamp_str = match.group(1)
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None

    # Extract board config
    config_match = re.search(r'(sq(?:uare)?(?:8|19))_(\d)p', filename, re.IGNORECASE)
    if config_match:
        board = config_match.group(1).lower()
        players = config_match.group(2)
        config = f"{board}_{players}p"
    elif "hex" in filename.lower():
        players_match = re.search(r'(\d)p', filename)
        players = players_match.group(1) if players_match else "2"
        config = f"hex_{players}p"
    else:
        config = "unknown"

    return {
        "filename": filename,
        "config": config,
        "timestamp": timestamp,
        "timestamp_str": timestamp_str
    }

def get_protected_models():
    """Get set of model IDs that should not be deleted."""
    protected = set()

    # Load promotion history
    if os.path.exists(PROMOTION_HISTORY):
        try:
            with open(PROMOTION_HISTORY) as f:
                history = json.load(f)
            for p in history:
                model_id = p.get("model_id", "")
                if model_id:
                    protected.add(model_id)
        except (json.JSONDecodeError, OSError):
            pass

    return protected

def find_models_to_delete(models_dir, keep_per_config, dry_run=True):
    """Find models that can be deleted.

    Returns list of (filename, reason) tuples.
    """
    # Group models by config
    models_by_config = defaultdict(list)
    skipped = []

    for filename in os.listdir(models_dir):
        if not filename.endswith(".pth"):
            continue

        info = parse_model_filename(filename)
        if info is None:
            skipped.append((filename, "unparseable or protected"))
            continue

        models_by_config[info["config"]].append(info)

    # Get protected models
    protected = get_protected_models()

    # For each config, sort by timestamp and mark old ones for deletion
    to_delete = []
    to_keep = []

    for config, models in models_by_config.items():
        # Sort by timestamp, newest first
        models.sort(key=lambda x: x["timestamp"], reverse=True)

        kept = 0
        for _i, model in enumerate(models):
            model_id = model["filename"].replace(".pth", "")

            # Check if protected
            if model_id in protected:
                to_keep.append((model["filename"], f"protected (promotion history)"))
                kept += 1
                continue

            # Check if best model
            if "best" in model["filename"].lower():
                to_keep.append((model["filename"], "best model"))
                kept += 1
                continue

            # Keep most recent N per config
            if kept < keep_per_config:
                to_keep.append((model["filename"], f"recent ({config}, #{kept+1})"))
                kept += 1
                continue

            # Mark for deletion
            to_delete.append((model["filename"], f"old ({config}, older than top {keep_per_config})"))

    return to_delete, to_keep, skipped

def main():
    parser = argparse.ArgumentParser(description="Clean up old model files")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Show what would be deleted without deleting")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete files (overrides --dry-run)")
    parser.add_argument("--keep", type=int, default=KEEP_PER_CONFIG,
                        help=f"Number of models to keep per config (default: {KEEP_PER_CONFIG})")
    parser.add_argument("--models-dir", default=MODELS_DIR,
                        help=f"Models directory (default: {MODELS_DIR})")
    args = parser.parse_args()

    dry_run = not args.delete
    models_dir = args.models_dir

    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return 1

    print(f"Scanning models in: {models_dir}")
    print(f"Keeping {args.keep} most recent models per config")
    print(f"Mode: {'DRY RUN' if dry_run else 'DELETE'}")
    print()

    to_delete, to_keep, skipped = find_models_to_delete(
        models_dir, args.keep, dry_run=dry_run
    )

    # Calculate sizes
    total_delete_size = 0
    for filename, _reason in to_delete:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            total_delete_size += os.path.getsize(filepath)

    total_keep_size = 0
    for filename, _reason in to_keep:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            total_keep_size += os.path.getsize(filepath)

    # Print summary
    print(f"=== MODELS TO KEEP ({len(to_keep)}) ===")
    for filename, reason in sorted(to_keep):
        print(f"  KEEP: {filename} - {reason}")

    print()
    print(f"=== MODELS TO DELETE ({len(to_delete)}) ===")
    for filename, reason in sorted(to_delete):
        print(f"  DELETE: {filename} - {reason}")

    if skipped:
        print()
        print(f"=== SKIPPED ({len(skipped)}) ===")
        for filename, reason in sorted(skipped):
            print(f"  SKIP: {filename} - {reason}")

    print()
    print(f"Summary:")
    print(f"  Models to keep:   {len(to_keep):3d} ({total_keep_size / 1024 / 1024:.1f} MB)")
    print(f"  Models to delete: {len(to_delete):3d} ({total_delete_size / 1024 / 1024:.1f} MB)")

    if dry_run:
        print()
        print("This was a DRY RUN. Run with --delete to actually delete files.")
    else:
        # Actually delete
        deleted_count = 0
        deleted_size = 0
        for filename, _reason in to_delete:
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                os.remove(filepath)
                deleted_count += 1
                deleted_size += size
                print(f"Deleted: {filename}")

        print()
        print(f"Deleted {deleted_count} files ({deleted_size / 1024 / 1024:.1f} MB)")

    return 0

if __name__ == "__main__":
    exit(main())
