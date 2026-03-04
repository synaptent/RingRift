#!/usr/bin/env python
"""
Backfill missing training_samples and training_games in generation_tracking.db.

Many generations were recorded with training_samples=0 and training_games=0 due to
a bug in training_executor.py that parsed validation log lines instead of actual
totals (see MEMORY.md "Training Sample Count Mismatch" Feb 28, 2026).

This script fixes those rows by:
1. Finding all generations where training_samples=0 or training_games=0
2. Checking if a corresponding NPZ file exists at data/training/{config_key}.npz
3. Reading the NPZ to get the actual sample count (len of 'policies' or 'boards' array)
4. Estimating training_games from training_samples using average samples-per-game
5. Updating the database row

Usage (from ai-service/):
    # Dry run (preview changes)
    python scripts/backfill_generation_tracking.py --dry-run

    # Apply changes
    python scripts/backfill_generation_tracking.py

    # Verbose output
    python scripts/backfill_generation_tracking.py --verbose

    # Custom paths
    python scripts/backfill_generation_tracking.py \
        --db-path data/generation_tracking.db \
        --training-dir data/training
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

# Average samples per game, estimated from typical game lengths.
# Used to back-calculate training_games from training_samples.
SAMPLES_PER_GAME = {
    "hex8_2p": 35,
    "hex8_3p": 40,
    "hex8_4p": 45,
    "square8_2p": 40,
    "square8_3p": 50,
    "square8_4p": 55,
    "square19_2p": 150,
    "square19_3p": 180,
    "square19_4p": 200,
    "hexagonal_2p": 250,
    "hexagonal_3p": 300,
    "hexagonal_4p": 350,
}


def get_npz_sample_count(npz_path: Path) -> int | None:
    """Read an NPZ file and return the number of training samples.

    Tries 'policies' array first (canonical key), then 'boards' as fallback.
    Returns None if the file cannot be read or contains no recognized arrays.

    Note: allow_pickle=True is required because this project stores policy_indices
    and policy_values as object-dtype arrays (variable-length per sample).
    """
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "policies" in data:
                return len(data["policies"])
            if "boards" in data:
                return len(data["boards"])
            # Try other common keys
            for key in ("features", "values", "policy_indices"):
                if key in data:
                    return len(data[key])
        return None
    except Exception as e:
        print(f"  WARNING: Failed to read {npz_path}: {e}", file=sys.stderr)
        return None


def estimate_games(config_key: str, sample_count: int) -> int:
    """Estimate the number of training games from sample count."""
    spg = SAMPLES_PER_GAME.get(config_key)
    if spg is None:
        # Fallback: use a generic estimate
        print(
            f"  WARNING: No samples-per-game estimate for {config_key}, using 50",
            file=sys.stderr,
        )
        spg = 50
    return max(1, sample_count // spg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill training_samples and training_games in generation_tracking.db"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/generation_tracking.db"),
        help="Path to generation_tracking.db (default: data/generation_tracking.db)",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path("data/training"),
        help="Directory containing NPZ training files (default: data/training)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without modifying the database",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about each generation",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"ERROR: Database not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)

    if not args.training_dir.is_dir():
        print(f"ERROR: Training directory not found: {args.training_dir}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row

    # Find generations with missing sample/game counts
    cursor = conn.execute(
        """
        SELECT generation_id, model_path, board_type, num_players,
               training_games, training_samples, created_at
        FROM model_generations
        WHERE training_samples IS NULL
           OR training_samples = 0
           OR training_games IS NULL
           OR training_games = 0
        ORDER BY generation_id
        """
    )
    rows = cursor.fetchall()

    if not rows:
        print("No generations with missing training_samples/training_games found.")
        conn.close()
        return

    print(f"Found {len(rows)} generation(s) with missing sample/game counts.\n")

    # Cache NPZ sample counts by config_key to avoid re-reading the same file
    npz_cache: dict[str, int | None] = {}

    updated_count = 0
    skipped_count = 0
    total_samples_recovered = 0

    for row in rows:
        gen_id = row["generation_id"]
        board_type = row["board_type"]
        num_players = row["num_players"]
        config_key = f"{board_type}_{num_players}p"
        model_path = row["model_path"]

        if args.verbose:
            print(
                f"  Gen {gen_id}: {config_key} "
                f"(model={model_path}, "
                f"samples={row['training_samples']}, "
                f"games={row['training_games']})"
            )

        # Look up or read the NPZ file
        if config_key not in npz_cache:
            npz_path = args.training_dir / f"{config_key}.npz"
            if npz_path.exists():
                npz_cache[config_key] = get_npz_sample_count(npz_path)
                if npz_cache[config_key] is not None and args.verbose:
                    print(f"    NPZ {npz_path}: {npz_cache[config_key]:,} samples")
            else:
                npz_cache[config_key] = None
                if args.verbose:
                    print(f"    NPZ not found: {npz_path}")

        sample_count = npz_cache[config_key]
        if sample_count is None:
            skipped_count += 1
            if args.verbose:
                print(f"    SKIP: No NPZ data available for {config_key}")
            continue

        game_count = estimate_games(config_key, sample_count)

        if args.dry_run:
            print(
                f"  [DRY RUN] Gen {gen_id} ({config_key}): "
                f"would set training_samples={sample_count:,}, "
                f"training_games={game_count:,}"
            )
        else:
            conn.execute(
                "UPDATE model_generations SET training_samples=?, training_games=? "
                "WHERE generation_id=?",
                (sample_count, game_count, gen_id),
            )
            if args.verbose:
                print(
                    f"    UPDATED: training_samples={sample_count:,}, "
                    f"training_games={game_count:,}"
                )

        updated_count += 1
        total_samples_recovered += sample_count

    if not args.dry_run:
        conn.commit()

    conn.close()

    # Summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total generations with missing data: {len(rows)}")
    print(f"  Updated:  {updated_count}")
    print(f"  Skipped (no NPZ): {skipped_count}")
    print(f"  Total samples recovered: {total_samples_recovered:,}")
    if args.dry_run:
        print("\n  (Dry run -- no changes were made to the database)")


if __name__ == "__main__":
    main()
