#!/usr/bin/env python3
"""Parity validation script for game databases.

Validates that recorded games can be replayed identically by the canonical
TypeScript engine, ensuring no divergence between Python and TS game logic.

Usage:
    python scripts/run_parity_validation.py --databases data/games/*.db --mode canonical
    python scripts/run_parity_validation.py --databases data/games/square8_2p.db --output-json results.json
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import (
    validate_game_parity,
    ParityValidationError,
)
from app.db.game_replay import GameReplayDB


def get_game_count(db_path: Path) -> int:
    """Get the number of games in a database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM games")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def validate_database(
    db_path: Path,
    mode: str = "canonical",
    progress_every: int = 100,
    max_games: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate all games in a database.

    Args:
        db_path: Path to the game database
        mode: Validation mode ('canonical' or 'strict')
        progress_every: Print progress every N games
        max_games: Maximum number of games to validate (None = all)

    Returns:
        Dict with validation results
    """
    results = {
        "database": str(db_path),
        "total_games_checked": 0,
        "games_passed": 0,
        "games_with_semantic_divergence": 0,
        "games_with_replay_errors": 0,
        "divergence_details": [],
    }

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Get game IDs
        query = "SELECT game_id FROM games ORDER BY game_id"
        if max_games:
            query += f" LIMIT {max_games}"
        cursor = conn.execute(query)
        game_ids = [row["game_id"] for row in cursor.fetchall()]
        conn.close()

        if not game_ids:
            print(f"  No games found in {db_path.name}")
            return results

    except Exception as e:
        print(f"  Error reading database {db_path}: {e}")
        return results

    # Validate each game
    replay_db = GameReplayDB(str(db_path))
    start_time = time.time()

    for i, game_id in enumerate(game_ids):
        try:
            # Verify game exists
            meta = replay_db.get_game_metadata(game_id)
            if not meta:
                results["games_with_replay_errors"] += 1
                continue

            # Run parity validation using the correct API
            try:
                divergence = validate_game_parity(
                    db_path=str(db_path),
                    game_id=str(game_id),
                    mode="warn",  # Don't raise, just return divergence
                )

                if divergence:
                    results["games_with_semantic_divergence"] += 1
                    results["divergence_details"].append({
                        "game_id": str(game_id),
                        "diverged_at": divergence.diverged_at,
                        "mismatch_kinds": divergence.mismatch_kinds,
                    })
                else:
                    results["games_passed"] += 1

            except ParityValidationError as pve:
                results["games_with_semantic_divergence"] += 1
                results["divergence_details"].append({
                    "game_id": str(game_id),
                    "error": str(pve),
                })

        except Exception as e:
            results["games_with_replay_errors"] += 1

        results["total_games_checked"] += 1

        # Progress update
        if progress_every > 0 and (i + 1) % progress_every == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Progress: {i + 1}/{len(game_ids)} games "
                  f"({rate:.1f}/s, passed={results['games_passed']}, "
                  f"failed={results['games_with_semantic_divergence']})")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate game databases for parity with canonical TS engine"
    )
    parser.add_argument(
        "--databases", "-d",
        nargs="+",
        required=True,
        help="Path(s) to game database files"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["canonical", "strict"],
        default="canonical",
        help="Validation mode (default: canonical)"
    )
    parser.add_argument(
        "--output-json", "-o",
        help="Path to write JSON results"
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N games (0 to disable)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum games to validate per database"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Aggregate results across all databases
    aggregate = {
        "total_games_checked": 0,
        "games_passed": 0,
        "games_with_semantic_divergence": 0,
        "games_with_replay_errors": 0,
        "databases_processed": 0,
        "per_database": [],
    }

    # Process each database
    for db_arg in args.databases:
        db_path = Path(db_arg)

        # Handle glob patterns
        if "*" in str(db_path):
            db_files = list(db_path.parent.glob(db_path.name))
        else:
            db_files = [db_path] if db_path.exists() else []

        for db_file in db_files:
            if not db_file.suffix == ".db":
                continue

            game_count = get_game_count(db_file)
            if game_count == 0:
                continue

            if not args.quiet:
                print(f"Validating {db_file.name} ({game_count} games)...")

            results = validate_database(
                db_file,
                mode=args.mode,
                progress_every=0 if args.quiet else args.progress_every,
                max_games=args.max_games,
            )

            # Aggregate
            aggregate["total_games_checked"] += results["total_games_checked"]
            aggregate["games_passed"] += results["games_passed"]
            aggregate["games_with_semantic_divergence"] += results["games_with_semantic_divergence"]
            aggregate["games_with_replay_errors"] += results["games_with_replay_errors"]
            aggregate["databases_processed"] += 1
            aggregate["per_database"].append({
                "database": str(db_file),
                "games_checked": results["total_games_checked"],
                "passed": results["games_passed"],
                "divergences": results["games_with_semantic_divergence"],
                "errors": results["games_with_replay_errors"],
            })

            if not args.quiet:
                print(f"  Done: {results['games_passed']}/{results['total_games_checked']} passed")

    # Calculate failure rate
    if aggregate["total_games_checked"] > 0:
        aggregate["failure_rate"] = (
            (aggregate["games_with_semantic_divergence"] + aggregate["games_with_replay_errors"]) /
            aggregate["total_games_checked"]
        )
    else:
        aggregate["failure_rate"] = 0.0

    # Determine pass/fail (< 5% failure rate = pass)
    aggregate["passed"] = aggregate["failure_rate"] < 0.05

    # Write output JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(aggregate, f, indent=2)
        if not args.quiet:
            print(f"\nResults written to {args.output_json}")

    # Summary
    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"PARITY VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Databases processed: {aggregate['databases_processed']}")
        print(f"Total games checked: {aggregate['total_games_checked']}")
        print(f"Games passed:        {aggregate['games_passed']}")
        print(f"Semantic divergence: {aggregate['games_with_semantic_divergence']}")
        print(f"Replay errors:       {aggregate['games_with_replay_errors']}")
        print(f"Failure rate:        {aggregate['failure_rate']:.2%}")
        print(f"{'='*60}")
        print(f"Result: {'PASSED' if aggregate['passed'] else 'FAILED'}")
        print(f"{'='*60}")

    # Exit with appropriate code
    sys.exit(0 if aggregate["passed"] else 1)


if __name__ == "__main__":
    main()
