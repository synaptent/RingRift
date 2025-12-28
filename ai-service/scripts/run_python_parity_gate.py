#!/usr/bin/env python3
"""Run Python-native parity gate on canonical databases.

This script validates games using Python-only replay (no Node.js required),
making it suitable for cluster nodes that lack npx.

PERMANENT PARITY GATE SOLUTION (December 2025):

The parity gate now has two modes:

1. COORDINATOR MODE (--populate-ts-hashes):
   Run on coordinator (which has Node.js) to:
   - Run full TypeScript parity validation
   - Store TS state hashes in database (ts_parity_hashes table)
   - These databases can then be synced to cluster nodes

2. CLUSTER MODE (--use-ts-hashes):
   Run on cluster nodes (no Node.js) to:
   - Validate Python replay against stored TS hashes
   - Full parity verification without running TypeScript

Workflow:
   1. Coordinator: python scripts/run_python_parity_gate.py --all --populate-ts-hashes
   2. Sync databases to cluster nodes
   3. Cluster: python scripts/run_python_parity_gate.py --all --use-ts-hashes

Usage:
    # COORDINATOR: Populate TS hashes (requires Node.js)
    python scripts/run_python_parity_gate.py --all --populate-ts-hashes

    # CLUSTER: Validate against stored TS hashes (no Node.js needed)
    python scripts/run_python_parity_gate.py --all --use-ts-hashes

    # Legacy: Python-only replay (checks Python consistency, not TS parity)
    python scripts/run_python_parity_gate.py --db data/games/canonical_hex8_2p.db

    # Validate specific board type
    python scripts/run_python_parity_gate.py --board-type hex8 --num-players 2 --use-ts-hashes

    # Dry run (don't update database status)
    python scripts/run_python_parity_gate.py --all --use-ts-hashes --dry-run

December 2025: Created permanent parity gate solution for cluster nodes without Node.js.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.parity_validator import (
    validate_database_python_only,
    validate_database_with_ts_hashes,
    populate_ts_hashes_from_validation,
    count_games_with_ts_hashes,
)
from app.db.game_replay import GameReplayDB


def find_canonical_databases(
    board_type: str | None = None,
    num_players: int | None = None,
) -> list[Path]:
    """Find canonical databases matching filters."""
    data_dir = Path(__file__).parent.parent / "data" / "games"

    pattern = "canonical_"
    if board_type:
        pattern += f"{board_type}_"
    if num_players:
        pattern += f"{num_players}p"
    pattern += "*.db"

    databases = list(data_dir.glob(pattern))
    return sorted(databases)


def main():
    parser = argparse.ArgumentParser(
        description="Run Python-native parity gate on canonical databases"
    )
    parser.add_argument(
        "--db",
        type=str,
        help="Path to specific database to validate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all canonical databases",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["hex8", "square8", "square19", "hexagonal"],
        help="Filter by board type",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        help="Filter by number of players",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum games to validate per database (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't update database status, just report results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    # New permanent parity gate options
    mode_group = parser.add_argument_group("Parity Gate Modes")
    mode_group.add_argument(
        "--use-ts-hashes",
        action="store_true",
        help="Validate against stored TypeScript hashes (for cluster nodes without Node.js)",
    )
    mode_group.add_argument(
        "--populate-ts-hashes",
        action="store_true",
        help="Run full TS validation and store hashes (for coordinator with Node.js)",
    )
    mode_group.add_argument(
        "--check-ts-hash-status",
        action="store_true",
        help="Check how many games have TS hashes stored",
    )

    args = parser.parse_args()

    # Validate mode arguments
    if args.use_ts_hashes and args.populate_ts_hashes:
        parser.error("Cannot use both --use-ts-hashes and --populate-ts-hashes")

    # Determine which databases to validate
    if args.db:
        databases = [Path(args.db)]
    elif args.all or args.board_type or args.num_players:
        databases = find_canonical_databases(args.board_type, args.num_players)
    else:
        parser.print_help()
        print("\nError: Specify --db, --all, or --board-type/--num-players")
        return 1

    if not databases:
        print("No databases found matching criteria")
        return 1

    print(f"Found {len(databases)} database(s) to validate")

    # Check TS hash status mode
    if args.check_ts_hash_status:
        print("\n" + "=" * 60)
        print("TS HASH STATUS")
        print("=" * 60)
        for db_path in databases:
            if not db_path.exists():
                print(f"{db_path.name}: NOT FOUND")
                continue
            try:
                db = GameReplayDB(str(db_path))
                games_with_hashes = count_games_with_ts_hashes(db)
                total_games = len(db.query_games(limit=100_000, require_moves=True))
                pct = (games_with_hashes / total_games * 100) if total_games > 0 else 0
                status = "READY" if games_with_hashes == total_games and total_games > 0 else "INCOMPLETE"
                print(f"{db_path.name}: {games_with_hashes}/{total_games} games ({pct:.1f}%) - {status}")
            except Exception as e:
                print(f"{db_path.name}: ERROR - {e}")
        return 0

    print("-" * 60)

    # Determine which validation function to use
    if args.populate_ts_hashes:
        print("\n*** COORDINATOR MODE: Populating TS hashes ***")
        print("This requires Node.js/npx to run TypeScript parity validation.")
        print("-" * 60)
        validation_fn = populate_ts_hashes_from_validation
        mode_name = "populate-ts-hashes"
    elif args.use_ts_hashes:
        print("\n*** CLUSTER MODE: Validating against stored TS hashes ***")
        print("No Node.js required - uses pre-computed TS hashes.")
        print("-" * 60)
        validation_fn = validate_database_with_ts_hashes
        mode_name = "use-ts-hashes"
    else:
        print("\n*** LEGACY MODE: Python-only replay validation ***")
        print("This only checks Python can replay games, not TS parity.")
        print("-" * 60)
        validation_fn = validate_database_python_only
        mode_name = "python-only"

    results = []
    total_passed = 0
    total_failed = 0
    total_hashes_stored = 0

    for db_path in databases:
        if not db_path.exists():
            print(f"Database not found: {db_path}")
            continue

        print(f"\nValidating: {db_path.name}")

        # Different argument signatures for each mode
        if args.populate_ts_hashes:
            # populate_ts_hashes_from_validation doesn't have update_status
            result = validation_fn(
                db_path,
                max_games=args.max_games,
                verbose=args.verbose,
            )
        else:
            result = validation_fn(
                db_path,
                max_games=args.max_games,
                update_status=not args.dry_run,
                verbose=args.verbose,
            )

        # Handle error case (e.g., Node.js not available for populate mode)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            if "detail" in result:
                print(f"  Detail: {result['detail']}")
            continue

        results.append(result)

        # Different result formats for different modes
        if args.populate_ts_hashes:
            games_processed = result.get("games_processed", 0)
            hashes_stored = result.get("hashes_stored", 0)
            failed = result.get("failed", 0)
            total_passed += games_processed
            total_failed += failed
            total_hashes_stored += hashes_stored
            status = "COMPLETE" if failed == 0 else "PARTIAL"
            print(f"  Result: {status} - {games_processed} games, {hashes_stored} hashes stored")
            if failed > 0:
                print(f"  Failed: {failed} games")
        else:
            total_passed += result["passed"]
            total_failed += result["failed"]

            status = "PASSED" if result["failed"] == 0 else "FAILED"
            games_info = f"{result['passed']}/{result.get('total_games', result['passed'] + result['failed'])} games"
            print(f"  Result: {status} - {games_info}")
            if result.get("failed_games"):
                print(f"  Failed games: {len(result['failed_games'])}")
                for game_id, error in result["failed_games"][:3]:
                    print(f"    - {game_id[:8]}: {error[:60]}")

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY ({mode_name})")
    print("=" * 60)
    print(f"Databases validated: {len(results)}")

    if args.populate_ts_hashes:
        print(f"Total games processed: {total_passed}")
        print(f"Total hashes stored: {total_hashes_stored}")
        print(f"Total games failed: {total_failed}")
    else:
        print(f"Total games passed: {total_passed}")
        print(f"Total games failed: {total_failed}")

    if total_passed + total_failed > 0:
        rate = total_passed / (total_passed + total_failed)
        print(f"Overall pass rate: {rate:.1%}")

    if not args.dry_run:
        print("\nDatabase status fields updated for all validated databases.")
    else:
        print("\n(Dry run - no database status changes made)")

    # Mode-specific guidance
    if args.populate_ts_hashes and total_failed == 0:
        print("\n*** TS hashes populated successfully! ***")
        print("Next steps:")
        print("  1. Sync these databases to cluster nodes")
        print("  2. On cluster: python scripts/run_python_parity_gate.py --all --use-ts-hashes")
    elif args.use_ts_hashes and total_failed == 0:
        print("\n*** Parity gate PASSED! ***")
        print("All games validated against TypeScript reference hashes.")

    # JSON output if requested
    if args.json:
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": mode_name,
            "databases": results,
            "summary": {
                "total_passed": total_passed,
                "total_failed": total_failed,
                "pass_rate": rate if total_passed + total_failed > 0 else 1.0,
            },
        }
        if args.populate_ts_hashes:
            output["summary"]["total_hashes_stored"] = total_hashes_stored
        print("\n" + json.dumps(output, indent=2))

    # Exit code based on failures
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
