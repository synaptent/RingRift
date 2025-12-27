#!/usr/bin/env python3
"""Run Python-native parity gate on canonical databases.

This script validates games using Python-only replay (no Node.js required),
making it suitable for cluster nodes that lack npx.

Usage:
    # Validate a specific database
    python scripts/run_python_parity_gate.py --db data/games/canonical_hex8_2p.db

    # Validate all canonical databases
    python scripts/run_python_parity_gate.py --all

    # Validate specific board type
    python scripts/run_python_parity_gate.py --board-type hex8 --num-players 2

    # Dry run (don't update database status)
    python scripts/run_python_parity_gate.py --all --dry-run

December 2025: Created to enable parity validation on cluster nodes without Node.js.
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

from app.db.parity_validator import validate_database_python_only


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

    args = parser.parse_args()

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
    print("-" * 60)

    results = []
    total_passed = 0
    total_failed = 0

    for db_path in databases:
        if not db_path.exists():
            print(f"Database not found: {db_path}")
            continue

        print(f"\nValidating: {db_path.name}")

        result = validate_database_python_only(
            db_path,
            max_games=args.max_games,
            update_status=not args.dry_run,
            verbose=args.verbose,
        )

        results.append(result)
        total_passed += result["passed"]
        total_failed += result["failed"]

        status = "PASSED" if result["failed"] == 0 else "FAILED"
        print(f"  Result: {status} - {result['passed']}/{result['total_games']} games")
        if result["failed_games"]:
            print(f"  Failed games: {len(result['failed_games'])}")
            for game_id, error in result["failed_games"][:3]:
                print(f"    - {game_id[:8]}: {error[:60]}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Databases validated: {len(results)}")
    print(f"Total games passed: {total_passed}")
    print(f"Total games failed: {total_failed}")
    if total_passed + total_failed > 0:
        rate = total_passed / (total_passed + total_failed)
        print(f"Overall pass rate: {rate:.1%}")

    if not args.dry_run:
        print("\nDatabase status fields updated for all validated databases.")
    else:
        print("\n(Dry run - no database status changes made)")

    # JSON output if requested
    if args.json:
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "databases": results,
            "summary": {
                "total_passed": total_passed,
                "total_failed": total_failed,
                "pass_rate": rate if total_passed + total_failed > 0 else 1.0,
            },
        }
        print("\n" + json.dumps(output, indent=2))

    # Exit code based on failures
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
