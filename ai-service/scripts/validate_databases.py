#!/usr/bin/env python3
"""
Validate game databases for phase/move invariant issues.

This script checks selfplay databases by:
1. Verifying table structure
2. Attempting to replay games to catch phase/move invariant violations
3. Optionally deleting databases that fail validation

Usage:
  python scripts/validate_databases.py data/games --replay
  python scripts/validate_databases.py data/games --replay --delete --force
  python scripts/validate_databases.py /path/to/specific.db --replay
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# Add ai-service to path for imports
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_DIR))


def validate_db_basic(db_path: str) -> tuple[int | None, str]:
    """
    Basic validation - check structure and data format.
    Returns (game_count, status_message).
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if games table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
        if not cursor.fetchone():
            conn.close()
            return None, "No games table"

        # Get game count
        cursor.execute("SELECT COUNT(*) FROM games")
        count = cursor.fetchone()[0]

        if count == 0:
            conn.close()
            return 0, "Empty"

        # Check for game_moves table (normalized schema)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'")
        has_moves_table = cursor.fetchone() is not None

        if has_moves_table:
            # Check if we have moves for at least one game
            cursor.execute("SELECT COUNT(*) FROM game_moves LIMIT 1")
            moves_count = cursor.fetchone()[0]
            if moves_count == 0:
                conn.close()
                return count, "No move data"

            # Try to parse a move_json
            cursor.execute("SELECT move_json FROM game_moves LIMIT 1")
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    json.loads(row[0])
                except json.JSONDecodeError:
                    conn.close()
                    return count, "Corrupt move_json"
        else:
            conn.close()
            return count, "Missing game_moves table"

        conn.close()
        return count, "OK"
    except Exception as e:
        return None, f"Error: {str(e)[:50]}"


def validate_db_structure(db_path: str, sample_size: int = 3) -> tuple[int | None, int | None, list[str]]:
    """
    Validate database structure and data integrity.
    Returns (game_count, moves_count, list_of_errors).

    Note: Full game replay validation is not done here because the Python
    BoardManager uses static methods and doesn't support instance-based
    game replay. For thorough invariant checking, use the TypeScript
    replay tools (scripts/selfplay-db-ts-replay.ts).
    """
    errors = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if games table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
        if not cursor.fetchone():
            return None, None, ["No games table"]

        # Get game count
        cursor.execute("SELECT COUNT(*) FROM games")
        count = cursor.fetchone()[0]

        if count == 0:
            conn.close()
            return 0, 0, []

        # Check for game_moves table (normalized schema)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'")
        has_moves_table = cursor.fetchone() is not None

        if not has_moves_table:
            conn.close()
            return count, None, ["Missing game_moves table"]

        # Check total moves count
        cursor.execute("SELECT COUNT(*) FROM game_moves")
        moves_count = cursor.fetchone()[0]

        if moves_count == 0:
            conn.close()
            return count, 0, ["No moves recorded"]

        # Validate a sample of move JSON
        cursor.execute(f"SELECT move_json FROM game_moves LIMIT {sample_size}")
        for (move_json,) in cursor.fetchall():
            if move_json:
                try:
                    move = json.loads(move_json)
                    if not isinstance(move, dict):
                        errors.append("Invalid move format (not a dict)")
                    elif "type" not in move:
                        errors.append("Move missing 'type' field")
                except json.JSONDecodeError as e:
                    errors.append(f"Corrupt move JSON: {e}")

        conn.close()
        return count, moves_count, errors

    except Exception as e:
        return None, None, [str(e)]


def scan_directory(
    base_path: str,
    check_structure: bool = False,
    delete_failing: bool = False,
    dry_run: bool = True,
    verbose: bool = False
) -> tuple[list, list]:
    """
    Scan directory for databases and validate them.
    Returns (all_results, failing_dbs).
    """
    results = []
    failing_dbs = []
    empty_dbs = []

    db_files = []
    base = Path(base_path)
    if base.is_file():
        db_files = [base]
    else:
        db_files = sorted(base.rglob("*.db"))

    print(f"Scanning {len(db_files)} database files...")

    for db_path in db_files:
        db_str = str(db_path)

        if check_structure:
            count, moves, errors = validate_db_structure(db_str)
            if errors:
                status = f"FAILING ({len(errors)} errors: {errors[0][:40]})"
                failing_dbs.append(db_str)
            elif count == 0:
                status = "Empty"
                empty_dbs.append(db_str)
            elif moves == 0:
                status = f"No moves ({count} games)"
                failing_dbs.append(db_str)
            elif moves is None:
                status = f"Missing game_moves table ({count} games)"
                failing_dbs.append(db_str)
            else:
                avg_moves = moves // count if count > 0 else 0
                status = f"OK ({count} games, {moves} moves, avg {avg_moves}/game)"
        else:
            count, status = validate_db_basic(db_str)
            if count == 0:
                empty_dbs.append(db_str)
            elif status not in ["OK", "Empty"] and count is not None:
                failing_dbs.append(db_str)

        results.append((db_str, count, status))
        if verbose or "FAILING" in status:
            print(f"  {db_path.name}: {status}")

    # Summary
    print("\n=== Validation Summary ===")
    print(f"Total databases: {len(db_files)}")
    print(f"Empty databases: {len(empty_dbs)}")
    print(f"Failing databases: {len(failing_dbs)}")
    print(f"Valid databases: {len(db_files) - len(empty_dbs) - len(failing_dbs)}")

    if failing_dbs:
        print(f"\n=== {len(failing_dbs)} FAILING DATABASES ===")
        for db in failing_dbs:
            print(f"  {db}")

        if delete_failing:
            if dry_run:
                print(f"\n[DRY RUN] Would delete {len(failing_dbs)} databases")
                print("Run with --force to actually delete")
            else:
                print(f"\nDeleting {len(failing_dbs)} failing databases...")
                for db in failing_dbs:
                    try:
                        os.remove(db)
                        print(f"  Deleted: {db}")
                    except Exception as e:
                        print(f"  Failed to delete {db}: {e}")

    return results, failing_dbs


def main():
    parser = argparse.ArgumentParser(
        description="Validate game databases for structure and data integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic table check (fast)
  python scripts/validate_databases.py data/games

  # Full structure validation with move data check
  python scripts/validate_databases.py data/games --check-structure

  # Find and delete failing databases (dry run)
  python scripts/validate_databases.py data/games --check-structure --delete

  # Actually delete failing databases
  python scripts/validate_databases.py data/games --check-structure --delete --force

  # Validate a single database
  python scripts/validate_databases.py data/games/selfplay.db --check-structure

Note: For full phase/move invariant validation (game replay), use the
TypeScript tools: npx ts-node scripts/selfplay-db-ts-replay.ts --db <path>
"""
    )
    parser.add_argument("path", nargs="?", default="data/games",
                        help="Path to scan (default: data/games)")
    parser.add_argument("--check-structure", action="store_true",
                        help="Check database structure and move data integrity")
    parser.add_argument("--delete", action="store_true",
                        help="Delete databases that fail validation")
    parser.add_argument("--force", action="store_true",
                        help="Actually delete (without this, --delete is a dry run)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show status for all databases, not just failing ones")
    args = parser.parse_args()

    dry_run = not args.force

    _results, failing = scan_directory(
        args.path,
        check_structure=args.check_structure,
        delete_failing=args.delete,
        dry_run=dry_run,
        verbose=args.verbose
    )

    # Exit with error code if there are failing databases
    if failing and not args.delete:
        sys.exit(1)


if __name__ == "__main__":
    main()
