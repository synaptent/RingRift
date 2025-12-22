#!/usr/bin/env python3
"""Validate and clean training data across databases and JSONLs.

This script identifies and removes game records that are missing move data
or other metadata required to train useful NNUE samples.

A valid training record requires:
1. game_status = 'completed' with valid winner
2. Either game_state_snapshots OR (game_initial_state + game_moves) for replay
3. total_moves >= 10 (min_game_length)
4. Board type and num_players metadata
"""

import argparse
import gzip
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationStats:
    """Statistics from validating a database or JSONL file."""
    path: str
    total_games: int = 0
    valid_games: int = 0
    invalid_games: int = 0
    missing_moves: int = 0
    missing_initial_state: int = 0
    missing_snapshots: int = 0
    missing_winner: int = 0
    not_completed: int = 0
    too_short: int = 0
    missing_board_type: int = 0
    corrupt_json: int = 0
    empty_file: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def entirely_unusable(self) -> bool:
        """Return True if entire file is unusable for training."""
        return self.valid_games == 0 and self.total_games > 0

    @property
    def usability_pct(self) -> float:
        """Percentage of games that are usable."""
        if self.total_games == 0:
            return 0.0
        return (self.valid_games / self.total_games) * 100


def validate_sqlite_db(db_path: str, fix: bool = False) -> ValidationStats:
    """Validate a SQLite game database.

    Args:
        db_path: Path to the SQLite database
        fix: If True, delete invalid records

    Returns:
        ValidationStats with results
    """
    stats = ValidationStats(path=db_path)

    if not os.path.exists(db_path):
        stats.errors.append("File does not exist")
        return stats

    file_size = os.path.getsize(db_path)
    if file_size == 0:
        stats.empty_file = True
        stats.errors.append("Empty file")
        return stats

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check if this is a valid game database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        if 'games' not in tables:
            stats.errors.append("Not a valid game database (no 'games' table)")
            conn.close()
            return stats

        # Check table schema
        cursor.execute("PRAGMA table_info(games)")
        columns = {row[1] for row in cursor.fetchall()}

        has_game_moves = 'game_moves' in tables
        has_snapshots = 'game_state_snapshots' in tables
        has_initial_state = 'game_initial_state' in tables

        # Get all games
        cursor.execute("SELECT * FROM games")
        games = cursor.fetchall()
        stats.total_games = len(games)

        if stats.total_games == 0:
            stats.empty_file = True
            stats.errors.append("No games in database")
            conn.close()
            return stats

        invalid_game_ids = []

        for game in games:
            game_id = game['game_id'] if 'game_id' in columns else game['id']
            is_valid = True

            # Check completion status
            # Valid statuses for training:
            # - "completed": naturally finished game
            # - "max_moves": hit move limit (still has valid data, just no natural winner)
            # Invalid statuses:
            # - "active": game still in progress (incomplete)
            # - "timeout": wall-clock timeout (may be incomplete)
            # - "abandoned": explicitly abandoned
            VALID_STATUSES = {'completed', 'max_moves'}
            game_status = game['game_status'] if 'game_status' in columns else None
            if game_status not in VALID_STATUSES:
                stats.not_completed += 1
                is_valid = False

            # Check winner
            winner = game['winner'] if 'winner' in columns else None
            if winner is None:
                stats.missing_winner += 1
                is_valid = False

            # Check total moves
            total_moves = game['total_moves'] if 'total_moves' in columns else 0
            if total_moves is None or total_moves < 10:
                stats.too_short += 1
                is_valid = False

            # Check board type
            board_type = game['board_type'] if 'board_type' in columns else None
            if board_type is None:
                stats.missing_board_type += 1
                is_valid = False

            # Check for move data availability
            has_move_data = False

            if has_snapshots:
                cursor.execute(
                    "SELECT COUNT(*) FROM game_state_snapshots WHERE game_id = ?",
                    (game_id,)
                )
                snapshot_count = cursor.fetchone()[0]
                if snapshot_count > 0:
                    has_move_data = True
                else:
                    stats.missing_snapshots += 1

            if not has_move_data and has_initial_state and has_game_moves:
                # Check for initial state
                cursor.execute(
                    "SELECT COUNT(*) FROM game_initial_state WHERE game_id = ?",
                    (game_id,)
                )
                has_init = cursor.fetchone()[0] > 0

                # Check for moves
                cursor.execute(
                    "SELECT COUNT(*) FROM game_moves WHERE game_id = ?",
                    (game_id,)
                )
                move_count = cursor.fetchone()[0]

                if has_init and move_count > 0:
                    has_move_data = True
                elif not has_init:
                    stats.missing_initial_state += 1
                else:
                    stats.missing_moves += 1

            if not has_move_data and has_snapshots:
                # Already counted missing_snapshots
                pass
            elif not has_move_data:
                if not has_snapshots and not (has_initial_state and has_game_moves):
                    stats.missing_moves += 1
                is_valid = False

            if is_valid:
                stats.valid_games += 1
            else:
                stats.invalid_games += 1
                invalid_game_ids.append(game_id)

        # Fix: delete invalid records
        if fix and invalid_game_ids:
            logger.info(f"Deleting {len(invalid_game_ids)} invalid games from {db_path}")

            placeholders = ",".join("?" * len(invalid_game_ids))

            # Delete from all related tables
            if has_snapshots:
                cursor.execute(
                    f"DELETE FROM game_state_snapshots WHERE game_id IN ({placeholders})",
                    invalid_game_ids
                )
            if has_game_moves:
                cursor.execute(
                    f"DELETE FROM game_moves WHERE game_id IN ({placeholders})",
                    invalid_game_ids
                )
            if has_initial_state:
                cursor.execute(
                    f"DELETE FROM game_initial_state WHERE game_id IN ({placeholders})",
                    invalid_game_ids
                )

            cursor.execute(
                f"DELETE FROM games WHERE game_id IN ({placeholders})",
                invalid_game_ids
            )

            conn.commit()

            # Vacuum to reclaim space
            cursor.execute("VACUUM")
            conn.commit()

            logger.info(f"Cleaned {len(invalid_game_ids)} invalid games, vacuumed database")

        conn.close()

    except sqlite3.Error as e:
        stats.errors.append(f"SQLite error: {e}")
    except Exception as e:
        stats.errors.append(f"Error: {e}")

    return stats


def validate_jsonl_file(jsonl_path: str, fix: bool = False) -> ValidationStats:
    """Validate a JSONL game file.

    Args:
        jsonl_path: Path to the JSONL file
        fix: If True, rewrite file without invalid records

    Returns:
        ValidationStats with results
    """
    stats = ValidationStats(path=jsonl_path)

    if not os.path.exists(jsonl_path):
        stats.errors.append("File does not exist")
        return stats

    file_size = os.path.getsize(jsonl_path)
    if file_size == 0:
        stats.empty_file = True
        stats.errors.append("Empty file")
        return stats

    valid_records = []

    try:
        # Handle both .jsonl and .jsonl.gz
        if jsonl_path.endswith('.gz'):
            opener = gzip.open
        else:
            opener = open

        with opener(jsonl_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                stats.total_games += 1
                is_valid = True

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    stats.corrupt_json += 1
                    stats.invalid_games += 1
                    continue

                # Check game status
                # Valid statuses: "completed" (natural end) or "max_moves" (hit limit)
                VALID_STATUSES = {'completed', 'max_moves'}
                game_status = record.get('game_status') or record.get('status')
                if game_status not in VALID_STATUSES:
                    stats.not_completed += 1
                    is_valid = False

                # Check winner
                winner = record.get('winner')
                if winner is None:
                    stats.missing_winner += 1
                    is_valid = False

                # Check moves
                moves = record.get('moves') or record.get('move_history') or record.get('moveHistory')
                total_moves = record.get('total_moves') or record.get('totalMoves') or (len(moves) if moves else 0)

                if not moves or len(moves) == 0:
                    stats.missing_moves += 1
                    is_valid = False
                elif total_moves < 10:
                    stats.too_short += 1
                    is_valid = False

                # Check board type
                board_type = record.get('board_type') or record.get('boardType')
                if not board_type:
                    stats.missing_board_type += 1
                    is_valid = False

                # Check initial state
                initial_state = record.get('initial_state') or record.get('initialState')
                if not initial_state and not record.get('states') and not record.get('snapshots'):
                    stats.missing_initial_state += 1
                    is_valid = False

                if is_valid:
                    stats.valid_games += 1
                    valid_records.append(line)
                else:
                    stats.invalid_games += 1

        if stats.total_games == 0:
            stats.empty_file = True
            stats.errors.append("No records in file")

        # Fix: rewrite file with only valid records
        if fix and stats.invalid_games > 0 and stats.valid_games > 0:
            logger.info(f"Rewriting {jsonl_path} with {stats.valid_games} valid records")

            with opener(jsonl_path, 'wt', encoding='utf-8') as f:
                for record in valid_records:
                    f.write(record + '\n')

            logger.info(f"Removed {stats.invalid_games} invalid records")

    except Exception as e:
        stats.errors.append(f"Error: {e}")

    return stats


def find_data_files(directory: str, recursive: bool = True) -> tuple[list[str], list[str]]:
    """Find all DB and JSONL files in a directory.

    Returns:
        Tuple of (db_paths, jsonl_paths)
    """
    db_paths = []
    jsonl_paths = []

    if recursive:
        for root, dirs, files in os.walk(directory):
            for f in files:
                path = os.path.join(root, f)
                if f.endswith('.db'):
                    db_paths.append(path)
                elif f.endswith('.jsonl') or f.endswith('.jsonl.gz'):
                    jsonl_paths.append(path)
    else:
        for f in os.listdir(directory):
            path = os.path.join(directory, f)
            if f.endswith('.db'):
                db_paths.append(path)
            elif f.endswith('.jsonl') or f.endswith('.jsonl.gz'):
                jsonl_paths.append(path)

    return db_paths, jsonl_paths


def print_stats(stats: ValidationStats, verbose: bool = False) -> None:
    """Print validation statistics."""
    status = "EMPTY" if stats.empty_file else (
        "UNUSABLE" if stats.entirely_unusable else (
            "PARTIAL" if stats.invalid_games > 0 else "OK"
        )
    )

    print(f"\n{stats.path}")
    print(f"  Status: {status}")
    print(f"  Total games: {stats.total_games}")
    print(f"  Valid games: {stats.valid_games} ({stats.usability_pct:.1f}%)")
    print(f"  Invalid games: {stats.invalid_games}")

    if verbose and stats.invalid_games > 0:
        if stats.missing_moves > 0:
            print(f"    - Missing moves: {stats.missing_moves}")
        if stats.missing_initial_state > 0:
            print(f"    - Missing initial state: {stats.missing_initial_state}")
        if stats.missing_snapshots > 0:
            print(f"    - Missing snapshots: {stats.missing_snapshots}")
        if stats.missing_winner > 0:
            print(f"    - Missing winner: {stats.missing_winner}")
        if stats.not_completed > 0:
            print(f"    - Not completed: {stats.not_completed}")
        if stats.too_short > 0:
            print(f"    - Too short (<10 moves): {stats.too_short}")
        if stats.missing_board_type > 0:
            print(f"    - Missing board type: {stats.missing_board_type}")
        if stats.corrupt_json > 0:
            print(f"    - Corrupt JSON: {stats.corrupt_json}")

    if stats.errors:
        for error in stats.errors:
            print(f"  ERROR: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and clean training data files"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Paths to validate (files or directories)"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Delete invalid records (otherwise just report)"
    )
    parser.add_argument(
        "--delete-unusable",
        action="store_true",
        help="Delete files that are entirely unusable"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed breakdown of issues"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search directories recursively"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    all_stats = []
    files_deleted = []

    for path in args.paths:
        if os.path.isfile(path):
            if path.endswith('.db'):
                stats = validate_sqlite_db(path, fix=args.fix)
                all_stats.append(stats)
            elif path.endswith('.jsonl') or path.endswith('.jsonl.gz'):
                stats = validate_jsonl_file(path, fix=args.fix)
                all_stats.append(stats)
        elif os.path.isdir(path):
            db_paths, jsonl_paths = find_data_files(path, recursive=not args.no_recursive)

            for db_path in db_paths:
                stats = validate_sqlite_db(db_path, fix=args.fix)
                all_stats.append(stats)

            for jsonl_path in jsonl_paths:
                stats = validate_jsonl_file(jsonl_path, fix=args.fix)
                all_stats.append(stats)

    # Delete unusable files if requested
    if args.delete_unusable:
        for stats in all_stats:
            if stats.entirely_unusable or stats.empty_file:
                try:
                    os.remove(stats.path)
                    files_deleted.append(stats.path)
                    logger.info(f"Deleted unusable file: {stats.path}")
                except Exception as e:
                    logger.error(f"Failed to delete {stats.path}: {e}")

    # Output results
    if args.json:
        results = {
            "total_files": len(all_stats),
            "files_with_issues": sum(1 for s in all_stats if s.invalid_games > 0),
            "entirely_unusable": sum(1 for s in all_stats if s.entirely_unusable),
            "empty_files": sum(1 for s in all_stats if s.empty_file),
            "total_games": sum(s.total_games for s in all_stats),
            "valid_games": sum(s.valid_games for s in all_stats),
            "invalid_games": sum(s.invalid_games for s in all_stats),
            "files_deleted": files_deleted,
            "details": [
                {
                    "path": s.path,
                    "total": s.total_games,
                    "valid": s.valid_games,
                    "invalid": s.invalid_games,
                    "usability_pct": s.usability_pct,
                    "entirely_unusable": s.entirely_unusable,
                    "empty": s.empty_file,
                }
                for s in all_stats
            ]
        }
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'='*60}")
        print("TRAINING DATA VALIDATION REPORT")
        print(f"{'='*60}")

        for stats in all_stats:
            print_stats(stats, verbose=args.verbose)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total files scanned: {len(all_stats)}")
        print(f"Files with issues: {sum(1 for s in all_stats if s.invalid_games > 0)}")
        print(f"Entirely unusable: {sum(1 for s in all_stats if s.entirely_unusable)}")
        print(f"Empty files: {sum(1 for s in all_stats if s.empty_file)}")
        print(f"Total games: {sum(s.total_games for s in all_stats)}")
        print(f"Valid games: {sum(s.valid_games for s in all_stats)}")
        print(f"Invalid games: {sum(s.invalid_games for s in all_stats)}")

        if files_deleted:
            print(f"\nDeleted {len(files_deleted)} unusable files")

    # Return exit code based on issues found
    has_issues = any(s.entirely_unusable or s.empty_file for s in all_stats)
    return 1 if has_issues else 0


if __name__ == "__main__":
    sys.exit(main())
