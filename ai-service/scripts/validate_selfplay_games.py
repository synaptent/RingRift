#!/usr/bin/env python3
"""Validate selfplay games and identify files with high invalid rates.

This script checks if games can be replayed successfully and reports
statistics on invalid games per file/directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import BoardType, GameState, Move, MoveType
from app.rules.default_engine import DefaultRulesEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_initial_state(board_type: BoardType, num_players: int = 2) -> GameState:
    """Create initial game state for the given board type."""
    from app.training.generate_data import create_initial_state as _create
    return _create(board_type, num_players)


def parse_move(move_data: dict, board_type: BoardType) -> Move | None:
    """Parse a move from JSONL data."""
    try:
        move_type_str = move_data.get("type", move_data.get("move_type", "PLACE"))
        if isinstance(move_type_str, str):
            move_type = MoveType[move_type_str.upper()]
        else:
            move_type = MoveType(move_type_str)

        return Move(
            player=move_data.get("player", 1),
            move_type=move_type,
            position=tuple(move_data["position"]) if move_data.get("position") else None,
            stack_index=move_data.get("stack_index"),
            target_position=tuple(move_data["target_position"]) if move_data.get("target_position") else None,
        )
    except Exception as e:
        return None


def validate_game(game_data: dict, engine: DefaultRulesEngine) -> tuple[bool, str]:
    """Validate a single game by replaying it.

    Returns:
        (is_valid, error_message)
    """
    try:
        # Get board type
        board_type_str = game_data.get("board_type", "hex8")
        if isinstance(board_type_str, str):
            board_type = BoardType[board_type_str.upper()]
        else:
            board_type = BoardType(board_type_str)

        num_players = game_data.get("num_players", 2)

        # Create initial state
        state = create_initial_state(board_type, num_players)

        # Get moves
        moves = game_data.get("moves", [])
        if not moves:
            return False, "No moves in game"

        # Replay moves
        for i, move_data in enumerate(moves):
            move = parse_move(move_data, board_type)
            if move is None:
                return False, f"Failed to parse move {i}"

            try:
                state = engine.apply_move(state, move)
            except Exception as e:
                return False, f"Move {i} failed: {str(e)[:100]}"

        return True, ""

    except Exception as e:
        return False, f"Game validation error: {str(e)[:100]}"


def validate_jsonl_file(filepath: Path, engine: DefaultRulesEngine,
                        max_games: int = 100) -> dict:
    """Validate games in a JSONL file.

    Returns:
        dict with validation statistics
    """
    stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": defaultdict(int),
        "filepath": str(filepath),
    }

    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                if max_games and stats["total"] >= max_games:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    game_data = json.loads(line)
                    is_valid, error = validate_game(game_data, engine)

                    stats["total"] += 1
                    if is_valid:
                        stats["valid"] += 1
                    else:
                        stats["invalid"] += 1
                        # Categorize error
                        if "chain capture" in error.lower():
                            stats["errors"]["chain_capture"] += 1
                        elif "no target stack" in error.lower():
                            stats["errors"]["no_target_stack"] += 1
                        elif "Move" in error and "failed" in error:
                            stats["errors"]["move_application"] += 1
                        else:
                            stats["errors"]["other"] += 1

                except json.JSONDecodeError:
                    stats["invalid"] += 1
                    stats["total"] += 1
                    stats["errors"]["json_parse"] += 1

    except Exception as e:
        stats["error"] = str(e)

    # Calculate rate
    if stats["total"] > 0:
        stats["invalid_rate"] = stats["invalid"] / stats["total"]
    else:
        stats["invalid_rate"] = 0.0

    return stats


def scan_directory(dirpath: Path, engine: DefaultRulesEngine,
                   max_games_per_file: int = 50) -> list[dict]:
    """Scan a directory for JSONL files and validate them."""
    results = []

    jsonl_files = list(dirpath.glob("*.jsonl"))
    if not jsonl_files:
        # Check subdirectories
        for subdir in dirpath.iterdir():
            if subdir.is_dir():
                jsonl_files.extend(subdir.glob("*.jsonl"))

    for filepath in jsonl_files:
        stats = validate_jsonl_file(filepath, engine, max_games_per_file)
        results.append(stats)

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate selfplay games")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to JSONL file or directory to scan")
    parser.add_argument("--max-games", type=int, default=50,
                        help="Max games to check per file")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Invalid rate threshold for deletion (default: 0.9)")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete files with high invalid rates")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    args = parser.parse_args()

    engine = DefaultRulesEngine()
    path = Path(args.path)

    files_to_delete = []
    dirs_to_delete = []

    if path.is_file():
        stats = validate_jsonl_file(path, engine, args.max_games)
        logger.info(f"File: {path}")
        logger.info(f"  Total: {stats['total']}, Valid: {stats['valid']}, "
                   f"Invalid: {stats['invalid']} ({stats['invalid_rate']:.1%})")
        if stats['invalid_rate'] >= args.threshold:
            files_to_delete.append(path)

    elif path.is_dir():
        # Check if this is a selfplay directory with subdirectories
        subdirs = [d for d in path.iterdir() if d.is_dir()]

        if subdirs:
            # Scan each subdirectory
            for subdir in sorted(subdirs):
                results = scan_directory(subdir, engine, args.max_games)

                if not results:
                    continue

                total_games = sum(r["total"] for r in results)
                total_invalid = sum(r["invalid"] for r in results)

                if total_games > 0:
                    invalid_rate = total_invalid / total_games

                    if args.verbose or invalid_rate >= args.threshold:
                        logger.info(f"\n{subdir.name}/")
                        logger.info(f"  Files: {len(results)}, Games: {total_games}, "
                                   f"Invalid: {total_invalid} ({invalid_rate:.1%})")

                        if invalid_rate >= args.threshold:
                            dirs_to_delete.append(subdir)
                            logger.info(f"  >>> MARKED FOR DELETION (>{args.threshold:.0%} invalid)")
        else:
            # Scan files in this directory directly
            results = scan_directory(path, engine, args.max_games)
            for stats in results:
                if stats['invalid_rate'] >= args.threshold:
                    files_to_delete.append(Path(stats['filepath']))
                    logger.info(f"INVALID: {stats['filepath']} ({stats['invalid_rate']:.1%})")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary:")
    logger.info(f"  Files to delete: {len(files_to_delete)}")
    logger.info(f"  Directories to delete: {len(dirs_to_delete)}")

    if args.delete and (files_to_delete or dirs_to_delete):
        logger.info(f"\nDeleting...")

        import shutil

        for f in files_to_delete:
            try:
                f.unlink()
                logger.info(f"  Deleted file: {f}")
            except Exception as e:
                logger.error(f"  Failed to delete {f}: {e}")

        for d in dirs_to_delete:
            try:
                shutil.rmtree(d)
                logger.info(f"  Deleted directory: {d}")
            except Exception as e:
                logger.error(f"  Failed to delete {d}: {e}")

        logger.info("Done!")
    elif files_to_delete or dirs_to_delete:
        logger.info("\nRun with --delete to actually remove these files/directories")


if __name__ == "__main__":
    main()
