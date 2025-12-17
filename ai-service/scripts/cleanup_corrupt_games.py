#!/usr/bin/env python3
"""Cleanup tool for corrupt game data in selfplay databases.

Identifies and removes games with corrupt move data (e.g., from_pos=None
for move_stack moves).

Usage:
    # Analyze corruption in a database
    python cleanup_corrupt_games.py analyze --db data/games/all_selfplay_v2.db

    # Delete corrupt games
    python cleanup_corrupt_games.py delete --db data/games/all_selfplay_v2.db --confirm

    # Export only valid games to new database
    python cleanup_corrupt_games.py export-valid --db data/games/all_selfplay_v2.db \
        --output data/games/all_selfplay_v2_clean.db
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add ai-service to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Unified logging setup
try:
    from app.core.logging_config import setup_logging
    logger = setup_logging("cleanup_corrupt_games", log_dir="logs")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)


def analyze_move_corruption(db_path: str, sample_size: int = 1000) -> Dict:
    """Analyze move data corruption in a database.

    Returns statistics about corrupt vs valid games.
    """
    from app.db import GameReplayDB
    from app.models import BoardType

    db = GameReplayDB(db_path)

    stats = {
        "total_games": 0,
        "games_with_moves": 0,
        "corrupt_games": 0,
        "valid_games": 0,
        "corrupt_by_type": {},
        "sample_corrupt_ids": [],
        "sample_valid_ids": [],
    }

    board_types = [BoardType.SQUARE8, BoardType.SQUARE19, BoardType.HEXAGONAL]

    for board_type in board_types:
        for num_players in [2, 3, 4]:
            count = 0
            for meta, initial_state, moves in db.iterate_games(
                board_type=board_type,
                num_players=num_players,
                require_moves=True,
            ):
                stats["total_games"] += 1
                game_id = meta.get("game_id")

                if not moves:
                    continue

                stats["games_with_moves"] += 1
                is_corrupt = False

                # Check for corrupt move_stack moves (from_pos=None)
                for m in moves:
                    if getattr(m, "type", None) == "move_stack":
                        if m.from_pos is None:
                            is_corrupt = True
                            break

                if is_corrupt:
                    stats["corrupt_games"] += 1
                    key = f"{board_type.value}_{num_players}p"
                    stats["corrupt_by_type"][key] = stats["corrupt_by_type"].get(key, 0) + 1
                    if len(stats["sample_corrupt_ids"]) < 10:
                        stats["sample_corrupt_ids"].append(game_id)
                else:
                    stats["valid_games"] += 1
                    if len(stats["sample_valid_ids"]) < 10:
                        stats["sample_valid_ids"].append(game_id)

                count += 1
                if count >= sample_size:
                    break

            if count > 0:
                logger.info(f"  {board_type.value} {num_players}p: checked {count} games")

    return stats


def delete_corrupt_games(db_path: str, dry_run: bool = True) -> int:
    """Delete games with corrupt move data.

    Args:
        db_path: Path to database
        dry_run: If True, only count games to delete without deleting

    Returns:
        Number of games deleted (or would be deleted in dry run)
    """
    from app.db import GameReplayDB
    from app.models import BoardType

    db = GameReplayDB(db_path)
    corrupt_ids: List[str] = []

    board_types = [BoardType.SQUARE8, BoardType.SQUARE19, BoardType.HEXAGONAL]

    logger.info("Scanning for corrupt games...")

    for board_type in board_types:
        for num_players in [2, 3, 4]:
            type_count = 0
            for meta, initial_state, moves in db.iterate_games(
                board_type=board_type,
                num_players=num_players,
                require_moves=True,
            ):
                game_id = meta.get("game_id")
                if not moves:
                    continue

                # Check for corrupt move_stack moves
                for m in moves:
                    if getattr(m, "type", None) == "move_stack":
                        if m.from_pos is None:
                            corrupt_ids.append(game_id)
                            type_count += 1
                            break

            if type_count > 0:
                logger.info(f"  {board_type.value} {num_players}p: {type_count} corrupt games")

    logger.info(f"Total corrupt games: {len(corrupt_ids)}")

    if dry_run:
        logger.info("Dry run - no games deleted")
        return len(corrupt_ids)

    # Delete corrupt games
    logger.info("Deleting corrupt games...")
    conn = sqlite3.connect(db_path, timeout=60.0)

    deleted = 0
    batch_size = 100

    for i in range(0, len(corrupt_ids), batch_size):
        batch = corrupt_ids[i:i + batch_size]
        placeholders = ",".join("?" * len(batch))

        # Delete from all related tables
        tables = [
            "game_moves",
            "game_players",
            "game_initial_state",
            "game_state_snapshots",
            "game_choices",
            "game_history_entries",
            "games",
        ]

        for table in tables:
            try:
                conn.execute(f"DELETE FROM {table} WHERE game_id IN ({placeholders})", batch)
            except sqlite3.OperationalError:
                pass  # Table might not exist

        conn.commit()
        deleted += len(batch)

        if deleted % 1000 == 0:
            logger.info(f"  Deleted {deleted}/{len(corrupt_ids)} games")

    conn.execute("VACUUM")
    conn.commit()
    conn.close()

    logger.info(f"Deleted {deleted} corrupt games")
    return deleted


def export_valid_games(
    db_path: str,
    output_path: str,
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
) -> int:
    """Export only valid games to a new database.

    Args:
        db_path: Source database path
        output_path: Destination database path
        board_type: Optional filter by board type
        num_players: Optional filter by player count

    Returns:
        Number of games exported
    """
    from app.db import GameReplayDB
    from app.models import BoardType

    source_db = GameReplayDB(db_path)

    # Create destination database
    if os.path.exists(output_path):
        os.remove(output_path)
    dest_db = GameReplayDB(output_path)

    board_types = [BoardType.SQUARE8, BoardType.SQUARE19, BoardType.HEXAGONAL]
    if board_type:
        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_types = [board_type_map[board_type]]

    player_counts = [2, 3, 4]
    if num_players:
        player_counts = [num_players]

    exported = 0
    skipped = 0

    for bt in board_types:
        for np in player_counts:
            type_exported = 0
            type_skipped = 0

            for meta, initial_state, moves in source_db.iterate_games(
                board_type=bt,
                num_players=np,
                require_moves=True,
            ):
                if not moves:
                    continue

                # Check for corrupt move_stack moves
                is_corrupt = False
                for m in moves:
                    if getattr(m, "type", None) == "move_stack":
                        if m.from_pos is None:
                            is_corrupt = True
                            break

                if is_corrupt:
                    type_skipped += 1
                    skipped += 1
                    continue

                # Export valid game
                game_id = meta.get("game_id")
                try:
                    dest_db.save_game(
                        game_id=game_id,
                        initial_state=initial_state,
                        moves=moves,
                        final_state=None,  # We don't have this
                        metadata={
                            "board_type": bt.value,
                            "num_players": np,
                            "total_moves": len(moves),
                        },
                    )
                    exported += 1
                    type_exported += 1
                except Exception as e:
                    logger.warning(f"Failed to export {game_id}: {e}")
                    type_skipped += 1
                    skipped += 1

            if type_exported > 0 or type_skipped > 0:
                logger.info(f"  {bt.value} {np}p: exported {type_exported}, skipped {type_skipped}")

    logger.info(f"Total: exported {exported}, skipped {skipped}")
    return exported


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup corrupt game data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze corruption")
    analyze_parser.add_argument("--db", required=True, help="Database path")
    analyze_parser.add_argument("--sample", type=int, default=1000, help="Games per type to sample")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete corrupt games")
    delete_parser.add_argument("--db", required=True, help="Database path")
    delete_parser.add_argument("--confirm", action="store_true", help="Actually delete (not dry run)")

    # Export-valid command
    export_parser = subparsers.add_parser("export-valid", help="Export only valid games")
    export_parser.add_argument("--db", required=True, help="Source database path")
    export_parser.add_argument("--output", required=True, help="Destination database path")
    export_parser.add_argument("--board-type", choices=["square8", "square19", "hexagonal"])
    export_parser.add_argument("--num-players", type=int, choices=[2, 3, 4])

    args = parser.parse_args()

    if args.command == "analyze":
        logger.info(f"Analyzing corruption in {args.db}...")
        stats = analyze_move_corruption(args.db, sample_size=args.sample)
        print("\n=== Corruption Analysis ===")
        print(f"Total games scanned: {stats['total_games']}")
        print(f"Games with moves: {stats['games_with_moves']}")
        print(f"Corrupt games: {stats['corrupt_games']}")
        print(f"Valid games: {stats['valid_games']}")
        if stats["corrupt_by_type"]:
            print("\nCorrupt by type:")
            for k, v in sorted(stats["corrupt_by_type"].items()):
                print(f"  {k}: {v}")
        if stats["sample_corrupt_ids"]:
            print(f"\nSample corrupt IDs: {stats['sample_corrupt_ids'][:5]}")
        if stats["sample_valid_ids"]:
            print(f"Sample valid IDs: {stats['sample_valid_ids'][:5]}")

    elif args.command == "delete":
        logger.info(f"{'Dry run: ' if not args.confirm else ''}Deleting corrupt games from {args.db}...")
        count = delete_corrupt_games(args.db, dry_run=not args.confirm)
        print(f"\n{'Would delete' if not args.confirm else 'Deleted'} {count} corrupt games")

    elif args.command == "export-valid":
        logger.info(f"Exporting valid games from {args.db} to {args.output}...")
        count = export_valid_games(
            args.db,
            args.output,
            board_type=args.board_type,
            num_players=args.num_players,
        )
        print(f"\nExported {count} valid games to {args.output}")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
