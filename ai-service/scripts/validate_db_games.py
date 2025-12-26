#!/usr/bin/env python3
"""Validate games in training databases and optionally delete invalid ones.

This script validates games stored in SQLite databases by attempting to replay
their move sequences. Games that cannot be replayed are marked as invalid.
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import BoardType, Move, MoveType
from app.rules.default_engine import DefaultRulesEngine
from app.training.generate_data import create_initial_state

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_game_moves(conn: sqlite3.Connection, game_id: str) -> list[dict]:
    """Get moves for a game from the database."""
    cur = conn.cursor()
    cur.execute("""
        SELECT move_number, player, move_type, move_json
        FROM game_moves
        WHERE game_id = ?
        ORDER BY move_number
    """, (game_id,))

    moves = []
    for row in cur.fetchall():
        move_number, player, move_type, move_json = row

        move = {
            "move_number": move_number,
            "player": player,
            "move_type": move_type,
        }

        # Parse JSON move data
        if move_json:
            try:
                move_data = json.loads(move_json)

                # Extract position from 'from' or 'to' field
                from_pos = move_data.get("from") or move_data.get("position")
                if from_pos and isinstance(from_pos, dict):
                    move["position"] = (from_pos.get("x"), from_pos.get("y"))
                elif from_pos and isinstance(from_pos, list):
                    move["position"] = tuple(from_pos[:2])

                # Extract target position
                to_pos = move_data.get("to") or move_data.get("target")
                if to_pos and isinstance(to_pos, dict):
                    move["target_position"] = (to_pos.get("x"), to_pos.get("y"))
                elif to_pos and isinstance(to_pos, list):
                    move["target_position"] = tuple(to_pos[:2])

                # Extract stack index
                move["stack_index"] = move_data.get("stackIndex") or move_data.get("stack_index")

                # Get capture target if present
                if move_data.get("captureTarget"):
                    ct = move_data["captureTarget"]
                    if isinstance(ct, dict):
                        move["target_position"] = (ct.get("x"), ct.get("y"))

                # Override move type from JSON if present
                if move_data.get("type"):
                    move["move_type"] = move_data["type"]

            except json.JSONDecodeError:
                pass

        moves.append(move)

    return moves


MOVE_TYPE_MAP = {
    "place_ring": "PLACE_RING",
    "place_marker": "PLACE_RING",
    "move_ring": "MOVE_RING",
    "move_stack": "MOVE_STACK",
    "capture": "OVERTAKING_CAPTURE",
    "chain_capture": "CHAIN_CAPTURE",
    "overtaking_capture": "OVERTAKING_CAPTURE",
    "place": "PLACE_RING",
    "move": "MOVE_RING",
    "build_stack": "BUILD_STACK",
    "swap": "SWAP_SIDES",
    "swap_sides": "SWAP_SIDES",
    "skip": "SKIP_PLACEMENT",
    "skip_placement": "SKIP_PLACEMENT",
    "line_formation": "LINE_FORMATION",
    "territory_claim": "TERRITORY_CLAIM",
}


def validate_game_from_db(conn: sqlite3.Connection, game_id: str,
                          board_type: BoardType, num_players: int,
                          engine: DefaultRulesEngine) -> tuple[bool, str]:
    """Validate a single game by replaying its moves.

    Returns:
        (is_valid, error_message)
    """
    try:
        moves = get_game_moves(conn, game_id)
        if not moves:
            return False, "No moves found"

        state = create_initial_state(board_type, num_players)

        for i, move_data in enumerate(moves):
            try:
                move_type_str = move_data.get("move_type", "PLACE")
                # Map to standard move types
                move_type_str = MOVE_TYPE_MAP.get(move_type_str.lower(), move_type_str.upper())
                if isinstance(move_type_str, str):
                    move_type = MoveType[move_type_str.upper()]
                else:
                    move_type = MoveType(move_type_str)

                move = Move(
                    id=f"move-{i}",
                    type=move_type,
                    player=move_data["player"],
                    from_pos=move_data.get("position"),
                    to=move_data.get("target_position"),
                    capture_target=move_data.get("target_position") if move_type == MoveType.CHAIN_CAPTURE else None,
                )

                state = engine.apply_move(state, move)

            except Exception as e:
                return False, f"Move {i} failed: {str(e)[:80]}"

        return True, ""

    except Exception as e:
        return False, f"Game error: {str(e)[:80]}"


def validate_database(db_path: str, max_games: int = 100,
                     delete_invalid: bool = False) -> dict:
    """Validate games in a database.

    Returns:
        Statistics dict
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    engine = DefaultRulesEngine()

    # Get database info
    cur.execute("SELECT COUNT(*) FROM games")
    total_games = cur.fetchone()[0]

    # Get sample of games
    cur.execute("""
        SELECT game_id, board_type, num_players
        FROM games
        WHERE winner IS NOT NULL
        LIMIT ?
    """, (max_games,))

    games = cur.fetchall()

    stats = {
        "total_in_db": total_games,
        "sampled": len(games),
        "valid": 0,
        "invalid": 0,
        "invalid_ids": [],
        "error_types": {},
    }

    for game_id, board_type_str, num_players in games:
        try:
            board_type = BoardType[board_type_str.upper()]
        except (KeyError, ValueError, AttributeError):
            board_type = BoardType.HEX8

        is_valid, error = validate_game_from_db(
            conn, game_id, board_type, num_players, engine
        )

        if is_valid:
            stats["valid"] += 1
        else:
            stats["invalid"] += 1
            stats["invalid_ids"].append(game_id)

            # Categorize error
            if "chain capture" in error.lower():
                stats["error_types"]["chain_capture"] = stats["error_types"].get("chain_capture", 0) + 1
            elif "no target" in error.lower():
                stats["error_types"]["no_target"] = stats["error_types"].get("no_target", 0) + 1
            else:
                stats["error_types"]["other"] = stats["error_types"].get("other", 0) + 1

    if stats["sampled"] > 0:
        stats["invalid_rate"] = stats["invalid"] / stats["sampled"]
    else:
        stats["invalid_rate"] = 0.0

    # Delete invalid games if requested
    if delete_invalid and stats["invalid_ids"]:
        logger.info(f"Deleting {len(stats['invalid_ids'])} invalid games...")

        # Delete in batches
        batch_size = 100
        for i in range(0, len(stats["invalid_ids"]), batch_size):
            batch = stats["invalid_ids"][i:i+batch_size]
            placeholders = ",".join("?" * len(batch))

            # Delete from all related tables
            for table in ["game_moves", "game_players", "game_initial_state",
                         "game_state_snapshots", "game_choices",
                         "game_history_entries", "game_nnue_features"]:
                try:
                    cur.execute(f"DELETE FROM {table} WHERE game_id IN ({placeholders})", batch)
                except sqlite3.OperationalError:
                    pass  # Table might not exist

            cur.execute(f"DELETE FROM games WHERE game_id IN ({placeholders})", batch)

        conn.commit()

        # Report remaining
        cur.execute("SELECT COUNT(*) FROM games")
        remaining = cur.fetchone()[0]
        stats["remaining_after_delete"] = remaining
        logger.info(f"Remaining games: {remaining}")

    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate games in training databases")
    parser.add_argument("--db", type=str, required=True, help="Path to database")
    parser.add_argument("--max-games", type=int, default=100, help="Max games to validate")
    parser.add_argument("--delete", action="store_true", help="Delete invalid games")
    parser.add_argument("--delete-all-invalid", action="store_true",
                        help="Delete ALL invalid games (validates all, not just sample)")
    args = parser.parse_args()

    db_path = args.db

    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        return 1

    logger.info(f"Validating: {db_path}")

    if args.delete_all_invalid:
        # Validate all games
        stats = validate_database(db_path, max_games=999999, delete_invalid=True)
    else:
        stats = validate_database(db_path, args.max_games, args.delete)

    logger.info(f"\nResults for {db_path}:")
    logger.info(f"  Total in DB: {stats['total_in_db']}")
    logger.info(f"  Sampled: {stats['sampled']}")
    logger.info(f"  Valid: {stats['valid']}")
    logger.info(f"  Invalid: {stats['invalid']} ({stats['invalid_rate']:.1%})")

    if stats["error_types"]:
        logger.info(f"  Error types: {stats['error_types']}")

    if "remaining_after_delete" in stats:
        logger.info(f"  Remaining after delete: {stats['remaining_after_delete']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
