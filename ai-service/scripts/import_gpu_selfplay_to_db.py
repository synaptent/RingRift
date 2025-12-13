#!/usr/bin/env python
"""Import GPU selfplay JSONL games into full GameReplayDB format.

This script converts GPU selfplay JSONL records (which contain full move data)
into the canonical GameReplayDB format with state snapshots, enabling parity
validation and training pipeline usage.

Usage:
    python scripts/import_gpu_selfplay_to_db.py \
        --input data/games/gpu_generated/square8_512.jsonl \
        --output data/games/gpu_square8_canonical.db
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.db.game_replay import GameReplayDB
from app.game_engine import GameEngine
from app.models import BoardType, GameState, Move, MoveType, Position


def parse_position(pos_dict: Dict[str, Any]) -> Position:
    """Parse a position dict into a Position object."""
    return Position(
        x=pos_dict["x"],
        y=pos_dict["y"],
        z=pos_dict.get("z"),
    )


def parse_move(move_dict: Dict[str, Any], move_number: int, timestamp: str) -> Move:
    """Parse a move dict into a Move object."""
    move_type_str = move_dict.get("type", "unknown")

    # Map move type strings to MoveType enum
    move_type_map = {
        "place_ring": MoveType.PLACE_RING,
        "move_stack": MoveType.MOVE_STACK,
        "overtaking_capture": MoveType.OVERTAKING_CAPTURE,
        "continue_capture_segment": MoveType.CONTINUE_CAPTURE_SEGMENT,
        "end_capture_chain": MoveType.END_CAPTURE_CHAIN,
        "choose_line_option": MoveType.CHOOSE_LINE_OPTION,
        "choose_line_reward": MoveType.CHOOSE_LINE_REWARD,
        "process_line": MoveType.PROCESS_LINE,
        "no_line_action": MoveType.NO_LINE_ACTION,
        "choose_territory_option": MoveType.CHOOSE_TERRITORY_OPTION,
        "process_territory_region": MoveType.PROCESS_TERRITORY_REGION,
        "no_territory_action": MoveType.NO_TERRITORY_ACTION,
        "eliminate_rings_from_stack": MoveType.ELIMINATE_RINGS_FROM_STACK,
        "recovery_slide": MoveType.RECOVERY_SLIDE,
        "skip_recovery": MoveType.SKIP_RECOVERY,
        "pass_turn": MoveType.PASS_TURN,
    }

    move_type = move_type_map.get(move_type_str, MoveType.PLACE_RING)

    # Parse positions
    from_pos = parse_position(move_dict["from"]) if "from" in move_dict else None
    to_pos = parse_position(move_dict["to"]) if "to" in move_dict else None
    capture_target = parse_position(move_dict["capture_target"]) if "capture_target" in move_dict else None

    return Move(
        id=f"move-{move_number}",
        type=move_type,
        player=move_dict.get("player", 1),
        from_pos=from_pos,
        to=to_pos,
        capture_target=capture_target,
        timestamp=timestamp,
        thinkTime=move_dict.get("think_time_ms", 0),
        moveNumber=move_number,
    )


def get_board_type(board_str: str) -> BoardType:
    """Convert board type string to BoardType enum."""
    board_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "square25": BoardType.SQUARE25,
        "hexagonal": BoardType.HEXAGONAL,
    }
    return board_map.get(board_str, BoardType.SQUARE8)


def import_game(
    db: GameReplayDB,
    game_record: Dict[str, Any],
    source_file: str,
) -> bool:
    """Import a single game record into the database.

    Returns True on success, False on failure.
    """
    game_id = game_record.get("game_id", "unknown")
    board_type_str = game_record.get("board_type", "square8")
    board_type = get_board_type(board_type_str)
    num_players = game_record.get("num_players", 2)

    # Get initial state - either from record or create default
    initial_state_dict = game_record.get("initial_state")
    if initial_state_dict:
        # Reconstruct initial state from dict
        try:
            initial_state = GameEngine.create_initial_state(
                board_type=board_type,
                num_players=num_players,
            )
        except Exception as e:
            print(f"  Warning: Failed to create initial state for {game_id}: {e}")
            return False
    else:
        initial_state = GameEngine.create_initial_state(
            board_type=board_type,
            num_players=num_players,
        )

    # Parse moves
    moves_data = game_record.get("moves", [])
    timestamp = game_record.get("timestamp", datetime.now().isoformat())

    moves: List[Move] = []
    for i, move_dict in enumerate(moves_data):
        try:
            move = parse_move(move_dict, i + 1, timestamp)
            moves.append(move)
        except Exception as e:
            print(f"  Warning: Failed to parse move {i} in {game_id}: {e}")
            return False

    # Apply moves to get final state
    current_state = initial_state
    for move in moves:
        try:
            current_state = GameEngine.apply_move(current_state, move)
        except Exception as e:
            print(f"  Warning: Failed to apply move in {game_id}: {e}")
            return False

    final_state = current_state

    # Prepare metadata
    metadata = {
        "source": f"gpu_import:{source_file}",
        "original_game_id": game_record.get("game_id"),
        "termination_reason": game_record.get("termination_reason", game_record.get("victory_type")),
        "victory_type": game_record.get("victory_type"),
        "engine_mode": game_record.get("engine_mode", "gpu_heuristic"),
        "batch_id": game_record.get("batch_id"),
        "device": game_record.get("device"),
    }

    # Store the game
    try:
        db.store_game(
            game_id=f"gpu_{board_type_str}_{game_id}",
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            metadata=metadata,
            store_history_entries=True,
        )
        return True
    except Exception as e:
        print(f"  Warning: Failed to store game {game_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Import GPU selfplay JSONL games into GameReplayDB format"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output SQLite DB path"
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=0,
        help="Maximum number of games to import (0 = all)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Create output database
    db = GameReplayDB(args.output)
    source_file = Path(args.input).stem

    print(f"Importing games from {args.input} to {args.output}")

    games_imported = 0
    games_failed = 0

    with open(args.input, "r") as f:
        for line_num, line in enumerate(f, 1):
            if args.limit > 0 and games_imported >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: Invalid JSON at line {line_num}: {e}")
                games_failed += 1
                continue

            if import_game(db, record, source_file):
                games_imported += 1
                if games_imported % 50 == 0:
                    print(f"  Imported {games_imported} games...")
            else:
                games_failed += 1

    print(f"\nImport complete:")
    print(f"  Successfully imported: {games_imported}")
    print(f"  Failed: {games_failed}")
    print(f"  Output: {args.output}")

    # Verify the database
    count = db.get_game_count()
    print(f"  Total games in DB: {count}")

    sys.exit(0 if games_imported > 0 else 1)


if __name__ == "__main__":
    main()
