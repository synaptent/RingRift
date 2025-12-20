#!/usr/bin/env python
"""Convert selfplay JSONL games to eval state pool format.

This script extracts mid-game states from selfplay JSONL files (from
generate_gumbel_selfplay.py or similar) and converts them to the JSONL
format expected by eval_pools for CMA-ES and other training scripts.

Usage:
    python scripts/convert_selfplay_to_state_pool.py \
        --input data/selfplay/gpu/games_square8_2p_*.jsonl \
        --output data/eval_pools/square8/pool_v1.jsonl \
        --sample-moves 20,40,60,80 \
        --max-states 500
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models import BoardType, GameState, GameStatus, Move, MoveType, Position
from app.rules.default_engine import DefaultRulesEngine
from app.training.generate_data import create_initial_state


def parse_position(pos_data: Optional[Dict[str, Any]]) -> Optional[Position]:
    """Convert position dict to Position object."""
    if pos_data is None:
        return None
    return Position(x=pos_data["x"], y=pos_data["y"], z=pos_data.get("z"))


def parse_move(move_data: Dict[str, Any], move_idx: int = 0) -> Move:
    """Convert move dict to Move object."""
    from datetime import datetime, timezone
    move_type = MoveType(move_data["type"])
    return Move(
        id=f"move_{move_idx}",
        type=move_type,
        player=move_data["player"],
        from_pos=parse_position(move_data.get("from")),
        to=parse_position(move_data.get("to")),
        timestamp=datetime.now(timezone.utc),
        think_time=0,
        move_number=move_idx,
    )


def extract_states_from_game(
    game: Dict[str, Any],
    sample_moves: List[int],
    engine: DefaultRulesEngine,
) -> List[Dict[str, Any]]:
    """Extract states at specified move numbers from a game.

    Returns list of state snapshots in eval pool format.
    """
    states = []

    # Get board type
    board_type_str = game.get("board_type", "square8")
    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
        "hex8": BoardType.HEX8,
    }
    board_type = board_type_map.get(board_type_str, BoardType.SQUARE8)
    num_players = game.get("num_players", 2)

    # Initialize state
    if "initial_state" in game and game["initial_state"]:
        state = GameState(**game["initial_state"])
    else:
        state = create_initial_state(board_type, num_players)

    moves = game.get("moves", [])
    game_id = game.get("game_id", "unknown")

    for move_num, move_data in enumerate(moves):
        # Check if we should sample this state
        if move_num in sample_moves:
            # Export state snapshot
            state_snapshot = {
                "id": f"{game_id}_move{move_num}",
                "board_type": board_type.value,
                "num_players": num_players,
                "move_number": move_num,
                "state": state.model_dump(),
                "source_game": game_id,
            }
            states.append(state_snapshot)

        # Apply move to advance state
        try:
            move = parse_move(move_data, move_num)
            state = engine.apply_move(state, move)
        except Exception as e:
            # Stop if move application fails
            print(f"    Move {move_num} failed: {e}")
            break

        # Stop if game is over
        if state.game_status != GameStatus.ACTIVE:
            break

    return states


def main():
    parser = argparse.ArgumentParser(
        description="Convert selfplay JSONL to eval state pool format"
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Input JSONL files (glob patterns supported)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output state pool JSONL file",
    )
    parser.add_argument(
        "--sample-moves",
        default="20,40,60,80",
        help="Comma-separated move numbers to sample states at",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=500,
        help="Maximum number of states to export",
    )
    parser.add_argument(
        "--min-game-length",
        type=int,
        default=50,
        help="Minimum game length to include",
    )

    args = parser.parse_args()

    # Parse sample moves
    sample_moves = [int(m.strip()) for m in args.sample_moves.split(",")]
    print(f"Sampling states at moves: {sample_moves}")

    # Expand input globs
    input_files = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        elif os.path.exists(pattern):
            input_files.append(pattern)

    if not input_files:
        print(f"No input files found matching: {args.input}")
        return 1

    print(f"Processing {len(input_files)} input files")

    engine = DefaultRulesEngine()
    all_states = []
    games_processed = 0

    for input_file in input_files:
        print(f"  Reading {input_file}...")
        try:
            with open(input_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        game = json.loads(line)

                        # Filter by minimum game length
                        if game.get("move_count", 0) < args.min_game_length:
                            continue

                        states = extract_states_from_game(game, sample_moves, engine)
                        all_states.extend(states)
                        games_processed += 1

                        if len(all_states) >= args.max_states:
                            break

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"    Warning: Error processing game: {e}")
                        continue

        except Exception as e:
            print(f"  Error reading {input_file}: {e}")
            continue

        if len(all_states) >= args.max_states:
            break

    # Truncate to max_states
    all_states = all_states[:args.max_states]

    # Write output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        for state in all_states:
            f.write(json.dumps(state) + "\n")

    print(f"\nExtracted {len(all_states)} states from {games_processed} games")
    print(f"Wrote to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
