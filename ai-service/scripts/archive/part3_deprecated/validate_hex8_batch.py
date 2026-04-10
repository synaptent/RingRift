#!/usr/bin/env python3
"""Batch validation script for hex8 games.

Validates that all games in a JSONL file replay correctly using DefaultRulesEngine.
"""

import json
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models import BoardType, GameStatus, Move, MoveType, Position
from app.rules.default_engine import DefaultRulesEngine
from app.training.initial_state import create_initial_state


def parse_move(move_data: dict) -> Move:
    """Parse a move from JSONL format."""
    import uuid
    move_type = MoveType(move_data["type"])
    player = move_data["player"]

    from_pos = None
    if "from" in move_data:
        f = move_data["from"]
        from_pos = Position(x=f["x"], y=f["y"], z=f.get("z"))

    to_pos = None
    if "to" in move_data:
        t = move_data["to"]
        to_pos = Position(x=t["x"], y=t["y"], z=t.get("z"))

    capture_target = None
    if "capture_target" in move_data:
        ct = move_data["capture_target"]
        capture_target = Position(x=ct["x"], y=ct["y"], z=ct.get("z"))

    # Parse placement_count for place_ring moves (critical for replay parity)
    placement_count = move_data.get("placement_count")

    return Move(
        id=str(uuid.uuid4()),
        type=move_type,
        player=player,
        from_pos=from_pos,
        to=to_pos,
        capture_target=capture_target,
        placement_count=placement_count,
    )


def validate_game(game_data: dict, game_idx: int, verbose: bool = False) -> tuple[bool, str]:
    """Validate a single game by replaying it.

    Returns (passed, message)
    """
    engine = DefaultRulesEngine()
    state = create_initial_state(
        board_type=BoardType.HEX8,
        num_players=2,
    )

    for move_idx, move_data in enumerate(game_data["moves"]):
        recorded_type = move_data["type"]
        recorded_player = move_data["player"]

        # Current engine state
        engine_player = state.current_player
        engine_phase = state.current_phase.value

        # Get valid moves
        valid_moves = engine.get_valid_moves(state, engine_player)
        valid_types = set(m.type.value for m in valid_moves)

        # Check phase/player match
        if recorded_player != engine_player:
            return False, f"Move {move_idx}: player mismatch (recorded={recorded_player}, engine={engine_player})"

        if recorded_type not in valid_types:
            return False, f"Move {move_idx}: type '{recorded_type}' not valid in phase '{engine_phase}' (valid: {sorted(valid_types)})"

        # Parse and apply move
        try:
            move = parse_move(move_data)
            # Use trace_mode=True to match generation semantics
            state = engine.apply_move(state, move, trace_mode=True)
        except Exception as e:
            return False, f"Move {move_idx}: apply error - {e}"

        if state.game_status != GameStatus.ACTIVE:
            break

    return True, f"{len(game_data['moves'])} moves, winner=P{game_data.get('winner', '?')}"


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_hex8_batch.py <jsonl_file> [--verbose]")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    verbose = "--verbose" in sys.argv

    print(f"Validating: {jsonl_path}")
    print()

    passed = 0
    failed = 0
    errors = []

    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            game_data = json.loads(line)
            success, message = validate_game(game_data, idx, verbose)

            if success:
                passed += 1
                print(f"Game {idx}: PASSED - {message}")
            else:
                failed += 1
                errors.append(f"Game {idx}: {message}")
                print(f"Game {idx}: FAILED - {message}")

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print()
        print("Errors:")
        for err in errors:
            print(f"  {err}")
        sys.exit(1)
    else:
        print("All games validated successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
