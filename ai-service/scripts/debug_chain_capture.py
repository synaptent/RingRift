#!/usr/bin/env python3
"""Debug chain capture detection divergence.

This script traces the exact state at the point where selfplay and replay
diverge in chain capture detection.
"""

import json
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.game_engine import GameEngine
from app.models import BoardType, GameStatus, Move, MoveType, Position
from app.rules.default_engine import DefaultRulesEngine
from app.rules.capture_chain import enumerate_capture_moves_py
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


def main():
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "data/selfplay/hex8_fixed/test_after_size_fix.jsonl"
    game_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # Load the specified game
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i == game_idx:
                game_data = json.loads(line)
                break

    print(f"Game ID: {game_data['game_id']}")
    print(f"Board type: {game_data['board_type']}")
    print(f"Total moves: {len(game_data['moves'])}")
    print()

    # Initialize engine
    engine = DefaultRulesEngine()
    state = create_initial_state(
        board_type=BoardType.HEX8,
        num_players=2,
    )

    # Find the first capture move that's followed by no_line_action
    for i, move_data in enumerate(game_data["moves"]):
        move = parse_move(move_data)

        if move.type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT):
            # Check next move
            next_move = game_data["moves"][i + 1] if i + 1 < len(game_data["moves"]) else None
            
            if next_move and next_move["type"] == "no_line_action":
                print(f"=== Move {i}: {move.type.value} followed by no_line_action ===")
                print(f"  From: {move.from_pos}")
                print(f"  Target: {move.capture_target}")
                print(f"  Landing: {move.to}")
                print()

                # Show state BEFORE applying the capture
                print("Board state BEFORE capture:")
                print(f"  Phase: {state.current_phase.value}")
                print(f"  Player: {state.current_player}")
                print(f"  chain_capture_state: {state.chain_capture_state}")
                print(f"  Stacks ({len(state.board.stacks)}):")
                for key, stack in sorted(state.board.stacks.items()):
                    print(f"    {key}: player={stack.controlling_player}, height={stack.stack_height}, rings={stack.rings}")
                print()

                # Apply the capture
                state = engine.apply_move(state, move, trace_mode=True)

                # Show state AFTER applying the capture
                print("Board state AFTER capture:")
                print(f"  Phase: {state.current_phase.value}")
                print(f"  Player: {state.current_player}")
                print(f"  chain_capture_state: {state.chain_capture_state}")
                if state.chain_capture_state:
                    print(f"    current_position: {state.chain_capture_state.current_position}")
                    print(f"    visited_positions: {state.chain_capture_state.visited_positions}")
                print()

                # Check what captures are available
                print("Enumerating chain captures...")
                if state.chain_capture_state:
                    attacker_pos = state.chain_capture_state.current_position
                    captures = enumerate_capture_moves_py(
                        state,
                        state.current_player,
                        attacker_pos,
                        kind="continuation",
                    )
                    print(f"  Raw captures from enumerate_capture_moves_py: {len(captures)}")
                    for cap in captures:
                        from_str = f"({cap.from_pos.x},{cap.from_pos.y},{cap.from_pos.z})" if cap.from_pos else "None"
                        to_str = f"({cap.to.x},{cap.to.y},{cap.to.z})" if cap.to else "None"
                        ct_str = f"({cap.capture_target.x},{cap.capture_target.y},{cap.capture_target.z})" if cap.capture_target else "None"
                        landing_key = cap.to.to_key() if cap.to else "None"
                        in_visited = landing_key in state.chain_capture_state.visited_positions if state.chain_capture_state else False
                        print(f"    {cap.type.value}: from={from_str} target={ct_str} landing={to_str} (key={landing_key}, in_visited={in_visited})")

                    # Now check what GameEngine._get_capture_moves returns
                    ge_captures = GameEngine._get_capture_moves(state, state.current_player)
                    print(f"  Filtered captures from _get_capture_moves: {len(ge_captures)}")
                    for cap in ge_captures:
                        from_str = f"({cap.from_pos.x},{cap.from_pos.y},{cap.from_pos.z})" if cap.from_pos else "None"
                        to_str = f"({cap.to.x},{cap.to.y},{cap.to.z})" if cap.to else "None"
                        ct_str = f"({cap.capture_target.x},{cap.capture_target.y},{cap.capture_target.z})" if cap.capture_target else "None"
                        print(f"    {cap.type.value}: from={from_str} target={ct_str} landing={to_str}")
                print()

                # What was recorded as the NEXT move?
                print(f"Next recorded move (move {i+1}): {next_move['type']}")
                print("  ERROR: Recording shows no_line_action, but replay finds chain captures!")
                print()

                # Check what valid moves the engine sees NOW
                valid_moves = engine.get_valid_moves(state, state.current_player)
                print(f"  Valid moves according to engine: {len(valid_moves)}")
                valid_types = set(vm.type.value for vm in valid_moves)
                print(f"  Types: {sorted(valid_types)}")

                break

        # Apply the move
        state = engine.apply_move(state, move, trace_mode=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
