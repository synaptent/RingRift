#!/usr/bin/env python
"""Minimal test to reproduce hex8 territory processing infinite loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.game_engine import GameEngine
from app.models import BoardType, GamePhase
from app.training.initial_state import create_initial_state

# Create minimal game state that triggers the bug
game_state = create_initial_state(BoardType.HEX8, num_players=2)

# Fast-forward to a state with some moves played
for i in range(5):
    valid_moves = GameEngine.get_valid_moves(game_state, game_state.current_player)
    if valid_moves:
        move = valid_moves[0]
        print(f"Move {i}: {move.type} by P{move.player}")
        game_state = GameEngine.apply_move(game_state, move)
    else:
        req = GameEngine.get_phase_requirement(game_state, game_state.current_player)
        if req:
            move = GameEngine.synthesize_bookkeeping_move(req, game_state)
            print(f"Move {i}: {move.type} (bookkeeping) by P{move.player}")
            game_state = GameEngine.apply_move(game_state, move)

print(f"\nCurrent phase: {game_state.current_phase}")
print(f"Current player: {game_state.current_player}")
print(f"Stacks: {len(game_state.board.stacks)}")
print(f"Markers: {len(game_state.board.markers)}")
print(f"Collapsed: {len(game_state.board.collapsed_spaces)}")

# Try to get territory processing moves
print("\n=== Testing territory processing ===")
if game_state.current_phase == GamePhase.TERRITORY_PROCESSING:
    print("Currently in TERRITORY_PROCESSING phase")

    from app.board_manager import BoardManager
    regions = BoardManager.find_disconnected_regions(game_state.board, game_state.current_player)
    print(f"Found {len(regions)} disconnected regions")

    for i, region in enumerate(regions[:3]):  # Show first 3
        print(f"Region {i}: {len(region.spaces)} spaces, controlling_player={region.controlling_player}")
        if region.spaces:
            print(f"  First space: {region.spaces[0]}")
            print(f"  First space key: {region.spaces[0].to_key()}")

    # Check for infinite territory detection
    print("\n=== Checking for infinite loop in get_valid_moves ===")
    import time
    start = time.time()
    valid_moves = GameEngine.get_valid_moves(game_state, game_state.current_player)
    elapsed = time.time() - start

    print(f"get_valid_moves returned {len(valid_moves)} moves in {elapsed:.3f}s")
    if len(valid_moves) > 10:
        print("WARNING: Many moves found - possible infinite loop in territory detection")

    # Show a few moves
    for i, move in enumerate(valid_moves[:5]):
        print(f"  Move {i}: {move.type} at {move.to}")
else:
    print(f"Not in TERRITORY_PROCESSING (phase: {game_state.current_phase})")

print("\nTest complete")
