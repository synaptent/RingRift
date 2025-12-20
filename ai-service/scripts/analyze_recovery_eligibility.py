#!/usr/bin/env python3
"""
Analyze recovery eligibility in games with forced elimination.

This script replays games that contain forced_elimination moves and analyzes
the game state to check if players meet the recovery eligibility conditions
per RR-CANON-R110 (3 conditions; eligibility is independent of rings in hand).
"""

import json
from pathlib import Path
from app.models import GameState, MoveType
from app.game_engine import GameEngine
from app.rules.core import is_eligible_for_recovery, count_buried_rings, player_has_markers, player_controls_any_stack
from app.training.generate_data import create_initial_state


def analyze_recovery_eligibility(state: GameState, player: int) -> dict:
    """Analyze recovery eligibility inputs (RR-CANON-R110)."""
    board = state.board
    player_state = next((p for p in state.players if p.player_number == player), None)

    rings_in_hand = player_state.rings_in_hand if player_state else 0
    controls_stacks = player_controls_any_stack(board, player)
    has_markers = player_has_markers(board, player)
    buried_rings = count_buried_rings(board, player)
    is_eligible = is_eligible_for_recovery(state, player)

    # Find the specific stacks with buried rings
    # Note: stack.rings is a list of player numbers (ints), not Ring objects
    # Note: board.stacks keys can be Position objects or strings like "x,y"
    buried_ring_locations = []
    for pos, stack in board.stacks.items():
        rings = stack.rings
        if len(rings) > 1:
            # Check if any non-top ring belongs to player
            for i, ring_owner in enumerate(rings[:-1]):  # Exclude top ring
                if ring_owner == player:
                    # Handle both Position objects and string keys
                    if isinstance(pos, str):
                        pos_str = pos
                    else:
                        pos_str = f"{pos.x},{pos.y}"
                    buried_ring_locations.append({
                        'position': pos_str,
                        'depth': i,
                        'controller': stack.controlling_player
                    })

    return {
        'rings_in_hand': rings_in_hand,
        'controls_stacks': controls_stacks,
        'has_markers': has_markers,
        'buried_rings': buried_rings,
        'buried_ring_locations': buried_ring_locations,
        'is_eligible': is_eligible,
    }


def get_bookkeeping_move(state, player):
    """Get bookkeeping move if one is needed."""
    req = GameEngine.get_phase_requirement(state, player)
    if req is not None:
        return GameEngine.synthesize_bookkeeping_move(req, state)
    return None


def get_move_position(move, attr_name):
    """Safely get position from move object."""
    # Move object may use different attribute names
    pos = getattr(move, attr_name, None)
    if pos is None and attr_name == 'from_position':
        # Try alternate name
        pos = getattr(move, 'from_pos', None)
    return pos


def main():
    # Load game
    games_file = Path("data/selfplay/test_recovery/games.jsonl")
    with open(games_file, 'r') as f:
        games = [json.loads(line.strip()) for line in f]

    # Find games with forced elimination
    fe_games = [g for g in games if any(m.get('type') == 'forced_elimination' for m in g.get('moves', []))]
    print(f"Found {len(fe_games)} games with forced elimination")

    # For Game 0, we need to replay properly using valid moves from engine
    game = fe_games[0]
    moves_json = game.get('moves', [])

    # Create initial state and replay
    state = create_initial_state(board_type="square8", num_players=2)

    # Find FE indices
    fe_indices = [i for i, m in enumerate(moves_json) if m.get('type') == 'forced_elimination']
    print(f"Forced elimination at indices: {fe_indices}")

    # Track when we want to analyze (after no_movement_action for FE player)
    analyze_points = {}
    for fe_idx in fe_indices:
        fe_player = moves_json[fe_idx].get('player')
        for j in range(fe_idx + 1, min(fe_idx + 10, len(moves_json))):
            m = moves_json[j]
            if m.get('type') == 'no_movement_action' and m.get('player') == fe_player:
                analyze_points[j] = {'fe_idx': fe_idx, 'fe_player': fe_player}
                break

    print(f"Will analyze at no_movement_action points: {list(analyze_points.keys())}")

    # First, let's inspect Move object to understand its attributes
    valid_moves = GameEngine.get_valid_moves(state, state.current_player)
    if valid_moves:
        sample_move = valid_moves[0]
        print(f"\nMove object attributes: {[a for a in dir(sample_move) if not a.startswith('_')]}")
        print(f"Sample move: type={sample_move.type}, to={sample_move.to}")
        # Check which attribute holds from position
        for attr in ['from_position', 'from_pos', 'source']:
            if hasattr(sample_move, attr):
                print(f"  {attr}={getattr(sample_move, attr)}")

    # Replay the game
    move_idx = 0
    applied = 0
    while move_idx < len(moves_json):
        m_json = moves_json[move_idx]
        move_type_str = m_json.get('type')
        move_player = m_json.get('player', state.current_player)

        # Get valid moves
        valid_moves = GameEngine.get_valid_moves(state, state.current_player)
        bookkeeping = get_bookkeeping_move(state, state.current_player)

        # Try to find a matching move
        matched_move = None

        try:
            move_type = MoveType(move_type_str)
        except ValueError:
            print(f"Unknown move type at {move_idx}: {move_type_str}")
            break

        # Check bookkeeping first
        if bookkeeping and bookkeeping.type == move_type:
            matched_move = bookkeeping
        else:
            # Filter by type
            candidates = [vm for vm in valid_moves if vm.type == move_type]

            # Filter by position
            if m_json.get('to'):
                to_x, to_y = m_json['to']['x'], m_json['to']['y']
                candidates = [vm for vm in candidates if vm.to and vm.to.x == to_x and vm.to.y == to_y]

            if m_json.get('from'):
                from_x, from_y = m_json['from']['x'], m_json['from']['y']
                # Use getattr to handle potential attribute name differences
                new_candidates = []
                for vm in candidates:
                    from_pos = getattr(vm, 'from_position', None) or getattr(vm, 'from_pos', None)
                    if from_pos and from_pos.x == from_x and from_pos.y == from_y:
                        new_candidates.append(vm)
                candidates = new_candidates

            if candidates:
                matched_move = candidates[0]

        if not matched_move:
            print(f"No match for move {move_idx}: {move_type_str} from {m_json.get('from')} to {m_json.get('to')}")
            print(f"  Current phase: {state.current_phase}, player: {state.current_player}")
            # Show valid move types
            move_types = set(vm.type.value for vm in valid_moves)
            print(f"  Valid move types: {move_types}")
            print(f"  Bookkeeping: {bookkeeping.type.value if bookkeeping else None}")
            break

        # Track markers BEFORE and AFTER the move
        markers_before = len(state.board.markers)

        # Analyze BEFORE applying the move if this is an analysis point
        if move_idx in analyze_points:
            info = analyze_points[move_idx]
            print(f"\n{'='*60}")
            print(f"ANALYSIS at move {move_idx} (no_movement_action)")
            print(f"This is after forced_elimination at move {info['fe_idx']} by player {info['fe_player']}")
            print(f"Current phase: {state.current_phase}, current player: {state.current_player}")

            for player in [1, 2]:
                analysis = analyze_recovery_eligibility(state, player)
                print(f"\n  Player {player} recovery eligibility:")
                print(f"    Rings in hand: {analysis['rings_in_hand']} (does not affect eligibility)")
                print(f"    Controls stacks: {analysis['controls_stacks']} (need False)")
                print(f"    Has markers: {analysis['has_markers']} (need True)")
                print(f"    Buried rings: {analysis['buried_rings']} (need >= 1)")
                print(f"    -> IS ELIGIBLE: {analysis['is_eligible']}")

                if analysis['buried_ring_locations']:
                    print(f"    Buried ring locations:")
                    for loc in analysis['buried_ring_locations']:
                        print(f"      - At {loc['position']} (depth {loc['depth']}, controlled by P{loc['controller']})")

            # Show all non-empty stacks
            # Note: stack.rings is already a list of player numbers (ints)
            # Note: pos can be string or Position object
            print(f"\n  Board state (all stacks):")
            for pos, stack in state.board.stacks.items():
                if stack.stack_height > 0:
                    ring_owners = list(stack.rings)  # Already ints
                    if isinstance(pos, str):
                        pos_str = pos
                    else:
                        pos_str = f"({pos.x},{pos.y})"
                    print(f"    {pos_str}: height={stack.stack_height}, controller=P{stack.controlling_player}, rings={ring_owners}")

            # Debug: Show markers on the board
            print(f"\n  Markers on board: {len(state.board.markers)}")
            if state.board.markers:
                for pos, marker in state.board.markers.items():
                    if isinstance(pos, str):
                        pos_str = pos
                    else:
                        pos_str = f"({pos.x},{pos.y})"
                    print(f"    {pos_str}: player={marker.player}")
            else:
                print("    (no markers found - this is suspicious!)")

        # Apply the move
        try:
            state = GameEngine.apply_move(state, matched_move)
            applied += 1

            # Track marker changes
            markers_after = len(state.board.markers)
            if markers_before != markers_after:
                print(f"  Move {move_idx}: {move_type_str} by P{move_player}, markers {markers_before} -> {markers_after}")

            move_idx += 1
        except Exception as e:
            print(f"Error applying move {move_idx}: {e}")
            import traceback
            traceback.print_exc()
            break

    print(f"\nReplayed {applied}/{len(moves_json)} moves")


if __name__ == "__main__":
    main()
