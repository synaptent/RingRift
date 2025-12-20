#!/usr/bin/env python3
"""
Debug why recovery_slide moves aren't being generated when players are eligible.

This script loads a specific recovery-eligible state and checks:
1. Is player eligible for recovery?
2. What does get_expanded_recovery_moves return?
3. What does get_valid_moves return?
"""

import json
from pathlib import Path
from app.models import MoveType, GamePhase
from app.game_engine import GameEngine
from app.rules.core import (
    is_eligible_for_recovery,
    count_buried_rings,
    player_has_markers,
    player_controls_any_stack,
)
from app.rules.recovery import (
    get_recovery_moves,
    get_expanded_recovery_moves,
    enumerate_recovery_slide_targets,
    enumerate_expanded_recovery_targets,
)
from app.training.generate_data import create_initial_state


def analyze_eligible_state(game_file: str, game_index: int, move_index: int, player: int):
    """Replay to the specified state and check recovery move generation."""
    games_file = Path(game_file)
    games = []
    with open(games_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    games.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # Skip corrupt lines

    game = games[game_index]
    moves_json = game.get('moves', [])

    # Get board/player config
    board_type = game.get('board_type', 'square8')
    num_players = game.get('num_players', 2)

    # Create initial state
    state = create_initial_state(board_type=board_type, num_players=num_players)

    # Replay to the target move
    for i in range(move_index):
        m_json = moves_json[i]
        move_type_str = m_json.get('type')

        try:
            move_type = MoveType(move_type_str)
        except ValueError:
            print(f"Unknown move type at {i}: {move_type_str}")
            return

        # Get valid moves and bookkeeping
        valid_moves = GameEngine.get_valid_moves(state, state.current_player)
        bookkeeping = None
        req = GameEngine.get_phase_requirement(state, state.current_player)
        if req is not None:
            bookkeeping = GameEngine.synthesize_bookkeeping_move(req, state)

        # Match move
        matched_move = None
        if bookkeeping and bookkeeping.type == move_type:
            matched_move = bookkeeping
        else:
            candidates = [vm for vm in valid_moves if vm.type == move_type]
            if m_json.get('to'):
                to_x, to_y = m_json['to']['x'], m_json['to']['y']
                candidates = [vm for vm in candidates if vm.to and vm.to.x == to_x and vm.to.y == to_y]
            if m_json.get('from'):
                from_x, from_y = m_json['from']['x'], m_json['from']['y']
                new_candidates = []
                for vm in candidates:
                    from_pos = getattr(vm, 'from_position', None) or getattr(vm, 'from_pos', None)
                    if from_pos and from_pos.x == from_x and from_pos.y == from_y:
                        new_candidates.append(vm)
                candidates = new_candidates
            if candidates:
                matched_move = candidates[0]

        if not matched_move:
            print(f"No match at move {i}: {move_type_str}")
            return

        state = GameEngine.apply_move(state, matched_move)

    # Now we're at the target state
    print(f"\n{'='*60}")
    print(f"STATE AT MOVE {move_index} (game {game_index} in {game_file})")
    print(f"Current phase: {state.current_phase}")
    print(f"Current player: {state.current_player}")
    print(f"Target player for recovery check: {player}")
    print(f"{'='*60}")

    # Check recovery eligibility
    player_state = next((p for p in state.players if p.player_number == player), None)
    board = state.board

    print(f"\nRECOVERY ELIGIBILITY CHECK (player {player}):")
    print(f"  Rings in hand: {player_state.rings_in_hand if player_state else 'N/A'} (does not affect eligibility)")
    print(f"  Controls stacks: {player_controls_any_stack(board, player)} (need False)")
    print(f"  Has markers: {player_has_markers(board, player)} (need True)")
    print(f"  Buried rings: {count_buried_rings(board, player)} (need >= 1)")
    print(f"  -> is_eligible_for_recovery: {is_eligible_for_recovery(state, player)}")

    # Check if we're in MOVEMENT phase
    print(f"\nPHASE CHECK:")
    print(f"  Current phase: {state.current_phase}")
    print(f"  Is MOVEMENT phase: {state.current_phase == GamePhase.MOVEMENT}")

    # Check if it's the player's turn
    print(f"\nTURN CHECK:")
    print(f"  Current player: {state.current_player}")
    print(f"  Is player's turn: {state.current_player == player}")

    # Try to generate recovery moves
    print(f"\nRECOVERY MOVE GENERATION:")

    # Direct call to enumerate_recovery_slide_targets
    targets = enumerate_recovery_slide_targets(state, player)
    print(f"  enumerate_recovery_slide_targets: {len(targets)} targets")

    # Direct call to enumerate_expanded_recovery_targets
    expanded_targets = enumerate_expanded_recovery_targets(state, player)
    print(f"  enumerate_expanded_recovery_targets: {len(expanded_targets)} targets")
    for t in expanded_targets[:5]:
        print(f"    - {t.recovery_mode}: {t.from_pos} -> {t.to_pos}")

    # Call get_recovery_moves
    recovery_moves = get_recovery_moves(state, player)
    print(f"  get_recovery_moves: {len(recovery_moves)} moves")

    # Call get_expanded_recovery_moves
    expanded_moves = get_expanded_recovery_moves(state, player)
    print(f"  get_expanded_recovery_moves: {len(expanded_moves)} moves")
    for m in expanded_moves[:5]:
        print(f"    - {m.type.value}: {m.from_pos} -> {m.to}")

    # Call get_valid_moves (this is what the AI uses)
    print(f"\nGET_VALID_MOVES OUTPUT:")
    valid_moves = GameEngine.get_valid_moves(state, state.current_player)
    print(f"  Total valid moves: {len(valid_moves)}")

    # Group by type
    move_types = {}
    for vm in valid_moves:
        t = vm.type.value
        move_types[t] = move_types.get(t, 0) + 1

    print(f"  Move types:")
    for t, count in sorted(move_types.items()):
        print(f"    {t}: {count}")

    # Specifically look for recovery moves
    recovery_in_valid = [vm for vm in valid_moves if vm.type == MoveType.RECOVERY_SLIDE]
    print(f"  RECOVERY_SLIDE moves: {len(recovery_in_valid)}")

    skip_recovery = [vm for vm in valid_moves if vm.type == MoveType.SKIP_RECOVERY]
    print(f"  SKIP_RECOVERY moves: {len(skip_recovery)}")


def main():
    # Load the analysis results to find eligible states
    results_file = Path("data/selfplay/recovery_analysis_results.json")
    with open(results_file, 'r') as f:
        results = json.load(f)

    eligible_states = results.get('eligible_states', [])
    if not eligible_states:
        print("No eligible states found in analysis results!")
        return

    # Find a state that's in MOVEMENT phase (most actionable)
    movement_states = [s for s in eligible_states if s.get('phase') == 'movement']
    print(f"Found {len(movement_states)} eligible states in MOVEMENT phase")

    # Analyze multiple states to check turn alignment
    print("\nChecking turn alignment for all movement-phase eligible states:")
    for s in movement_states[:5]:  # Check first 5
        print(f"\n  Game {s['game_index']} move {s['move_index']} player {s['player']}:")
        try:
            analyze_eligible_state(
                s['game_file'],
                s['game_index'],
                s['move_index'],
                s['player']
            )
        except Exception as e:
            print(f"    Error: {e}")


if __name__ == "__main__":
    main()
