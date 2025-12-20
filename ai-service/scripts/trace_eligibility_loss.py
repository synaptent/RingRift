#!/usr/bin/env python3
"""
Trace what changes between when a player is eligible for recovery (e.g., in line_processing)
and their next MOVEMENT phase turn, causing them to lose eligibility.
"""

import contextlib
import json
from pathlib import Path

from app.game_engine import GameEngine
from app.models import GamePhase, GameState, MoveType
from app.rules.core import (
    count_buried_rings,
    is_eligible_for_recovery,
    player_controls_any_stack,
    player_has_markers,
)
from app.training.generate_data import create_initial_state


def is_eligible_no_rings_requirement(state: GameState, player: int) -> bool:
    """Check recovery eligibility WITHOUT the rings_in_hand == 0 requirement."""
    board = state.board
    if player_controls_any_stack(board, player):
        return False
    if not player_has_markers(board, player):
        return False
    return not count_buried_rings(board, player) < 1


def get_eligibility_details(state: GameState, player: int) -> dict:
    """Get detailed eligibility status for a player."""
    board = state.board
    player_state = next((p for p in state.players if p.player_number == player), None)

    rings_in_hand = player_state.rings_in_hand if player_state else 0
    controls_stacks = player_controls_any_stack(board, player)
    has_markers = player_has_markers(board, player)
    buried_rings = count_buried_rings(board, player)

    # Count controlled stacks
    controlled_stack_count = 0
    for stack in board.stacks.values():
        if stack.controlling_player == player and stack.stack_height > 0:
            controlled_stack_count += 1

    return {
        'rings_in_hand': rings_in_hand,
        'controls_stacks': controls_stacks,
        'controlled_stack_count': controlled_stack_count,
        'has_markers': has_markers,
        'buried_rings': buried_rings,
        'is_eligible': is_eligible_for_recovery(state, player),
        'eligible_no_rings_req': is_eligible_no_rings_requirement(state, player),
    }


def trace_game(game_file: str, game_index: int, eligible_move_idx: int, eligible_player: int):
    """
    Trace a game from an eligible state (in line/territory processing) to
    the player's next MOVEMENT phase, tracking what changes.
    """
    games_file = Path(game_file)
    games = []
    with open(games_file) as f:
        for line in f:
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    games.append(json.loads(line))

    game = games[game_index]
    moves_json = game.get('moves', [])
    board_type = game.get('board_type', 'square8')
    num_players = game.get('num_players', 2)

    state = create_initial_state(board_type=board_type, num_players=num_players)

    print(f"\n{'='*70}")
    print(f"TRACING: Game {game_index}, eligible at move {eligible_move_idx}, player {eligible_player}")
    print(f"{'='*70}")

    # Replay to just before the eligible state
    for i in range(eligible_move_idx):
        m_json = moves_json[i]
        move_type_str = m_json.get('type')

        try:
            move_type = MoveType(move_type_str)
        except ValueError:
            print(f"Unknown move type at {i}: {move_type_str}")
            return

        valid_moves = GameEngine.get_valid_moves(state, state.current_player)
        bookkeeping = None
        req = GameEngine.get_phase_requirement(state, state.current_player)
        if req is not None:
            bookkeeping = GameEngine.synthesize_bookkeeping_move(req, state)

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

    # Now at the eligible state
    print(f"\n--- AT ELIGIBLE STATE (move {eligible_move_idx}) ---")
    print(f"Phase: {state.current_phase.value}, Current player: {state.current_player}")
    details = get_eligibility_details(state, eligible_player)
    print(f"Player {eligible_player} status:")
    for k, v in details.items():
        print(f"  {k}: {v}")

    # Continue replaying, printing each step until player reaches MOVEMENT phase
    found_movement = False
    for i in range(eligible_move_idx, min(eligible_move_idx + 20, len(moves_json))):
        m_json = moves_json[i]
        move_type_str = m_json.get('type')

        try:
            move_type = MoveType(move_type_str)
        except ValueError:
            break

        # Check state BEFORE applying this move
        if state.current_player == eligible_player and state.current_phase == GamePhase.MOVEMENT:
            print(f"\n--- PLAYER {eligible_player} IN MOVEMENT PHASE (before move {i}) ---")
            details = get_eligibility_details(state, eligible_player)
            print(f"Player {eligible_player} status:")
            for k, v in details.items():
                print(f"  {k}: {v}")
            found_movement = True
            break

        valid_moves = GameEngine.get_valid_moves(state, state.current_player)
        bookkeeping = None
        req = GameEngine.get_phase_requirement(state, state.current_player)
        if req is not None:
            bookkeeping = GameEngine.synthesize_bookkeeping_move(req, state)

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
            break

        # Print move being applied
        print(f"\n  Move {i}: {move_type_str} by P{state.current_player} (phase: {state.current_phase.value})")

        old_details = get_eligibility_details(state, eligible_player)

        state = GameEngine.apply_move(state, matched_move)

        new_details = get_eligibility_details(state, eligible_player)

        # Check what changed for the eligible player
        changes = []
        for k in old_details:
            if old_details[k] != new_details[k]:
                changes.append(f"{k}: {old_details[k]} -> {new_details[k]}")

        if changes:
            print(f"    P{eligible_player} changes: {', '.join(changes)}")

        # Check after applying
        if state.current_player == eligible_player and state.current_phase == GamePhase.MOVEMENT:
            print(f"\n--- PLAYER {eligible_player} IN MOVEMENT PHASE (after move {i}) ---")
            details = get_eligibility_details(state, eligible_player)
            print(f"Player {eligible_player} status:")
            for k, v in details.items():
                print(f"  {k}: {v}")
            found_movement = True
            break

    if not found_movement:
        print(f"\n  (Player {eligible_player} did not reach MOVEMENT phase in next 20 moves)")


def main():
    # Load the analysis results
    results_file = Path("data/selfplay/actual_recovery_opportunities.json")
    with open(results_file) as f:
        results = json.load(f)

    eligible_states = results.get('eligible_but_not_movement', [])

    print(f"Found {len(eligible_states)} eligible states NOT in MOVEMENT phase")
    print("These are where player is eligible AND it's their turn, but phase is line/territory processing")

    # Trace each one
    for state_info in eligible_states[:4]:  # First 4
        trace_game(
            state_info['game_file'],
            state_info['game_index'],
            state_info['move_index'],
            state_info['player'],
        )


if __name__ == "__main__":
    main()
