#!/usr/bin/env python
"""Test GPU to CPU import parity with multiple seeds."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models import BoardType, MoveType, Position, GamePhase
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)

# GPU bookkeeping moves that CPU phase machine handles implicitly
# Note: choose_line_option and choose_territory_option are NOT bookkeeping - they are
# real actions that need to be matched and applied. process_line is automatic on CPU
# when lines are detected, but territory decisions remain explicit canonical moves.
GPU_BOOKKEEPING_MOVES = {
    'skip_capture',
    'skip_recovery',
    'no_placement_action',
    'no_movement_action',
    'no_line_action',
    'no_territory_action',
    'process_line',  # Line processing is automatic on CPU
}

# Phase ordering for comparison (lower index = earlier in turn)
PHASE_ORDER = [
    'ring_placement',
    'movement',
    'capture',
    'chain_capture',
    'line_processing',
    'territory_processing',
    'forced_elimination',
    'game_over',
]

# Map move types to the phase they should occur in
MOVE_TYPE_TO_PHASE = {
    'place_ring': 'ring_placement',
    'skip_placement': 'ring_placement',
    'move_stack': 'movement',
    'recovery_slide': 'movement',
    'overtaking_capture': 'capture',
    'continue_capture_segment': 'chain_capture',
    'choose_line_option': 'line_processing',
    'process_line': 'line_processing',
    'choose_territory_option': 'territory_processing',
    'eliminate_rings_from_stack': 'forced_elimination',
    'forced_elimination': 'forced_elimination',
}


def get_phase_index(phase_str: str) -> int:
    """Get the index of a phase in the turn order."""
    try:
        return PHASE_ORDER.index(phase_str)
    except ValueError:
        return 0


def is_phase_later(phase_a: str, phase_b: str) -> bool:
    """Return True if phase_a comes after phase_b in turn order."""
    return get_phase_index(phase_a) > get_phase_index(phase_b)


def advance_cpu_through_phases(state, target_phase_str: str, target_player: int):
    """Advance CPU state through bookkeeping phases until reaching target phase/player.

    GPU and CPU have different phase structures. When GPU exports a move for a certain
    phase/player, CPU might still be in an earlier phase. This function advances CPU
    by applying bookkeeping moves until it reaches the target phase/player.
    """
    from app.board_manager import BoardManager
    max_iterations = 20  # Increased to handle complex phase chains

    for _ in range(max_iterations):
        current_phase = state.current_phase.value
        current_player = state.current_player

        # If we're at the exact target phase/player, we're done
        if current_phase == target_phase_str and current_player == target_player:
            return state

        # If target player is different and we're at ring_placement for target, done
        if current_phase == 'ring_placement' and current_player == target_player:
            if target_phase_str == 'ring_placement':
                return state

        # Try to advance via bookkeeping first
        req = GameEngine.get_phase_requirement(state, state.current_player)
        if req:
            synth = GameEngine.synthesize_bookkeeping_move(req, state)
            state = GameEngine.apply_move(state, synth)
            continue

        # Determine if we need to skip current phase
        target_is_different_player = target_player != current_player
        target_is_later_phase = is_phase_later(target_phase_str, current_phase)
        should_skip = target_is_different_player or target_is_later_phase

        if not should_skip:
            # Target is same player and same or earlier phase - stop here
            break

        valid = GameEngine.get_valid_moves(state, state.current_player)

        # Handle movement phase - skip if target is later
        if current_phase == 'movement':
            # NO_MOVEMENT_ACTION is used when player has no movement options
            skip_moves = [v for v in valid if v.type == MoveType.NO_MOVEMENT_ACTION]
            if skip_moves:
                state = GameEngine.apply_move(state, skip_moves[0])
                continue
            # If no skip available but we need to advance, force phase change
            if should_skip and not any(v.type in (MoveType.MOVE_STACK, MoveType.RECOVERY_SLIDE) for v in valid):
                state.current_phase = GamePhase.CAPTURE
                continue

        # Handle capture/chain_capture phase
        if current_phase in ('capture', 'chain_capture'):
            skip_moves = [v for v in valid if v.type == MoveType.SKIP_CAPTURE]
            capture_moves = [v for v in valid if v.type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT)]

            # Check if all captures are self-captures (GPU skips these)
            all_self_captures = True
            for c in capture_moves:
                if c.capture_target:
                    target_stack = BoardManager.get_stack(c.capture_target, state.board)
                    if target_stack and target_stack.controlling_player != current_player:
                        all_self_captures = False
                        break

            if should_skip or all_self_captures:
                if skip_moves:
                    state = GameEngine.apply_move(state, skip_moves[0])
                    continue
                elif current_phase == 'chain_capture':
                    state.current_phase = GamePhase.LINE_PROCESSING
                    continue

        # Handle line_processing phase
        if current_phase == 'line_processing':
            # Check if there are line processing moves available
            line_moves = [v for v in valid if v.type in (MoveType.CHOOSE_LINE_OPTION, MoveType.PROCESS_LINE)]
            if not line_moves and should_skip:
                # Only force transition if no line processing moves exist
                state.current_phase = GamePhase.TERRITORY_PROCESSING
                continue

        # Handle territory_processing phase
        if current_phase == 'territory_processing':
            skip_moves = [v for v in valid if v.type == MoveType.SKIP_TERRITORY_PROCESSING]
            if should_skip:
                if skip_moves:
                    state = GameEngine.apply_move(state, skip_moves[0])
                    continue
                # Force transition to next player's turn
                state.current_phase = GamePhase.RING_PLACEMENT
                state.current_player = (current_player % 2) + 1
                continue

        # Handle forced_elimination phase
        if current_phase == 'forced_elimination':
            if should_skip:
                # Force transition to next player
                state.current_phase = GamePhase.RING_PLACEMENT
                state.current_player = (current_player % 2) + 1
                continue

        # No advancement possible
        break

    return state


def find_matching_move(valid_moves, move_type, from_pos, to_pos):
    """Find a matching move from valid moves list."""
    for v in valid_moves:
        if v.type != move_type:
            continue
        v_to = v.to.to_key() if v.to else None
        m_to = to_pos.to_key() if to_pos else None
        if move_type == MoveType.PLACE_RING:
            if v_to == m_to:
                return v
        elif move_type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT):
            v_from = v.from_pos.to_key() if v.from_pos else None
            m_from = from_pos.to_key() if from_pos else None
            if v_from == m_from and v_to == m_to:
                return v
        elif move_type == MoveType.SKIP_PLACEMENT:
            return v
        elif move_type == MoveType.RECOVERY_SLIDE:
            v_from = v.from_pos.to_key() if v.from_pos else None
            m_from = from_pos.to_key() if from_pos else None
            if v_from == m_from and v_to == m_to:
                return v
        elif move_type in (MoveType.CHOOSE_LINE_OPTION, MoveType.PROCESS_LINE):
            return v
        elif move_type == MoveType.CHOOSE_TERRITORY_OPTION:
            return v
        elif move_type == MoveType.ELIMINATE_RINGS_FROM_STACK:
            if v_to == m_to:
                return v
        else:
            v_from = v.from_pos.to_key() if v.from_pos else None
            m_from = from_pos.to_key() if from_pos else None
            if v_from == m_from and v_to == m_to:
                return v
    return None


def capture_gpu_state(runner, move_idx: int) -> dict:
    """Capture GPU state snapshot for debugging."""
    state = {}
    for y in range(8):
        for x in range(8):
            owner = runner.state.stack_owner[0, y, x].item()
            height = runner.state.stack_height[0, y, x].item()
            if height > 0:
                cap = runner.state.cap_height[0, y, x].item()
                ring_under = runner.state.ring_under_cap[0, y, x].item()
                state[(y, x)] = {'owner': owner, 'height': height, 'cap': cap, 'ring_under': ring_under}
    return state


def test_seed(seed: int, debug: bool = False) -> tuple[int, int, int, int, list]:
    """Test GPU to CPU parity for a given seed."""
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    # Capture GPU state at each move for debugging
    gpu_states = {}
    for step in range(100):
        move_count = int(runner.state.move_count[0].item())
        if debug and move_count not in gpu_states:
            gpu_states[move_count] = capture_gpu_state(runner, move_count)
        game_status = runner.state.game_status[0].item()
        if game_status != 0:
            break
        if move_count >= 60:
            break
        runner._step_games([{}])

    total_moves = int(runner.state.move_count[0].item())
    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)

    if debug:
        # Print GPU final state for key positions
        print(f"\n=== GPU Final State (seed {seed}) ===")
        for y in range(8):
            for x in range(8):
                owner = runner.state.stack_owner[0, y, x].item()
                height = runner.state.stack_height[0, y, x].item()
                if height > 0:
                    cap = runner.state.cap_height[0, y, x].item()
                    ring_under = runner.state.ring_under_cap[0, y, x].item()
                    print(f"  ({y},{x}): owner={owner}, height={height}, cap={cap}, ring_under={ring_under}")
    exported_moves = len(game_dict['moves'])

    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state
    errors = []
    skipped = 0

    for i, m in enumerate(game_dict['moves']):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        if debug and move_type_str not in GPU_BOOKKEEPING_MOVES:
            # Print every non-bookkeeping move
            from_p = Position(**m['from']) if 'from' in m and m['from'] else None
            to_p = Position(**m['to']) if 'to' in m and m['to'] else None
            print(f"Move {i}: {move_type_str} from={from_p.to_key() if from_p else None} to={to_p.to_key() if to_p else None} (player={gpu_player})")
            # Check key positions for divergence tracking (2,4 is problematic position)
            key_pos_24 = state.board.stacks.get('2,4')
            key_pos_61 = state.board.stacks.get('6,1')
            if key_pos_24:
                print(f"  >> 2,4 before: owner={key_pos_24.controlling_player}, h={key_pos_24.stack_height}, cap={key_pos_24.cap_height}")
            if key_pos_61:
                print(f"  >> 6,1 before: owner={key_pos_61.controlling_player}, h={key_pos_61.stack_height}, cap={key_pos_61.cap_height}")

        # Skip pure GPU bookkeeping moves that don't affect game state
        if move_type_str in GPU_BOOKKEEPING_MOVES:
            skipped += 1
            # Bookkeeping moves mean "we're done with this phase, advance to next"
            target_phase = gpu_phase
            target_player = gpu_player
            if move_type_str == 'no_line_action' and gpu_phase == 'line_processing':
                target_phase = 'territory_processing'
            elif move_type_str == 'no_territory_action' and gpu_phase == 'territory_processing':
                target_phase = 'ring_placement'
                target_player = (gpu_player % 2) + 1
            elif move_type_str == 'skip_capture' and gpu_phase in ('capture', 'chain_capture'):
                target_phase = 'line_processing'
            elif move_type_str == 'skip_recovery' and gpu_phase == 'movement':
                target_phase = 'capture'
            state = advance_cpu_through_phases(state, target_phase, target_player)
            continue

        move_type = MoveType(move_type_str)
        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

        # Infer correct phase from move type (GPU phase labels can be wrong)
        inferred_phase = MOVE_TYPE_TO_PHASE.get(move_type_str, gpu_phase)

        # First try: check if move is valid in current state (without advancing)
        valid = GameEngine.get_valid_moves(state, state.current_player)
        matched = find_matching_move(valid, move_type, from_pos, to_pos)

        # Second try: advance to inferred phase based on move type
        if not matched:
            state = advance_cpu_through_phases(state, inferred_phase, gpu_player)
            valid = GameEngine.get_valid_moves(state, state.current_player)
            matched = find_matching_move(valid, move_type, from_pos, to_pos)

        # Third try: advance to GPU's stated phase (might be different player's turn)
        if not matched and gpu_phase != inferred_phase:
            state = advance_cpu_through_phases(state, gpu_phase, gpu_player)
            valid = GameEngine.get_valid_moves(state, state.current_player)
            matched = find_matching_move(valid, move_type, from_pos, to_pos)

        # Fourth try: check if we need to advance through bookkeeping then retry
        if not matched:
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)
                valid = GameEngine.get_valid_moves(state, state.current_player)
                matched = find_matching_move(valid, move_type, from_pos, to_pos)

        if matched:
            state = GameEngine.apply_move(state, matched)
        else:
            # Record error - move could not be matched
            if debug:
                print(f"  Move {i}: {m['type']} (GPU phase={gpu_phase}, player={gpu_player})")
                print(f"    CPU: phase={state.current_phase.value}, player={state.current_player}")
                valid_positions = [(v.type.value, v.to.to_key() if v.to else None) for v in valid[:15]]
                print(f"    Valid moves (first 15): {valid_positions}")
                if from_pos:
                    print(f"    GPU move: from={from_pos.to_key()}, to={to_pos.to_key() if to_pos else None}")
                # Check stack at target position
                if to_pos:
                    stack = state.board.stacks.get(to_pos.to_key())
                    if stack:
                        print(f"    Stack at target: height={stack.stack_height}, owner={stack.controlling_player}, cap={stack.cap_height}")
                    else:
                        print(f"    No stack at target")
                # For placement moves, check all stacks owned by current player
                if move_type_str == 'place_ring':
                    my_stacks = [(k, s.stack_height, s.cap_height) for k, s in state.board.stacks.items()
                                 if s.controlling_player == state.current_player]
                    print(f"    Player {state.current_player} stacks: {my_stacks[:10]}")
                    print(f"    Rings in hand: {[p.rings_in_hand for p in state.players]}")
                # For eliminate_rings_from_stack, show GPU state at that point
                if move_type_str == 'eliminate_rings_from_stack' and to_pos:
                    # GPU coordinates: y = to_pos.y, x = to_pos.x
                    gpu_y, gpu_x = to_pos.y, to_pos.x
                    if gpu_states and i in gpu_states:
                        gpu_state_at_move = gpu_states.get(i, {})
                        if (gpu_y, gpu_x) in gpu_state_at_move:
                            print(f"    GPU state at move {i} for ({gpu_y},{gpu_x}): {gpu_state_at_move[(gpu_y, gpu_x)]}")
            errors.append((i, m['type'], state.current_phase.value))

    return total_moves, exported_moves, skipped, len(errors), errors[:3]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test GPU-CPU parity')
    parser.add_argument('--seeds', type=int, default=6, help='Number of seeds to test')
    parser.add_argument('--start-seed', type=int, default=42, help='Starting seed value')
    parser.add_argument('--specific-seed', type=int, help='Test a specific seed with debug output')
    parser.add_argument('--debug', action='store_true', help='Enable debug output for failures')
    args = parser.parse_args()

    # If specific seed requested, run just that one with debug
    if args.specific_seed is not None:
        moves, exported, skipped, error_count, errors = test_seed(args.specific_seed, debug=True)
        status = 'PASS' if error_count == 0 else 'FAIL'
        print(f'\nSeed {args.specific_seed}: {status} ({error_count} errors)')
        print(f'  Total moves: {moves}, Exported: {exported}, Skipped: {skipped}')
        return

    # Generate seeds: either use default list or generate random ones
    if args.seeds <= 6:
        seeds = [42, 123, 456, 789, 1000, 2024][:args.seeds]
    else:
        import random
        random.seed(args.start_seed)
        seeds = [random.randint(0, 100000) for _ in range(args.seeds)]

    results = []
    for seed in seeds:
        moves, exported, skipped, error_count, errors = test_seed(seed)
        status = 'PASS' if error_count == 0 else 'FAIL'
        results.append((seed, moves, exported, skipped, status, error_count))
        if errors:
            print(f'Seed {seed}: FAIL ({error_count} errors)')
            for i, mtype, phase in errors:
                print(f'  Move {i}: {mtype} in phase {phase}')

    print('\nSummary:')
    print('Seed  | Moves | Exported | Skipped | Status')
    print('-' * 50)
    for seed, moves, exported, skipped, status, ec in results:
        print(f'{seed:5} | {moves:5} | {exported:8} | {skipped:7} | {status}')

    passed = sum(1 for r in results if r[4] == 'PASS')
    print(f'\nPassed: {passed}/{len(results)}')


if __name__ == '__main__':
    main()
