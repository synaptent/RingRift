#!/usr/bin/env python
"""Debug script to find exact state divergence between GPU and CPU."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models import BoardType, MoveType, Position
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)

GPU_BOOKKEEPING_MOVES = {
    'skip_capture', 'skip_recovery', 'no_placement_action',
    'no_movement_action', 'no_line_action', 'no_territory_action', 'process_line',
}


def get_board_state_summary(board) -> dict:
    """Extract stack positions from CPU board state."""
    stacks = {}
    for key, stack in board.stacks.items():
        stacks[key] = {
            'owner': stack.controlling_player,
            'height': stack.stack_height,
            'cap': stack.cap_height,
        }
    return stacks


def get_gpu_state_summary(state, game_idx: int, board_size: int) -> dict:
    """Extract stack positions from GPU state."""
    stacks = {}
    for y in range(board_size):
        for x in range(board_size):
            owner = int(state.stack_owner[game_idx, y, x].item())
            height = int(state.stack_height[game_idx, y, x].item())
            if height > 0:
                cap = int(state.cap_height[game_idx, y, x].item())
                key = f"{x},{y}"  # CPU uses x,y format
                stacks[key] = {'owner': owner, 'height': height, 'cap': cap}
    return stacks


def compare_states(gpu_stacks: dict, cpu_stacks: dict) -> list:
    """Compare GPU and CPU stack states, return differences."""
    diffs = []
    all_keys = set(gpu_stacks.keys()) | set(cpu_stacks.keys())

    for key in sorted(all_keys):
        gpu = gpu_stacks.get(key)
        cpu = cpu_stacks.get(key)

        if gpu and not cpu:
            diffs.append(f"  {key}: GPU has stack {gpu}, CPU has none")
        elif cpu and not gpu:
            diffs.append(f"  {key}: CPU has stack {cpu}, GPU has none")
        elif gpu != cpu:
            diffs.append(f"  {key}: GPU={gpu}, CPU={cpu}")

    return diffs


def find_matching_move(state, move_type_str, from_pos, to_pos):
    """Find a matching valid move on CPU side."""
    valid_moves = state.valid_moves
    m_to = to_pos.to_key() if to_pos else None
    move_type = getattr(MoveType, move_type_str.upper(), None)

    for v in valid_moves:
        if v.type != move_type:
            continue
        v_to = v.to_pos.to_key() if v.to_pos else None

        if move_type in (MoveType.PLACE_RING,):
            if v_to == m_to:
                return v
        elif move_type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT):
            if v_to == m_to:
                return v
        else:
            v_from = v.from_pos.to_key() if v.from_pos else None
            m_from = from_pos.to_key() if from_pos else None
            if v_from == m_from and v_to == m_to:
                return v
    return None


def debug_seed(seed: int):
    """Run detailed debug trace for a specific seed."""
    print(f"\n{'='*60}")
    print(f"DEBUGGING SEED {seed}")
    print(f"{'='*60}\n")

    # Run GPU game
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    for step in range(100):
        move_count = int(runner.state.move_count[0].item())
        if runner.state.game_status[0].item() != 0 or move_count >= 60:
            break
        runner._step_games([{}])

    # Export GPU moves
    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)
    gpu_moves = game_dict['moves']

    print(f"GPU generated {len(gpu_moves)} moves")

    # Replay on CPU with state comparison
    cpu_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    engine = GameEngine()

    # Also run GPU again step-by-step to capture intermediate states
    torch.manual_seed(seed)
    gpu_runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    gpu_move_idx = 0
    first_divergence = None

    for i, m in enumerate(gpu_moves):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        # Skip bookkeeping moves
        if move_type_str in GPU_BOOKKEEPING_MOVES:
            continue

        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

        # Advance GPU runner to this move
        while int(gpu_runner.state.move_count[0].item()) < i:
            if gpu_runner.state.game_status[0].item() != 0:
                break
            gpu_runner._step_games([{}])

        # Capture states before this move
        gpu_stacks = get_gpu_state_summary(gpu_runner.state, 0, 8)
        cpu_stacks = get_board_state_summary(cpu_state.board)

        # Compare states
        diffs = compare_states(gpu_stacks, cpu_stacks)

        # Try to find matching CPU move
        matching_move = find_matching_move(cpu_state, move_type_str, from_pos, to_pos)

        move_desc = f"{move_type_str} from={from_pos.to_key() if from_pos else None} to={to_pos.to_key() if to_pos else None}"

        if diffs:
            print(f"\nMove {i}: {move_desc} (player={gpu_player}, phase={gpu_phase})")
            print(f"  STATE DIVERGENCE DETECTED:")
            for d in diffs[:5]:  # Show first 5 differences
                print(d)
            if len(diffs) > 5:
                print(f"  ... and {len(diffs)-5} more differences")

            if first_divergence is None:
                first_divergence = i

        if matching_move is None:
            print(f"\nMove {i}: {move_desc} (player={gpu_player}, phase={gpu_phase})")
            print(f"  CPU phase: {cpu_state.current_phase}, player: {cpu_state.current_player}")
            print(f"  NO MATCHING CPU MOVE FOUND")
            print(f"  CPU valid moves ({len(cpu_state.valid_moves)}):")
            for v in cpu_state.valid_moves[:10]:
                v_from = v.from_pos.to_key() if v.from_pos else None
                v_to = v.to_pos.to_key() if v.to_pos else None
                print(f"    {v.type.name}: from={v_from} to={v_to}")
            if len(cpu_state.valid_moves) > 10:
                print(f"    ... and {len(cpu_state.valid_moves)-10} more")

            if first_divergence is None:
                first_divergence = i
            break
        else:
            # Apply move to CPU state
            cpu_state = engine.apply_move(cpu_state, matching_move)

    if first_divergence is not None:
        print(f"\n{'='*60}")
        print(f"FIRST DIVERGENCE AT MOVE {first_divergence}")
        print(f"{'='*60}")
    else:
        print(f"\nNo divergence found - all moves matched!")

    return first_divergence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Seed to debug')
    args = parser.parse_args()

    debug_seed(args.seed)


if __name__ == '__main__':
    main()
