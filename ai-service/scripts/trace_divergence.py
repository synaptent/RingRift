#!/usr/bin/env python
"""Trace exact point of board state divergence between GPU and CPU."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models import BoardType, MoveType, Position
from app.board_manager import BoardManager
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)

GPU_BOOKKEEPING_MOVES = {
    'skip_capture', 'skip_recovery', 'no_placement_action', 'no_movement_action',
    'no_line_action', 'no_territory_action', 'process_line', 'process_territory_region',
    'choose_line_option', 'choose_territory_option',
}


def get_cpu_stacks(state):
    """Get CPU stack state as dict."""
    stacks = {}
    for y in range(8):
        for x in range(8):
            pos = Position(x=x, y=y)
            stack = BoardManager.get_stack(pos, state.board)
            if stack and stack.stack_height > 0:
                stacks[(x, y)] = (stack.controlling_player, stack.stack_height)
    return stacks


def get_gpu_stacks(runner):
    """Get GPU stack state as dict."""
    stacks = {}
    for y in range(8):
        for x in range(8):
            owner = int(runner.state.stack_owner[0, y, x].item())
            height = int(runner.state.stack_height[0, y, x].item())
            if owner > 0 and height > 0:
                stacks[(x, y)] = (owner, height)
    return stacks


def compare_stacks(cpu_stacks, gpu_stacks):
    """Compare two stack dicts, return differences."""
    all_positions = set(cpu_stacks.keys()) | set(gpu_stacks.keys())
    diffs = []
    for pos in sorted(all_positions):
        cpu_val = cpu_stacks.get(pos)
        gpu_val = gpu_stacks.get(pos)
        if cpu_val != gpu_val:
            diffs.append((pos, cpu_val, gpu_val))
    return diffs


def advance_cpu_through_phases(state, target_phase_str, target_player):
    max_iterations = 10
    for _ in range(max_iterations):
        current_phase = state.current_phase.value
        current_player = state.current_player
        if current_phase == 'ring_placement' and current_player == target_player:
            return state
        if current_phase == target_phase_str and current_player == target_player:
            return state
        req = GameEngine.get_phase_requirement(state, state.current_player)
        if req:
            synth = GameEngine.synthesize_bookkeeping_move(req, state)
            state = GameEngine.apply_move(state, synth)
        else:
            if current_phase == 'capture':
                valid = GameEngine.get_valid_moves(state, state.current_player)
                skip_moves = [v for v in valid if v.type == MoveType.SKIP_CAPTURE]
                captures = [v for v in valid if v.type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT)]
                all_self_captures = True
                for c in captures:
                    if c.capture_target:
                        target_stack = BoardManager.get_stack(c.capture_target, state.board)
                        if target_stack and target_stack.controlling_player != state.current_player:
                            all_self_captures = False
                            break
                if all_self_captures and skip_moves:
                    state = GameEngine.apply_move(state, skip_moves[0])
                    continue
            break
    return state


def main():
    seed = 456

    # Step 1: Run GPU game to completion and get export
    print("Running GPU game...")
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    for step in range(100):
        if runner.state.move_count[0].item() >= 60:
            break
        runner._step_games([{}])

    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)
    print(f"GPU game has {len(game_dict['moves'])} exported moves")

    # Step 2: Replay CPU following GPU export, compare board states at key points
    print("\nReplaying CPU and comparing states...")

    # Fresh GPU runner to compare states incrementally
    torch.manual_seed(seed)
    gpu_runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    # CPU state
    cpu_state = create_initial_state(BoardType.SQUARE8, num_players=2)

    cpu_move_idx = 0
    last_gpu_move_count = 0

    for i, m in enumerate(game_dict['moves']):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        # Step GPU to this point
        while int(gpu_runner.state.move_count[0].item()) <= i:
            gpu_runner._step_games([{}])
            if gpu_runner.state.game_status[0].item() != 0:
                break

        # Apply to CPU
        if move_type_str in GPU_BOOKKEEPING_MOVES:
            cpu_state = advance_cpu_through_phases(cpu_state, gpu_phase, gpu_player)
            continue

        move_type = MoveType(move_type_str)
        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

        cpu_state = advance_cpu_through_phases(cpu_state, gpu_phase, gpu_player)

        valid = GameEngine.get_valid_moves(cpu_state, cpu_state.current_player)
        matched = None

        for v in valid:
            if v.type != move_type:
                continue
            v_to = v.to.to_key() if v.to else None
            m_to = to_pos.to_key() if to_pos else None
            if move_type == MoveType.PLACE_RING:
                if v_to == m_to:
                    matched = v
                    break
            else:
                v_from = v.from_pos.to_key() if v.from_pos else None
                m_from = from_pos.to_key() if from_pos else None
                if v_from == m_from and v_to == m_to:
                    matched = v
                    break

        if matched:
            cpu_state = GameEngine.apply_move(cpu_state, matched)
            cpu_move_idx += 1

            # Compare stacks after each non-bookkeeping move
            cpu_stacks = get_cpu_stacks(cpu_state)
            gpu_stacks = get_gpu_stacks(gpu_runner)
            diffs = compare_stacks(cpu_stacks, gpu_stacks)

            if diffs:
                print(f"\n=== DIVERGENCE after move {i}: {move_type_str} ===")
                print(f"Move: from={from_pos} to={to_pos}")
                print(f"\nStack differences ({len(diffs)}):")
                for pos, cpu_val, gpu_val in diffs[:20]:
                    cpu_str = f"P{cpu_val[0]}h{cpu_val[1]}" if cpu_val else "empty"
                    gpu_str = f"P{gpu_val[0]}h{gpu_val[1]}" if gpu_val else "empty"
                    print(f"  ({pos[0]},{pos[1]}): CPU={cpu_str}, GPU={gpu_str}")
                return
            else:
                if i % 10 == 0:
                    print(f"  Move {i}: OK ({move_type_str})")
        else:
            print(f"\n=== MISMATCH at move {i} ===")
            print(f"GPU wants: {move_type_str} from={from_pos} to={to_pos}")
            print(f"CPU phase: {cpu_state.current_phase.value} player: {cpu_state.current_player}")
            print(f"\nCPU valid moves ({len(valid)}):")
            for v in valid[:10]:
                print(f"  {v.type.value} from={v.from_pos} to={v.to}")

            # Show stack comparison
            cpu_stacks = get_cpu_stacks(cpu_state)
            gpu_stacks = get_gpu_stacks(gpu_runner)
            diffs = compare_stacks(cpu_stacks, gpu_stacks)

            print(f"\nStack differences ({len(diffs)}):")
            for pos, cpu_val, gpu_val in diffs[:20]:
                cpu_str = f"P{cpu_val[0]}h{cpu_val[1]}" if cpu_val else "empty"
                gpu_str = f"P{gpu_val[0]}h{gpu_val[1]}" if gpu_val else "empty"
                print(f"  ({pos[0]},{pos[1]}): CPU={cpu_str}, GPU={gpu_str}")
            return

    print(f"\nCompleted {cpu_move_idx} moves without divergence")


if __name__ == '__main__':
    main()
