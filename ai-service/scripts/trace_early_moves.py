#!/usr/bin/env python
"""Trace early moves to find divergence."""
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
    stacks = {}
    for y in range(8):
        for x in range(8):
            pos = Position(x=x, y=y)
            stack = BoardManager.get_stack(pos, state.board)
            if stack and stack.stack_height > 0:
                stacks[(x, y)] = (stack.controlling_player, stack.stack_height)
    return stacks


def print_stacks(stacks, label):
    print(f"  {label}:")
    for (x, y), (owner, height) in sorted(stacks.items()):
        print(f"    ({x},{y}): P{owner}h{height}")


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

    # Run GPU game and export
    print("Running GPU game...")
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    for step in range(100):
        if runner.state.move_count[0].item() >= 60:
            break
        runner._step_games([{}])

    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)

    # Print first 10 moves
    print("\nGPU exported moves 0-9:")
    for i, m in enumerate(game_dict['moves'][:10]):
        from_str = f"({m['from']['x']},{m['from']['y']})" if m.get('from') else "None"
        to_str = f"({m['to']['x']},{m['to']['y']})" if m.get('to') else "None"
        print(f"  {i}: [{m.get('phase','?'):20}] {m['type']:25} player={m['player']} from={from_str:10} to={to_str}")

    # Replay CPU with detailed output
    print("\n\nCPU Replay with trace:")
    cpu_state = create_initial_state(BoardType.SQUARE8, num_players=2)

    for i, m in enumerate(game_dict['moves'][:10]):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        print(f"\n--- Move {i}: {move_type_str} ---")
        print(f"  GPU phase={gpu_phase} player={gpu_player}")

        if move_type_str in GPU_BOOKKEEPING_MOVES:
            print(f"  (bookkeeping, skipping)")
            cpu_state = advance_cpu_through_phases(cpu_state, gpu_phase, gpu_player)
            continue

        move_type = MoveType(move_type_str)
        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

        print(f"  GPU from={from_pos} to={to_pos}")
        print(f"  CPU before: phase={cpu_state.current_phase.value} player={cpu_state.current_player}")

        cpu_state = advance_cpu_through_phases(cpu_state, gpu_phase, gpu_player)
        print(f"  CPU after advance: phase={cpu_state.current_phase.value} player={cpu_state.current_player}")

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
            print(f"  MATCHED: {matched.type.value} from={matched.from_pos} to={matched.to}")
            cpu_state = GameEngine.apply_move(cpu_state, matched)
            print_stacks(get_cpu_stacks(cpu_state), "CPU stacks after")
        else:
            print(f"  NO MATCH FOUND!")
            print(f"  Valid moves of type {move_type.value}:")
            for v in valid:
                if v.type == move_type:
                    print(f"    from={v.from_pos} to={v.to}")


if __name__ == '__main__':
    main()
