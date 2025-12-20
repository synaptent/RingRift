#!/usr/bin/env python
"""Check board state after move 13."""
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
    'skip_capture', 'skip_recovery',
    'no_placement_action', 'no_movement_action', 'no_line_action', 'no_territory_action',
    'process_line', 'process_territory_region', 'choose_line_option', 'choose_territory_option',
}


def get_stacks(state):
    stacks = []
    for y in range(8):
        for x in range(8):
            pos = Position(x=x, y=y)
            stack = BoardManager.get_stack(pos, state.board)
            if stack and stack.stack_height > 0:
                stacks.append(f"({x},{y}):P{stack.controlling_player}h{stack.stack_height}")
    return sorted(stacks)


def main():
    seed = 42

    # Run GPU step by step
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    # Run until move 14 (which is after move 13)
    while runner.state.move_count[0].item() < 14:
        runner._step_games([{}])

    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)

    print("GPU state after move 13:")
    print(f"  Move count: {int(runner.state.move_count[0].item())}")
    print("  Stacks:")
    for y in range(8):
        for x in range(8):
            owner = int(runner.state.stack_owner[0, y, x].item())
            height = int(runner.state.stack_height[0, y, x].item())
            if owner > 0 and height > 0:
                print(f"    ({x},{y}): P{owner}h{height}")

    # Now replay CPU to move 13
    print("\n\nReplay CPU to move 13...")
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    for i, m in enumerate(game_dict['moves'][:14]):
        move_type_str = m['type']

        if move_type_str in GPU_BOOKKEEPING_MOVES:
            for _ in range(10):
                req = GameEngine.get_phase_requirement(state, state.current_player)
                if req:
                    synth = GameEngine.synthesize_bookkeeping_move(req, state)
                    state = GameEngine.apply_move(state, synth)
                else:
                    break
            continue

        move_type = MoveType(move_type_str)
        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

        for _ in range(10):
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)
            else:
                break

        valid = GameEngine.get_valid_moves(state, state.current_player)
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
            state = GameEngine.apply_move(state, matched)
        else:
            print(f"  FAILED at move {i}: {move_type_str}")
            break

    print("\nCPU state after replaying through move 13:")
    print(f"  Phase: {state.current_phase.value}")
    print(f"  Player: {state.current_player}")
    cpu_stacks = get_stacks(state)
    print(f"  Stacks: {cpu_stacks}")

    # Check captures available
    valid = GameEngine.get_valid_moves(state, state.current_player)
    captures = [v for v in valid if v.type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT)]
    print(f"\n  Captures available: {len(captures)}")
    for c in captures[:5]:
        print(f"    {c.type.value} from={c.from_pos} to={c.to}")


if __name__ == '__main__':
    main()
