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
from app.models import BoardType, MoveType, Position
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)

GPU_BOOKKEEPING_MOVES = {'skip_capture'}


def test_seed(seed: int) -> tuple[int, int, int, int, list]:
    """Test GPU to CPU parity for a given seed."""
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    for step in range(100):
        game_status = runner.state.game_status[0].item()
        move_count = runner.state.move_count[0].item()
        if game_status != 0:
            break
        if move_count >= 60:
            break
        runner._step_games([{}])

    total_moves = int(runner.state.move_count[0].item())
    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)
    exported_moves = len(game_dict['moves'])

    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state
    errors = []
    skipped = 0

    for i, m in enumerate(game_dict['moves']):
        move_type_str = m['type']

        if move_type_str in GPU_BOOKKEEPING_MOVES:
            skipped += 1
            continue

        move_type = MoveType(move_type_str)
        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

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
            elif move_type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT):
                v_from = v.from_pos.to_key() if v.from_pos else None
                m_from = from_pos.to_key() if from_pos else None
                if v_from == m_from and v_to == m_to:
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
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)
            else:
                errors.append((i, m['type'], state.current_phase.value))

    return total_moves, exported_moves, skipped, len(errors), errors[:3]


def main():
    # Test multiple seeds
    results = []
    for seed in [42, 123, 456, 789, 1000, 2024]:
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
