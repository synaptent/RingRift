#!/usr/bin/env python
"""Analyze parity failures to categorize divergence types."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from collections import defaultdict
from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models import BoardType, MoveType, Position
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)

GPU_BOOKKEEPING_MOVES = {
    'skip_capture', 'skip_recovery', 'no_placement_action',
    'no_movement_action', 'no_line_action', 'no_territory_action',
    'process_line', 'process_territory_region',
}


def advance_cpu_through_phases(state, target_phase_str: str, target_player: int):
    """Advance CPU state through bookkeeping phases."""
    from app.models import GamePhase

    for _ in range(10):
        current_phase = state.current_phase.value
        current_player = state.current_player

        if current_phase == target_phase_str and current_player == target_player:
            return state

        req = GameEngine.get_phase_requirement(state, state.current_player)
        if req:
            synth = GameEngine.synthesize_bookkeeping_move(req, state)
            state = GameEngine.apply_move(state, synth)
        else:
            if current_phase in ('capture', 'chain_capture'):
                valid = GameEngine.get_valid_moves(state, state.current_player)
                skip_moves = [v for v in valid if v.type == MoveType.SKIP_CAPTURE]
                if skip_moves:
                    state = GameEngine.apply_move(state, skip_moves[0])
                    continue
            elif current_phase == 'territory_processing':
                valid = GameEngine.get_valid_moves(state, state.current_player)
                skip_moves = [v for v in valid if v.type == MoveType.SKIP_TERRITORY_PROCESSING]
                if skip_moves and target_player != current_player:
                    state = GameEngine.apply_move(state, skip_moves[0])
                    continue
            break

    return state


def analyze_seed(seed: int) -> dict:
    """Analyze a single seed for divergence patterns."""
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    for step in range(100):
        game_status = runner.state.game_status[0].item()
        move_count = runner.state.move_count[0].item()
        if game_status != 0 or move_count >= 60:
            break
        runner._step_games([{}])

    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    errors = []

    for i, m in enumerate(game_dict['moves']):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        if move_type_str in GPU_BOOKKEEPING_MOVES:
            state = advance_cpu_through_phases(state, gpu_phase, gpu_player)
            continue

        move_type = MoveType(move_type_str)
        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

        state = advance_cpu_through_phases(state, gpu_phase, gpu_player)

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
            elif move_type == MoveType.SKIP_PLACEMENT:
                matched = v
                break
            elif move_type == MoveType.RECOVERY_SLIDE:
                v_from = v.from_pos.to_key() if v.from_pos else None
                m_from = from_pos.to_key() if from_pos else None
                if v_from == m_from and v_to == m_to:
                    matched = v
                    break
            elif move_type in (MoveType.CHOOSE_LINE_OPTION, MoveType.PROCESS_LINE,
                               MoveType.CHOOSE_TERRITORY_OPTION, MoveType.PROCESS_TERRITORY_REGION,
                               MoveType.TERRITORY_CLAIM):
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
            # Categorize the error
            cpu_phase = state.current_phase.value
            cpu_player = state.current_player

            error_info = {
                'move_idx': i,
                'gpu_move_type': move_type_str,
                'gpu_phase': gpu_phase,
                'gpu_player': gpu_player,
                'cpu_phase': cpu_phase,
                'cpu_player': cpu_player,
                'gpu_from': m.get('from'),
                'gpu_to': m.get('to'),
            }

            # Check what CPU offers
            cpu_move_types = set(v.type.value for v in valid)
            error_info['cpu_move_types'] = list(cpu_move_types)
            error_info['cpu_move_count'] = len(valid)

            # Determine divergence type
            if gpu_phase != cpu_phase:
                error_info['divergence_type'] = 'phase_mismatch'
            elif gpu_player != cpu_player:
                error_info['divergence_type'] = 'player_mismatch'
            elif move_type_str in cpu_move_types:
                # Same type available but different positions
                error_info['divergence_type'] = 'position_mismatch'
            else:
                error_info['divergence_type'] = 'move_type_unavailable'

            errors.append(error_info)

            # Try to continue
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)

    return {'seed': seed, 'errors': errors}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=200)
    parser.add_argument('--start-seed', type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.start_seed)
    seeds = [random.randint(0, 100000) for _ in range(args.seeds)]

    divergence_counts = defaultdict(int)
    phase_mismatch_details = defaultdict(int)
    position_mismatch_details = defaultdict(int)
    all_failures = []

    for seed in seeds:
        result = analyze_seed(seed)
        if result['errors']:
            all_failures.append(result)
            for err in result['errors']:
                divergence_counts[err['divergence_type']] += 1
                if err['divergence_type'] == 'phase_mismatch':
                    key = f"GPU:{err['gpu_phase']} CPU:{err['cpu_phase']}"
                    phase_mismatch_details[key] += 1
                elif err['divergence_type'] == 'position_mismatch':
                    position_mismatch_details[err['gpu_move_type']] += 1

    print(f"\n{'='*60}")
    print(f"Parity Analysis: {args.seeds} seeds")
    print(f"{'='*60}")
    print(f"\nFailing seeds: {len(all_failures)}/{args.seeds} ({100*len(all_failures)/args.seeds:.1f}%)")

    print(f"\nDivergence Types:")
    for dtype, count in sorted(divergence_counts.items(), key=lambda x: -x[1]):
        print(f"  {dtype}: {count}")

    if phase_mismatch_details:
        print(f"\nPhase Mismatch Details:")
        for detail, count in sorted(phase_mismatch_details.items(), key=lambda x: -x[1]):
            print(f"  {detail}: {count}")

    if position_mismatch_details:
        print(f"\nPosition Mismatch by Move Type:")
        for mtype, count in sorted(position_mismatch_details.items(), key=lambda x: -x[1]):
            print(f"  {mtype}: {count}")

    # Show example failures
    print(f"\nExample Failures (first 3):")
    for result in all_failures[:3]:
        print(f"\n  Seed {result['seed']}:")
        for err in result['errors'][:2]:
            print(f"    Move {err['move_idx']}: {err['divergence_type']}")
            print(f"      GPU: {err['gpu_move_type']} @ {err['gpu_phase']} P{err['gpu_player']}")
            print(f"      CPU: phase={err['cpu_phase']} P{err['cpu_player']}, offers {err['cpu_move_types'][:3]}...")


if __name__ == '__main__':
    main()
