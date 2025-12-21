#!/usr/bin/env python
"""Debug chain capture phase divergences between GPU and CPU."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
from app.ai.gpu_move_generation import generate_chain_capture_moves_from_position
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models import BoardType, MoveType, Position
from app.board_manager import BoardManager
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)

GPU_BOOKKEEPING_MOVES = {
    'skip_capture', 'skip_recovery', 'no_placement_action',
    'no_movement_action', 'no_line_action', 'no_territory_action',
    'process_line', 'process_territory_region',
}


def analyze_chain_divergence(seed: int) -> dict | None:
    """Analyze chain capture divergence for a seed.

    Returns dict with divergence info, or None if parity passes.
    """
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')
    for _ in range(100):
        if runner.state.game_status[0].item() != 0 or runner.state.move_count[0].item() >= 60:
            break
        runner._step_games([{}])

    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    cpu_state = initial_state

    for i, m in enumerate(game_dict['moves'][:60]):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        if move_type_str in GPU_BOOKKEEPING_MOVES:
            for _ in range(10):
                if cpu_state.current_phase.value == gpu_phase and cpu_state.current_player == gpu_player:
                    break
                req = GameEngine.get_phase_requirement(cpu_state, cpu_state.current_player)
                if req:
                    synth = GameEngine.synthesize_bookkeeping_move(req, cpu_state)
                    cpu_state = GameEngine.apply_move(cpu_state, synth)
                elif cpu_state.current_phase.value in ('capture', 'chain_capture'):
                    valid = GameEngine.get_valid_moves(cpu_state, cpu_state.current_player)
                    skip_moves = [v for v in valid if v.type == MoveType.SKIP_CAPTURE]
                    if skip_moves:
                        cpu_state = GameEngine.apply_move(cpu_state, skip_moves[0])
                    else:
                        break
                else:
                    break
            continue

        for _ in range(10):
            if cpu_state.current_phase.value == gpu_phase and cpu_state.current_player == gpu_player:
                break
            req = GameEngine.get_phase_requirement(cpu_state, cpu_state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, cpu_state)
                cpu_state = GameEngine.apply_move(cpu_state, synth)
            elif cpu_state.current_phase.value in ('capture', 'chain_capture'):
                valid = GameEngine.get_valid_moves(cpu_state, cpu_state.current_player)
                skip_moves = [v for v in valid if v.type == MoveType.SKIP_CAPTURE]
                if skip_moves:
                    cpu_state = GameEngine.apply_move(cpu_state, skip_moves[0])
                else:
                    break
            else:
                break

        move_type = MoveType(move_type_str)
        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

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
            else:
                v_from = v.from_pos.to_key() if v.from_pos else None
                m_from = from_pos.to_key() if from_pos else None
                if v_from == m_from and v_to == m_to:
                    matched = v
                    break

        if not matched:
            # Found divergence - analyze it
            cpu_phase = cpu_state.current_phase.value
            cpu_player = cpu_state.current_player

            # Collect CPU capture info
            cpu_captures = []
            for v in valid:
                if v.type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT):
                    cpu_captures.append({
                        'type': v.type.value,
                        'from': v.from_pos.to_key() if v.from_pos else None,
                        'to': v.to.to_key() if v.to else None,
                    })

            return {
                'seed': seed,
                'move_idx': i,
                'gpu_move': {
                    'type': move_type_str,
                    'phase': gpu_phase,
                    'player': gpu_player,
                    'from': m.get('from'),
                    'to': m.get('to'),
                },
                'cpu_phase': cpu_phase,
                'cpu_player': cpu_player,
                'cpu_captures': cpu_captures[:5],  # First 5
                'cpu_valid_count': len(valid),
            }

        cpu_state = GameEngine.apply_move(cpu_state, matched)

    return None  # Passed


def main():
    # Find first few failing seeds and analyze them
    print("Analyzing chain capture divergences...")
    print("=" * 70)

    divergences = []
    for seed in range(100):
        result = analyze_chain_divergence(seed)
        if result:
            divergences.append(result)
            if len(divergences) >= 5:
                break

    # Categorize divergences
    categories = {}
    for d in divergences:
        key = f"GPU:{d['gpu_move']['phase']} CPU:{d['cpu_phase']}"
        if key not in categories:
            categories[key] = []
        categories[key].append(d)

    print(f"\nFound {len(divergences)} divergences in first 100 seeds")
    print("\nDivergence Categories:")
    for key, items in sorted(categories.items()):
        print(f"  {key}: {len(items)}")

    # Show details of first few
    print("\n" + "=" * 70)
    print("Detailed Analysis of First 3 Divergences:")
    print("=" * 70)

    for d in divergences[:3]:
        print(f"\nSeed {d['seed']}, Move {d['move_idx']}:")
        print(f"  GPU move: {d['gpu_move']['type']} @ {d['gpu_move']['phase']} P{d['gpu_move']['player']}")
        print(f"           from {d['gpu_move']['from']} to {d['gpu_move']['to']}")
        print(f"  CPU phase: {d['cpu_phase']} P{d['cpu_player']}")
        print(f"  CPU valid moves: {d['cpu_valid_count']}")
        if d['cpu_captures']:
            print(f"  CPU captures available:")
            for c in d['cpu_captures']:
                print(f"    {c['type']}: {c['from']} -> {c['to']}")


if __name__ == '__main__':
    main()
