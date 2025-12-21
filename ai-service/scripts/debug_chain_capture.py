#!/usr/bin/env python
"""Debug chain capture divergence between GPU and CPU."""
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

GPU_BOOKKEEPING_MOVES = {
    'skip_capture', 'skip_recovery', 'no_placement_action',
    'no_movement_action', 'no_line_action', 'no_territory_action',
    'process_line', 'process_territory_region',
}


def debug_seed(seed: int) -> None:
    """Debug chain capture divergence for a seed."""
    print(f"\n{'='*70}")
    print(f"Debug Chain Capture Divergence: Seed {seed}")
    print(f"{'='*70}")

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
    cpu_state = initial_state

    print(f"\nTotal GPU moves: {len(game_dict['moves'])}")

    for i, m in enumerate(game_dict['moves']):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        # Skip bookkeeping moves
        if move_type_str in GPU_BOOKKEEPING_MOVES:
            for _ in range(10):
                if cpu_state.current_phase.value == gpu_phase and cpu_state.current_player == gpu_player:
                    break
                req = GameEngine.get_phase_requirement(cpu_state, cpu_state.current_player)
                if req:
                    synth = GameEngine.synthesize_bookkeeping_move(req, cpu_state)
                    cpu_state = GameEngine.apply_move(cpu_state, synth)
                else:
                    break
            continue

        # Advance CPU to match GPU phase/player
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
                    continue
                break
            elif cpu_state.current_phase.value == 'territory_processing':
                valid = GameEngine.get_valid_moves(cpu_state, cpu_state.current_player)
                skip_moves = [v for v in valid if v.type == MoveType.SKIP_TERRITORY_PROCESSING]
                if skip_moves and gpu_player != cpu_state.current_player:
                    cpu_state = GameEngine.apply_move(cpu_state, skip_moves[0])
                    continue
                break
            else:
                break

        # Check for phase divergence
        if cpu_state.current_phase.value != gpu_phase or cpu_state.current_player != gpu_player:
            print(f"\n*** PHASE DIVERGENCE at move {i} ***")
            print(f"  GPU: {move_type_str} @ phase={gpu_phase} player={gpu_player}")
            print(f"  CPU: phase={cpu_state.current_phase.value} player={cpu_state.current_player}")

            valid = GameEngine.get_valid_moves(cpu_state, cpu_state.current_player)
            cpu_move_types = set(v.type.value for v in valid)
            print(f"  CPU offers: {list(cpu_move_types)[:5]}")

            if cpu_state.current_phase.value == 'chain_capture':
                print("\n  CPU is in CHAIN_CAPTURE - checking what captures are available:")
                for v in valid[:3]:
                    print(f"    {v.type.value}: from={v.from_pos} to={v.to}")

            print(f"\n  Last 5 GPU moves before divergence:")
            for j in range(max(0, i-5), i+1):
                gm = game_dict['moves'][j]
                print(f"    Move {j}: {gm['type']} @ {gm.get('phase')} P{gm.get('player')}")
                if gm.get('from'):
                    print(f"             from={gm.get('from')} to={gm.get('to')}")

            return

        # Try to match and apply the move
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
            cpu_state = GameEngine.apply_move(cpu_state, matched)
        else:
            print(f"\n*** MOVE MISMATCH at move {i} ***")
            print(f"  GPU: {move_type_str} from={m.get('from')} to={m.get('to')}")
            print(f"  CPU phase={cpu_state.current_phase.value} player={cpu_state.current_player}")
            print(f"  CPU valid moves of type {move_type_str}:")
            for v in valid:
                if v.type.value == move_type_str:
                    print(f"    from={v.from_pos} to={v.to}")
            return

    print(f"\nSeed {seed}: All moves matched successfully!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('seeds', type=int, nargs='+', help='Seeds to debug')
    args = parser.parse_args()

    for seed in args.seeds:
        debug_seed(seed)
