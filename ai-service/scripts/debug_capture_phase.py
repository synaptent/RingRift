#!/usr/bin/env python
"""Debug capture phase stuck issue."""
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
    'skip_capture', 'skip_recovery',
    'no_placement_action', 'no_movement_action', 'no_line_action', 'no_territory_action',
    'process_line', 'process_territory_region', 'choose_line_option', 'choose_territory_option',
}


def advance_cpu_through_phases(state, target_phase_str: str, target_player: int):
    from app.models import GamePhase
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
            break

    return state


def main():
    seed = 42
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    for step in range(100):
        if runner.state.move_count[0].item() >= 60:
            break
        runner._step_games([{}])

    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)

    # Show moves 14-18
    print("GPU moves 10-20:")
    for i, m in enumerate(game_dict['moves'][10:21], start=10):
        from_str = f"({m['from']['x']},{m['from']['y']})" if 'from' in m and m['from'] else "None"
        to_str = f"({m['to']['x']},{m['to']['y']})" if 'to' in m and m['to'] else "None"
        print(f"  {i:2}: [{m.get('phase','?'):20}] {m['type']:25} player={m['player']}")

    # Replay CPU with detailed tracing
    print("\n\nCPU Replay:")
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    for i, m in enumerate(game_dict['moves'][:25]):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        if move_type_str in GPU_BOOKKEEPING_MOVES:
            state = advance_cpu_through_phases(state, gpu_phase, gpu_player)
            if i >= 10:
                print(f"  {i:2}: SKIP {move_type_str:25} cpu={state.current_phase.value:15} player={state.current_player}")
            continue

        move_type = MoveType(move_type_str)
        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

        old_phase = state.current_phase.value
        old_player = state.current_player
        state = advance_cpu_through_phases(state, gpu_phase, gpu_player)
        new_phase = state.current_phase.value
        new_player = state.current_player

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
            if i >= 10:
                print(f"  {i:2}: OK   {move_type_str:25} advanced {old_phase:12}→{new_phase:12} now={state.current_phase.value:12}")
        else:
            if i >= 10:
                print(f"  {i:2}: FAIL {move_type_str:25} gpu_phase={gpu_phase:12} cpu_phase={state.current_phase.value:12}")
                print(f"        GPU wants player={gpu_player}, CPU has player={state.current_player}")

                # Check phase requirement
                req = GameEngine.get_phase_requirement(state, state.current_player)
                if req:
                    print(f"        CPU phase requirement: {req}")
                else:
                    print(f"        CPU phase requirement: None")

                # Show valid moves
                print(f"        Valid moves ({len(valid)}):")
                for v in valid[:5]:
                    print(f"          {v.type.value}")

            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)


if __name__ == '__main__':
    main()
