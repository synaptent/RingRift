#!/usr/bin/env python
"""Debug capture source divergence - GPU only checks landing, CPU checks all stacks."""
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
from app.rules.capture_chain import enumerate_capture_moves_py
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)


def debug_seed(seed: int) -> None:
    """Debug capture source for a seed to find where GPU misses captures."""
    print(f"\n{'='*70}")
    print(f"Debug Capture Source: Seed {seed}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    # Run a few steps
    for step in range(100):
        game_status = runner.state.game_status[0].item()
        move_count = runner.state.move_count[0].item()
        if game_status != 0 or move_count >= 60:
            break
        runner._step_games([{}])

    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    cpu_state = initial_state

    for i, m in enumerate(game_dict['moves']):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        # Skip bookkeeping
        if move_type_str in {'skip_capture', 'skip_recovery', 'no_placement_action',
                             'no_movement_action', 'no_line_action', 'no_territory_action',
                             'process_line', 'process_territory_region'}:
            # Advance CPU through bookkeeping
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

        # For move_stack, check what captures CPU sees after the move
        if move_type_str == 'move_stack':
            # Advance CPU to match
            for _ in range(10):
                if cpu_state.current_phase.value == gpu_phase and cpu_state.current_player == gpu_player:
                    break
                req = GameEngine.get_phase_requirement(cpu_state, cpu_state.current_player)
                if req:
                    synth = GameEngine.synthesize_bookkeeping_move(req, cpu_state)
                    cpu_state = GameEngine.apply_move(cpu_state, synth)
                else:
                    break

            # Apply the move on CPU
            from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
            to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

            valid = GameEngine.get_valid_moves(cpu_state, cpu_state.current_player)
            matched = None
            for v in valid:
                if v.type.value == move_type_str:
                    v_from = v.from_pos.to_key() if v.from_pos else None
                    v_to = v.to.to_key() if v.to else None
                    m_from = from_pos.to_key() if from_pos else None
                    m_to = to_pos.to_key() if to_pos else None
                    if v_from == m_from and v_to == m_to:
                        matched = v
                        break

            if not matched:
                print(f"Move {i}: Could not match move_stack {m}")
                return

            cpu_state = GameEngine.apply_move(cpu_state, matched)

            # Now CPU should be in CAPTURE phase
            if cpu_state.current_phase.value == 'capture':
                # Get all CPU captures
                cpu_captures = GameEngine.get_valid_moves(cpu_state, cpu_state.current_player)
                cpu_capture_moves = [c for c in cpu_captures if c.type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT)]

                # Check captures from landing position only (what GPU does)
                landing_captures = enumerate_capture_moves_py(
                    cpu_state,
                    cpu_state.current_player,
                    to_pos,
                    move_number=len(cpu_state.move_history) + 1,
                    kind="initial",
                )

                # Check captures from all stacks (what CPU does)
                all_stack_captures = []
                for stack in cpu_state.board.stacks.values():
                    if stack.controlling_player != cpu_state.current_player:
                        continue
                    if stack.stack_height <= 0:
                        continue
                    caps = enumerate_capture_moves_py(
                        cpu_state,
                        cpu_state.current_player,
                        stack.position,
                        move_number=len(cpu_state.move_history) + 1,
                        kind="initial",
                    )
                    if caps:
                        all_stack_captures.append((stack.position, caps))

                # Look for divergence: CPU has captures but landing position has none
                has_cpu_captures = len(cpu_capture_moves) > 0
                has_landing_captures = len(landing_captures) > 0

                if has_cpu_captures and not has_landing_captures:
                    print(f"\n*** CAPTURE SOURCE DIVERGENCE at move {i} ***")
                    print(f"  Move: {move_type_str} from {from_pos} to {to_pos}")
                    print(f"  CPU phase: {cpu_state.current_phase.value}")
                    print(f"  CPU has {len(cpu_capture_moves)} captures total")

                    print(f"\n  Landing position ({to_pos}) has {len(landing_captures)} captures")

                    print(f"\n  Other stacks with captures:")
                    for pos, caps in all_stack_captures:
                        if pos.to_key() != to_pos.to_key():
                            print(f"    Stack at {pos} has {len(caps)} captures:")
                            for c in caps[:2]:
                                print(f"      to={c.to}")

                    # Look ahead to see if GPU emitted captures from landing or skipped
                    next_moves = game_dict['moves'][i+1:i+5]
                    print(f"\n  Next GPU moves:")
                    for nm in next_moves:
                        print(f"    {nm['type']} @ {nm.get('phase')} P{nm.get('player')}")

                    return
            continue

        # Apply other move types
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
            else:
                v_from = v.from_pos.to_key() if v.from_pos else None
                m_from = from_pos.to_key() if from_pos else None
                if v_from == m_from and v_to == m_to:
                    matched = v
                    break

        if matched:
            cpu_state = GameEngine.apply_move(cpu_state, matched)

    print(f"\nSeed {seed}: No capture source divergence found")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('seeds', type=int, nargs='+', help='Seeds to debug')
    args = parser.parse_args()

    for seed in args.seeds:
        debug_seed(seed)
