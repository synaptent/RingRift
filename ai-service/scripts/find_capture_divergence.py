#!/usr/bin/env python
"""Find seeds that have capture source divergence (CPU has captures from non-landing stack)."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models import BoardType, MoveType, Position
from app.rules.capture_chain import enumerate_capture_moves_py
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)


def check_seed(seed: int) -> dict | None:
    """Check a seed for capture source divergence. Returns info if found."""
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
                return None  # Earlier divergence caused mismatch

            cpu_state = GameEngine.apply_move(cpu_state, matched)

            # Now CPU should be in CAPTURE phase
            if cpu_state.current_phase.value == 'capture':
                cpu_captures = GameEngine.get_valid_moves(cpu_state, cpu_state.current_player)
                cpu_capture_moves = [c for c in cpu_captures if c.type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT)]

                landing_captures = enumerate_capture_moves_py(
                    cpu_state,
                    cpu_state.current_player,
                    to_pos,
                    move_number=len(cpu_state.move_history) + 1,
                    kind="initial",
                )

                has_cpu_captures = len(cpu_capture_moves) > 0
                has_landing_captures = len(landing_captures) > 0

                if has_cpu_captures and not has_landing_captures:
                    # Found divergence - identify which stacks have captures
                    other_stacks = []
                    for stack in cpu_state.board.stacks.values():
                        if stack.controlling_player != cpu_state.current_player:
                            continue
                        if stack.stack_height <= 0:
                            continue
                        if stack.position.to_key() == to_pos.to_key():
                            continue
                        caps = enumerate_capture_moves_py(
                            cpu_state,
                            cpu_state.current_player,
                            stack.position,
                            move_number=len(cpu_state.move_history) + 1,
                            kind="initial",
                        )
                        if caps:
                            other_stacks.append((stack.position, len(caps)))

                    return {
                        'seed': seed,
                        'move_idx': i,
                        'from': str(from_pos),
                        'to': str(to_pos),
                        'cpu_captures': len(cpu_capture_moves),
                        'landing_captures': len(landing_captures),
                        'other_stacks': other_stacks,
                    }
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

    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=100)
    parser.add_argument('--start', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.start)
    seeds = [random.randint(0, 100000) for _ in range(args.seeds)]

    divergences = []
    for seed in seeds:
        result = check_seed(seed)
        if result:
            divergences.append(result)

    print(f"\nFound {len(divergences)} seeds with capture source divergence:")
    for d in divergences[:10]:
        print(f"\n  Seed {d['seed']} move {d['move_idx']}:")
        print(f"    Move: from {d['from']} to {d['to']}")
        print(f"    CPU total: {d['cpu_captures']}, Landing: {d['landing_captures']}")
        print(f"    Other stacks with captures: {d['other_stacks']}")


if __name__ == '__main__':
    main()
