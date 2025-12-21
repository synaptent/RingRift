#!/usr/bin/env python
"""Debug chain capture to see why CPU doesn't find more captures."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models import BoardType, MoveType, Position, GamePhase
from app.board_manager import BoardManager
from app.rules.capture_chain import enumerate_capture_moves_py
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)

GPU_BOOKKEEPING_MOVES = {
    'skip_capture', 'skip_recovery', 'no_placement_action',
    'no_movement_action', 'no_line_action', 'no_territory_action',
    'process_line', 'process_territory_region',
}


def debug_seed(seed: int, target_move: int) -> None:
    """Debug chain capture at target move."""
    print(f"\n{'='*70}")
    print(f"Debug Seed {seed}, detailed chain capture analysis at move {target_move}")
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

    for i, m in enumerate(game_dict['moves']):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        # Skip bookkeeping
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
                else:
                    break
            else:
                break

        # Apply move
        move_type = MoveType(move_type_str)
        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

        # If this is the target move, do detailed analysis BEFORE applying
        if i == target_move:
            print(f"\n=== BEFORE move {i}: {move_type_str} ===")
            print(f"  GPU: from {from_pos} to {to_pos} @ {gpu_phase} P{gpu_player}")
            print(f"  CPU phase: {cpu_state.current_phase.value} P{cpu_state.current_player}")
            if cpu_state.chain_capture_state:
                ccs = cpu_state.chain_capture_state
                print(f"  CPU chain state: pos={ccs.current_position}, visited={ccs.visited_positions}")

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

            # If this is the target move, do detailed analysis AFTER applying
            if i == target_move:
                print(f"\n=== AFTER move {i}: {move_type_str} ===")
                print(f"  CPU phase: {cpu_state.current_phase.value} P{cpu_state.current_player}")
                if cpu_state.chain_capture_state:
                    ccs = cpu_state.chain_capture_state
                    print(f"  CPU chain state: pos={ccs.current_position}, visited={ccs.visited_positions}")
                else:
                    print(f"  CPU chain state: None")

                # Check what captures exist from the landing position
                player = cpu_state.current_player
                landing_pos = to_pos

                # Get stack at landing
                stack = BoardManager.get_stack(landing_pos, cpu_state.board)
                if stack:
                    print(f"\n  Stack at landing {landing_pos}: h={stack.stack_height} cap={stack.cap_height} owner={stack.controlling_player}")
                else:
                    print(f"  No stack at landing {landing_pos}")

                # Enumerate captures from landing
                caps = enumerate_capture_moves_py(
                    cpu_state,
                    player,
                    landing_pos,
                    move_number=len(cpu_state.move_history) + 1,
                    kind="continuation",
                )
                print(f"\n  Captures from landing ({len(caps)}):")
                for c in caps[:5]:
                    print(f"    {c.type.value} to {c.to}")
                    if c.capture_target:
                        target_stack = BoardManager.get_stack(c.capture_target, cpu_state.board)
                        if target_stack:
                            print(f"      target at {c.capture_target}: h={target_stack.stack_height} cap={target_stack.cap_height} owner={target_stack.controlling_player}")

                # What GPU expects next
                if i + 1 < len(game_dict['moves']):
                    next_m = game_dict['moves'][i + 1]
                    print(f"\n  Next GPU move: {next_m['type']} @ {next_m.get('phase')} P{next_m.get('player')}")
                    if next_m.get('from'):
                        print(f"    from {next_m['from']} to {next_m['to']}")

                # Show board state around landing
                print(f"\n  Board near landing {landing_pos}:")
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = landing_pos.y + dy, landing_pos.x + dx
                        if 0 <= ny < 8 and 0 <= nx < 8:
                            pos = Position(x=nx, y=ny)
                            s = BoardManager.get_stack(pos, cpu_state.board)
                            if s:
                                marker = "*" if pos.to_key() == landing_pos.to_key() else ""
                                print(f"    ({ny},{nx}): h={s.stack_height} cap={s.cap_height} owner={s.controlling_player} {marker}")

                return
        else:
            print(f"\n*** MOVE MISMATCH at move {i} ***")
            print(f"  GPU: {move_type_str} from={m.get('from')} to={m.get('to')}")
            print(f"  CPU phase={cpu_state.current_phase.value} player={cpu_state.current_player}")
            return

    print(f"\nSeed {seed}: Completed without reaching target move")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int, help='Seed to debug')
    parser.add_argument('--move', type=int, required=True, help='Target move to analyze')
    args = parser.parse_args()

    debug_seed(args.seed, args.move)
