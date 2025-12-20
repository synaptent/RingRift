#!/usr/bin/env python
"""Detect exact point of GPU vs CPU state divergence.

This script runs GPU games and after each move, converts to CPU state
and compares. This identifies exactly where divergence begins.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_game_types import GamePhase, MoveType
from app._game_engine_legacy import GameEngine
from app.models import Position
import torch


def compare_states(gpu_state, game_idx: int, move_num: int) -> list[str]:
    """Compare GPU state to CPU state, return list of divergences."""
    divergences = []

    try:
        cpu_state = gpu_state.to_game_state(game_idx)
    except Exception as e:
        return [f"Failed to convert GPU state to CPU: {e}"]

    board = cpu_state.board
    board_size = gpu_state.board_size

    # Compare stacks - BoardState uses dict[str, RingStack] with "x,y" keys
    for y in range(board_size):
        for x in range(board_size):
            gpu_owner = gpu_state.stack_owner[game_idx, y, x].item()
            gpu_height = gpu_state.stack_height[game_idx, y, x].item()
            gpu_cap = gpu_state.cap_height[game_idx, y, x].item()

            pos_key = f"{x},{y}"
            cpu_stack = board.stacks.get(pos_key)

            if gpu_owner > 0 and gpu_height > 0:
                if cpu_stack is None:
                    divergences.append(f"Stack at ({x},{y}): GPU has P{gpu_owner} h={gpu_height}, CPU has nothing")
                else:
                    cpu_owner = cpu_stack.controlling_player
                    cpu_height = cpu_stack.stack_height
                    cpu_cap = cpu_stack.cap_height
                    if gpu_owner != cpu_owner:
                        divergences.append(f"Stack at ({x},{y}): GPU owner={gpu_owner}, CPU owner={cpu_owner}")
                    if gpu_height != cpu_height:
                        divergences.append(f"Stack at ({x},{y}): GPU height={gpu_height}, CPU height={cpu_height}")
                    if gpu_cap != cpu_cap:
                        divergences.append(f"Stack at ({x},{y}): GPU cap={gpu_cap}, CPU cap={cpu_cap}")
            elif cpu_stack is not None and cpu_stack.stack_height > 0:
                divergences.append(f"Stack at ({x},{y}): GPU has nothing, CPU has P{cpu_stack.controlling_player} h={cpu_stack.stack_height}")

    # Compare markers - BoardState uses dict[str, MarkerInfo] with "x,y" keys
    for y in range(board_size):
        for x in range(board_size):
            gpu_marker = gpu_state.marker_owner[game_idx, y, x].item()

            pos_key = f"{x},{y}"
            cpu_marker = board.markers.get(pos_key)
            cpu_marker_owner = cpu_marker.player if cpu_marker else 0

            if gpu_marker != cpu_marker_owner:
                divergences.append(f"Marker at ({x},{y}): GPU={gpu_marker}, CPU={cpu_marker_owner}")

    # Compare collapsed spaces - BoardState uses dict[str, int] with "x,y" keys
    for y in range(board_size):
        for x in range(board_size):
            gpu_collapsed = gpu_state.is_collapsed[game_idx, y, x].item()

            pos_key = f"{x},{y}"
            cpu_collapsed = pos_key in board.collapsed_spaces

            if gpu_collapsed != cpu_collapsed:
                divergences.append(f"Collapsed at ({x},{y}): GPU={gpu_collapsed}, CPU={cpu_collapsed}")

    return divergences


def trace_game_with_validation(seed: int = 42, max_moves: int = 50):
    """Run a game and validate after each move."""
    torch.manual_seed(seed)

    runner = ParallelGameRunner(
        batch_size=1,
        board_size=8,
        num_players=2,
        device="cpu"
    )

    print(f"=== Tracing game with seed {seed} ===\n")

    last_move_count = 0
    step = 0

    while runner.state.game_status[0].item() == 0 and step < 500:
        step += 1

        # Get state before step
        move_count_before = runner.state.move_count[0].item()

        # Execute one phase
        runner._step_games([{}])

        # Get state after step
        move_count_after = runner.state.move_count[0].item()

        # If moves were made, check for divergence
        if move_count_after > move_count_before:
            for m in range(move_count_before, move_count_after):
                # Get move info
                move = runner.state.move_history[0, m]
                move_type = int(move[0].item())
                player = int(move[1].item())
                from_y = int(move[2].item())
                from_x = int(move[3].item())
                to_y = int(move[4].item())
                to_x = int(move[5].item())
                phase = int(move[6].item())

                move_type_name = MoveType(move_type).name if move_type >= 0 else "UNKNOWN"
                phase_name = GamePhase(phase).name if phase >= 0 else "UNKNOWN"

                print(f"Move {m}: {move_type_name} P{player} ({from_x},{from_y})->({to_x},{to_y}) phase={phase_name}")

                # Check for divergence
                divergences = compare_states(runner.state, 0, m)
                if divergences:
                    print(f"  *** DIVERGENCE DETECTED ***")
                    for d in divergences[:5]:
                        print(f"    - {d}")
                    if len(divergences) > 5:
                        print(f"    ... and {len(divergences) - 5} more")
                    return m, divergences

        if move_count_after >= max_moves:
            break

    print(f"\nNo divergence detected in {move_count_after} moves")
    return -1, []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-moves", type=int, default=50)
    args = parser.parse_args()

    move_idx, divergences = trace_game_with_validation(args.seed, args.max_moves)
    if move_idx >= 0:
        print(f"\n*** First divergence at move {move_idx} ***")
