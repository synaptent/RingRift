#!/usr/bin/env python
"""Debug GPU marker state at move 48."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)


def print_state(runner, label):
    move_count = int(runner.state.move_count[0].item())
    print(f"\n{label} (move_count={move_count}):")

    print("  Stacks:")
    for y in range(8):
        for x in range(8):
            owner = int(runner.state.stack_owner[0, y, x].item())
            height = int(runner.state.stack_height[0, y, x].item())
            if owner > 0 and height > 0:
                print(f"    ({x},{y}): P{owner}h{height}")

    print("  Markers:")
    for y in range(8):
        for x in range(8):
            marker_owner = int(runner.state.marker_owner[0, y, x].item())
            if marker_owner > 0:
                print(f"    ({x},{y}): P{marker_owner}")


def main():
    seed = 456
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    # Step one move at a time and print state around move 47-50
    for step in range(100):
        move_count = int(runner.state.move_count[0].item())

        if move_count == 47:
            print_state(runner, "Before move 47 is applied")

        if move_count == 48:
            print_state(runner, "Before move 48 is applied")

        if move_count == 49:
            print_state(runner, "After move 48 (before move 49)")

        if move_count == 50:
            print_state(runner, "After move 49 (before move 50)")
            break

        runner._step_games([{}])
        if runner.state.game_status[0].item() != 0:
            break

    # Export and print moves 46-50
    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)
    print("\n\nExported moves 46-51:")
    for i, m in enumerate(game_dict['moves'][46:52], start=46):
        from_str = f"({m['from']['x']},{m['from']['y']})" if m.get('from') else "None"
        to_str = f"({m['to']['x']},{m['to']['y']})" if m.get('to') else "None"
        cap = m.get('captureTarget')
        cap_str = f"cap=({cap['x']},{cap['y']})" if cap else ""
        print(f"  {i}: [{m.get('phase','?'):20}] {m['type']:25} P{m['player']} {from_str:10} -> {to_str:10} {cap_str}")


if __name__ == '__main__':
    main()
