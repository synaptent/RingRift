#!/usr/bin/env python
"""Test GPU move expansion to canonical format."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
from app.training.initial_state import create_initial_state
from app.models import BoardType

from scripts.import_gpu_selfplay_to_db import expand_gpu_jsonl_moves_to_canonical, parse_move


def main():
    # Run a short GPU game
    torch.manual_seed(42)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device="cpu")

    for step in range(150):
        if runner.state.game_status[0].item() != 0:
            break
        runner._step_games([{}])

    # Export to canonical format
    game_dict = export_game_to_canonical_dict(runner.state, 0, "square8", 2)
    print(f"GPU game: {len(game_dict['moves'])} moves")

    # Parse GPU moves to Move objects
    timestamp = "2024-01-01T00:00:00Z"
    gpu_moves = []
    for i, m in enumerate(game_dict["moves"]):
        try:
            parsed = parse_move(m, i, timestamp, BoardType.SQUARE8)
            if parsed:
                gpu_moves.append(parsed)
        except Exception as e:
            print(f"  Parse error at move {i}: {e}")
            break

    print(f"Parsed {len(gpu_moves)} moves")

    # Create initial state
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)

    # Try expanding to canonical (this handles phase divergence)
    try:
        canonical, final = expand_gpu_jsonl_moves_to_canonical(
            gpu_moves,
            initial_state,
            verbose=True,
            skip_invalid_moves=False
        )
        print()
        print(f"SUCCESS: Expanded to {len(canonical)} canonical moves")
        print(f"Final phase: {final.current_phase.value}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
