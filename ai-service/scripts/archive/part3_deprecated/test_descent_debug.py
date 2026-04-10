#!/usr/bin/env python3
"""Debug script to test DescentAI initialization and select_move."""

import os
import sys

# Allow imports from app/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.descent_ai import DescentAI
from app.models import AIConfig, BoardType


def main():
    config = AIConfig(
        difficulty=5,
        think_time=0,
        randomness=0.0,
        rngSeed=42,
        heuristic_profile_id=None,
    )

    print("Creating DescentAI...")
    ai = DescentAI(1, config)
    print(f"Created. Type: {type(ai)}")
    print(f'Has use_incremental_search: {hasattr(ai, "use_incremental_search")}')
    print(f"use_incremental_search value: {ai.use_incremental_search}")
    print(f'Has neural_net: {hasattr(ai, "neural_net")}')
    print(f"neural_net value: {ai.neural_net}")

    # Test player_number reassignment like in the evaluation script
    print()
    print("Testing player_number reassignment...")
    ai.player_number = 2
    print(f"After reassignment, player_number: {ai.player_number}")
    print(f'Has use_incremental_search after reassignment: {hasattr(ai, "use_incremental_search")}')

    # Load checkpoint (using safe loader for security)
    ckpt = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "checkpoint_final_epoch_5.pth")
    if os.path.exists(ckpt):
        from app.utils.torch_utils import safe_load_checkpoint

        print(f"\nLoading checkpoint: {ckpt}")
        checkpoint = safe_load_checkpoint(ckpt, map_location="cpu")
        if ai.neural_net is not None and "model_state_dict" in checkpoint:
            ai.neural_net.model.load_state_dict(checkpoint["model_state_dict"])
            ai.neural_net.model.eval()
            print("Checkpoint loaded successfully")
        print(f'Has use_incremental_search after checkpoint load: {hasattr(ai, "use_incremental_search")}')
    else:
        print(f"Checkpoint not found: {ckpt}")

    # Now test select_move with a real game state
    print()
    print("Testing select_move...")
    from app.training.env import RingRiftEnv

    env = RingRiftEnv(board_type=BoardType.SQUARE8)
    game_state = env.reset()

    print(f"Game state created. Current player: {game_state.current_player}")
    ai.player_number = game_state.current_player

    try:
        move = ai.select_move(game_state)
        print(f"select_move succeeded! Move: {move}")
    except Exception as e:
        print(f"select_move FAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
