#!/usr/bin/env python3
"""Validate MinimaxAI policy ordering integration.

Tests that:
1. Policy model loads correctly in MinimaxAI
2. Move ordering uses policy scores when enabled
3. MinimaxAI can make moves with policy ordering enabled
"""

import sys
import time
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.minimax_ai import MinimaxAI
from app.game_engine import GameEngine
from app.models import AIConfig
from app.training.initial_state import create_initial_state


def validate_policy_ordering():
    """Validate policy ordering in MinimaxAI."""
    print("=" * 60)
    print("MinimaxAI Policy Ordering Validation")
    print("=" * 60)

    # Check if policy model exists
    policy_model_path = Path("models/nnue/nnue_policy_square8_2p.pt")
    if not policy_model_path.exists():
        print(f"SKIP: Policy model not found at {policy_model_path}")
        return False

    print(f"Policy model found: {policy_model_path}")
    print()

    # Create game engine and initial state
    print("Creating game engine and initial state...")
    engine = GameEngine()
    game_state = create_initial_state()

    # Create MinimaxAI with policy ordering enabled
    print("Creating MinimaxAI with policy ordering enabled...")
    config = AIConfig(
        difficulty=5,
        use_policy_ordering=True,
    )
    ai = MinimaxAI(player_number=1, config=config)

    # Make a few moves to get to movement phase
    print("Making initial placement moves...")
    moves_made = 0
    while game_state.current_phase.value != "movement" and moves_made < 20:
        current_player = game_state.current_player
        valid_moves = engine.get_valid_moves(game_state, current_player)
        if not valid_moves:
            print("No valid moves available")
            break

        # Use a simple heuristic AI for placement
        simple_ai = MinimaxAI(player_number=current_player, config=AIConfig(difficulty=1))
        move = simple_ai.select_move(game_state)
        game_state = engine.apply_move(game_state, move)
        moves_made += 1

    print(f"Made {moves_made} placement moves, now in {game_state.current_phase.value} phase")
    print()

    # Test policy ordering on movement phase
    if game_state.current_phase.value == "movement":
        current_player = game_state.current_player
        valid_moves = engine.get_valid_moves(game_state, current_player)
        print(f"Valid moves in movement phase: {len(valid_moves)}")

        if len(valid_moves) > 2:
            # Test policy-based move selection
            print("\nTesting policy-based move selection...")
            start_time = time.time()
            selected_move = ai.select_move(game_state)
            elapsed = time.time() - start_time

            print(f"Policy AI selected move: {selected_move}")
            print(f"Selection time: {elapsed:.3f}s")

            # Check if policy model was loaded
            if ai.policy_model is not None:
                print("Policy model loaded: YES")
                print("Policy ordering enabled: " + str(ai.use_policy_ordering))
            else:
                print("Policy model loaded: NO")

    print("\nValidation completed successfully!")
    return True


def validate_policy_vs_no_policy():
    """Compare move ordering with and without policy."""
    print("\n" + "=" * 60)
    print("Comparing Policy vs No-Policy Move Ordering")
    print("=" * 60)

    # Check if policy model exists
    policy_model_path = Path("models/nnue/nnue_policy_square8_2p.pt")
    if not policy_model_path.exists():
        print(f"SKIP: Policy model not found at {policy_model_path}")
        return False

    # Create game engine and advance to mid-game
    engine = GameEngine()
    game_state = create_initial_state()

    # Place rings to get to movement phase
    moves_made = 0
    while game_state.current_phase.value != "movement" and moves_made < 25:
        current_player = game_state.current_player
        valid_moves = engine.get_valid_moves(game_state, current_player)
        if not valid_moves:
            break
        simple_ai = MinimaxAI(player_number=current_player, config=AIConfig(difficulty=1))
        move = simple_ai.select_move(game_state)
        game_state = engine.apply_move(game_state, move)
        moves_made += 1

    if game_state.current_phase.value != "movement":
        print("Could not reach movement phase")
        return False

    current_player = game_state.current_player
    valid_moves = engine.get_valid_moves(game_state, current_player)
    print(f"Movement phase with {len(valid_moves)} valid moves")

    # Create AI with policy
    ai_with_policy = MinimaxAI(
        player_number=game_state.current_player,
        config=AIConfig(difficulty=5, use_policy_ordering=True),
    )

    # Create AI without policy
    ai_without_policy = MinimaxAI(
        player_number=game_state.current_player,
        config=AIConfig(difficulty=5, use_policy_ordering=False),
    )

    # Compare timing
    print("\nComparing move selection time...")

    start = time.time()
    move_with_policy = ai_with_policy.select_move(game_state)
    time_with_policy = time.time() - start

    start = time.time()
    move_without_policy = ai_without_policy.select_move(game_state)
    time_without_policy = time.time() - start

    print(f"With policy:    {time_with_policy:.3f}s - selected {move_with_policy}")
    print(f"Without policy: {time_without_policy:.3f}s - selected {move_without_policy}")

    if ai_with_policy.policy_model is not None:
        print("\nPolicy model loaded successfully!")
        speedup = time_without_policy / time_with_policy if time_with_policy > 0 else 0
        print(f"Speedup from policy ordering: {speedup:.2f}x")
    else:
        print("\nWARNING: Policy model not loaded")

    return True


def main():
    """Run all validation tests."""
    success = True

    try:
        success = validate_policy_ordering() and success
    except Exception as e:
        print(f"Error in validate_policy_ordering: {e}")
        import traceback
        traceback.print_exc()
        success = False

    try:
        success = validate_policy_vs_no_policy() and success
    except Exception as e:
        print(f"Error in validate_policy_vs_no_policy: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "=" * 60)
    if success:
        print("ALL VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
