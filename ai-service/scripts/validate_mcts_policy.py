#!/usr/bin/env python3
"""Validate NNUE policy integration in MCTS.

Tests that:
1. NNUE policy model loads correctly in MCTSAI
2. Policy priors are used when no neural network is available
3. MCTSAI can make moves with NNUE policy priors
"""

import sys
import time
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.mcts_ai import MCTSAI
from app.game_engine import GameEngine
from app.models import AIConfig
from app.training.initial_state import create_initial_state


def validate_nnue_policy_priors():
    """Validate NNUE policy priors in MCTSAI."""
    print("=" * 60)
    print("MCTSAI NNUE Policy Priors Validation")
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

    # Create MCTSAI with NNUE policy priors enabled and NO neural net
    print("Creating MCTSAI with NNUE policy priors (no neural net)...")
    config = AIConfig(
        difficulty=4,  # Below D6 so no neural net
        use_neural_net=False,  # Explicitly disable neural net
        use_nnue_policy_priors=True,
    )
    ai = MCTSAI(player_number=1, config=config)

    # Verify neural net is disabled
    print(f"Neural net loaded: {ai.neural_net is not None}")
    print(f"NNUE policy pending init: {ai._pending_nnue_policy_init}")

    # Make first move to trigger NNUE policy initialization
    print("\nMaking first move to trigger NNUE policy initialization...")
    start_time = time.time()
    move = ai.select_move(game_state)
    elapsed = time.time() - start_time

    print(f"Selected move: {move}")
    print(f"Selection time: {elapsed:.3f}s")
    print(f"NNUE policy model loaded: {ai.nnue_policy_model is not None}")

    if ai.nnue_policy_model is None:
        print("FAIL: NNUE policy model not loaded")
        return False

    print("\nValidation PASSED!")
    return True


def validate_policy_priors_vs_uniform():
    """Compare MCTS with NNUE policy priors vs uniform priors."""
    print("\n" + "=" * 60)
    print("Comparing Policy Priors vs Uniform Priors")
    print("=" * 60)

    # Check if policy model exists
    policy_model_path = Path("models/nnue/nnue_policy_square8_2p.pt")
    if not policy_model_path.exists():
        print(f"SKIP: Policy model not found at {policy_model_path}")
        return False

    # Create game engine
    engine = GameEngine()
    game_state = create_initial_state()

    # Advance to mid-game
    print("Advancing to movement phase...")
    moves_made = 0
    while game_state.current_phase.value != "movement" and moves_made < 25:
        current_player = game_state.current_player
        valid_moves = engine.get_valid_moves(game_state, current_player)
        if not valid_moves:
            break
        # Use simple heuristic to place rings
        from app.ai.heuristic_ai import HeuristicAI
        simple_ai = HeuristicAI(player_number=current_player, config=AIConfig(difficulty=1))
        move = simple_ai.select_move(game_state)
        game_state = engine.apply_move(game_state, move)
        moves_made += 1

    if game_state.current_phase.value != "movement":
        print("Could not reach movement phase")
        return False

    print(f"Reached movement phase after {moves_made} moves")
    valid_moves = engine.get_valid_moves(game_state, game_state.current_player)
    print(f"Valid moves in movement phase: {len(valid_moves)}")

    # MCTS with NNUE policy
    print("\nTesting MCTS with NNUE policy priors...")
    ai_with_policy = MCTSAI(
        player_number=game_state.current_player,
        config=AIConfig(
            difficulty=4,
            use_neural_net=False,
            use_nnue_policy_priors=True,
            think_time=500,  # 0.5s
        ),
    )

    start = time.time()
    move_with_policy, policy_dist = ai_with_policy.select_move_and_policy(game_state)
    time_with_policy = time.time() - start

    # MCTS without NNUE policy (uniform priors)
    print("Testing MCTS with uniform priors...")
    ai_without_policy = MCTSAI(
        player_number=game_state.current_player,
        config=AIConfig(
            difficulty=4,
            use_neural_net=False,
            use_nnue_policy_priors=False,
            think_time=500,  # 0.5s
        ),
    )

    start = time.time()
    move_without_policy, _ = ai_without_policy.select_move_and_policy(game_state)
    time_without_policy = time.time() - start

    print(f"\nWith NNUE policy: {time_with_policy:.3f}s - {move_with_policy}")
    print(f"Without policy:   {time_without_policy:.3f}s - {move_without_policy}")

    if ai_with_policy.nnue_policy_model is not None:
        print("\nNNUE policy model loaded successfully!")

        # Show top policy moves
        if policy_dist:
            sorted_policy = sorted(policy_dist.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\nTop 5 moves by MCTS policy:")
            for move_str, prob in sorted_policy:
                print(f"  {prob:.3f}: {move_str[:50]}...")
    else:
        print("\nWARNING: NNUE policy model not loaded")

    return True


def main():
    """Run all validation tests."""
    success = True

    try:
        success = validate_nnue_policy_priors() and success
    except Exception as e:
        print(f"Error in validate_nnue_policy_priors: {e}")
        import traceback
        traceback.print_exc()
        success = False

    try:
        success = validate_policy_priors_vs_uniform() and success
    except Exception as e:
        print(f"Error in validate_policy_priors_vs_uniform: {e}")
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
