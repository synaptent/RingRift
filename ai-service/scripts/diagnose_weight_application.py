#!/usr/bin/env python
"""Diagnostic script to verify that weight modifications affect HeuristicAI behavior.

This script tests the critical hypothesis that different weight vectors produce
different game evaluations and move selections. If weights have no effect, we've
found the root cause of the zero-variance CMA-ES training issue.
"""

import os
import sys

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.ai.heuristic_ai import HeuristicAI
from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS
from app.models import AIConfig, BoardType
from app.training.eval_pools import load_state_pool


def test_weight_application():
    """Test that different weights produce different evaluations."""

    print("=" * 70)
    print("WEIGHT APPLICATION DIAGNOSTIC TEST")
    print("=" * 70)
    print()

    # Load test states from the evaluation pool
    print("Loading test states from evaluation pool...")
    try:
        states = load_state_pool(BoardType.SQUARE8, "v1", max_states=3)
        if not states:
            print("⚠️  WARNING: No states in evaluation pool, creating initial state")
            from datetime import datetime

            from app.models import BoardState, GamePhase, GameState, GameStatus, Player, TimeControl

            board = BoardState(
                type=BoardType.SQUARE8,
                size=8,
                stacks={},
                markers={},
                collapsedSpaces={},
                eliminatedRings={},
            )

            players = [
                Player(
                    id=f"player{i}",
                    username=f"Test Player {i}",
                    type="ai",
                    playerNumber=i,
                    isReady=True,
                    timeRemaining=600000,
                    aiDifficulty=5,
                    ringsInHand=18,
                    eliminatedRings=0,
                    territorySpaces=0,
                )
                for i in [1, 2]
            ]

            now = datetime.now()
            states = [GameState(
                id="diagnostic-test",
                boardType=BoardType.SQUARE8,
                rngSeed=None,
                board=board,
                players=players,
                currentPhase=GamePhase.RING_PLACEMENT,
                currentPlayer=1,
                moveHistory=[],
                timeControl=TimeControl(initialTime=600, increment=5, type="standard"),
                gameStatus=GameStatus.ACTIVE,
                createdAt=now,
                lastMoveAt=now,
                isRated=False,
                maxPlayers=2,
                totalRingsInPlay=0,
                totalRingsEliminated=0,
                victoryThreshold=18,  # RR-CANON-R061: = ringsPerPlayer for 2-player games
                territoryVictoryThreshold=33,
                chainCaptureState=None,
                mustMoveFromStackKey=None,
                zobristHash=None,
                lpsRoundIndex=0,
                lpsExclusivePlayerForCompletedRound=None,
            )]
    except FileNotFoundError as e:
        print(f"⚠️  Pool file not found: {e}")
        print("This diagnostic requires the v1 state pool to exist")
        return False

    print(f"Loaded {len(states)} test state(s)")
    print()

    # Use the first state for testing
    test_state = states[0]

    print(f"Test state: Player {test_state.current_player}'s turn")
    print(f"  Phase: {test_state.current_phase}")
    print(f"  Stacks on board: {len(test_state.board.stacks)}")
    print(f"  Markers on board: {len(test_state.board.markers)}")
    print()

    # Test 1: Baseline weights
    print("Test 1: Baseline weights (BASE_V1_BALANCED_WEIGHTS)")
    print("-" * 70)
    baseline_ai = HeuristicAI(1, AIConfig(
        difficulty=5,
        think_time=0,
        randomness=0.0,  # Deterministic for testing
        heuristic_profile_id=None,  # Will infer v1-heuristic-5
    ))
    # Override with explicit baseline weights
    for k, v in BASE_V1_BALANCED_WEIGHTS.items():
        setattr(baseline_ai, k, v)

    baseline_eval = baseline_ai.evaluate_position(test_state)
    baseline_move = baseline_ai.select_move(test_state)

    print(f"  Evaluation: {baseline_eval:.4f}")
    print(f"  Selected move: {baseline_move}")
    print()

    # Test 2: Scaled 2x (all weights doubled)
    print("Test 2: Scaled 2x (all weights doubled)")
    print("-" * 70)
    scaled_ai = HeuristicAI(1, AIConfig(
        difficulty=5,
        think_time=0,
        randomness=0.0,
        heuristic_profile_id=None,
    ))
    for k, v in BASE_V1_BALANCED_WEIGHTS.items():
        setattr(scaled_ai, k, v * 2.0)

    scaled_eval = scaled_ai.evaluate_position(test_state)
    scaled_move = scaled_ai.select_move(test_state)

    print(f"  Evaluation: {scaled_eval:.4f}")
    print(f"  Selected move: {scaled_move}")
    print()

    # Test 3: All zeros (completely random play)
    print("Test 3: All zeros (random play)")
    print("-" * 70)
    zero_ai = HeuristicAI(1, AIConfig(
        difficulty=5,
        think_time=0,
        randomness=0.0,
        heuristic_profile_id=None,
    ))
    for k in BASE_V1_BALANCED_WEIGHTS:
        setattr(zero_ai, k, 0.0)

    zero_eval = zero_ai.evaluate_position(test_state)
    zero_move = zero_ai.select_move(test_state)

    print(f"  Evaluation: {zero_eval:.4f}")
    print(f"  Selected move: {zero_move}")
    print()

    # Test 4: Negative weights (inverted heuristics)
    print("Test 4: Negative weights (inverted heuristics)")
    print("-" * 70)
    negative_ai = HeuristicAI(1, AIConfig(
        difficulty=5,
        think_time=0,
        randomness=0.0,
        heuristic_profile_id=None,
    ))
    for k, v in BASE_V1_BALANCED_WEIGHTS.items():
        setattr(negative_ai, k, -v)

    negative_eval = negative_ai.evaluate_position(test_state)
    negative_move = negative_ai.select_move(test_state)

    print(f"  Evaluation: {negative_eval:.4f}")
    print(f"  Selected move: {negative_move}")
    print()

    # Analysis
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    # Check if evaluations differ
    evals = [baseline_eval, scaled_eval, zero_eval, negative_eval]
    unique_evals = len(set(evals))

    print(f"Unique evaluation scores: {unique_evals} out of 4 tests")
    print(f"  Baseline: {baseline_eval:.4f}")
    print(f"  Scaled:   {scaled_eval:.4f} (diff: {scaled_eval - baseline_eval:+.4f})")
    print(f"  Zero:     {zero_eval:.4f} (diff: {zero_eval - baseline_eval:+.4f})")
    print(f"  Negative: {negative_eval:.4f} (diff: {negative_eval - baseline_eval:+.4f})")
    print()

    # Check if moves differ
    moves = [baseline_move, scaled_move, zero_move, negative_move]
    move_strs = [str(m) for m in moves if m is not None]
    unique_moves = len(set(move_strs))

    print(f"Unique moves selected: {unique_moves} out of 4 tests")
    print()

    # Verdict
    success = True

    if unique_evals == 1:
        print("🚨 CRITICAL BUG: All weights produce IDENTICAL evaluations!")
        print("   This explains the zero variance in CMA-ES training.")
        print("   Weights are NOT affecting position evaluation.")
        success = False
    elif unique_evals == 2 and baseline_eval == scaled_eval:
        print("⚠️  POTENTIAL BUG: Scaling weights has no effect on evaluation")
        print("   This suggests weights are being applied incorrectly.")
        success = False
    else:
        print("✅ PASS: Different weights produce different evaluations")
        print(f"   Found {unique_evals} unique evaluation scores")

    print()

    if unique_moves == 1:
        print("⚠️  WARNING: All weights produce the SAME move choice")
        print("   Even though evaluations differ, move selection converges.")
        print("   This could contribute to low variance but is less critical.")
    else:
        print("✅ PASS: Different weights produce different move selections")
        print(f"   Found {unique_moves} unique moves")

    print()

    # Additional diagnostic: Check if weights are actually set
    print("=" * 70)
    print("WEIGHT SETTING VERIFICATION")
    print("=" * 70)
    print()

    sample_weights = ["WEIGHT_STACK_CONTROL", "WEIGHT_TERRITORY", "WEIGHT_VICTORY_PROXIMITY"]

    for weight_name in sample_weights:
        baseline_val = getattr(baseline_ai, weight_name)
        scaled_val = getattr(scaled_ai, weight_name)
        zero_val = getattr(zero_ai, weight_name)

        print(f"{weight_name}:")
        print(f"  Baseline: {baseline_val:.2f}")
        print(f"  Scaled:   {scaled_val:.2f} (expected: {BASE_V1_BALANCED_WEIGHTS[weight_name] * 2.0:.2f})")
        print(f"  Zero:     {zero_val:.2f} (expected: 0.00)")

        if scaled_val != BASE_V1_BALANCED_WEIGHTS[weight_name] * 2.0:
            print("  ⚠️  Scaled value mismatch!")
            success = False
        if zero_val != 0.0:
            print("  ⚠️  Zero value mismatch!")
            success = False
        print()

    return success


if __name__ == "__main__":
    success = test_weight_application()

    print("=" * 70)
    if success:
        print("✅ DIAGNOSTIC COMPLETE: Weight application is working correctly")
        print()
        print("The zero-variance issue is likely caused by:")
        print("  - Population size too small (2)")
        print("  - Games per eval too small (4)")
        print("  - Randomness too low (0.02)")
        print()
        print("Proceed to Task 2: Retry with larger parameters")
        sys.exit(0)
    else:
        print("🚨 DIAGNOSTIC FAILED: Weight application bug detected")
        print()
        print("The zero-variance issue is caused by weights not affecting gameplay.")
        print("Review the weight application code in:")
        print("  - scripts/run_cmaes_optimization.py::create_heuristic_ai_with_weights()")
        print("  - app/ai/heuristic_ai.py::_apply_weight_profile()")
        sys.exit(1)
