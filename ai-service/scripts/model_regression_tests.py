#!/usr/bin/env python3
"""Model performance regression tests for RingRift AI.

Ensures new models don't regress against baseline difficulty tiers.
Runs automated checks after model promotions.

Usage:
    python scripts/model_regression_tests.py --model models/ringrift_best_sq8_2p.pth
    python scripts/model_regression_tests.py --all --threshold 0.45
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RegressionTest:
    """Definition of a regression test."""
    name: str
    opponent_type: str  # 'heuristic', 'minimax', 'nn'
    opponent_config: Dict
    min_win_rate: float
    games: int
    board_type: str = "square8"
    num_players: int = 2


# Baseline regression tests - model must beat these opponents
REGRESSION_TESTS = [
    # Must beat random easily
    RegressionTest(
        name="vs_random",
        opponent_type="random",
        opponent_config={},
        min_win_rate=0.95,
        games=20,
    ),
    # Must beat D2 heuristic
    RegressionTest(
        name="vs_d2_heuristic",
        opponent_type="heuristic",
        opponent_config={"difficulty": 2},
        min_win_rate=0.80,
        games=30,
    ),
    # Must beat D4 minimax
    RegressionTest(
        name="vs_d4_minimax",
        opponent_type="minimax",
        opponent_config={"difficulty": 4, "depth": 3},
        min_win_rate=0.60,
        games=30,
    ),
    # Shouldn't lose too badly to D6
    RegressionTest(
        name="vs_d6_minimax",
        opponent_type="minimax",
        opponent_config={"difficulty": 6, "depth": 4},
        min_win_rate=0.40,
        games=30,
    ),
]


def run_regression_test(
    model_path: str,
    test: RegressionTest,
) -> Dict:
    """Run a single regression test.

    Returns:
        Dict with test results
    """
    from app.rules import create_game_state, apply_action, is_game_over, get_winner
    from app.ai.heuristic_ai import HeuristicAI
    from app.models import BoardType

    print(f"  Running {test.name}: {test.games} games, min_win_rate={test.min_win_rate:.0%}")

    # Create test opponent
    if test.opponent_type == "random":
        from app.ai.random_ai import RandomAI
        opponent = RandomAI()
    elif test.opponent_type == "heuristic":
        opponent = HeuristicAI(difficulty=test.opponent_config.get("difficulty", 5))
    elif test.opponent_type == "minimax":
        from app.ai.minimax import MinimaxAI
        opponent = MinimaxAI(
            difficulty=test.opponent_config.get("difficulty", 5),
            max_depth=test.opponent_config.get("depth", 3),
        )
    else:
        raise ValueError(f"Unknown opponent type: {test.opponent_type}")

    # Load model being tested
    import torch
    from app.ai.neural_net import NeuralNetAI

    test_ai = NeuralNetAI(
        difficulty=10,
        board_type=test.board_type,
        num_players=test.num_players,
    )
    # Load specific model
    test_ai._load_model(model_path)

    # Run games
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0

    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }
    board_type = board_type_map.get(test.board_type, BoardType.SQUARE8)

    for game_idx in range(test.games):
        # Alternate sides
        test_player = game_idx % 2

        state = create_game_state(
            board_type=board_type,
            num_players=test.num_players,
        )

        move_count = 0
        max_moves = 300

        while not is_game_over(state) and move_count < max_moves:
            current_player = state.current_player_index

            if current_player == test_player:
                action = test_ai.get_move(state)
            else:
                action = opponent.get_move(state)

            state = apply_action(state, action)
            move_count += 1

        total_moves += move_count

        winner = get_winner(state)
        if winner == test_player:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1

    # Calculate results
    total_games = wins + losses + draws
    win_rate = wins / total_games if total_games > 0 else 0
    avg_moves = total_moves / total_games if total_games > 0 else 0

    passed = win_rate >= test.min_win_rate

    return {
        "name": test.name,
        "passed": passed,
        "win_rate": round(win_rate, 4),
        "min_win_rate": test.min_win_rate,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": total_games,
        "avg_moves": round(avg_moves, 1),
        "margin": round(win_rate - test.min_win_rate, 4),
    }


def run_all_tests(
    model_path: str,
    tests: List[RegressionTest] = REGRESSION_TESTS,
) -> Dict:
    """Run all regression tests on a model.

    Returns:
        Dict with overall results and per-test details
    """
    print(f"Running regression tests on: {model_path}")
    print("=" * 60)

    results = []
    all_passed = True
    start_time = time.time()

    for test in tests:
        result = run_regression_test(model_path, test)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"    {status}: win_rate={result['win_rate']:.0%} "
              f"(min={result['min_win_rate']:.0%}, margin={result['margin']:+.0%})")

        if not result["passed"]:
            all_passed = False

    duration = time.time() - start_time

    print("=" * 60)
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    print(f"Tests: {sum(1 for r in results if r['passed'])}/{len(results)} passed")
    print(f"Duration: {duration:.1f}s")

    return {
        "model_path": model_path,
        "all_passed": all_passed,
        "tests_passed": sum(1 for r in results if r["passed"]),
        "tests_total": len(results),
        "duration_seconds": round(duration, 1),
        "results": results,
    }


def find_models_to_test() -> List[Path]:
    """Find all best models to test."""
    models_dir = AI_SERVICE_ROOT / "models"
    return list(models_dir.glob("ringrift_best_*.pth"))


def main():
    parser = argparse.ArgumentParser(description="Model regression tests")
    parser.add_argument("--model", help="Model path to test")
    parser.add_argument("--all", action="store_true", help="Test all best models")
    parser.add_argument("--threshold", type=float, help="Override min win rate threshold")
    parser.add_argument("--games", type=int, help="Override games per test")
    parser.add_argument("--output", help="Output JSON file")

    args = parser.parse_args()

    # Modify tests if threshold/games overridden
    tests = REGRESSION_TESTS.copy()
    if args.threshold:
        for test in tests:
            test.min_win_rate = args.threshold
    if args.games:
        for test in tests:
            test.games = args.games

    all_results = []

    if args.all:
        models = find_models_to_test()
        if not models:
            print("No models found to test")
            return 1

        for model_path in models:
            print(f"\n{'='*60}")
            results = run_all_tests(str(model_path), tests)
            all_results.append(results)

    elif args.model:
        results = run_all_tests(args.model, tests)
        all_results.append(results)

    else:
        parser.print_help()
        return 1

    # Summary
    print(f"\n{'='*60}")
    print("REGRESSION TEST SUMMARY")
    print("=" * 60)

    total_passed = sum(1 for r in all_results if r["all_passed"])
    print(f"Models: {total_passed}/{len(all_results)} passed all tests")

    for result in all_results:
        status = "PASS" if result["all_passed"] else "FAIL"
        model_name = Path(result["model_path"]).name
        print(f"  {status}: {model_name} ({result['tests_passed']}/{result['tests_total']} tests)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Return non-zero if any model failed
    return 0 if all(r["all_passed"] for r in all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
