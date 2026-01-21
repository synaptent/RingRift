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
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RegressionTest:
    """Definition of a regression test."""
    name: str
    opponent_type: str  # 'heuristic', 'minimax', 'nn'
    opponent_config: dict
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

# Quick tests for fast sanity checks (fewer games, relaxed thresholds)
QUICK_REGRESSION_TESTS = [
    RegressionTest(
        name="vs_random_quick",
        opponent_type="random",
        opponent_config={},
        min_win_rate=0.80,  # Relaxed for small sample
        games=5,
    ),
    RegressionTest(
        name="vs_d2_heuristic_quick",
        opponent_type="heuristic",
        opponent_config={"difficulty": 2},
        min_win_rate=0.60,  # Relaxed for small sample
        games=5,
    ),
]


def run_regression_test(
    model_path: str,
    test: RegressionTest,
) -> dict:
    """Run a single regression test.

    Returns:
        Dict with test results
    """
    import os
    from datetime import datetime

    import torch

    from app.ai.descent_ai import DescentAI
    from app.ai.heuristic_ai import HeuristicAI
    from app.game_engine import GameEngine
    from app.models import AIConfig, BoardState, BoardType, GamePhase, GameState, GameStatus, Player, TimeControl
    from app.rules.core import (
        BOARD_CONFIGS,
        get_territory_victory_threshold,
        get_victory_threshold,
    )

    print(f"  Running {test.name}: {test.games} games, min_win_rate={test.min_win_rate:.0%}")

    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }
    board_type = board_type_map.get(test.board_type, BoardType.SQUARE8)

    def create_initial_state() -> GameState:
        """Create initial game state."""
        if board_type in BOARD_CONFIGS:
            config = BOARD_CONFIGS[board_type]
            size = config.size
            rings_per_player = config.rings_per_player
        else:
            size = 8
            rings_per_player = 18

        players = [
            Player(
                id=f"p{idx}",
                username=f"AI {idx}",
                type="ai",
                playerNumber=idx,
                isReady=True,
                timeRemaining=600,
                ringsInHand=rings_per_player,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=10,
            )
            for idx in range(1, test.num_players + 1)
        ]

        total_rings = rings_per_player * test.num_players
        victory_threshold = get_victory_threshold(board_type, test.num_players)
        territory_threshold = get_territory_victory_threshold(board_type)

        return GameState(
            id="regression_test",
            boardType=board_type,
            rngSeed=None,
            board=BoardState(
                type=board_type,
                size=size,
                stacks={},
                markers={},
                collapsedSpaces={},
                eliminatedRings={},
            ),
            players=players,
            currentPhase=GamePhase.RING_PLACEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=600,
                increment=0,
                type="blitz",
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt=datetime.now(),
            lastMoveAt=datetime.now(),
            isRated=False,
            maxPlayers=test.num_players,
            totalRingsInPlay=total_rings,
            totalRingsEliminated=0,
            victoryThreshold=victory_threshold,
            territoryVictoryThreshold=territory_threshold,
            chainCaptureState=None,
            mustMoveFromStackKey=None,
            zobristHash=None,
            lpsRoundIndex=0,
            lpsExclusivePlayerForCompletedRound=None,
        )

    def create_opponent(player_num: int, game_idx: int = 0):
        """Create test opponent.

        Args:
            player_num: Player number (1-indexed)
            game_idx: Game index for unique RNG seeding (Jan 2026 fix).
                     Without this, RandomAI/HeuristicAI produce identical games.
        """
        # Per-game seed for varied randomness
        rng_seed = (game_idx * 10000 + player_num * 1000) & 0xFFFFFFFF
        config = AIConfig(
            difficulty=test.opponent_config.get("difficulty", 5),
            randomness=0.0,
            think_time=100,
            rng_seed=rng_seed,
        )
        if test.opponent_type == "random":
            from app.ai.random_ai import RandomAI
            return RandomAI(player_num, config)
        elif test.opponent_type == "heuristic":
            return HeuristicAI(player_num, config)
        elif test.opponent_type == "minimax":
            from app.ai.minimax_ai import MinimaxAI
            config.think_time = test.opponent_config.get("depth", 3) * 100
            return MinimaxAI(player_num, config)
        else:
            raise ValueError(f"Unknown opponent type: {test.opponent_type}")

    def create_test_ai(player_num: int) -> DescentAI:
        """Create AI with the model being tested."""
        model_id = os.path.splitext(os.path.basename(model_path))[0]
        config = AIConfig(
            difficulty=10,
            randomness=0.1,
            think_time=500,
            nn_model_id=model_id,
        )
        ai = DescentAI(player_num, config)

        # Trigger lazy initialization by providing board type
        if ai.neural_net:
            try:
                # Initialize the model architecture for this board type
                ai.neural_net._ensure_model_initialized(board_type, test.num_players)

                # Now load specific model weights if different from default
                if os.path.exists(model_path) and ai.neural_net.model is not None:
                    from app.utils.torch_utils import safe_load_checkpoint
                    checkpoint = safe_load_checkpoint(model_path, map_location="cpu")
                    # Handle both raw state_dict and checkpoint format
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        ai.neural_net.model.load_state_dict(checkpoint["model_state_dict"])
                    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        ai.neural_net.model.load_state_dict(checkpoint["state_dict"])
                    else:
                        # Assume it's a raw state dict
                        ai.neural_net.model.load_state_dict(checkpoint)
                    ai.neural_net.model.eval()
            except Exception as e:
                print(f"    Warning: Failed to load model {model_path}: {e}")

        return ai

    # Run games
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0

    for game_idx in range(test.games):
        # Alternate sides (1-indexed player numbers)
        test_player = (game_idx % 2) + 1

        state = create_initial_state()

        # Create AIs (pass game_idx for unique seeding - Jan 2026 fix)
        if test_player == 1:
            ais = {1: create_test_ai(1), 2: create_opponent(2, game_idx)}
        else:
            ais = {1: create_opponent(1, game_idx), 2: create_test_ai(2)}

        move_count = 0
        max_moves = 300

        while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
            current = state.current_player
            ai = ais.get(current)
            if ai is None:
                break

            try:
                move = ai.select_move(state)
                if move:
                    state = GameEngine.apply_move(state, move)
                else:
                    # No move available - handle phase requirement
                    req = GameEngine.get_phase_requirement(state)
                    if req:
                        from app.game_engine import PhaseRequirementType
                        from app.models import Move, MoveType
                        if req.type == PhaseRequirementType.NO_PLACEMENT_ACTION_REQUIRED:
                            state = GameEngine.apply_move(state, Move(
                                type=MoveType.NO_PLACEMENT_ACTION,
                                playerId=f"p{req.player}",
                                playerNumber=req.player,
                            ))
                        elif req.type == PhaseRequirementType.NO_MOVEMENT_ACTION_REQUIRED:
                            state = GameEngine.apply_move(state, Move(
                                type=MoveType.NO_MOVEMENT_ACTION,
                                playerId=f"p{req.player}",
                                playerNumber=req.player,
                            ))
                        else:
                            break
                    else:
                        break
            except (ValueError, RuntimeError, AttributeError):
                break

            move_count += 1

        total_moves += move_count

        winner = state.winner
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
    tests: list[RegressionTest] = REGRESSION_TESTS,
) -> dict:
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


def find_models_to_test() -> list[Path]:
    """Find all best models to test."""
    models_dir = AI_SERVICE_ROOT / "models"
    return list(models_dir.glob("ringrift_best_*.pth"))


def main():
    parser = argparse.ArgumentParser(description="Model regression tests")
    parser.add_argument("--model", help="Model path to test")
    parser.add_argument("--all", action="store_true", help="Test all best models")
    parser.add_argument("--quick", action="store_true", help="Run quick tests (fewer games)")
    parser.add_argument("--threshold", type=float, help="Override min win rate threshold")
    parser.add_argument("--games", type=int, help="Override games per test")
    parser.add_argument("--output", help="Output JSON file")

    args = parser.parse_args()

    # Select test suite
    if args.quick:
        tests = [RegressionTest(**asdict(t)) for t in QUICK_REGRESSION_TESTS]
    else:
        tests = [RegressionTest(**asdict(t)) for t in REGRESSION_TESTS]

    # Modify tests if threshold/games overridden
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
