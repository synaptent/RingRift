#!/usr/bin/env python
"""A/B Test: Compare NNUE Policy Models.

This script runs head-to-head matches between different policy models to validate
improvements. Useful for:
1. Comparing new policy model against baseline
2. Validating distillation quality
3. Comparing different model sizes (small/medium/large)
4. Curriculum stage advancement decisions

Usage:
    # Compare two policy models
    python scripts/ab_test_policy_models.py \
        --model-a models/nnue/nnue_policy_square8_2p.pt \
        --model-b models/nnue/nnue_policy_square8_2p_new.pt \
        --num-games 100

    # Compare against no policy (baseline heuristic)
    python scripts/ab_test_policy_models.py \
        --model-a models/nnue/nnue_policy_square8_2p.pt \
        --model-b none \
        --num-games 100

    # Quick test
    python scripts/ab_test_policy_models.py --quick

Output:
    - JSON report with win rates, confidence intervals, and statistical analysis
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.mcts_ai import MCTSAI
from app.models import AIConfig, BoardType, GameStatus
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("ab_test_policy_models")


@dataclass
class MatchResult:
    """Result of a single match."""
    game_id: str
    winner: Optional[int]  # 1 or 2, None for draw
    model_a_player: int  # Which player number model A was
    model_b_player: int
    num_moves: int
    game_duration_sec: float
    model_a_think_time_sec: float
    model_b_think_time_sec: float


@dataclass
class ABTestResults:
    """Results of policy A/B test."""
    model_a_path: str
    model_b_path: str
    board_type: str
    num_players: int
    num_games: int

    # Win/loss/draw counts
    model_a_wins: int = 0
    model_b_wins: int = 0
    draws: int = 0

    # Win rates
    model_a_win_rate: float = 0.0
    model_b_win_rate: float = 0.0
    draw_rate: float = 0.0

    # Statistical analysis
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)
    p_value: float = 1.0
    significant_at_95: bool = False

    # Game statistics
    avg_game_length: float = 0.0
    avg_model_a_think_time: float = 0.0
    avg_model_b_think_time: float = 0.0

    # Individual match results
    matches: List[Dict[str, Any]] = field(default_factory=list)

    def compute_statistics(self):
        """Compute win rates and statistical significance."""
        total = self.model_a_wins + self.model_b_wins + self.draws
        if total == 0:
            return

        self.model_a_win_rate = self.model_a_wins / total
        self.model_b_win_rate = self.model_b_wins / total
        self.draw_rate = self.draws / total

        # Compute 95% confidence interval for model A win rate
        # Using Wilson score interval for better small-sample behavior
        n = total
        p = self.model_a_win_rate
        z = 1.96  # 95% confidence

        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        spread = z * math.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denominator

        self.confidence_interval_95 = (
            max(0, center - spread),
            min(1, center + spread)
        )

        # Compute p-value using binomial test (one-sided)
        # H0: model A win rate = 0.5
        # H1: model A win rate > 0.5
        try:
            from scipy import stats
            # Use number of wins vs losses (excluding draws)
            decisive = self.model_a_wins + self.model_b_wins
            if decisive > 0:
                result = stats.binomtest(
                    self.model_a_wins, decisive, p=0.5, alternative='greater'
                )
                self.p_value = result.pvalue
                self.significant_at_95 = self.p_value < 0.05
        except ImportError:
            # Fallback: approximate z-test
            decisive = self.model_a_wins + self.model_b_wins
            if decisive > 0:
                observed_rate = self.model_a_wins / decisive
                z_score = (observed_rate - 0.5) / math.sqrt(0.25 / decisive)
                # One-sided p-value approximation
                self.p_value = 0.5 * (1 - math.erf(z_score / math.sqrt(2)))
                self.significant_at_95 = self.p_value < 0.05


def create_mcts_ai(
    player_number: int,
    policy_model_path: Optional[str],
    think_time_ms: int = 500,
    board_type: BoardType = BoardType.SQUARE8,
    policy_temperature: float = 2.0,
    prior_uniform_mix: float = 0.3,
) -> MCTSAI:
    """Create MCTS AI with optional policy model."""
    config = AIConfig(
        difficulty=4,
        think_time=think_time_ms,
        use_neural_net=False,  # Don't use full neural net
        use_nnue_policy_priors=(policy_model_path is not None),
    )
    ai = MCTSAI(player_number=player_number, config=config)

    # Apply policy temperature/mixing settings
    ai.policy_temperature = policy_temperature
    ai.prior_uniform_mix = prior_uniform_mix

    # Manually load policy model if specified
    if policy_model_path and os.path.exists(policy_model_path):
        try:
            from app.ai.nnue_policy import RingRiftNNUEWithPolicy
            import torch
            import re

            checkpoint = torch.load(policy_model_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint
            hidden_dim = 256
            num_hidden_layers = 2

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                hidden_dim = int(checkpoint.get("hidden_dim") or hidden_dim)
                num_hidden_layers = int(checkpoint.get("num_hidden_layers") or num_hidden_layers)

            # Infer from state dict
            if isinstance(state_dict, dict):
                try:
                    accumulator_weight = state_dict.get("accumulator.weight")
                    if accumulator_weight is not None and hasattr(accumulator_weight, "shape"):
                        hidden_dim = int(accumulator_weight.shape[0])
                except Exception:
                    pass

                try:
                    layer_indices = set()
                    for key in state_dict:
                        match = re.match(r"hidden\.(\d+)\.weight$", key)
                        if match:
                            layer_indices.add(int(match.group(1)))
                    if layer_indices:
                        num_hidden_layers = len(layer_indices)
                except Exception:
                    pass

            ai.nnue_policy_model = RingRiftNNUEWithPolicy(
                board_type=board_type,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
            )
            if not isinstance(state_dict, dict):
                raise TypeError(f"Unexpected checkpoint: {type(state_dict).__name__}")
            ai.nnue_policy_model.load_state_dict(state_dict)
            ai.nnue_policy_model.eval()
            ai._pending_nnue_policy_init = False
            logger.debug(f"Loaded policy model: {policy_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load policy model {policy_model_path}: {e}")

    return ai


def play_match(
    engine: GameEngine,
    ai_a: MCTSAI,
    ai_b: MCTSAI,
    model_a_player: int,
    max_moves: int = 10000,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
) -> MatchResult:
    """Play a single match between two AIs."""
    game_state = create_initial_state(board_type, num_players)
    game_id = game_state.id

    model_a_think_time = 0.0
    model_b_think_time = 0.0
    start_time = time.time()
    num_moves = 0

    while game_state.game_status == GameStatus.ACTIVE and num_moves < max_moves:
        current_player = game_state.current_player

        # Select AI based on player
        if current_player == model_a_player:
            ai = ai_a
            ai.player_number = current_player
        else:
            ai = ai_b
            ai.player_number = current_player

        # Get move
        move_start = time.time()
        try:
            move = ai.select_move(game_state)
            move_time = time.time() - move_start

            if current_player == model_a_player:
                model_a_think_time += move_time
            else:
                model_b_think_time += move_time

            if move is None:
                break

            game_state = engine.apply_move(game_state, move)
            num_moves += 1
        except Exception as e:
            logger.warning(f"Move failed: {e}")
            break

    game_duration = time.time() - start_time

    # Use the game state's winner field (set by game engine on completion)
    winner = game_state.winner

    return MatchResult(
        game_id=game_id,
        winner=winner,
        model_a_player=model_a_player,
        model_b_player=3 - model_a_player,  # 2 if a is 1, 1 if a is 2
        num_moves=num_moves,
        game_duration_sec=game_duration,
        model_a_think_time_sec=model_a_think_time,
        model_b_think_time_sec=model_b_think_time,
    )


def run_ab_test(
    model_a_path: str,
    model_b_path: Optional[str],
    num_games: int,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    think_time_ms: int = 500,
    max_moves: int = 2000,
    policy_temperature: float = 2.0,
    prior_uniform_mix: float = 0.3,
) -> ABTestResults:
    """Run A/B test between two policy models."""
    logger.info(f"Starting A/B test:")
    logger.info(f"  Model A: {model_a_path}")
    logger.info(f"  Model B: {model_b_path or 'none (baseline)'}")
    logger.info(f"  Games: {num_games}")

    engine = GameEngine()
    results = ABTestResults(
        model_a_path=model_a_path,
        model_b_path=model_b_path or "none",
        board_type=board_type.value,
        num_players=num_players,
        num_games=num_games,
    )

    total_game_length = 0
    total_model_a_time = 0.0
    total_model_b_time = 0.0

    for i in range(num_games):
        # Alternate which player model A is to eliminate first-move advantage
        model_a_player = 1 if i % 2 == 0 else 2

        # Create fresh AIs for each game
        ai_a = create_mcts_ai(
            model_a_player, model_a_path, think_time_ms, board_type,
            policy_temperature, prior_uniform_mix
        )
        ai_b = create_mcts_ai(
            3 - model_a_player, model_b_path, think_time_ms, board_type,
            policy_temperature, prior_uniform_mix
        )

        # Play match
        match_result = play_match(
            engine, ai_a, ai_b, model_a_player, max_moves,
            board_type, num_players
        )

        # Record result
        if match_result.winner is None:
            results.draws += 1
        elif match_result.winner == model_a_player:
            results.model_a_wins += 1
        else:
            results.model_b_wins += 1

        total_game_length += match_result.num_moves
        total_model_a_time += match_result.model_a_think_time_sec
        total_model_b_time += match_result.model_b_think_time_sec

        results.matches.append(asdict(match_result))

        # Progress update
        if (i + 1) % 10 == 0:
            logger.info(
                f"Progress: {i+1}/{num_games} games | "
                f"A wins: {results.model_a_wins} | "
                f"B wins: {results.model_b_wins} | "
                f"Draws: {results.draws}"
            )

    # Compute final statistics
    results.avg_game_length = total_game_length / num_games
    results.avg_model_a_think_time = total_model_a_time / num_games
    results.avg_model_b_think_time = total_model_b_time / num_games
    results.compute_statistics()

    return results


def run_multi_time_test(
    model_a_path: str,
    model_b_path: Optional[str],
    num_games: int,
    board_type: BoardType,
    think_times: List[int],
    max_moves: int = 300,
    output_path: Optional[str] = None,
    policy_temperature: float = 2.0,
    prior_uniform_mix: float = 0.3,
) -> int:
    """Run A/B tests at multiple think times.

    Policy models typically benefit more at shorter think times where MCTS
    has less time to search. This function runs tests at each think time
    and outputs a comparison table.

    Returns:
        Exit code (0 for success)
    """
    logger.info("=" * 60)
    logger.info("MULTI-TIME A/B TEST")
    logger.info("=" * 60)
    logger.info(f"Model A: {model_a_path}")
    logger.info(f"Model B: {model_b_path or 'baseline (no policy)'}")
    logger.info(f"Games per think time: {num_games}")
    logger.info(f"Think times: {think_times} ms")
    logger.info("")

    all_results = {}

    for think_time in think_times:
        logger.info(f"Testing at {think_time}ms think time...")

        results = run_ab_test(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            num_games=num_games,
            board_type=board_type,
            think_time_ms=think_time,
            max_moves=max_moves,
            policy_temperature=policy_temperature,
            prior_uniform_mix=prior_uniform_mix,
        )

        all_results[think_time] = results
        logger.info(f"  Model A win rate: {results.model_a_win_rate:.1%} "
                   f"(95% CI: [{results.confidence_interval_95[0]:.1%}, {results.confidence_interval_95[1]:.1%}])")

    # Print comparison table
    print("\n" + "=" * 80)
    print("MULTI-TIME A/B TEST RESULTS")
    print("=" * 80)
    print(f"Model A: {model_a_path}")
    print(f"Model B: {model_b_path or 'baseline (no policy)'}")
    print(f"Games per think time: {num_games}")
    print()

    # Table header
    print(f"{'Think Time':>12} | {'A Wins':>8} | {'B Wins':>8} | {'Draws':>8} | {'A Win%':>10} | {'95% CI':>20} | {'Sig?':>5}")
    print("-" * 80)

    for think_time in think_times:
        r = all_results[think_time]
        ci_str = f"[{r.confidence_interval_95[0]:.1%}, {r.confidence_interval_95[1]:.1%}]"
        sig = "YES" if r.significant_at_95 else "no"
        print(f"{think_time:>10}ms | {r.model_a_wins:>8} | {r.model_b_wins:>8} | {r.draws:>8} | "
              f"{r.model_a_win_rate:>9.1%} | {ci_str:>20} | {sig:>5}")

    print("-" * 80)
    print()

    # Summary
    best_time = max(think_times, key=lambda t: all_results[t].model_a_win_rate)
    worst_time = min(think_times, key=lambda t: all_results[t].model_a_win_rate)

    print(f"Best performance for A: {all_results[best_time].model_a_win_rate:.1%} at {best_time}ms")
    print(f"Worst performance for A: {all_results[worst_time].model_a_win_rate:.1%} at {worst_time}ms")

    # Check if policy benefits more at shorter think times (expected for policy models)
    if len(think_times) >= 2:
        shortest = min(think_times)
        longest = max(think_times)
        diff = all_results[shortest].model_a_win_rate - all_results[longest].model_a_win_rate
        if diff > 0.05:
            print(f"Policy shows {diff:.1%} stronger advantage at shorter think times (expected)")
        elif diff < -0.05:
            print(f"Policy shows {abs(diff):.1%} weaker advantage at shorter think times (unexpected)")

    print("=" * 80)

    # Save to file if requested
    if output_path:
        # Convert to serializable format
        report = {
            "model_a_path": model_a_path,
            "model_b_path": model_b_path,
            "num_games_per_time": num_games,
            "board_type": board_type.value,
            "timestamp": datetime.now().isoformat(),
            "results_by_think_time": {}
        }

        for think_time, r in all_results.items():
            # Convert numpy types
            def convert_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy(v) for v in obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj

            report["results_by_think_time"][think_time] = convert_numpy(asdict(r))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to: {output_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="A/B test NNUE policy models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model-a",
        type=str,
        default="models/nnue/nnue_policy_square8_2p.pt",
        help="Path to policy model A (default: models/nnue/nnue_policy_square8_2p.pt)",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default=None,
        help="Path to policy model B (default: none = baseline without policy)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=50,
        help="Number of games to play (default: 50)",
    )
    parser.add_argument(
        "--think-time",
        type=int,
        default=500,
        help="Think time per move in ms (default: 500)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=300,
        help="Maximum moves per game (default: 300)",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type (default: square8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer games (10)",
    )
    parser.add_argument(
        "--multi-time",
        action="store_true",
        help="Run tests at multiple think times (50ms, 100ms, 200ms, 500ms). "
             "Policy models typically benefit more at shorter think times. "
             "Output includes results for each think time.",
    )
    parser.add_argument(
        "--multi-time-values",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500],
        help="Think times (ms) to test with --multi-time (default: 50 100 200 500)",
    )
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=1.5,
        help="Temperature for policy softmax (higher = flatter distribution). Default: 1.5",
    )
    parser.add_argument(
        "--prior-uniform-mix",
        type=float,
        default=0.5,
        help="Mix ratio with uniform distribution (0=pure policy, 1=pure uniform). Default: 0.5",
    )

    args = parser.parse_args()

    if args.quick:
        args.num_games = 10
        args.think_time = 500  # 500ms for quick mode (still fast but games complete)

    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }
    board_type = board_type_map[args.board_type]

    # Check model A exists
    if not os.path.exists(args.model_a):
        logger.error(f"Model A not found: {args.model_a}")
        return 1

    # Multi-time test mode
    if args.multi_time:
        return run_multi_time_test(
            model_a_path=args.model_a,
            model_b_path=args.model_b,
            num_games=args.num_games,
            board_type=board_type,
            think_times=args.multi_time_values,
            max_moves=args.max_moves,
            output_path=args.output,
            policy_temperature=args.policy_temperature,
            prior_uniform_mix=args.prior_uniform_mix,
        )

    # Run single test
    results = run_ab_test(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        num_games=args.num_games,
        board_type=board_type,
        think_time_ms=args.think_time,
        max_moves=args.max_moves,
        policy_temperature=args.policy_temperature,
        prior_uniform_mix=args.prior_uniform_mix,
    )

    # Output results - convert numpy types to Python natives for JSON serialization
    def convert_numpy(obj):
        """Recursively convert numpy types to Python natives."""
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy(v) for v in obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj

    report = convert_numpy(asdict(results))
    report_json = json.dumps(report, indent=2)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report_json)
        logger.info(f"Report saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("A/B TEST RESULTS")
        print("=" * 60)
        print(f"Model A: {results.model_a_path}")
        print(f"Model B: {results.model_b_path}")
        print(f"Games: {results.num_games}")
        print()
        print(f"Model A wins: {results.model_a_wins} ({results.model_a_win_rate:.1%})")
        print(f"Model B wins: {results.model_b_wins} ({results.model_b_win_rate:.1%})")
        print(f"Draws: {results.draws} ({results.draw_rate:.1%})")
        print()
        print(f"95% CI for A win rate: [{results.confidence_interval_95[0]:.1%}, {results.confidence_interval_95[1]:.1%}]")
        print(f"P-value (A > B): {results.p_value:.4f}")
        print(f"Significant at 95%: {results.significant_at_95}")
        print()
        print(f"Avg game length: {results.avg_game_length:.1f} moves")
        print(f"Avg Model A think time: {results.avg_model_a_think_time:.2f}s")
        print(f"Avg Model B think time: {results.avg_model_b_think_time:.2f}s")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
