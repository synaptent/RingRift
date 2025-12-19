#!/usr/bin/env python3
"""
Model Elo Comparison Pipeline

Compares two models head-to-head to determine Elo difference.
Used to validate training improvements.

Usage:
    # Compare new model vs current best
    python scripts/compare_models_elo.py \\
        --model-a models/nnue/new_model.pt \\
        --model-b models/nnue/square8_2p_best.pt \\
        --games 100 \\
        --board-type square8 --num-players 2

    # Quick comparison (fewer games)
    python scripts/compare_models_elo.py --model-a new.pt --model-b best.pt --games 20 --quick
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.logging_config import (
    setup_script_logging,
    get_logger,
    get_metrics_logger,
)
from scripts.lib.config import BoardConfig

logger = get_logger(__name__)


@dataclass
class MatchResult:
    """Result of a single game."""
    game_id: int
    winner: Optional[int]  # 1 = model_a, 2 = model_b, None = draw
    moves: int
    time_seconds: float
    model_a_color: int  # Which player model_a was (1 or 2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonResult:
    """Result of full comparison."""
    model_a_path: str
    model_b_path: str
    board_type: str
    num_players: int
    total_games: int
    model_a_wins: int
    model_b_wins: int
    draws: int
    model_a_win_rate: float
    elo_difference: float
    confidence_interval: Tuple[float, float]
    avg_game_length: float
    total_time: float

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["confidence_interval"] = list(self.confidence_interval)
        return result

    def is_significant_improvement(self, threshold: float = 50.0) -> bool:
        """Check if model A is significantly better."""
        return self.elo_difference > threshold

    def is_regression(self, threshold: float = -20.0) -> bool:
        """Check if model A is significantly worse."""
        return self.elo_difference < threshold


class EloCalculator:
    """Elo rating calculations."""

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def elo_diff_from_win_rate(win_rate: float) -> float:
        """Calculate Elo difference from win rate."""
        if win_rate <= 0:
            return -400
        if win_rate >= 1:
            return 400
        return -400 * math.log10(1.0 / win_rate - 1.0)

    @staticmethod
    def wilson_confidence_interval(
        wins: int,
        total: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval for win rate."""
        if total == 0:
            return (0.0, 1.0)

        z = 1.96 if confidence == 0.95 else 1.645  # 95% or 90%
        p = wins / total

        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

        return (max(0, center - spread), min(1, center + spread))


def play_single_game(
    model_a_path: str,
    model_b_path: str,
    board_type: str,
    num_players: int,
    game_id: int,
    model_a_player: int,
    mcts_simulations: int = 100,
) -> MatchResult:
    """Play a single game between two models.

    Args:
        model_a_path: Path to model A
        model_b_path: Path to model B
        board_type: Type of board
        num_players: Number of players
        game_id: Game identifier
        model_a_player: Which player (1 or 2) model_a plays as
        mcts_simulations: Number of MCTS simulations per move

    Returns:
        MatchResult with game outcome
    """
    from app.game_engine import GameEngine
    from app.ai.nnue import NNUEEvaluator
    from app.ai.mcts_ai import MCTSAI
    from app.training.generate_data import create_initial_state
    from app.models import AIConfig, BoardType, GameStatus

    start_time = time.time()

    # Load models
    evaluator_a = NNUEEvaluator(model_a_path, board_type=BoardType(board_type))
    evaluator_b = NNUEEvaluator(model_b_path, board_type=BoardType(board_type))

    # Create AIs
    config = AIConfig(difficulty=5, think_time=1000)

    if model_a_player == 1:
        ai_1 = MCTSAI(1, config, evaluator=evaluator_a, simulations=mcts_simulations)
        ai_2 = MCTSAI(2, config, evaluator=evaluator_b, simulations=mcts_simulations)
    else:
        ai_1 = MCTSAI(1, config, evaluator=evaluator_b, simulations=mcts_simulations)
        ai_2 = MCTSAI(2, config, evaluator=evaluator_a, simulations=mcts_simulations)

    # Create game
    state = create_initial_state(BoardType(board_type), num_players=num_players)
    engine = GameEngine()

    # Play game
    move_count = 0
    max_moves = 500

    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_ai = ai_1 if state.current_player == 1 else ai_2
        move = current_ai.select_move(state)

        if move is None:
            break

        state = engine.apply_move(state, move)
        move_count += 1

    # Determine winner relative to model_a
    winner = None
    if state.winner is not None:
        if (state.winner == 1 and model_a_player == 1) or \
           (state.winner == 2 and model_a_player == 2):
            winner = 1  # model_a won
        else:
            winner = 2  # model_b won

    elapsed = time.time() - start_time

    return MatchResult(
        game_id=game_id,
        winner=winner,
        moves=move_count,
        time_seconds=elapsed,
        model_a_color=model_a_player,
    )


class ModelComparator:
    """Compares two models via head-to-head games."""

    def __init__(
        self,
        model_a_path: str,
        model_b_path: str,
        board_type: str = "square8",
        num_players: int = 2,
        mcts_simulations: int = 100,
        parallel_games: int = 4,
    ):
        """Initialize the comparator.

        Args:
            model_a_path: Path to first model
            model_b_path: Path to second model (baseline)
            board_type: Board type for games
            num_players: Number of players
            mcts_simulations: MCTS simulations per move
            parallel_games: Number of games to run in parallel
        """
        self.model_a_path = model_a_path
        self.model_b_path = model_b_path
        self.board_type = board_type
        self.num_players = num_players
        self.mcts_simulations = mcts_simulations
        self.parallel_games = parallel_games
        self.metrics = get_metrics_logger("model_comparison", log_interval=60)
        self.calculator = EloCalculator()

    def run(self, num_games: int) -> ComparisonResult:
        """Run the full comparison.

        Args:
            num_games: Number of games to play

        Returns:
            ComparisonResult with statistics
        """
        config_key = f"{self.board_type}_{self.num_players}p"

        logger.info(f"Starting model comparison")
        logger.info(f"  Model A: {self.model_a_path}")
        logger.info(f"  Model B: {self.model_b_path}")
        logger.info(f"  Config: {config_key}")
        logger.info(f"  Games: {num_games}")
        logger.info(f"  MCTS simulations: {self.mcts_simulations}")

        start_time = time.time()
        results: List[MatchResult] = []

        # Alternate colors for fairness
        game_configs = [
            (i, 1 if i % 2 == 0 else 2)
            for i in range(num_games)
        ]

        # Run games
        if self.parallel_games > 1:
            results = self._run_parallel(game_configs)
        else:
            results = self._run_sequential(game_configs)

        total_time = time.time() - start_time

        # Calculate statistics
        return self._compute_statistics(results, total_time)

    def _run_parallel(
        self,
        game_configs: List[Tuple[int, int]],
    ) -> List[MatchResult]:
        """Run games in parallel."""
        results: List[MatchResult] = []

        with ProcessPoolExecutor(max_workers=self.parallel_games) as executor:
            futures = {
                executor.submit(
                    play_single_game,
                    self.model_a_path,
                    self.model_b_path,
                    self.board_type,
                    self.num_players,
                    game_id,
                    model_a_player,
                    self.mcts_simulations,
                ): game_id
                for game_id, model_a_player in game_configs
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)

                    # Progress update
                    a_wins = sum(1 for r in results if r.winner == 1)
                    b_wins = sum(1 for r in results if r.winner == 2)
                    draws = sum(1 for r in results if r.winner is None)

                    self.metrics.set("games_completed", len(results))
                    self.metrics.set("model_a_wins", a_wins)
                    self.metrics.set("model_b_wins", b_wins)

                    logger.debug(
                        f"Game {len(results)}/{len(game_configs)}: "
                        f"A={a_wins} B={b_wins} D={draws}"
                    )

                except Exception as e:
                    logger.error(f"Game failed: {e}")

        return results

    def _run_sequential(
        self,
        game_configs: List[Tuple[int, int]],
    ) -> List[MatchResult]:
        """Run games sequentially."""
        results: List[MatchResult] = []

        for game_id, model_a_player in game_configs:
            try:
                result = play_single_game(
                    self.model_a_path,
                    self.model_b_path,
                    self.board_type,
                    self.num_players,
                    game_id,
                    model_a_player,
                    self.mcts_simulations,
                )
                results.append(result)

                a_wins = sum(1 for r in results if r.winner == 1)
                b_wins = sum(1 for r in results if r.winner == 2)
                draws = sum(1 for r in results if r.winner is None)

                self.metrics.set("games_completed", len(results))
                logger.debug(
                    f"Game {len(results)}/{len(game_configs)}: "
                    f"A={a_wins} B={b_wins} D={draws}"
                )

            except Exception as e:
                logger.error(f"Game {game_id} failed: {e}")

        return results

    def _compute_statistics(
        self,
        results: List[MatchResult],
        total_time: float,
    ) -> ComparisonResult:
        """Compute final statistics from game results."""
        a_wins = sum(1 for r in results if r.winner == 1)
        b_wins = sum(1 for r in results if r.winner == 2)
        draws = sum(1 for r in results if r.winner is None)
        total = len(results)

        # Win rate (counting draws as 0.5)
        a_score = a_wins + 0.5 * draws
        win_rate = a_score / total if total > 0 else 0.5

        # Elo difference
        elo_diff = self.calculator.elo_diff_from_win_rate(win_rate)

        # Confidence interval
        ci_low, ci_high = self.calculator.wilson_confidence_interval(
            a_wins,
            total - draws,
        )
        elo_ci_low = (
            self.calculator.elo_diff_from_win_rate(ci_low)
            if ci_low > 0 else -400
        )
        elo_ci_high = (
            self.calculator.elo_diff_from_win_rate(ci_high)
            if ci_high < 1 else 400
        )

        # Average game length
        avg_moves = sum(r.moves for r in results) / len(results) if results else 0

        return ComparisonResult(
            model_a_path=self.model_a_path,
            model_b_path=self.model_b_path,
            board_type=self.board_type,
            num_players=self.num_players,
            total_games=total,
            model_a_wins=a_wins,
            model_b_wins=b_wins,
            draws=draws,
            model_a_win_rate=win_rate,
            elo_difference=elo_diff,
            confidence_interval=(elo_ci_low, elo_ci_high),
            avg_game_length=avg_moves,
            total_time=total_time,
        )


class ComparisonReporter:
    """Formats and outputs comparison reports."""

    def print_report(self, result: ComparisonResult) -> None:
        """Print a formatted comparison report."""
        total = result.total_games

        print(f"\n{'=' * 60}")
        print(f"  MODEL COMPARISON RESULTS")
        print(f"{'=' * 60}")
        print(f"  Model A: {result.model_a_path}")
        print(f"  Model B: {result.model_b_path}")
        print(f"  Config:  {result.board_type}_{result.num_players}p")
        print(f"{'=' * 60}")
        print(f"")
        print(f"  Model A wins: {result.model_a_wins:>4} ({result.model_a_wins/total*100:.1f}%)")
        print(f"  Model B wins: {result.model_b_wins:>4} ({result.model_b_wins/total*100:.1f}%)")
        print(f"  Draws:        {result.draws:>4} ({result.draws/total*100:.1f}%)")
        print(f"")
        print(f"  Model A win rate: {result.model_a_win_rate:.1%}")
        print(f"  Elo difference:   {result.elo_difference:+.0f} (A vs B)")
        print(f"  95% CI:           [{result.confidence_interval[0]:+.0f}, {result.confidence_interval[1]:+.0f}]")
        print(f"")
        print(f"  Avg game length:  {result.avg_game_length:.1f} moves")
        print(f"  Total time:       {result.total_time:.1f}s ({result.total_time/total:.1f}s/game)")
        print(f"{'=' * 60}")

        # Interpretation
        self._print_interpretation(result)

    def _print_interpretation(self, result: ComparisonResult) -> None:
        """Print interpretation of results."""
        elo_diff = result.elo_difference

        if elo_diff > 50:
            print(f"\n  Model A is SIGNIFICANTLY STRONGER (+{elo_diff:.0f} Elo)")
        elif elo_diff > 20:
            print(f"\n  Model A is moderately stronger (+{elo_diff:.0f} Elo)")
        elif elo_diff > -20:
            print(f"\n  Models are roughly equal ({elo_diff:+.0f} Elo)")
        elif elo_diff > -50:
            print(f"\n  Model B is moderately stronger ({elo_diff:+.0f} Elo)")
        else:
            print(f"\n  Model B is SIGNIFICANTLY STRONGER ({elo_diff:+.0f} Elo)")

    def save_result(self, result: ComparisonResult, output_path: Path) -> None:
        """Save comparison result to JSON."""
        data = result.to_dict()
        data["timestamp"] = datetime.now().isoformat()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Result saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare two models via Elo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model-a new.pt --model-b best.pt --games 100
  %(prog)s --model-a new.pt --model-b best.pt --quick
  %(prog)s --model-a new.pt --model-b best.pt --board-type hex8 --num-players 3
        """,
    )

    parser.add_argument(
        "--model-a",
        required=True,
        help="Path to model A",
    )
    parser.add_argument(
        "--model-b",
        required=True,
        help="Path to model B (baseline)",
    )
    parser.add_argument(
        "--board-type",
        default="square8",
        help="Board type (default: square8)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games (default: 100)",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=100,
        help="MCTS simulations per move (default: 100)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Parallel games (default: 4)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (fewer sims and games)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Use JSON format for log files",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_script_logging(
        script_name="compare_models",
        level=log_level,
        json_logs=args.json_logs,
    )

    # Validate model paths
    model_a = Path(args.model_a)
    model_b = Path(args.model_b)

    if not model_a.exists():
        logger.error(f"Model A not found: {model_a}")
        return 1

    if not model_b.exists():
        logger.error(f"Model B not found: {model_b}")
        return 1

    # Quick mode adjustments
    mcts_sims = args.mcts_sims
    num_games = args.games

    if args.quick:
        mcts_sims = 50
        num_games = min(num_games, 20)
        logger.info("Quick mode enabled: 50 sims, max 20 games")

    # Run comparison
    comparator = ModelComparator(
        model_a_path=str(model_a),
        model_b_path=str(model_b),
        board_type=args.board_type,
        num_players=args.num_players,
        mcts_simulations=mcts_sims,
        parallel_games=args.parallel,
    )

    try:
        result = comparator.run(num_games)
    except Exception as e:
        logger.exception(f"Comparison failed: {e}")
        return 1

    # Output results
    reporter = ComparisonReporter()
    reporter.print_report(result)

    if args.output:
        reporter.save_result(result, Path(args.output))

    # Log summary
    logger.info(
        f"Comparison complete: {result.total_games} games, "
        f"Elo diff: {result.elo_difference:+.0f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
