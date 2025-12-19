#!/usr/bin/env python3
"""
Model Elo Comparison Pipeline

Compares two models head-to-head to determine Elo difference.
Used to validate training improvements.

Usage:
    # Compare new model vs current best
    python scripts/compare_models_elo.py \
        --model-a models/nnue/new_model.pt \
        --model-b models/nnue/square8_2p_best.pt \
        --games 100 \
        --board-type square8 --num-players 2

    # Quick comparison (fewer games)
    python scripts/compare_models_elo.py --model-a new.pt --model-b best.pt --games 20 --quick
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import BoardType, GameStatus


@dataclass
class MatchResult:
    """Result of a single game."""
    game_id: int
    winner: Optional[int]  # 1 = model_a, 2 = model_b, None = draw
    moves: int
    time_seconds: float
    model_a_color: int  # Which player model_a was (1 or 2)


@dataclass
class ComparisonResult:
    """Result of full comparison."""
    model_a_path: str
    model_b_path: str
    total_games: int
    model_a_wins: int
    model_b_wins: int
    draws: int
    model_a_win_rate: float
    elo_difference: float
    confidence_interval: Tuple[float, float]
    avg_game_length: float
    total_time: float


def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_diff_from_win_rate(win_rate: float) -> float:
    """Calculate Elo difference from win rate."""
    if win_rate <= 0:
        return -400
    if win_rate >= 1:
        return 400
    return -400 * math.log10(1.0 / win_rate - 1.0)


def wilson_confidence_interval(wins: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
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
    model_a_player: int,  # Which player (1 or 2) is model_a
    mcts_simulations: int = 100,
) -> MatchResult:
    """Play a single game between two models."""
    from app.game_engine import GameEngine
    from app.ai.nnue import NNUEEvaluator
    from app.ai.mcts_ai import MCTSAI
    from app.training.generate_data import create_initial_state
    from app.models import AIConfig

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
        if (state.winner == 1 and model_a_player == 1) or (state.winner == 2 and model_a_player == 2):
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


def run_comparison(
    model_a_path: str,
    model_b_path: str,
    board_type: str,
    num_players: int,
    num_games: int,
    mcts_simulations: int = 100,
    parallel: int = 4,
) -> ComparisonResult:
    """Run full comparison between two models."""
    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"  Model A: {model_a_path}")
    print(f"  Model B: {model_b_path}")
    print(f"  Games: {num_games} ({board_type}_{num_players}p)")
    print(f"  MCTS sims: {mcts_simulations}")
    print(f"{'='*60}\n")

    start_time = time.time()
    results: List[MatchResult] = []

    # Alternate colors for fairness
    game_configs = []
    for i in range(num_games):
        model_a_player = 1 if i % 2 == 0 else 2
        game_configs.append((i, model_a_player))

    # Run games
    if parallel > 1:
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    play_single_game,
                    model_a_path, model_b_path,
                    board_type, num_players,
                    game_id, model_a_player,
                    mcts_simulations
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
                    print(f"  Game {len(results)}/{num_games}: A={a_wins} B={b_wins} D={draws}", end="\r")

                except Exception as e:
                    print(f"  Game failed: {e}")
    else:
        for game_id, model_a_player in game_configs:
            try:
                result = play_single_game(
                    model_a_path, model_b_path,
                    board_type, num_players,
                    game_id, model_a_player,
                    mcts_simulations
                )
                results.append(result)

                a_wins = sum(1 for r in results if r.winner == 1)
                b_wins = sum(1 for r in results if r.winner == 2)
                draws = sum(1 for r in results if r.winner is None)
                print(f"  Game {len(results)}/{num_games}: A={a_wins} B={b_wins} D={draws}", end="\r")

            except Exception as e:
                print(f"  Game {game_id} failed: {e}")

    print()  # New line after progress

    # Calculate statistics
    a_wins = sum(1 for r in results if r.winner == 1)
    b_wins = sum(1 for r in results if r.winner == 2)
    draws = sum(1 for r in results if r.winner is None)
    total = len(results)

    # Win rate (counting draws as 0.5)
    a_score = a_wins + 0.5 * draws
    win_rate = a_score / total if total > 0 else 0.5

    # Elo difference
    elo_diff = elo_diff_from_win_rate(win_rate)

    # Confidence interval
    ci_low, ci_high = wilson_confidence_interval(a_wins, total - draws)
    elo_ci_low = elo_diff_from_win_rate(ci_low) if ci_low > 0 else -400
    elo_ci_high = elo_diff_from_win_rate(ci_high) if ci_high < 1 else 400

    # Average game length
    avg_moves = sum(r.moves for r in results) / len(results) if results else 0

    total_time = time.time() - start_time

    comparison = ComparisonResult(
        model_a_path=model_a_path,
        model_b_path=model_b_path,
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

    # Print results
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Model A wins: {a_wins:>4} ({a_wins/total*100:.1f}%)")
    print(f"  Model B wins: {b_wins:>4} ({b_wins/total*100:.1f}%)")
    print(f"  Draws:        {draws:>4} ({draws/total*100:.1f}%)")
    print(f"")
    print(f"  Model A win rate: {win_rate:.1%}")
    print(f"  Elo difference:   {elo_diff:+.0f} (A vs B)")
    print(f"  95% CI:           [{elo_ci_low:+.0f}, {elo_ci_high:+.0f}]")
    print(f"")
    print(f"  Avg game length:  {avg_moves:.1f} moves")
    print(f"  Total time:       {total_time:.1f}s ({total_time/total:.1f}s/game)")
    print(f"{'='*60}")

    # Interpretation
    if elo_diff > 50:
        print(f"\n  ✅ Model A is SIGNIFICANTLY STRONGER (+{elo_diff:.0f} Elo)")
    elif elo_diff > 20:
        print(f"\n  📈 Model A is moderately stronger (+{elo_diff:.0f} Elo)")
    elif elo_diff > -20:
        print(f"\n  🔄 Models are roughly equal ({elo_diff:+.0f} Elo)")
    elif elo_diff > -50:
        print(f"\n  📉 Model B is moderately stronger ({elo_diff:+.0f} Elo)")
    else:
        print(f"\n  ❌ Model B is SIGNIFICANTLY STRONGER ({elo_diff:+.0f} Elo)")

    return comparison


def save_comparison_result(result: ComparisonResult, output_path: Path):
    """Save comparison result to JSON."""
    data = {
        "model_a": result.model_a_path,
        "model_b": result.model_b_path,
        "total_games": result.total_games,
        "model_a_wins": result.model_a_wins,
        "model_b_wins": result.model_b_wins,
        "draws": result.draws,
        "model_a_win_rate": result.model_a_win_rate,
        "elo_difference": result.elo_difference,
        "confidence_interval": result.confidence_interval,
        "avg_game_length": result.avg_game_length,
        "total_time_seconds": result.total_time,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResult saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare two models via Elo")
    parser.add_argument("--model-a", required=True, help="Path to model A")
    parser.add_argument("--model-b", required=True, help="Path to model B (baseline)")
    parser.add_argument("--board-type", default="square8", help="Board type")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--mcts-sims", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel games")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer sims)")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    if args.quick:
        args.mcts_sims = 50
        args.games = min(args.games, 20)

    result = run_comparison(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        board_type=args.board_type,
        num_players=args.num_players,
        num_games=args.games,
        mcts_simulations=args.mcts_sims,
        parallel=args.parallel,
    )

    if args.output:
        save_comparison_result(result, Path(args.output))


if __name__ == "__main__":
    main()
