#!/usr/bin/env python3
"""Benchmark search algorithms (Descent, MCTS, Gumbel MCTS) to determine optimal strength.

This script runs head-to-head matches between different search algorithms at
various think-time budgets to empirically determine which provides the strongest
play. This is essential for selecting the optimal algorithm for difficulty 11
(Ultimate) mode.

Usage:
    # Quick benchmark (10 games per matchup)
    python scripts/benchmark_search_algorithms.py --games 10

    # Full benchmark (50 games per matchup)
    python scripts/benchmark_search_algorithms.py --games 50 --board square8

    # Compare at specific think times
    python scripts/benchmark_search_algorithms.py --think-times 5000,10000,30000

    # Run with verbose output
    python scripts/benchmark_search_algorithms.py --verbose

Key findings will be written to data/search_algorithm_benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.ai.base import BaseAI

# Disable torch dynamo to avoid triton compilation issues on some systems
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Add ai-service to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.models import (
    AIConfig,
    AIType,
    BoardType,
)
from app.rules.default_engine import DefaultRulesEngine
from app.training.generate_data import create_initial_state


@dataclass
class MatchResult:
    """Result of a single match."""
    algo1: str
    algo2: str
    think_time_ms: int
    winner: str | None  # "algo1", "algo2", "draw"
    game_length: int
    duration_sec: float
    algo1_time_total: float  # Total think time for algo1
    algo2_time_total: float  # Total think time for algo2


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for an algorithm pair."""
    algo1: str
    algo2: str
    think_time_ms: int
    games_played: int
    algo1_wins: int
    algo2_wins: int
    draws: int
    algo1_win_rate: float
    avg_game_length: float
    avg_duration_sec: float


def get_best_model_id(board_type: BoardType, num_players: int = 2) -> str | None:
    """Get the best model ID for a board/player configuration.

    Args:
        board_type: Board type
        num_players: Number of players

    Returns:
        Model ID string or None if no model available
    """
    # Map board types to model naming conventions (use short names that match actual files)
    board_prefix_map = {
        BoardType.SQUARE8: "sq8",
        BoardType.SQUARE19: "sq19",
        BoardType.HEXAGONAL: "hex",
        BoardType.HEX8: "hex8",
    }
    prefix = board_prefix_map.get(board_type, "sq8")

    # Standard model ID format used by the training pipeline
    model_id = f"ringrift_best_{prefix}_{num_players}p"
    return model_id


def create_ai_for_algorithm(
    algorithm: str,
    player_number: int,
    think_time_ms: int,
    board_type: BoardType,
    num_players: int = 2,
    game_seed: int = 0,
) -> BaseAI:
    """Create an AI instance for the specified algorithm.

    Args:
        algorithm: One of "descent", "mcts", "gumbel_mcts", "maxn", "brs"
        player_number: Player number (1, 2, 3, or 4)
        think_time_ms: Think time budget in milliseconds
        board_type: Board type for the game
        num_players: Number of players in the game
        game_seed: Per-game seed for varied randomness (Jan 2026 fix).
                   Without this, RandomAI produces identical games.

    Returns:
        Configured AI instance
    """
    from app.ai.descent_ai import DescentAI
    from app.ai.gumbel_mcts_ai import GumbelMCTSAI
    from app.ai.maxn_ai import BRSAI, MaxNAI
    from app.ai.mcts_ai import MCTSAI

    algorithm = algorithm.lower()

    # Get best model for this configuration
    model_id = get_best_model_id(board_type, num_players)

    if algorithm == "maxn":
        config = AIConfig(
            ai_type=AIType.MAXN,
            board_type=board_type,
            difficulty=10,
            think_time=think_time_ms,
            use_neural_net=False,  # MaxN uses heuristic evaluation
            randomness=0.0,
        )
        return MaxNAI(player_number, config)

    elif algorithm == "brs":
        config = AIConfig(
            ai_type=AIType.BRS,
            board_type=board_type,
            difficulty=10,
            think_time=think_time_ms,
            use_neural_net=False,  # BRS uses heuristic evaluation
            randomness=0.0,
        )
        return BRSAI(player_number, config)

    elif algorithm == "descent":
        config = AIConfig(
            ai_type=AIType.DESCENT,
            board_type=board_type,
            difficulty=10,
            think_time=think_time_ms,
            use_neural_net=True,
            randomness=0.0,
            nn_model_id=model_id,
            allow_fresh_weights=True,  # Allow fallback if model not found
        )
        return DescentAI(player_number, config)

    elif algorithm == "mcts":
        config = AIConfig(
            ai_type=AIType.MCTS,
            board_type=board_type,
            difficulty=10,
            think_time=think_time_ms,
            use_neural_net=True,
            randomness=0.0,
            nn_model_id=model_id,
            allow_fresh_weights=True,  # Allow fallback if model not found
        )
        return MCTSAI(player_number, config)

    elif algorithm == "gumbel_mcts":
        # For Gumbel MCTS, translate think_time to simulation budget
        # Rough estimate: 1 simulation ~= 5ms on average
        # Cap at 1000 due to AIConfig validation limit
        simulation_budget = min(1000, max(50, think_time_ms // 5))
        config = AIConfig(
            ai_type=AIType.GUMBEL_MCTS,
            board_type=board_type,
            difficulty=10,
            gumbel_simulation_budget=simulation_budget,
            gumbel_num_sampled_actions=16,
            use_neural_net=True,
            randomness=0.0,
            nn_model_id=model_id,
            allow_fresh_weights=True,  # Allow fallback if model not found
        )
        return GumbelMCTSAI(player_number, config, board_type)

    elif algorithm == "random":
        # Random player for neutral fill-in positions in multiplayer games
        from app.ai.random_ai import RandomAI
        # Per-game seed for varied randomness (Jan 2026 fix)
        rng_seed = (game_seed * 10000 + player_number * 1000) & 0xFFFFFFFF
        config = AIConfig(
            ai_type=AIType.RANDOM,
            board_type=board_type,
            difficulty=1,
            randomness=1.0,
            rng_seed=rng_seed,
        )
        return RandomAI(player_number, config)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def play_game(
    algo1: str,
    algo2: str,
    think_time_ms: int,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    max_moves: int = 500,
    verbose: bool = False,
    game_idx: int = 0,
) -> MatchResult:
    """Play a single game between two algorithms.

    Args:
        algo1: Algorithm name for player 1
        algo2: Algorithm name for player 2
        think_time_ms: Think time budget per move
        board_type: Board type
        num_players: Number of players
        max_moves: Maximum moves before declaring draw
        verbose: Print progress
        game_idx: Game index for unique RNG seeding (Jan 2026 fix).

    Returns:
        MatchResult with game outcome
    """
    start_time = time.time()

    # Create initial state
    state = create_initial_state(board_type, num_players)
    engine = DefaultRulesEngine()

    # Create AIs with proper model for this board/player combo (pass game_idx for unique seeding)
    # P1 = algo1, P2 = algo2, P3+ = random (neutral fill-in for fair comparison)
    ais = {}
    for player_num in range(1, num_players + 1):
        if player_num == 1:
            ais[player_num] = create_ai_for_algorithm(algo1, player_num, think_time_ms, board_type, num_players, game_idx)
        elif player_num == 2:
            ais[player_num] = create_ai_for_algorithm(algo2, player_num, think_time_ms, board_type, num_players, game_idx)
        else:
            # Neutral random players for positions 3+
            ais[player_num] = create_ai_for_algorithm("random", player_num, think_time_ms, board_type, num_players, game_idx)
    move_count = 0
    algo1_think_time = 0.0
    algo2_think_time = 0.0

    while state.game_status != "completed" and move_count < max_moves:
        current_player = state.current_player
        ai = ais[current_player]

        move_start = time.time()
        move = ai.select_move(state)
        move_end = time.time()

        # Track think time by algorithm (P1 = algo1, P2 = algo2, P3+ = random)
        if current_player == 1:
            algo1_think_time += (move_end - move_start)
        elif current_player == 2:
            algo2_think_time += (move_end - move_start)
        # P3+ are random players, don't track their think time

        if move is None:
            # No valid moves - skip turn or end game
            valid_moves = engine.get_valid_moves(state, current_player)
            if not valid_moves:
                # Pass turn
                state.current_player = 3 - current_player  # Toggle between 1 and 2
                continue
            break

        state = engine.apply_move(state, move)
        move_count += 1

        if verbose and move_count % 20 == 0:
            print(f"  Move {move_count}: Player {current_player} played")

    end_time = time.time()

    # Determine winner (P1 = algo1, P2 = algo2, P3+ = random)
    winner = None
    if state.game_status == "completed":
        if state.winner == 1:
            winner = "algo1"
        elif state.winner == 2:
            winner = "algo2"
        else:
            # Either no winner (draw) or random player (P3+) won
            # In both cases, neither algo1 nor algo2 won
            winner = "draw"
    else:
        winner = "draw"  # Max moves reached

    return MatchResult(
        algo1=algo1,
        algo2=algo2,
        think_time_ms=think_time_ms,
        winner=winner,
        game_length=move_count,
        duration_sec=end_time - start_time,
        algo1_time_total=algo1_think_time,
        algo2_time_total=algo2_think_time,
    )


def run_benchmark(
    algorithms: list[str],
    think_times: list[int],
    games_per_matchup: int,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    verbose: bool = False,
) -> list[BenchmarkResult]:
    """Run full benchmark between all algorithm pairs.

    Args:
        algorithms: List of algorithm names to compare
        think_times: List of think times to test
        games_per_matchup: Number of games per matchup
        board_type: Board type
        num_players: Number of players
        verbose: Print progress

    Returns:
        List of BenchmarkResult for each matchup
    """
    results = []

    # Log which model we're using
    model_id = get_best_model_id(board_type, num_players)
    print(f"\nUsing model: {model_id}", flush=True)

    # Generate all matchups (each pair plays in both color assignments)
    for think_time in think_times:
        print(f"\n{'='*60}", flush=True)
        print(f"Testing at think_time = {think_time}ms", flush=True)
        print(f"{'='*60}", flush=True)

        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i+1:]:
                print(f"\n  {algo1} vs {algo2} ({games_per_matchup} games each way)", flush=True)

                algo1_wins = 0
                algo2_wins = 0
                draws = 0
                total_length = 0
                total_duration = 0.0

                # Play half the games with algo1 as player 1
                half = games_per_matchup // 2
                for game_num in range(half):
                    if verbose:
                        print(f"    Game {game_num + 1}/{half} ({algo1} as P1)")

                    result = play_game(
                        algo1, algo2, think_time, board_type,
                        num_players=num_players,
                        verbose=verbose,
                        game_idx=game_num,  # Jan 2026: unique randomness per game
                    )

                    if result.winner == "algo1":
                        algo1_wins += 1
                    elif result.winner == "algo2":
                        algo2_wins += 1
                    else:
                        draws += 1

                    total_length += result.game_length
                    total_duration += result.duration_sec

                # Play remaining games with algo2 as player 1
                remaining = games_per_matchup - half
                for game_num in range(remaining):
                    if verbose:
                        print(f"    Game {game_num + 1}/{remaining} ({algo2} as P1)")

                    result = play_game(
                        algo2, algo1, think_time, board_type,
                        num_players=num_players,
                        verbose=verbose,
                        game_idx=half + game_num,  # Jan 2026: unique randomness (offset by half)
                    )

                    # Note: roles are swapped
                    if result.winner == "algo1":  # algo2 wins (was player 1)
                        algo2_wins += 1
                    elif result.winner == "algo2":  # algo1 wins (was player 2)
                        algo1_wins += 1
                    else:
                        draws += 1

                    total_length += result.game_length
                    total_duration += result.duration_sec

                total_games = games_per_matchup

                bench_result = BenchmarkResult(
                    algo1=algo1,
                    algo2=algo2,
                    think_time_ms=think_time,
                    games_played=total_games,
                    algo1_wins=algo1_wins,
                    algo2_wins=algo2_wins,
                    draws=draws,
                    algo1_win_rate=algo1_wins / total_games if total_games > 0 else 0.0,
                    avg_game_length=total_length / total_games if total_games > 0 else 0.0,
                    avg_duration_sec=total_duration / total_games if total_games > 0 else 0.0,
                )

                results.append(bench_result)

                # Print summary
                print(f"    Results: {algo1}={algo1_wins}, {algo2}={algo2_wins}, draws={draws}", flush=True)
                print(f"    {algo1} win rate: {bench_result.algo1_win_rate:.1%}", flush=True)

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary of benchmark results."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    # Group by think time
    by_think_time: dict[int, list[BenchmarkResult]] = {}
    for r in results:
        if r.think_time_ms not in by_think_time:
            by_think_time[r.think_time_ms] = []
        by_think_time[r.think_time_ms].append(r)

    for think_time, group in sorted(by_think_time.items()):
        print(f"\n  Think Time: {think_time}ms")
        print("  " + "-"*40)

        for r in group:
            if r.algo1_wins > r.algo2_wins:
                winner = r.algo1
                win_margin = r.algo1_wins - r.algo2_wins
            elif r.algo2_wins > r.algo1_wins:
                winner = r.algo2
                win_margin = r.algo2_wins - r.algo1_wins
            else:
                winner = "TIE"
                win_margin = 0

            print(f"    {r.algo1:15} vs {r.algo2:15}: "
                  f"{r.algo1_wins:2}-{r.algo2_wins:2}-{r.draws:2} "
                  f"→ {winner} (+{win_margin})")

    # Overall algorithm rankings (sum of wins across all matchups)
    algo_total_wins: dict[str, int] = {}
    algo_total_games: dict[str, int] = {}

    for r in results:
        for algo, wins in [(r.algo1, r.algo1_wins), (r.algo2, r.algo2_wins)]:
            if algo not in algo_total_wins:
                algo_total_wins[algo] = 0
                algo_total_games[algo] = 0
            algo_total_wins[algo] += wins
            algo_total_games[algo] += r.games_played

    print("\n" + "="*70)
    print("OVERALL RANKINGS (by total wins)")
    print("="*70)

    sorted_algos = sorted(
        algo_total_wins.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for rank, (algo, wins) in enumerate(sorted_algos, 1):
        games = algo_total_games[algo]
        win_rate = wins / games if games > 0 else 0.0
        print(f"  {rank}. {algo:20} {wins:3} wins / {games:3} games ({win_rate:.1%})")

    # Recommend best algorithm
    if sorted_algos:
        best_algo = sorted_algos[0][0]
        print(f"\n  RECOMMENDED for Ultimate difficulty: {best_algo}")


def save_results(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark search algorithms for optimal AI strength"
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="descent,mcts,gumbel_mcts",
        help="Comma-separated list of algorithms to compare (descent,mcts,gumbel_mcts,maxn,brs)"
    )
    parser.add_argument(
        "--think-times",
        type=str,
        default="5000,10000,20000",
        help="Comma-separated list of think times in ms"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games per matchup (default: 10)"
    )
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal", "hex8"],
        help="Board type (default: square8)"
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players (default: 2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    # Parse arguments
    algorithms = [a.strip() for a in args.algorithms.split(",")]
    think_times = [int(t.strip()) for t in args.think_times.split(",")]

    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
        "hex8": BoardType.HEX8,
    }
    board_type = board_type_map[args.board]

    print("="*70, flush=True)
    print("SEARCH ALGORITHM BENCHMARK", flush=True)
    print("="*70, flush=True)
    print(f"Algorithms:  {', '.join(algorithms)}", flush=True)
    print(f"Think times: {', '.join(str(t) + 'ms' for t in think_times)}", flush=True)
    print(f"Games/matchup: {args.games}", flush=True)
    print(f"Board type:  {args.board}", flush=True)
    print(f"Players:     {args.players}", flush=True)
    print("="*70, flush=True)

    # Run benchmark
    results = run_benchmark(
        algorithms=algorithms,
        think_times=think_times,
        games_per_matchup=args.games,
        board_type=board_type,
        num_players=args.players,
        verbose=args.verbose,
    )

    # Print summary
    print_summary(results)

    # Save results
    output_path = Path(args.output) if args.output else (
        AI_SERVICE_ROOT / "data" / f"benchmark_{args.board}_{args.players}p.json"
    )
    save_results(results, output_path)

    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)


if __name__ == "__main__":
    main()
