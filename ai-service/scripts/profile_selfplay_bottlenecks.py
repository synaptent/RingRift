#!/usr/bin/env python3
"""Profile self-play bottlenecks to identify optimization opportunities.

This script measures where time is spent during self-play games to help
identify the best targets for GPU/MPS acceleration or other optimizations.

Usage:
    python scripts/profile_selfplay_bottlenecks.py --board square8 --games 5
    python scripts/profile_selfplay_bottlenecks.py --board square19 --games 2
    python scripts/profile_selfplay_bottlenecks.py --board square8 --detailed
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.heuristic_ai import HeuristicAI
from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS
from app.models import AIConfig, BoardType, GameStatus
from app.rules.default_engine import DefaultRulesEngine
from app.training.initial_state import create_initial_state
from scripts.lib.metrics import TimingStats


class SelfPlayProfiler:
    """Profiles self-play performance with detailed timing breakdown."""

    def __init__(self):
        self.timings: dict[str, TimingStats] = defaultdict(TimingStats)
        self.move_counts: list[int] = []
        self.game_times: list[float] = []

    def time_section(self, name: str):
        """Context manager to time a code section."""
        return self.timings[name].time()

    def play_profiled_game(
        self,
        board_type: BoardType,
        num_players: int,
        rules: DefaultRulesEngine,
        detailed: bool = False,
    ) -> int:
        """Play a single game with detailed timing instrumentation."""

        game_start = time.perf_counter()

        # Create initial state
        with self.time_section("state_creation"):
            state = create_initial_state(board_type, num_players)

        # Create AIs (HeuristicAI takes player_number and AIConfig)
        ais = []
        for i in range(num_players):
            player_number = i + 1
            config = AIConfig(
                difficulty=5,
                think_time=0,
                randomness=0.15,
                rngSeed=None,
            )
            ai = HeuristicAI(player_number, config)
            # Apply weights via setattr (like CMA-ES does)
            for name, value in BASE_V1_BALANCED_WEIGHTS.items():
                setattr(ai, name, value)
            ais.append(ai)

        move_count = 0
        max_moves = 10000

        while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
            current_player = state.current_player
            ai = ais[current_player - 1]  # Convert 1-indexed to 0-indexed
            ai.player_number = current_player

            # Time move selection (the main bottleneck)
            with self.time_section("select_move"):
                move = ai.select_move(state)

            if move is None:
                break

            # Apply move to advance game
            with self.time_section("apply_move"):
                state = rules.apply_move(state, move)

            move_count += 1

            if detailed and move_count % 20 == 0:
                print(f"  Move {move_count} completed")

        game_time = time.perf_counter() - game_start
        self.move_counts.append(move_count)
        self.game_times.append(game_time)

        return move_count

    def print_report(self, board_type: str):
        """Print profiling report."""
        print("\n" + "=" * 70)
        print(f"SELF-PLAY PROFILING REPORT - {board_type}")
        print("=" * 70)

        total_games = len(self.game_times)
        total_moves = sum(self.move_counts)
        total_time = sum(self.game_times)

        print("\nOverall Statistics:")
        print(f"  Games played: {total_games}")
        print(f"  Total moves: {total_moves}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg game time: {total_time/total_games:.2f}s")
        print(f"  Avg moves/game: {total_moves/total_games:.1f}")
        print(f"  Moves/second: {total_moves/total_time:.2f}")

        print(f"\n{'Section':<30} {'Total':<12} {'Calls':<10} {'Avg':<12} {'% of Total':<10}")
        print("-" * 74)

        # Sort by total time
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1].total_time, reverse=True)

        sum(t.total_time for _, t in sorted_timings)

        for name, stats in sorted_timings:
            pct = (stats.total_time / total_time) * 100 if total_time > 0 else 0
            print(
                f"{name:<30} "
                f"{stats.total_time_ms:>8.1f}ms  "
                f"{stats.count:>8}  "
                f"{stats.avg_time_ms:>8.3f}ms  "
                f"{pct:>8.1f}%"
            )

        # Analysis
        print("\n" + "=" * 70)
        print("BOTTLENECK ANALYSIS")
        print("=" * 70)

        select_move_time = self.timings.get("select_move", TimingStats()).total_time
        apply_time = self.timings.get("apply_move", TimingStats()).total_time

        print("\nTime breakdown:")
        if total_moves > 0:
            print(f"  select_move: {select_move_time/total_moves*1000:.2f}ms avg per call")
            print(f"  apply_move: {apply_time/total_moves*1000:.2f}ms avg per call")
            print(f"  select_move as % of total: {select_move_time/total_time*100:.1f}%")
            print(f"  apply_move as % of total: {apply_time/total_time*100:.1f}%")

        print("\nRun with --cprofile for detailed function-level breakdown")


def run_cprofile_analysis(board_type: BoardType, num_players: int, num_games: int):
    """Run cProfile analysis for detailed function-level profiling."""

    print("\n" + "=" * 70)
    print("cProfile DETAILED ANALYSIS")
    print("=" * 70)

    rules = DefaultRulesEngine()

    def play_games():
        for _ in range(num_games):
            state = create_initial_state(board_type, num_players)
            ais = []
            for i in range(num_players):
                player_number = i + 1
                config = AIConfig(
                    difficulty=5,
                    think_time=0,
                    randomness=0.15,
                    rngSeed=None,
                )
                ai = HeuristicAI(player_number, config)
                for name, value in BASE_V1_BALANCED_WEIGHTS.items():
                    setattr(ai, name, value)
                ais.append(ai)

            move_count = 0
            while state.game_status == GameStatus.ACTIVE and move_count < 200:
                current_player = state.current_player
                ai = ais[current_player - 1]
                ai.player_number = current_player
                move = ai.select_move(state)
                if move is None:
                    break
                state = rules.apply_move(state, move)
                move_count += 1

    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    play_games()
    profiler.disable()

    # Print stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    print(stream.getvalue())

    # Also print by tottime
    print("\nTop functions by total time (excluding subcalls):")
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats("tottime")
    stats2.print_stats(20)
    print(stream2.getvalue())


def main():
    parser = argparse.ArgumentParser(
        description="Profile self-play bottlenecks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--board",
        "-b",
        choices=["square8", "square19", "hex"],
        default="square8",
        help="Board type to profile",
    )
    parser.add_argument(
        "--games",
        "-g",
        type=int,
        default=3,
        help="Number of games to play",
    )
    parser.add_argument(
        "--players",
        "-p",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed move-by-move progress",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Also run cProfile analysis",
    )

    args = parser.parse_args()

    # Map board type
    board_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hex": BoardType.HEXAGONAL,
    }
    board_type = board_map[args.board]

    print(f"Profiling {args.games} games on {args.board} with {args.players} players...")

    # Run custom profiling
    profiler = SelfPlayProfiler()
    rules = DefaultRulesEngine()

    for game_idx in range(args.games):
        print(f"\nGame {game_idx + 1}/{args.games}...")
        moves = profiler.play_profiled_game(
            board_type,
            args.players,
            rules,
            detailed=args.detailed,
        )
        print(f"  Completed in {profiler.game_times[-1]:.2f}s with {moves} moves")

    profiler.print_report(args.board)

    # Optional cProfile
    if args.cprofile:
        run_cprofile_analysis(board_type, args.players, args.games)

    # Check if skip_shadow_contracts is enabled
    skip_shadow = os.getenv("RINGRIFT_SKIP_SHADOW_CONTRACTS", "").lower() in {"1", "true", "yes", "on"}

    print("\n" + "=" * 70)
    print("OPTIMIZATION STATUS")
    print("=" * 70)
    print(
        f"""
  RINGRIFT_SKIP_SHADOW_CONTRACTS: {"ENABLED ✓" if skip_shadow else "DISABLED"}

  To enable shadow contract skip for 2-3x speedup:
    export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
"""
    )

    print("=" * 70)
    print("RECOMMENDATIONS FOR GPU/MPS ACCELERATION")
    print("=" * 70)
    print(
        """
0. SKIP SHADOW CONTRACTS (Already implemented - HIGHEST IMPACT!)
   - Set RINGRIFT_SKIP_SHADOW_CONTRACTS=true
   - Skips validation deep-copies in DefaultRulesEngine
   - Provides 2-3x speedup with zero risk to accuracy
   - Use for all training/benchmarking runs

1. BATCH POSITION EVALUATION (Medium effort, High impact)
   - Collect all (move, next_state) pairs
   - Extract features to NumPy arrays in batch
   - Compute heuristic scores via vectorized operations
   - Works well on CPU with NumPy, can extend to GPU

2. NUMPY VECTORIZATION FOR HEURISTICS (Low effort, Medium impact)
   - Convert board state to NumPy arrays once
   - Use vectorized operations for:
     * Stack counting/summing
     * Territory area calculation
     * Center distance computation
     * Adjacency checks

3. NUMBA JIT COMPILATION (Medium effort, High impact)
   - @numba.jit decorators on hot evaluation functions
   - Compile Python loops to machine code
   - No GPU needed, but significant speedup

4. PYTORCH/JAX FOR HEURISTICS (High effort, High impact)
   - Rewrite heuristic eval as tensor operations
   - Batch multiple states together
   - Use MPS/CUDA for parallel evaluation
   - Best for very large boards (square19)

5. STATE POOLING / CACHING (Low effort, Medium impact)
   - Reuse GameState objects instead of creating new ones
   - Cache intermediate calculations (adjacency, territories)
   - Use Zobrist hashing for transposition table

6. MOVE SAMPLING (Already implemented)
   - training_move_sample_limit already exists
   - Reduces evaluations needed per move
"""
    )


if __name__ == "__main__":
    main()
