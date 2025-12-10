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
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    TimeControl,
)
from app.ai.heuristic_ai import HeuristicAI
from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS
from app.rules.default_engine import DefaultRulesEngine


def create_game_state(
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
) -> GameState:
    """Create an initial game state for profiling."""
    # RR-CANON-R061: victoryThreshold = ringsPerPlayer
    if board_type == BoardType.SQUARE8:
        size = 8
        rings_per_player = 18
        victory_threshold = 18  # ringsPerPlayer
        territory_threshold = 33
    elif board_type == BoardType.SQUARE19:
        size = 19
        rings_per_player = 48
        victory_threshold = 48  # ringsPerPlayer
        territory_threshold = 181
    elif board_type == BoardType.HEXAGONAL:
        size = 13  # Canonical hex: size=13, radius=12
        rings_per_player = 72
        victory_threshold = 72  # ringsPerPlayer
        territory_threshold = 235  # >234 for 469 cells
    else:
        size = 8
        rings_per_player = 18
        victory_threshold = 18  # ringsPerPlayer
        territory_threshold = 33

    now = datetime.now()

    board = BoardState(
        type=board_type,
        size=size,
        stacks={},
        markers={},
        collapsedSpaces={},
        eliminatedRings={},
    )

    players = [
        Player(
            id=f"player{i}",
            username=f"AI {i}",
            type="ai",
            playerNumber=i,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=rings_per_player,
            eliminatedRings=0,
            territorySpaces=0,
        )
        for i in range(1, num_players + 1)
    ]

    return GameState(
        id="profiling-game",
        boardType=board_type,
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
        maxPlayers=num_players,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=victory_threshold,
        territoryVictoryThreshold=territory_threshold,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsExclusivePlayerForCompletedRound=None,
    )


@dataclass
class TimingStats:
    """Accumulated timing statistics for a code section."""

    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0

    def record(self, elapsed: float) -> None:
        self.total_time += elapsed
        self.call_count += 1
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)

    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    def __str__(self) -> str:
        if self.call_count == 0:
            return "no calls"
        return (
            f"{self.total_time*1000:.1f}ms total, "
            f"{self.call_count} calls, "
            f"{self.avg_time*1000:.3f}ms avg, "
            f"[{self.min_time*1000:.3f}-{self.max_time*1000:.3f}]ms range"
        )


class SelfPlayProfiler:
    """Profiles self-play performance with detailed timing breakdown."""

    def __init__(self):
        self.timings: Dict[str, TimingStats] = defaultdict(TimingStats)
        self.move_counts: List[int] = []
        self.game_times: List[float] = []

    @contextmanager
    def time_section(self, name: str):
        """Context manager to time a code section."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[name].record(elapsed)

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
            state = create_game_state(board_type, num_players)

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
        max_moves = 500

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

        print(f"\nOverall Statistics:")
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

        total_measured = sum(t.total_time for _, t in sorted_timings)

        for name, stats in sorted_timings:
            pct = (stats.total_time / total_time) * 100 if total_time > 0 else 0
            print(
                f"{name:<30} "
                f"{stats.total_time*1000:>8.1f}ms  "
                f"{stats.call_count:>8}  "
                f"{stats.avg_time*1000:>8.3f}ms  "
                f"{pct:>8.1f}%"
            )

        # Analysis
        print("\n" + "=" * 70)
        print("BOTTLENECK ANALYSIS")
        print("=" * 70)

        select_move_time = self.timings.get("select_move", TimingStats()).total_time
        apply_time = self.timings.get("apply_move", TimingStats()).total_time

        print(f"\nTime breakdown:")
        if total_moves > 0:
            print(f"  select_move: {select_move_time/total_moves*1000:.2f}ms avg per call")
            print(f"  apply_move: {apply_time/total_moves*1000:.2f}ms avg per call")
            print(f"  select_move as % of total: {select_move_time/total_time*100:.1f}%")
            print(f"  apply_move as % of total: {apply_time/total_time*100:.1f}%")

        print(f"\nRun with --cprofile for detailed function-level breakdown")


def run_cprofile_analysis(board_type: BoardType, num_players: int, num_games: int):
    """Run cProfile analysis for detailed function-level profiling."""

    print("\n" + "=" * 70)
    print("cProfile DETAILED ANALYSIS")
    print("=" * 70)

    rules = DefaultRulesEngine()

    def play_games():
        for _ in range(num_games):
            state = create_game_state(board_type, num_players)
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
