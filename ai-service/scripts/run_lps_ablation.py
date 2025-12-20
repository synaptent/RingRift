#!/usr/bin/env python
"""LPS (Last Player Standing) rounds threshold + rings ablation experiment.

This script runs self-play games with configurable LPS rounds threshold
AND rings-per-player to analyze the impact on termination reason distribution.

Background:
- LPS victory occurs when only one player has "real actions" available
  for consecutive rounds while opponents can only pass/skip.
- The default threshold is 2 rounds (traditional rule).
- This experiment tests whether increasing to 3+ rounds and/or changing
  ring counts produces better game termination diversity.

Usage examples:
    # Compare 2 vs 3 LPS rounds on square8 2p (default rings)
    python scripts/run_lps_ablation.py \
        --num-games 200 \
        --board-type square8 \
        --num-players 2 \
        --lps-rounds 2 3 \
        --engine-mode heuristic-only

    # Compare LPS AND rings on square19 (96 vs default 72)
    python scripts/run_lps_ablation.py \
        --num-games 100 \
        --board-type square19 \
        --num-players 2 \
        --lps-rounds 2 3 \
        --rings-per-player default 96 \
        --engine-mode heuristic-only

    # Test hexagonal with increased rings (120 vs default 96)
    python scripts/run_lps_ablation.py \
        --num-games 100 \
        --board-type hexagonal \
        --num-players 2 \
        --lps-rounds 2 3 \
        --rings-per-player default 120 \
        --engine-mode heuristic-only

    # Full cross-product experiment on all boards
    python scripts/run_lps_ablation.py \
        --num-games 50 \
        --board-type square8 square19 hexagonal \
        --num-players 2 \
        --lps-rounds 2 3 \
        --rings-per-player default 96 120 \
        --engine-mode random-only

Environment variables:
    RINGRIFT_DISABLE_FSM_VALIDATION=1  - Disable FSM validation for faster runs
"""
from __future__ import annotations

# Disable FSM validation for faster experiments
import os
os.environ.setdefault("RINGRIFT_FSM_VALIDATION_MODE", "off")

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure `app.*` imports resolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.main import _create_ai_instance
from app.models import (
    AIConfig,
    AIType,
    BoardType,
    GameState,
    GameStatus,
)
from app.training.env import (
    TrainingEnvConfig,
    make_env,
    get_theoretical_max_moves,
)
from app.game_engine import GameEngine
from scripts.lib.cli import BOARD_TYPE_MAP


# Engine mode to AI type mapping
ENGINE_MODE_TO_AI = {
    "heuristic-only": AIType.HEURISTIC,
    "mcts-only": AIType.MCTS,
    "descent-only": AIType.DESCENT,
    "random-only": AIType.RANDOM,
    "hybrid-gpu": None,  # Special case: uses HybridGPUEvaluator
}


@dataclass
class GameResult:
    """Result of a single game."""
    game_id: int
    lps_rounds: int
    rings_per_player: int | None  # None means default
    board_type: str
    num_players: int
    termination_reason: str
    winner: int | None
    move_count: int
    duration_ms: float


@dataclass
class ExperimentConfig:
    """Configuration for an LPS ablation experiment."""
    num_games: int
    board_types: list[str]
    num_players: int
    lps_rounds_values: list[int]
    rings_per_player_values: list[int | None]  # None means use default
    engine_mode: str
    seed: int
    output_dir: str | None
    verbose: bool = False
    progress_interval: int = 10


@dataclass
class ExperimentResults:
    """Aggregated results from an experiment run."""
    config: dict[str, Any]
    results_by_condition: dict[str, dict[str, Any]]
    raw_results: list[dict[str, Any]]


def get_termination_reason(state: GameState) -> str:
    """Extract termination reason from final game state."""
    if state.game_status != GameStatus.COMPLETED:
        return "incomplete"

    # Check for early LPS (one player with stacks, others eliminated)
    if hasattr(state, 'termination_reason') and state.termination_reason:
        return state.termination_reason

    # Infer from state
    board = state.board
    players_with_stacks = set()
    for stack in board.stacks.values():
        players_with_stacks.add(stack.controlling_player)

    if len(players_with_stacks) <= 1:
        # Check if others have any rings
        active_rings = {p: 0 for p in range(1, state.max_players + 1)}
        for stack in board.stacks.values():
            for ring in stack.rings:
                active_rings[ring] += 1
        for player in state.players:
            active_rings[player.player_number] += player.rings_in_hand

        players_with_rings = sum(1 for r in active_rings.values() if r > 0)
        if players_with_rings <= 1:
            return "elimination"

    # Check move count for timeout
    max_moves = get_theoretical_max_moves(state.board_type, state.max_players)
    if len(state.move_history) >= max_moves:
        return "timeout"

    # Default to LPS
    return "lps"


def run_single_game(
    board_type: BoardType,
    num_players: int,
    lps_rounds: int,
    rings_per_player: int | None,
    engine_mode: str,
    seed: int,
    game_id: int,
    verbose: bool = False,
    progress_interval: int = 10,
    hybrid_evaluator=None,
) -> GameResult:
    """Run a single self-play game with specified LPS threshold and rings."""
    start_time = time.time()

    # Create environment with experimental overrides
    max_moves = get_theoretical_max_moves(board_type, num_players)
    env_config = TrainingEnvConfig(
        board_type=board_type,
        num_players=num_players,
        max_moves=max_moves,
        seed=seed + game_id,
        rings_per_player=rings_per_player,
        lps_rounds_required=lps_rounds,
    )

    env = make_env(env_config)
    state = env.reset()

    # For hybrid-gpu mode, use GPU-accelerated move selection
    use_hybrid = engine_mode == "hybrid-gpu" and hybrid_evaluator is not None

    if not use_hybrid:
        # Create standard CPU AI instances
        ai_type = ENGINE_MODE_TO_AI.get(engine_mode, AIType.HEURISTIC)
        ai_config = AIConfig(
            type=ai_type,
            difficulty=5,
            useNeuralNetwork=False,
        )
        ais = {}
        for p in range(1, num_players + 1):
            ais[p] = _create_ai_instance(ai_type, p, ai_config)

    # Play the game
    move_count = 0
    done = False
    while not done and move_count < max_moves:
        current_player = state.current_player

        # Get move from AI
        move_start = time.time()

        if use_hybrid:
            # Use hybrid GPU evaluator for move selection
            # Get valid moves from rules engine
            valid_moves = GameEngine.get_valid_moves(state, current_player)
            if not valid_moves:
                # Check for bookkeeping requirements
                req = GameEngine.get_phase_requirement(state, current_player)
                if req is not None:
                    move = GameEngine.synthesize_bookkeeping_move(req, state)
                else:
                    move = None
            else:
                # Evaluate moves with GPU-accelerated heuristic
                move_scores = hybrid_evaluator.evaluate_moves(
                    state, valid_moves, current_player, GameEngine
                )
                if move_scores:
                    move = max(move_scores, key=lambda x: x[1])[0]
                else:
                    move = valid_moves[0]
        else:
            ai = ais[current_player]
            move = ai.select_move(state)

        move_time_ms = (time.time() - move_start) * 1000
        if move is None:
            break

        # Apply move through environment
        state, _reward, done, _info = env.step(move)
        move_count += 1

        # Verbose progress output
        if verbose and (move_count % progress_interval == 0 or move_time_ms > 1000):
            elapsed = time.time() - start_time
            moves_per_sec = move_count / elapsed if elapsed > 0 else 0
            print(
                f"    [G{game_id}] Move {move_count}: "
                f"{move.type.value if hasattr(move.type, 'value') else move.type} "
                f"by P{move.player} ({move_time_ms:.0f}ms, {moves_per_sec:.1f} m/s)",
                flush=True,
            )

    duration_ms = (time.time() - start_time) * 1000
    termination = get_termination_reason(state)

    return GameResult(
        game_id=game_id,
        lps_rounds=lps_rounds,
        rings_per_player=rings_per_player,
        board_type=board_type.value if hasattr(board_type, 'value') else str(board_type),
        num_players=num_players,
        termination_reason=termination,
        winner=state.winner,
        move_count=move_count,
        duration_ms=duration_ms,
    )


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """Run the full LPS + rings ablation experiment."""
    print(f"\n{'='*60}")
    print("LPS + RINGS ABLATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Games per condition: {config.num_games}")
    print(f"Board types: {config.board_types}")
    print(f"Players: {config.num_players}")
    print(f"LPS rounds values: {config.lps_rounds_values}")
    print(f"Rings per player values: {config.rings_per_player_values}")
    print(f"Engine mode: {config.engine_mode}")
    print(f"{'='*60}\n")

    # Initialize hybrid GPU evaluator if using hybrid-gpu mode
    hybrid_evaluator = None
    if config.engine_mode == "hybrid-gpu":
        try:
            from app.ai.hybrid_gpu import create_hybrid_evaluator
            print("Initializing Hybrid GPU Evaluator...")
            # We'll create per-board-type evaluators in the loop
        except ImportError as e:
            print(f"Warning: Could not import hybrid_gpu: {e}")
            print("Falling back to heuristic-only mode")
            config.engine_mode = "heuristic-only"

    all_results: list[GameResult] = []
    results_by_condition: dict[str, dict[str, Any]] = {}

    for board_type_str in config.board_types:
        board_type = BOARD_TYPE_MAP.get(board_type_str)
        if board_type is None:
            print(f"Warning: Unknown board type {board_type_str}, skipping")
            continue

        # Create hybrid evaluator for this board type if using hybrid-gpu mode
        if config.engine_mode == "hybrid-gpu":
            try:
                from app.ai.hybrid_gpu import create_hybrid_evaluator
                hybrid_evaluator = create_hybrid_evaluator(
                    board_type=board_type_str,
                    num_players=config.num_players,
                    prefer_gpu=True,
                )
                print(f"  Created hybrid evaluator for {board_type_str}")
            except Exception as e:
                print(f"  Warning: Failed to create hybrid evaluator: {e}")
                hybrid_evaluator = None

        for lps_rounds in config.lps_rounds_values:
            for rings_per_player in config.rings_per_player_values:
                # Build condition key
                rings_label = f"r{rings_per_player}" if rings_per_player else "rdef"
                condition_key = f"{board_type_str}_{config.num_players}p_lps{lps_rounds}_{rings_label}"
                print(f"\n--- Running condition: {condition_key} ---")

                condition_results: list[GameResult] = []
                termination_counts = Counter()

                for game_id in range(config.num_games):
                    if (game_id + 1) % 10 == 0:
                        print(f"  Game {game_id + 1}/{config.num_games}...")

                    result = run_single_game(
                        board_type=board_type,
                        num_players=config.num_players,
                        lps_rounds=lps_rounds,
                        rings_per_player=rings_per_player,
                        engine_mode=config.engine_mode,
                        seed=config.seed,
                        game_id=game_id,
                        verbose=config.verbose,
                        progress_interval=config.progress_interval,
                        hybrid_evaluator=hybrid_evaluator,
                    )
                    condition_results.append(result)
                    all_results.append(result)
                    termination_counts[result.termination_reason] += 1

                # Compute statistics for this condition
                total_games = len(condition_results)
                avg_moves = sum(r.move_count for r in condition_results) / total_games
                avg_duration = sum(r.duration_ms for r in condition_results) / total_games

                stats = {
                    "total_games": total_games,
                    "lps_rounds": lps_rounds,
                    "rings_per_player": rings_per_player,
                    "avg_move_count": round(avg_moves, 1),
                    "avg_duration_ms": round(avg_duration, 1),
                    "termination_distribution": {
                        k: {"count": v, "pct": round(100 * v / total_games, 1)}
                        for k, v in termination_counts.most_common()
                    },
                }
                results_by_condition[condition_key] = stats

                # Print summary for this condition
                print(f"\n  Results for {condition_key}:")
                print(f"    Average moves: {stats['avg_move_count']}")
                for term, info in stats["termination_distribution"].items():
                    print(f"    {term}: {info['count']} ({info['pct']}%)")

    return ExperimentResults(
        config=asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config),
        results_by_condition=results_by_condition,
        raw_results=[asdict(r) for r in all_results],
    )


def print_comparison_table(results: ExperimentResults) -> None:
    """Print a comparison table of termination distributions."""
    print(f"\n{'='*90}")
    print("COMPARISON TABLE: Termination Distribution by LPS Threshold and Rings")
    print(f"{'='*90}")

    # Group by board type
    conditions = list(results.results_by_condition.keys())
    board_types = set()
    for cond in conditions:
        # Extract board type (everything before _Np_)
        parts = cond.split('_')
        if len(parts) >= 2:
            board_types.add(parts[0])

    for bt in sorted(board_types):
        print(f"\n{bt}:")
        print(f"{'Condition':<30} {'Territory':>10} {'Elimination':>12} {'LPS':>10} {'Other':>10} {'Avg Moves':>10}")
        print("-" * 86)

        for cond in sorted(conditions):
            if not cond.startswith(bt + "_"):
                continue

            stats = results.results_by_condition[cond]
            term_dist = stats["termination_distribution"]

            territory = term_dist.get("territory", {}).get("pct", 0)
            elimination = term_dist.get("elimination", {}).get("pct", 0)
            lps = term_dist.get("lps", {}).get("pct", 0)
            other = 100 - territory - elimination - lps

            # Extract condition suffix for display
            display_label = cond[len(bt)+1:]  # Remove board type prefix
            print(f"{display_label:<30} {territory:>9.1f}% {elimination:>11.1f}% {lps:>9.1f}% {other:>9.1f}% {stats['avg_move_count']:>10.1f}")


def parse_rings_value(val: str) -> int | None:
    """Parse a rings-per-player value from CLI.

    - 'default', '0', or empty string -> None (use default from BOARD_CONFIGS)
    - positive integer -> that value
    """
    val = val.strip().lower()
    if val in ('', 'default', '0', 'none'):
        return None
    return int(val)


def main():
    parser = argparse.ArgumentParser(
        description="Run LPS + rings ablation experiment"
    )
    parser.add_argument(
        "--num-games", "-n",
        type=int,
        default=100,
        help="Number of games per condition (default: 100)"
    )
    parser.add_argument(
        "--board-type", "-b",
        nargs="+",
        default=["square8"],
        help="Board type(s): square8, square19, hexagonal (default: square8)"
    )
    parser.add_argument(
        "--num-players", "-p",
        type=int,
        default=2,
        help="Number of players (default: 2)"
    )
    parser.add_argument(
        "--lps-rounds", "-l",
        nargs="+",
        type=int,
        default=[2, 3],
        help="LPS rounds threshold values to test (default: 2 3)"
    )
    parser.add_argument(
        "--rings-per-player", "-r",
        nargs="+",
        default=["default"],
        help="Rings per player values to test. Use 'default' or 0 for board default. (default: default)"
    )
    parser.add_argument(
        "--engine-mode", "-e",
        default="heuristic-only",
        choices=["heuristic-only", "mcts-only", "descent-only", "random-only", "hybrid-gpu"],
        help="AI engine mode. hybrid-gpu uses GPU-accelerated heuristic evaluation (default: heuristic-only)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for results (default: logs/lps_ablation)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable move-by-move progress output for monitoring"
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print progress every N moves when verbose (default: 10)"
    )

    args = parser.parse_args()

    # Parse rings values
    rings_values = [parse_rings_value(v) for v in args.rings_per_player]

    config = ExperimentConfig(
        num_games=args.num_games,
        board_types=args.board_type,
        num_players=args.num_players,
        lps_rounds_values=args.lps_rounds,
        rings_per_player_values=rings_values,
        engine_mode=args.engine_mode,
        seed=args.seed,
        output_dir=args.output_dir or "logs/lps_ablation",
        verbose=args.verbose,
        progress_interval=args.progress_interval,
    )

    # Run experiment
    results = run_experiment(config)

    # Print comparison table
    print_comparison_table(results)

    # Save results
    if config.output_dir:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"lps_ablation_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump({
                "config": config.__dict__,
                "results_by_condition": results.results_by_condition,
                "raw_results": results.raw_results,
            }, f, indent=2)

        print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
