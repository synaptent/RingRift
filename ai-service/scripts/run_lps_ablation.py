#!/usr/bin/env python
"""LPS (Last Player Standing) rounds threshold ablation experiment.

This script runs self-play games with configurable LPS rounds threshold
to analyze the impact on termination reason distribution.

Background:
- LPS victory occurs when only one player has "real actions" available
  for consecutive rounds while opponents can only pass/skip.
- The default threshold is 2 rounds (traditional rule).
- This experiment tests whether increasing to 3+ rounds produces better
  game termination diversity (more territory/elimination, less LPS).

Usage examples:
    # Compare 2 vs 3 rounds on square8 2p
    python scripts/run_lps_ablation.py \
        --num-games 200 \
        --board-type square8 \
        --num-players 2 \
        --lps-rounds 2 3 \
        --engine-mode heuristic-only \
        --output-dir logs/lps_ablation

    # Run on multiple board types
    python scripts/run_lps_ablation.py \
        --num-games 100 \
        --board-type square8 square19 hexagonal \
        --num-players 2 \
        --lps-rounds 2 3 \
        --engine-mode heuristic-only

    # Test with increased rings (for larger boards)
    python scripts/run_lps_ablation.py \
        --num-games 100 \
        --board-type square19 \
        --num-players 2 \
        --lps-rounds 2 3 \
        --rings-per-player 72 96 \
        --engine-mode heuristic-only

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
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
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
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
)
from app.training.env import (
    TrainingEnvConfig,
    make_env,
    get_theoretical_max_moves,
)
from app.game_engine import GameEngine


# Board type mapping
BOARD_TYPE_MAP = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hexagonal": BoardType.HEXAGONAL,
}

# Engine mode to AI type mapping
ENGINE_MODE_TO_AI = {
    "heuristic-only": AIType.HEURISTIC,
    "mcts-only": AIType.MCTS,
    "descent-only": AIType.DESCENT,
    "random-only": AIType.RANDOM,
}


@dataclass
class GameResult:
    """Result of a single game."""
    game_id: int
    lps_rounds: int
    board_type: str
    num_players: int
    termination_reason: str
    winner: Optional[int]
    move_count: int
    duration_ms: float


@dataclass
class ExperimentConfig:
    """Configuration for an LPS ablation experiment."""
    num_games: int
    board_types: List[str]
    num_players: int
    lps_rounds_values: List[int]
    engine_mode: str
    seed: int
    output_dir: Optional[str]


@dataclass
class ExperimentResults:
    """Aggregated results from an experiment run."""
    config: Dict[str, Any]
    results_by_condition: Dict[str, Dict[str, Any]]
    raw_results: List[Dict[str, Any]]


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
    engine_mode: str,
    seed: int,
    game_id: int,
) -> GameResult:
    """Run a single self-play game with specified LPS threshold."""
    start_time = time.time()

    # Create environment
    max_moves = get_theoretical_max_moves(board_type, num_players)
    env_config = TrainingEnvConfig(
        board_type=board_type,
        num_players=num_players,
        max_moves=max_moves,
        seed=seed + game_id,
    )

    env = make_env(env_config)
    state = env.reset()

    # Patch the LPS threshold on the environment's internal state
    # The env holds a reference to the state that gets updated on step
    def patch_lps_threshold(s: GameState) -> GameState:
        """Patch LPS threshold on a state."""
        if hasattr(s, 'model_dump'):
            # For pydantic models, we need to reconstruct
            state_dict = s.model_dump(by_alias=True)
            state_dict['lpsRoundsRequired'] = lps_rounds
            return GameState(**state_dict)
        return s

    state = patch_lps_threshold(state)
    # Also patch the env's internal state if accessible
    if hasattr(env, '_state'):
        env._state = state

    # Create AI instances
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
        ai = ais[current_player]

        # Get move from AI
        move = ai.select_move(state)
        if move is None:
            break

        # Apply move through environment
        state, _reward, done, _info = env.step(move)

        # Re-patch LPS threshold after each step (state is replaced)
        if not done:
            state = patch_lps_threshold(state)
            if hasattr(env, '_state'):
                env._state = state

        move_count += 1

    duration_ms = (time.time() - start_time) * 1000
    termination = get_termination_reason(state)

    return GameResult(
        game_id=game_id,
        lps_rounds=lps_rounds,
        board_type=board_type.value if hasattr(board_type, 'value') else str(board_type),
        num_players=num_players,
        termination_reason=termination,
        winner=state.winner,
        move_count=move_count,
        duration_ms=duration_ms,
    )


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """Run the full LPS ablation experiment."""
    print(f"\n{'='*60}")
    print("LPS ABLATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Games per condition: {config.num_games}")
    print(f"Board types: {config.board_types}")
    print(f"Players: {config.num_players}")
    print(f"LPS rounds values: {config.lps_rounds_values}")
    print(f"Engine mode: {config.engine_mode}")
    print(f"{'='*60}\n")

    all_results: List[GameResult] = []
    results_by_condition: Dict[str, Dict[str, Any]] = {}

    for board_type_str in config.board_types:
        board_type = BOARD_TYPE_MAP.get(board_type_str)
        if board_type is None:
            print(f"Warning: Unknown board type {board_type_str}, skipping")
            continue

        for lps_rounds in config.lps_rounds_values:
            condition_key = f"{board_type_str}_{config.num_players}p_lps{lps_rounds}"
            print(f"\n--- Running condition: {condition_key} ---")

            condition_results: List[GameResult] = []
            termination_counts = Counter()

            for game_id in range(config.num_games):
                if (game_id + 1) % 10 == 0:
                    print(f"  Game {game_id + 1}/{config.num_games}...")

                result = run_single_game(
                    board_type=board_type,
                    num_players=config.num_players,
                    lps_rounds=lps_rounds,
                    engine_mode=config.engine_mode,
                    seed=config.seed,
                    game_id=game_id,
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
    print(f"\n{'='*70}")
    print("COMPARISON TABLE: Termination Distribution by LPS Threshold")
    print(f"{'='*70}")

    # Group by board type
    conditions = list(results.results_by_condition.keys())
    board_types = set()
    for cond in conditions:
        parts = cond.rsplit('_lps', 1)
        if len(parts) == 2:
            board_types.add(parts[0])

    for bt in sorted(board_types):
        print(f"\n{bt}:")
        print(f"{'LPS Rounds':<12} {'Territory':>10} {'Elimination':>12} {'LPS':>10} {'Other':>10} {'Avg Moves':>10}")
        print("-" * 66)

        for cond in sorted(conditions):
            if not cond.startswith(bt + "_"):
                continue

            lps_val = cond.rsplit('_lps', 1)[1]
            stats = results.results_by_condition[cond]
            term_dist = stats["termination_distribution"]

            territory = term_dist.get("territory", {}).get("pct", 0)
            elimination = term_dist.get("elimination", {}).get("pct", 0)
            lps = term_dist.get("lps", {}).get("pct", 0)
            other = 100 - territory - elimination - lps

            print(f"{lps_val:>12} {territory:>9.1f}% {elimination:>11.1f}% {lps:>9.1f}% {other:>9.1f}% {stats['avg_move_count']:>10.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run LPS rounds threshold ablation experiment"
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
        "--engine-mode", "-e",
        default="heuristic-only",
        choices=["heuristic-only", "mcts-only", "descent-only", "random-only"],
        help="AI engine mode (default: heuristic-only)"
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

    args = parser.parse_args()

    config = ExperimentConfig(
        num_games=args.num_games,
        board_types=args.board_type,
        num_players=args.num_players,
        lps_rounds_values=args.lps_rounds,
        engine_mode=args.engine_mode,
        seed=args.seed,
        output_dir=args.output_dir or "logs/lps_ablation",
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
