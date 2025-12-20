#!/usr/bin/env python
"""GPU-accelerated LPS + rings ablation experiment.

This script runs GPU-parallel self-play games with configurable LPS rounds threshold
AND rings per player to analyze impact on termination reason distribution.
~100x faster than CPU version.

Background:
- LPS victory occurs when only one player has "real actions" available
  for consecutive rounds while opponents can only pass/skip.
- The default threshold is 2 rounds (traditional rule).
- This experiment tests whether increasing to 3+ rounds and/or changing
  ring counts produces better game termination diversity.

Usage examples:
    # Compare 2 vs 3 LPS rounds on square8 (fastest)
    python scripts/run_gpu_lps_ablation.py \
        --num-games 1000 \
        --board-type square8 \
        --num-players 2 \
        --lps-rounds 2 3 \
        --batch-size 256

    # Full cross-product: LPS rounds AND rings per player
    python scripts/run_gpu_lps_ablation.py \
        --num-games 500 \
        --board-type square19 \
        --num-players 2 \
        --lps-rounds 2 3 \
        --rings-per-player default 96

    # Test hexagonal with increased rings
    python scripts/run_gpu_lps_ablation.py \
        --num-games 500 \
        --board-type hexagonal \
        --num-players 2 \
        --lps-rounds 2 3 \
        --rings-per-player default 120

Performance:
    GPU (RTX 3090): ~100-500 games/sec
    CPU (heuristic): ~0.1-1 games/sec
    Speedup: 100-1000x
"""
from __future__ import annotations

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

import numpy as np

# Add app/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from app.ai.gpu_batch import get_device, clear_gpu_memory
from app.ai.gpu_parallel_games import ParallelGameRunner


# Board type mapping
BOARD_SIZE_MAP = {
    "square8": 8,
    "square19": 19,
    "hexagonal": 25,
}


@dataclass
class GPUExperimentConfig:
    """Configuration for GPU LPS + rings ablation experiment."""
    num_games: int
    board_type: str
    num_players: int
    lps_rounds_values: list[int]
    rings_per_player_values: list[int | None]  # None = board default
    batch_size: int
    seed: int
    output_dir: str | None


@dataclass
class GPUExperimentResults:
    """Aggregated results from GPU experiment run."""
    config: dict[str, Any]
    results_by_condition: dict[str, dict[str, Any]]
    raw_results: list[dict[str, Any]]


def run_gpu_condition(
    board_size: int,
    num_players: int,
    lps_rounds: int,
    rings_per_player: int | None,
    num_games: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Run a single experimental condition using GPU parallel games.

    Returns dict with:
        - termination_counts: Counter of termination reasons
        - move_counts: List of move counts
        - duration_ms: Total runtime
        - games_per_sec: Throughput
    """
    start_time = time.time()

    # Create runner with specific LPS threshold and rings
    runner = ParallelGameRunner(
        batch_size=batch_size,
        board_size=board_size,
        num_players=num_players,
        device=device,
        lps_victory_rounds=lps_rounds,
        rings_per_player=rings_per_player,
        swap_enabled=True,  # Enable pie rule for 2p
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Run games in batches
    all_results = []
    games_completed = 0

    while games_completed < num_games:
        remaining = num_games - games_completed
        actual_batch = min(batch_size, remaining)

        # Reset for new batch
        runner.reset_games()

        # Run batch
        results = runner.run_games(max_moves=500)

        # Collect results
        for i in range(actual_batch):
            vtype, tiebreaker = runner.state.derive_victory_type(i, max_moves=500)
            all_results.append({
                "winner": int(results["winners"][i]),
                "move_count": int(results["move_counts"][i]),
                "victory_type": vtype,
                "tiebreaker": tiebreaker,
            })

        games_completed += actual_batch

    duration_ms = (time.time() - start_time) * 1000

    # Aggregate termination counts
    termination_counts = Counter()
    move_counts = []
    for r in all_results:
        termination_counts[r["victory_type"]] += 1
        move_counts.append(r["move_count"])

    return {
        "termination_counts": dict(termination_counts),
        "move_counts": move_counts,
        "duration_ms": duration_ms,
        "games_per_sec": num_games / (duration_ms / 1000),
        "raw_results": all_results,
    }


def run_gpu_experiment(config: GPUExperimentConfig) -> GPUExperimentResults:
    """Run the full GPU LPS + rings ablation experiment."""
    print(f"\n{'='*60}")
    print("GPU LPS + RINGS ABLATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Games per condition: {config.num_games}")
    print(f"Board type: {config.board_type}")
    print(f"Players: {config.num_players}")
    print(f"LPS rounds values: {config.lps_rounds_values}")
    print(f"Rings per player values: {config.rings_per_player_values}")
    print(f"Batch size: {config.batch_size}")

    device = get_device()
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    board_size = BOARD_SIZE_MAP.get(config.board_type, 8)

    all_results: list[dict[str, Any]] = []
    results_by_condition: dict[str, dict[str, Any]] = {}

    for lps_rounds in config.lps_rounds_values:
        for rings_per_player in config.rings_per_player_values:
            # Build condition key
            rings_label = f"r{rings_per_player}" if rings_per_player else "rdef"
            condition_key = f"{config.board_type}_{config.num_players}p_lps{lps_rounds}_{rings_label}"
            print(f"\n--- Running condition: {condition_key} ---")

            condition_data = run_gpu_condition(
                board_size=board_size,
                num_players=config.num_players,
                lps_rounds=lps_rounds,
                rings_per_player=rings_per_player,
                num_games=config.num_games,
                batch_size=config.batch_size,
                device=device,
                seed=config.seed,
            )

            # Compute statistics
            total_games = len(condition_data["raw_results"])
            avg_moves = sum(condition_data["move_counts"]) / total_games

            stats = {
                "total_games": total_games,
                "lps_rounds": lps_rounds,
                "rings_per_player": rings_per_player,
                "avg_move_count": round(avg_moves, 1),
                "duration_ms": round(condition_data["duration_ms"], 1),
                "games_per_sec": round(condition_data["games_per_sec"], 1),
                "termination_distribution": {
                    k: {"count": v, "pct": round(100 * v / total_games, 1)}
                    for k, v in sorted(condition_data["termination_counts"].items(), key=lambda x: -x[1])
                },
            }
            results_by_condition[condition_key] = stats

            # Store raw results
            for r in condition_data["raw_results"]:
                r["lps_rounds"] = lps_rounds
                r["rings_per_player"] = rings_per_player
                r["condition"] = condition_key
                all_results.append(r)

            # Print summary
            print(f"\n  Results for {condition_key}:")
            print(f"    Games: {total_games} in {stats['duration_ms']/1000:.1f}s ({stats['games_per_sec']:.1f} g/s)")
            print(f"    Average moves: {stats['avg_move_count']}")
            for term, info in stats["termination_distribution"].items():
                print(f"    {term}: {info['count']} ({info['pct']}%)")

            # Clear GPU memory between conditions
            clear_gpu_memory()

    return GPUExperimentResults(
        config=asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config),
        results_by_condition=results_by_condition,
        raw_results=all_results,
    )


def print_comparison_table(results: GPUExperimentResults) -> None:
    """Print a comparison table of termination distributions."""
    print(f"\n{'='*90}")
    print("COMPARISON TABLE: Termination Distribution by LPS Threshold and Rings")
    print(f"{'='*90}")

    print(f"\n{'Condition':<30} {'Territory':>10} {'Elimination':>12} {'LPS':>10} {'Timeout':>10} {'Avg Moves':>10}")
    print("-" * 85)

    for cond, stats in sorted(results.results_by_condition.items()):
        term_dist = stats["termination_distribution"]

        territory = term_dist.get("territory", {}).get("pct", 0)
        elimination = term_dist.get("ring_elimination", {}).get("pct", 0)
        lps = term_dist.get("lps", {}).get("pct", 0)
        timeout = term_dist.get("timeout", {}).get("pct", 0)

        print(f"{cond:<30} {territory:>9.1f}% {elimination:>11.1f}% {lps:>9.1f}% {timeout:>9.1f}% {stats['avg_move_count']:>10.1f}")


def parse_rings_value(val: str) -> int | None:
    """Parse a rings-per-player value from CLI.

    - 'default', '0', or empty string -> None (use default from board config)
    - positive integer -> that value
    """
    val = val.strip().lower()
    if val in ('', 'default', '0', 'none'):
        return None
    return int(val)


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated LPS + rings ablation experiment"
    )
    parser.add_argument(
        "--num-games", "-n",
        type=int,
        default=1000,
        help="Number of games per condition (default: 1000)"
    )
    parser.add_argument(
        "--board-type", "-b",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type (default: square8)"
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
        "--batch-size", "-bs",
        type=int,
        default=256,
        help="GPU batch size (default: 256)"
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
        help="Output directory for results (default: logs/gpu_lps_ablation)"
    )

    args = parser.parse_args()

    # Parse rings values
    rings_values = [parse_rings_value(v) for v in args.rings_per_player]

    config = GPUExperimentConfig(
        num_games=args.num_games,
        board_type=args.board_type,
        num_players=args.num_players,
        lps_rounds_values=args.lps_rounds,
        rings_per_player_values=rings_values,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir or "logs/gpu_lps_ablation",
    )

    # Run experiment
    results = run_gpu_experiment(config)

    # Print comparison table
    print_comparison_table(results)

    # Save results
    if config.output_dir:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"gpu_lps_ablation_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump({
                "config": config.__dict__,
                "results_by_condition": results.results_by_condition,
                "summary": {
                    cond: {
                        "lps_pct": stats["termination_distribution"].get("lps", {}).get("pct", 0),
                        "avg_moves": stats["avg_move_count"],
                    }
                    for cond, stats in results.results_by_condition.items()
                },
            }, f, indent=2)

        print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
