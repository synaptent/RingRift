#!/usr/bin/env python
"""GPU-accelerated self-play data generation.

This script generates training data using GPU parallel game simulation,
achieving 10-100x speedup compared to CPU-based self-play.

The generated data can be used for:
1. Neural network training (policy/value targets)
2. CMA-ES fitness evaluation baselines
3. Game analysis and statistics

Usage:
    # Generate 1000 games on GPU
    python scripts/run_gpu_selfplay.py \\
        --num-games 1000 \\
        --board square8 \\
        --num-players 2 \\
        --output-dir data/selfplay/gpu_square8_2p

    # With specific heuristic weights
    python scripts/run_gpu_selfplay.py \\
        --num-games 500 \\
        --board square8 \\
        --weights-file config/trained_heuristic_profiles.json \\
        --profile heuristic_v1_2p \\
        --output-dir data/selfplay/trained_2p

    # Benchmark mode
    python scripts/run_gpu_selfplay.py --benchmark-only

Output:
    - games.jsonl: Game records in JSONL format
    - stats.json: Aggregated statistics
    - training_data.npz: NumPy arrays ready for NN training (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add app/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from app.ai.gpu_batch import get_device, clear_gpu_memory
from app.ai.gpu_parallel_games import (
    ParallelGameRunner,
    BatchGameState,
    benchmark_parallel_games,
)

# NOTE: Victory type derivation is now handled internally by the
# BatchGameState class in gpu_parallel_games.py, which derives
# victory_type and stalemate_tiebreaker from the final GPU state.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Default Heuristic Weights
# =============================================================================

DEFAULT_WEIGHTS = {
    "material_weight": 1.0,
    "ring_count_weight": 0.5,
    "stack_height_weight": 0.3,
    "center_control_weight": 0.4,
    "territory_weight": 0.8,
    "mobility_weight": 0.2,
    "line_potential_weight": 0.6,
    "defensive_weight": 0.3,
}


def load_weights_from_profile(
    weights_file: str,
    profile_name: str,
) -> Dict[str, float]:
    """Load heuristic weights from a profile file."""
    if not os.path.exists(weights_file):
        logger.warning(f"Weights file not found: {weights_file}, using defaults")
        return DEFAULT_WEIGHTS.copy()

    with open(weights_file, "r") as f:
        data = json.load(f)

    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        logger.warning(f"Profile {profile_name} not found, using defaults")
        return DEFAULT_WEIGHTS.copy()

    return profiles[profile_name].get("weights", DEFAULT_WEIGHTS.copy())


# =============================================================================
# GPU Self-Play Generator
# =============================================================================


class GPUSelfPlayGenerator:
    """Generate self-play games using GPU parallel simulation."""

    def __init__(
        self,
        board_size: int = 8,
        num_players: int = 2,
        batch_size: int = 256,
        max_moves: int = 500,
        device: Optional[torch.device] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.board_size = board_size
        self.num_players = num_players
        self.batch_size = batch_size
        self.max_moves = max_moves
        self.device = device or get_device()
        self.weights = weights or DEFAULT_WEIGHTS.copy()

        self.runner = ParallelGameRunner(
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
            device=self.device,
        )

        # Statistics
        self.total_games = 0
        self.total_moves = 0
        self.total_time = 0.0
        self.wins_by_player = {i: 0 for i in range(1, num_players + 1)}
        self.draws = 0

    def generate_batch(
        self,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a batch of games.

        Returns dict with:
            - winners: List of winning player numbers (0 = draw)
            - move_counts: List of move counts per game
            - games_per_second: Throughput
            - elapsed_seconds: Wall time
        """
        if seed is not None:
            torch.manual_seed(seed)

        start = time.time()
        results = self.runner.run_games(
            weights_list=[self.weights] * self.batch_size,
            max_moves=self.max_moves,
        )
        elapsed = time.time() - start

        # Update statistics
        self.total_games += self.batch_size
        self.total_moves += sum(results["move_counts"])
        self.total_time += elapsed

        for winner in results["winners"]:
            if winner == 0:
                self.draws += 1
            else:
                self.wins_by_player[winner] = self.wins_by_player.get(winner, 0) + 1

        return results

    def generate_games(
        self,
        num_games: int,
        output_file: Optional[str] = None,
        progress_interval: int = 10,
    ) -> List[Dict[str, Any]]:
        """Generate multiple batches of games.

        Args:
            num_games: Total number of games to generate
            output_file: Optional JSONL file to stream results to
            progress_interval: Log progress every N batches

        Returns:
            List of game records
        """
        all_records = []
        num_batches = (num_games + self.batch_size - 1) // self.batch_size

        logger.info(f"Generating {num_games} games in {num_batches} batches...")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Device: {self.device}")
        logger.info("")

        start_time = time.time()
        file_handle = None

        if output_file:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            file_handle = open(output_file, "w")

        try:
            for batch_idx in range(num_batches):
                # Adjust batch size for last batch
                remaining = num_games - len(all_records)
                actual_batch = min(self.batch_size, remaining)

                # Generate batch
                results = self.generate_batch(seed=batch_idx * 1000)

                # Create game records
                for i in range(actual_batch):
                    record = {
                        "game_id": len(all_records),
                        "batch_id": batch_idx,
                        "board_size": self.board_size,
                        "num_players": self.num_players,
                        "winner": int(results["winners"][i]),
                        "move_count": int(results["move_counts"][i]),
                        "max_moves": self.max_moves,
                        "victory_type": results["victory_types"][i],
                        "stalemate_tiebreaker": results["stalemate_tiebreakers"][i],
                        "moves": results["move_histories"][i],  # Full move history for training
                        "timestamp": datetime.now().isoformat(),
                    }
                    all_records.append(record)

                    if file_handle:
                        file_handle.write(json.dumps(record) + "\n")

                # Progress logging
                if (batch_idx + 1) % progress_interval == 0:
                    elapsed = time.time() - start_time
                    games_done = len(all_records)
                    games_per_sec = games_done / elapsed if elapsed > 0 else 0
                    eta = (num_games - games_done) / games_per_sec if games_per_sec > 0 else 0

                    logger.info(
                        f"  Batch {batch_idx + 1}/{num_batches}: "
                        f"{games_done}/{num_games} games, "
                        f"{games_per_sec:.1f} g/s, "
                        f"ETA: {eta:.0f}s"
                    )

        finally:
            if file_handle:
                file_handle.close()

        return all_records

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        total_decided = sum(self.wins_by_player.values())

        stats = {
            "total_games": self.total_games,
            "total_moves": self.total_moves,
            "total_time_seconds": self.total_time,
            "games_per_second": self.total_games / self.total_time if self.total_time > 0 else 0,
            "moves_per_game": self.total_moves / self.total_games if self.total_games > 0 else 0,
            "wins_by_player": self.wins_by_player,
            "draws": self.draws,
            "draw_rate": self.draws / self.total_games if self.total_games > 0 else 0,
            "board_size": self.board_size,
            "num_players": self.num_players,
            "batch_size": self.batch_size,
            "max_moves": self.max_moves,
            "device": str(self.device),
            "weights": self.weights,
        }

        # Add win rates by player
        for p in range(1, self.num_players + 1):
            wins = self.wins_by_player.get(p, 0)
            stats[f"p{p}_win_rate"] = wins / total_decided if total_decided > 0 else 0

        return stats


# =============================================================================
# Main Entry Point
# =============================================================================


def run_gpu_selfplay(
    board_type: str,
    num_players: int,
    num_games: int,
    output_dir: str,
    batch_size: int = 256,
    max_moves: int = 500,
    weights: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run GPU-accelerated self-play generation.

    Args:
        board_type: Board type (square8, square19, hex)
        num_players: Number of players
        num_games: Total games to generate
        output_dir: Output directory
        batch_size: GPU batch size
        max_moves: Max moves per game
        weights: Heuristic weights
        seed: Random seed

    Returns:
        Statistics dict
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    board_size = {"square8": 8, "square19": 19, "hex": 25}.get(board_type.lower(), 8)

    logger.info("=" * 60)
    logger.info("GPU-ACCELERATED SELF-PLAY GENERATION")
    logger.info("=" * 60)
    logger.info(f"Board: {board_type} ({board_size}x{board_size})")
    logger.info(f"Players: {num_players}")
    logger.info(f"Games: {num_games}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max moves: {max_moves}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Create generator
    generator = GPUSelfPlayGenerator(
        board_size=board_size,
        num_players=num_players,
        batch_size=batch_size,
        max_moves=max_moves,
        weights=weights,
    )

    # Generate games
    games_file = os.path.join(output_dir, "games.jsonl")
    records = generator.generate_games(
        num_games=num_games,
        output_file=games_file,
        progress_interval=10,
    )

    # Get and save statistics
    stats = generator.get_statistics()
    stats["timestamp"] = datetime.now().isoformat()
    stats["seed"] = seed

    stats_file = os.path.join(output_dir, "stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total games: {stats['total_games']}")
    logger.info(f"Total moves: {stats['total_moves']}")
    logger.info(f"Avg moves/game: {stats['moves_per_game']:.1f}")
    logger.info(f"Total time: {stats['total_time_seconds']:.1f}s")
    logger.info(f"Throughput: {stats['games_per_second']:.1f} games/sec")
    logger.info(f"Draw rate: {stats['draw_rate']:.1%}")
    logger.info("")
    logger.info("Win rates by player:")
    for p in range(1, num_players + 1):
        logger.info(f"  Player {p}: {stats[f'p{p}_win_rate']:.1%}")
    logger.info("")
    logger.info(f"Games saved to: {games_file}")
    logger.info(f"Stats saved to: {stats_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated self-play data generation"
    )
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex"],
        help="Board type",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1000,
        help="Number of games to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="GPU batch size (games per batch)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=500,
        help="Maximum moves per game",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/selfplay/gpu",
        help="Output directory",
    )
    parser.add_argument(
        "--weights-file",
        type=str,
        help="Path to heuristic weights JSON file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="Profile name in weights file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run GPU benchmark",
    )

    args = parser.parse_args()

    if args.benchmark_only:
        logger.info("Running GPU benchmark...")
        device = get_device()
        board_size = {"square8": 8, "square19": 19, "hex": 25}.get(args.board.lower(), 8)
        results = benchmark_parallel_games(
            batch_sizes=[32, 64, 128, 256, 512, 1024],
            board_size=board_size,
            max_moves=100,
            device=device,
        )
        logger.info("Benchmark results:")
        for i, bs in enumerate(results["batch_size"]):
            logger.info(
                f"  Batch {bs}: {results['games_per_second'][i]:.1f} games/sec, "
                f"{results['moves_per_second'][i]:.1f} moves/sec"
            )
        return

    # Load weights if specified
    weights = None
    if args.weights_file and args.profile:
        weights = load_weights_from_profile(args.weights_file, args.profile)
        logger.info(f"Loaded weights from {args.weights_file}:{args.profile}")

    run_gpu_selfplay(
        board_type=args.board,
        num_players=args.num_players,
        num_games=args.num_games,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_moves=args.max_moves,
        weights=weights,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
