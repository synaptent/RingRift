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
import fcntl
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
from app.models.core import BoardType
from app.training.generate_data import create_initial_state

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
    """Generate self-play games using GPU parallel simulation.

    Shadow Validation (Phase 2):
        When enabled, a subset of GPU-generated moves are validated against
        the canonical CPU rules engine. This catches GPU/CPU divergence early.

        Configuration:
            shadow_validation: Enable shadow validation
            shadow_sample_rate: Fraction of moves to validate (default 0.05 = 5%)
            shadow_threshold: Max divergence rate before error (default 0.001 = 0.1%)

        See GPU_PIPELINE_ROADMAP.md Section 7.4.2 for architecture details.
    """

    def __init__(
        self,
        board_size: int = 8,
        num_players: int = 2,
        batch_size: int = 256,
        max_moves: int = 500,
        device: Optional[torch.device] = None,
        weights: Optional[Dict[str, float]] = None,
        engine_mode: str = "heuristic-only",
        shadow_validation: bool = False,
        shadow_sample_rate: float = 0.05,
        shadow_threshold: float = 0.001,
        lps_victory_rounds: int = 2,
        rings_per_player: Optional[int] = None,
    ):
        self.board_size = board_size
        self.num_players = num_players
        self.batch_size = batch_size
        self.max_moves = max_moves
        self.device = device or get_device()
        self.engine_mode = engine_mode
        self.shadow_validation = shadow_validation
        self.lps_victory_rounds = lps_victory_rounds
        self.rings_per_player = rings_per_player
        # For random-only mode, use None weights (uniform random)
        # For heuristic-only mode, use provided weights or defaults
        if engine_mode == "random-only":
            self.weights = None  # Triggers uniform random selection
        else:
            self.weights = weights or DEFAULT_WEIGHTS.copy()

        self.runner = ParallelGameRunner(
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
            device=self.device,
            shadow_validation=shadow_validation,
            shadow_sample_rate=shadow_sample_rate,
            shadow_threshold=shadow_threshold,
            lps_victory_rounds=lps_victory_rounds,
            rings_per_player=rings_per_player,
        )

        # Log shadow validation status
        if shadow_validation:
            logger.info(f"Shadow validation ENABLED: sample_rate={shadow_sample_rate}, threshold={shadow_threshold}")
        else:
            logger.info("Shadow validation disabled")

        # Statistics
        self.total_games = 0
        self.total_moves = 0
        self.total_time = 0.0
        self.wins_by_player = {i: 0 for i in range(1, num_players + 1)}
        self.draws = 0

        # Pre-compute initial state for training data compatibility
        # All GPU games start from the same initial state with custom rules applied
        board_type_map = {8: BoardType.SQUARE8, 19: BoardType.SQUARE19}
        board_type = board_type_map.get(board_size, BoardType.SQUARE8)
        self._initial_state = create_initial_state(
            board_type,
            num_players,
            rings_per_player_override=rings_per_player,
            lps_rounds_required=lps_victory_rounds,
        )
        self._initial_state_json = self._initial_state.model_dump(mode="json")

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
        # Pass None for random mode (uniform random), weights for heuristic mode
        weights_list = None if self.weights is None else [self.weights] * self.batch_size
        results = self.runner.run_games(
            weights_list=weights_list,
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
            # Acquire exclusive lock to prevent JSONL corruption from concurrent writes
            try:
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.error(f"Cannot acquire lock on {output_file} - another process is writing")
                file_handle.close()
                sys.exit(1)

        try:
            for batch_idx in range(num_batches):
                # Adjust batch size for last batch
                remaining = num_games - len(all_records)
                actual_batch = min(self.batch_size, remaining)

                # Generate batch
                results = self.generate_batch(seed=batch_idx * 1000)

                # Create game records
                board_type_str = {8: "square8", 19: "square19"}.get(self.board_size, "square8")
                for i in range(actual_batch):
                    game_idx = len(all_records)
                    vtype = results["victory_types"][i]
                    record = {
                        # === Core game identifiers ===
                        "game_id": f"gpu_{board_type_str}_{self.num_players}p_{game_idx}_{int(datetime.now().timestamp())}",
                        "board_type": board_type_str,  # Standardized: square8, square19, hexagonal
                        "board_size": self.board_size,  # Legacy field for compatibility
                        "num_players": self.num_players,
                        # === Game outcome ===
                        "winner": int(results["winners"][i]),
                        "move_count": int(results["move_counts"][i]),
                        "max_moves": self.max_moves,
                        "status": "completed",
                        "game_status": "completed",
                        "victory_type": vtype,
                        "stalemate_tiebreaker": results["stalemate_tiebreakers"][i],
                        "termination_reason": f"status:completed:{vtype}",
                        # === Engine/opponent metadata ===
                        "engine_mode": "gpu_heuristic",
                        "opponent_type": "selfplay",
                        "player_types": ["gpu_batch"] * self.num_players,
                        "batch_id": batch_idx,
                        # === Training data (required for NPZ export) ===
                        "moves": results["move_histories"][i],
                        "initial_state": self._initial_state_json,
                        # === Timing metadata ===
                        "timestamp": datetime.now().isoformat(),
                        "created_at": datetime.now().isoformat(),
                        # === Source tracking ===
                        "source": "run_gpu_selfplay.py",
                        "device": str(self.device),
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

        # Add shadow validation stats if enabled
        shadow_report = self.runner.get_shadow_validation_report()
        if shadow_report:
            stats["shadow_validation"] = shadow_report
            logger.info(f"Shadow validation report: {shadow_report['status']}")
            if shadow_report.get("divergence_rate", 0) > 0:
                logger.warning(f"  Divergence rate: {shadow_report['divergence_rate']:.4%}")
                logger.warning(f"  Total divergences: {shadow_report.get('total_divergences', 0)}")

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
    engine_mode: str = "heuristic-only",
    seed: int = 42,
    shadow_validation: bool = False,
    shadow_sample_rate: float = 0.05,
    shadow_threshold: float = 0.001,
    lps_victory_rounds: int = 2,
    rings_per_player: Optional[int] = None,
) -> Dict[str, Any]:
    """Run GPU-accelerated self-play generation.

    Args:
        board_type: Board type (square8, square19, hex, hexagonal)
        num_players: Number of players
        num_games: Total games to generate
        output_dir: Output directory
        batch_size: GPU batch size
        max_moves: Max moves per game
        weights: Heuristic weights (ignored in random-only mode)
        engine_mode: Engine mode (random-only or heuristic-only)
        seed: Random seed
        shadow_validation: Enable shadow validation (GPU/CPU parity checking)
        shadow_sample_rate: Fraction of moves to validate (default 5%)
        shadow_threshold: Max divergence rate before error (default 0.1%)
        lps_victory_rounds: LPS victory threshold (default 2)
        rings_per_player: Starting rings per player (None = board default)

    Returns:
        Statistics dict
    """
    # Create output directory with explicit error handling
    output_dir = os.path.abspath(output_dir)  # Resolve to absolute path
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory created/verified: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise

    np.random.seed(seed)
    torch.manual_seed(seed)

    board_size = {"square8": 8, "square19": 19, "hex": 25, "hexagonal": 25}.get(board_type.lower(), 8)

    logger.info("=" * 60)
    logger.info("GPU-ACCELERATED SELF-PLAY GENERATION")
    logger.info("=" * 60)
    logger.info(f"Board: {board_type} ({board_size}x{board_size})")
    logger.info(f"Players: {num_players}")
    logger.info(f"Games: {num_games}")
    logger.info(f"Engine mode: {engine_mode}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max moves: {max_moves}")
    logger.info(f"LPS victory rounds: {lps_victory_rounds}")
    logger.info(f"Rings per player: {rings_per_player or 'board default'}")
    logger.info(f"Shadow validation: {shadow_validation}")
    if shadow_validation:
        logger.info(f"  Sample rate: {shadow_sample_rate:.1%}")
        logger.info(f"  Threshold: {shadow_threshold:.2%}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Create generator
    generator = GPUSelfPlayGenerator(
        board_size=board_size,
        num_players=num_players,
        batch_size=batch_size,
        max_moves=max_moves,
        weights=weights,
        engine_mode=engine_mode,
        shadow_validation=shadow_validation,
        shadow_sample_rate=shadow_sample_rate,
        shadow_threshold=shadow_threshold,
        lps_victory_rounds=lps_victory_rounds,
        rings_per_player=rings_per_player,
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
    # Ensure directory exists right before write (handles race conditions on remote hosts)
    os.makedirs(os.path.dirname(stats_file) or ".", exist_ok=True)
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
        choices=["square8", "square19", "hex", "hexagonal"],
        help="Board type",
    )
    parser.add_argument(
        "--engine-mode",
        type=str,
        default="heuristic-only",
        choices=["random-only", "heuristic-only"],
        help="Engine mode: random-only (uniform random moves) or heuristic-only (weighted by GPU heuristic)",
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

    # Shadow validation options
    parser.add_argument(
        "--shadow-validation",
        action="store_true",
        help="Enable shadow validation (GPU/CPU parity checking)",
    )
    parser.add_argument(
        "--shadow-sample-rate",
        type=float,
        default=0.05,
        help="Fraction of moves to validate against CPU (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--shadow-threshold",
        type=float,
        default=0.001,
        help="Max divergence rate before error (default: 0.001 = 0.1%%)",
    )

    # Game rule configuration
    parser.add_argument(
        "--lps-victory-rounds",
        type=int,
        default=2,
        help="LPS victory threshold in consecutive rounds (default: 2)",
    )
    parser.add_argument(
        "--rings-per-player",
        type=int,
        default=None,
        help="Starting rings per player (default: board default - 18/72/96)",
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
        engine_mode=args.engine_mode,
        seed=args.seed,
        shadow_validation=args.shadow_validation,
        shadow_sample_rate=args.shadow_sample_rate,
        shadow_threshold=args.shadow_threshold,
        lps_victory_rounds=args.lps_victory_rounds,
        rings_per_player=args.rings_per_player,
    )


if __name__ == "__main__":
    main()
