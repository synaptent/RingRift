#!/usr/bin/env python
"""GPU-accelerated CMA-ES heuristic weight optimization.

This script leverages GPU parallel game simulation for 10-100x faster
fitness evaluation compared to CPU-based CMA-ES optimization.

Uses ParallelGameRunner from gpu_parallel_games.py to run batch games
on GPU, significantly accelerating the CMA-ES optimization loop.

Usage:
    # Local GPU (single GPU)
    python scripts/run_gpu_cmaes.py \\
        --board square8 \\
        --num-players 2 \\
        --generations 50 \\
        --population-size 20 \\
        --games-per-eval 50 \\
        --output-dir logs/cmaes/gpu_square8_2p

    # Multi-GPU on same machine
    python scripts/run_gpu_cmaes.py \\
        --board square8 \\
        --num-players 2 \\
        --generations 50 \\
        --multi-gpu \\
        --output-dir logs/cmaes/gpu_multi

Requirements:
    - PyTorch with CUDA support
    - RTX 3090/4090/5090 or similar for best performance
    - For RTX 5090: PyTorch nightly with sm_120 support
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

from app.ai.gpu_batch import get_device, get_all_cuda_devices, clear_gpu_memory
from app.ai.gpu_parallel_games import (
    ParallelGameRunner,
    evaluate_candidate_fitness_gpu,
    benchmark_parallel_games,
)
from app.models import BoardType

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

WEIGHT_NAMES = list(DEFAULT_WEIGHTS.keys())
NUM_WEIGHTS = len(WEIGHT_NAMES)


def weights_to_vector(weights: Dict[str, float]) -> np.ndarray:
    """Convert weight dict to numpy vector."""
    return np.array([weights.get(name, DEFAULT_WEIGHTS[name]) for name in WEIGHT_NAMES])


def vector_to_weights(vec: np.ndarray) -> Dict[str, float]:
    """Convert numpy vector to weight dict."""
    return {name: float(vec[i]) for i, name in enumerate(WEIGHT_NAMES)}


# =============================================================================
# GPU Fitness Evaluation
# =============================================================================


class GPUFitnessEvaluator:
    """GPU-accelerated fitness evaluator for CMA-ES."""

    def __init__(
        self,
        board_size: int = 8,
        num_players: int = 2,
        games_per_eval: int = 50,
        max_moves: int = 500,
        device: Optional[torch.device] = None,
        baseline_weights: Optional[Dict[str, float]] = None,
    ):
        self.board_size = board_size
        self.num_players = num_players
        self.games_per_eval = games_per_eval
        self.max_moves = max_moves
        self.device = device or get_device()
        self.baseline_weights = baseline_weights or DEFAULT_WEIGHTS.copy()

        # Pre-create runner for reuse
        self.runner = ParallelGameRunner(
            batch_size=games_per_eval,
            board_size=board_size,
            num_players=num_players,
            device=self.device,
        )

        # Statistics
        self.total_games = 0
        self.total_time = 0.0

    def evaluate(self, candidate_weights: Dict[str, float]) -> float:
        """Evaluate a candidate's fitness against baseline."""
        start = time.time()

        win_rate = evaluate_candidate_fitness_gpu(
            candidate_weights=candidate_weights,
            opponent_weights=self.baseline_weights,
            num_games=self.games_per_eval,
            board_size=self.board_size,
            num_players=self.num_players,
            max_moves=self.max_moves,
            device=self.device,
        )

        elapsed = time.time() - start
        self.total_games += self.games_per_eval
        self.total_time += elapsed

        return win_rate

    def evaluate_population(
        self,
        population: List[np.ndarray],
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Evaluate an entire population in parallel-ish manner.

        For now, evaluates sequentially but could be parallelized across
        multiple GPUs in the future.

        Returns:
            (fitness_scores, stats_dict)
        """
        start = time.time()
        fitness_scores = []

        for i, vec in enumerate(population):
            weights = vector_to_weights(vec)
            fitness = self.evaluate(weights)
            fitness_scores.append(fitness)

            if (i + 1) % 5 == 0:
                logger.info(
                    f"  Evaluated {i+1}/{len(population)} candidates, "
                    f"best so far: {max(fitness_scores):.3f}"
                )

        elapsed = time.time() - start
        stats = {
            "population_size": len(population),
            "total_games": len(population) * self.games_per_eval,
            "elapsed_seconds": elapsed,
            "games_per_second": (len(population) * self.games_per_eval) / elapsed,
            "best_fitness": max(fitness_scores),
            "mean_fitness": np.mean(fitness_scores),
            "std_fitness": np.std(fitness_scores),
        }

        return fitness_scores, stats


class MultiGPUFitnessEvaluator:
    """Distribute fitness evaluation across multiple GPUs."""

    def __init__(
        self,
        board_size: int = 8,
        num_players: int = 2,
        games_per_eval: int = 50,
        max_moves: int = 500,
        baseline_weights: Optional[Dict[str, float]] = None,
    ):
        self.devices = get_all_cuda_devices()
        if not self.devices:
            raise RuntimeError("No CUDA devices available for multi-GPU evaluation")

        logger.info(f"Multi-GPU evaluator using {len(self.devices)} GPUs")

        self.evaluators = [
            GPUFitnessEvaluator(
                board_size=board_size,
                num_players=num_players,
                games_per_eval=games_per_eval,
                max_moves=max_moves,
                device=device,
                baseline_weights=baseline_weights,
            )
            for device in self.devices
        ]

    def evaluate_population(
        self,
        population: List[np.ndarray],
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Evaluate population distributed across GPUs."""
        import concurrent.futures

        start = time.time()
        fitness_scores = [0.0] * len(population)

        def eval_candidate(args):
            idx, vec, evaluator = args
            weights = vector_to_weights(vec)
            return idx, evaluator.evaluate(weights)

        # Distribute candidates across GPUs
        tasks = []
        for i, vec in enumerate(population):
            evaluator = self.evaluators[i % len(self.evaluators)]
            tasks.append((i, vec, evaluator))

        # Run in parallel threads (GPU operations release GIL)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = [executor.submit(eval_candidate, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                idx, fitness = future.result()
                fitness_scores[idx] = fitness

        elapsed = time.time() - start
        total_games = len(population) * self.evaluators[0].games_per_eval

        stats = {
            "population_size": len(population),
            "num_gpus": len(self.devices),
            "total_games": total_games,
            "elapsed_seconds": elapsed,
            "games_per_second": total_games / elapsed,
            "best_fitness": max(fitness_scores),
            "mean_fitness": np.mean(fitness_scores),
            "std_fitness": np.std(fitness_scores),
        }

        return fitness_scores, stats


# =============================================================================
# CMA-ES Optimization Loop
# =============================================================================


def run_gpu_cmaes(
    board_type: str,
    num_players: int,
    generations: int,
    population_size: int,
    games_per_eval: int,
    output_dir: str,
    sigma: float = 0.5,
    max_moves: int = 500,
    baseline_weights: Optional[Dict[str, float]] = None,
    multi_gpu: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run GPU-accelerated CMA-ES optimization.

    Args:
        board_type: Board type (square8, square19, hex)
        num_players: Number of players (2, 3, 4)
        generations: Number of CMA-ES generations
        population_size: Population size per generation
        games_per_eval: Games per fitness evaluation
        output_dir: Directory for checkpoints and logs
        sigma: Initial step size
        max_moves: Max moves per game
        baseline_weights: Starting weights (defaults to DEFAULT_WEIGHTS)
        multi_gpu: Use multiple GPUs if available
        seed: Random seed

    Returns:
        Dict with optimization results
    """
    try:
        import cma
    except ImportError:
        logger.error("CMA-ES library not installed. Run: pip install cma")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    # Parse board type
    board_size = {"square8": 8, "square19": 19, "hex": 25}.get(board_type.lower(), 8)

    logger.info("=" * 60)
    logger.info("GPU-ACCELERATED CMA-ES OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Board: {board_type} ({board_size}x{board_size})")
    logger.info(f"Players: {num_players}")
    logger.info(f"Generations: {generations}")
    logger.info(f"Population size: {population_size}")
    logger.info(f"Games per eval: {games_per_eval}")
    logger.info(f"Sigma: {sigma}")
    logger.info(f"Multi-GPU: {multi_gpu}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Initialize evaluator
    if multi_gpu and torch.cuda.device_count() > 1:
        evaluator = MultiGPUFitnessEvaluator(
            board_size=board_size,
            num_players=num_players,
            games_per_eval=games_per_eval,
            max_moves=max_moves,
            baseline_weights=baseline_weights,
        )
    else:
        evaluator = GPUFitnessEvaluator(
            board_size=board_size,
            num_players=num_players,
            games_per_eval=games_per_eval,
            max_moves=max_moves,
            baseline_weights=baseline_weights,
        )

    # Benchmark first
    logger.info("Running GPU benchmark...")
    bench = benchmark_parallel_games(
        batch_sizes=[games_per_eval],
        board_size=board_size,
        max_moves=50,
    )
    logger.info(f"  GPU throughput: {bench['games_per_second'][0]:.1f} games/sec")
    logger.info("")

    # Initialize CMA-ES
    x0 = weights_to_vector(baseline_weights or DEFAULT_WEIGHTS)
    es = cma.CMAEvolutionStrategy(
        x0,
        sigma,
        {
            "popsize": population_size,
            "seed": seed,
            "verbose": -1,
            "bounds": [0.0, 5.0],  # Reasonable weight bounds
        },
    )

    # Tracking
    best_fitness = 0.0
    best_weights = baseline_weights or DEFAULT_WEIGHTS.copy()
    history = []
    start_time = time.time()

    for gen in range(generations):
        gen_start = time.time()

        # Get candidate solutions
        solutions = es.ask()

        # Evaluate fitness on GPU
        logger.info(f"Generation {gen + 1}/{generations}")
        fitness_scores, stats = evaluator.evaluate_population(solutions)

        # CMA-ES uses minimization, so negate fitness (we want to maximize win rate)
        # Also clip to avoid numerical issues
        minimization_scores = [-max(0.001, f) for f in fitness_scores]
        es.tell(solutions, minimization_scores)

        # Track best
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_best_weights = vector_to_weights(solutions[gen_best_idx])

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_weights = gen_best_weights.copy()

        gen_elapsed = time.time() - gen_start

        # Log progress
        logger.info(
            f"  Best: {gen_best_fitness:.3f}, Mean: {stats['mean_fitness']:.3f}, "
            f"Std: {stats['std_fitness']:.3f}"
        )
        logger.info(
            f"  Games: {stats['total_games']}, Speed: {stats['games_per_second']:.1f} g/s, "
            f"Time: {gen_elapsed:.1f}s"
        )
        logger.info(f"  Overall best: {best_fitness:.3f}")
        logger.info("")

        # Save checkpoint
        checkpoint = {
            "generation": gen + 1,
            "fitness": gen_best_fitness,
            "best_fitness": best_fitness,
            "weights": gen_best_weights,
            "best_weights": best_weights,
            "stats": stats,
            "timestamp": datetime.now().isoformat(),
        }
        history.append(checkpoint)

        checkpoint_path = os.path.join(output_dir, f"checkpoint_gen{gen + 1:03d}.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        # Check for early stopping
        if es.stop():
            logger.info("CMA-ES converged, stopping early")
            break

    total_time = time.time() - start_time

    # Final results
    results = {
        "board_type": board_type,
        "num_players": num_players,
        "generations_completed": len(history),
        "best_fitness": best_fitness,
        "best_weights": best_weights,
        "total_time_seconds": total_time,
        "total_games": sum(h["stats"]["total_games"] for h in history),
        "history": history,
    }

    # Save final results
    results_path = os.path.join(output_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best fitness: {best_fitness:.3f}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Total games: {results['total_games']}")
    logger.info(f"Results saved to: {results_path}")
    logger.info("")
    logger.info("Best weights:")
    for name, value in best_weights.items():
        logger.info(f"  {name}: {value:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated CMA-ES heuristic optimization"
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
        "--generations",
        type=int,
        default=50,
        help="Number of CMA-ES generations",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=20,
        help="Population size per generation",
    )
    parser.add_argument(
        "--games-per-eval",
        type=int,
        default=50,
        help="Games per fitness evaluation",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=500,
        help="Maximum moves per game",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Initial CMA-ES step size",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/cmaes/gpu",
        help="Output directory for results",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use multiple GPUs if available",
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
        help="Only run GPU benchmark, don't optimize",
    )

    args = parser.parse_args()

    if args.benchmark_only:
        logger.info("Running GPU benchmark...")
        device = get_device()
        board_size = {"square8": 8, "square19": 19, "hex": 25}.get(args.board.lower(), 8)
        results = benchmark_parallel_games(
            batch_sizes=[16, 32, 64, 128, 256, 512],
            board_size=board_size,
            max_moves=100,
            device=device,
        )
        logger.info("Benchmark results:")
        for i, bs in enumerate(results["batch_size"]):
            logger.info(
                f"  Batch {bs}: {results['games_per_second'][i]:.1f} games/sec"
            )
        return

    run_gpu_cmaes(
        board_type=args.board,
        num_players=args.num_players,
        generations=args.generations,
        population_size=args.population_size,
        games_per_eval=args.games_per_eval,
        output_dir=args.output_dir,
        sigma=args.sigma,
        max_moves=args.max_moves,
        multi_gpu=args.multi_gpu,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
