#!/usr/bin/env python3
"""Benchmark GPU-accelerated Gumbel MCTS.

Compares sequential vs batched neural network evaluation performance.

Usage:
    # Run with GPU batching (default)
    python scripts/benchmark_gumbel_gpu.py

    # Force CPU/sequential mode
    RINGRIFT_GPU_GUMBEL_DISABLE=1 python scripts/benchmark_gumbel_gpu.py

    # Enable shadow validation
    RINGRIFT_GPU_GUMBEL_SHADOW_VALIDATE=1 python scripts/benchmark_gumbel_gpu.py
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.models import AIConfig
from app.training.initial_state import create_initial_state
from app.rules.default_engine import DefaultRulesEngine
from scripts.lib.logging_config import setup_script_logging


def benchmark_gumbel_mcts(
    num_moves: int = 10,
    budget: int = 50,
    num_sampled: int = 8,
    verbose: bool = False,
) -> dict:
    """Benchmark Gumbel MCTS move selection.

    Args:
        num_moves: Number of moves to benchmark.
        budget: Simulation budget per move.
        num_sampled: Number of Gumbel-sampled actions.
        verbose: Enable verbose logging.

    Returns:
        Dict with benchmark results.
    """
    from app.ai.gumbel_mcts_ai import GumbelMCTSAI

    # Create AI config (using fresh weights since no model may be available)
    config = AIConfig(
        type="gumbel-mcts",
        difficulty=7,
        gumbel_num_sampled_actions=num_sampled,
        gumbel_simulation_budget=budget,
        allow_fresh_weights=True,
    )

    # Create AI instance
    ai = GumbelMCTSAI(player_number=1, config=config)
    print(f"AI: {ai}")

    # Create game state
    game_state = create_initial_state()
    engine = DefaultRulesEngine()

    # Warmup (first move may include initialization overhead)
    print("\nWarming up...")
    warmup_state = create_initial_state()
    ai.select_move(warmup_state)

    # Benchmark moves
    print(f"\nBenchmarking {num_moves} moves (budget={budget}, m={num_sampled})...")
    times = []
    current_state = game_state

    for i in range(num_moves):
        # Make a move
        start = time.perf_counter()
        move = ai.select_move(current_state)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if move is None:
            print(f"  Move {i+1}: No valid moves, game over")
            break

        if verbose:
            print(f"  Move {i+1}: {move} ({elapsed*1000:.1f}ms)")

        # Apply move
        current_state = engine.apply_move(current_state, move)
        if current_state.game_status == "completed":
            print(f"  Game completed after {i+1} moves")
            break

    if not times:
        return {"error": "No moves made"}

    avg_time = sum(times) / len(times)
    total_time = sum(times)

    results = {
        "num_moves": len(times),
        "total_time_s": total_time,
        "avg_time_ms": avg_time * 1000,
        "moves_per_sec": len(times) / total_time,
        "budget": budget,
        "num_sampled": num_sampled,
        "gpu_available": ai._gpu_available,
        "gpu_device": str(ai._gpu_device) if ai._gpu_device else "N/A",
    }

    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg time per move: {avg_time*1000:.1f}ms")
    print(f"  Moves/sec: {results['moves_per_sec']:.2f}")
    print(f"  GPU available: {results['gpu_available']}")
    print(f"  GPU device: {results['gpu_device']}")

    return results


def compare_gpu_vs_cpu(
    num_moves: int = 5,
    budget: int = 50,
    num_sampled: int = 8,
) -> None:
    """Compare GPU batch vs CPU sequential performance."""
    print("=" * 60)
    print("Gumbel MCTS GPU Acceleration Benchmark")
    print("=" * 60)

    # Run with GPU batching enabled
    print("\n--- GPU Batch Mode ---")
    os.environ.pop("RINGRIFT_GPU_GUMBEL_DISABLE", None)
    # Need to reimport to pick up env var change
    import importlib
    from app.ai import gumbel_mcts_ai
    importlib.reload(gumbel_mcts_ai)

    gpu_results = benchmark_gumbel_mcts(num_moves, budget, num_sampled)

    # Run with CPU/sequential mode
    print("\n--- CPU Sequential Mode ---")
    os.environ["RINGRIFT_GPU_GUMBEL_DISABLE"] = "1"
    importlib.reload(gumbel_mcts_ai)

    cpu_results = benchmark_gumbel_mcts(num_moves, budget, num_sampled)

    # Compare
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)

    if "error" not in gpu_results and "error" not in cpu_results:
        speedup = cpu_results["avg_time_ms"] / gpu_results["avg_time_ms"]
        print(f"GPU avg: {gpu_results['avg_time_ms']:.1f}ms")
        print(f"CPU avg: {cpu_results['avg_time_ms']:.1f}ms")
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("Error during benchmark")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Gumbel MCTS GPU acceleration")
    parser.add_argument("-n", "--num-moves", type=int, default=10, help="Number of moves")
    parser.add_argument("-b", "--budget", type=int, default=50, help="Simulation budget")
    parser.add_argument("-m", "--num-sampled", type=int, default=8, help="Gumbel-sampled actions")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--compare", action="store_true", help="Compare GPU vs CPU")

    args = parser.parse_args()

    setup_script_logging("benchmark_gumbel_gpu", level="DEBUG" if args.verbose else "INFO")

    if args.compare:
        compare_gpu_vs_cpu(args.num_moves, args.budget, args.num_sampled)
    else:
        benchmark_gumbel_mcts(args.num_moves, args.budget, args.num_sampled, args.verbose)
