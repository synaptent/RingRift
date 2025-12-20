#!/usr/bin/env python
"""Benchmark GPU self-play on cluster with optimized batch sizes.

This script benchmarks GPU-accelerated parallel self-play using the
optimized batch sizes for different GPU types (GH200, H100, A100, RTX).

Usage:
    # Run locally (auto-detects GPU and batch size)
    python scripts/benchmark_gpu_selfplay_cluster.py --num-games 1000

    # Run on cluster node with specific batch size
    python scripts/benchmark_gpu_selfplay_cluster.py --num-games 10000 --batch-size 2048

    # Generate training data
    python scripts/benchmark_gpu_selfplay_cluster.py --num-games 100000 --output data/gpu_selfplay.npz
"""

import argparse
import os
import sys
import time

# Ensure app imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def detect_gpu_and_batch_size() -> tuple:
    """Detect GPU type and return optimal batch size."""
    try:
        import torch
        if not torch.cuda.is_available():
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "MPS", 256
            return "CPU", 64

        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name.upper()
        total_memory_gb = props.total_memory / (1024 ** 3)

        # Match GPU type and set batch size
        if "GH200" in gpu_name or total_memory_gb > 90:
            return gpu_name, 4096  # 64 * 64 multiplier
        elif "H100" in gpu_name or "H200" in gpu_name or total_memory_gb > 70:
            return gpu_name, 2048  # 64 * 32 multiplier
        elif "A100" in gpu_name or total_memory_gb > 30:
            return gpu_name, 1024  # 64 * 16 multiplier
        elif total_memory_gb > 16:
            return gpu_name, 512   # 64 * 8 multiplier (RTX class)
        else:
            return gpu_name, 256   # Consumer GPU
    except Exception as e:
        print(f"GPU detection failed: {e}")
        return "Unknown", 64


def run_benchmark(num_games: int, batch_size: int, output_file: str | None = None) -> dict:
    """Run GPU self-play benchmark."""
    import torch

    from app.ai.gpu_parallel_games import ParallelGameRunner
    from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS

    print(f"\n{'='*60}")
    print("GPU Self-Play Benchmark")
    print(f"{'='*60}")

    gpu_name, auto_batch = detect_gpu_and_batch_size()
    if batch_size is None:
        batch_size = auto_batch

    print(f"GPU: {gpu_name}")
    print(f"Batch size: {batch_size}")
    print(f"Target games: {num_games}")

    # Initialize runner
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    runner = ParallelGameRunner(
        batch_size=batch_size,
        board_size=8,
        num_players=2,
        device=device,
        use_heuristic_selection=True,
        weight_noise=0.1,  # Some diversity
        temperature=1.0,
        random_opening_moves=2,  # Diverse openings
    )

    weights = dict(BASE_V1_BALANCED_WEIGHTS)

    # Run games
    total_games = 0
    total_moves = 0
    start_time = time.perf_counter()

    batches = (num_games + batch_size - 1) // batch_size

    for batch_idx in range(batches):
        batch_start = time.perf_counter()

        results = runner.run_games(
            weights_list=[weights] * batch_size,
            max_moves=10000,
        )

        batch_time = time.perf_counter() - batch_start
        batch_games = len(results['winners'])
        batch_moves = sum(results['move_counts'])

        total_games += batch_games
        total_moves += batch_moves

        games_per_sec = batch_games / batch_time
        moves_per_sec = batch_moves / batch_time

        print(f"Batch {batch_idx+1}/{batches}: {batch_games} games, "
              f"{games_per_sec:.1f} games/sec, {moves_per_sec:.0f} moves/sec")

        if total_games >= num_games:
            break

    elapsed = time.perf_counter() - start_time

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Total games: {total_games}")
    print(f"Total moves: {total_moves}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Games/sec: {total_games/elapsed:.1f}")
    print(f"Moves/sec: {total_moves/elapsed:.0f}")
    print(f"Avg moves/game: {total_moves/total_games:.1f}")

    # Estimate hourly throughput
    games_per_hour = (total_games / elapsed) * 3600
    print(f"\nProjected hourly throughput: {games_per_hour:,.0f} games/hour")

    return {
        'gpu_name': gpu_name,
        'batch_size': batch_size,
        'total_games': total_games,
        'total_moves': total_moves,
        'elapsed_seconds': elapsed,
        'games_per_second': total_games / elapsed,
        'moves_per_second': total_moves / elapsed,
        'games_per_hour': games_per_hour,
    }


def main():
    parser = argparse.ArgumentParser(description="GPU Self-Play Benchmark")
    parser.add_argument("--num-games", type=int, default=1000,
                        help="Number of games to run")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (auto-detected if not set)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for training data (optional)")
    args = parser.parse_args()

    run_benchmark(args.num_games, args.batch_size, args.output)

    print(f"\n{'='*60}")
    print("Benchmark Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
