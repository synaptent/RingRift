#!/usr/bin/env python3
"""Benchmark GPU MCTS implementations on various devices.

Usage:
    # Local (CPU/MPS)
    PYTHONPATH=. python scripts/benchmark_gpu_mcts.py

    # On cluster with CUDA
    ssh ubuntu@<gpu-node-ip>
    cd ~/ringrift/ai-service
    PYTHONPATH=. python scripts/benchmark_gpu_mcts.py --device cuda
"""

import argparse
import time
import torch

def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU MCTS")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: cpu, mps, cuda, or auto")
    parser.add_argument("--batch-sizes", type=str, default="4,8,16,32,64",
                       help="Comma-separated batch sizes to test")
    parser.add_argument("--budget", type=int, default=64,
                       help="Simulation budget per tree")
    parser.add_argument("--sampled-actions", type=int, default=8,
                       help="Number of sampled actions (k)")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("=" * 70)
    print("GPU MCTS Benchmark")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Simulation budget: {args.budget}")
    print(f"Sampled actions: {args.sampled_actions}")
    print()

    # Import after device check
    from app.testing.fixtures import create_game_state
    from app.models import BoardType
    from app.ai.tensor_gumbel_tree import (
        MultiTreeMCTS, MultiTreeMCTSConfig,
        GPUGumbelMCTS, GPUGumbelMCTSConfig,
    )

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print("Batch Size | Phase 2 (seq) | Phase 3 (par) | Speedup | Games/sec")
    print("-----------|---------------|---------------|---------|----------")

    for batch_size in batch_sizes:
        game_states = [
            create_game_state(BoardType.SQUARE8, num_players=2)
            for _ in range(batch_size)
        ]

        # Phase 2: Sequential single-tree
        try:
            config2 = GPUGumbelMCTSConfig(
                simulation_budget=args.budget,
                num_sampled_actions=args.sampled_actions,
                device=device,
            )
            mcts2 = GPUGumbelMCTS(config2)

            # Warmup
            _ = mcts2.search(game_states[0], neural_net=None)
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for gs in game_states:
                _ = mcts2.search(gs, neural_net=None)
            if device == "cuda":
                torch.cuda.synchronize()
            time_phase2 = time.perf_counter() - start
        except Exception as e:
            print(f"Phase 2 error: {e}")
            time_phase2 = float('inf')

        # Phase 3: Parallel multi-tree
        try:
            config3 = MultiTreeMCTSConfig(
                simulation_budget=args.budget,
                num_sampled_actions=args.sampled_actions,
                device=device,
            )
            mcts3 = MultiTreeMCTS(config3)

            # Warmup
            _ = mcts3.search_batch(game_states[:min(4, batch_size)], neural_net=None)
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = mcts3.search_batch(game_states, neural_net=None)
            if device == "cuda":
                torch.cuda.synchronize()
            time_phase3 = time.perf_counter() - start
        except Exception as e:
            print(f"Phase 3 error: {e}")
            time_phase3 = float('inf')

        # Results
        speedup = time_phase2 / time_phase3 if time_phase3 > 0 else 0
        games_per_sec = batch_size / time_phase3 if time_phase3 > 0 else 0

        print(f"{batch_size:^10} | {time_phase2*1000:>10.1f}ms | {time_phase3*1000:>10.1f}ms | {speedup:>6.1f}x | {games_per_sec:>8.1f}")

    print()
    print("=" * 70)
    print("Benchmark complete")


if __name__ == "__main__":
    main()
