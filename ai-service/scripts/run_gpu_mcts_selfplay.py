#!/usr/bin/env python3
"""Run selfplay using GPU-accelerated MultiTreeMCTS.

This script runs selfplay games using the Phase 3 GPU MCTS implementation,
which provides 3-4x speedup over sequential processing.

Usage:
    # Run on CPU (local testing)
    PYTHONPATH=. python scripts/run_gpu_mcts_selfplay.py --device cpu --games 10

    # Run on CUDA GPU (cluster)
    PYTHONPATH=. python scripts/run_gpu_mcts_selfplay.py --device cuda --games 100

    # With specific configuration
    PYTHONPATH=. python scripts/run_gpu_mcts_selfplay.py \
        --board-type hex8 \
        --num-players 2 \
        --batch-size 16 \
        --budget 64 \
        --device cuda
"""

import argparse
import asyncio
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Run GPU MCTS Selfplay")
    parser.add_argument("--board-type", type=str, default="square8",
                       help="Board type: square8, hex8, square19, hexagonal")
    parser.add_argument("--num-players", type=int, default=2,
                       help="Number of players: 2, 3, or 4")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Number of games to run in parallel")
    parser.add_argument("--budget", type=int, default=64,
                       help="MCTS simulation budget per move")
    parser.add_argument("--games", type=int, default=10,
                       help="Total number of games to run")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cpu, cuda, mps")
    parser.add_argument("--output-dir", type=str, default="data/selfplay/gpu_mcts",
                       help="Output directory for games")
    parser.add_argument("--eval-mode", type=str, default="heuristic",
                       choices=["heuristic", "nn", "hybrid"],
                       help="Evaluation mode: heuristic (fast), nn (accurate), hybrid")
    args = parser.parse_args()

    print("=" * 70)
    print("GPU MCTS Selfplay")
    print("=" * 70)
    print(f"Board type: {args.board_type}")
    print(f"Players: {args.num_players}")
    print(f"Batch size: {args.batch_size}")
    print(f"Budget: {args.budget}")
    print(f"Total games: {args.games}")
    print(f"Device: {args.device}")
    print(f"Eval mode: {args.eval_mode}")
    print()

    from app.training.event_driven_selfplay import EventDrivenSelfplay

    # Create manager with GPU MCTS
    manager = EventDrivenSelfplay(
        board_type=args.board_type,
        num_players=args.num_players,
        batch_size=args.batch_size,
        mcts_sims=args.budget,
        output_dir=args.output_dir,
        use_gpu_mcts=True,
        gpu_device=args.device,
        gpu_eval_mode=args.eval_mode,
    )

    # Run games
    start_time = time.perf_counter()
    await manager.start()

    try:
        games = await manager.run_games(num_games=args.games)
    finally:
        await manager.stop()

    elapsed = time.perf_counter() - start_time

    # Print summary
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Games completed: {len(games)}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Games per second: {len(games)/elapsed:.2f}")
    print(f"Total moves: {manager._stats.total_moves}")
    print(f"Average moves per game: {manager._stats.total_moves / max(len(games), 1):.1f}")
    print()
    print("Win distribution:")
    for player, wins in sorted(manager._stats.wins_by_player.items()):
        print(f"  Player {player}: {wins} wins ({wins/len(games)*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
