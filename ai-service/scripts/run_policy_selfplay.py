#!/usr/bin/env python3
"""
Policy-only selfplay using neural network for move selection.

Uses GPUMCTSSelfplayRunner with Gumbel MCTS for high-quality training data.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.selfplay_config import SelfplayConfig, EngineMode
from app.training.gpu_mcts_selfplay import GPUMCTSSelfplayRunner
from app.db.game_replay import GameReplayDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run policy-only selfplay with neural network")
    parser.add_argument("--board-type", "-b", type=str, required=True,
                       choices=["hex8", "square8", "square19", "hexagonal"],
                       help="Board type")
    parser.add_argument("--num-players", "-p", type=int, required=True,
                       choices=[2, 3, 4], help="Number of players")
    parser.add_argument("--num-games", "-n", type=int, default=100,
                       help="Number of games to play")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for parallel games")
    parser.add_argument("--weights-file", "-w", type=str, default=None,
                       help="Path to model weights (auto-detected if not specified)")
    parser.add_argument("--simulation-budget", type=int, default=200,
                       help="MCTS simulation budget per move")
    parser.add_argument("--output-db", "-o", type=str, default=None,
                       help="Output database path")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")

    args = parser.parse_args()

    # Auto-detect model weights if not specified
    weights_file = args.weights_file
    if weights_file is None:
        weights_file = f"models/canonical_{args.board_type}_{args.num_players}p.pth"

    if not os.path.exists(weights_file):
        logger.error(f"Model weights not found: {weights_file}")
        sys.exit(1)

    # Auto-generate output db if not specified
    output_db = args.output_db
    if output_db is None:
        timestamp = int(time.time())
        output_db = f"data/games/policy_{args.board_type}_{args.num_players}p_{timestamp}.db"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_db), exist_ok=True)

    logger.info(f"Policy-only selfplay configuration:")
    logger.info(f"  Board type: {args.board_type}")
    logger.info(f"  Players: {args.num_players}")
    logger.info(f"  Games: {args.num_games}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Model: {weights_file}")
    logger.info(f"  Simulation budget: {args.simulation_budget}")
    logger.info(f"  Output: {output_db}")

    # Create selfplay config
    config = SelfplayConfig(
        board_type=args.board_type,
        num_players=args.num_players,
        num_games=args.num_games,
        batch_size=args.batch_size,
        engine_mode=EngineMode.POLICY_ONLY,
        weights_file=weights_file,
        simulation_budget=args.simulation_budget,
        seed=args.seed or 42,
    )

    # Create runner
    logger.info("Initializing GPUMCTSSelfplayRunner...")
    runner = GPUMCTSSelfplayRunner(config)

    # Run selfplay
    start_time = time.time()
    total_games = 0
    total_samples = 0

    # Run in batches
    games_remaining = args.num_games
    batch_num = 0

    # Open database
    db = GameReplayDB(output_db)

    while games_remaining > 0:
        batch_games = min(args.batch_size, games_remaining)
        batch_num += 1

        logger.info(f"Running batch {batch_num}: {batch_games} games ({total_games}/{args.num_games} complete)")

        batch_start = time.time()
        records = runner.run_batch(num_games=batch_games)
        batch_time = time.time() - batch_start

        if records:
            # Store games in database
            for record in records:
                try:
                    db.store_game(
                        game_state_json=record.to_json(),
                        board_type=args.board_type,
                        num_players=args.num_players,
                        winner=record.winner,
                        total_moves=record.total_moves,
                        source="policy_selfplay",
                    )
                    total_games += 1
                    total_samples += len(record.samples)
                except Exception as e:
                    logger.warning(f"Failed to store game: {e}")

            games_per_sec = batch_games / batch_time if batch_time > 0 else 0
            samples_per_game = sum(len(r.samples) for r in records) / len(records) if records else 0
            logger.info(f"  Batch complete: {len(records)} games, {samples_per_game:.1f} samples/game, {games_per_sec:.2f} games/sec")
        else:
            logger.warning("Batch returned no records")

        games_remaining -= batch_games

    elapsed = time.time() - start_time

    logger.info(f"\n=== Policy Selfplay Complete ===")
    logger.info(f"Total games: {total_games}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Elapsed time: {elapsed:.1f}s")
    logger.info(f"Games/sec: {total_games / elapsed:.2f}")
    logger.info(f"Samples/game: {total_samples / total_games:.1f}" if total_games > 0 else "N/A")
    logger.info(f"Output: {output_db}")

    db.close()


if __name__ == "__main__":
    main()
