#!/usr/bin/env python3
"""High-parallelism gauntlet for Vast instances with many CPU cores.

DEPRECATED: Use run_gauntlet.py instead with --parallel flag:
    python scripts/run_gauntlet.py --config square8_2p --parallel 128

This script will be removed in a future release.

---
Original description:
Designed for 200+ vCPU instances. Runs many parallel games to maximize throughput.

Usage:
    python run_vast_gauntlet.py --config square8_2p --parallel 128
"""
import warnings

warnings.warn(
    "run_vast_gauntlet.py is deprecated. Use run_gauntlet.py --parallel instead.",
    DeprecationWarning,
    stacklevel=2
)
import argparse
import asyncio
import sys
import time
from pathlib import Path

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("run_vast_gauntlet")

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tournament.distributed_gauntlet import GauntletConfig, get_gauntlet


async def run_vast_gauntlet(
    config_key: str,
    parallel_games: int = 128,
    games_per_matchup: int = 2,
    num_baselines: int = 4,
):
    """Run high-parallelism gauntlet on Vast instance."""

    config = GauntletConfig(
        games_per_matchup=games_per_matchup,
        num_baselines=num_baselines,
        reserved_workers=parallel_games // 16,  # Approximate
        parallel_games=parallel_games,
        timeout_seconds=120,
        min_games_for_rating=games_per_matchup * 2,
    )

    gauntlet = get_gauntlet()
    gauntlet.config = config

    # Get unrated models
    all_unrated = gauntlet.get_unrated_models(config_key)
    total_games = len(all_unrated) * num_baselines * games_per_matchup

    logger.info(f"Vast Gauntlet: {len(all_unrated)} models, {total_games} games")
    logger.info(f"Parallelism: {parallel_games} concurrent games")

    if not all_unrated:
        logger.info("No unrated models")
        return

    # Calculate expected time
    game_time_estimate = 5  # seconds per game with enough CPUs
    batch_count = total_games / parallel_games
    estimated_time = batch_count * game_time_estimate
    logger.info(f"Estimated time: {estimated_time:.0f}s ({estimated_time/60:.1f}min)")

    start = time.time()
    result = await gauntlet.run_gauntlet(config_key)
    duration = time.time() - start

    rate = result.total_games / duration if duration > 0 else 0
    logger.info(f"Complete: {result.models_evaluated} models, {result.total_games} games")
    logger.info(f"Duration: {duration:.1f}s ({rate:.2f} games/sec)")
    logger.info(f"Throughput: {rate * 60:.1f} games/min")

    return result


def main():
    parser = argparse.ArgumentParser(description="High-parallelism Vast gauntlet")
    parser.add_argument("--config", type=str, default="square8_2p", help="Config key")
    parser.add_argument("--parallel", type=int, default=128, help="Parallel games")
    parser.add_argument("--games", type=int, default=2, help="Games per matchup")
    parser.add_argument("--baselines", type=int, default=4, help="Number of baselines")

    args = parser.parse_args()

    asyncio.run(run_vast_gauntlet(
        config_key=args.config,
        parallel_games=args.parallel,
        games_per_matchup=args.games,
        num_baselines=args.baselines,
    ))


if __name__ == "__main__":
    main()
