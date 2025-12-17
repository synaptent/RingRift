#!/usr/bin/env python3
"""Run gauntlet on a partition of models for parallel execution.

Usage:
    python run_partitioned_gauntlet.py --partition 0 --total-partitions 5 --config square8_2p

Each instance evaluates models where hash(model_id) % total_partitions == partition.
"""
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Unified logging setup
try:
    from app.core.logging_config import setup_logging
    logger = setup_logging("run_partitioned_gauntlet", log_dir="logs")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tournament.distributed_gauntlet import get_gauntlet, GauntletConfig


def hash_model_id(model_id: str) -> int:
    """Deterministic hash for partitioning (Python's hash() varies across runs)."""
    import hashlib
    return int(hashlib.md5(model_id.encode()).hexdigest(), 16)


async def run_partitioned_gauntlet(
    partition: int,
    total_partitions: int,
    config_key: str,
    games_per_matchup: int = 2,
    num_baselines: int = 4,
    parallel_games: int = 16,
):
    """Run gauntlet on a partition of models."""

    config = GauntletConfig(
        games_per_matchup=games_per_matchup,
        num_baselines=num_baselines,
        reserved_workers=4,
        parallel_games=parallel_games,
        timeout_seconds=90,
        min_games_for_rating=games_per_matchup * 2,  # Need at least half
    )

    gauntlet = get_gauntlet()
    gauntlet.config = config

    # Get all unrated models for this config
    all_unrated = gauntlet.get_unrated_models(config_key)

    # Filter to this partition
    partition_models = [
        m for m in all_unrated
        if hash_model_id(m) % total_partitions == partition
    ]

    logger.info(f"Partition {partition}/{total_partitions}: {len(partition_models)}/{len(all_unrated)} models")

    if not partition_models:
        logger.info("No models in this partition")
        return

    # Run gauntlet on partition
    # We need to override get_unrated_models to return only our partition
    original_get_unrated = gauntlet.get_unrated_models
    gauntlet.get_unrated_models = lambda ck: [m for m in original_get_unrated(ck)
                                               if hash_model_id(m) % total_partitions == partition]

    start = time.time()
    result = await gauntlet.run_gauntlet(config_key)
    duration = time.time() - start

    rate = result.total_games / duration if duration > 0 else 0
    logger.info(f"Complete: {result.models_evaluated} models, {result.total_games} games")
    logger.info(f"Duration: {duration:.1f}s ({rate:.2f} games/sec)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run partitioned gauntlet")
    parser.add_argument("--partition", type=int, required=True, help="Partition index (0-based)")
    parser.add_argument("--total-partitions", type=int, default=5, help="Total number of partitions")
    parser.add_argument("--config", type=str, default="square8_2p", help="Config key")
    parser.add_argument("--games", type=int, default=2, help="Games per matchup")
    parser.add_argument("--baselines", type=int, default=4, help="Number of baselines")
    parser.add_argument("--parallel", type=int, default=16, help="Parallel games")

    args = parser.parse_args()

    asyncio.run(run_partitioned_gauntlet(
        partition=args.partition,
        total_partitions=args.total_partitions,
        config_key=args.config,
        games_per_matchup=args.games,
        num_baselines=args.baselines,
        parallel_games=args.parallel,
    ))


if __name__ == "__main__":
    main()
