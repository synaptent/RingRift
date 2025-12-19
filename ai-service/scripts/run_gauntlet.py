#!/usr/bin/env python3
"""Run distributed NN gauntlet evaluation across the P2P cluster.

This script evaluates unrated neural network models against fixed baselines
using O(n) efficiency instead of O(n²) round-robin.

Usage:
    # Evaluate all configs
    python scripts/run_gauntlet.py

    # Evaluate specific config
    python scripts/run_gauntlet.py --config square8_2p

    # Run locally (no cluster)
    python scripts/run_gauntlet.py --local

    # Dry run (show what would be evaluated)
    python scripts/run_gauntlet.py --dry-run
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tournament.distributed_gauntlet import (
    DistributedNNGauntlet,
    GauntletConfig,
    CONFIG_KEYS,
    get_gauntlet,
)
from app.tournament.model_culling import (
    ModelCullingController,
    get_culling_controller,
)

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("run_gauntlet")


async def run_gauntlet_for_config(
    gauntlet: DistributedNNGauntlet,
    culler: ModelCullingController,
    config_key: str,
    distributed: bool = True,
    p2p_url: str = None,
) -> dict:
    """Run gauntlet and culling for a single config.

    Args:
        gauntlet: Gauntlet evaluator
        culler: Model culler
        config_key: Config to evaluate
        distributed: Whether to use distributed execution
        p2p_url: P2P orchestrator URL

    Returns:
        Result dict
    """
    logger.info(f"=== Processing {config_key} ===")

    # Check current state
    model_count = gauntlet.count_models(config_key)
    unrated = gauntlet.get_unrated_models(config_key)
    baselines = gauntlet.select_baselines(config_key)

    logger.info(f"Models: {model_count}, Unrated: {len(unrated)}")
    logger.info(f"Baselines: {baselines}")

    if not unrated:
        logger.info(f"No unrated models for {config_key}")
        return {
            "config_key": config_key,
            "status": "no_work",
            "models_evaluated": 0,
        }

    # Run gauntlet
    if distributed:
        result = await gauntlet.run_gauntlet_distributed(config_key, p2p_url)
    else:
        result = await gauntlet.run_gauntlet(config_key)

    logger.info(
        f"Gauntlet {result.run_id}: {result.models_evaluated} models, "
        f"{result.total_games} games, status={result.status}"
    )

    # Check if culling needed
    new_count = gauntlet.count_models(config_key)
    if new_count > culler.CULL_THRESHOLD:
        logger.info(f"Model count ({new_count}) > threshold ({culler.CULL_THRESHOLD}), culling...")
        cull_result = culler.check_and_cull(config_key)
        logger.info(f"Culled {cull_result.culled} models, kept {cull_result.kept}")
    else:
        cull_result = None

    return {
        "config_key": config_key,
        "status": result.status,
        "run_id": result.run_id,
        "models_evaluated": result.models_evaluated,
        "total_games": result.total_games,
        "duration_sec": (result.completed_at or 0) - result.started_at,
        "culled": cull_result.culled if cull_result else 0,
    }


async def main():
    parser = argparse.ArgumentParser(description="Run distributed NN gauntlet evaluation")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Specific config to evaluate (e.g., square8_2p). If not specified, evaluates all.",
    )
    parser.add_argument(
        "--local", "-l",
        action="store_true",
        help="Run locally without distributed execution",
    )
    parser.add_argument(
        "--p2p-url",
        type=str,
        default=os.environ.get("RINGRIFT_P2P_URL", "http://localhost:8770"),
        help="P2P orchestrator URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be evaluated without running",
    )
    parser.add_argument(
        "--games-per-matchup",
        type=int,
        default=10,
        help="Games per model vs baseline matchup",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=5,
        help="Minimum games for a model to be considered rated",
    )

    args = parser.parse_args()

    # Configure gauntlet
    config = GauntletConfig(
        games_per_matchup=args.games_per_matchup,
        min_games_for_rating=args.min_games,
    )

    # Initialize components
    gauntlet = get_gauntlet()
    gauntlet.config = config
    culler = get_culling_controller()

    # Determine configs to process
    if args.config:
        if args.config not in CONFIG_KEYS:
            logger.error(f"Invalid config: {args.config}. Valid: {CONFIG_KEYS}")
            sys.exit(1)
        configs = [args.config]
    else:
        configs = CONFIG_KEYS

    logger.info(f"Configs to process: {configs}")
    logger.info(f"Distributed: {not args.local}")
    logger.info(f"P2P URL: {args.p2p_url}")

    if args.dry_run:
        logger.info("\n=== DRY RUN - Not executing ===\n")
        for config_key in configs:
            model_count = gauntlet.count_models(config_key)
            unrated = gauntlet.get_unrated_models(config_key)
            baselines = gauntlet.select_baselines(config_key)

            total_games = len(unrated) * len(baselines) * config.games_per_matchup

            logger.info(f"{config_key}:")
            logger.info(f"  Total models: {model_count}")
            logger.info(f"  Unrated models: {len(unrated)}")
            logger.info(f"  Baselines: {baselines}")
            logger.info(f"  Games to play: {total_games}")
            logger.info(f"  Needs culling: {model_count > culler.CULL_THRESHOLD}")
            logger.info("")
        return

    # Run gauntlet for each config
    results = []
    for config_key in configs:
        try:
            result = await run_gauntlet_for_config(
                gauntlet=gauntlet,
                culler=culler,
                config_key=config_key,
                distributed=not args.local,
                p2p_url=args.p2p_url,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {config_key}: {e}")
            results.append({
                "config_key": config_key,
                "status": "error",
                "error": str(e),
            })

    # Summary
    logger.info("\n=== Summary ===")
    total_evaluated = 0
    total_games = 0
    total_culled = 0

    for result in results:
        config_key = result["config_key"]
        status = result["status"]
        evaluated = result.get("models_evaluated", 0)
        games = result.get("total_games", 0)
        culled = result.get("culled", 0)
        duration = result.get("duration_sec", 0)

        total_evaluated += evaluated
        total_games += games
        total_culled += culled

        logger.info(
            f"{config_key}: {status}, {evaluated} models, "
            f"{games} games, {culled} culled ({duration:.1f}s)"
        )

    logger.info(f"\nTotal: {total_evaluated} models evaluated, {total_games} games, {total_culled} culled")


if __name__ == "__main__":
    asyncio.run(main())
