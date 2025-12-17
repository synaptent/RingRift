#!/usr/bin/env python3
"""Run targeted gauntlet evaluation for models with insufficient games.

This script focuses on models with the fewest games played, running more
games to reach a statistically sound threshold before culling.

Usage:
    python scripts/run_targeted_gauntlet.py --config square8_2p --min-target 50
"""

import argparse
import asyncio
import logging
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tournament.distributed_gauntlet import (
    DistributedNNGauntlet,
    GauntletConfig,
    get_gauntlet,
)

# Unified logging setup
try:
    from app.core.logging_config import setup_logging
    logger = setup_logging("run_targeted_gauntlet", log_dir="logs")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)


def get_low_game_models(db_path: Path, config_key: str, max_games: int = 20) -> list:
    """Get models with games_played < max_games."""
    parts = config_key.split("_")
    board_type = parts[0]
    num_players = int(parts[1].replace("p", ""))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute("""
            SELECT participant_id, rating, games_played
            FROM elo_ratings
            WHERE board_type = ? AND num_players = ?
            AND (archived_at IS NULL OR archived_at = 0)
            AND games_played < ?
            ORDER BY games_played ASC
        """, (board_type, num_players, max_games))
        return list(cursor.fetchall())
    finally:
        conn.close()


async def run_targeted_round(
    gauntlet: DistributedNNGauntlet,
    config_key: str,
    target_models: list,
    games_per_matchup: int,
    p2p_url: str,
):
    """Run gauntlet for specific models only."""
    # Temporarily override get_unrated_models to return our target models
    original_get_unrated = gauntlet.get_unrated_models

    def targeted_unrated(cfg):
        if cfg == config_key:
            return [m['participant_id'] for m in target_models]
        return original_get_unrated(cfg)

    gauntlet.get_unrated_models = targeted_unrated
    gauntlet.config.games_per_matchup = games_per_matchup
    gauntlet.config.min_games_for_rating = 1  # Include all low-game models

    try:
        result = await gauntlet.run_gauntlet_distributed(config_key, p2p_url)
        return result
    finally:
        gauntlet.get_unrated_models = original_get_unrated


async def main():
    parser = argparse.ArgumentParser(description="Run targeted gauntlet for low-game models")
    parser.add_argument("--config", "-c", default="square8_2p", help="Config to evaluate")
    parser.add_argument("--p2p-url", default="https://p2p.ringrift.ai", help="P2P URL")
    parser.add_argument("--max-games", type=int, default=20, help="Target models with < this many games")
    parser.add_argument("--games-per-matchup", type=int, default=10, help="Games per model vs baseline")
    parser.add_argument("--rounds", type=int, default=5, help="Number of evaluation rounds")
    parser.add_argument("--batch-size", type=int, default=20, help="Models per round")
    args = parser.parse_args()

    gauntlet = get_gauntlet()
    db_path = Path(__file__).parent.parent / "data" / "unified_elo.db"

    total_evaluated = 0
    total_games = 0

    for round_num in range(args.rounds):
        # Get current low-game models
        low_game_models = get_low_game_models(db_path, args.config, args.max_games)

        if not low_game_models:
            logger.info(f"No more models with < {args.max_games} games")
            break

        # Take batch_size models with fewest games
        batch = low_game_models[:args.batch_size]

        logger.info(f"\n=== Round {round_num + 1}/{args.rounds} ===")
        logger.info(f"Evaluating {len(batch)} models with lowest game counts")
        logger.info(f"Game counts: {[m['games_played'] for m in batch[:10]]}...")

        try:
            result = await run_targeted_round(
                gauntlet=gauntlet,
                config_key=args.config,
                target_models=batch,
                games_per_matchup=args.games_per_matchup,
                p2p_url=args.p2p_url,
            )

            total_evaluated += result.models_evaluated
            total_games += result.total_games

            logger.info(f"Round {round_num + 1}: {result.models_evaluated} models, {result.total_games} games")

        except Exception as e:
            logger.error(f"Round {round_num + 1} failed: {e}")
            continue

        # Brief pause between rounds
        await asyncio.sleep(2)

    # Final stats
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total: {total_evaluated} model evaluations, {total_games} games")

    # Show remaining low-game models
    remaining = get_low_game_models(db_path, args.config, args.max_games)
    logger.info(f"Remaining models with < {args.max_games} games: {len(remaining)}")


if __name__ == "__main__":
    asyncio.run(main())
