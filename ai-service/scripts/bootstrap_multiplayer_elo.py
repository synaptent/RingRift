#!/usr/bin/env python3
"""Bootstrap Elo ratings for 3p/4p configurations.

This script initializes Elo ratings for multiplayer (3p/4p) configurations
that have trained models but no tournament evaluation history.

Usage:
    PYTHONPATH=. python scripts/bootstrap_multiplayer_elo.py

    # Dry run (show what would be done)
    PYTHONPATH=. python scripts/bootstrap_multiplayer_elo.py --dry-run

    # Run quick evaluations
    PYTHONPATH=. python scripts/bootstrap_multiplayer_elo.py --evaluate

December 2025: Created to address missing 3p/4p Elo visibility.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Canonical model naming
BOARD_TYPES = ["hex8", "square8", "square19", "hexagonal"]
PLAYER_COUNTS = [2, 3, 4]
MODELS_DIR = PROJECT_ROOT / "models"


def find_missing_elo_configs() -> list[tuple[str, int]]:
    """Find configs that have models but no Elo ratings."""
    try:
        from app.training.elo_service import get_elo_service
    except ImportError:
        logger.error("Failed to import elo_service")
        return []

    elo = get_elo_service()
    missing = []

    for board in BOARD_TYPES:
        for players in PLAYER_COUNTS:
            model_path = MODELS_DIR / f"canonical_{board}_{players}p.pth"

            if not model_path.exists():
                logger.debug(f"No model for {board}_{players}p")
                continue

            # Check if any ratings exist for this config
            try:
                leaderboard = elo.get_leaderboard(board, players, limit=1)
                if not leaderboard:
                    missing.append((board, players))
                    logger.info(f"Missing Elo: {board}_{players}p (model exists)")
            except Exception as e:
                logger.warning(f"Error checking {board}_{players}p: {e}")
                missing.append((board, players))

    return missing


def init_baseline_ratings(board_type: str, num_players: int, dry_run: bool = False) -> None:
    """Initialize baseline ratings for a config.

    Creates initial Elo entries for:
    - random baseline (1200)
    - heuristic baseline (1500)
    - canonical model (1500)
    """
    try:
        from app.training.elo_service import get_elo_service
    except ImportError:
        logger.error("Failed to import elo_service")
        return

    elo = get_elo_service()
    config_key = f"{board_type}_{num_players}p"

    # Random baseline is anchored at 400 Elo (set by BASELINE_ELO_RANDOM in elo_service)
    # Heuristic and models start at 1500
    baselines = [
        ("none:random", "random", "Random baseline (anchored at 400)"),
        ("none:heuristic:d2", "heuristic", "Heuristic baseline (depth 2)"),
        (f"canonical_{config_key}", "neural_net", f"Canonical {config_key} model"),
    ]

    for participant_id, ai_type, description in baselines:
        if dry_run:
            logger.info(f"  [DRY RUN] Would create: {participant_id}")
            continue

        try:
            # Register participant (idempotent)
            elo.register_participant(
                participant_id=participant_id,
                ai_type=ai_type,
            )

            # Create rating entry via get_rating
            # Random participants get anchored at 400, others at 1500
            rating = elo.get_rating(participant_id, board_type, num_players)

            logger.info(f"  Created: {participant_id} at {rating.rating}")
        except Exception as e:
            logger.warning(f"  Failed to create {participant_id}: {e}")


async def run_quick_evaluation(board_type: str, num_players: int) -> None:
    """Run quick gauntlet evaluation for a config."""
    try:
        from app.training.game_gauntlet import GameGauntlet, GauntletConfig
    except ImportError:
        logger.error("Failed to import GameGauntlet")
        return

    model_path = MODELS_DIR / f"canonical_{board_type}_{num_players}p.pth"

    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return

    config = GauntletConfig(
        model_path=str(model_path),
        board_type=board_type,
        num_players=num_players,
        games_per_baseline=5,  # Quick evaluation
        baselines=["random", "heuristic"],
    )

    gauntlet = GameGauntlet(config)

    try:
        logger.info(f"Running quick evaluation for {board_type}_{num_players}p...")
        results = await gauntlet.run()
        logger.info(f"  Results: {results.summary()}")
    except Exception as e:
        logger.error(f"  Evaluation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap multiplayer Elo ratings")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--evaluate", action="store_true", help="Run quick gauntlet evaluations")
    parser.add_argument("--config", type=str, help="Specific config (e.g., hex8_3p)")
    args = parser.parse_args()

    # Find missing configs
    missing = find_missing_elo_configs()

    if not missing:
        logger.info("All configs have Elo ratings!")
        return

    logger.info(f"Found {len(missing)} configs missing Elo ratings:")
    for board, players in missing:
        logger.info(f"  - {board}_{players}p")

    # Filter by specific config if requested
    if args.config:
        parts = args.config.replace("_", " ").replace("p", "").split()
        if len(parts) == 2:
            board, players = parts[0], int(parts[1])
            missing = [(board, players)] if (board, players) in missing else []

    if not missing:
        logger.info("No matching configs to process")
        return

    # Initialize baseline ratings
    logger.info("\nInitializing baseline ratings...")
    for board, players in missing:
        logger.info(f"\n{board}_{players}p:")
        init_baseline_ratings(board, players, dry_run=args.dry_run)

    # Optionally run evaluations
    if args.evaluate and not args.dry_run:
        logger.info("\nRunning quick evaluations...")
        for board, players in missing:
            asyncio.run(run_quick_evaluation(board, players))

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
