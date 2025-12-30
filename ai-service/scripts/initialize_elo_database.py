#!/usr/bin/env python3
"""Initialize Elo database with all canonical models and baselines.

This script seeds the unified_elo.db with:
- All 12 canonical neural network models (one per board/player config)
- Baseline AI participants (random, heuristic) for evaluation

Usage:
    python scripts/initialize_elo_database.py

    # Verify:
    sqlite3 data/elo/unified_elo.db "SELECT COUNT(*) FROM elo_ratings; SELECT COUNT(*) FROM participants;"
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.elo_service import get_elo_service

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# All 12 canonical model configurations
CANONICAL_MODELS = [
    ("canonical_hex8_2p", "hex8", 2),
    ("canonical_hex8_3p", "hex8", 3),
    ("canonical_hex8_4p", "hex8", 4),
    ("canonical_square8_2p", "square8", 2),
    ("canonical_square8_3p", "square8", 3),
    ("canonical_square8_4p", "square8", 4),
    ("canonical_square19_2p", "square19", 2),
    ("canonical_square19_3p", "square19", 3),
    ("canonical_square19_4p", "square19", 4),
    ("canonical_hexagonal_2p", "hexagonal", 2),
    ("canonical_hexagonal_3p", "hexagonal", 3),
    ("canonical_hexagonal_4p", "hexagonal", 4),
]

# Baseline AI types for evaluation (random anchored at 400 Elo)
BASELINES = [
    ("random", "RANDOM"),
    ("heuristic", "HEURISTIC"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize Elo database with canonical models")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--force", action="store_true", help="Re-register even if models already exist")
    args = parser.parse_args()

    elo = get_elo_service()
    logger.info(f"Using Elo database: {elo.db_path}")

    # Check existing state
    try:
        existing_count = elo.get_total_ratings()
        logger.info(f"Existing ratings in database: {existing_count}")
    except (sqlite3.Error, OSError, ValueError):
        existing_count = 0

    if existing_count > 0 and not args.force:
        logger.info("Database already has ratings. Use --force to re-register.")
        # Still show current state
        _show_current_state(elo)
        return 0

    if args.dry_run:
        logger.info("\n=== DRY RUN - No changes will be made ===\n")

    # Register baselines first (random anchored at 400 Elo)
    logger.info("\nRegistering baseline AI participants...")
    for participant_id, ai_type in BASELINES:
        if args.dry_run:
            logger.info(f"  Would register: {participant_id} (type={ai_type})")
        else:
            elo.register_participant(participant_id, ai_type=ai_type)
            logger.info(f"  Registered: {participant_id} (type={ai_type})")

    # Register all canonical models
    logger.info("\nRegistering canonical neural network models...")
    for model_id, board_type, num_players in CANONICAL_MODELS:
        model_path = f"models/{model_id}.pth"
        config_key = f"{board_type}_{num_players}p"

        if args.dry_run:
            logger.info(f"  Would register: {model_id} for {config_key}")
        else:
            elo.register_model(model_id, board_type, num_players, model_path)
            # Get the initial rating that was created
            rating = elo.get_rating(model_id, board_type, num_players)
            logger.info(f"  Registered: {model_id} for {config_key} (initial Elo: {rating.rating:.0f})")

    if not args.dry_run:
        _show_current_state(elo)

    return 0


def _show_current_state(elo) -> None:
    """Show current state of the Elo database."""
    logger.info("\n=== Current Elo Database State ===")

    try:
        total_ratings = elo.get_total_ratings()
        logger.info(f"Total ratings: {total_ratings}")
    except (sqlite3.Error, OSError, ValueError) as e:
        logger.warning(f"Could not get total ratings: {e}")

    # Show leaderboard for each config
    for board_type in ["hex8", "square8", "square19", "hexagonal"]:
        for num_players in [2, 3, 4]:
            try:
                leaderboard = elo.get_leaderboard(board_type, num_players, limit=5)
                if leaderboard:
                    logger.info(f"\n{board_type}_{num_players}p leaderboard:")
                    for entry in leaderboard:
                        logger.info(f"  {entry.rank}. {entry.participant_id}: {entry.rating:.0f} ({entry.games_played} games)")
            except (sqlite3.Error, OSError, ValueError, KeyError):
                pass


if __name__ == "__main__":
    sys.exit(main())
