#!/usr/bin/env python3
"""Register canonical models with composite IDs for proper Elo tracking.

This script registers all canonical models with composite participant IDs
that include harness information (algorithm and configuration).

Format: {nn_model_id}:{ai_algorithm}:{config}
Example: canonical_hex8_2p:gumbel_mcts:b800

Usage:
    # Dry run (default) - show what would be registered
    python scripts/register_composite_models.py

    # Actually register models
    python scripts/register_composite_models.py --execute

    # Register with specific harness
    python scripts/register_composite_models.py --harness gumbel_mcts --budget 800

    # Register single config
    python scripts/register_composite_models.py --config hex8_2p --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.elo_service import get_elo_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# All 12 canonical configurations
CONFIG_KEYS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]

# Model directory
MODELS_DIR = Path(__file__).parent.parent / "models"


def get_canonical_model_path(config_key: str) -> Path | None:
    """Find canonical model for a configuration.

    Args:
        config_key: Config like 'hex8_2p'

    Returns:
        Path to model file or None if not found
    """
    # Try different naming patterns
    patterns = [
        f"canonical_{config_key}.pth",
        f"ringrift_best_{config_key}.pth",
        f"canonical_{config_key}_v2.pth",
    ]

    for pattern in patterns:
        path = MODELS_DIR / pattern
        if path.exists():
            return path

    return None


def create_composite_id(
    nn_model_id: str,
    harness: str,
    config_str: str,
) -> str:
    """Create composite participant ID.

    Args:
        nn_model_id: Base model ID (e.g., 'canonical_hex8_2p')
        harness: Algorithm name (e.g., 'gumbel_mcts')
        config_str: Configuration string (e.g., 'b800')

    Returns:
        Composite ID like 'canonical_hex8_2p:gumbel_mcts:b800'
    """
    return f"{nn_model_id}:{harness}:{config_str}"


def register_model_with_harness(
    config_key: str,
    harness: str,
    budget: int,
    dry_run: bool = True,
) -> bool:
    """Register a model with its harness configuration.

    Args:
        config_key: Config like 'hex8_2p'
        harness: Algorithm like 'gumbel_mcts'
        budget: Simulation budget like 800
        dry_run: If True, don't actually register

    Returns:
        True if successful (or would be successful in dry run)
    """
    # Parse config
    parts = config_key.split("_")
    board_type = parts[0]
    num_players = int(parts[1].replace("p", ""))

    # Find model file
    model_path = get_canonical_model_path(config_key)
    if not model_path:
        logger.warning(f"No canonical model found for {config_key}")
        return False

    # Create IDs
    nn_model_id = f"canonical_{config_key}"
    config_str = f"b{budget}"
    composite_id = create_composite_id(nn_model_id, harness, config_str)

    if dry_run:
        logger.info(f"[DRY RUN] Would register: {composite_id}")
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  Board: {board_type}, Players: {num_players}")
        return True

    # Register with EloService
    elo = get_elo_service()

    try:
        # Register participant
        elo.register_participant(
            participant_id=composite_id,
            ai_type=harness,
            difficulty=8,  # Expert level
            use_neural_net=True,
            model_path=str(model_path),
            metadata={
                "nn_model_id": nn_model_id,
                "harness": harness,
                "budget": budget,
                "board_type": board_type,
                "num_players": num_players,
                "is_composite": True,
            },
            is_deployable=True,
        )

        # Initialize rating at 1500 for the config
        conn = elo._get_connection()
        conn.execute("""
            INSERT OR IGNORE INTO elo_ratings
            (participant_id, board_type, num_players, rating, games_played,
             wins, losses, draws, peak_rating, last_update)
            VALUES (?, ?, ?, 1500.0, 0, 0, 0, 0, 1500.0, ?)
        """, (composite_id, board_type, num_players, __import__('time').time()))
        conn.commit()

        # Update participant last_seen
        conn.execute("""
            UPDATE participants SET last_seen = ? WHERE participant_id = ?
        """, (__import__('time').time(), composite_id))
        conn.commit()

        logger.info(f"Registered: {composite_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to register {composite_id}: {e}")
        return False


def also_register_baselines(
    config_key: str,
    dry_run: bool = True,
) -> None:
    """Register random and heuristic baselines for a config.

    Args:
        config_key: Config like 'hex8_2p'
        dry_run: If True, don't actually register
    """
    parts = config_key.split("_")
    board_type = parts[0]
    num_players = int(parts[1].replace("p", ""))

    baselines = [
        ("random", "d1", 400.0),
        ("heuristic", "d5", 1000.0),
    ]

    elo = get_elo_service()

    for ai_type, config_str, default_rating in baselines:
        composite_id = f"none:{ai_type}:{config_str}"

        if dry_run:
            logger.info(f"[DRY RUN] Would register baseline: {composite_id}")
            continue

        try:
            elo.register_participant(
                participant_id=composite_id,
                ai_type=ai_type,
                difficulty=1 if ai_type == "random" else 5,
                use_neural_net=False,
                is_deployable=True,
            )

            # Initialize rating
            conn = elo._get_connection()
            conn.execute("""
                INSERT OR IGNORE INTO elo_ratings
                (participant_id, board_type, num_players, rating, games_played,
                 wins, losses, draws, peak_rating, last_update)
                VALUES (?, ?, ?, ?, 0, 0, 0, 0, ?, ?)
            """, (composite_id, board_type, num_players, default_rating, default_rating, __import__('time').time()))
            conn.commit()

            logger.info(f"Registered baseline: {composite_id} @ {default_rating} Elo")

        except Exception as e:
            logger.warning(f"Failed to register baseline {composite_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Register canonical models with composite IDs for Elo tracking"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually register models (default is dry run)"
    )
    parser.add_argument(
        "--harness",
        type=str,
        default="gumbel_mcts",
        help="AI harness algorithm (default: gumbel_mcts)"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=800,
        help="Simulation budget (default: 800)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Register single config (e.g., 'hex8_2p')"
    )
    parser.add_argument(
        "--with-baselines",
        action="store_true",
        help="Also register random/heuristic baselines"
    )

    args = parser.parse_args()
    dry_run = not args.execute

    if dry_run:
        print("\n=== DRY RUN MODE (use --execute to register) ===\n")

    configs = [args.config] if args.config else CONFIG_KEYS
    registered = 0
    skipped = 0

    for config_key in configs:
        success = register_model_with_harness(
            config_key=config_key,
            harness=args.harness,
            budget=args.budget,
            dry_run=dry_run,
        )

        if success:
            registered += 1
        else:
            skipped += 1

        if args.with_baselines:
            also_register_baselines(config_key, dry_run=dry_run)

    print(f"\n{'Would register' if dry_run else 'Registered'}: {registered} models")
    if skipped:
        print(f"Skipped: {skipped} (no model file found)")

    # Show current state
    if not dry_run:
        print("\nCurrent Elo entries with games > 0:")
        elo = get_elo_service()
        rows = elo.execute_query("""
            SELECT participant_id, board_type, num_players, rating, games_played
            FROM elo_ratings
            WHERE games_played > 0
            ORDER BY rating DESC
            LIMIT 15
        """)
        for row in rows:
            print(f"  {row[0]}: {row[3]:.0f} Elo ({row[4]} games) [{row[1]}_{row[2]}p]")


if __name__ == "__main__":
    main()
