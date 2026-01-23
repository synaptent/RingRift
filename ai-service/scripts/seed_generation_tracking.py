#!/usr/bin/env python3
"""Seed the generation tracking database with existing canonical models.

This script populates the generation_tracking.db with the current canonical
models as generation 1 (baseline). Future training runs will build on this.

Usage:
    python scripts/seed_generation_tracking.py

    # Dry run (preview only)
    python scripts/seed_generation_tracking.py --dry-run

    # Force re-seed (clear and rebuild)
    python scripts/seed_generation_tracking.py --force

January 2026 - Created for MVP research demonstration.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.coordination.generation_tracker import (
    GenerationTracker,
    get_generation_tracker,
    reset_generation_tracker,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Canonical model configurations
CONFIGS = [
    ("hex8", 2),
    ("hex8", 3),
    ("hex8", 4),
    ("square8", 2),
    ("square8", 3),
    ("square8", 4),
    ("square19", 2),
    ("square19", 3),
    ("square19", 4),
    ("hexagonal", 2),
    ("hexagonal", 3),
    ("hexagonal", 4),
]


def find_canonical_model(board_type: str, num_players: int, models_dir: Path) -> Path | None:
    """Find the canonical model for a configuration."""
    # Check for main canonical model
    canonical = models_dir / f"canonical_{board_type}_{num_players}p.pth"
    if canonical.exists():
        return canonical

    # Check for v2 variant
    canonical_v2 = models_dir / f"canonical_{board_type}_{num_players}p_v2.pth"
    if canonical_v2.exists():
        return canonical_v2

    return None


def seed_generations(
    tracker: GenerationTracker,
    models_dir: Path,
    dry_run: bool = False,
) -> int:
    """Seed the generation database with existing canonical models.

    Args:
        tracker: GenerationTracker instance
        models_dir: Path to models directory
        dry_run: If True, only preview actions

    Returns:
        Number of generations seeded
    """
    seeded = 0

    for board_type, num_players in CONFIGS:
        config_key = f"{board_type}_{num_players}p"

        # Check if already seeded
        existing = tracker.get_latest_generation(board_type, num_players)
        if existing:
            logger.info(f"  {config_key}: Already has generation {existing.generation_id} (v{existing.version})")
            continue

        # Find canonical model
        model_path = find_canonical_model(board_type, num_players, models_dir)
        if not model_path:
            logger.warning(f"  {config_key}: No canonical model found")
            continue

        # Get model size as proxy for training effort
        size_mb = model_path.stat().st_size / (1024 * 1024)

        if dry_run:
            logger.info(f"  {config_key}: Would seed from {model_path.name} ({size_mb:.1f}MB)")
        else:
            gen_id = tracker.record_generation(
                model_path=str(model_path),
                board_type=board_type,
                num_players=num_players,
                parent_generation_id=None,  # No parent for v1
                training_games=None,  # Unknown for existing models
                training_samples=None,
            )
            logger.info(f"  {config_key}: Seeded as generation {gen_id} from {model_path.name}")

        seeded += 1

    return seeded


def main():
    parser = argparse.ArgumentParser(
        description="Seed generation tracking database with existing canonical models"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without making changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear existing data and re-seed",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Path to models directory (default: models)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RingRift Generation Tracking - Seed Script")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    # Get or create tracker
    if args.force and not args.dry_run:
        logger.info("Force mode: Clearing existing generation data...")
        db_path = Path("data/generation_tracking.db")
        if db_path.exists():
            db_path.unlink()
        reset_generation_tracker()

    tracker = get_generation_tracker()
    stats = tracker.get_stats()

    logger.info(f"Current database state:")
    logger.info(f"  Generations: {stats['total_generations']}")
    logger.info(f"  Tournaments: {stats['total_tournaments']}")
    logger.info(f"  Elo snapshots: {stats['total_elo_snapshots']}")

    if stats["total_generations"] > 0 and not args.force:
        logger.info("Database already has generations. Use --force to re-seed.")

    logger.info("")
    logger.info("Seeding canonical models...")

    seeded = seed_generations(tracker, args.models_dir, dry_run=args.dry_run)

    logger.info("")
    logger.info(f"Seeded {seeded} new generation(s)")

    # Show final state
    if not args.dry_run:
        stats = tracker.get_stats()
        logger.info("")
        logger.info("Final database state:")
        logger.info(f"  Generations: {stats['total_generations']}")
        for config_key, count in sorted(stats["generations_by_config"].items()):
            logger.info(f"    {config_key}: {count} generation(s)")

        # Show improvement report
        logger.info("")
        logger.info(tracker.get_improvement_report())


if __name__ == "__main__":
    main()
