#!/usr/bin/env python3
"""Consolidate game databases on a cluster node.

This script merges jsonl_aggregated.db into selfplay.db with parity validation.
It's designed to run on cluster nodes to consolidate all available training data.

RR-CANONICAL-MERGE-2025-12-22

Usage:
    # Run on a cluster node to consolidate games
    python scripts/consolidate_cluster_games.py

    # With validation and upgrade
    python scripts/consolidate_cluster_games.py --validate --upgrade

    # Dry run
    python scripts/consolidate_cluster_games.py --dry-run

    # Focus on specific board types
    python scripts/consolidate_cluster_games.py --board-type hexagonal --board-type square19
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default data directories
DATA_DIR = PROJECT_ROOT / "data"
GAMES_DIR = DATA_DIR / "games"

# Databases to merge into selfplay.db
MERGE_SOURCES = [
    "jsonl_aggregated.db",
    "jsonl_converted_hexagonal_2p.db",
    "jsonl_converted_hexagonal_3p.db",
    "jsonl_converted_hexagonal_4p.db",
    "jsonl_converted_square19_2p.db",
    "jsonl_converted_square19_3p.db",
    "jsonl_converted_square19_4p.db",
    "jsonl_converted_square8_2p.db",
    "jsonl_converted_square8_3p.db",
    "jsonl_converted_square8_4p.db",
    "tournament_hexagonal_2p.db",
    "tournament_hexagonal_3p.db",
    "tournament_hexagonal_4p.db",
    "tournament_square19_2p.db",
    "tournament_square19_3p.db",
    "tournament_square19_4p.db",
]

TARGET_DB = "selfplay.db"


def find_source_databases(games_dir: Path) -> list[Path]:
    """Find source databases to merge."""
    sources = []

    for db_name in MERGE_SOURCES:
        db_path = games_dir / db_name
        if db_path.exists():
            sources.append(db_path)
            logger.info(f"Found source: {db_path}")

    # Also look for any other potential sources
    for db_file in games_dir.glob("*.db"):
        name = db_file.name
        if name == TARGET_DB:
            continue
        # Skip glob patterns or special files
        if "*" in name or name.startswith("."):
            continue
        if db_file in sources:
            continue
        if name.startswith("canonical_"):
            sources.append(db_file)
            logger.info(f"Found canonical source: {db_file}")
        elif "tournament" in name:
            sources.append(db_file)
            logger.info(f"Found tournament source: {db_file}")

    return sources


def main():
    parser = argparse.ArgumentParser(description="Consolidate cluster game databases")
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=GAMES_DIR,
        help="Games directory",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate games against canonical engine",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Attempt to upgrade failed games to canonical format",
    )
    parser.add_argument(
        "--board-type",
        action="append",
        help="Filter by board type (can specify multiple)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit games per source",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't modify target database",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    games_dir = Path(args.games_dir)
    target_db = games_dir / TARGET_DB

    logger.info("=" * 60)
    logger.info("CONSOLIDATE CLUSTER GAMES")
    logger.info("=" * 60)
    logger.info(f"Games directory: {games_dir}")
    logger.info(f"Target database: {target_db}")

    # Find sources
    sources = find_source_databases(games_dir)
    if not sources:
        logger.error("No source databases found!")
        sys.exit(1)

    logger.info(f"Found {len(sources)} source databases")

    # Import merge function directly to avoid heavy dependencies
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "merge_and_validate_games",
        SCRIPT_DIR / "merge_and_validate_games.py"
    )
    merge_module = importlib.util.module_from_spec(spec)
    # Register the module before exec to allow dataclasses introspection
    sys.modules["merge_and_validate_games"] = merge_module
    spec.loader.exec_module(merge_module)
    merge_databases = merge_module.merge_databases

    stats = merge_databases(
        sources=[str(s) for s in sources],
        target=str(target_db),
        validate=args.validate,
        upgrade=args.upgrade,
        board_types=args.board_type,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    logger.info("=" * 60)
    logger.info("CONSOLIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Source games: {stats.source_games}")
    logger.info(f"Already exists: {stats.already_exists}")
    logger.info(f"Passed validation: {stats.passed_validation}")
    logger.info(f"Failed validation: {stats.failed_validation}")
    logger.info(f"Errors: {stats.errors}")
    logger.info(f"Upgraded: {stats.upgraded}")
    logger.info(f"Merged: {stats.merged}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
