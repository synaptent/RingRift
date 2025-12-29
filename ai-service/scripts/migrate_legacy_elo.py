#!/usr/bin/env python3
"""Migrate legacy Elo databases to unified EloService.

This script consolidates Elo data from various legacy databases into the
canonical unified_elo.db used by EloService.

Legacy databases to migrate:
- data/elo/comprehensive_elo.db
- data/elo/comprehensive_elo_3p.db
- data/elo/comprehensive_elo_4p.db
- data/elo/tournament_20251214.db
- data/elo_leaderboard.db (if exists)

Usage:
    python scripts/migrate_legacy_elo.py --dry-run    # Preview changes
    python scripts/migrate_legacy_elo.py --execute    # Actually migrate

December 2025: Created as part of Elo unification initiative.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.elo_service import get_elo_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Legacy database locations
LEGACY_DATABASES = [
    ("data/elo/comprehensive_elo.db", "comprehensive_elo"),
    ("data/elo/comprehensive_elo_3p.db", "comprehensive_elo_3p"),
    ("data/elo/comprehensive_elo_4p.db", "comprehensive_elo_4p"),
    ("data/elo/tournament_20251214.db", "tournament_20251214"),
    ("data/elo_leaderboard.db", "elo_leaderboard"),
    ("data/gauntlet_results.db", "gauntlet_results"),
]

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def get_db_connection(db_path: Path) -> sqlite3.Connection | None:
    """Get database connection if file exists."""
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.warning(f"Failed to connect to {db_path}: {e}")
        return None


def get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Get column names for a table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]


def fetch_legacy_matches(
    conn: sqlite3.Connection,
    source_name: str,
) -> list[dict[str, Any]]:
    """Fetch match records from a legacy database."""
    matches = []

    # Check which table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('match_history', 'matches', 'gauntlet_matches')"
    )
    tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        logger.debug(f"No match tables found in {source_name}")
        return []

    table = tables[0]
    columns = get_table_columns(conn, table)

    try:
        cursor = conn.execute(f"SELECT * FROM {table}")
        for row in cursor:
            match = dict(row)
            match["_source"] = source_name
            match["_table"] = table
            matches.append(match)
    except sqlite3.Error as e:
        logger.warning(f"Error fetching matches from {source_name}: {e}")

    return matches


def normalize_match(match: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a legacy match record to EloService format.

    Returns None if match cannot be normalized (missing required fields).
    """
    # Extract participant IDs
    participant_a = match.get("participant_a") or match.get("model_a")
    participant_b = match.get("participant_b") or match.get("model_b")

    # Try participant_ids JSON array
    if not participant_a or not participant_b:
        participant_ids = match.get("participant_ids")
        if participant_ids:
            try:
                ids = json.loads(participant_ids) if isinstance(participant_ids, str) else participant_ids
                if len(ids) >= 2:
                    participant_a = ids[0]
                    participant_b = ids[1]
            except (json.JSONDecodeError, TypeError):
                pass

    if not participant_a or not participant_b:
        return None

    # Determine winner
    winner = match.get("winner") or match.get("winner_id")

    # Try rankings to determine winner
    if not winner:
        rankings = match.get("rankings")
        if rankings:
            try:
                ranks = json.loads(rankings) if isinstance(rankings, str) else rankings
                if len(ranks) >= 2:
                    if ranks[0] < ranks[1]:
                        winner = participant_a
                    elif ranks[1] < ranks[0]:
                        winner = participant_b
                    # else draw (winner = None)
            except (json.JSONDecodeError, TypeError):
                pass

    # Get board type and num players
    board_type = match.get("board_type")
    num_players = match.get("num_players")

    if not board_type or not num_players:
        return None

    # Build game_id for deduplication
    game_id = match.get("game_id")
    if not game_id:
        # Generate from source + match ID
        match_id = match.get("id") or match.get("match_id")
        source = match.get("_source", "unknown")
        game_id = f"{source}_{match_id}"

    return {
        "participant_a": participant_a,
        "participant_b": participant_b,
        "winner": winner,
        "board_type": board_type,
        "num_players": int(num_players),
        "game_length": match.get("game_length", 0) or 0,
        "duration_sec": match.get("duration_sec", 0.0) or 0.0,
        "tournament_id": match.get("tournament_id") or match.get("_source"),
        "game_id": game_id,
        "metadata": {
            "migrated_from": match.get("_source"),
            "original_timestamp": match.get("timestamp"),
        },
    }


def migrate_matches(
    dry_run: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Migrate all legacy matches to EloService.

    Args:
        dry_run: If True, only preview changes without writing
        verbose: If True, print each match being migrated

    Returns:
        Migration statistics
    """
    stats = {
        "sources_checked": 0,
        "sources_with_data": 0,
        "matches_found": 0,
        "matches_normalized": 0,
        "matches_migrated": 0,
        "matches_skipped_duplicate": 0,
        "matches_skipped_error": 0,
        "errors": [],
    }

    # Get EloService
    if not dry_run:
        elo_service = get_elo_service()
    else:
        elo_service = None

    # Process each legacy database
    for db_rel_path, source_name in LEGACY_DATABASES:
        db_path = PROJECT_ROOT / db_rel_path
        stats["sources_checked"] += 1

        conn = get_db_connection(db_path)
        if not conn:
            logger.debug(f"Skipping {db_rel_path} (not found or can't connect)")
            continue

        try:
            matches = fetch_legacy_matches(conn, source_name)
            if matches:
                stats["sources_with_data"] += 1
                stats["matches_found"] += len(matches)
                logger.info(f"Found {len(matches)} matches in {source_name}")

            for match in matches:
                normalized = normalize_match(match)
                if not normalized:
                    stats["matches_skipped_error"] += 1
                    continue

                stats["matches_normalized"] += 1

                if verbose:
                    logger.info(
                        f"  {normalized['participant_a']} vs {normalized['participant_b']} "
                        f"({normalized['board_type']}_{normalized['num_players']}p)"
                    )

                if not dry_run:
                    try:
                        # Check for duplicate via game_id
                        result = elo_service.record_match(
                            participant_a=normalized["participant_a"],
                            participant_b=normalized["participant_b"],
                            winner=normalized["winner"],
                            board_type=normalized["board_type"],
                            num_players=normalized["num_players"],
                            game_length=normalized["game_length"],
                            duration_sec=normalized["duration_sec"],
                            tournament_id=normalized["tournament_id"],
                            metadata=normalized["metadata"],
                        )
                        stats["matches_migrated"] += 1
                    except Exception as e:
                        if "UNIQUE constraint failed" in str(e) or "duplicate" in str(e).lower():
                            stats["matches_skipped_duplicate"] += 1
                        else:
                            stats["matches_skipped_error"] += 1
                            stats["errors"].append(f"{source_name}: {e}")
                            logger.warning(f"Failed to migrate match: {e}")
        finally:
            conn.close()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate legacy Elo databases to unified EloService"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing (default)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show each match being processed",
    )

    args = parser.parse_args()

    # Default to dry-run unless --execute specified
    dry_run = not args.execute

    if dry_run:
        logger.info("=== DRY RUN MODE - No changes will be made ===")
    else:
        logger.info("=== EXECUTING MIGRATION ===")

    stats = migrate_matches(dry_run=dry_run, verbose=args.verbose)

    # Print summary
    print("\n" + "=" * 50)
    print("Migration Summary")
    print("=" * 50)
    print(f"Sources checked:           {stats['sources_checked']}")
    print(f"Sources with data:         {stats['sources_with_data']}")
    print(f"Matches found:             {stats['matches_found']}")
    print(f"Matches normalized:        {stats['matches_normalized']}")

    if dry_run:
        print(f"\n[DRY RUN] Would migrate:   {stats['matches_normalized']} matches")
        print("\nRun with --execute to perform the migration.")
    else:
        print(f"Matches migrated:          {stats['matches_migrated']}")
        print(f"Skipped (duplicate):       {stats['matches_skipped_duplicate']}")
        print(f"Skipped (error):           {stats['matches_skipped_error']}")

    if stats["errors"]:
        print("\nErrors:")
        for err in stats["errors"][:10]:  # Limit to 10
            print(f"  - {err}")
        if len(stats["errors"]) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
