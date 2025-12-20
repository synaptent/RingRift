#!/usr/bin/env python3
"""Migrate data from legacy elo_leaderboard.db to unified Elo database.

This script migrates:
1. Models -> Participants
2. Elo ratings (with composite key transformation)
3. Match history
4. Rating history

Usage:
    python scripts/migrate_elo_to_unified.py [--dry-run] [--force]

Options:
    --dry-run   Show what would be migrated without making changes
    --force     Overwrite existing records in unified database
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.tournament.unified_elo_db import EloDatabase, UnifiedEloRating

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("migrate_elo_to_unified")

# Database paths
LEGACY_DB_PATH = AI_SERVICE_ROOT / "data" / "elo_leaderboard.db"
UNIFIED_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"


def get_legacy_connection() -> sqlite3.Connection:
    """Get connection to legacy database."""
    if not LEGACY_DB_PATH.exists():
        raise FileNotFoundError(f"Legacy database not found: {LEGACY_DB_PATH}")
    conn = sqlite3.connect(str(LEGACY_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def count_legacy_records() -> dict:
    """Count records in legacy database."""
    conn = get_legacy_connection()
    counts = {
        "models": conn.execute("SELECT COUNT(*) FROM models").fetchone()[0],
        "elo_ratings": conn.execute("SELECT COUNT(*) FROM elo_ratings").fetchone()[0],
        "match_history": conn.execute("SELECT COUNT(*) FROM match_history").fetchone()[0],
        "rating_history": conn.execute("SELECT COUNT(*) FROM rating_history").fetchone()[0],
    }
    conn.close()
    return counts


def count_unified_records(db: EloDatabase) -> dict:
    """Count records in unified database."""
    conn = db._get_connection()
    counts = {
        "participants": conn.execute("SELECT COUNT(*) FROM participants").fetchone()[0],
        "elo_ratings": conn.execute("SELECT COUNT(*) FROM elo_ratings").fetchone()[0],
        "match_history": conn.execute("SELECT COUNT(*) FROM match_history").fetchone()[0],
        "rating_history": conn.execute("SELECT COUNT(*) FROM rating_history").fetchone()[0],
    }
    return counts


def migrate_models_to_participants(
    legacy_conn: sqlite3.Connection,
    unified_db: EloDatabase,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Migrate models table to participants table."""
    logger.info("Migrating models -> participants...")

    rows = legacy_conn.execute("""
        SELECT model_id, model_path, board_type, num_players, model_version,
               created_at, last_seen
        FROM models
    """).fetchall()

    migrated = 0
    skipped = 0

    for row in rows:
        model_id = row["model_id"]

        # Check if already exists
        existing = unified_db.get_participant(model_id)
        if existing and not force:
            skipped += 1
            continue

        if dry_run:
            logger.info(f"  [DRY-RUN] Would migrate model: {model_id}")
            migrated += 1
            continue

        # Determine participant type and AI type from model_id
        if "baseline_random" in model_id:
            participant_type = "baseline"
            ai_type = "random"
        elif "baseline_heuristic" in model_id:
            participant_type = "baseline"
            ai_type = "heuristic"
        elif "baseline_mcts" in model_id:
            participant_type = "baseline"
            ai_type = "mcts"
        else:
            participant_type = "model"
            ai_type = "neural_net"

        unified_db.register_participant(
            participant_id=model_id,
            name=model_id,
            participant_type=participant_type,
            ai_type=ai_type,
            use_neural_net=(ai_type == "neural_net"),
            model_id=model_id,
            model_path=row["model_path"],
            model_version=row["model_version"],
        )
        migrated += 1

    logger.info(f"  Models migrated: {migrated}, skipped: {skipped}")
    return migrated


def migrate_elo_ratings(
    legacy_conn: sqlite3.Connection,
    unified_db: EloDatabase,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Migrate elo_ratings table.

    Legacy schema has single model_id PK, unified has composite
    (participant_id, board_type, num_players) PK.
    """
    logger.info("Migrating elo_ratings...")

    rows = legacy_conn.execute("""
        SELECT model_id, board_type, num_players, rating, games_played,
               wins, losses, draws, last_update
        FROM elo_ratings
    """).fetchall()

    migrated = 0
    skipped = 0

    for row in rows:
        model_id = row["model_id"]
        board_type = row["board_type"] or "square8"  # Default if NULL
        num_players = row["num_players"] or 2

        # Check if already exists
        existing = unified_db.get_rating(model_id, board_type, num_players)
        if existing.games_played > 0 and not force:
            skipped += 1
            continue

        if dry_run:
            logger.info(f"  [DRY-RUN] Would migrate rating: {model_id} "
                       f"({board_type}/{num_players}p) = {row['rating']:.1f}")
            migrated += 1
            continue

        # Create and update rating
        rating = UnifiedEloRating(
            participant_id=model_id,
            board_type=board_type,
            num_players=num_players,
            rating=row["rating"],
            games_played=row["games_played"],
            wins=row["wins"],
            losses=row["losses"],
            draws=row["draws"],
            last_update=row["last_update"],
        )
        unified_db.update_rating(rating)
        migrated += 1

    logger.info(f"  Ratings migrated: {migrated}, skipped: {skipped}")
    return migrated


def migrate_match_history(
    legacy_conn: sqlite3.Connection,
    unified_db: EloDatabase,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Migrate match_history table."""
    logger.info("Migrating match_history...")

    rows = legacy_conn.execute("""
        SELECT id, model_a, model_b, board_type, num_players, winner,
               game_length, duration_sec, timestamp, tournament_id
        FROM match_history
        ORDER BY id
    """).fetchall()

    # Track existing match IDs (by legacy ID mapping)
    unified_conn = unified_db._get_connection()

    migrated = 0
    skipped = 0

    for row in rows:
        legacy_id = row["id"]

        # Generate deterministic match_id from legacy ID
        match_id = f"legacy_{legacy_id}"

        # Check if already migrated
        existing = unified_conn.execute(
            "SELECT id FROM match_history WHERE id = ?",
            (match_id,)
        ).fetchone()

        if existing and not force:
            skipped += 1
            continue

        if dry_run:
            logger.info(f"  [DRY-RUN] Would migrate match #{legacy_id}: "
                       f"{row['model_a']} vs {row['model_b']}")
            migrated += 1
            continue

        # Convert timestamp to ISO format if numeric
        timestamp = row["timestamp"]
        if isinstance(timestamp, (int, float)):
            from datetime import datetime
            timestamp = datetime.fromtimestamp(timestamp).isoformat()

        # Determine rankings from winner
        model_a = row["model_a"]
        model_b = row["model_b"]
        winner = row["winner"]

        if winner == model_a:
            rankings = [0, 1]  # A won
        elif winner == model_b:
            rankings = [1, 0]  # B won
        else:
            rankings = [0, 0]  # Draw or no winner recorded

        board_type = row["board_type"] or "square8"
        num_players = row["num_players"] or 2

        # Ensure participants exist
        unified_db.ensure_participant(model_a, participant_type="model")
        unified_db.ensure_participant(model_b, participant_type="model")

        # Insert into unified database (using backwards-compatible schema)
        unified_conn.execute("""
            INSERT OR REPLACE INTO match_history
            (participant_a, participant_b, participant_ids, rankings, winner,
             board_type, num_players, game_length, duration_sec, timestamp,
             tournament_id, worker, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_a,
            model_b,
            json.dumps([model_a, model_b]),
            json.dumps(rankings),
            winner,
            board_type,
            num_players,
            row["game_length"],
            row["duration_sec"],
            timestamp,
            row["tournament_id"],
            None,  # worker
            json.dumps({"legacy_id": legacy_id}),  # metadata
        ))
        migrated += 1

    if not dry_run:
        unified_conn.commit()

    logger.info(f"  Matches migrated: {migrated}, skipped: {skipped}")
    return migrated


def migrate_rating_history(
    legacy_conn: sqlite3.Connection,
    unified_db: EloDatabase,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Migrate rating_history table."""
    logger.info("Migrating rating_history...")

    rows = legacy_conn.execute("""
        SELECT id, model_id, rating, games_played, timestamp, tournament_id
        FROM rating_history
        ORDER BY id
    """).fetchall()

    unified_conn = unified_db._get_connection()

    # We need to infer board_type and num_players from model_id or elo_ratings
    # First, build a mapping from model_id to (board_type, num_players)
    ratings_map = {}
    ratings_rows = legacy_conn.execute("""
        SELECT model_id, board_type, num_players FROM elo_ratings
    """).fetchall()
    for r in ratings_rows:
        ratings_map[r["model_id"]] = (
            r["board_type"] or "square8",
            r["num_players"] or 2,
        )

    migrated = 0
    skipped = 0

    for row in rows:
        legacy_id = row["id"]
        model_id = row["model_id"]

        # Get board_type and num_players
        board_type, num_players = ratings_map.get(model_id, ("square8", 2))

        # Check if similar entry exists (by participant, timestamp, rating)
        existing = unified_conn.execute("""
            SELECT id FROM rating_history
            WHERE participant_id = ? AND ABS(timestamp - ?) < 1
            AND ABS(rating - ?) < 0.1
        """, (model_id, row["timestamp"], row["rating"])).fetchone()

        if existing and not force:
            skipped += 1
            continue

        if dry_run:
            logger.info(f"  [DRY-RUN] Would migrate rating history #{legacy_id}: "
                       f"{model_id} = {row['rating']:.1f}")
            migrated += 1
            continue

        unified_conn.execute("""
            INSERT INTO rating_history
            (participant_id, board_type, num_players, rating, rating_change,
             games_played, timestamp, match_id, tournament_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            board_type,
            num_players,
            row["rating"],
            0.0,  # rating_change not available in legacy
            row["games_played"],
            row["timestamp"],
            None,  # match_id not available in legacy
            row["tournament_id"],
        ))
        migrated += 1

    if not dry_run:
        unified_conn.commit()

    logger.info(f"  Rating history migrated: {migrated}, skipped: {skipped}")
    return migrated


def run_migration(dry_run: bool = False, force: bool = False):
    """Run full migration from legacy to unified database."""
    logger.info("=" * 60)
    logger.info("Elo Database Migration: elo_leaderboard.db -> unified_elo.db")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    if force:
        logger.info("FORCE MODE - Existing records will be overwritten")

    # Check legacy database
    logger.info(f"\nLegacy database: {LEGACY_DB_PATH}")
    if not LEGACY_DB_PATH.exists():
        logger.error("Legacy database not found!")
        return False

    legacy_counts = count_legacy_records()
    logger.info(f"  Records to migrate:")
    logger.info(f"    Models: {legacy_counts['models']}")
    logger.info(f"    Elo ratings: {legacy_counts['elo_ratings']}")
    logger.info(f"    Match history: {legacy_counts['match_history']}")
    logger.info(f"    Rating history: {legacy_counts['rating_history']}")

    # Initialize unified database
    logger.info(f"\nUnified database: {UNIFIED_DB_PATH}")
    unified_db = EloDatabase(UNIFIED_DB_PATH)

    unified_counts_before = count_unified_records(unified_db)
    logger.info(f"  Current records:")
    logger.info(f"    Participants: {unified_counts_before['participants']}")
    logger.info(f"    Elo ratings: {unified_counts_before['elo_ratings']}")
    logger.info(f"    Match history: {unified_counts_before['match_history']}")
    logger.info(f"    Rating history: {unified_counts_before['rating_history']}")

    # Open legacy connection
    legacy_conn = get_legacy_connection()

    # Run migrations
    logger.info("\n" + "-" * 40)
    logger.info("Running migrations...")
    logger.info("-" * 40)

    total_migrated = 0
    total_migrated += migrate_models_to_participants(legacy_conn, unified_db, dry_run, force)
    total_migrated += migrate_elo_ratings(legacy_conn, unified_db, dry_run, force)
    total_migrated += migrate_match_history(legacy_conn, unified_db, dry_run, force)
    total_migrated += migrate_rating_history(legacy_conn, unified_db, dry_run, force)

    legacy_conn.close()

    # Report final counts
    if not dry_run:
        unified_counts_after = count_unified_records(unified_db)
        logger.info("\n" + "-" * 40)
        logger.info("Migration complete!")
        logger.info("-" * 40)
        logger.info(f"  Final unified database records:")
        logger.info(f"    Participants: {unified_counts_after['participants']} "
                   f"(+{unified_counts_after['participants'] - unified_counts_before['participants']})")
        logger.info(f"    Elo ratings: {unified_counts_after['elo_ratings']} "
                   f"(+{unified_counts_after['elo_ratings'] - unified_counts_before['elo_ratings']})")
        logger.info(f"    Match history: {unified_counts_after['match_history']} "
                   f"(+{unified_counts_after['match_history'] - unified_counts_before['match_history']})")
        logger.info(f"    Rating history: {unified_counts_after['rating_history']} "
                   f"(+{unified_counts_after['rating_history'] - unified_counts_before['rating_history']})")
    else:
        logger.info(f"\n[DRY-RUN] Would have migrated {total_migrated} total records")

    unified_db.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy Elo database to unified system")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be migrated without making changes")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing records in unified database")
    args = parser.parse_args()

    success = run_migration(dry_run=args.dry_run, force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
