#!/usr/bin/env python3
"""Archive stale Elo entries - participants with 0 games played.

This script identifies and archives Elo entries that:
1. Have games_played = 0 (registered but never evaluated)
2. Have last_seen = NULL in participants table

These entries bloat the database and obscure the actual leaderboard.

Usage:
    # Dry run (default) - show stale entries without archiving
    python scripts/archive_stale_elo_entries.py

    # Actually archive stale entries
    python scripts/archive_stale_elo_entries.py --execute

    # Archive entries older than 30 days (default: 0)
    python scripts/archive_stale_elo_entries.py --min-age-days 30

    # Only archive entries for a specific config
    python scripts/archive_stale_elo_entries.py --config square8_2p

    # Show statistics only
    python scripts/archive_stale_elo_entries.py --stats
"""

import argparse
import logging
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_ELO_DB = Path("data/unified_elo.db")

# Baseline participant IDs to preserve (built-in AI types)
PRESERVE_PARTICIPANTS = {
    "random_ai",
    "random",
    "heuristic",
    "heuristic_ai",
    "baseline_random",
    "baseline_heuristic",
}


def get_stale_elo_entries(
    conn: sqlite3.Connection,
    min_age_days: int = 0,
    config_key: str | None = None,
) -> list[tuple]:
    """Find Elo entries with 0 games played.

    Args:
        conn: SQLite connection
        min_age_days: Only archive entries older than this
        config_key: Filter to specific config (e.g., 'square8_2p')

    Returns:
        List of (participant_id, board_type, num_players, rating, created_at) tuples
    """
    query = """
        SELECT e.participant_id, e.board_type, e.num_players, e.rating,
               e.games_played, e.wins, e.losses, e.draws, e.last_update,
               p.created_at, p.last_seen
        FROM elo_ratings e
        LEFT JOIN participants p ON e.participant_id = p.participant_id
        WHERE e.games_played = 0
    """
    params = []

    if config_key:
        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))
        query += " AND e.board_type = ? AND e.num_players = ?"
        params.extend([board_type, num_players])

    if min_age_days > 0:
        cutoff = datetime.now().timestamp() - (min_age_days * 86400)
        query += " AND (p.created_at IS NULL OR p.created_at < ?)"
        params.append(cutoff)

    query += " ORDER BY e.board_type, e.num_players, e.participant_id"

    cursor = conn.execute(query, params)
    return cursor.fetchall()


def get_stale_participants(conn: sqlite3.Connection) -> list[tuple]:
    """Find participants with last_seen = NULL that aren't baselines.

    Returns:
        List of (participant_id, participant_type, created_at) tuples
    """
    query = """
        SELECT p.participant_id, p.participant_type, p.created_at
        FROM participants p
        WHERE p.last_seen IS NULL
        ORDER BY p.participant_id
    """
    cursor = conn.execute(query)

    results = []
    for row in cursor.fetchall():
        pid = row[0]
        # Skip baseline/built-in AIs
        if pid in PRESERVE_PARTICIPANTS:
            continue
        if any(pid.lower().startswith(prefix) for prefix in ["d1", "d2", "d3", "d4", "d5", "tier"]):
            continue
        results.append(row)

    return results


def create_archive_table(conn: sqlite3.Connection) -> None:
    """Create archive table for stale Elo entries if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS archived_elo_ratings (
            participant_id TEXT NOT NULL,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            rating REAL DEFAULT 1500.0,
            games_played INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            draws INTEGER DEFAULT 0,
            last_update REAL,
            archived_at REAL NOT NULL,
            archive_reason TEXT,
            PRIMARY KEY (participant_id, board_type, num_players)
        )
    """)
    conn.commit()


def archive_elo_entries(
    conn: sqlite3.Connection,
    entries: list[tuple],
    dry_run: bool = True,
) -> int:
    """Archive stale Elo entries.

    Args:
        conn: SQLite connection
        entries: List of stale entries from get_stale_elo_entries()
        dry_run: If True, don't actually modify database

    Returns:
        Number of entries archived
    """
    if not entries:
        return 0

    create_archive_table(conn)
    archived_count = 0
    now = datetime.now().timestamp()

    for entry in entries:
        pid, board_type, num_players = entry[0], entry[1], entry[2]

        # Skip baselines
        if pid in PRESERVE_PARTICIPANTS:
            continue

        if dry_run:
            logger.info(f"  [DRY RUN] Would archive: {pid} ({board_type}_{num_players}p)")
            archived_count += 1
            continue

        # Move to archive table
        try:
            conn.execute("""
                INSERT OR REPLACE INTO archived_elo_ratings
                (participant_id, board_type, num_players, rating, games_played,
                 wins, losses, draws, last_update, archived_at, archive_reason)
                SELECT participant_id, board_type, num_players, rating, games_played,
                       wins, losses, draws, last_update, ?, 'stale_zero_games'
                FROM elo_ratings
                WHERE participant_id = ? AND board_type = ? AND num_players = ?
            """, (now, pid, board_type, num_players))

            # Delete from main table
            conn.execute("""
                DELETE FROM elo_ratings
                WHERE participant_id = ? AND board_type = ? AND num_players = ?
            """, (pid, board_type, num_players))

            archived_count += 1
            logger.info(f"  Archived: {pid} ({board_type}_{num_players}p)")

        except sqlite3.Error as e:
            logger.warning(f"  Failed to archive {pid}: {e}")

    if not dry_run:
        conn.commit()

    return archived_count


def delete_orphan_participants(
    conn: sqlite3.Connection,
    participants: list[tuple],
    dry_run: bool = True,
) -> int:
    """Delete participant entries with no Elo ratings or activity.

    Args:
        conn: SQLite connection
        participants: List of stale participants from get_stale_participants()
        dry_run: If True, don't actually modify database

    Returns:
        Number of participants deleted
    """
    if not participants:
        return 0

    deleted_count = 0

    for entry in participants:
        pid = entry[0]

        # Check if participant has any remaining Elo entries
        cursor = conn.execute(
            "SELECT COUNT(*) FROM elo_ratings WHERE participant_id = ?",
            (pid,)
        )
        elo_count = cursor.fetchone()[0]

        if elo_count > 0:
            logger.debug(f"  Skipping {pid}: has {elo_count} Elo entries")
            continue

        if dry_run:
            logger.info(f"  [DRY RUN] Would delete participant: {pid}")
            deleted_count += 1
            continue

        try:
            conn.execute("DELETE FROM participants WHERE participant_id = ?", (pid,))
            deleted_count += 1
            logger.info(f"  Deleted participant: {pid}")
        except sqlite3.Error as e:
            logger.warning(f"  Failed to delete {pid}: {e}")

    if not dry_run:
        conn.commit()

    return deleted_count


def show_statistics(conn: sqlite3.Connection) -> None:
    """Show current database statistics."""
    # Total participants
    cursor = conn.execute("SELECT COUNT(*) FROM participants")
    total_participants = cursor.fetchone()[0]

    # Participants with last_seen
    cursor = conn.execute("SELECT COUNT(*) FROM participants WHERE last_seen IS NOT NULL")
    active_participants = cursor.fetchone()[0]

    # Total Elo entries
    cursor = conn.execute("SELECT COUNT(*) FROM elo_ratings")
    total_elo = cursor.fetchone()[0]

    # Elo entries with games > 0
    cursor = conn.execute("SELECT COUNT(*) FROM elo_ratings WHERE games_played > 0")
    active_elo = cursor.fetchone()[0]

    # Archived entries (if table exists)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM archived_elo_ratings")
        archived_count = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        archived_count = 0

    # Top 10 by games played
    cursor = conn.execute("""
        SELECT participant_id, board_type, num_players, rating, games_played
        FROM elo_ratings
        WHERE games_played > 0
        ORDER BY games_played DESC
        LIMIT 10
    """)
    top_entries = cursor.fetchall()

    print("\n=== Elo Database Statistics ===")
    print(f"\nParticipants:")
    print(f"  Total: {total_participants}")
    print(f"  Active (with last_seen): {active_participants}")
    print(f"  Stale (no last_seen): {total_participants - active_participants}")

    print(f"\nElo Ratings:")
    print(f"  Total entries: {total_elo}")
    print(f"  With games > 0: {active_elo}")
    print(f"  Stale (games = 0): {total_elo - active_elo}")
    print(f"  Archived: {archived_count}")

    if top_entries:
        print(f"\nTop 10 by games played:")
        for pid, bt, np, rating, games in top_entries:
            print(f"  {pid}: {rating:.0f} Elo, {games} games ({bt}_{np}p)")


def main():
    parser = argparse.ArgumentParser(
        description="Archive stale Elo entries with 0 games played"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually archive entries (default is dry run)"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_ELO_DB,
        help=f"Path to Elo database (default: {DEFAULT_ELO_DB})"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Filter to specific config (e.g., 'square8_2p')"
    )
    parser.add_argument(
        "--min-age-days",
        type=int,
        default=0,
        help="Only archive entries older than N days"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics only, no archiving"
    )
    parser.add_argument(
        "--delete-orphan-participants",
        action="store_true",
        help="Also delete participants with no Elo entries"
    )

    args = parser.parse_args()

    # Ensure database exists
    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        sys.exit(1)

    # Create backup if executing
    if args.execute:
        backup_path = args.db.with_suffix(f".db.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(args.db, backup_path)
        logger.info(f"Created backup: {backup_path}")

    conn = sqlite3.connect(args.db)

    try:
        if args.stats:
            show_statistics(conn)
            return

        dry_run = not args.execute

        if dry_run:
            print("\n=== DRY RUN MODE (use --execute to apply changes) ===\n")

        # Find stale Elo entries
        stale_entries = get_stale_elo_entries(
            conn,
            min_age_days=args.min_age_days,
            config_key=args.config,
        )

        print(f"Found {len(stale_entries)} stale Elo entries (games_played = 0)")

        if stale_entries:
            archived = archive_elo_entries(conn, stale_entries, dry_run=dry_run)
            print(f"{'Would archive' if dry_run else 'Archived'} {archived} entries")

        # Optionally delete orphan participants
        if args.delete_orphan_participants:
            stale_participants = get_stale_participants(conn)
            print(f"\nFound {len(stale_participants)} stale participants (last_seen = NULL)")

            if stale_participants:
                deleted = delete_orphan_participants(conn, stale_participants, dry_run=dry_run)
                print(f"{'Would delete' if dry_run else 'Deleted'} {deleted} orphan participants")

        # Show final stats
        print()
        show_statistics(conn)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
