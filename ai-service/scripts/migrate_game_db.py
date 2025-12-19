#!/usr/bin/env python3
"""CLI tool for managing GameReplayDB schema migrations.

Usage:
    # Check current schema status
    python scripts/migrate_game_db.py --db data/games/selfplay.db --status

    # Run migration (automatic, creates backup first)
    python scripts/migrate_game_db.py --db data/games/selfplay.db --migrate

    # Dry run (show what would be done)
    python scripts/migrate_game_db.py --db data/games/selfplay.db --migrate --dry-run

    # Force re-run migration (use with caution)
    python scripts/migrate_game_db.py --db data/games/selfplay.db --migrate --force
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.game_replay import SCHEMA_VERSION, GameReplayDB

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("migrate_game_db")


def get_db_schema_version(db_path: str) -> int:
    """Get schema version from database without triggering migration."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Check for schema_metadata table
        has_metadata = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_metadata'"
        ).fetchone()

        if has_metadata:
            row = conn.execute("SELECT value FROM schema_metadata WHERE key = 'schema_version'").fetchone()
            return int(row["value"]) if row else 1
        else:
            # Check if games table exists (v1 schema)
            has_games = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'").fetchone()
            return 1 if has_games else 0
    finally:
        conn.close()


def get_db_stats(db_path: str) -> dict:
    """Get basic stats from database without triggering migration."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        stats = {}

        # Check tables exist
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        stats["tables"] = [row["name"] for row in tables]

        if "games" in stats["tables"]:
            stats["game_count"] = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]

            stats["move_count"] = (
                conn.execute("SELECT COUNT(*) FROM game_moves").fetchone()[0] if "game_moves" in stats["tables"] else 0
            )

        # Check which columns exist in game_moves
        if "game_moves" in stats["tables"]:
            columns = conn.execute("PRAGMA table_info(game_moves)").fetchall()
            stats["game_moves_columns"] = [row["name"] for row in columns]

        # Check which columns exist in games
        if "games" in stats["tables"]:
            columns = conn.execute("PRAGMA table_info(games)").fetchall()
            stats["games_columns"] = [row["name"] for row in columns]

        return stats
    finally:
        conn.close()


def create_backup(db_path: str) -> str:
    """Create a timestamped backup of the database."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup.{timestamp}"
    shutil.copy2(db_path, backup_path)
    return backup_path


def show_status(db_path: str) -> None:
    """Show current database status."""
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return

    current_version = get_db_schema_version(db_path)
    stats = get_db_stats(db_path)

    print(f"\n{'=' * 60}")
    print(f"Database: {db_path}")
    print(f"{'=' * 60}")
    print(f"Current schema version: {current_version}")
    print(f"Target schema version:  {SCHEMA_VERSION}")
    print(f"Migration needed:       {'Yes' if current_version < SCHEMA_VERSION else 'No'}")
    print()

    print("Tables:")
    for table in stats.get("tables", []):
        print(f"  - {table}")
    print()

    if "game_count" in stats:
        print(f"Games:    {stats['game_count']:,}")
        print(f"Moves:    {stats['move_count']:,}")
        print()

    # Show v2 column status
    if "game_moves_columns" in stats:
        v2_move_columns = [
            "time_remaining_ms",
            "engine_eval",
            "engine_eval_type",
            "engine_depth",
            "engine_nodes",
            "engine_pv",
            "engine_time_ms",
        ]
        print("V2 columns in game_moves:")
        for col in v2_move_columns:
            status = "present" if col in stats["game_moves_columns"] else "MISSING"
            print(f"  - {col}: {status}")
        print()

    if "games_columns" in stats:
        v2_game_columns = ["time_control_type", "initial_time_ms", "time_increment_ms"]
        print("V2 columns in games:")
        for col in v2_game_columns:
            status = "present" if col in stats["games_columns"] else "MISSING"
            print(f"  - {col}: {status}")
        print()


def run_migration(db_path: str, dry_run: bool = False, force: bool = False) -> bool:
    """Run schema migration on database."""
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return False

    current_version = get_db_schema_version(db_path)

    if current_version >= SCHEMA_VERSION and not force:
        print(f"Database is already at schema version {current_version}")
        print("Use --force to re-run migration anyway")
        return True

    print(f"\nMigration plan: v{current_version} -> v{SCHEMA_VERSION}")

    if dry_run:
        print("\n[DRY RUN] Would perform the following:")
        print("  1. Create backup of database")
        print("  2. Add schema_metadata table (if needed)")
        print("  3. Add v2 columns to games table")
        print("  4. Add v2 columns to game_moves table")
        print("  5. Update schema_version to 2")
        print("\nNo changes made.")
        return True

    # Create backup first
    backup_path = create_backup(db_path)
    print(f"Created backup: {backup_path}")

    try:
        # Opening the database triggers migration
        print("Running migration...")
        db = GameReplayDB(db_path)
        stats = db.get_stats()

        print(f"\nMigration complete!")
        print(f"Schema version: {stats['schema_version']}")
        print(f"Games: {stats['total_games']:,}")
        print(f"Moves: {stats['total_moves']:,}")
        return True

    except Exception as e:
        print(f"\nMigration failed: {e}")
        print(f"Backup available at: {backup_path}")
        print("You can restore with: cp {backup_path} {db_path}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Manage GameReplayDB schema migrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current schema status",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Run schema migration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what migration would do without making changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run migration even if already at target version",
    )

    args = parser.parse_args()

    if not args.status and not args.migrate:
        parser.print_help()
        return 1

    if args.status:
        show_status(args.db)

    if args.migrate:
        success = run_migration(args.db, dry_run=args.dry_run, force=args.force)
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
