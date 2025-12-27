#!/usr/bin/env python3
"""Consolidate jsonl_aggregated.db files into per-config canonical databases.

This script merges the scattered jsonl_aggregated.db files (44 files, 147GB total)
into per-config canonical databases for faster training data export.

Features:
- Deduplicates by game_id
- Validates game completeness before merging
- Updates existing canonical databases incrementally
- Tracks merge statistics

Usage:
    # Dry run - show what would be merged
    python scripts/consolidate_jsonl_databases.py --dry-run

    # Merge all configs
    python scripts/consolidate_jsonl_databases.py

    # Merge specific config
    python scripts/consolidate_jsonl_databases.py --config hex8_4p

    # Custom paths
    python scripts/consolidate_jsonl_databases.py \
        --source-dir /path/to/jsonl_dbs \
        --dest-dir /path/to/canonical

December 2025: Created for database consolidation.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Configuration
OWC_SELFPLAY_DIR = Path(os.getenv("OWC_SELFPLAY_DIR", "/Volumes/RingRift-Data/selfplay_repository"))
OWC_CANONICAL_DIR = Path(os.getenv("OWC_CANONICAL_DIR", "/Volumes/RingRift-Data/canonical_games"))
MIN_MOVES_FOR_VALID = 5

# All supported configs
ALL_CONFIGS = [
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("consolidate")


@dataclass
class MergeStats:
    """Statistics for merge operation."""
    source_db: str = ""
    config: str = ""
    games_scanned: int = 0
    games_valid: int = 0
    games_merged: int = 0
    games_duplicate: int = 0
    games_invalid: int = 0
    errors: list = field(default_factory=list)


@dataclass
class ConsolidationResult:
    """Result of consolidation for a config."""
    config: str = ""
    source_dbs: int = 0
    total_games_before: int = 0
    total_games_after: int = 0
    games_added: int = 0
    merge_stats: list = field(default_factory=list)


def find_jsonl_databases(base_dir: Path) -> list[Path]:
    """Find all jsonl_aggregated.db files."""
    databases = []
    for pattern in ["**/jsonl_aggregated.db", "**/jsonl_*.db"]:
        databases.extend(base_dir.glob(pattern))
    return list(set(databases))


def get_existing_game_ids(db_path: Path) -> set[str]:
    """Get set of existing game IDs in database."""
    if not db_path.exists():
        return set()

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT game_id FROM games")
        game_ids = {row[0] for row in cursor.fetchall()}
        conn.close()
        return game_ids
    except sqlite3.Error as e:
        logger.warning(f"Could not read game IDs from {db_path}: {e}")
        return set()


def count_games_for_config(db_path: Path, board_type: str, num_players: int) -> int:
    """Count valid games for a specific config in a database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("""
            SELECT COUNT(*) FROM games
            WHERE board_type = ? AND num_players = ?
            AND game_status IN ('completed', 'finished')
            AND total_moves >= ?
        """, (board_type, num_players, MIN_MOVES_FOR_VALID))
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except sqlite3.Error:
        return 0


def ensure_canonical_schema(db_path: Path, reference_db: Path = None) -> None:
    """Ensure the canonical database has the correct schema.

    Uses the normalized schema compatible with jsonl_aggregated databases.
    """
    conn = sqlite3.connect(str(db_path))

    # Use normalized schema matching jsonl_aggregated format
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            rng_seed INTEGER,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            game_status TEXT NOT NULL,
            winner INTEGER,
            termination_reason TEXT,
            total_moves INTEGER NOT NULL,
            total_turns INTEGER NOT NULL,
            duration_ms INTEGER,
            source TEXT,
            schema_version INTEGER NOT NULL DEFAULT 5,
            time_control_type TEXT DEFAULT 'none',
            initial_time_ms INTEGER,
            time_increment_ms INTEGER,
            metadata_json TEXT,
            quality_score REAL,
            quality_category TEXT
        )
    """)

    # Use the full v11 schema to preserve move_json (required for training export)
    # The minimal schema was causing data loss during consolidation
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_moves (
            game_id TEXT NOT NULL,
            move_number INTEGER NOT NULL,
            turn_number INTEGER NOT NULL DEFAULT 0,
            player INTEGER NOT NULL,
            phase TEXT NOT NULL DEFAULT 'unknown',
            move_type TEXT NOT NULL DEFAULT 'unknown',
            move_json TEXT NOT NULL DEFAULT '{}',
            timestamp TEXT,
            think_time_ms INTEGER,
            time_remaining_ms INTEGER,
            engine_eval REAL,
            engine_eval_type TEXT,
            engine_depth INTEGER,
            engine_nodes INTEGER,
            engine_pv TEXT,
            engine_time_ms INTEGER,
            move_probs TEXT,
            search_stats_json TEXT,
            PRIMARY KEY (game_id, move_number),
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_initial_state (
            game_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_state_snapshots (
            game_id TEXT NOT NULL,
            move_number INTEGER NOT NULL,
            state_json TEXT NOT NULL,
            PRIMARY KEY (game_id, move_number),
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_players (
            game_id TEXT NOT NULL,
            player_index INTEGER NOT NULL,
            player_type TEXT,
            model_version TEXT,
            PRIMARY KEY (game_id, player_index),
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_games_config
        ON games(board_type, num_players)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_games_status
        ON games(game_status)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_games_created
        ON games(created_at)
    """)

    conn.commit()
    conn.close()


def merge_games(
    source_db: Path,
    dest_db: Path,
    board_type: str,
    num_players: int,
    existing_ids: set[str],
    dry_run: bool = False,
) -> MergeStats:
    """Merge valid games from source to destination database.

    This handles the normalized schema with separate tables:
    - games: main game metadata
    - game_moves: individual moves
    - game_initial_state: initial board state
    - game_state_snapshots: state snapshots
    - game_players: player info
    """
    stats = MergeStats(
        source_db=str(source_db),
        config=f"{board_type}_{num_players}p",
    )

    # Related tables to copy along with games
    RELATED_TABLES = [
        "game_moves",
        "game_initial_state",
        "game_state_snapshots",
        "game_players",
        "game_choices",
        "game_history_entries",
    ]

    try:
        src_conn = sqlite3.connect(str(source_db))
        src_conn.row_factory = sqlite3.Row

        # Check if game_moves table exists
        cursor = src_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'"
        )
        has_game_moves_table = cursor.fetchone() is not None

        # Get valid games for this config
        cursor = src_conn.execute("""
            SELECT * FROM games
            WHERE board_type = ? AND num_players = ?
            AND game_status IN ('completed', 'finished')
            AND total_moves >= ?
        """, (board_type, num_players, MIN_MOVES_FOR_VALID))

        games_to_merge = []
        game_ids_to_merge = []

        for row in cursor:
            stats.games_scanned += 1
            game_id = row["game_id"]

            if game_id in existing_ids:
                stats.games_duplicate += 1
                continue

            # CRITICAL: Verify game_moves actually exist for this game
            # This prevents merging orphan games that have total_moves>0 but no actual move data
            if has_game_moves_table:
                move_cursor = src_conn.execute(
                    "SELECT COUNT(*) FROM game_moves WHERE game_id = ?",
                    (game_id,)
                )
                actual_move_count = move_cursor.fetchone()[0]

                if actual_move_count == 0:
                    # Game claims to have moves but game_moves table is empty for it
                    stats.games_invalid += 1
                    logger.debug(
                        f"  Skipping orphan game {game_id}: total_moves={row['total_moves']} "
                        f"but game_moves has 0 entries"
                    )
                    continue
            else:
                # No game_moves table at all - can't merge this game
                stats.games_invalid += 1
                continue

            # Convert row to dict for easier access
            row_dict = dict(row)
            stats.games_valid += 1
            games_to_merge.append(row_dict)
            game_ids_to_merge.append(game_id)
            existing_ids.add(game_id)  # Track for dedup within this merge

        if dry_run:
            stats.games_merged = len(games_to_merge)
            logger.debug(f"  [DRY-RUN] Would merge {len(games_to_merge)} games from {source_db.parent.name}")
            src_conn.close()
            return stats

        if not games_to_merge:
            src_conn.close()
            return stats

        # Insert games into destination
        dest_conn = sqlite3.connect(str(dest_db))

        # Get column names from games table
        columns = list(games_to_merge[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        col_names = ", ".join(columns)

        insert_sql = f"INSERT OR IGNORE INTO games ({col_names}) VALUES ({placeholders})"

        for game in games_to_merge:
            try:
                values = [game.get(col) for col in columns]
                dest_conn.execute(insert_sql, values)
                stats.games_merged += 1
            except sqlite3.Error as e:
                stats.errors.append(f"Insert error: {e}")

        # Copy related tables for merged games
        for table_name in RELATED_TABLES:
            try:
                # Check if source table exists
                cursor = src_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                if not cursor.fetchone():
                    continue

                # Check if dest table exists
                cursor = dest_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                if not cursor.fetchone():
                    continue

                # Get column names from BOTH source and destination tables
                # This handles schema differences between source (v11+) and canonical (v5) schemas
                cursor = src_conn.execute(f"PRAGMA table_info({table_name})")
                src_cols = [row[1] for row in cursor.fetchall()]

                cursor = dest_conn.execute(f"PRAGMA table_info({table_name})")
                dest_cols = [row[1] for row in cursor.fetchall()]

                if not src_cols or not dest_cols:
                    continue

                # Find columns that exist in BOTH tables (intersection)
                # This allows copying between schemas with different column sets
                common_cols = [c for c in src_cols if c in dest_cols]
                if not common_cols:
                    logger.debug(f"  No common columns between source and dest for {table_name}")
                    continue

                # Log schema difference for debugging
                src_only = set(src_cols) - set(dest_cols)
                if src_only:
                    logger.debug(f"  {table_name}: source-only columns will be dropped: {src_only}")

                # Build select/insert using only common columns
                select_cols = ", ".join(common_cols)
                placeholders = ", ".join(["?" for _ in common_cols])
                col_indices = [src_cols.index(c) for c in common_cols]

                # Copy rows for merged games
                for game_id in game_ids_to_merge:
                    cursor = src_conn.execute(
                        f"SELECT {select_cols} FROM {table_name} WHERE game_id = ?",
                        (game_id,)
                    )
                    rows = cursor.fetchall()

                    for row in rows:
                        try:
                            dest_conn.execute(
                                f"INSERT OR IGNORE INTO {table_name} ({select_cols}) VALUES ({placeholders})",
                                tuple(row)
                            )
                        except sqlite3.Error as e:
                            logger.debug(f"  Insert error for {table_name}: {e}")

            except sqlite3.Error as e:
                logger.debug(f"  Could not copy {table_name}: {e}")

        dest_conn.commit()
        dest_conn.close()
        src_conn.close()

    except sqlite3.Error as e:
        stats.errors.append(f"Database error: {e}")
        logger.error(f"Error merging from {source_db}: {e}")

    return stats


def consolidate_config(
    board_type: str,
    num_players: int,
    source_dir: Path = OWC_SELFPLAY_DIR,
    dest_dir: Path = OWC_CANONICAL_DIR,
    dry_run: bool = False,
) -> ConsolidationResult:
    """Consolidate all jsonl databases for a specific config."""
    config = f"{board_type}_{num_players}p"
    result = ConsolidationResult(config=config)

    logger.info(f"Consolidating {config}...")

    # Find all source databases
    source_dbs = find_jsonl_databases(source_dir)
    result.source_dbs = len(source_dbs)

    if not source_dbs:
        logger.warning(f"  No source databases found in {source_dir}")
        return result

    # Setup destination database
    dest_db = dest_dir / f"canonical_{board_type}_{num_players}p.db"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        ensure_canonical_schema(dest_db)

    # Get existing game IDs for deduplication
    existing_ids = get_existing_game_ids(dest_db)
    result.total_games_before = len(existing_ids)

    logger.info(f"  Destination: {dest_db}")
    logger.info(f"  Existing games: {result.total_games_before}")
    logger.info(f"  Scanning {len(source_dbs)} source databases...")

    # First pass: count games per source
    sources_with_data = []
    for db in source_dbs:
        count = count_games_for_config(db, board_type, num_players)
        if count > 0:
            sources_with_data.append((db, count))
            logger.info(f"    {db.parent.name}: {count} games")

    if not sources_with_data:
        logger.info(f"  No {config} games found in any source database")
        return result

    # Second pass: merge games
    for db, count in sources_with_data:
        stats = merge_games(db, dest_db, board_type, num_players, existing_ids, dry_run)
        result.merge_stats.append(stats)
        result.games_added += stats.games_merged

        if stats.games_merged > 0:
            logger.info(
                f"    Merged {stats.games_merged}/{stats.games_scanned} from {db.parent.name} "
                f"({stats.games_duplicate} dups, {stats.games_invalid} invalid)"
            )

    # Final count
    if not dry_run:
        result.total_games_after = len(get_existing_game_ids(dest_db))
    else:
        result.total_games_after = result.total_games_before + result.games_added

    logger.info(
        f"  {config}: {result.total_games_before} -> {result.total_games_after} games "
        f"(+{result.games_added})"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate jsonl_aggregated databases into canonical databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--config", type=str, help="Specific config (e.g., hex8_4p)")
    parser.add_argument("--source-dir", type=str, default=str(OWC_SELFPLAY_DIR))
    parser.add_argument("--dest-dir", type=str, default=str(OWC_CANONICAL_DIR))
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)

    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")

    # Determine which configs to process
    if args.config:
        parts = args.config.split("_")
        if len(parts) == 2 and parts[1].endswith("p"):
            board_type = parts[0]
            num_players = int(parts[1].rstrip("p"))
            configs = [(board_type, num_players)]
        else:
            logger.error(f"Invalid config format: {args.config} (expected: hex8_2p)")
            sys.exit(1)
    else:
        configs = ALL_CONFIGS

    # Process each config
    results = []
    total_added = 0

    for board_type, num_players in configs:
        result = consolidate_config(
            board_type, num_players,
            source_dir, dest_dir,
            args.dry_run,
        )
        results.append(result)
        total_added += result.games_added

    # Summary
    print("\n" + "=" * 60)
    print("CONSOLIDATION SUMMARY")
    print("=" * 60)
    for r in results:
        if r.games_added > 0 or r.total_games_after > 0:
            print(f"  {r.config}: {r.total_games_before} -> {r.total_games_after} (+{r.games_added})")
    print(f"\nTotal games added: {total_added}")
    if args.dry_run:
        print("\n(DRY RUN - no changes were made)")


if __name__ == "__main__":
    main()
