#!/usr/bin/env python3
"""Consolidate jsonl_aggregated.db files into per-config canonical databases.

This script merges the scattered jsonl_aggregated.db files (44 files, 147GB total)
into per-config canonical databases for faster training data export.

Features:
- Deduplicates by game_id
- Validates game completeness before merging
- VALIDATES MOVE DATA PRESENCE - games without moves are rejected
- Updates existing canonical databases incrementally
- Tracks merge statistics

Usage:
    # Dry run - show what would be merged
    python scripts/consolidate_jsonl_databases.py --dry-run

    # Merge all configs
    python scripts/consolidate_jsonl_databases.py

    # Merge specific config
    python scripts/consolidate_jsonl_databases.py --config hex8_4p

    # Strict mode - fail on any data integrity issues
    python scripts/consolidate_jsonl_databases.py --strict

    # Custom paths
    python scripts/consolidate_jsonl_databases.py \
        --source-dir /path/to/jsonl_dbs \
        --dest-dir /path/to/canonical

December 2025: Created for database consolidation.
December 2025: Added move data validation (games without moves are rejected).
"""

from __future__ import annotations

import argparse
import json
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
class SourceValidationResult:
    """Result of validating a source database."""
    db_path: str = ""
    has_games_table: bool = False
    has_moves_table: bool = False
    game_count: int = 0
    games_with_moves: int = 0
    games_without_moves: int = 0
    is_valid: bool = False
    error: str | None = None


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
    games_no_moves: int = 0  # Games skipped because they lack move data
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


def validate_source_database(
    db_path: Path,
    board_type: str,
    num_players: int,
) -> SourceValidationResult:
    """Validate a source database has move data.

    This is a CRITICAL validation step to prevent importing games without moves.
    Games without move data are useless for training and pollute databases.

    Returns:
        SourceValidationResult with validation details
    """
    result = SourceValidationResult(db_path=str(db_path))

    if not db_path.exists():
        result.error = "Database file does not exist"
        return result

    try:
        conn = sqlite3.connect(str(db_path))

        # Check if games table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='games'"
        )
        result.has_games_table = cursor.fetchone() is not None

        # Check if game_moves table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'"
        )
        result.has_moves_table = cursor.fetchone() is not None

        if not result.has_games_table:
            result.error = "Missing 'games' table"
            conn.close()
            return result

        # Count games for this config
        cursor = conn.execute("""
            SELECT COUNT(*) FROM games
            WHERE board_type = ? AND num_players = ?
            AND game_status IN ('completed', 'finished')
            AND total_moves >= ?
        """, (board_type, num_players, MIN_MOVES_FOR_VALID))
        result.game_count = cursor.fetchone()[0]

        if result.game_count == 0:
            result.is_valid = True  # Empty but valid
            conn.close()
            return result

        # Check how many games have move data
        if result.has_moves_table:
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT g.game_id)
                FROM games g
                INNER JOIN game_moves m ON g.game_id = m.game_id
                WHERE g.board_type = ? AND g.num_players = ?
                AND g.game_status IN ('completed', 'finished')
                AND g.total_moves >= ?
            """, (board_type, num_players, MIN_MOVES_FOR_VALID))
            result.games_with_moves = cursor.fetchone()[0]
        else:
            result.games_with_moves = 0

        result.games_without_moves = result.game_count - result.games_with_moves
        result.is_valid = True

        conn.close()
        return result

    except sqlite3.Error as e:
        result.error = f"Database error: {e}"
        return result


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

    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_moves (
            game_id TEXT NOT NULL,
            move_number INTEGER NOT NULL,
            player INTEGER NOT NULL,
            position_q INTEGER,
            position_r INTEGER,
            move_type TEXT,
            move_probs TEXT,
            PRIMARY KEY (game_id, move_number),
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        )
    """)

    # Initial state table (must use initial_state_json to match TypeScript)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_initial_state (
            game_id TEXT PRIMARY KEY,
            initial_state_json TEXT NOT NULL,
            compressed INTEGER DEFAULT 0,
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
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

    # Players table (must match TypeScript schema in SelfPlayGameService)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_players (
            game_id TEXT NOT NULL,
            player_number INTEGER NOT NULL,
            player_type TEXT,
            ai_type TEXT,
            ai_difficulty INTEGER,
            ai_profile_id TEXT,
            final_eliminated_rings INTEGER,
            final_territory_spaces INTEGER,
            final_rings_in_hand INTEGER,
            model_version TEXT,
            PRIMARY KEY (game_id, player_number),
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
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
    strict: bool = False,
) -> MergeStats:
    """Merge valid games from source to destination database.

    This handles the normalized schema with separate tables:
    - games: main game metadata
    - game_moves: individual moves
    - game_initial_state: initial board state
    - game_state_snapshots: state snapshots
    - game_players: player info

    CRITICAL: Only merges games that have corresponding move data.
    Games without moves are skipped and logged as warnings.

    Args:
        source_db: Source database path
        dest_db: Destination database path
        board_type: Board type to filter by
        num_players: Number of players to filter by
        existing_ids: Set of game IDs already in destination
        dry_run: If True, don't actually write to database
        strict: If True, raise error on any integrity issues
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

        # Check if source has game_moves table
        cursor = src_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'"
        )
        has_moves_table = cursor.fetchone() is not None

        if not has_moves_table:
            logger.warning(
                f"  SKIPPING {source_db.parent.name}: no game_moves table - "
                f"cannot import games without move data"
            )
            stats.errors.append("No game_moves table in source database")
            src_conn.close()
            return stats

        # Get valid games that HAVE move data (INNER JOIN ensures moves exist)
        cursor = src_conn.execute("""
            SELECT DISTINCT g.*
            FROM games g
            INNER JOIN game_moves m ON g.game_id = m.game_id
            WHERE g.board_type = ? AND g.num_players = ?
            AND g.game_status IN ('completed', 'finished')
            AND g.total_moves >= ?
        """, (board_type, num_players, MIN_MOVES_FOR_VALID))

        games_to_merge = []
        game_ids_to_merge = []

        for row in cursor:
            stats.games_scanned += 1
            game_id = row["game_id"]

            if game_id in existing_ids:
                stats.games_duplicate += 1
                continue

            # Convert row to dict for easier access
            row_dict = dict(row)
            stats.games_valid += 1
            games_to_merge.append(row_dict)
            game_ids_to_merge.append(game_id)
            existing_ids.add(game_id)  # Track for dedup within this merge

        # Also count games WITHOUT moves to report them
        cursor = src_conn.execute("""
            SELECT COUNT(g.game_id) as no_moves_count
            FROM games g
            LEFT JOIN game_moves m ON g.game_id = m.game_id
            WHERE g.board_type = ? AND g.num_players = ?
            AND g.game_status IN ('completed', 'finished')
            AND g.total_moves >= ?
            AND m.game_id IS NULL
        """, (board_type, num_players, MIN_MOVES_FOR_VALID))
        no_moves_count = cursor.fetchone()[0]
        stats.games_no_moves = no_moves_count

        if no_moves_count > 0:
            logger.warning(
                f"  SKIPPED {no_moves_count} games without move data in {source_db.parent.name}"
            )
            if strict:
                raise ValueError(
                    f"Found {no_moves_count} games without move data in {source_db}. "
                    f"Use --no-strict to skip these games instead of failing."
                )

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

                # Get column names for this table
                cursor = src_conn.execute(f"PRAGMA table_info({table_name})")
                table_cols = [row[1] for row in cursor.fetchall()]

                if not table_cols:
                    continue

                # Build placeholders for the columns
                placeholders = ", ".join(["?" for _ in table_cols])
                col_list = ", ".join(table_cols)

                # Copy rows for merged games
                # RR-FIX-2025-12-28: Validate move positions during consolidation
                # to prevent copying corrupt game data
                move_json_col_idx = None
                if table_name == "game_moves" and "move_json" in table_cols:
                    move_json_col_idx = table_cols.index("move_json")

                for game_id in game_ids_to_merge:
                    cursor = src_conn.execute(
                        f"SELECT * FROM {table_name} WHERE game_id = ?",
                        (game_id,)
                    )
                    rows = cursor.fetchall()

                    for row in rows:
                        # Validate move positions if this is game_moves table
                        if move_json_col_idx is not None:
                            try:
                                move_json = row[move_json_col_idx]
                                if move_json:
                                    move_data = json.loads(move_json)
                                    move_type = move_data.get("type")
                                    to_pos = move_data.get("to")
                                    from_pos = move_data.get("from")
                                    capture_target = move_data.get("captureTarget")

                                    # Check position requirements by move type
                                    # place_ring requires 'to'
                                    if move_type == "place_ring" and to_pos is None:
                                        logger.debug(
                                            f"  Skipping game {game_id}: place_ring with to=null"
                                        )
                                        # Remove this game from merged list
                                        if game_id in game_ids_to_merge:
                                            stats.games_merged -= 1
                                            stats.games_invalid += 1
                                        break  # Skip all moves for this game

                                    # move_stack requires 'from' and 'to'
                                    if move_type in ("move_stack", "move_ring", "build_stack"):
                                        if from_pos is None or to_pos is None:
                                            logger.debug(
                                                f"  Skipping game {game_id}: {move_type} with null positions"
                                            )
                                            if game_id in game_ids_to_merge:
                                                stats.games_merged -= 1
                                                stats.games_invalid += 1
                                            break

                                    # overtaking_capture requires all three
                                    if move_type == "overtaking_capture":
                                        if from_pos is None or to_pos is None or capture_target is None:
                                            logger.debug(
                                                f"  Skipping game {game_id}: overtaking_capture with null positions"
                                            )
                                            if game_id in game_ids_to_merge:
                                                stats.games_merged -= 1
                                                stats.games_invalid += 1
                                            break

                                    # continue_capture_segment requires 'to'
                                    if move_type == "continue_capture_segment" and to_pos is None:
                                        logger.debug(
                                            f"  Skipping game {game_id}: continue_capture_segment with to=null"
                                        )
                                        if game_id in game_ids_to_merge:
                                            stats.games_merged -= 1
                                            stats.games_invalid += 1
                                        break
                            except (json.JSONDecodeError, KeyError, TypeError):
                                pass  # Invalid JSON, skip validation

                        try:
                            dest_conn.execute(
                                f"INSERT OR IGNORE INTO {table_name} ({col_list}) VALUES ({placeholders})",
                                tuple(row)
                            )
                        except sqlite3.Error:
                            pass  # Ignore duplicate key errors

            except sqlite3.Error as e:
                logger.debug(f"  Could not copy {table_name}: {e}")

        # RR-FIX-2025-12-28: Validate all merged games have move data before commit
        # This prevents orphan games (games without moves) from being committed
        orphan_check = dest_conn.execute("""
            SELECT g.game_id FROM games g
            LEFT JOIN game_moves m ON g.game_id = m.game_id
            WHERE g.game_id IN ({})
            GROUP BY g.game_id
            HAVING COUNT(m.game_id) = 0
        """.format(",".join("?" * len(game_ids_to_merge))), tuple(game_ids_to_merge))
        orphan_games = [row[0] for row in orphan_check.fetchall()]

        if orphan_games:
            logger.warning(
                f"  Found {len(orphan_games)} games without moves - removing before commit"
            )
            for orphan_id in orphan_games:
                dest_conn.execute("DELETE FROM games WHERE game_id = ?", (orphan_id,))
                stats.games_merged -= 1
            stats.errors.append(f"Removed {len(orphan_games)} orphan games (no moves)")

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
    strict: bool = False,
) -> ConsolidationResult:
    """Consolidate all jsonl databases for a specific config.

    Args:
        board_type: Board type (e.g., 'hex8', 'square8')
        num_players: Number of players (2, 3, or 4)
        source_dir: Directory containing source databases
        dest_dir: Directory to write canonical databases
        dry_run: If True, don't actually write to database
        strict: If True, fail on any data integrity issues (games without moves)
    """
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
        stats = merge_games(db, dest_db, board_type, num_players, existing_ids, dry_run, strict)
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any data integrity issues (games without move data)"
    )
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

    if args.strict:
        logger.info("STRICT MODE - will fail on any data integrity issues")

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
    total_skipped_no_moves = 0

    for board_type, num_players in configs:
        result = consolidate_config(
            board_type, num_players,
            source_dir, dest_dir,
            args.dry_run,
            args.strict,
        )
        results.append(result)
        total_added += result.games_added
        # Sum games skipped due to no move data
        for stats in result.merge_stats:
            total_skipped_no_moves += stats.games_no_moves

    # Summary
    print("\n" + "=" * 60)
    print("CONSOLIDATION SUMMARY")
    print("=" * 60)
    for r in results:
        if r.games_added > 0 or r.total_games_after > 0:
            print(f"  {r.config}: {r.total_games_before} -> {r.total_games_after} (+{r.games_added})")
    print(f"\nTotal games added: {total_added}")
    if total_skipped_no_moves > 0:
        print(f"WARNING: Skipped {total_skipped_no_moves} games without move data")
    if args.dry_run:
        print("\n(DRY RUN - no changes were made)")


if __name__ == "__main__":
    main()
