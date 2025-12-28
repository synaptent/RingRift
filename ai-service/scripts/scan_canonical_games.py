#!/usr/bin/env python3
"""
Canonical Game Scanner & Migrator

Scans scattered selfplay databases, validates each game for canonical compliance,
and migrates qualifying games to per-config canonical databases.

Usage:
    # Full scan with TS parity (thorough, requires Node.js)
    PYTHONPATH=. python scripts/scan_canonical_games.py \
        --source-dirs /Volumes/RingRift-Data/selfplay_repository \
                      /Volumes/RingRift-Data/cluster_games \
        --target-dir /Volumes/RingRift-Data/canonical_games \
        --state-file /Volumes/RingRift-Data/scan_state.json

    # Fast mode (Python-only, skip TS parity)
    PYTHONPATH=. python scripts/scan_canonical_games.py \
        --source-dirs /Volumes/RingRift-Data/selfplay_repository \
        --target-dir /Volumes/RingRift-Data/canonical_games \
        --skip-ts-parity

    # Dry run (preview without migrating)
    PYTHONPATH=. python scripts/scan_canonical_games.py \
        --source-dirs /Volumes/RingRift-Data/cluster_games \
        --target-dir /Volumes/RingRift-Data/canonical_games \
        --dry-run

December 28, 2025: Initial implementation for recovering canonical games.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ScanConfig:
    """Configuration for canonical game scanner."""
    source_dirs: list[Path]
    target_dir: Path
    state_file: Path
    batch_size: int = 100
    skip_ts_parity: bool = False
    dry_run: bool = False
    checkpoint_interval: int = 500


@dataclass
class GameInfo:
    """Basic game metadata for inventory."""
    game_id: str
    board_type: str
    num_players: int
    total_moves: int
    source: str | None
    parity_status: str | None
    source_db: Path


@dataclass
class ValidationResult:
    """Result of validating a single game."""
    game_id: str
    status: str  # "canonical", "invalid", "non_canonical_history", "parity_failed", "error"
    reason: str | None = None


@dataclass
class ScanState:
    """Persistent state for resumable scanning."""
    scanned_dbs: list[str] = field(default_factory=list)
    validated_games: dict[str, str] = field(default_factory=dict)  # game_id -> status
    migrated_count: int = 0
    error_count: int = 0
    last_checkpoint: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScanState":
        return cls(
            scanned_dbs=data.get("scanned_dbs", []),
            validated_games=data.get("validated_games", {}),
            migrated_count=data.get("migrated_count", 0),
            error_count=data.get("error_count", 0),
            last_checkpoint=data.get("last_checkpoint", ""),
        )


@dataclass
class ScanSummary:
    """Summary statistics from scan."""
    total_source_dbs: int = 0
    total_games_inventoried: int = 0
    games_with_moves: int = 0
    games_already_canonical: int = 0
    candidates_fast_validated: int = 0
    candidates_invalid_structure: int = 0
    candidates_non_canonical_history: int = 0
    candidates_passed_ts_parity: int = 0
    candidates_failed_ts_parity: int = 0
    candidates_ts_error: int = 0
    games_migrated: int = 0
    games_already_in_target: int = 0
    errors: int = 0
    configs_updated: dict[str, int] = field(default_factory=dict)


# =============================================================================
# State Management
# =============================================================================

def save_state(state: ScanState, path: Path) -> None:
    """Save scan state to JSON file."""
    state.last_checkpoint = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(state.to_dict(), f, indent=2)
    logger.debug(f"State saved to {path}")


def load_state(path: Path) -> ScanState | None:
    """Load scan state from JSON file if it exists."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        state = ScanState.from_dict(data)
        logger.info(f"Resumed state from {path} (last checkpoint: {state.last_checkpoint})")
        return state
    except Exception as e:
        logger.warning(f"Failed to load state from {path}: {e}")
        return None


# =============================================================================
# Database Discovery
# =============================================================================

def discover_source_databases(source_dirs: list[Path]) -> list[Path]:
    """Find all .db files containing games tables."""
    databases = []

    for source_dir in source_dirs:
        if not source_dir.exists():
            logger.warning(f"Source directory does not exist: {source_dir}")
            continue

        # Find all .db files recursively
        for db_path in source_dir.rglob("*.db"):
            # Skip canonical DBs (they're targets, not sources)
            if db_path.name.startswith("canonical_"):
                continue

            # Skip known non-game databases
            skip_names = {"elo_registry.db", "model_registry.db", "coordination.db", "event_log.db"}
            if db_path.name in skip_names:
                continue

            # Validate it has games and game_moves tables
            if has_games_tables(db_path):
                databases.append(db_path)
            else:
                logger.debug(f"Skipping {db_path} - no games/game_moves tables")

    return sorted(databases)


def has_games_tables(db_path: Path) -> bool:
    """Check if database has both games and game_moves tables."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('games', 'game_moves')")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        return "games" in tables and "game_moves" in tables
    except Exception:
        return False


# =============================================================================
# Game Inventory
# =============================================================================

def inventory_games(db_path: Path, state: ScanState) -> Iterator[GameInfo]:
    """Get all games with basic metadata from a database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check available columns
        cursor.execute("PRAGMA table_info(games)")
        columns = {row["name"] for row in cursor.fetchall()}

        # Build query based on available columns
        has_parity = "parity_status" in columns
        has_source = "source" in columns
        has_total_moves = "total_moves" in columns

        select_parts = ["game_id", "board_type", "num_players"]
        if has_total_moves:
            select_parts.append("total_moves")
        if has_source:
            select_parts.append("source")
        if has_parity:
            select_parts.append("parity_status")

        query = f"SELECT {', '.join(select_parts)} FROM games"

        cursor.execute(query)

        for row in cursor:
            game_id = row["game_id"]

            # Skip if already validated in this session
            if game_id in state.validated_games:
                continue

            # Get move count if not in games table
            if has_total_moves:
                total_moves = row["total_moves"] or 0
            else:
                cursor.execute("SELECT COUNT(*) FROM game_moves WHERE game_id = ?", (game_id,))
                total_moves = cursor.fetchone()[0]

            yield GameInfo(
                game_id=game_id,
                board_type=row["board_type"],
                num_players=row["num_players"],
                total_moves=total_moves,
                source=row.get("source") if has_source else None,
                parity_status=row.get("parity_status") if has_parity else None,
                source_db=db_path,
            )

        conn.close()

    except Exception as e:
        logger.error(f"Error inventorying {db_path}: {e}")


# =============================================================================
# Fast Pre-Validation (Python-only)
# =============================================================================

def fast_validate_game(db_path: Path, game_id: str) -> ValidationResult:
    """
    Python-only validation (no TS required).

    Checks:
    1. Game structure is valid (can be replayed)
    2. Canonical history (phase-move alignment)
    """
    try:
        # Import validation functions
        from scripts.check_ts_python_replay_parity import (
            classify_game_structure,
            validate_canonical_history_for_game,
        )
        from app.db.game_replay import GameReplayDB

        db = GameReplayDB(str(db_path), enforce_canonical_history=False)

        # Check structure
        structure, reason = classify_game_structure(db, game_id)
        if structure not in ("good", "mid_snapshot"):
            return ValidationResult(
                game_id=game_id,
                status="invalid",
                reason=f"Invalid structure: {structure} - {reason}",
            )

        # Check canonical history (phase-move alignment)
        canonical_report = validate_canonical_history_for_game(db, game_id)
        if not canonical_report.is_canonical:
            return ValidationResult(
                game_id=game_id,
                status="non_canonical_history",
                reason=f"Non-canonical at move {canonical_report.first_violation_move}: {canonical_report.first_violation_reason}",
            )

        return ValidationResult(
            game_id=game_id,
            status="candidate",  # Passed fast validation, needs TS parity
            reason=None,
        )

    except Exception as e:
        return ValidationResult(
            game_id=game_id,
            status="error",
            reason=str(e),
        )


# =============================================================================
# Full TS Parity Validation
# =============================================================================

def full_validate_game(db_path: Path, game_id: str, timeout: int = 30) -> ValidationResult:
    """
    Full TS/Python parity check (requires npx).
    """
    try:
        from scripts.check_ts_python_replay_parity import check_game_parity

        result = check_game_parity(db_path, game_id)

        # Check if passed
        if result.diverged_at is None and result.structure in ("good", "mid_snapshot"):
            return ValidationResult(
                game_id=game_id,
                status="canonical",
                reason=None,
            )

        # Check structure issues
        if result.structure not in ("good", "mid_snapshot"):
            return ValidationResult(
                game_id=game_id,
                status="invalid",
                reason=f"Structure: {result.structure} - {result.structure_reason}",
            )

        # Divergence detected
        return ValidationResult(
            game_id=game_id,
            status="parity_failed",
            reason=f"Diverged at move {result.diverged_at}: {', '.join(result.mismatch_kinds)}",
        )

    except Exception as e:
        return ValidationResult(
            game_id=game_id,
            status="error",
            reason=str(e),
        )


# =============================================================================
# Game Migration
# =============================================================================

def get_target_db_path(target_dir: Path, board_type: str, num_players: int) -> Path:
    """Get path to canonical database for this config."""
    return target_dir / f"canonical_{board_type}_{num_players}p.db"


def get_existing_game_ids(db_path: Path) -> set[str]:
    """Get all game_ids already in target database."""
    if not db_path.exists():
        return set()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT game_id FROM games")
        ids = {row[0] for row in cursor.fetchall()}
        conn.close()
        return ids
    except Exception as e:
        logger.warning(f"Error reading existing IDs from {db_path}: {e}")
        return set()


def ensure_target_schema(db_path: Path) -> None:
    """Ensure target database has proper schema."""
    from app.db.game_replay import GameReplayDB

    # Creating a GameReplayDB instance will ensure schema exists
    db = GameReplayDB(str(db_path))
    # Just accessing ensures initialization
    del db


def migrate_game(
    source_db_path: Path,
    target_db_path: Path,
    game_id: str,
    dry_run: bool = False,
) -> bool:
    """
    Copy a game and its moves from source to target database.

    Returns True if migration was successful.
    """
    if dry_run:
        return True

    try:
        source_conn = sqlite3.connect(source_db_path)
        source_conn.row_factory = sqlite3.Row

        # Ensure target exists with proper schema
        ensure_target_schema(target_db_path)

        target_conn = sqlite3.connect(target_db_path)

        # Get games columns from source
        cursor = source_conn.cursor()
        cursor.execute("PRAGMA table_info(games)")
        source_columns = [row[1] for row in cursor.fetchall()]

        # Get games columns from target
        target_cursor = target_conn.cursor()
        target_cursor.execute("PRAGMA table_info(games)")
        target_columns = {row[1] for row in target_cursor.fetchall()}

        # Find common columns
        common_columns = [c for c in source_columns if c in target_columns]

        # Copy games row
        columns_str = ", ".join(common_columns)
        placeholders = ", ".join(["?" for _ in common_columns])

        cursor.execute(f"SELECT {columns_str} FROM games WHERE game_id = ?", (game_id,))
        game_row = cursor.fetchone()

        if game_row:
            # Insert with parity_status = 'passed'
            values = list(game_row)

            # Update parity_status if it's in the columns
            if "parity_status" in common_columns:
                idx = common_columns.index("parity_status")
                values[idx] = "passed"

            target_cursor.execute(
                f"INSERT OR REPLACE INTO games ({columns_str}) VALUES ({placeholders})",
                values
            )

        # Copy game_moves rows
        cursor.execute("PRAGMA table_info(game_moves)")
        move_columns = [row[1] for row in cursor.fetchall()]

        target_cursor.execute("PRAGMA table_info(game_moves)")
        target_move_columns = {row[1] for row in target_cursor.fetchall()}

        common_move_columns = [c for c in move_columns if c in target_move_columns]
        move_columns_str = ", ".join(common_move_columns)
        move_placeholders = ", ".join(["?" for _ in common_move_columns])

        cursor.execute(f"SELECT {move_columns_str} FROM game_moves WHERE game_id = ?", (game_id,))
        move_rows = cursor.fetchall()

        for move_row in move_rows:
            target_cursor.execute(
                f"INSERT OR REPLACE INTO game_moves ({move_columns_str}) VALUES ({move_placeholders})",
                list(move_row)
            )

        # Copy game_initial_state if exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_initial_state'")
        if cursor.fetchone():
            cursor.execute("SELECT * FROM game_initial_state WHERE game_id = ?", (game_id,))
            initial_state_row = cursor.fetchone()
            if initial_state_row:
                target_cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='game_initial_state'"
                )
                if target_cursor.fetchone():
                    # Get columns
                    cursor.execute("PRAGMA table_info(game_initial_state)")
                    init_columns = [row[1] for row in cursor.fetchall()]
                    init_columns_str = ", ".join(init_columns)
                    init_placeholders = ", ".join(["?" for _ in init_columns])
                    target_cursor.execute(
                        f"INSERT OR REPLACE INTO game_initial_state ({init_columns_str}) VALUES ({init_placeholders})",
                        list(initial_state_row)
                    )

        target_conn.commit()
        source_conn.close()
        target_conn.close()

        return True

    except Exception as e:
        logger.error(f"Error migrating game {game_id}: {e}")
        return False


# =============================================================================
# Main Scanner
# =============================================================================

def run_scanner(config: ScanConfig) -> ScanSummary:
    """Run the canonical game scanner."""
    summary = ScanSummary()

    # Load or create state
    state = load_state(config.state_file) or ScanState()

    # Discover source databases
    logger.info("Discovering source databases...")
    source_dbs = discover_source_databases(config.source_dirs)
    summary.total_source_dbs = len(source_dbs)
    logger.info(f"Found {len(source_dbs)} source databases")

    # Filter out already scanned DBs
    remaining_dbs = [db for db in source_dbs if str(db) not in state.scanned_dbs]
    logger.info(f"Remaining to scan: {len(remaining_dbs)} databases")

    # Cache of existing game IDs per target DB
    target_cache: dict[Path, set[str]] = {}

    def get_target_ids(target_path: Path) -> set[str]:
        if target_path not in target_cache:
            target_cache[target_path] = get_existing_game_ids(target_path)
        return target_cache[target_path]

    # Process each source database
    games_processed = 0

    for db_idx, db_path in enumerate(remaining_dbs):
        logger.info(f"[{db_idx + 1}/{len(remaining_dbs)}] Processing {db_path.name}...")

        # Collect games from this DB
        games_in_db = []
        for game_info in inventory_games(db_path, state):
            summary.total_games_inventoried += 1

            # Skip games without moves
            if game_info.total_moves < 1:
                continue
            summary.games_with_moves += 1

            # Skip already quarantined games
            if game_info.parity_status == "non_canonical_history":
                continue

            # Skip already passed games (they're canonical)
            if game_info.parity_status == "passed":
                summary.games_already_canonical += 1
                continue

            games_in_db.append(game_info)

        logger.info(f"  Found {len(games_in_db)} candidate games")

        # Process games in batches
        for batch_start in range(0, len(games_in_db), config.batch_size):
            batch = games_in_db[batch_start:batch_start + config.batch_size]

            for game_info in batch:
                games_processed += 1

                # Fast pre-validation
                result = fast_validate_game(game_info.source_db, game_info.game_id)

                if result.status == "invalid":
                    summary.candidates_invalid_structure += 1
                    state.validated_games[game_info.game_id] = result.status
                    continue

                if result.status == "non_canonical_history":
                    summary.candidates_non_canonical_history += 1
                    state.validated_games[game_info.game_id] = result.status
                    continue

                if result.status == "error":
                    summary.errors += 1
                    state.validated_games[game_info.game_id] = result.status
                    continue

                summary.candidates_fast_validated += 1

                # Full TS parity (unless skipped)
                if not config.skip_ts_parity:
                    result = full_validate_game(game_info.source_db, game_info.game_id)

                    if result.status == "error":
                        summary.candidates_ts_error += 1
                        state.validated_games[game_info.game_id] = result.status
                        continue

                    if result.status == "parity_failed":
                        summary.candidates_failed_ts_parity += 1
                        state.validated_games[game_info.game_id] = result.status
                        continue

                    summary.candidates_passed_ts_parity += 1
                else:
                    # In skip mode, treat fast-validated as canonical
                    result = ValidationResult(
                        game_id=game_info.game_id,
                        status="canonical",
                        reason=None,
                    )

                # Migrate canonical game
                if result.status == "canonical":
                    target_path = get_target_db_path(
                        config.target_dir,
                        game_info.board_type,
                        game_info.num_players,
                    )

                    # Check if already in target
                    target_ids = get_target_ids(target_path)
                    if game_info.game_id in target_ids:
                        summary.games_already_in_target += 1
                        state.validated_games[game_info.game_id] = "already_migrated"
                        continue

                    # Migrate
                    if migrate_game(
                        game_info.source_db,
                        target_path,
                        game_info.game_id,
                        dry_run=config.dry_run,
                    ):
                        summary.games_migrated += 1
                        state.migrated_count += 1

                        # Track per-config
                        config_key = f"{game_info.board_type}_{game_info.num_players}p"
                        summary.configs_updated[config_key] = summary.configs_updated.get(config_key, 0) + 1

                        # Update cache
                        target_ids.add(game_info.game_id)

                    state.validated_games[game_info.game_id] = "canonical"

                # Checkpoint periodically
                if games_processed % config.checkpoint_interval == 0:
                    save_state(state, config.state_file)
                    logger.info(f"  Checkpoint: {games_processed} games processed, {summary.games_migrated} migrated")

        # Mark DB as scanned
        state.scanned_dbs.append(str(db_path))
        save_state(state, config.state_file)

    # Final state save
    state.error_count = summary.errors
    save_state(state, config.state_file)

    return summary


def print_summary(summary: ScanSummary, config: ScanConfig) -> None:
    """Print scan summary."""
    print("\n" + "=" * 60)
    print("CANONICAL GAME SCANNER SUMMARY")
    print("=" * 60)

    print(f"\nSource databases: {summary.total_source_dbs}")
    print(f"Total games inventoried: {summary.total_games_inventoried}")
    print(f"Games with moves: {summary.games_with_moves}")
    print(f"Already canonical: {summary.games_already_canonical}")

    print(f"\nFast validation:")
    print(f"  - Passed (candidates): {summary.candidates_fast_validated}")
    print(f"  - Invalid structure: {summary.candidates_invalid_structure}")
    print(f"  - Non-canonical history: {summary.candidates_non_canonical_history}")

    if not config.skip_ts_parity:
        print(f"\nTS parity validation:")
        print(f"  - Passed: {summary.candidates_passed_ts_parity}")
        print(f"  - Failed: {summary.candidates_failed_ts_parity}")
        print(f"  - Errors: {summary.candidates_ts_error}")

    print(f"\nMigration:")
    print(f"  - Games migrated: {summary.games_migrated}")
    print(f"  - Already in target: {summary.games_already_in_target}")
    print(f"  - Errors: {summary.errors}")

    if summary.configs_updated:
        print(f"\nPer-config additions:")
        for config_key, count in sorted(summary.configs_updated.items()):
            print(f"  - {config_key}: +{count} games")

    if config.dry_run:
        print(f"\n[DRY RUN - no games were actually migrated]")

    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scan scattered selfplay games and migrate canonical ones"
    )
    parser.add_argument(
        "--source-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Directories containing source databases to scan",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Directory for canonical databases",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("scan_state.json"),
        help="File to store scan state for resumability",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of games to process per batch",
    )
    parser.add_argument(
        "--skip-ts-parity",
        action="store_true",
        help="Skip TS parity validation (faster, less thorough)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without actually migrating games",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Save state every N games processed",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate directories
    for source_dir in args.source_dirs:
        if not source_dir.exists():
            logger.warning(f"Source directory does not exist: {source_dir}")

    args.target_dir.mkdir(parents=True, exist_ok=True)

    config = ScanConfig(
        source_dirs=args.source_dirs,
        target_dir=args.target_dir,
        state_file=args.state_file,
        batch_size=args.batch_size,
        skip_ts_parity=args.skip_ts_parity,
        dry_run=args.dry_run,
        checkpoint_interval=args.checkpoint_interval,
    )

    logger.info("Starting canonical game scanner...")
    if config.dry_run:
        logger.info("[DRY RUN MODE - no changes will be made]")
    if config.skip_ts_parity:
        logger.info("[FAST MODE - skipping TS parity validation]")

    start_time = time.time()
    summary = run_scanner(config)
    elapsed = time.time() - start_time

    print_summary(summary, config)

    logger.info(f"Completed in {elapsed:.1f}s")
    logger.info(f"State saved to {config.state_file}")


if __name__ == "__main__":
    main()
