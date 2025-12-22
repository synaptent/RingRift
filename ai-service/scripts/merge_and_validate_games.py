#!/usr/bin/env python3
"""Merge game databases with parity validation and canonical upgrade.

This script:
1. Reads games from source databases (jsonl_aggregated.db, etc.)
2. Validates each game against canonical rules (Python engine replay)
3. For games that fail validation, attempts to "upgrade" by replaying moves
   through the canonical engine and recording the corrected trajectory
4. Merges valid/upgraded games into the target selfplay.db
5. Tags all games with their parity status

RR-CANONICAL-MERGE-2025-12-22

Usage:
    # Merge jsonl_aggregated.db into selfplay.db with validation
    python scripts/merge_and_validate_games.py \
        --source data/games/jsonl_aggregated.db \
        --target data/games/selfplay.db \
        --validate

    # Merge with upgrade attempt for failed games
    python scripts/merge_and_validate_games.py \
        --source data/games/jsonl_aggregated.db \
        --target data/games/selfplay.db \
        --validate --upgrade

    # Merge multiple sources
    python scripts/merge_and_validate_games.py \
        --source data/games/jsonl_aggregated.db \
        --source data/games/tournament.db \
        --target data/games/selfplay.db

    # Dry run (don't modify target)
    python scripts/merge_and_validate_games.py \
        --source data/games/jsonl_aggregated.db \
        --target data/games/selfplay.db \
        --dry-run

    # Filter by board type
    python scripts/merge_and_validate_games.py \
        --source data/games/jsonl_aggregated.db \
        --target data/games/selfplay.db \
        --board-type hexagonal --board-type square19
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.game_engine import GameEngine
from app.models import BoardType, GameStatus, Move, MoveType
from app.training.initial_state import create_initial_state
from app.rules.serialization import serialize_game_state, deserialize_game_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class GameRecord:
    """A game record from the database."""
    game_id: str
    board_type: str
    num_players: int
    winner: int | None
    termination_reason: str | None
    total_moves: int
    created_at: str
    initial_state_json: str | None
    moves_json: str | None  # JSON list of moves
    metadata_json: str | None
    source: str | None
    schema_version: int


@dataclass
class ValidationResult:
    """Result of validating a game."""
    game_id: str
    status: str  # 'passed', 'failed', 'error', 'upgraded'
    divergence_move: int | None = None
    error_message: str | None = None
    upgraded_moves: list[dict] | None = None
    upgraded_states: list[dict] | None = None


@dataclass
class MergeStats:
    """Statistics from merge operation."""
    source_games: int = 0
    already_exists: int = 0
    passed_validation: int = 0
    failed_validation: int = 0
    upgraded: int = 0
    upgrade_failed: int = 0
    errors: int = 0
    merged: int = 0
    skipped_config: int = 0


# -----------------------------------------------------------------------------
# Database Operations
# -----------------------------------------------------------------------------

def get_existing_game_ids(db_path: str) -> set[str]:
    """Get set of game IDs already in database."""
    if not os.path.exists(db_path):
        return set()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT game_id FROM games")
        return {row[0] for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def iter_games(
    db_path: str,
    board_types: list[str] | None = None,
    limit: int | None = None,
) -> Iterator[GameRecord]:
    """Iterate over games in a database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if games table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
    if not cursor.fetchone():
        logger.warning(f"No 'games' table in {db_path}, skipping")
        conn.close()
        return

    query = "SELECT * FROM games"
    params = []

    if board_types:
        placeholders = ",".join("?" * len(board_types))
        query += f" WHERE board_type IN ({placeholders})"
        params.extend(board_types)

    if limit:
        query += f" LIMIT {limit}"

    # Get column names to check what exists
    cursor.execute(query, params)
    col_names = [desc[0] for desc in cursor.description] if cursor.description else []

    def safe_get(row, key, default=None):
        """Safely get a value from sqlite3.Row."""
        if key in col_names:
            val = row[key]
            return val if val is not None else default
        return default

    try:
        for row in cursor:
            yield GameRecord(
                game_id=row["game_id"],
                board_type=row["board_type"],
                num_players=row["num_players"],
                winner=row["winner"],
                termination_reason=row["termination_reason"],
                total_moves=row["total_moves"],
                created_at=row["created_at"],
                initial_state_json=safe_get(row, "initial_state_json"),
                moves_json=None,  # Will load separately
                metadata_json=safe_get(row, "metadata_json"),
                source=safe_get(row, "source"),
                schema_version=safe_get(row, "schema_version", 1),
            )
    finally:
        conn.close()


def get_game_moves(db_path: str, game_id: str) -> list[dict]:
    """Get moves for a game from database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT * FROM game_moves WHERE game_id = ? ORDER BY move_number",
            (game_id,)
        )
        moves = []
        for row in cursor:
            move_data = {
                "move_number": row["move_number"],
                "player": row["player"],
                "move_type": row["move_type"],
                "move_json": row["move_json"],
            }
            moves.append(move_data)
        return moves
    finally:
        conn.close()


def get_game_initial_state(db_path: str, game_id: str) -> dict | None:
    """Get initial state for a game."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Check what tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        # Try game_state_snapshots first (move_number = 0)
        if "game_state_snapshots" in tables:
            cursor.execute(
                "SELECT state_json FROM game_state_snapshots WHERE game_id = ? AND move_number = 0",
                (game_id,)
            )
            row = cursor.fetchone()
            if row and row["state_json"]:
                return json.loads(row["state_json"])

        # Try games table - check if column exists
        if "games" in tables:
            cursor.execute("PRAGMA table_info(games)")
            columns = {row[1] for row in cursor.fetchall()}

            if "initial_state_json" in columns:
                cursor.execute(
                    "SELECT initial_state_json FROM games WHERE game_id = ?",
                    (game_id,)
                )
                row = cursor.fetchone()
                if row and row["initial_state_json"]:
                    return json.loads(row["initial_state_json"])

        return None
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Validation and Upgrade
# -----------------------------------------------------------------------------

def parse_move(move_data: dict) -> Move | None:
    """Parse a move from database format to Move object."""
    try:
        move_json = move_data.get("move_json")
        if not move_json:
            return None

        if isinstance(move_json, str):
            move_dict = json.loads(move_json)
        else:
            move_dict = move_json

        move_type = MoveType(move_dict.get("type", move_dict.get("move_type")))

        return Move(
            type=move_type,
            player=move_dict.get("player", move_data.get("player")),
            from_pos=move_dict.get("from_pos") or move_dict.get("from"),
            to_pos=move_dict.get("to_pos") or move_dict.get("to"),
            ring_color=move_dict.get("ring_color"),
            piece_type=move_dict.get("piece_type"),
            position=move_dict.get("position"),
            target_player=move_dict.get("target_player"),
            target_position=move_dict.get("target_position"),
            direction=move_dict.get("direction"),
            eliminated_player=move_dict.get("eliminated_player"),
            pieces_eliminated=move_dict.get("pieces_eliminated"),
            line_positions=move_dict.get("line_positions"),
            swap_piece_type=move_dict.get("swap_piece_type"),
        )
    except Exception as e:
        logger.debug(f"Failed to parse move: {e}")
        return None


def validate_game(
    game: GameRecord,
    moves: list[dict],
    initial_state_dict: dict | None = None,
) -> ValidationResult:
    """Validate a game by replaying through canonical engine.

    Returns ValidationResult with status and any divergence info.
    """
    try:
        # Create initial state
        if initial_state_dict:
            state = deserialize_game_state(initial_state_dict)
        else:
            state = create_initial_state(
                board_type=game.board_type,
                num_players=game.num_players,
            )

        # Replay each move
        for i, move_data in enumerate(moves):
            if state.game_status != GameStatus.ACTIVE:
                # Game ended early - check if this is expected
                break

            move = parse_move(move_data)
            if move is None:
                return ValidationResult(
                    game_id=game.game_id,
                    status="error",
                    error_message=f"Failed to parse move {i}",
                )

            # Check if move is valid
            current_player = state.current_player
            valid_moves = GameEngine.get_valid_moves(state, current_player)

            # Also check bookkeeping moves
            req = GameEngine.get_phase_requirement(state, current_player)
            bookkeeping = None
            if req is not None:
                bookkeeping = GameEngine.synthesize_bookkeeping_move(req, state)

            # Try to match move
            move_matched = False
            for vm in valid_moves:
                if _moves_match(move, vm):
                    move_matched = True
                    break

            if not move_matched and bookkeeping and _moves_match(move, bookkeeping):
                move_matched = True

            if not move_matched:
                return ValidationResult(
                    game_id=game.game_id,
                    status="failed",
                    divergence_move=i,
                    error_message=f"Move {i} not valid in canonical engine",
                )

            # Apply move
            try:
                state = GameEngine.apply_move(state, move)
            except Exception as e:
                return ValidationResult(
                    game_id=game.game_id,
                    status="failed",
                    divergence_move=i,
                    error_message=f"Move {i} failed to apply: {e}",
                )

        return ValidationResult(
            game_id=game.game_id,
            status="passed",
        )

    except Exception as e:
        return ValidationResult(
            game_id=game.game_id,
            status="error",
            error_message=str(e),
        )


def upgrade_game(
    game: GameRecord,
    moves: list[dict],
) -> ValidationResult:
    """Attempt to upgrade a game by replaying and finding canonical equivalents.

    For each move that doesn't match exactly, tries to find a semantically
    equivalent valid move in the canonical engine.
    """
    try:
        state = create_initial_state(
            board_type=game.board_type,
            num_players=game.num_players,
        )

        upgraded_moves = []
        upgraded_states = [serialize_game_state(state)]

        for i, move_data in enumerate(moves):
            if state.game_status != GameStatus.ACTIVE:
                break

            move = parse_move(move_data)
            if move is None:
                return ValidationResult(
                    game_id=game.game_id,
                    status="upgrade_failed",
                    divergence_move=i,
                    error_message=f"Cannot parse move {i}",
                )

            current_player = state.current_player
            valid_moves = GameEngine.get_valid_moves(state, current_player)

            # Check bookkeeping
            req = GameEngine.get_phase_requirement(state, current_player)
            bookkeeping = None
            if req is not None:
                bookkeeping = GameEngine.synthesize_bookkeeping_move(req, state)

            # Try exact match first
            matched_move = None
            for vm in valid_moves:
                if _moves_match(move, vm):
                    matched_move = vm
                    break

            if matched_move is None and bookkeeping and _moves_match(move, bookkeeping):
                matched_move = bookkeeping

            # If no exact match, try semantic match
            if matched_move is None:
                matched_move = _find_semantic_match(move, valid_moves, bookkeeping)

            if matched_move is None:
                return ValidationResult(
                    game_id=game.game_id,
                    status="upgrade_failed",
                    divergence_move=i,
                    error_message=f"No valid equivalent for move {i}",
                )

            # Apply matched move
            try:
                state = GameEngine.apply_move(state, matched_move)
                upgraded_moves.append(matched_move.model_dump(mode="json"))
                upgraded_states.append(serialize_game_state(state))
            except Exception as e:
                return ValidationResult(
                    game_id=game.game_id,
                    status="upgrade_failed",
                    divergence_move=i,
                    error_message=f"Failed to apply upgraded move {i}: {e}",
                )

        return ValidationResult(
            game_id=game.game_id,
            status="upgraded",
            upgraded_moves=upgraded_moves,
            upgraded_states=upgraded_states,
        )

    except Exception as e:
        return ValidationResult(
            game_id=game.game_id,
            status="upgrade_failed",
            error_message=str(e),
        )


def _moves_match(m1: Move, m2: Move) -> bool:
    """Check if two moves are equivalent."""
    if m1.type != m2.type:
        return False
    if m1.player != m2.player:
        return False

    # Type-specific matching
    if m1.type == MoveType.RING_PLACEMENT:
        return m1.position == m2.position and m1.ring_color == m2.ring_color
    elif m1.type == MoveType.CAP_PLACEMENT:
        return m1.position == m2.position
    elif m1.type == MoveType.MOVEMENT:
        return m1.from_pos == m2.from_pos and m1.to_pos == m2.to_pos
    elif m1.type == MoveType.CAPTURE:
        return (m1.from_pos == m2.from_pos and
                m1.to_pos == m2.to_pos and
                m1.direction == m2.direction)
    elif m1.type == MoveType.ELIMINATION:
        return (m1.target_player == m2.target_player and
                m1.target_position == m2.target_position)
    elif m1.type in (MoveType.LINE_COLLAPSE, MoveType.TERRITORY_COLLAPSE):
        # Compare line positions as sets
        lp1 = set(tuple(p) if isinstance(p, list) else p for p in (m1.line_positions or []))
        lp2 = set(tuple(p) if isinstance(p, list) else p for p in (m2.line_positions or []))
        return lp1 == lp2
    elif m1.type == MoveType.SWAP:
        return m1.position == m2.position and m1.swap_piece_type == m2.swap_piece_type
    elif m1.type == MoveType.PASS:
        return True

    # Default: compare all fields
    return m1.model_dump() == m2.model_dump()


def _find_semantic_match(
    original: Move,
    valid_moves: list[Move],
    bookkeeping: Move | None,
) -> Move | None:
    """Find a semantically equivalent move from valid moves.

    This handles cases where move representation differs slightly
    (e.g., position format "1,2" vs [1, 2]).
    """
    # Try bookkeeping first for phase transitions
    if bookkeeping and original.type == bookkeeping.type:
        return bookkeeping

    # For movements, try matching source/dest with flexible position format
    if original.type == MoveType.MOVEMENT:
        orig_from = _normalize_pos(original.from_pos)
        orig_to = _normalize_pos(original.to_pos)

        for vm in valid_moves:
            if vm.type == MoveType.MOVEMENT:
                vm_from = _normalize_pos(vm.from_pos)
                vm_to = _normalize_pos(vm.to_pos)
                if orig_from == vm_from and orig_to == vm_to:
                    return vm

    # For placements, try matching position
    if original.type in (MoveType.RING_PLACEMENT, MoveType.CAP_PLACEMENT):
        orig_pos = _normalize_pos(original.position)

        for vm in valid_moves:
            if vm.type == original.type:
                vm_pos = _normalize_pos(vm.position)
                if orig_pos == vm_pos:
                    if original.type == MoveType.RING_PLACEMENT:
                        if original.ring_color == vm.ring_color:
                            return vm
                    else:
                        return vm

    return None


def _normalize_pos(pos: Any) -> tuple | None:
    """Normalize position to tuple format."""
    if pos is None:
        return None
    if isinstance(pos, tuple):
        return pos
    if isinstance(pos, list):
        return tuple(pos)
    if isinstance(pos, str):
        parts = pos.split(",")
        return tuple(int(p) for p in parts)
    return None


# -----------------------------------------------------------------------------
# Merge Operations
# -----------------------------------------------------------------------------

def ensure_target_schema(db_path: str) -> None:
    """Ensure target database has required schema."""
    from app.db.game_replay import GameReplayDB

    # Just opening with GameReplayDB will run migrations
    # No need to close explicitly - uses internal connection management
    _ = GameReplayDB(db_path)


def insert_game(
    target_conn: sqlite3.Connection,
    game: GameRecord,
    moves: list[dict],
    validation: ValidationResult,
    upgraded: bool = False,
) -> bool:
    """Insert a game into target database."""
    try:
        cursor = target_conn.cursor()

        # Determine source tag
        source = game.source or "merged"
        if upgraded:
            source = f"{source}_upgraded"

        # Insert game record
        # game_status and total_turns are NOT NULL in schema, so provide defaults
        cursor.execute("""
            INSERT OR IGNORE INTO games (
                game_id, board_type, num_players, winner, termination_reason,
                total_moves, total_turns, game_status, created_at, source,
                schema_version, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game.game_id,
            game.board_type,
            game.num_players,
            game.winner,
            game.termination_reason,
            len(moves),
            len(moves),  # total_turns = total_moves for merged games
            "completed",  # game_status for merged games
            game.created_at,
            source,
            9,  # Current schema version
            game.metadata_json,
        ))

        if cursor.rowcount == 0:
            return False  # Already exists

        # Insert moves
        use_moves = validation.upgraded_moves if upgraded else moves
        for i, move_data in enumerate(use_moves):
            if isinstance(move_data, dict):
                move_json = json.dumps(move_data)
                player = move_data.get("player", 0)
                move_type = move_data.get("type", move_data.get("move_type", "unknown"))
                phase = move_data.get("phase", "play")
            else:
                move_json = json.dumps(move_data)
                player = 0
                move_type = "unknown"
                phase = "play"

            cursor.execute("""
                INSERT INTO game_moves (
                    game_id, move_number, turn_number, player, phase, move_type, move_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (game.game_id, i, i, player, phase, move_type, move_json))

        # Add parity status
        try:
            cursor.execute("""
                UPDATE games SET parity_status = ?, parity_checked_at = ?
                WHERE game_id = ?
            """, (
                validation.status,
                datetime.utcnow().isoformat(),
                game.game_id,
            ))
        except sqlite3.OperationalError:
            pass  # Column might not exist

        return True

    except Exception as e:
        logger.error(f"Failed to insert game {game.game_id}: {e}")
        return False


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def merge_databases(
    sources: list[str],
    target: str,
    validate: bool = True,
    upgrade: bool = False,
    board_types: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> MergeStats:
    """Merge source databases into target with validation."""
    stats = MergeStats()

    # Ensure target schema
    if not dry_run:
        ensure_target_schema(target)

    # Get existing games
    existing_ids = get_existing_game_ids(target)
    logger.info(f"Target has {len(existing_ids)} existing games")

    # Open target connection
    if not dry_run:
        target_conn = sqlite3.connect(target)
        target_conn.execute("PRAGMA journal_mode=WAL")
    else:
        target_conn = None

    try:
        for source_path in sources:
            if not os.path.exists(source_path):
                logger.warning(f"Source not found: {source_path}")
                continue

            logger.info(f"Processing source: {source_path}")

            for game in iter_games(source_path, board_types, limit):
                stats.source_games += 1

                if game.game_id in existing_ids:
                    stats.already_exists += 1
                    continue

                # Get moves
                moves = get_game_moves(source_path, game.game_id)
                if not moves:
                    logger.debug(f"No moves found for game {game.game_id}")
                    stats.errors += 1
                    continue
                logger.debug(f"Game {game.game_id}: {len(moves)} moves")

                # Validate if requested
                validation = ValidationResult(game_id=game.game_id, status="skipped")

                if validate:
                    initial_state = get_game_initial_state(source_path, game.game_id)
                    validation = validate_game(game, moves, initial_state)

                    if validation.status == "passed":
                        stats.passed_validation += 1
                    elif validation.status == "failed":
                        stats.failed_validation += 1

                        if upgrade:
                            # Try to upgrade
                            upgrade_result = upgrade_game(game, moves)
                            if upgrade_result.status == "upgraded":
                                validation = upgrade_result
                                stats.upgraded += 1
                            else:
                                stats.upgrade_failed += 1
                                logger.debug(
                                    f"Upgrade failed for {game.game_id}: "
                                    f"{upgrade_result.error_message}"
                                )
                                continue
                        else:
                            continue
                    elif validation.status == "error":
                        stats.errors += 1
                        continue

                # Insert game
                if not dry_run and target_conn:
                    upgraded = validation.status == "upgraded"
                    logger.debug(f"Inserting game {game.game_id}...")
                    if insert_game(target_conn, game, moves, validation, upgraded):
                        stats.merged += 1
                        existing_ids.add(game.game_id)
                        logger.debug(f"Successfully merged {game.game_id}")
                    else:
                        logger.debug(f"Insert returned False for {game.game_id}")
                else:
                    stats.merged += 1

                # Progress logging
                if stats.source_games % 1000 == 0:
                    logger.info(
                        f"Progress: {stats.source_games} processed, "
                        f"{stats.merged} merged, {stats.failed_validation} failed"
                    )

        # Commit
        if target_conn:
            target_conn.commit()

    finally:
        if target_conn:
            target_conn.close()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge game databases with validation")
    parser.add_argument(
        "--source", "-s",
        action="append",
        required=True,
        help="Source database path (can specify multiple)",
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Target database path",
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
        help="Limit number of games per source",
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

    logger.info("=" * 60)
    logger.info("MERGE AND VALIDATE GAMES")
    logger.info("=" * 60)
    logger.info(f"Sources: {args.source}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Validate: {args.validate}")
    logger.info(f"Upgrade: {args.upgrade}")
    logger.info(f"Board types: {args.board_type}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)

    start_time = time.time()

    stats = merge_databases(
        sources=args.source,
        target=args.target,
        validate=args.validate,
        upgrade=args.upgrade,
        board_types=args.board_type,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    duration = time.time() - start_time

    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Source games: {stats.source_games}")
    logger.info(f"Already exists: {stats.already_exists}")
    logger.info(f"Passed validation: {stats.passed_validation}")
    logger.info(f"Failed validation: {stats.failed_validation}")
    logger.info(f"Upgraded: {stats.upgraded}")
    logger.info(f"Upgrade failed: {stats.upgrade_failed}")
    logger.info(f"Errors: {stats.errors}")
    logger.info(f"Merged: {stats.merged}")
    logger.info(f"Duration: {duration:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
