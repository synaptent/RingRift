#!/usr/bin/env python3
"""
Upgrade phase annotations in existing games to match canonical spec.

This script fixes `phase_move_mismatch` parity errors by re-deriving phases
using the correct phase state machine. Old games may have incorrect phase
annotations due to bugs in earlier selfplay recording code.

The upgrade process:
1. Load each game's initial state and move sequence
2. Replay moves using GameEngine.apply_move() which calls advance_phases()
3. Extract the correct current_phase at each move
4. Update the database with corrected phase annotations
5. Re-validate via parity check

Usage:
    PYTHONPATH=. python scripts/upgrade_phase_annotations.py \
        --db data/games/canonical_square8_2p.db \
        --dry-run

    # Commit changes
    PYTHONPATH=. python scripts/upgrade_phase_annotations.py \
        --db data/games/canonical_square8_2p.db \
        --commit

December 28, 2025: Initial implementation for parity fix.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class UpgradeResult:
    """Result of upgrading a single game."""
    game_id: str
    success: bool
    moves_checked: int
    moves_fixed: int
    error: str | None = None


@dataclass
class UpgradeSummary:
    """Summary of upgrade operation."""
    total_games: int
    successful: int
    failed: int
    moves_fixed: int
    errors: list[str]


def load_game_data(conn: sqlite3.Connection, game_id: str) -> dict[str, Any] | None:
    """Load game initial state and moves from database."""
    cursor = conn.cursor()

    # Get initial state
    cursor.execute(
        """
        SELECT initial_state_json FROM game_initial_state WHERE game_id = ?
        """,
        (game_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None

    initial_state_json = row[0]

    # Get all moves in order
    cursor.execute(
        """
        SELECT move_number, move_json, phase
        FROM game_moves
        WHERE game_id = ?
        ORDER BY move_number
        """,
        (game_id,),
    )
    moves = cursor.fetchall()

    # Get history entries if available (v4+ schema)
    cursor.execute(
        """
        SELECT move_number, phase_before, phase_after
        FROM game_history_entries
        WHERE game_id = ?
        ORDER BY move_number
        """,
        (game_id,),
    )
    history_entries = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

    return {
        "initial_state_json": initial_state_json,
        "moves": moves,
        "history_entries": history_entries,
    }


def upgrade_game_phases(
    conn: sqlite3.Connection,
    game_id: str,
    dry_run: bool = True,
) -> UpgradeResult:
    """
    Upgrade phase annotations for a single game.

    Returns UpgradeResult with details of the upgrade.
    """
    from app.game_engine import GameEngine
    from app.db.game_replay import _deserialize_state, _deserialize_move, _decompress_json

    try:
        game_data = load_game_data(conn, game_id)
        if not game_data:
            return UpgradeResult(
                game_id=game_id,
                success=False,
                moves_checked=0,
                moves_fixed=0,
                error="Game not found or missing initial state",
            )

        # Parse initial state using canonical deserializer
        initial_state_json = game_data["initial_state_json"]
        if isinstance(initial_state_json, bytes):
            initial_state_json = _decompress_json(initial_state_json)

        state = _deserialize_state(initial_state_json)

        moves_checked = 0
        moves_fixed = 0
        phase_updates: list[tuple[str, int, str]] = []  # (game_id, move_number, phase)
        history_updates: list[tuple[str, str, int, str]] = []  # (game_id, phase_before, move_number)

        for move_number, move_json_raw, recorded_phase in game_data["moves"]:
            # Parse move using canonical deserializer
            if isinstance(move_json_raw, bytes):
                move_json_raw = _decompress_json(move_json_raw)

            move = _deserialize_move(move_json_raw)

            # Get current (correct) phase before move
            correct_phase = state.current_phase.value if hasattr(state.current_phase, "value") else str(state.current_phase)

            moves_checked += 1

            # Check if phase needs updating
            if recorded_phase != correct_phase:
                moves_fixed += 1
                phase_updates.append((game_id, move_number, correct_phase))

                # Also update history entry if exists
                if move_number in game_data["history_entries"]:
                    history_updates.append((correct_phase, game_id, move_number))

                if not dry_run:
                    logger.debug(
                        f"  Move {move_number}: {recorded_phase} -> {correct_phase} "
                        f"(move_type={move.type.value if hasattr(move.type, 'value') else move.type})"
                    )

            # Apply move to advance state
            state = GameEngine.apply_move(state, move)

        # Commit updates if not dry run
        if not dry_run and (phase_updates or history_updates):
            cursor = conn.cursor()

            # Update game_moves table
            cursor.executemany(
                """
                UPDATE game_moves SET phase = ? WHERE game_id = ? AND move_number = ?
                """,
                [(phase, gid, mn) for gid, mn, phase in phase_updates],
            )

            # Update game_history_entries table
            if history_updates:
                cursor.executemany(
                    """
                    UPDATE game_history_entries SET phase_before = ? WHERE game_id = ? AND move_number = ?
                    """,
                    history_updates,
                )

            conn.commit()

        return UpgradeResult(
            game_id=game_id,
            success=True,
            moves_checked=moves_checked,
            moves_fixed=moves_fixed,
        )

    except Exception as e:
        logger.error(f"Error upgrading game {game_id}: {e}")
        return UpgradeResult(
            game_id=game_id,
            success=False,
            moves_checked=0,
            moves_fixed=0,
            error=str(e),
        )


def get_games_needing_upgrade(conn: sqlite3.Connection, limit: int | None = None) -> list[str]:
    """
    Get list of game IDs that need phase upgrade.

    Returns games that have:
    1. parity_status = 'error' or
    2. parity_status = 'pending' (haven't been checked yet)
    """
    cursor = conn.cursor()

    query = """
        SELECT game_id FROM games
        WHERE parity_status IN ('error', 'pending', 'pending_gate')
        OR parity_status IS NULL
        ORDER BY created_at DESC
    """

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    return [row[0] for row in cursor.fetchall()]


def run_upgrade(
    db_path: Path,
    dry_run: bool = True,
    limit: int | None = None,
    game_ids: list[str] | None = None,
) -> UpgradeSummary:
    """
    Run phase upgrade on database.

    Args:
        db_path: Path to SQLite database
        dry_run: If True, don't commit changes
        limit: Max number of games to process
        game_ids: Specific game IDs to process (overrides limit)

    Returns:
        UpgradeSummary with results
    """
    conn = sqlite3.connect(db_path)

    try:
        if game_ids:
            games = game_ids
        else:
            games = get_games_needing_upgrade(conn, limit=limit)

        logger.info(f"Found {len(games)} games to check in {db_path}")

        if not games:
            return UpgradeSummary(
                total_games=0,
                successful=0,
                failed=0,
                moves_fixed=0,
                errors=[],
            )

        successful = 0
        failed = 0
        total_moves_fixed = 0
        errors: list[str] = []

        for i, game_id in enumerate(games):
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(games)} games...")

            result = upgrade_game_phases(conn, game_id, dry_run=dry_run)

            if result.success:
                successful += 1
                total_moves_fixed += result.moves_fixed

                if result.moves_fixed > 0:
                    logger.info(
                        f"Game {game_id}: fixed {result.moves_fixed}/{result.moves_checked} moves"
                    )
            else:
                failed += 1
                if result.error:
                    errors.append(f"{game_id}: {result.error}")

        # Update parity status for upgraded games if not dry run
        if not dry_run and successful > 0:
            cursor = conn.cursor()
            cursor.executemany(
                """
                UPDATE games SET parity_status = 'upgraded', parity_checked_at = datetime('now')
                WHERE game_id = ?
                """,
                [(gid,) for gid in games if gid not in [e.split(":")[0] for e in errors]],
            )
            conn.commit()
            logger.info(f"Marked {successful} games as 'upgraded'")

        return UpgradeSummary(
            total_games=len(games),
            successful=successful,
            failed=failed,
            moves_fixed=total_moves_fixed,
            errors=errors[:10],  # Limit error list
        )

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Upgrade phase annotations in existing games to match canonical spec"
    )
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Check phases without committing changes (default)",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Actually commit phase updates to database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of games to process",
    )
    parser.add_argument(
        "--game-ids",
        nargs="+",
        help="Specific game IDs to process",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        sys.exit(1)

    dry_run = not args.commit

    if dry_run:
        logger.info("DRY RUN - no changes will be made")
    else:
        logger.info("COMMIT MODE - changes will be written to database")

    summary = run_upgrade(
        db_path=args.db,
        dry_run=dry_run,
        limit=args.limit,
        game_ids=args.game_ids,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"UPGRADE SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total games checked: {summary.total_games}")
    logger.info(f"Successful: {summary.successful}")
    logger.info(f"Failed: {summary.failed}")
    logger.info(f"Total moves fixed: {summary.moves_fixed}")

    if summary.errors:
        logger.warning(f"\nErrors ({len(summary.errors)} shown):")
        for err in summary.errors:
            logger.warning(f"  - {err}")

    if dry_run and summary.moves_fixed > 0:
        logger.info(f"\nRe-run with --commit to apply {summary.moves_fixed} phase fixes")


if __name__ == "__main__":
    main()
