#!/usr/bin/env python3
"""Backfill periodic state snapshots for existing games.

This script processes games in the database that don't have periodic snapshots
(e.g., older games recorded before snapshot support was added), replays the moves
to regenerate intermediate states, and stores snapshots at configurable intervals.

This enables NNUE training on historical game data.

Usage:
    python scripts/backfill_snapshots.py --db data/games/selfplay.db
    python scripts/backfill_snapshots.py --db data/games/selfplay.db --interval 10 --limit 100
    python scripts/backfill_snapshots.py --db data/games/selfplay.db --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.game_engine import GameEngine
from app.models import GameState, Move
from app.rules.serialization import deserialize_game_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_INTERVAL = 20


def get_games_needing_snapshots(
    conn: sqlite3.Connection,
    snapshot_interval: int,
    limit: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """Find games that need snapshot backfill.

    Returns list of (game_id, move_count) tuples for games that:
    1. Have enough moves to warrant snapshots (> snapshot_interval)
    2. Don't have snapshots at the expected intervals
    """
    cursor = conn.cursor()

    # Get games with move counts
    query = """
        SELECT g.game_id, COUNT(m.game_id) as move_count
        FROM games g
        LEFT JOIN game_moves m ON g.game_id = m.game_id
        GROUP BY g.game_id
        HAVING move_count >= ?
    """
    if limit:
        query += f" LIMIT {limit * 2}"  # Get extra to filter out those with snapshots

    cursor.execute(query, (snapshot_interval,))
    candidates = cursor.fetchall()

    # Filter out games that already have sufficient snapshots
    result = []
    for game_id, move_count in candidates:
        expected_snapshots = move_count // snapshot_interval
        if expected_snapshots == 0:
            continue

        # Count existing snapshots
        cursor.execute(
            "SELECT COUNT(*) FROM game_state_snapshots WHERE game_id = ?",
            (game_id,)
        )
        existing_count = cursor.fetchone()[0]

        # If we have fewer snapshots than expected, add to backfill list
        if existing_count < expected_snapshots:
            result.append((game_id, move_count))
            if limit and len(result) >= limit:
                break

    return result


def get_game_data(
    conn: sqlite3.Connection,
    game_id: str,
) -> Tuple[Optional[GameState], List[Move]]:
    """Load initial state and moves for a game."""
    cursor = conn.cursor()

    # Get initial state - try both schema versions
    # v5+: separate game_initial_state table
    cursor.execute(
        "SELECT initial_state_json FROM game_initial_state WHERE game_id = ?",
        (game_id,)
    )
    row = cursor.fetchone()

    if not row:
        # v4-: initial_state_json in games table
        cursor.execute(
            "SELECT initial_state_json FROM games WHERE game_id = ?",
            (game_id,)
        )
        row = cursor.fetchone()

    if not row or not row[0]:
        return None, []

    try:
        import json
        state_data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        initial_state = deserialize_game_state(state_data)
    except Exception as e:
        logger.warning(f"Failed to deserialize initial state for {game_id}: {e}")
        return None, []

    # Get moves in order
    cursor.execute(
        """SELECT move_json FROM game_moves
           WHERE game_id = ?
           ORDER BY turn_number ASC""",
        (game_id,)
    )
    moves = []
    for (move_json,) in cursor.fetchall():
        try:
            move = Move.model_validate_json(move_json)
            moves.append(move)
        except Exception as e:
            logger.warning(f"Failed to parse move in {game_id}: {e}")
            continue

    return initial_state, moves


def store_snapshot(
    conn: sqlite3.Connection,
    game_id: str,
    move_number: int,
    state: GameState,
) -> bool:
    """Store a snapshot for the given game and move number."""
    try:
        state_json = state.model_dump_json()
        conn.execute(
            """INSERT OR REPLACE INTO game_state_snapshots
               (game_id, move_number, state_json, compressed)
               VALUES (?, ?, ?, 0)""",
            (game_id, move_number, state_json)
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to store snapshot for {game_id} at move {move_number}: {e}")
        return False


def backfill_game(
    conn: sqlite3.Connection,
    game_id: str,
    snapshot_interval: int,
    dry_run: bool = False,
) -> int:
    """Backfill snapshots for a single game.

    Returns the number of snapshots created.
    """
    initial_state, moves = get_game_data(conn, game_id)
    if initial_state is None or not moves:
        logger.warning(f"Skipping {game_id}: no initial state or moves")
        return 0

    # Get existing snapshot move numbers to avoid duplicates
    cursor = conn.cursor()
    cursor.execute(
        "SELECT move_number FROM game_state_snapshots WHERE game_id = ?",
        (game_id,)
    )
    existing_snapshots = {row[0] for row in cursor.fetchall()}

    state = initial_state
    snapshots_created = 0

    for i, move in enumerate(moves):
        try:
            state = GameEngine.apply_move(state, move, trace_mode=True)
        except Exception as e:
            logger.warning(f"Failed to apply move {i} in {game_id}: {e}")
            break

        move_number = i + 1  # 1-indexed for interval check

        # Store snapshot at interval if not already present
        if move_number % snapshot_interval == 0 and i not in existing_snapshots:
            if dry_run:
                logger.info(f"  Would create snapshot at move {i}")
                snapshots_created += 1
            else:
                if store_snapshot(conn, game_id, i, state):
                    snapshots_created += 1

    return snapshots_created


def backfill_database(
    db_path: str,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Backfill snapshots for all games in the database.

    Returns (games_processed, snapshots_created).
    """
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return 0, 0

    conn = sqlite3.connect(db_path)

    # Ensure snapshot table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_state_snapshots (
            game_id TEXT NOT NULL,
            move_number INTEGER NOT NULL,
            state_json TEXT NOT NULL,
            compressed INTEGER DEFAULT 0,
            state_hash TEXT,
            PRIMARY KEY (game_id, move_number),
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        )
    """)
    conn.commit()

    # Find games needing backfill
    games = get_games_needing_snapshots(conn, snapshot_interval, limit)
    logger.info(f"Found {len(games)} games needing snapshot backfill")

    if not games:
        return 0, 0

    games_processed = 0
    total_snapshots = 0

    for game_id, move_count in games:
        logger.info(f"Processing {game_id} ({move_count} moves)...")

        snapshots = backfill_game(conn, game_id, snapshot_interval, dry_run)
        if snapshots > 0:
            games_processed += 1
            total_snapshots += snapshots

            if not dry_run:
                conn.commit()

            logger.info(f"  Created {snapshots} snapshots")

    conn.close()

    action = "Would create" if dry_run else "Created"
    logger.info(f"{action} {total_snapshots} snapshots across {games_processed} games")

    return games_processed, total_snapshots


def main():
    parser = argparse.ArgumentParser(
        description="Backfill periodic snapshots for existing games"
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to the game database",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_SNAPSHOT_INTERVAL,
        help=f"Snapshot interval in moves (default: {DEFAULT_SNAPSHOT_INTERVAL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of games to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    logger.info(f"Backfilling snapshots in {args.db}")
    logger.info(f"Snapshot interval: {args.interval}")
    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")

    games, snapshots = backfill_database(
        args.db,
        snapshot_interval=args.interval,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"  Games processed: {games}")
    logger.info(f"  Snapshots created: {snapshots}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
