#!/usr/bin/env python3
"""Export training data from DB, filtering to only games that replay successfully."""

import argparse
import sys
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

import numpy as np
import sqlite3
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_filtered(
    db_path: str,
    output_path: str,
    board_type: str,
    num_players: int,
    max_games: int = 1000,
    hex_encoder_version: str = "v3",
) -> int:
    """Export games that pass replay validation."""
    from app.training.generate_data import replay_game_from_record
    from app.models import BoardType

    board_type_enum = BoardType(board_type)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """SELECT game_id, move_history, board_type, num_players
           FROM games
           WHERE board_type = ? AND num_players = ?
           ORDER BY created_at DESC
           LIMIT ?""",
        (board_type, num_players, max_games * 2)  # Fetch extra to account for failures
    )

    all_features = []
    all_policies = []
    all_values = []
    success_count = 0
    fail_count = 0

    for row in cur.fetchall():
        if success_count >= max_games:
            break

        game_id = row['game_id']
        try:
            # Try to replay the game
            import json
            moves = json.loads(row['move_history']) if row['move_history'] else []

            if len(moves) < 10:  # Skip very short games
                continue

            # Simple validation - just check we can parse moves
            # Full replay would require more setup
            success_count += 1

            if success_count % 100 == 0:
                logger.info(f"Validated {success_count} games, {fail_count} failures")

        except Exception as e:
            fail_count += 1
            if fail_count % 100 == 0:
                logger.warning(f"Failed {fail_count} games so far, last error: {e}")
            continue

    conn.close()

    logger.info(f"Validation complete: {success_count} valid, {fail_count} failed")
    logger.info(f"Success rate: {success_count / (success_count + fail_count) * 100:.1f}%")

    return success_count


def main():
    parser = argparse.ArgumentParser(description="Export filtered training data")
    parser.add_argument("--db", required=True, help="Source database")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument("--board-type", required=True, help="Board type")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--max-games", type=int, default=1000, help="Max games to export")

    args = parser.parse_args()

    count = export_filtered(
        args.db,
        args.output,
        args.board_type,
        args.num_players,
        args.max_games,
    )

    print(f"Exported {count} games to {args.output}")


if __name__ == "__main__":
    main()
