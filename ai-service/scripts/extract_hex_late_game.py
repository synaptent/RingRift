#!/usr/bin/env python3
"""Extract late-game hex positions for targeted training and evaluation.

Creates eval pools and training samples from existing hex game databases,
focusing on positions from the last N moves of completed games.

Usage:
    # Extract positions from moves 100+ for eval pool
    python scripts/extract_hex_late_game.py --db data/games/hex8_*.db --min-moves 100 --output eval

    # Create NPZ training data from late-game positions
    python scripts/extract_hex_late_game.py --db data/games/hex8_*.db --min-moves 120 --output npz
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_late_game_positions(
    db_path: Path,
    min_move_number: int = 100,
    max_positions: int | None = None,
) -> list[dict[str, Any]]:
    """Extract late-game positions from a game database.

    Handles multiple database schemas:
    - hex8_consolidated.db: games.num_moves, moves table
    - hex8_test_mcts.db: games.total_moves, game_moves table
    - jsonl_converted_*.db: games.move_count, no moves table (game-level only)

    Args:
        db_path: Path to SQLite game database
        min_move_number: Minimum move number to include (default: 100)
        max_positions: Maximum positions to extract (None = all)

    Returns:
        List of position dicts with state, move info, and metadata
    """
    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return []

    positions = []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Detect schema by checking column names
        cursor.execute("PRAGMA table_info(games)")
        game_cols = {r[1] for r in cursor.fetchall()}

        # Detect move count column
        if "move_count" in game_cols:
            move_count_col = "move_count"
        elif "num_moves" in game_cols:
            move_count_col = "num_moves"
        elif "total_moves" in game_cols:
            move_count_col = "total_moves"
        else:
            logger.warning(f"{db_path.name}: No move count column found")
            conn.close()
            return []

        # Detect moves table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cursor.fetchall()}

        if "moves" in tables:
            moves_table = "moves"
            move_num_col = "move_number"
        elif "game_moves" in tables:
            moves_table = "game_moves"
            move_num_col = "move_number"  # game_moves uses move_number too
        else:
            moves_table = None
            logger.info(f"{db_path.name}: No moves table, extracting game-level data only")

        # Find completed games with sufficient moves
        cursor.execute(f"""
            SELECT game_id, board_type, num_players, winner, {move_count_col} as move_count
            FROM games
            WHERE winner IS NOT NULL AND {move_count_col} >= ?
        """, (min_move_number,))

        games = cursor.fetchall()
        logger.info(f"{db_path.name}: Found {len(games)} games with {min_move_number}+ moves")

        for game in games:
            game_id = game["game_id"]

            if moves_table:
                # Get moves from the late-game portion
                try:
                    cursor.execute(f"""
                        SELECT {move_num_col} as move_number, player, move_type,
                               COALESCE(from_pos, from_position) as from_pos,
                               COALESCE(to_pos, to_position) as to_pos,
                               COALESCE(state_before, '') as state_before
                        FROM {moves_table}
                        WHERE game_id = ? AND {move_num_col} >= ?
                        ORDER BY {move_num_col}
                    """, (game_id, min_move_number))
                except sqlite3.OperationalError:
                    # Try simpler query for different schema
                    cursor.execute(f"""
                        SELECT {move_num_col} as move_number,
                               COALESCE(player, 1) as player,
                               COALESCE(move_type, 'unknown') as move_type,
                               '' as from_pos, '' as to_pos, '' as state_before
                        FROM {moves_table}
                        WHERE game_id = ? AND {move_num_col} >= ?
                        ORDER BY {move_num_col}
                    """, (game_id, min_move_number))

                moves = cursor.fetchall()
                for move in moves:
                    position = {
                        "game_id": game_id,
                        "board_type": game["board_type"],
                        "num_players": game["num_players"],
                        "winner": game["winner"],
                        "total_moves": game["move_count"],
                        "move_number": move["move_number"],
                        "player": move["player"],
                        "move_type": move["move_type"],
                        "from_pos": move["from_pos"],
                        "to_pos": move["to_pos"],
                        "state_before": move["state_before"],
                    }
                    positions.append(position)

                    if max_positions and len(positions) >= max_positions:
                        break
            else:
                # No moves table - just add game-level entry
                position = {
                    "game_id": game_id,
                    "board_type": game["board_type"],
                    "num_players": game["num_players"],
                    "winner": game["winner"],
                    "total_moves": game["move_count"],
                    "move_number": game["move_count"],  # Use total moves as position
                    "player": game["winner"],
                    "move_type": "game_end",
                    "from_pos": "",
                    "to_pos": "",
                    "state_before": "",
                }
                positions.append(position)

            if max_positions and len(positions) >= max_positions:
                break

        conn.close()

    except Exception as e:
        logger.error(f"Error reading {db_path}: {e}")
        import traceback
        traceback.print_exc()

    return positions


def save_eval_pool(
    positions: list[dict[str, Any]],
    output_dir: Path,
    pool_name: str = "hex_late_game",
) -> Path:
    """Save positions as JSONL eval pool.

    Args:
        positions: List of position dicts
        output_dir: Directory to save pool
        pool_name: Name for the pool file

    Returns:
        Path to saved pool file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pool_name}.jsonl"

    with open(output_path, "w") as f:
        for pos in positions:
            f.write(json.dumps(pos) + "\n")

    logger.info(f"Saved {len(positions)} positions to {output_path}")
    return output_path


def save_training_npz(
    positions: list[dict[str, Any]],
    output_path: Path,
    board_type: str = "hex8",
) -> Path:
    """Save positions as NPZ training data.

    Note: Requires encoding the states, which needs the full encoder pipeline.
    This is a simplified version that saves raw position data.

    Args:
        positions: List of position dicts
        output_path: Path to output NPZ file
        board_type: Board type for encoding

    Returns:
        Path to saved NPZ file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract arrays
    game_ids = np.array([p["game_id"] for p in positions], dtype=object)
    move_numbers = np.array([p["move_number"] for p in positions], dtype=np.int32)
    total_moves = np.array([p["total_moves"] for p in positions], dtype=np.int32)
    players = np.array([p["player"] for p in positions], dtype=np.int32)
    winners = np.array([p["winner"] for p in positions], dtype=np.int32)

    np.savez(
        output_path,
        game_ids=game_ids,
        move_numbers=move_numbers,
        total_moves=total_moves,
        players=players,
        winners=winners,
        board_type=np.asarray(board_type),
        extraction_timestamp=np.asarray(datetime.now().isoformat()),
    )

    logger.info(f"Saved {len(positions)} positions to {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract late-game hex positions for training/eval"
    )
    parser.add_argument(
        "--db",
        type=str,
        nargs="+",
        required=True,
        help="Glob patterns for database files (e.g., 'data/games/hex8_*.db')",
    )
    parser.add_argument(
        "--min-moves",
        type=int,
        default=100,
        help="Minimum move number to include (default: 100)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Maximum positions to extract (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["eval", "npz", "both"],
        default="eval",
        help="Output format: eval pool (JSONL) or NPZ training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eval_pools/hex",
        help="Output directory for eval pools",
    )
    parser.add_argument(
        "--npz-path",
        type=str,
        default="data/training/hex8_late_game.npz",
        help="Output path for NPZ file",
    )

    args = parser.parse_args(argv)

    # Expand glob patterns
    db_paths = []
    for pattern in args.db:
        matches = list(Path(".").glob(pattern))
        if not matches:
            # Try as literal path
            path = Path(pattern)
            if path.exists():
                matches = [path]
        db_paths.extend(matches)

    if not db_paths:
        logger.error(f"No databases found matching: {args.db}")
        return 1

    logger.info(f"Processing {len(db_paths)} database(s)")

    # Extract positions
    all_positions = []
    for db_path in db_paths:
        positions = get_late_game_positions(
            db_path,
            min_move_number=args.min_moves,
            max_positions=args.max_positions,
        )
        all_positions.extend(positions)

    if not all_positions:
        logger.warning("No late-game positions found")
        return 0

    logger.info(f"Extracted {len(all_positions)} total positions")

    # Save outputs
    if args.output in ("eval", "both"):
        save_eval_pool(
            all_positions,
            output_dir=Path(args.output_dir),
            pool_name=f"hex_late_game_{args.min_moves}plus",
        )

    if args.output in ("npz", "both"):
        save_training_npz(
            all_positions,
            output_path=Path(args.npz_path),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
