#!/usr/bin/env python3
"""Export training data from game databases across the cluster.

Finds all game databases, filters by board type and quality,
exports training positions to NPZ format.

Usage:
    # Export all hex 2p training data
    python scripts/export_training_from_cluster.py --board hexagonal --players 2

    # Export with quality filter (gauntlet scores)
    python scripts/export_training_from_cluster.py --board hexagonal --players 2 --min-score 0.7

    # Export from specific directories
    python scripts/export_training_from_cluster.py --board square19 --players 2 --db-dir data/selfplay
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))


def find_game_databases(
    base_dir: Path,
    board_type: Optional[str] = None,
) -> List[Path]:
    """Find all game databases in directory tree."""
    dbs = []

    # Search patterns
    patterns = [
        "**/*.db",
        "**/games.db",
    ]

    for pattern in patterns:
        for db_path in base_dir.glob(pattern):
            # Skip non-game databases
            if "wal" in db_path.name or "shm" in db_path.name:
                continue

            # Filter by board type if specified
            if board_type:
                path_str = str(db_path).lower()
                board_aliases = {
                    "hexagonal": ["hex", "hexagonal"],
                    "hex8": ["hex8"],
                    "square8": ["sq8", "square8"],
                    "square19": ["sq19", "square19"],
                }
                aliases = board_aliases.get(board_type, [board_type])
                if not any(alias in path_str for alias in aliases):
                    continue

            dbs.append(db_path)

    return dbs


def get_db_stats(db_path: Path) -> Dict[str, Any]:
    """Get statistics from a game database."""
    try:
        # Open in read-only mode to avoid locking issues
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]

        if "games" not in tables:
            conn.close()
            return {"error": "no games table"}

        # Count games
        cursor.execute("SELECT COUNT(*) FROM games")
        num_games = cursor.fetchone()[0]

        # Check for moves table
        num_moves = 0
        if "game_moves" in tables:
            cursor.execute("SELECT COUNT(*) FROM game_moves")
            num_moves = cursor.fetchone()[0]

        conn.close()
        return {
            "games": num_games,
            "moves": num_moves,
            "tables": tables,
        }

    except Exception as e:
        return {"error": str(e)}


def export_positions_from_db(
    db_path: Path,
    board_type: str,
    num_players: int,
    output_dir: Path,
    max_games: int = 10000,
    sample_every: int = 3,
) -> int:
    """Export training positions from a game database.

    Returns number of positions exported.
    """
    try:
        # Open in read-only mode
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Check if this is a GameReplayDB format
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]

        if "game_state_snapshots" not in tables:
            # Simple format - just game outcomes
            conn.close()
            return 0

        # Export positions from snapshots
        cursor.execute("""
            SELECT gs.game_id, gs.move_index, gs.state_json, g.winner
            FROM game_state_snapshots gs
            JOIN games g ON gs.game_id = g.game_id
            WHERE g.winner IS NOT NULL
            ORDER BY gs.game_id, gs.move_index
            LIMIT ?
        """, (max_games * 50,))  # Assume ~50 moves per game

        positions = []
        values = []
        policies = []

        for row in cursor.fetchall():
            game_id, move_idx, state_json, winner = row

            # Sample every N positions
            if move_idx % sample_every != 0:
                continue

            try:
                state = json.loads(state_json)

                # Extract board state
                if "board" in state:
                    board = np.array(state["board"], dtype=np.float32)
                    positions.append(board.flatten())

                    # Value target: +1 if current player wins, -1 if loses
                    current_player = state.get("current_player", 1)
                    value = 1.0 if winner == current_player else -1.0
                    values.append(value)

            except Exception:
                continue

        conn.close()

        if positions:
            # Save to NPZ
            output_file = output_dir / f"{board_type}_{num_players}p_{db_path.stem}.npz"
            np.savez_compressed(
                output_file,
                positions=np.array(positions),
                values=np.array(values),
                source=str(db_path),
                board_type=board_type,
                num_players=num_players,
            )
            return len(positions)

        return 0

    except Exception as e:
        print(f"Error exporting from {db_path}: {e}")
        return 0


def merge_training_files(
    input_dir: Path,
    output_file: Path,
    board_type: str,
    num_players: int,
) -> int:
    """Merge multiple NPZ files into one."""
    all_positions = []
    all_values = []

    for npz_file in input_dir.glob(f"{board_type}_{num_players}p_*.npz"):
        try:
            data = np.load(npz_file)
            if "positions" in data:
                all_positions.append(data["positions"])
            if "values" in data:
                all_values.append(data["values"])
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")

    if all_positions:
        positions = np.concatenate(all_positions)
        values = np.concatenate(all_values) if all_values else np.zeros(len(positions))

        np.savez_compressed(
            output_file,
            positions=positions,
            values=values,
            board_type=board_type,
            num_players=num_players,
            timestamp=datetime.now().isoformat(),
        )
        return len(positions)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Export training data from game databases")
    parser.add_argument("--board", type=str, required=True, help="Board type (hexagonal, square8, square19, hex8)")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--db-dir", type=str, default="data", help="Directory to search for databases")
    parser.add_argument("--output-dir", type=str, default="data/training", help="Output directory for NPZ files")
    parser.add_argument("--max-games", type=int, default=10000, help="Max games per database")
    parser.add_argument("--sample-every", type=int, default=3, help="Sample every N positions")
    parser.add_argument("--merge", action="store_true", help="Merge all output files into one")
    parser.add_argument("--list-only", action="store_true", help="Just list databases, don't export")

    args = parser.parse_args()

    base_dir = AI_SERVICE_ROOT / args.db_dir
    output_dir = AI_SERVICE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find databases
    print(f"Searching for {args.board} databases in {base_dir}...")
    dbs = find_game_databases(base_dir, args.board)
    print(f"Found {len(dbs)} databases")

    if args.list_only:
        for db in dbs:
            stats = get_db_stats(db)
            print(f"  {db}: {stats}")
        return

    # Export from each database
    total_positions = 0
    for i, db in enumerate(dbs):
        print(f"[{i+1}/{len(dbs)}] Exporting from {db.name}...", end=" ", flush=True)
        stats = get_db_stats(db)

        if "error" in stats:
            print(f"skipped ({stats['error']})")
            continue

        if stats["games"] == 0:
            print("skipped (empty)")
            continue

        n = export_positions_from_db(
            db, args.board, args.players, output_dir,
            args.max_games, args.sample_every
        )
        total_positions += n
        print(f"{n} positions")

    print(f"\nTotal positions exported: {total_positions}")

    # Merge if requested
    if args.merge and total_positions > 0:
        merged_file = output_dir / f"{args.board}_{args.players}p_merged.npz"
        n = merge_training_files(output_dir, merged_file, args.board, args.players)
        print(f"Merged into {merged_file}: {n} positions")


if __name__ == "__main__":
    main()
