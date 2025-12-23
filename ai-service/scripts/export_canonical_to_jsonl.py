#!/usr/bin/env python3
"""Export games from canonical DB to JSONL format for GMO training.

This script extracts games with initial_state and moves from the canonical
SQLite database and outputs them in the JSONL format expected by GMO trainers.

Usage:
    python scripts/export_canonical_to_jsonl.py --board-type square8 --num-players 2 --output data/training/gmo_full_sq8_2p.jsonl
"""

import argparse
import json
import sqlite3
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def decompress_if_needed(data: str | bytes, compressed: int) -> str:
    """Decompress data if it was stored compressed."""
    if compressed and isinstance(data, bytes):
        return zlib.decompress(data).decode('utf-8')
    if isinstance(data, bytes):
        return data.decode('utf-8')
    return data


def export_games(
    db_path: Path,
    output_path: Path,
    board_type: str = "square8",
    num_players: int = 2,
    min_moves: int = 10,
    include_draws: bool = False,
) -> int:
    """Export games to JSONL format.

    Returns number of games exported.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Query for games with winner
    where_clause = "g.game_status = 'completed'"
    if not include_draws:
        where_clause += " AND g.winner IS NOT NULL"
    if board_type:
        where_clause += f" AND g.board_type = '{board_type}'"
    if num_players:
        where_clause += f" AND g.num_players = {num_players}"
    if min_moves:
        where_clause += f" AND g.total_moves >= {min_moves}"

    # Get game IDs
    game_query = f"""
        SELECT g.game_id, g.board_type, g.num_players, g.winner, g.total_moves,
               i.initial_state_json, i.compressed
        FROM games g
        JOIN game_initial_state i ON g.game_id = i.game_id
        WHERE {where_clause}
        ORDER BY g.game_id
    """

    games = conn.execute(game_query).fetchall()
    print(f"Found {len(games)} games matching criteria")

    exported = 0
    with open(output_path, 'w') as f:
        for game in games:
            game_id = game['game_id']

            # Get initial state
            try:
                initial_state_json = decompress_if_needed(
                    game['initial_state_json'],
                    game['compressed']
                )
                initial_state = json.loads(initial_state_json)
            except Exception as e:
                print(f"  Skipping {game_id}: failed to parse initial_state - {e}")
                continue

            # Get moves
            moves_query = """
                SELECT move_json
                FROM game_moves
                WHERE game_id = ?
                ORDER BY move_number
            """
            move_rows = conn.execute(moves_query, (game_id,)).fetchall()

            if not move_rows:
                print(f"  Skipping {game_id}: no moves found")
                continue

            moves = []
            for row in move_rows:
                try:
                    move = json.loads(row['move_json'])
                    moves.append(move)
                except json.JSONDecodeError:
                    continue

            if len(moves) < min_moves:
                continue

            # Build JSONL record
            record = {
                "game_id": game_id,
                "winner": game['winner'],
                "board_type": game['board_type'],
                "num_players": game['num_players'],
                "initial_state": initial_state,
                "moves": moves,
            }

            f.write(json.dumps(record) + '\n')
            exported += 1

            if exported % 100 == 0:
                print(f"  Exported {exported} games...")

    conn.close()
    return exported


def main():
    parser = argparse.ArgumentParser(description="Export canonical games to JSONL")
    parser.add_argument(
        "--db-path", type=Path,
        default=Path("data/training/canonical_games.db"),
        help="Path to canonical games database"
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        default=Path("data/training/gmo_full.jsonl"),
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--board-type", type=str, default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type to export"
    )
    parser.add_argument(
        "--num-players", type=int, default=2,
        choices=[2, 3, 4],
        help="Number of players"
    )
    parser.add_argument(
        "--min-moves", type=int, default=10,
        help="Minimum moves per game"
    )
    parser.add_argument(
        "--include-draws", action="store_true",
        help="Include games without a winner"
    )

    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: Database not found at {args.db_path}")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting games from {args.db_path}")
    print(f"  Board type: {args.board_type}")
    print(f"  Num players: {args.num_players}")
    print(f"  Min moves: {args.min_moves}")
    print(f"  Output: {args.output}")
    print()

    count = export_games(
        db_path=args.db_path,
        output_path=args.output,
        board_type=args.board_type,
        num_players=args.num_players,
        min_moves=args.min_moves,
        include_draws=args.include_draws,
    )

    print(f"\nExported {count} games to {args.output}")

    # Print stats
    file_size = args.output.stat().st_size
    print(f"File size: {file_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
