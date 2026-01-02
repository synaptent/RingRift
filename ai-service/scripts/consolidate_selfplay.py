#!/usr/bin/env python3
"""Consolidate scattered selfplay games into canonical databases.

Usage:
    python scripts/consolidate_selfplay.py
"""
import sqlite3
from pathlib import Path
import sys

def consolidate_config(config: str, base_dir: Path = Path("data/selfplay/p2p_hybrid")) -> int:
    """Consolidate games for a single config."""
    source_dir = base_dir / config
    dest_db = Path(f"data/games/canonical_{config}.db")

    if not source_dir.exists():
        print(f"No source directory for {config}")
        return 0

    # Remove existing dest to ensure clean schema
    dest_db.unlink(missing_ok=True)

    # Find first source DB to copy schema from
    source_dbs = list(source_dir.rglob("games.db"))
    if not source_dbs:
        print(f"No source databases for {config}")
        return 0

    # Copy schema from first source
    first_src = sqlite3.connect(str(source_dbs[0]))
    dest_conn = sqlite3.connect(str(dest_db))

    # Copy games table schema
    games_schema = first_src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='games'").fetchone()
    if games_schema:
        dest_conn.execute(games_schema[0])

    # Copy game_moves table schema
    moves_schema = first_src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='game_moves'").fetchone()
    if moves_schema:
        dest_conn.execute(moves_schema[0])

    first_src.close()

    # Find all source DBs and import
    total_imported = 0
    for db_path in source_dbs:
        try:
            src_conn = sqlite3.connect(str(db_path))

            # Import completed games with all columns
            games = src_conn.execute(
                "SELECT * FROM games WHERE game_status IN ('complete', 'completed')"
            ).fetchall()

            for game in games:
                game_id = game[0]  # First column is game_id
                try:
                    # Build insert with available columns
                    placeholders = ",".join(["?"] * len(game))
                    dest_conn.execute(f"INSERT OR IGNORE INTO games VALUES ({placeholders})", game)

                    # Copy moves for this game
                    moves = src_conn.execute(
                        "SELECT * FROM game_moves WHERE game_id = ?", (game_id,)
                    ).fetchall()
                    for move in moves:
                        placeholders = ",".join(["?"] * len(move))
                        dest_conn.execute(f"INSERT OR IGNORE INTO game_moves VALUES ({placeholders})", move)

                    total_imported += 1
                except Exception:
                    pass
            src_conn.close()
        except Exception as e:
            print(f"Error with {db_path}: {e}")

    dest_conn.commit()
    dest_conn.close()

    # Count final
    final_count = sqlite3.connect(str(dest_db)).execute("SELECT COUNT(*) FROM games").fetchone()[0]
    print(f"{config}: Imported {total_imported} games, total now: {final_count}")
    return final_count


def main():
    configs = [
        "hex8_2p", "hex8_3p", "hex8_4p",
        "square8_2p", "square8_3p", "square8_4p",
        "square19_2p", "square19_3p", "square19_4p",
        "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
    ]

    total = 0
    for config in configs:
        count = consolidate_config(config)
        total += count

    print(f"\nTotal games consolidated: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
