#!/usr/bin/env python3
"""Consolidate scattered selfplay games into canonical databases.

Usage:
    python scripts/consolidate_selfplay.py
"""
import sqlite3
from pathlib import Path
import sys

def consolidate_config(config: str, base_dir: Path = Path("data/selfplay/p2p_hybrid")) -> int:
    """Consolidate games for a single config.

    IMPORTANT: This function MERGES games into existing canonical DBs.
    It does NOT delete/replace existing data.
    """
    source_dir = base_dir / config
    dest_db = Path(f"data/games/canonical_{config}.db")

    if not source_dir.exists():
        print(f"No source directory for {config}")
        return 0

    # CRITICAL: Do NOT add dest_db.unlink() here!
    # See incident on Jan 2, 2026 where 235K+ games were lost.
    # This script MERGES games into existing canonical DBs.

    # Check if destination already exists and has data
    existing_count = 0
    if dest_db.exists():
        try:
            existing_conn = sqlite3.connect(str(dest_db))
            existing_count = existing_conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            existing_conn.close()
            print(f"{config}: Existing DB has {existing_count} games, will merge new games")
        except Exception:
            # DB exists but is corrupt/empty - safe to initialize fresh
            print(f"{config}: Existing DB is corrupt/empty, will initialize")
            dest_db.unlink(missing_ok=True)

    # Find first source DB to copy schema from
    source_dbs = list(source_dir.rglob("games.db"))
    if not source_dbs:
        print(f"No source databases for {config}")
        return 0

    # Only create schema if dest doesn't exist or was corrupt
    if not dest_db.exists():
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
        dest_conn.commit()
        dest_conn.close()

    # Open dest for merging
    dest_conn = sqlite3.connect(str(dest_db))

    # Find all source DBs and import
    total_imported = 0
    total_skipped_no_moves = 0
    min_moves_required = 3  # DATA QUALITY GATE: Require at least 3 moves per game

    for db_path in source_dbs:
        try:
            src_conn = sqlite3.connect(str(db_path))

            # DATA QUALITY GATE: Only import games that HAVE move data
            # Uses INNER JOIN to ensure moves exist before importing
            games = src_conn.execute("""
                SELECT DISTINCT g.*
                FROM games g
                INNER JOIN game_moves m ON g.game_id = m.game_id
                WHERE g.game_status IN ('complete', 'completed')
                GROUP BY g.game_id
                HAVING COUNT(m.game_id) >= ?
            """, (min_moves_required,)).fetchall()

            for game in games:
                game_id = game[0]  # First column is game_id
                try:
                    # Build insert with available columns
                    placeholders = ",".join(["?"] * len(game))
                    dest_conn.execute(f"INSERT OR IGNORE INTO games VALUES ({placeholders})", game)

                    # Copy moves for this game (guaranteed to exist by INNER JOIN above)
                    moves = src_conn.execute(
                        "SELECT * FROM game_moves WHERE game_id = ?", (game_id,)
                    ).fetchall()
                    for move in moves:
                        placeholders = ",".join(["?"] * len(move))
                        dest_conn.execute(f"INSERT OR IGNORE INTO game_moves VALUES ({placeholders})", move)

                    total_imported += 1
                except Exception:
                    pass

            # Track skipped games for reporting
            skipped = src_conn.execute("""
                SELECT COUNT(*) FROM games g
                WHERE g.game_status IN ('complete', 'completed')
                AND g.game_id NOT IN (
                    SELECT DISTINCT game_id FROM game_moves
                    GROUP BY game_id
                    HAVING COUNT(*) >= ?
                )
            """, (min_moves_required,)).fetchone()[0]
            total_skipped_no_moves += skipped

            src_conn.close()
        except Exception as e:
            print(f"Error with {db_path}: {e}")

    dest_conn.commit()
    dest_conn.close()

    # Count final
    final_count = sqlite3.connect(str(dest_db)).execute("SELECT COUNT(*) FROM games").fetchone()[0]
    new_games = final_count - existing_count
    skipped_msg = f", skipped {total_skipped_no_moves} without moves" if total_skipped_no_moves > 0 else ""
    print(f"{config}: Added {new_games} new games (tried {total_imported}{skipped_msg}), total now: {final_count}")
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
