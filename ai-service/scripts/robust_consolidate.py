#!/usr/bin/env python3
"""Robust database consolidation with WAL mode and proper locking.

Creates a fully-compatible GameReplayDB with games, game_moves, and game_choices tables.
"""

import sqlite3
import sys
from pathlib import Path
import time
import fcntl

SCHEMA_VERSION = 2  # Must match GameReplayDB expected version

def main():
    synced_dir = Path("data/games/synced")
    output_path = Path("data/games/consolidated_training_v2.db")
    lock_path = Path("/tmp/consolidate.lock")

    # Acquire exclusive lock
    lock_file = open(lock_path, 'w')
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another consolidation is running, exiting")
        return

    try:
        # Remove existing output
        if output_path.exists():
            output_path.unlink()

        # Get all source databases
        dbs = list(synced_dir.rglob("*.db"))
        print(f"Found {len(dbs)} databases to consolidate")

        # Create output database with WAL mode for crash safety
        conn_out = sqlite3.connect(str(output_path), timeout=60.0)
        conn_out.execute("PRAGMA journal_mode=WAL")
        conn_out.execute("PRAGMA synchronous=NORMAL")
        conn_out.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn_out.execute("PRAGMA foreign_keys=OFF")  # Speed up bulk inserts

        # Create full schema compatible with GameReplayDB
        conn_out.execute("""
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn_out.execute("INSERT OR REPLACE INTO schema_metadata (key, value) VALUES ('version', ?)", (str(SCHEMA_VERSION),))

        conn_out.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER,
                rng_seed INTEGER,
                created_at TEXT,
                source TEXT,
                canonical_history TEXT,
                metadata_json TEXT
            )
        """)
        conn_out.execute("CREATE INDEX IF NOT EXISTS idx_board_players ON games(board_type, num_players)")

        # game_moves table (v2 schema)
        conn_out.execute("""
            CREATE TABLE IF NOT EXISTS game_moves (
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                turn_number INTEGER NOT NULL,
                player INTEGER NOT NULL,
                phase TEXT NOT NULL,
                move_type TEXT NOT NULL,
                move_json TEXT NOT NULL,
                timestamp TEXT,
                think_time_ms INTEGER,
                time_remaining_ms INTEGER,
                engine_eval REAL,
                engine_eval_type TEXT,
                engine_depth INTEGER,
                engine_nodes INTEGER,
                engine_pv TEXT,
                engine_time_ms INTEGER,
                PRIMARY KEY (game_id, move_number),
                FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
            )
        """)
        conn_out.execute("CREATE INDEX IF NOT EXISTS idx_moves_game_turn ON game_moves(game_id, turn_number)")

        # game_choices table
        conn_out.execute("""
            CREATE TABLE IF NOT EXISTS game_choices (
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                choice_type TEXT NOT NULL,
                player INTEGER NOT NULL,
                options_json TEXT NOT NULL,
                selected_option_json TEXT NOT NULL,
                ai_reasoning TEXT,
                PRIMARY KEY (game_id, move_number, choice_type),
                FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
            )
        """)
        conn_out.commit()
        
        total_inserted = 0
        total_moves = 0
        errors = 0
        start_time = time.time()

        # Track which game_ids we've inserted to avoid duplicate moves/choices
        inserted_game_ids = set()

        for i, db_path in enumerate(dbs):
            try:
                # Skip known corrupted databases
                if 'gh200_b' in str(db_path) and 'selfplay.db' in str(db_path):
                    print(f"Skipping potentially corrupted: {db_path.name}")
                    continue

                # Use read-only connection to source
                src_uri = f"file:{db_path}?mode=ro"
                conn_src = sqlite3.connect(src_uri, uri=True, timeout=10.0)
                conn_src.row_factory = sqlite3.Row

                # Check if games table exists
                cursor = conn_src.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
                if not cursor.fetchone():
                    conn_src.close()
                    continue

                # Get source columns for games table
                cursor = conn_src.execute("PRAGMA table_info(games)")
                src_cols = [row[1] for row in cursor.fetchall()]

                # Use only columns that exist in both
                target_cols = ["game_id", "board_type", "num_players", "rng_seed", "created_at", "source", "canonical_history", "metadata_json"]
                common_cols = [c for c in target_cols if c in src_cols]

                if "game_id" not in common_cols:
                    conn_src.close()
                    continue

                col_list = ", ".join(common_cols)
                placeholders = ", ".join("?" * len(common_cols))

                # Read games in batches and track new game_ids
                new_game_ids = []
                cursor = conn_src.execute(f"SELECT {col_list} FROM games")
                batch = []
                batch_size = 1000

                for row in cursor:
                    game_id = row[0]  # game_id is first column
                    if game_id not in inserted_game_ids:
                        batch.append(tuple(row))
                        new_game_ids.append(game_id)
                        inserted_game_ids.add(game_id)
                    if len(batch) >= batch_size:
                        conn_out.executemany(
                            f"INSERT OR IGNORE INTO games ({col_list}) VALUES ({placeholders})",
                            batch
                        )
                        batch = []

                if batch:
                    conn_out.executemany(
                        f"INSERT OR IGNORE INTO games ({col_list}) VALUES ({placeholders})",
                        batch
                    )

                # Copy game_moves for new games if table exists
                cursor = conn_src.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'")
                if cursor.fetchone() and new_game_ids:
                    cursor = conn_src.execute("PRAGMA table_info(game_moves)")
                    moves_cols = [row[1] for row in cursor.fetchall()]
                    target_moves_cols = ["game_id", "move_number", "turn_number", "player", "phase", "move_type", "move_json",
                                         "timestamp", "think_time_ms", "time_remaining_ms", "engine_eval", "engine_eval_type",
                                         "engine_depth", "engine_nodes", "engine_pv", "engine_time_ms"]
                    common_moves = [c for c in target_moves_cols if c in moves_cols]

                    if "game_id" in common_moves:
                        moves_col_list = ", ".join(common_moves)
                        moves_placeholders = ", ".join("?" * len(common_moves))

                        # Fetch moves for new games only
                        placeholders_list = ",".join("?" * len(new_game_ids))
                        cursor = conn_src.execute(f"SELECT {moves_col_list} FROM game_moves WHERE game_id IN ({placeholders_list})", new_game_ids)
                        moves_batch = []
                        for row in cursor:
                            moves_batch.append(tuple(row))
                            if len(moves_batch) >= batch_size:
                                conn_out.executemany(
                                    f"INSERT OR IGNORE INTO game_moves ({moves_col_list}) VALUES ({moves_placeholders})",
                                    moves_batch
                                )
                                total_moves += len(moves_batch)
                                moves_batch = []
                        if moves_batch:
                            conn_out.executemany(
                                f"INSERT OR IGNORE INTO game_moves ({moves_col_list}) VALUES ({moves_placeholders})",
                                moves_batch
                            )
                            total_moves += len(moves_batch)

                # Copy game_choices for new games if table exists
                cursor = conn_src.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_choices'")
                if cursor.fetchone() and new_game_ids:
                    cursor = conn_src.execute("PRAGMA table_info(game_choices)")
                    choices_cols = [row[1] for row in cursor.fetchall()]
                    target_choices_cols = ["game_id", "move_number", "choice_type", "player", "options_json", "selected_option_json", "ai_reasoning"]
                    common_choices = [c for c in target_choices_cols if c in choices_cols]

                    if "game_id" in common_choices:
                        choices_col_list = ", ".join(common_choices)
                        choices_placeholders = ", ".join("?" * len(common_choices))

                        placeholders_list = ",".join("?" * len(new_game_ids))
                        cursor = conn_src.execute(f"SELECT {choices_col_list} FROM game_choices WHERE game_id IN ({placeholders_list})", new_game_ids)
                        choices_batch = list(cursor)
                        if choices_batch:
                            conn_out.executemany(
                                f"INSERT OR IGNORE INTO game_choices ({choices_col_list}) VALUES ({choices_placeholders})",
                                [tuple(row) for row in choices_batch]
                            )

                conn_out.commit()
                conn_src.close()

                if (i + 1) % 20 == 0:
                    cursor = conn_out.execute("SELECT COUNT(*) FROM games")
                    total_games = cursor.fetchone()[0]
                    elapsed = time.time() - start_time
                    print(f"[{i+1}/{len(dbs)}] {total_games} unique games, {total_moves} moves, {elapsed:.1f}s elapsed")

            except Exception as e:
                errors += 1
                print(f"Error processing {db_path.name}: {str(e)[:50]}")
        
        # Checkpoint WAL to main database
        conn_out.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        # Final stats
        cursor = conn_out.execute("SELECT COUNT(*) FROM games")
        final_count = cursor.fetchone()[0]

        cursor = conn_out.execute("SELECT COUNT(*) FROM game_moves")
        final_moves = cursor.fetchone()[0]

        cursor = conn_out.execute("""
            SELECT board_type, num_players, COUNT(*) as count
            FROM games GROUP BY board_type, num_players ORDER BY count DESC
        """)

        print(f"\n=== Consolidation Complete ===")
        print(f"Total unique games: {final_count}")
        print(f"Total moves: {final_moves}")
        print(f"Errors: {errors}")
        print(f"Time: {time.time() - start_time:.1f}s")
        print("\nBreakdown by config:")
        for row in cursor.fetchall():
            print(f"  {row[0]}_{row[1]}p: {row[2]} games")
        
        conn_out.close()
        
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()

if __name__ == "__main__":
    main()
