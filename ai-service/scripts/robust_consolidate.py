#!/usr/bin/env python3
"""Robust database consolidation with WAL mode and proper locking."""

import sqlite3
import sys
from pathlib import Path
import time
import fcntl

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
        conn_out.commit()
        
        total_inserted = 0
        errors = 0
        start_time = time.time()
        
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
                
                # Get source columns
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
                
                # Read in batches and insert
                cursor = conn_src.execute(f"SELECT {col_list} FROM games")
                batch = []
                batch_size = 1000
                
                for row in cursor:
                    batch.append(tuple(row))
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
                
                conn_out.commit()
                conn_src.close()
                
                if (i + 1) % 20 == 0:
                    cursor = conn_out.execute("SELECT COUNT(*) FROM games")
                    total_games = cursor.fetchone()[0]
                    elapsed = time.time() - start_time
                    print(f"[{i+1}/{len(dbs)}] {total_games} unique games, {elapsed:.1f}s elapsed")
                    
            except Exception as e:
                errors += 1
                print(f"Error processing {db_path.name}: {str(e)[:50]}")
        
        # Checkpoint WAL to main database
        conn_out.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        
        # Final stats
        cursor = conn_out.execute("SELECT COUNT(*) FROM games")
        final_count = cursor.fetchone()[0]
        
        cursor = conn_out.execute("""
            SELECT board_type, num_players, COUNT(*) as count 
            FROM games GROUP BY board_type, num_players ORDER BY count DESC
        """)
        
        print(f"\n=== Consolidation Complete ===")
        print(f"Total unique games: {final_count}")
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
