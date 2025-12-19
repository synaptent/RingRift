#!/usr/bin/env python3
"""Fast database consolidation using SQL ATTACH and INSERT OR IGNORE."""

import sqlite3
import sys
from pathlib import Path
import time

def get_schema(conn):
    """Get column names from games table."""
    cursor = conn.execute("PRAGMA table_info(games)")
    return [row[1] for row in cursor.fetchall()]

def main():
    synced_dir = Path("data/games/synced")
    output_path = Path("data/games/fast_consolidated.db")
    
    # Remove existing output
    if output_path.exists():
        output_path.unlink()
    
    # Get all source databases
    dbs = list(synced_dir.rglob("*.db"))
    print(f"Found {len(dbs)} databases to consolidate")
    
    # Create output database with minimal schema
    conn_out = sqlite3.connect(str(output_path))
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
    conn_out.execute("PRAGMA journal_mode=WAL")
    conn_out.execute("PRAGMA synchronous=NORMAL")
    conn_out.commit()
    
    total_inserted = 0
    start_time = time.time()
    
    for i, db_path in enumerate(dbs):
        try:
            # Attach source database
            conn_out.execute(f"ATTACH DATABASE '{db_path}' AS src")
            
            # Check if games table exists in source
            cursor = conn_out.execute("SELECT name FROM src.sqlite_master WHERE type='table' AND name='games'")
            if not cursor.fetchone():
                conn_out.execute("DETACH DATABASE src")
                continue
            
            # Get source columns
            cursor = conn_out.execute("PRAGMA src.table_info(games)")
            src_cols = [row[1] for row in cursor.fetchall()]
            
            # Use only columns that exist in both
            target_cols = ["game_id", "board_type", "num_players", "rng_seed", "created_at", "source", "canonical_history", "metadata_json"]
            common_cols = [c for c in target_cols if c in src_cols]
            
            if "game_id" not in common_cols:
                conn_out.execute("DETACH DATABASE src")
                continue
            
            col_list = ", ".join(common_cols)
            
            # Insert with deduplication
            cursor = conn_out.execute(f"""
                INSERT OR IGNORE INTO games ({col_list})
                SELECT {col_list} FROM src.games
            """)
            inserted = cursor.rowcount
            total_inserted += max(0, inserted)
            
            conn_out.commit()
            conn_out.execute("DETACH DATABASE src")
            
            if (i + 1) % 10 == 0 or i == len(dbs) - 1:
                elapsed = time.time() - start_time
                cursor = conn_out.execute("SELECT COUNT(*) FROM games")
                total_games = cursor.fetchone()[0]
                print(f"[{i+1}/{len(dbs)}] {total_games} unique games, {elapsed:.1f}s elapsed")
                
        except Exception as e:
            print(f"Error processing {db_path}: {e}")
            try:
                conn_out.execute("DETACH DATABASE src")
            except sqlite3.Error:
                pass
    
    # Final stats
    cursor = conn_out.execute("SELECT COUNT(*) FROM games")
    final_count = cursor.fetchone()[0]
    
    cursor = conn_out.execute("""
        SELECT board_type, num_players, COUNT(*) as count 
        FROM games GROUP BY board_type, num_players ORDER BY count DESC
    """)
    
    print(f"\n=== Consolidation Complete ===")
    print(f"Total unique games: {final_count}")
    print(f"Time: {time.time() - start_time:.1f}s")
    print("\nBreakdown by config:")
    for row in cursor.fetchall():
        print(f"  {row[0]}_{row[1]}p: {row[2]} games")
    
    conn_out.close()

if __name__ == "__main__":
    main()
