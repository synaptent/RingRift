#!/usr/bin/env python3
"""
Aggregate games from JSONL files to SQLite database for STATISTICS ONLY.

IMPORTANT: This script creates a simple schema optimized for monitoring and stats.
It is NOT compatible with the training pipeline, which requires the full
GameReplayDB schema with game_moves table.

For training data:
- Use run_self_play_soak.py --record-db to generate canonical games
- Or use data from data/selfplay/diverse/*.db or data/canonical/*.db

Designed for automated cron execution.
"""
import os
import json
import sqlite3
import glob
import sys
from datetime import datetime

DATA_DIR = os.environ.get("DATA_DIR", "data/games")
# Use _stats suffix to clearly indicate this is for monitoring, not training
DB_PATH = os.path.join(DATA_DIR, "selfplay_stats.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS games (
        game_id TEXT PRIMARY KEY,
        board_type TEXT,
        num_players INTEGER,
        moves TEXT,
        winner INTEGER,
        final_scores TEXT,
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source_file TEXT
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_board_players ON games(board_type, num_players)")
    conn.commit()
    return conn

def aggregate():
    conn = init_db()
    total_imported = 0

    # Find all JSONL files recursively
    jsonl_files = glob.glob(os.path.join(DATA_DIR, "*.jsonl"))
    jsonl_files += glob.glob(os.path.join(DATA_DIR, "**", "*.jsonl"), recursive=True)
    jsonl_files = list(set(jsonl_files))  # Deduplicate

    for jsonl_file in jsonl_files:
        imported = 0
        with open(jsonl_file) as f:
            for line in f:
                try:
                    g = json.loads(line.strip())
                    conn.execute(
                        "INSERT OR IGNORE INTO games (game_id, board_type, num_players, moves, winner, final_scores, metadata, source_file) VALUES (?,?,?,?,?,?,?,?)",
                        (g.get("game_id"), g.get("board_type"), g.get("num_players"),
                         json.dumps(g.get("moves", [])), g.get("winner"),
                         json.dumps(g.get("final_scores", {})), json.dumps(g),
                         os.path.basename(jsonl_file))
                    )
                    imported += 1
                except (json.JSONDecodeError, sqlite3.Error, KeyError):
                    pass  # Skip malformed lines or duplicate entries
        conn.commit()
        if imported > 0:
            print(f"  {os.path.basename(jsonl_file)}: +{imported}")
            total_imported += imported
    
    # Print stats
    print(f"\nTotal imported: {total_imported}")
    print("\nDatabase totals:")
    for row in conn.execute("SELECT board_type, num_players, COUNT(*) FROM games GROUP BY 1,2 ORDER BY 3 DESC"):
        print(f"  {row[0]} {row[1]}p: {row[2]}")
    total = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    print(f"TOTAL: {total}")
    
    conn.close()
    return total

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Aggregating games to {DB_PATH}")
    aggregate()
