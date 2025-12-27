#!/usr/bin/env python3
"""Consolidate Lambda GH200 backup data into training-ready databases.

This script extracts and consolidates game data from the Lambda cluster backup
(cluster_collected_backup/) which contains 327,000+ high-quality selfplay games.

Usage:
    # Full consolidation (all games)
    python scripts/consolidate_lambda_backup.py

    # Specific board type
    python scripts/consolidate_lambda_backup.py --board-type square8 --num-players 2

    # Export to NPZ after consolidation
    python scripts/consolidate_lambda_backup.py --export-npz

December 2025: Created to utilize untapped Lambda training data.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Lambda backup location
LAMBDA_BACKUP_ROOT = Path("/Volumes/RingRift-Data/cluster_collected_backup")

# Lambda nodes with significant data
LAMBDA_NODES = [
    "lambda-gh200-a",
    "lambda-gh200-c",
    "lambda-gh200-d",
    "lambda-gh200-e",
    "lambda-gh200-f",
    "lambda-gh200-g",
    "lambda-gh200-h",
    "lambda-gh200-i",
    "lambda-gh200-k",
    "lambda-gh200-l",
    "lambda-h100",
    "lambda-a10",
]

# Output directory
OUTPUT_DIR = Path("/Volumes/RingRift-Data/consolidated_training")


def find_databases(backup_root: Path) -> list[Path]:
    """Find all game databases in Lambda backup."""
    databases = []
    for node in LAMBDA_NODES:
        node_dir = backup_root / node
        if not node_dir.exists():
            continue

        # Find all .db files
        for db_path in node_dir.rglob("*.db"):
            # Skip WAL and SHM files
            if db_path.suffix in (".db-wal", ".db-shm"):
                continue
            # Check if it has a games table
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='games'"
                )
                if cursor.fetchone():
                    databases.append(db_path)
                conn.close()
            except Exception:
                pass

    return databases


def get_game_stats(db_path: Path) -> dict[str, Any]:
    """Get statistics from a game database."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.execute("""
            SELECT board_type, num_players, COUNT(*) as count
            FROM games
            GROUP BY board_type, num_players
        """)
        stats = {}
        for row in cursor:
            key = f"{row[0]}_{row[1]}p"
            stats[key] = row[2]
        conn.close()
        return stats
    except Exception as e:
        return {"error": str(e)}


def detect_schema_format(db_path: Path) -> str:
    """Detect whether database uses old (moves column) or new (game_moves table) schema."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        # Check if games table has moves column
        cursor = conn.execute("PRAGMA table_info(games)")
        columns = [row[1] for row in cursor]

        if "moves" in columns:
            conn.close()
            return "old"

        # Check if game_moves table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'"
        )
        if cursor.fetchone():
            conn.close()
            return "new"

        conn.close()
        return "unknown"
    except Exception:
        return "unknown"


def extract_game_from_new_schema(source_conn: sqlite3.Connection, game_id: str) -> tuple | None:
    """Extract game data from new schema (with separate game_moves table)."""
    import json

    try:
        # Get game metadata
        cursor = source_conn.execute("""
            SELECT board_type, num_players, winner
            FROM games
            WHERE game_id = ?
        """, (game_id,))
        row = cursor.fetchone()
        if not row:
            return None

        board_type, num_players, winner = row

        # Get all moves for this game
        cursor = source_conn.execute("""
            SELECT move_number, turn_number, player, phase, move_type, move_json
            FROM game_moves
            WHERE game_id = ?
            ORDER BY move_number
        """, (game_id,))

        moves = []
        for move_row in cursor:
            move_number, turn_number, player, phase, move_type, move_json = move_row
            move_data = json.loads(move_json)
            moves.append({
                "moveNumber": move_number,
                "turnNumber": turn_number,
                "player": player,
                "phase": phase,
                "moveType": move_type,
                **move_data
            })

        # Construct moves JSON
        moves_json = json.dumps(moves)

        # Get scores (calculate from final state if not stored)
        # For now, we'll just use empty scores
        scores_json = json.dumps([])

        # Construct metadata
        metadata_json = json.dumps({"source": "lambda_backup"})

        return (game_id, board_type, num_players, moves_json, winner, scores_json, metadata_json)

    except Exception as e:
        logger.warning(f"Error extracting game {game_id}: {e}")
        return None


def consolidate_to_config(
    databases: list[Path],
    board_type: str,
    num_players: int,
    output_path: Path,
) -> int:
    """Consolidate games for a specific config into one database."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create output database
    conn = sqlite3.connect(str(output_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT,
            num_players INTEGER,
            moves TEXT,
            winner INTEGER,
            scores TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_board_players ON games(board_type, num_players)")

    total_games = 0
    duplicates = 0

    for db_path in databases:
        try:
            # Detect schema format
            schema_format = detect_schema_format(db_path)

            if schema_format == "unknown":
                logger.warning(f"Unknown schema format in {db_path}, skipping")
                continue

            # Attach source database
            source_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

            if schema_format == "old":
                # Old schema: direct SELECT
                cursor = source_conn.execute("""
                    SELECT game_id, board_type, num_players, moves, winner, scores, metadata
                    FROM games
                    WHERE board_type = ? AND num_players = ?
                """, (board_type, num_players))

                batch = []
                for row in cursor:
                    batch.append(row)
                    if len(batch) >= 1000:
                        # Insert batch
                        for game in batch:
                            try:
                                conn.execute("""
                                    INSERT INTO games (game_id, board_type, num_players, moves, winner, scores, metadata)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, game)
                                total_games += 1
                            except sqlite3.IntegrityError:
                                duplicates += 1
                        batch = []
                        conn.commit()

                # Insert remaining
                for game in batch:
                    try:
                        conn.execute("""
                            INSERT INTO games (game_id, board_type, num_players, moves, winner, scores, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, game)
                        total_games += 1
                    except sqlite3.IntegrityError:
                        duplicates += 1
                conn.commit()

            else:  # new schema
                # Get list of game_ids for this config
                cursor = source_conn.execute("""
                    SELECT game_id
                    FROM games
                    WHERE board_type = ? AND num_players = ?
                """, (board_type, num_players))

                game_ids = [row[0] for row in cursor]
                logger.info(f"  Extracting {len(game_ids)} games from {db_path.name}...")

                batch = []
                for game_id in game_ids:
                    game_data = extract_game_from_new_schema(source_conn, game_id)
                    if game_data:
                        batch.append(game_data)

                    if len(batch) >= 100:
                        # Insert batch
                        for game in batch:
                            try:
                                conn.execute("""
                                    INSERT INTO games (game_id, board_type, num_players, moves, winner, scores, metadata)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, game)
                                total_games += 1
                            except sqlite3.IntegrityError:
                                duplicates += 1
                        batch = []
                        conn.commit()

                # Insert remaining
                for game in batch:
                    try:
                        conn.execute("""
                            INSERT INTO games (game_id, board_type, num_players, moves, winner, scores, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, game)
                        total_games += 1
                    except sqlite3.IntegrityError:
                        duplicates += 1
                conn.commit()

            source_conn.close()

        except Exception as e:
            logger.warning(f"Error processing {db_path}: {e}")

    conn.close()
    logger.info(f"Consolidated {total_games} games to {output_path} ({duplicates} duplicates skipped)")
    return total_games


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate Lambda backup for training")
    parser.add_argument("--backup-root", type=Path, default=LAMBDA_BACKUP_ROOT,
                        help="Lambda backup root directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory for consolidated databases")
    parser.add_argument("--board-type", type=str, help="Specific board type to consolidate")
    parser.add_argument("--num-players", type=int, help="Specific player count")
    parser.add_argument("--scan-only", action="store_true", help="Only scan and report stats")
    parser.add_argument("--export-npz", action="store_true", help="Export to NPZ after consolidation")
    args = parser.parse_args()

    # Find databases
    logger.info(f"Scanning Lambda backup at {args.backup_root}...")
    databases = find_databases(args.backup_root)
    logger.info(f"Found {len(databases)} game databases")

    # Gather stats
    all_stats: dict[str, int] = {}
    for db_path in databases:
        stats = get_game_stats(db_path)
        for key, count in stats.items():
            if key != "error":
                all_stats[key] = all_stats.get(key, 0) + count

    # Report stats
    logger.info("\n=== Lambda Backup Game Statistics ===")
    total_games = 0
    for key in sorted(all_stats.keys()):
        count = all_stats[key]
        total_games += count
        logger.info(f"  {key}: {count:,} games")
    logger.info(f"  TOTAL: {total_games:,} games")

    if args.scan_only:
        return

    # Determine configs to consolidate
    if args.board_type and args.num_players:
        configs = [(args.board_type, args.num_players)]
    else:
        # All configs with data
        configs = []
        for key in all_stats.keys():
            parts = key.rsplit("_", 1)
            if len(parts) == 2 and parts[1].endswith("p"):
                board_type = parts[0]
                num_players = int(parts[1][:-1])
                configs.append((board_type, num_players))

    # Consolidate each config
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for board_type, num_players in configs:
        key = f"{board_type}_{num_players}p"
        output_path = args.output_dir / f"lambda_{key}.db"
        logger.info(f"\nConsolidating {key}...")
        games = consolidate_to_config(databases, board_type, num_players, output_path)

        if args.export_npz and games > 0:
            npz_path = args.output_dir / f"lambda_{key}.npz"
            logger.info(f"Exporting to {npz_path}...")
            os.system(
                f"PYTHONPATH=. python scripts/export_replay_dataset.py "
                f"--db {output_path} --board-type {board_type} --num-players {num_players} "
                f"--output {npz_path}"
            )

    logger.info("\n=== Consolidation Complete ===")
    logger.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
