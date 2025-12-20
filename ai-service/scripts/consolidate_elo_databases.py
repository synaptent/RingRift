#!/usr/bin/env python3
"""Consolidate all Elo databases and tournament data into unified_elo.db.

This script:
1. Imports matches from existing elo_leaderboard.db
2. Imports matches from tournament JSONL files
3. Imports from any other Elo databases found
4. Cleans up participants with 0 games
5. Generates consolidated leaderboard report

Usage:
    python scripts/consolidate_elo_databases.py

    # Dry run (don't modify unified DB)
    python scripts/consolidate_elo_databases.py --dry-run

    # Skip cleanup
    python scripts/consolidate_elo_databases.py --no-cleanup
"""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
from pathlib import Path

# Add ai-service to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.tournament.unified_elo_db import (
    DEFAULT_DB_PATH as UNIFIED_ELO_DB_PATH,
    EloDatabase,
)


def import_from_old_db(db: EloDatabase, old_db_path: Path) -> int:
    """Import matches from an old Elo database."""
    if not old_db_path.exists():
        return 0

    old_conn = sqlite3.connect(str(old_db_path))
    cursor = old_conn.cursor()

    # Check what tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}

    imported = 0

    if "match_history" not in tables:
        old_conn.close()
        return 0

    try:
        # Check schema
        cursor.execute("PRAGMA table_info(match_history)")
        columns = {row[1] for row in cursor.fetchall()}

        if "model_a" in columns:
            cursor.execute("""
                SELECT model_a, model_b, board_type, num_players, winner,
                       game_length, duration_sec, timestamp, tournament_id
                FROM match_history
            """)
        elif "participant_a" in columns:
            cursor.execute("""
                SELECT participant_a, participant_b, board_type, num_players, winner,
                       game_length, duration_sec, timestamp, tournament_id
                FROM match_history
            """)
        else:
            old_conn.close()
            return 0

        for row in cursor.fetchall():
            try:
                participant_a = row[0]
                participant_b = row[1]
                board_type = row[2] or "square8"
                num_players = row[3] or 2
                winner = row[4]
                game_length = row[5] or 0
                duration_sec = row[6] or 0.0
                tournament_id = row[8] or "imported"

                if not participant_a or not participant_b:
                    continue

                # Determine rankings from winner
                if winner == participant_a:
                    rankings = [0, 1]
                elif winner == participant_b:
                    rankings = [1, 0]
                else:
                    rankings = [0, 0]  # draw

                db.record_match_and_update(
                    participant_ids=[participant_a, participant_b],
                    rankings=rankings,
                    board_type=board_type,
                    num_players=num_players,
                    tournament_id=tournament_id,
                    game_length=game_length,
                    duration_sec=duration_sec,
                )
                imported += 1
            except Exception:
                continue

    except Exception as e:
        print(f"  Error reading from {old_db_path}: {e}")

    old_conn.close()
    return imported


def import_from_jsonl(db: EloDatabase, jsonl_path: Path, tournament_id: str | None = None) -> int:
    """Import matches from tournament JSONL file."""
    if not jsonl_path.exists():
        return 0

    imported = 0

    # Handle gzip-compressed files
    try:
        with open(jsonl_path, "rb") as test_f:
            magic = test_f.read(2)
        if magic == b'\x1f\x8b':  # Gzip magic bytes
            f = gzip.open(jsonl_path, "rt", encoding="utf-8")
        else:
            f = open(jsonl_path, encoding="utf-8")
    except Exception:
        return 0

    try:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            board_type = record.get("board_type", "square8")
            num_players = record.get("num_players", 2)
            winner = record.get("winner")
            game_length = record.get("move_count") or record.get("game_length") or 0
            duration = record.get("duration_sec") or record.get("game_time_seconds") or 0.0
            game_id = record.get("game_id")
            tid = tournament_id or record.get("tournament_id") or "imported"

            model_a = record.get("model_a")
            model_b = record.get("model_b")

            if not model_a or not model_b:
                continue

            # Extract model name from path
            participant_a = Path(model_a).stem if "/" in model_a else model_a
            participant_b = Path(model_b).stem if "/" in model_b else model_b

            # Determine rankings from winner
            if winner == 1 or winner == "model_a" or winner == model_a:
                rankings = [0, 1]
            elif winner == 2 or winner == "model_b" or winner == model_b:
                rankings = [1, 0]
            else:
                rankings = [0, 0]

            try:
                db.record_match_and_update(
                    participant_ids=[participant_a, participant_b],
                    rankings=rankings,
                    board_type=board_type,
                    num_players=num_players,
                    tournament_id=tid,
                    game_length=game_length,
                    duration_sec=duration,
                    game_id=game_id,
                )
                imported += 1
            except Exception:
                continue
    finally:
        f.close()

    return imported


def find_tournament_jsonl_files(data_dir: Path) -> list:
    """Find all tournament JSONL files recursively."""
    files = []

    locations = [
        data_dir / "selfplay" / "elo_tournaments",
        data_dir / "holdouts" / "elo_tournaments",
        data_dir / "tournaments",
        data_dir / "selfplay" / "diverse",
    ]

    for loc in locations:
        if loc.exists():
            files.extend(loc.glob("*.jsonl"))
            files.extend(loc.glob("**/*.jsonl"))

    return list(set(files))


def find_elo_databases(data_dir: Path) -> list:
    """Find all Elo-related databases."""
    dbs = []

    patterns = [
        "elo_leaderboard.db",
        "elo_tournament.db",
        "elo_ratings.db",
        "**/elo*.db",
    ]

    for pattern in patterns:
        dbs.extend(data_dir.glob(pattern))

    # Filter out the target unified DB
    unified = UNIFIED_ELO_DB_PATH.resolve()
    return [db for db in set(dbs) if db.resolve() != unified]


def main():
    parser = argparse.ArgumentParser(description="Consolidate Elo databases")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify unified DB")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup of 0-game participants")
    parser.add_argument("--output", type=str, help="Custom output DB path")
    args = parser.parse_args()

    data_dir = ROOT / "data"
    output_path = Path(args.output) if args.output else UNIFIED_ELO_DB_PATH

    print("=" * 80)
    print(" Elo Database Consolidation")
    print("=" * 80)
    print(f"Output DB: {output_path}")
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        print("\n[DRY RUN] No changes will be made to the unified database.")
        db = EloDatabase(Path(":memory:"))
    else:
        db = EloDatabase(output_path)

    # Find all data sources
    print("\n--- Finding Data Sources ---")

    old_dbs = find_elo_databases(data_dir)
    print(f"Found {len(old_dbs)} Elo databases:")
    for old_db in old_dbs:
        print(f"  - {old_db}")

    jsonl_files = find_tournament_jsonl_files(data_dir)
    print(f"\nFound {len(jsonl_files)} tournament JSONL files:")
    for f in jsonl_files[:10]:
        print(f"  - {f}")
    if len(jsonl_files) > 10:
        print(f"  ... and {len(jsonl_files) - 10} more")

    # Import from old databases
    print("\n--- Importing from Old Databases ---")
    total_db_imports = 0
    for db_path in old_dbs:
        print(f"Importing from {db_path.name}...", end=" ")
        imported = import_from_old_db(db, db_path)
        print(f"{imported} matches")
        total_db_imports += imported

    print(f"Total from databases: {total_db_imports} matches")

    # Import from JSONL files
    print("\n--- Importing from Tournament JSONL Files ---")
    total_jsonl_imports = 0
    for jsonl_path in jsonl_files:
        print(f"Importing from {jsonl_path.name}...", end=" ")
        imported = import_from_jsonl(db, jsonl_path)
        print(f"{imported} matches")
        total_jsonl_imports += imported

    print(f"Total from JSONL: {total_jsonl_imports} matches")

    # Generate leaderboard reports
    print("\n--- Leaderboard Summary ---")

    configs = [
        ("square8", 2),
        ("square8", 3),
        ("square8", 4),
        ("square19", 2),
        ("square19", 3),
        ("square19", 4),
        ("hexagonal", 2),
        ("hexagonal", 3),
        ("hexagonal", 4),
    ]

    for board_type, num_players in configs:
        leaderboard = db.get_leaderboard(board_type, num_players, min_games=1, limit=5)
        if leaderboard:
            print(f"\n{board_type} {num_players}p (top 5):")
            for i, entry in enumerate(leaderboard, 1):
                pid = entry.get('participant_id', 'unknown')
                rating = entry.get('rating', 1500)
                games = entry.get('games_played', 0)
                print(f"  {i}. {pid[:35]:<35} {rating:>7.1f} ({games} games)")

    # Summary
    stats = db.get_stats()

    print("\n" + "=" * 80)
    print(" Summary")
    print("=" * 80)
    print(f"Total participants: {stats['total_participants']}")
    print(f"Active ratings (games > 0): {stats['rated_participants']}")
    print(f"Total matches recorded: {stats['total_matches']}")
    print(f"Imported from DBs: {total_db_imports}")
    print(f"Imported from JSONL: {total_jsonl_imports}")

    if not args.dry_run:
        print(f"\nUnified Elo database saved to: {output_path}")
    else:
        print("\n[DRY RUN] No changes were made.")


if __name__ == "__main__":
    main()
