#!/usr/bin/env python3
"""Migrate moves from metadata_json to game_moves table.

This script fixes JSONL-imported games that have moves stored in metadata_json
but not in the game_moves table. This is a critical data quality fix.

Usage:
    python scripts/migrate_jsonl_moves.py --db data/games/canonical_hexagonal_2p.db
    python scripts/migrate_jsonl_moves.py --db data/games/canonical_hexagonal_2p.db --dry-run
    python scripts/migrate_jsonl_moves.py --all  # Process all canonical DBs
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# Move type mapping from JSONL format to game_moves format
MOVE_TYPE_MAP = {
    "PLACEMENT": "place_ring",
    "PLACE_RING": "place_ring",
    "MOVEMENT": "move_stack",
    "MOVE_STACK": "move_stack",
    "LINE_PROCESSING": "no_line_action",
    "NO_LINE_ACTION": "no_line_action",
    "MARKER_COLLAPSE": "collapse_markers",
    "COLLAPSE_MARKERS": "collapse_markers",
    "CAPTURE": "capture_stack",
    "CAPTURE_STACK": "capture_stack",
    "RING_REMOVAL": "remove_ring",
    "REMOVE_RING": "remove_ring",
}

# Phase mapping from JSONL format to game_moves format
PHASE_MAP = {
    "RING_PLACEMENT": "ring_placement",
    "MOVEMENT": "movement",
    "LINE_PROCESSING": "line_processing",
    "MARKER_COLLAPSE": "marker_collapse",
    "GAME_OVER": "game_over",
}


def convert_pos_to_xyz(pos: list[int] | None) -> dict | None:
    """Convert [x, y] position to {x, y, z} format for hex coords."""
    if pos is None or len(pos) < 2:
        return None
    x, y = pos[0], pos[1]
    # For hex coordinates, z = -x - y (axial to cube conversion)
    z = -x - y
    return {"x": x, "y": y, "z": z}


def convert_move_to_game_moves_format(move: dict, game_id: str) -> dict:
    """Convert a move from metadata_json format to game_moves format.

    Args:
        move: Move dict from metadata_json
        game_id: The game ID

    Returns:
        Dict ready for insertion into game_moves table
    """
    # Get and normalize move_type
    raw_type = move.get("move_type", move.get("type", "unknown"))
    move_type = MOVE_TYPE_MAP.get(raw_type.upper(), raw_type.lower())

    # Get and normalize phase
    raw_phase = move.get("phase", "unknown")
    phase = PHASE_MAP.get(raw_phase.upper(), raw_phase.lower())

    # Convert positions
    from_pos = convert_pos_to_xyz(move.get("from_pos") or move.get("from"))
    to_pos = convert_pos_to_xyz(move.get("to_pos") or move.get("to"))

    # Build the move_json in the expected format
    move_json = {
        "id": f"migrated-{move.get('moveNumber', 0)}",
        "type": move_type,
        "player": move.get("player", 1),
        "from": from_pos,
        "to": to_pos,
        "captureTarget": move.get("captureTarget"),
        "capturedStacks": move.get("capturedStacks"),
        "captureChain": move.get("captureChain"),
        "overtakenRings": move.get("overtakenRings"),
        "placedOnStack": move.get("placedOnStack"),
        "placementCount": move.get("placementCount"),
        "stackMoved": move.get("stackMoved"),
        "minimumDistance": move.get("minimumDistance"),
        "actualDistance": move.get("actualDistance"),
        "markerLeft": move.get("markerLeft"),
        "lineIndex": move.get("lineIndex"),
        "formedLines": move.get("formedLines"),
        "collapsedMarkers": move.get("collapsedMarkers"),
        "claimedTerritory": move.get("claimedTerritory"),
        "disconnectedRegions": move.get("disconnectedRegions"),
        "recoveryOption": move.get("recoveryOption"),
        "recoveryMode": move.get("recoveryMode"),
        "collapsePositions": move.get("collapsePositions"),
        "extractionStacks": move.get("extractionStacks"),
        "eliminatedRings": move.get("eliminatedRings"),
        "eliminationContext": move.get("eliminationContext"),
        "timestamp": move.get("timestamp", datetime.now().isoformat()),
        "thinkTime": move.get("thinkTime", 0),
        "moveNumber": move.get("moveNumber", 0),
        "phase": phase,
    }

    return {
        "game_id": game_id,
        "move_number": move.get("moveNumber", 0) - 1,  # 0-indexed in game_moves
        "turn_number": 0,  # Not tracked in JSONL format
        "player": move.get("player", 1),
        "phase": phase,
        "move_type": move_type,
        "move_json": json.dumps(move_json),
        "timestamp": move.get("timestamp"),
        "think_time_ms": move.get("thinkTime", 0),
    }


def get_games_needing_migration(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Find games that have moves in metadata_json but not in game_moves.

    Returns:
        List of (game_id, metadata_json) tuples
    """
    cursor = conn.cursor()

    # Find games with total_moves > 0 but no entries in game_moves
    cursor.execute("""
        SELECT g.game_id, g.metadata_json
        FROM games g
        WHERE g.total_moves > 5
          AND g.metadata_json IS NOT NULL
          AND json_extract(g.metadata_json, '$.moves') IS NOT NULL
          AND json_array_length(json_extract(g.metadata_json, '$.moves')) > 0
          AND NOT EXISTS (
              SELECT 1 FROM game_moves gm WHERE gm.game_id = g.game_id
          )
    """)

    return cursor.fetchall()


def migrate_game_moves(
    conn: sqlite3.Connection,
    game_id: str,
    metadata_json: str,
    dry_run: bool = False,
) -> int:
    """Migrate moves for a single game.

    Args:
        conn: Database connection
        game_id: The game ID
        metadata_json: The metadata JSON string
        dry_run: If True, don't actually insert

    Returns:
        Number of moves migrated
    """
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        return 0

    moves = metadata.get("moves", [])
    if not moves:
        return 0

    cursor = conn.cursor()
    migrated = 0

    for move in moves:
        try:
            row = convert_move_to_game_moves_format(move, game_id)

            if not dry_run:
                cursor.execute("""
                    INSERT OR IGNORE INTO game_moves (
                        game_id, move_number, turn_number, player, phase,
                        move_type, move_json, timestamp, think_time_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["game_id"],
                    row["move_number"],
                    row["turn_number"],
                    row["player"],
                    row["phase"],
                    row["move_type"],
                    row["move_json"],
                    row["timestamp"],
                    row["think_time_ms"],
                ))
            migrated += 1
        except Exception as e:
            print(f"    Error migrating move {move.get('moveNumber', '?')}: {e}")

    return migrated


def process_database(db_path: str, dry_run: bool = False) -> tuple[int, int]:
    """Process a single database.

    Args:
        db_path: Path to the database
        dry_run: If True, don't actually modify

    Returns:
        Tuple of (games_processed, total_moves_migrated)
    """
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return 0, 0

    conn = sqlite3.connect(db_path)

    games = get_games_needing_migration(conn)
    if not games:
        print(f"  No games need migration in {Path(db_path).name}")
        conn.close()
        return 0, 0

    print(f"  Found {len(games)} games needing migration in {Path(db_path).name}")

    total_moves = 0
    games_processed = 0

    for i, (game_id, metadata_json) in enumerate(games):
        moves_migrated = migrate_game_moves(conn, game_id, metadata_json, dry_run)
        if moves_migrated > 0:
            total_moves += moves_migrated
            games_processed += 1

        if (i + 1) % 500 == 0:
            print(f"    Progress: {i + 1}/{len(games)} games, {total_moves} moves")
            if not dry_run:
                conn.commit()

    if not dry_run:
        conn.commit()

    conn.close()
    return games_processed, total_moves


def find_canonical_databases() -> list[str]:
    """Find all canonical game databases."""
    data_dir = Path("data/games")
    if not data_dir.exists():
        return []

    return sorted([
        str(p) for p in data_dir.glob("canonical_*.db")
        if not p.name.endswith(".db-journal")
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Migrate moves from metadata_json to game_moves table"
    )
    parser.add_argument(
        "--db",
        help="Database path to process",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all canonical databases",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually modify, just show what would be done",
    )

    args = parser.parse_args()

    if not args.db and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN MODE - No changes will be made ===\n")

    databases = []
    if args.all:
        databases = find_canonical_databases()
        if not databases:
            print("No canonical databases found in data/games/")
            sys.exit(1)
        print(f"Found {len(databases)} canonical databases\n")
    else:
        databases = [args.db]

    total_games = 0
    total_moves = 0

    for db_path in databases:
        print(f"\nProcessing {db_path}...")
        games, moves = process_database(db_path, args.dry_run)
        total_games += games
        total_moves += moves
        if games > 0:
            print(f"  Migrated {moves} moves from {games} games")

    print(f"\n{'=' * 50}")
    print(f"Total: {total_moves} moves migrated from {total_games} games")
    if args.dry_run:
        print("(Dry run - no changes were made)")


if __name__ == "__main__":
    main()
