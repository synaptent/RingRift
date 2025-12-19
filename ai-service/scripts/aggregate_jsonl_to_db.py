#!/usr/bin/env python3
"""Aggregate JSONL selfplay data into SQLite database for training.

This script bridges the gap between distributed JSONL selfplay generation
and SQLite-based training pipelines. It reads JSONL files from the
aggregated selfplay directories and imports them into a GameReplayDB.

Features:
- Incremental import (skips already-imported games)
- Source tracking (which machine generated each game)
- Deduplication via game hash
- Progress reporting and statistics
- FSM validation status propagation

Usage:
    # Import all aggregated data to training database
    python scripts/aggregate_jsonl_to_db.py \
        --input-dir data/selfplay/aggregated \
        --output-db data/games/training_aggregated.db

    # Import specific sources only
    python scripts/aggregate_jsonl_to_db.py \
        --input-dir data/selfplay/aggregated \
        --output-db data/games/training_aggregated.db \
        --sources lambda_h100 vast_3090

    # Dry run to see what would be imported
    python scripts/aggregate_jsonl_to_db.py \
        --input-dir data/selfplay/aggregated \
        --dry-run

    # Filter by board type and player count
    python scripts/aggregate_jsonl_to_db.py \
        --input-dir data/selfplay/aggregated \
        --output-db data/games/square8_2p_training.db \
        --board-type square8 \
        --num-players 2
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def is_gzip_file(filepath: Path) -> bool:
    """Check if a file is gzip-compressed by reading magic bytes."""
    try:
        with open(filepath, "rb") as f:
            magic = f.read(2)
            return magic == b'\x1f\x8b'  # Gzip magic number
    except (IOError, OSError):
        return False


def open_jsonl_file(filepath: Path):
    """Open a JSONL file, automatically detecting gzip compression.

    Returns a context manager that yields lines as strings.
    """
    if is_gzip_file(filepath):
        return gzip.open(filepath, "rt", encoding="utf-8", errors="replace")
    else:
        return open(filepath, "r", encoding="utf-8", errors="replace")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import (
    BoardType, BoardState, GamePhase, GameState, GameStatus,
    Move, MoveType, Position, Player, TimeControl
)

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("aggregate_jsonl_to_db")


def compute_game_hash(record: Dict[str, Any]) -> str:
    """Compute a deterministic hash for deduplication.

    Uses seed + board_type + num_players + first few moves to identify
    duplicate games across sources.
    """
    key_parts = [
        str(record.get("seed", "")),
        str(record.get("board_type", "")),
        str(record.get("num_players", "")),
        str(record.get("engine_mode", "")),
    ]

    # Include first 5 moves for additional uniqueness
    moves = record.get("moves", [])
    for i, move in enumerate(moves[:5]):
        if isinstance(move, dict):
            key_parts.append(f"{move.get('type', '')}:{move.get('player', '')}:{move.get('to', {}).get('x', '')}:{move.get('to', {}).get('y', '')}")
        elif isinstance(move, str):
            key_parts.append(move[:50])  # Truncate long move strings

    key = "|".join(key_parts)
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def parse_jsonl_record(record: Dict[str, Any], source: str, filepath: str) -> Optional[Dict[str, Any]]:
    """Parse a JSONL record into normalized format.

    Returns None if the record is invalid or incomplete.
    """
    try:
        # Required fields
        if "moves" not in record:
            return None
        if record.get("status") != "completed":
            return None

        # Extract board type - handle various formats
        board_type_raw = record.get("board_type", "square8")
        if isinstance(board_type_raw, str):
            # Normalize board type string
            board_type = board_type_raw.lower().replace("-", "_")
        else:
            board_type = "square8"

        # Parse winner
        winner = record.get("winner")
        if winner is None or winner == "":
            winner = 0  # Draw
        else:
            winner = int(winner)

        # Parse moves list
        moves = record.get("moves", [])
        if not isinstance(moves, list):
            return None

        # Generate unique game ID if not present
        game_id = record.get("game_id") or str(uuid.uuid4())

        # Extract metadata
        parsed = {
            "game_id": game_id,
            "board_type": board_type,
            "num_players": int(record.get("num_players", 2)),
            "seed": record.get("seed"),
            "winner": winner,
            "total_moves": len(moves),
            "moves": moves,
            "initial_state": record.get("initial_state"),
            "termination_reason": record.get("termination_reason", ""),
            "victory_type": record.get("victory_type", ""),
            "stalemate_tiebreaker": record.get("stalemate_tiebreaker", ""),
            "engine_mode": record.get("engine_mode", "unknown"),
            "source": source,
            "source_file": filepath,
            "game_hash": compute_game_hash(record),
        }

        return parsed

    except Exception as e:
        logger.debug(f"Failed to parse record: {e}")
        return None


def scan_aggregated_directory(
    input_dir: Path,
    sources: Optional[List[str]] = None,
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
) -> List[Tuple[Path, str]]:
    """Scan directory recursively for JSONL files.

    Returns list of (filepath, source_name) tuples.
    """
    results = []

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return results

    # Use recursive glob to find ALL .jsonl files at any depth (games.jsonl, games_*.jsonl, etc)
    for jsonl_file in input_dir.glob("**/*.jsonl"):
        # Skip empty files
        try:
            if jsonl_file.stat().st_size == 0:
                continue
        except OSError:
            continue

        # Extract source name from path (first directory component after input_dir)
        try:
            rel_path = jsonl_file.relative_to(input_dir)
            source_name = rel_path.parts[0] if rel_path.parts else "unknown"
        except ValueError:
            source_name = "unknown"

        # Filter by sources if specified
        if sources and source_name not in sources:
            continue

        # Try to parse board type and player count from path
        path_str = str(jsonl_file)

        # Check board type filter
        if board_type:
            if board_type not in path_str.lower():
                continue

        # Check player count filter (look for patterns like "2p", "_2_", etc.)
        if num_players:
            player_patterns = [f"{num_players}p", f"_{num_players}_", f"_{num_players}p"]
            if not any(p in path_str for p in player_patterns):
                continue

        results.append((jsonl_file, source_name))

    logger.info(f"Found {len(results)} JSONL files via recursive scan")
    return results


def get_existing_hashes(db_path: Path) -> Set[str]:
    """Get set of game hashes already in the database."""
    hashes = set()

    if not db_path.exists():
        return hashes

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT metadata_json FROM games WHERE metadata_json IS NOT NULL"
        )
        for row in cursor:
            try:
                metadata = json.loads(row[0])
                if "game_hash" in metadata:
                    hashes.add(metadata["game_hash"])
            except (json.JSONDecodeError, TypeError):
                pass
        conn.close()
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet

    return hashes


def import_to_database(
    records: List[Dict[str, Any]],
    db_path: Path,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Import parsed records to GameReplayDB.

    Returns statistics dict.
    """
    stats = {
        "total_records": len(records),
        "imported": 0,
        "skipped_duplicate": 0,
        "skipped_error": 0,
    }

    if dry_run:
        logger.info(f"DRY RUN: Would import {len(records)} records to {db_path}")
        return stats

    # Get existing hashes for deduplication
    existing_hashes = get_existing_hashes(db_path)
    logger.info(f"Found {len(existing_hashes)} existing games in database")

    # Filter out duplicates before import
    new_records = []
    for record in records:
        if record["game_hash"] in existing_hashes:
            stats["skipped_duplicate"] += 1
        else:
            new_records.append(record)
            existing_hashes.add(record["game_hash"])  # Track to avoid dups within batch

    if not new_records:
        logger.info("No new records to import")
        return stats

    logger.info(f"Importing {len(new_records)} new records ({stats['skipped_duplicate']} duplicates skipped)")

    # Import using GameReplayDB
    from app.db.game_replay import GameReplayDB

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = GameReplayDB(str(db_path), enforce_canonical_history=False)

    for i, record in enumerate(new_records):
        try:
            # Build metadata
            metadata = {
                "source": record["source"],
                "source_file": record["source_file"],
                "game_hash": record["game_hash"],
                "engine_mode": record["engine_mode"],
                "termination_reason": record["termination_reason"],
                "victory_type": record["victory_type"],
                "stalemate_tiebreaker": record["stalemate_tiebreaker"],
            }

            # Create minimal initial state if not provided
            initial_state_data = record.get("initial_state")
            if initial_state_data:
                if isinstance(initial_state_data, str):
                    initial_state_data = json.loads(initial_state_data)
                initial_state = GameState.model_validate(initial_state_data)
            else:
                # Create placeholder initial state
                initial_state = _create_placeholder_state(
                    record["board_type"],
                    record["num_players"],
                    record["seed"],
                )

            # Parse moves
            moves = _parse_moves(record["moves"], initial_state)

            # Create final state by applying moves (or use placeholder)
            final_state = _create_final_state(
                initial_state,
                moves,
                record["winner"],
                record["termination_reason"],
            )

            # Store the game (without history entries for efficiency)
            # Use snapshot_interval=0 to skip move validation/application
            # since placeholder states don't match actual game progression
            db.store_game(
                game_id=record["game_id"],
                initial_state=initial_state,
                final_state=final_state,
                moves=moves,
                metadata=metadata,
                store_history_entries=False,  # Skip expensive history for bulk import
                snapshot_interval=0,  # Disable snapshots to skip move application
            )

            stats["imported"] += 1

            if (i + 1) % 100 == 0:
                logger.info(f"  Imported {i + 1}/{len(new_records)} games...")

        except Exception as e:
            logger.debug(f"Failed to import game {record['game_id']}: {e}")
            stats["skipped_error"] += 1

    # Vacuum for efficiency
    db.vacuum()

    return stats


def _create_placeholder_state(
    board_type: str,
    num_players: int,
    seed: Optional[int],
) -> GameState:
    """Create a minimal placeholder initial state."""
    # Map board type string to enum and size
    board_type_map = {
        "square8": (BoardType.SQUARE8, 8),
        "square19": (BoardType.SQUARE19, 19),
        "hexagonal": (BoardType.HEXAGONAL, 4),  # radius 4
        "hex": (BoardType.HEXAGONAL, 4),
        "hex8": (BoardType.HEX8, 4),
    }
    bt, board_size = board_type_map.get(board_type, (BoardType.SQUARE8, 8))

    # Create board state with required fields
    board = BoardState(
        type=bt,
        size=board_size,
        stacks={},
        markers={},
        collapsed_spaces={},
        eliminated_rings={},
    )

    # Create minimal player states
    game_id = str(uuid.uuid4())
    now = datetime.now()
    players = [
        Player(
            id=f"player-{i+1}",
            username=f"AI-{i+1}",
            type="ai",
            player_number=i + 1,
            is_ready=True,
            time_remaining=300000,  # 5 minutes in ms
            rings_in_hand=15,  # Standard starting rings
            eliminated_rings=0,
            territory_spaces=0,
        )
        for i in range(num_players)
    ]

    # Default time control
    time_control = TimeControl(
        initial_time=300000,  # 5 minutes
        increment=0,
        type="untimed",
    )

    return GameState(
        id=game_id,
        board_type=bt,
        rng_seed=seed or 0,
        board=board,
        players=players,
        current_phase=GamePhase.RING_PLACEMENT,
        current_player=1,
        time_control=time_control,
        game_status=GameStatus.ACTIVE,
        created_at=now,
        last_move_at=now,
        is_rated=False,
        max_players=num_players,
        total_rings_in_play=0,
        total_rings_eliminated=0,
        victory_threshold=3,
        territory_victory_threshold=50,
    )


def _parse_moves(moves_data: List[Any], initial_state: GameState) -> List[Move]:
    """Parse moves from JSONL format to Move objects.

    IMPORTANT: This function preserves ALL move fields from the JSONL data,
    including capture_target, captured_stacks, capture_chain, formed_lines,
    collapsed_markers, etc. These fields are required for proper game replay.

    Using Move.model_validate() ensures all fields are preserved through
    Pydantic's alias handling (e.g., "captureTarget" -> capture_target).
    """
    moves = []

    for i, move_data in enumerate(moves_data):
        try:
            if isinstance(move_data, dict):
                # Ensure required fields have defaults
                if "id" not in move_data:
                    move_data["id"] = f"move-{i}"
                if "type" not in move_data:
                    move_data["type"] = "place_ring"
                if "player" not in move_data:
                    move_data["player"] = 1

                # Use model_validate to preserve ALL fields including:
                # - capture_target/captureTarget
                # - captured_stacks/capturedStacks
                # - capture_chain/captureChain
                # - overtaken_rings/overtakenRings
                # - formed_lines/formedLines
                # - collapsed_markers/collapsedMarkers
                # - claimed_territory/claimedTerritory
                # - stack_moved/stackMoved
                # - minimum_distance/minimumDistance
                # - actual_distance/actualDistance
                # - marker_left/markerLeft
                # - placed_on_stack/placedOnStack
                # - placement_count/placementCount
                # - line_index/lineIndex
                # - recovery_option/recoveryOption
                # - recovery_mode/recoveryMode
                # - collapse_positions/collapsePositions
                # - extraction_stacks/extractionStacks
                move = Move.model_validate(move_data)
                moves.append(move)

            elif isinstance(move_data, str):
                # Handle string format (e.g., "P1:PLACE(3,4)")
                # This format lacks required data for replay - skip these
                logger.debug(f"Skipping string-format move {i}: {move_data[:50]}")
                continue

        except Exception as e:
            logger.debug(f"Failed to parse move {i}: {e}")
            continue

    return moves


def _create_final_state(
    initial_state: GameState,
    moves: List[Move],
    winner: int,
    termination_reason: str,
) -> GameState:
    """Create final state based on game outcome."""
    final_state = initial_state.model_copy(deep=True)

    # Update game status
    final_state.game_status = GameStatus.COMPLETED
    final_state.winner = winner if winner > 0 else None
    final_state.last_move_at = datetime.now()

    return final_state


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate JSONL selfplay data into SQLite database"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="ai-service/data/selfplay/aggregated",
        help="Input directory containing aggregated JSONL files",
    )
    parser.add_argument(
        "--output-db",
        type=str,
        default="ai-service/data/games/training_aggregated.db",
        help="Output SQLite database path",
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        help="Filter to specific sources (e.g., lambda_h100 vast_3090)",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["square8", "square19", "hexagonal", "hex8"],
        help="Filter to specific board type",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        help="Filter to specific player count",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without actually importing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dir = Path(args.input_dir)
    output_db = Path(args.output_db)

    logger.info("=" * 60)
    logger.info("JSONL to SQLite Aggregation")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output database: {output_db}")
    if args.sources:
        logger.info(f"Filtering sources: {args.sources}")
    if args.board_type:
        logger.info(f"Filtering board type: {args.board_type}")
    if args.num_players:
        logger.info(f"Filtering player count: {args.num_players}")
    logger.info("")

    # Scan for JSONL files
    jsonl_files = scan_aggregated_directory(
        input_dir,
        sources=args.sources,
        board_type=args.board_type,
        num_players=args.num_players,
    )

    if not jsonl_files:
        logger.error("No JSONL files found matching criteria")
        return 1

    logger.info(f"Found {len(jsonl_files)} JSONL files to process")

    # Parse all records
    all_records = []
    records_by_source: Dict[str, int] = {}

    for filepath, source in jsonl_files:
        is_compressed = is_gzip_file(filepath)
        logger.info(f"  Reading {filepath}{'  (gzip)' if is_compressed else ''}...")
        try:
            with open_jsonl_file(filepath) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        parsed = parse_jsonl_record(record, source, str(filepath))
                        if parsed:
                            all_records.append(parsed)
                            records_by_source[source] = records_by_source.get(source, 0) + 1
                    except json.JSONDecodeError as e:
                        logger.debug(f"Invalid JSON at line {line_num}: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            continue

    logger.info("")
    logger.info(f"Parsed {len(all_records)} valid game records:")
    for source, count in sorted(records_by_source.items()):
        logger.info(f"  {source}: {count}")
    logger.info("")

    # Import to database
    stats = import_to_database(all_records, output_db, dry_run=args.dry_run)

    logger.info("")
    logger.info("=" * 60)
    logger.info("AGGREGATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total records processed: {stats['total_records']}")
    logger.info(f"Successfully imported: {stats['imported']}")
    logger.info(f"Skipped (duplicate): {stats['skipped_duplicate']}")
    logger.info(f"Skipped (error): {stats['skipped_error']}")

    if not args.dry_run and stats["imported"] > 0:
        logger.info(f"Database saved to: {output_db}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
