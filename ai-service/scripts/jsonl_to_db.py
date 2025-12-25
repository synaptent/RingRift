#!/usr/bin/env python
"""Convert JSONL game records to SQLite DB format with full move data.

This script converts JSONL game records (from selfplay scripts) to the SQLite
DB format used by GameReplayDB, including proper storage of moves in the
game_moves table for training data export.

Features:
- Stores games with full move data (usable for training export)
- Supports both GPU selfplay format and canonical JSONL format
- Creates initial states from record or as placeholders
- Deduplication via game hash

Usage:
    # Basic conversion
    python scripts/jsonl_to_db.py --input games.jsonl --output games.db

    # With board type filter
    python scripts/jsonl_to_db.py --input games.jsonl --output games.db \
        --board-type square8 --num-players 2

    # Process directory of JSONL files
    python scripts/jsonl_to_db.py --input-dir data/selfplay/ --output games.db
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("jsonl_to_db")

from app.db.game_replay import GameReplayDB
from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    Player,
    TimeControl,
)


def is_gzip_file(filepath: Path) -> bool:
    """Check if a file is gzip-compressed by reading magic bytes."""
    try:
        with open(filepath, "rb") as f:
            magic = f.read(2)
            return magic == b"\x1f\x8b"
    except OSError:
        return False


def open_jsonl_file(filepath: Path):
    """Open a JSONL file, automatically detecting gzip compression."""
    if is_gzip_file(filepath):
        return gzip.open(filepath, "rt", encoding="utf-8", errors="replace")
    else:
        return open(filepath, encoding="utf-8", errors="replace")


def compute_game_hash(record: dict[str, Any]) -> str:
    """Compute a deterministic hash for deduplication."""
    game_id = record.get("game_id", "")
    if game_id and not game_id.startswith("gpu_"):
        return hashlib.sha256(game_id.encode()).hexdigest()[:32]

    key_parts = [
        str(record.get("seed", "")),
        str(record.get("board_type", "")),
        str(record.get("num_players", "")),
        str(record.get("engine_mode", "")),
        game_id,
    ]

    moves = record.get("moves", [])
    for move in moves[:5]:
        if isinstance(move, dict):
            to_dict = move.get("to", {})
            if to_dict:
                key_parts.append(
                    f"{move.get('type', '')}:{move.get('player', '')}:"
                    f"{to_dict.get('x', '')}:{to_dict.get('y', '')}"
                )
            elif "to_pos" in move:
                to_pos = move.get("to_pos") or []
                x = to_pos[0] if len(to_pos) > 0 else ""
                y = to_pos[1] if len(to_pos) > 1 else ""
                key_parts.append(
                    f"{move.get('move_type', '')}:{move.get('player', '')}:{x}:{y}"
                )

    key = "|".join(key_parts)
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def create_placeholder_state(
    board_type: str,
    num_players: int,
    seed: int | None,
) -> GameState:
    """Create a minimal placeholder initial state."""
    board_type_map = {
        "square8": (BoardType.SQUARE8, 8),
        "square19": (BoardType.SQUARE19, 19),
        "hexagonal": (BoardType.HEXAGONAL, 4),
        "hex": (BoardType.HEXAGONAL, 4),
        "hex8": (BoardType.HEX8, 4),
    }
    bt, board_size = board_type_map.get(board_type, (BoardType.SQUARE8, 8))

    board = BoardState(
        type=bt,
        size=board_size,
        stacks={},
        markers={},
        collapsed_spaces={},
        eliminated_rings={},
    )

    game_id = str(uuid.uuid4())
    now = datetime.now()
    players = [
        Player(
            id=f"player-{i+1}",
            username=f"AI-{i+1}",
            type="ai",
            player_number=i + 1,
            is_ready=True,
            time_remaining=300000,
            rings_in_hand=15,
            eliminated_rings=0,
            territory_spaces=0,
        )
        for i in range(num_players)
    ]

    time_control = TimeControl(
        initial_time=300000,
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


# GPU move type to canonical move type mapping
GPU_TO_CANONICAL_MOVE_TYPE = {
    "PLACEMENT": "place_ring",
    "MOVEMENT": "move_stack",
    "CAPTURE": "overtaking_capture",
    "LINE_FORMATION": "process_line",
    "TERRITORY_CLAIM": "choose_territory_option",
    "SKIP": "skip_placement",
    "NO_ACTION": "no_territory_action",
    "RECOVERY_SLIDE": "recovery_slide",
    "NO_PLACEMENT_ACTION": "no_placement_action",
    "NO_MOVEMENT_ACTION": "no_movement_action",
    "NO_LINE_ACTION": "no_line_action",
    "NO_TERRITORY_ACTION": "no_territory_action",
    "OVERTAKING_CAPTURE": "overtaking_capture",
    "CONTINUE_CAPTURE_SEGMENT": "continue_capture_segment",
    "SKIP_CAPTURE": "skip_capture",
    "SKIP_RECOVERY": "skip_recovery",
    "FORCED_ELIMINATION": "forced_elimination",
    "CHOOSE_LINE_OPTION": "choose_line_option",
    "CHOOSE_TERRITORY_OPTION": "choose_territory_option",
    "SKIP_PLACEMENT": "skip_placement",
    "ELIMINATE_RINGS_FROM_STACK": "eliminate_rings_from_stack",
}


def parse_moves(moves_data: list[Any]) -> list[Move]:
    """Parse moves from JSONL format to Move objects."""
    moves = []

    for i, move_data in enumerate(moves_data):
        try:
            if isinstance(move_data, dict):
                # Convert GPU JSONL format to canonical format
                if "move_type" in move_data and "type" not in move_data:
                    gpu_type = move_data["move_type"]
                    move_data["type"] = GPU_TO_CANONICAL_MOVE_TYPE.get(
                        gpu_type, gpu_type.lower()
                    )

                # Convert to_pos [y, x] to to {x, y}
                if "to_pos" in move_data and "to" not in move_data:
                    to_pos = move_data["to_pos"]
                    if to_pos and len(to_pos) >= 2:
                        move_data["to"] = {"x": to_pos[1], "y": to_pos[0]}

                # Convert from_pos [y, x] to from {x, y}
                if "from_pos" in move_data and "from" not in move_data:
                    from_pos = move_data["from_pos"]
                    if from_pos and len(from_pos) >= 2:
                        move_data["from"] = {"x": from_pos[1], "y": from_pos[0]}

                # Convert phase to lowercase if uppercase
                if "phase" in move_data:
                    phase = move_data["phase"]
                    if isinstance(phase, str) and phase.isupper():
                        move_data["phase"] = phase.lower()

                # Compute captureTarget for capture moves
                move_type = move_data.get("type", "")
                if move_type in (
                    "overtaking_capture",
                    "continue_capture_segment",
                    "capture",
                ):
                    from_pos = move_data.get("from_pos") or move_data.get("from")
                    to_pos = move_data.get("to_pos") or move_data.get("to")
                    if from_pos and to_pos and "captureTarget" not in move_data:
                        if isinstance(from_pos, list):
                            from_y, from_x = from_pos[0], from_pos[1]
                        else:
                            from_y = from_pos.get("y", 0)
                            from_x = from_pos.get("x", 0)
                        if isinstance(to_pos, list):
                            to_y, to_x = to_pos[0], to_pos[1]
                        else:
                            to_y, to_x = to_pos.get("y", 0), to_pos.get("x", 0)

                        dy = to_y - from_y
                        dx = to_x - from_x
                        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
                        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
                        target_y = from_y + step_y
                        target_x = from_x + step_x
                        move_data["captureTarget"] = {"x": target_x, "y": target_y}

                # Ensure required fields have defaults
                if "id" not in move_data:
                    move_data["id"] = f"move-{i}"
                if "type" not in move_data:
                    move_data["type"] = "place_ring"
                if "player" not in move_data:
                    move_data["player"] = 1

                move = Move.model_validate(move_data)
                moves.append(move)

            elif isinstance(move_data, str):
                # Skip string format moves - they lack required data
                logger.debug(f"Skipping string-format move {i}: {move_data[:50]}")
                continue

        except Exception as e:
            logger.debug(f"Failed to parse move {i}: {e}")
            continue

    return moves


def create_final_state(
    initial_state: GameState,
    winner: int,
) -> GameState:
    """Create final state based on game outcome."""
    final_state = initial_state.model_copy(deep=True)
    final_state.game_status = GameStatus.COMPLETED
    final_state.winner = winner if winner > 0 else None
    final_state.last_move_at = datetime.now()
    return final_state


def convert_jsonl_to_db(
    input_paths: list[Path],
    output_path: Path,
    board_type_filter: str | None = None,
    num_players_filter: int | None = None,
    max_games: int | None = None,
) -> dict[str, int]:
    """Convert JSONL files to SQLite DB with full move data.

    Returns statistics dict.
    """
    stats = {
        "files_processed": 0,
        "games_imported": 0,
        "games_skipped_filter": 0,
        "games_skipped_no_moves": 0,
        "games_skipped_duplicate": 0,
        "games_skipped_error": 0,
        "moves_stored": 0,
    }

    # Initialize GameReplayDB
    output_path.parent.mkdir(parents=True, exist_ok=True)
    db = GameReplayDB(str(output_path), enforce_canonical_history=False)

    # Track existing game hashes for deduplication
    existing_hashes: set[str] = set()

    for filepath in input_paths:
        if max_games and stats["games_imported"] >= max_games:
            break

        logger.info(f"Processing {filepath}...")
        stats["files_processed"] += 1

        try:
            with open_jsonl_file(filepath) as f:
                for line_num, line in enumerate(f, 1):
                    if max_games and stats["games_imported"] >= max_games:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        stats["games_skipped_error"] += 1
                        continue

                    # Check required fields
                    moves_data = record.get("moves", [])
                    if not moves_data:
                        stats["games_skipped_no_moves"] += 1
                        continue

                    # Apply filters
                    board_type = record.get("board_type", "square8")
                    num_players = record.get("num_players", 2)

                    if board_type_filter and board_type != board_type_filter:
                        stats["games_skipped_filter"] += 1
                        continue
                    if num_players_filter and num_players != num_players_filter:
                        stats["games_skipped_filter"] += 1
                        continue

                    # Check for duplicates
                    game_hash = compute_game_hash(record)
                    if game_hash in existing_hashes:
                        stats["games_skipped_duplicate"] += 1
                        continue
                    existing_hashes.add(game_hash)

                    try:
                        # Generate game ID
                        game_id = record.get("game_id") or str(uuid.uuid4())

                        # Parse initial state or create placeholder
                        initial_state_data = record.get("initial_state")
                        if initial_state_data:
                            if isinstance(initial_state_data, str):
                                initial_state_data = json.loads(initial_state_data)
                            # Ensure board type is set correctly
                            if not initial_state_data.get("boardType"):
                                initial_state_data["boardType"] = board_type
                                if "board" in initial_state_data:
                                    initial_state_data["board"]["type"] = board_type
                            initial_state = GameState.model_validate(initial_state_data)
                        else:
                            initial_state = create_placeholder_state(
                                board_type, num_players, record.get("seed")
                            )

                        # Parse moves
                        moves = parse_moves(moves_data)
                        if not moves:
                            stats["games_skipped_no_moves"] += 1
                            continue

                        # Parse winner
                        winner = record.get("winner_player") or record.get("winner", 0)
                        if isinstance(winner, str):
                            try:
                                winner = int(winner)
                            except ValueError:
                                winner = 0

                        # Create final state
                        final_state = create_final_state(initial_state, winner)

                        # Build metadata
                        metadata = {
                            "source": f"jsonl:{filepath.stem}",
                            "game_hash": game_hash,
                            "engine_mode": record.get("engine_mode", "unknown"),
                            "termination_reason": record.get("termination_reason", ""),
                            "victory_type": record.get("victory_type", ""),
                        }

                        # Store the game with moves
                        db.store_game(
                            game_id=game_id,
                            initial_state=initial_state,
                            final_state=final_state,
                            moves=moves,
                            metadata=metadata,
                            store_history_entries=False,  # Skip expensive history
                            snapshot_interval=0,  # Skip state tracking
                        )

                        stats["games_imported"] += 1
                        stats["moves_stored"] += len(moves)

                        if stats["games_imported"] % 100 == 0:
                            logger.info(
                                f"  Imported {stats['games_imported']} games, "
                                f"{stats['moves_stored']} moves..."
                            )

                    except Exception as e:
                        logger.debug(f"Failed to import game at line {line_num}: {e}")
                        stats["games_skipped_error"] += 1
                        continue

        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            continue

    # Vacuum for efficiency
    db.vacuum()

    return stats


def find_jsonl_files(input_path: Path) -> list[Path]:
    """Find all JSONL files in the given path."""
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("**/*.jsonl"))


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL game records to SQLite DB with full move data"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory containing JSONL files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output SQLite DB path",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["square8", "square19", "hexagonal", "hex8"],
        help="Filter games by board type",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        help="Filter games by player count",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        help="Maximum number of games to import",
    )

    args = parser.parse_args()

    # Collect input files
    input_paths: list[Path] = []
    if args.input:
        input_paths.append(Path(args.input))
    if args.input_dir:
        input_paths.extend(find_jsonl_files(Path(args.input_dir)))

    if not input_paths:
        parser.error("Must specify --input or --input-dir")

    logger.info("=" * 60)
    logger.info("JSONL to SQLite DB Conversion")
    logger.info("=" * 60)
    logger.info(f"Input files: {len(input_paths)}")
    logger.info(f"Output database: {args.output}")
    if args.board_type:
        logger.info(f"Board type filter: {args.board_type}")
    if args.num_players:
        logger.info(f"Player count filter: {args.num_players}")
    logger.info("")

    stats = convert_jsonl_to_db(
        input_paths=input_paths,
        output_path=Path(args.output),
        board_type_filter=args.board_type,
        num_players_filter=args.num_players,
        max_games=args.max_games,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Games imported: {stats['games_imported']}")
    logger.info(f"Moves stored: {stats['moves_stored']}")
    logger.info(f"Skipped (no moves): {stats['games_skipped_no_moves']}")
    logger.info(f"Skipped (filter): {stats['games_skipped_filter']}")
    logger.info(f"Skipped (duplicate): {stats['games_skipped_duplicate']}")
    logger.info(f"Skipped (error): {stats['games_skipped_error']}")

    if stats["games_imported"] > 0:
        logger.info(f"\nDatabase saved to: {args.output}")

    sys.exit(0 if stats["games_imported"] > 0 else 1)


if __name__ == "__main__":
    main()
