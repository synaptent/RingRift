#!/usr/bin/env python
"""
Parallel export of training samples from GameReplayDB replays.

This is a parallelized version of export_replay_dataset.py that uses
ProcessPoolExecutor to encode games across multiple CPU cores.

Performance improvement: ~10-20x faster on multi-core systems.

Usage:
    PYTHONPATH=. python scripts/export_replay_dataset_parallel.py \\
        --db data/games/jsonl_aggregated.db \\
        --board-type hexagonal \\
        --num-players 2 \\
        --output data/training/hex_2p_parallel.npz \\
        --encoder-version v3 \\
        --max-games 10000 \\
        --workers 16
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Setup path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("RINGRIFT_FORCE_CPU", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from app.db import GameReplayDB
from app.models import BoardType
from app.training.export_cache import get_export_cache


BOARD_TYPE_MAP: Dict[str, BoardType] = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hex8": BoardType.HEX8,
    "hexagonal": BoardType.HEXAGONAL,
}


def load_games_from_db(
    db_paths: List[str],
    board_type: BoardType,
    num_players: int,
    max_games: Optional[int] = None,
    require_completed: bool = False,
    min_moves: Optional[int] = None,
    max_moves: Optional[int] = None,
    require_moves: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load games from database(s) into memory for parallel processing.

    Returns list of dicts with 'initial_state', 'moves', 'game_id'.
    NOTE: final_state is NOT computed here - workers compute it during encoding.
    This keeps the loading phase fast.
    """
    games = []
    seen_game_ids = set()
    total_loaded = 0

    query_filters = {
        "board_type": board_type,
        "num_players": num_players,
        "require_moves": require_moves,
    }
    if min_moves is not None:
        query_filters["min_moves"] = min_moves
    if max_moves is not None:
        query_filters["max_moves"] = max_moves

    for db_path in db_paths:
        if not os.path.exists(db_path):
            logger.warning(f"Skipping missing: {db_path}")
            continue

        logger.info(f"Loading from: {os.path.basename(db_path)}")

        try:
            db = GameReplayDB(db_path)
        except Exception as e:
            logger.warning(f"Error opening {db_path}: {e}")
            continue

        for meta, initial_state, moves in db.iterate_games(**query_filters):
            game_id = meta.get("game_id")

            # Deduplication
            if game_id in seen_game_ids:
                continue
            seen_game_ids.add(game_id)

            if require_completed:
                status = str(meta.get("game_status", ""))
                if status != "completed":
                    continue

            if not moves:
                continue

            # Don't replay here - let workers compute final_state during encoding
            # This is the key optimization that makes loading fast
            games.append({
                "game_id": game_id,
                "initial_state": initial_state,
                "moves": moves,
                "final_state": None,  # Computed by worker
            })

            total_loaded += 1
            if max_games is not None and total_loaded >= max_games:
                break

        if max_games is not None and total_loaded >= max_games:
            break

    logger.info(f"Loaded {len(games)} games from {len(db_paths)} database(s)")
    return games


def export_parallel(
    db_paths: List[str],
    board_type: BoardType,
    num_players: int,
    output_path: str,
    *,
    num_workers: Optional[int] = None,
    encoder_version: str = "v3",
    history_length: int = 3,
    sample_every: int = 1,
    max_games: Optional[int] = None,
    require_completed: bool = False,
    min_moves: Optional[int] = None,
    max_moves: Optional[int] = None,
    use_board_aware_encoding: bool = False,
    require_moves: bool = True,
    use_cache: bool = False,
    force_export: bool = False,
) -> int:
    """
    Export training samples using parallel encoding.

    This is the main export function that:
    1. Optionally checks cache to skip if unchanged
    2. Loads games from database(s)
    3. Encodes them in parallel using ProcessPoolExecutor
    4. Saves to NPZ format
    5. Updates cache if enabled

    Returns:
        Number of samples exported (0 if cache hit)
    """
    from app.training.parallel_encoding import (
        ParallelEncoder,
        samples_to_arrays,
    )

    # Check cache if enabled
    if use_cache:
        cache = get_export_cache()
        if not cache.needs_export(
            db_paths=db_paths,
            output_path=output_path,
            board_type=board_type.value,
            num_players=num_players,
            force=force_export,
        ):
            cache_info = cache.get_cache_info(
                output_path, board_type.value, num_players
            )
            samples = cache_info.get("samples_exported", "?") if cache_info else "?"
            logger.info(f"[CACHE HIT] Skipping export - source DBs unchanged")
            logger.info(f"  Output: {output_path}")
            logger.info(f"  Cached samples: {samples}")
            return 0
        logger.info(f"[CACHE MISS] Export needed - source DBs have changed")

    start_time = time.time()

    # Load games
    logger.info("Loading games from database(s)...")
    games = load_games_from_db(
        db_paths=db_paths,
        board_type=board_type,
        num_players=num_players,
        max_games=max_games,
        require_completed=require_completed,
        min_moves=min_moves,
        max_moves=max_moves,
        require_moves=require_moves,
    )

    if not games:
        logger.warning("No games found!")
        return

    load_time = time.time() - start_time
    logger.info(f"Loaded {len(games)} games in {load_time:.1f}s")

    # Encode in parallel
    logger.info(f"Encoding with {num_workers or 'auto'} workers...")
    encode_start = time.time()

    with ParallelEncoder(
        board_type=board_type,
        num_workers=num_workers,
        encoder_version=encoder_version,
        history_length=history_length,
        sample_every=sample_every,
        use_board_aware_encoding=use_board_aware_encoding,
    ) as encoder:
        samples, errors = encoder.encode_games_batch(
            games, num_players, show_progress=True
        )

    encode_time = time.time() - encode_start
    logger.info(f"Encoded {len(samples)} samples in {encode_time:.1f}s")

    if errors:
        logger.warning(f"Encountered {len(errors)} errors during encoding")

    if not samples:
        logger.warning("No samples generated!")
        return

    # Convert to arrays
    arrays = samples_to_arrays(samples)

    # Add metadata
    arrays["board_type"] = np.asarray(board_type.value)
    arrays["board_size"] = np.asarray(int(arrays["features"].shape[-1]))
    arrays["policy_encoding"] = np.asarray(
        "board_aware" if use_board_aware_encoding else "legacy_max_n"
    )

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, **arrays)

    total_time = time.time() - start_time
    logger.info(
        f"Exported {len(samples)} samples from {len(games)} games "
        f"to {output_path} in {total_time:.1f}s"
    )
    logger.info(
        f"Performance: {len(games) / total_time:.1f} games/s, "
        f"{len(samples) / total_time:.1f} samples/s"
    )

    # Update cache if enabled
    if use_cache:
        cache.record_export(
            db_paths=db_paths,
            output_path=output_path,
            board_type=board_type.value,
            num_players=num_players,
            samples_exported=len(samples),
            games_exported=len(games),
        )
        logger.info(f"[CACHE] Recorded export: {len(samples)} samples")

    return len(samples)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel export of training samples from GameReplayDB.",
    )
    parser.add_argument(
        "--db",
        type=str,
        action="append",
        dest="db_paths",
        required=True,
        help="Path to GameReplayDB (can specify multiple)",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["square8", "square19", "hex8", "hexagonal"],
        required=True,
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)",
    )
    parser.add_argument(
        "--encoder-version",
        type=str,
        choices=["v2", "v3"],
        default="v3",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--require-completed",
        action="store_true",
    )
    parser.add_argument(
        "--min-moves",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--board-aware-encoding",
        action="store_true",
    )
    parser.add_argument(
        "--no-require-moves",
        action="store_true",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable incremental caching (skip if DBs unchanged)",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Force re-export even with valid cache",
    )

    args = parser.parse_args()

    board_type = BOARD_TYPE_MAP[args.board_type]

    export_parallel(
        db_paths=args.db_paths,
        board_type=board_type,
        num_players=args.num_players,
        output_path=args.output,
        num_workers=args.workers,
        encoder_version=args.encoder_version,
        history_length=args.history_length,
        sample_every=args.sample_every,
        max_games=args.max_games,
        require_completed=args.require_completed,
        min_moves=args.min_moves,
        max_moves=args.max_moves,
        use_board_aware_encoding=args.board_aware_encoding,
        require_moves=not args.no_require_moves,
        use_cache=args.use_cache,
        force_export=args.force_export,
    )


if __name__ == "__main__":
    main()
