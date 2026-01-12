#!/usr/bin/env python3
"""Backfill heuristic features for existing games in GameReplayDB.

This script computes and stores heuristic features for all moves in a database
that don't already have them. Once backfilled, exports with --full-heuristics
will be 10-20x faster because they can load pre-computed values instead of
recomputing O(50) position evaluations per state.

Usage:
    # Backfill a single database with full (49) heuristics
    python scripts/backfill_heuristics.py --db data/games/canonical_hex8_2p.db --full

    # Backfill with fast (21) heuristics
    python scripts/backfill_heuristics.py --db data/games/canonical_hex8_2p.db

    # Backfill all canonical databases
    python scripts/backfill_heuristics.py --use-discovery --full

    # Dry run to see what would be backfilled
    python scripts/backfill_heuristics.py --db data/games/canonical_hex8_2p.db --dry-run

    # Backfill with parallelism
    python scripts/backfill_heuristics.py --db data/games/canonical_hex8_2p.db --full --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Add ai-service to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from app.db.game_replay import GameReplayDB
from app.models import BoardType, GameState

logger = logging.getLogger(__name__)


def extract_heuristics_for_state(
    state: GameState,
    player_number: int,
    full_heuristics: bool = True,
) -> np.ndarray:
    """Extract heuristic features for a single game state.

    Args:
        state: The game state to evaluate
        player_number: Player number for perspective
        full_heuristics: If True, extract all 49 features. If False, extract 21 fast features.

    Returns:
        numpy float32 array of heuristic features
    """
    if full_heuristics:
        from app.training.fast_heuristic_features import extract_full_heuristic_features

        return extract_full_heuristic_features(
            state, player_number=player_number, normalize=True
        )
    else:
        from app.training.fast_heuristic_features import extract_heuristic_features

        return extract_heuristic_features(
            state, player_number=player_number, eval_mode="full", normalize=True
        )


def get_moves_without_heuristics(db: GameReplayDB, game_id: str) -> list[int]:
    """Get list of move numbers that don't have stored heuristics.

    Args:
        db: Database connection
        game_id: Game identifier

    Returns:
        List of move numbers without heuristics
    """
    with db._get_conn() as conn:
        rows = conn.execute(
            """
            SELECT move_number
            FROM game_moves
            WHERE game_id = ? AND heuristic_features IS NULL
            ORDER BY move_number
            """,
            (game_id,),
        ).fetchall()
        return [row["move_number"] for row in rows]


def get_games_needing_backfill(db: GameReplayDB) -> list[tuple[str, int]]:
    """Get list of (game_id, missing_count) for games needing backfill.

    Args:
        db: Database connection

    Returns:
        List of (game_id, count_of_moves_without_heuristics) tuples
    """
    with db._get_conn() as conn:
        rows = conn.execute(
            """
            SELECT game_id, COUNT(*) as missing_count
            FROM game_moves
            WHERE heuristic_features IS NULL
            GROUP BY game_id
            ORDER BY game_id
            """,
        ).fetchall()
        return [(row["game_id"], row["missing_count"]) for row in rows]


def backfill_game(
    db_path: str,
    game_id: str,
    full_heuristics: bool = True,
    dry_run: bool = False,
) -> tuple[str, int, int, float]:
    """Backfill heuristics for a single game.

    Args:
        db_path: Path to database
        game_id: Game identifier
        full_heuristics: If True, compute 49 features. If False, 21 features.
        dry_run: If True, don't actually store values

    Returns:
        Tuple of (game_id, moves_processed, moves_skipped, duration_seconds)
    """
    from app.game_engine import GameEngine

    start_time = time.time()
    moves_processed = 0
    moves_skipped = 0

    db = GameReplayDB(db_path)

    # Get moves without heuristics
    missing_moves = set(get_moves_without_heuristics(db, game_id))

    if not missing_moves:
        return game_id, 0, 0, 0.0

    # Get game data using batch methods
    initial_states = db.get_initial_states_batch([game_id])
    moves_map = db.get_moves_batch([game_id])

    initial_state = initial_states.get(game_id)
    moves = moves_map.get(game_id, [])

    if initial_state is None:
        logger.warning(f"Game {game_id} has no initial state")
        return game_id, 0, 0, 0.0

    # Replay game and compute heuristics for missing moves
    state = initial_state
    for move_num, move in enumerate(moves):
        if move_num in missing_moves:
            try:
                # Compute heuristics for state before move
                features = extract_heuristics_for_state(
                    state, state.current_player, full_heuristics
                )

                if not dry_run:
                    db.store_move_heuristics(game_id, move_num, features)

                moves_processed += 1
            except Exception as e:
                logger.debug(f"Failed to compute heuristics for {game_id} move {move_num}: {e}")
                moves_skipped += 1
        else:
            moves_skipped += 1

        # Apply move to get next state
        try:
            state = GameEngine.apply_move(state, move)
        except Exception as e:
            logger.debug(f"Failed to apply move {move_num} in {game_id}: {e}")
            break

    duration = time.time() - start_time
    return game_id, moves_processed, moves_skipped, duration


def backfill_database(
    db_path: str,
    full_heuristics: bool = True,
    dry_run: bool = False,
    workers: int = 1,
    batch_size: int = 100,
) -> dict[str, Any]:
    """Backfill heuristics for all games in a database.

    Args:
        db_path: Path to database
        full_heuristics: If True, compute 49 features. If False, 21 features.
        dry_run: If True, don't actually store values
        workers: Number of parallel workers (1 = sequential)
        batch_size: Number of games per batch for progress reporting

    Returns:
        Dict with stats: total_games, total_moves, duration_seconds
    """
    db = GameReplayDB(db_path)

    # Get games needing backfill
    games_needing_backfill = get_games_needing_backfill(db)

    if not games_needing_backfill:
        logger.info(f"No games need backfilling in {db_path}")
        return {"total_games": 0, "total_moves": 0, "duration_seconds": 0.0}

    total_missing_moves = sum(count for _, count in games_needing_backfill)
    logger.info(
        f"Found {len(games_needing_backfill)} games with {total_missing_moves} moves "
        f"needing heuristics in {db_path}"
    )

    start_time = time.time()
    total_processed = 0
    total_skipped = 0

    game_ids = [game_id for game_id, _ in games_needing_backfill]

    if workers == 1:
        # Sequential processing
        with tqdm(total=len(game_ids), desc="Backfilling games") as pbar:
            for game_id in game_ids:
                _, processed, skipped, _ = backfill_game(
                    db_path, game_id, full_heuristics, dry_run
                )
                total_processed += processed
                total_skipped += skipped
                pbar.update(1)
                pbar.set_postfix(moves=total_processed)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    backfill_game, db_path, game_id, full_heuristics, dry_run
                ): game_id
                for game_id in game_ids
            }

            with tqdm(total=len(game_ids), desc="Backfilling games") as pbar:
                for future in as_completed(futures):
                    try:
                        _, processed, skipped, _ = future.result()
                        total_processed += processed
                        total_skipped += skipped
                    except Exception as e:
                        logger.warning(f"Game backfill failed: {e}")
                    pbar.update(1)
                    pbar.set_postfix(moves=total_processed)

    duration = time.time() - start_time

    logger.info(
        f"Backfill complete: {total_processed} moves processed, "
        f"{total_skipped} skipped in {duration:.1f}s"
    )

    return {
        "total_games": len(game_ids),
        "total_moves": total_processed,
        "moves_skipped": total_skipped,
        "duration_seconds": duration,
    }


def find_databases_to_backfill(
    board_type: str | None = None,
    num_players: int | None = None,
) -> list[str]:
    """Find all canonical databases that might need backfilling.

    Args:
        board_type: Optional filter by board type
        num_players: Optional filter by number of players

    Returns:
        List of database paths
    """
    from app.utils.game_discovery import GameDiscovery

    discovery = GameDiscovery()
    all_dbs = discovery.find_all_databases()

    # Filter by criteria
    result = []
    for db_info in all_dbs:
        if board_type and db_info.board_type.value != board_type:
            continue
        if num_players and db_info.num_players != num_players:
            continue
        if db_info.game_count > 0:
            result.append(str(db_info.path))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Backfill heuristic features for existing games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--db",
        type=str,
        help="Path to database to backfill",
    )
    parser.add_argument(
        "--use-discovery",
        action="store_true",
        help="Use GameDiscovery to find all databases",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["hex8", "square8", "square19", "hexagonal"],
        help="Filter databases by board type (with --use-discovery)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        help="Filter databases by number of players (with --use-discovery)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Compute full (49) heuristics instead of fast (21)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually store values, just report what would be done",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate args
    if not args.db and not args.use_discovery:
        parser.error("Either --db or --use-discovery must be specified")

    # Find databases
    if args.db:
        db_paths = [args.db]
    else:
        db_paths = find_databases_to_backfill(args.board_type, args.num_players)
        if not db_paths:
            logger.error("No databases found matching criteria")
            return 1

    logger.info(f"Processing {len(db_paths)} database(s)")

    # Process each database
    total_stats = {
        "databases": 0,
        "total_games": 0,
        "total_moves": 0,
        "duration_seconds": 0.0,
    }

    for db_path in db_paths:
        if not os.path.exists(db_path):
            logger.warning(f"Database not found: {db_path}")
            continue

        logger.info(f"\nProcessing: {db_path}")

        stats = backfill_database(
            db_path=db_path,
            full_heuristics=args.full,
            dry_run=args.dry_run,
            workers=args.workers,
        )

        total_stats["databases"] += 1
        total_stats["total_games"] += stats["total_games"]
        total_stats["total_moves"] += stats["total_moves"]
        total_stats["duration_seconds"] += stats["duration_seconds"]

    # Print summary
    print("\n" + "=" * 60)
    print("BACKFILL SUMMARY")
    print("=" * 60)
    print(f"Databases processed: {total_stats['databases']}")
    print(f"Games backfilled:    {total_stats['total_games']}")
    print(f"Moves processed:     {total_stats['total_moves']}")
    print(f"Total time:          {total_stats['duration_seconds']:.1f}s")

    if total_stats["total_moves"] > 0 and total_stats["duration_seconds"] > 0:
        rate = total_stats["total_moves"] / total_stats["duration_seconds"]
        print(f"Processing rate:     {rate:.1f} moves/sec")

    if args.dry_run:
        print("\n(DRY RUN - no values were stored)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
