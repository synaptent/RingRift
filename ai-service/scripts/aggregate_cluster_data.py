#!/usr/bin/env python
"""Aggregate training data from all cluster databases into NPZ files.

Scans all cluster_* directories and combines games by board type.

Usage:
    python scripts/aggregate_cluster_data.py --output-dir data/training/cluster_aggregated
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("RINGRIFT_FORCE_CPU", "1")

from scripts.db_to_training_npz import HexEncoderWrapper, get_encoder
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("aggregate_cluster_data")

# Unified game discovery
try:
    from app.utils.game_discovery import GameDiscovery
    HAS_GAME_DISCOVERY = True
except ImportError:
    HAS_GAME_DISCOVERY = False
    GameDiscovery = None


def discover_databases(selfplay_dir: Path) -> dict[str, list[Path]]:
    """Discover all databases grouped by board type.

    Uses unified GameDiscovery if available, falls back to legacy cluster-only scan.
    """
    databases: dict[str, list[Path]] = defaultdict(list)

    # Use unified GameDiscovery if available
    if HAS_GAME_DISCOVERY:
        logger.info("Using unified GameDiscovery...")
        discovery = GameDiscovery()
        for db_info in discovery.find_all_databases():
            if db_info.game_count == 0:
                continue
            # Group by board_type + num_players
            for (board_type, num_players), count in db_info.config_counts.items():
                if count > 0:
                    key = f"{board_type}_{num_players}p"
                    if db_info.path not in databases[key]:
                        databases[key].append(db_info.path)
        return databases

    # Fallback: Legacy cluster-only discovery
    logger.info("Falling back to legacy cluster discovery...")
    for cluster_dir in sorted(selfplay_dir.iterdir()):
        if not cluster_dir.name.startswith("cluster_100."):
            continue
        if not cluster_dir.is_dir():
            continue

        for db_path in cluster_dir.rglob("games.db"):
            # Extract board type from path
            path_str = str(db_path)
            board_type = None
            for bt in ["square8_2p", "square8_3p", "hex_2p", "hex_3p"]:
                if bt in path_str:
                    board_type = bt
                    break

            if board_type:
                databases[board_type].append(db_path)

    return databases


def count_games_in_db(db_path: Path) -> int:
    """Count games with winners in a database."""
    try:
        conn = sqlite3.connect(str(db_path))
        count = conn.execute(
            "SELECT COUNT(*) FROM games WHERE winner IS NOT NULL AND total_moves > 10"
        ).fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def aggregate_board_type(
    board_type: str,
    db_paths: list[Path],
    output_path: Path,
    sample_every: int = 3,
    max_positions: int = 1000000,
) -> int:
    """Aggregate all databases for a board type into a single NPZ file."""
    from app.db.game_replay import GameReplayDB

    # Parse num_players from board type
    num_players = 3 if "3p" in board_type else 2
    base_board_type = board_type.replace("_2p", "").replace("_3p", "")

    logger.info(f"Processing {board_type}: {len(db_paths)} databases")

    encoder = get_encoder(base_board_type, num_players)

    all_features = []
    all_globals = []
    all_values = []
    all_move_numbers = []
    all_total_moves = []
    all_num_players = []
    all_game_ids = []

    total_games_processed = 0

    for db_path in db_paths:
        game_count = count_games_in_db(db_path)
        if game_count == 0:
            continue

        logger.info(f"  Loading {db_path.name} ({game_count} games)")

        try:
            replay = GameReplayDB(str(db_path))

            # Get game IDs
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT game_id, winner, total_moves
                FROM games
                WHERE winner IS NOT NULL AND total_moves > 10
                ORDER BY game_id
            """)
            games = cursor.fetchall()
            conn.close()

            for game_id, winner, total_moves in games:
                if len(all_features) >= max_positions:
                    break

                try:
                    # Sample positions throughout the game
                    for move_num in range(0, total_moves, sample_every):
                        if len(all_features) >= max_positions:
                            break

                        state = replay.get_state_at_move(game_id, move_num)
                        if state is None:
                            continue

                        # Encode state
                        if hasattr(encoder, 'encode_state'):
                            encoded = encoder.encode_state(state)
                        else:
                            features, global_features = encoder._extract_features(state)
                            encoded = (features, global_features)

                        if encoded is None:
                            continue

                        features, global_features = encoded

                        # Compute value target
                        current_player = state.current_player
                        if winner == current_player:
                            value = 1.0
                        elif winner == 0:
                            value = 0.0
                        else:
                            value = -1.0

                        all_features.append(features)
                        all_globals.append(global_features)
                        all_values.append(value)
                        all_move_numbers.append(move_num)
                        all_total_moves.append(total_moves)
                        all_num_players.append(num_players)
                        all_game_ids.append(f"{db_path.parent.name}/{game_id}")

                    total_games_processed += 1

                except Exception as e:
                    logger.debug(f"Error processing game {game_id}: {e}")
                    continue

            if len(all_features) >= max_positions:
                logger.info(f"  Reached max positions ({max_positions})")
                break

        except Exception as e:
            logger.warning(f"  Error loading {db_path}: {e}")
            continue

    if not all_features:
        logger.warning(f"No positions extracted for {board_type}")
        return 0

    # Stack arrays
    features_arr = np.stack(all_features).astype(np.float32)
    globals_arr = np.stack(all_globals).astype(np.float32)
    values_arr = np.array(all_values, dtype=np.float32)
    move_numbers_arr = np.array(all_move_numbers, dtype=np.int32)
    total_moves_arr = np.array(all_total_moves, dtype=np.int32)
    num_players_arr = np.array(all_num_players, dtype=np.int32)

    # Save to NPZ
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features_arr,
        globals=globals_arr,
        values=values_arr,
        move_numbers=move_numbers_arr,
        total_game_moves=total_moves_arr,
        num_players=num_players_arr,
        board_type=board_type,
        games_processed=total_games_processed,
        databases_used=len(db_paths),
    )

    logger.info(f"Saved {len(all_features)} positions from {total_games_processed} games to {output_path}")
    logger.info(f"  Features shape: {features_arr.shape}")
    logger.info(f"  Values distribution: win={np.mean(values_arr > 0):.1%}, loss={np.mean(values_arr < 0):.1%}")

    return len(all_features)


def main():
    parser = argparse.ArgumentParser(description="Aggregate cluster databases to NPZ")
    parser.add_argument(
        "--selfplay-dir",
        type=str,
        default="data/selfplay",
        help="Directory containing cluster_* folders"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training/cluster_aggregated",
        help="Output directory for NPZ files"
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=3,
        help="Sample every N moves"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=500000,
        help="Max positions per board type"
    )
    parser.add_argument(
        "--board-types",
        type=str,
        nargs="*",
        default=None,
        help="Only process these board types (e.g., hex_2p square8_3p)"
    )

    args = parser.parse_args()

    selfplay_dir = Path(args.selfplay_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover databases
    databases = discover_databases(selfplay_dir)

    logger.info("=== Discovered Databases ===")
    for board_type, paths in sorted(databases.items()):
        total_games = sum(count_games_in_db(p) for p in paths)
        logger.info(f"  {board_type}: {len(paths)} databases, {total_games} games")

    # Process each board type
    results = {}
    for board_type, db_paths in sorted(databases.items()):
        if args.board_types and board_type not in args.board_types:
            logger.info(f"Skipping {board_type} (not in filter)")
            continue

        output_path = output_dir / f"cluster_{board_type}.npz"

        count = aggregate_board_type(
            board_type=board_type,
            db_paths=db_paths,
            output_path=output_path,
            sample_every=args.sample_every,
            max_positions=args.max_positions,
        )

        results[board_type] = count

    # Summary
    logger.info("\n=== Aggregation Summary ===")
    total = 0
    for board_type, count in sorted(results.items()):
        logger.info(f"  {board_type}: {count:,} positions")
        total += count
    logger.info(f"  TOTAL: {total:,} positions")

    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
