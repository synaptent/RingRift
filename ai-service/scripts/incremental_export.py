#!/usr/bin/env python
"""Incremental NPZ export - only processes new games since last export.

This dramatically reduces export time by:
1. Tracking which game_ids have already been exported
2. Only processing new games
3. Saving to shards that can be efficiently merged

Usage:
    # Export only new games for square8 2p
    python scripts/incremental_export.py --board-type square8 --num-players 2

    # Merge all shards into final NPZ for training
    python scripts/incremental_export.py --board-type square8 --num-players 2 --merge-only

    # Show export statistics
    python scripts/incremental_export.py --board-type square8 --num-players 2 --stats

    # Reset tracking (re-export everything)
    python scripts/incremental_export.py --board-type square8 --num-players 2 --reset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Ensure app.* imports resolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("incremental_export")

from app.training.incremental_export import (
    ExportStats,
    get_incremental_exporter,
)

# Import conversion functions from jsonl_to_npz
from scripts.jsonl_to_npz import (
    build_encoder,
    BOARD_TYPE_MAP,
)
from app.models import BoardType


# JSONL source directories (same as multi_config_training_loop.py)
CONFIG_JSONL_DIRS: dict[tuple[str, int], list[str]] = {
    ("square8", 2): [
        "data/selfplay/canonical",
        "data/games",
        "data/selfplay/gpu",
        "data/selfplay/mcts_square8_2p",
        "data/selfplay/mcts_cluster_collected_v3",
        "data/selfplay/hybrid_test",
        "data/selfplay/reanalyzed_square8_2p",
        "data/selfplay/cluster_h100",
    ],
    ("square8", 3): ["data/selfplay/canonical", "data/games", "data/selfplay/gpu"],
    ("square8", 4): ["data/selfplay/canonical", "data/games", "data/selfplay/gpu"],
    ("square19", 2): ["data/selfplay/canonical", "data/games", "data/selfplay/gpu"],
    ("square19", 3): ["data/selfplay/canonical", "data/games"],
    ("square19", 4): ["data/selfplay/canonical", "data/games"],
    ("hexagonal", 2): ["data/selfplay/canonical", "data/games"],
    ("hexagonal", 3): ["data/selfplay/canonical", "data/games"],
    ("hexagonal", 4): ["data/selfplay/canonical", "data/games"],
    ("hex8", 2): ["data/selfplay/hex8_policy_c", "data/selfplay/hex8_combined"],
    ("hex8", 3): ["data/selfplay/hex8_policy_c", "data/selfplay/hex8_combined"],
    ("hex8", 4): ["data/selfplay/hex8_policy_c", "data/selfplay/hex8_combined"],
}


def find_jsonl_files(base_dir: str, paths: list[str]) -> list[Path]:
    """Find all JSONL files in the given paths."""
    jsonl_files = []
    for path in paths:
        full_path = Path(base_dir) / path
        if full_path.is_file() and full_path.suffix == ".jsonl":
            jsonl_files.append(full_path)
        elif full_path.is_dir():
            jsonl_files.extend(full_path.glob("**/*.jsonl"))
    return sorted(jsonl_files)


def scan_game_ids_from_jsonl(
    jsonl_path: Path,
    board_type: str,
    num_players: int,
    max_lines: int = 50000,
) -> set[str]:
    """Scan a JSONL file for game IDs matching the config."""
    game_ids = set()

    # Board type variants for matching
    board_variants = {
        "square8": ["square8", "sq8"],
        "square19": ["square19", "sq19"],
        "hexagonal": ["hex", "hexagonal"],
        "hex8": ["hex8"],
    }
    variants = board_variants.get(board_type, [board_type])

    try:
        lines_read = 0
        with open(jsonl_path, "r") as f:
            for line in f:
                lines_read += 1
                if lines_read > max_lines:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    game_board = record.get("board_type", "")
                    game_players = record.get("num_players", 0)
                    game_id = record.get("game_id", "")
                    has_moves = "moves" in record and len(record.get("moves", [])) > 0

                    # Check if matches our config
                    board_match = game_board in variants or game_board == board_type
                    players_match = game_players == num_players

                    if board_match and players_match and has_moves and game_id:
                        game_ids.add(game_id)

                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Error scanning {jsonl_path}: {e}")

    return game_ids


def process_new_games_from_jsonl(
    jsonl_path: Path,
    new_game_ids: set[str],
    encoder,
    board_type: BoardType,
    board_type_str: str,
    num_players: int,
    sample_every: int,
    history_length: int,
) -> tuple[
    list[np.ndarray],  # features
    list[np.ndarray],  # globals
    list[float],       # values
    list[np.ndarray],  # values_mp
    list[int],         # num_players_list
    list[np.ndarray],  # policy_indices
    list[np.ndarray],  # policy_values
    list[int],         # move_numbers
    list[int],         # total_game_moves
    list[str],         # phases
    list[str],         # processed_game_ids
    int,               # positions_extracted
]:
    """Process only new games from a JSONL file."""
    features_list = []
    globals_list = []
    values_list = []
    values_mp_list = []
    num_players_list = []
    policy_indices_list = []
    policy_values_list = []
    move_numbers_list = []
    total_game_moves_list = []
    phases_list = []
    processed_game_ids = []
    positions_extracted = 0

    # Board type variants for matching
    board_variants = {
        "square8": ["square8", "sq8"],
        "square19": ["square19", "sq19"],
        "hexagonal": ["hex", "hexagonal"],
        "hex8": ["hex8"],
    }
    variants = board_variants.get(board_type_str, [board_type_str])

    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    game_id = record.get("game_id", "")

                    # Skip if not in our new games set
                    if game_id not in new_game_ids:
                        continue

                    # Verify config match
                    game_board = record.get("board_type", "")
                    game_players = record.get("num_players", 0)

                    board_match = game_board in variants or game_board == board_type_str
                    if not board_match or game_players != num_players:
                        continue

                    # Process this game using the GPU selfplay processor
                    from scripts.jsonl_to_npz import _process_gpu_selfplay_record

                    (
                        gpu_features, gpu_globals, gpu_values, gpu_values_mp,
                        gpu_num_players, gpu_policy_idx, gpu_policy_val,
                        gpu_move_nums, gpu_total_moves, gpu_phases, extracted
                    ) = _process_gpu_selfplay_record(
                        record, encoder, history_length, sample_every
                    )

                    if extracted > 0:
                        features_list.extend(gpu_features)
                        globals_list.extend(gpu_globals)
                        values_list.extend(gpu_values)
                        values_mp_list.extend(gpu_values_mp)
                        num_players_list.extend(gpu_num_players)
                        policy_indices_list.extend(gpu_policy_idx)
                        policy_values_list.extend(gpu_policy_val)
                        move_numbers_list.extend(gpu_move_nums)
                        total_game_moves_list.extend(gpu_total_moves)
                        phases_list.extend(gpu_phases)
                        processed_game_ids.append(game_id)
                        positions_extracted += extracted

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"Error processing game {game_id}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error reading {jsonl_path}: {e}")

    return (
        features_list, globals_list, values_list, values_mp_list,
        num_players_list, policy_indices_list, policy_values_list,
        move_numbers_list, total_game_moves_list, phases_list,
        processed_game_ids, positions_extracted
    )


def run_incremental_export(
    board_type: str,
    num_players: int,
    sample_every: int = 5,
    history_length: int = 3,
    encoder_version: str = "v2",
    base_dir: str = ROOT,
) -> ExportStats:
    """Run incremental export for a config.

    Only exports games that haven't been exported before.
    """
    start_time = time.time()
    stats = ExportStats()

    logger.info(f"Starting incremental export for {board_type}_{num_players}p")

    # Get incremental exporter
    exporter = get_incremental_exporter(board_type, num_players)

    # Find JSONL sources
    config = (board_type, num_players)
    jsonl_dirs = CONFIG_JSONL_DIRS.get(config, [])
    jsonl_files = find_jsonl_files(base_dir, jsonl_dirs)

    if not jsonl_files:
        logger.warning(f"No JSONL files found for {board_type}_{num_players}p")
        return stats

    logger.info(f"Found {len(jsonl_files)} JSONL files")

    # Scan for all game IDs
    all_game_ids: set[str] = set()
    for jsonl_path in jsonl_files:
        ids = scan_game_ids_from_jsonl(jsonl_path, board_type, num_players)
        all_game_ids.update(ids)

    stats.total_games_seen = len(all_game_ids)
    logger.info(f"Found {len(all_game_ids)} total games")

    # Get unexported game IDs
    new_game_ids = exporter.get_unexported_game_ids(all_game_ids)
    stats.new_games_found = len(new_game_ids)

    if not new_game_ids:
        logger.info("No new games to export")
        stats.export_time_seconds = time.time() - start_time
        return stats

    logger.info(f"Found {len(new_game_ids)} new games to export")

    # Build encoder
    board_type_enum = BOARD_TYPE_MAP.get(board_type, BoardType.SQUARE8)
    encoder = build_encoder(board_type_enum, encoder_version=encoder_version)

    # Process new games from each file
    all_features = []
    all_globals = []
    all_values = []
    all_values_mp = []
    all_num_players = []
    all_policy_indices = []
    all_policy_values = []
    all_move_numbers = []
    all_total_game_moves = []
    all_phases = []
    all_processed_ids = []

    for jsonl_path in jsonl_files:
        (
            features, globals_vec, values, values_mp, np_list,
            policy_idx, policy_val, move_nums, total_moves, phases,
            processed_ids, extracted
        ) = process_new_games_from_jsonl(
            jsonl_path, new_game_ids, encoder, board_type_enum, board_type,
            num_players, sample_every, history_length
        )

        if extracted > 0:
            all_features.extend(features)
            all_globals.extend(globals_vec)
            all_values.extend(values)
            all_values_mp.extend(values_mp)
            all_num_players.extend(np_list)
            all_policy_indices.extend(policy_idx)
            all_policy_values.extend(policy_val)
            all_move_numbers.extend(move_nums)
            all_total_game_moves.extend(total_moves)
            all_phases.extend(phases)
            all_processed_ids.extend(processed_ids)
            stats.positions_extracted += extracted
            stats.games_exported += len(processed_ids)

            logger.info(f"  Processed {len(processed_ids)} games, {extracted} positions from {jsonl_path.name}")

    # Save as shard if we have data
    if all_features:
        features_arr = np.stack(all_features, axis=0).astype(np.float32)
        globals_arr = np.stack(all_globals, axis=0).astype(np.float32)
        values_arr = np.array(all_values, dtype=np.float32)
        values_mp_arr = np.stack(all_values_mp, axis=0).astype(np.float32)
        num_players_arr = np.array(all_num_players, dtype=np.int32)
        policy_indices_arr = np.array(all_policy_indices, dtype=object)
        policy_values_arr = np.array(all_policy_values, dtype=object)
        move_numbers_arr = np.array(all_move_numbers, dtype=np.int32)
        total_game_moves_arr = np.array(all_total_game_moves, dtype=np.int32)
        phases_arr = np.array(all_phases, dtype=object)

        shard_path = exporter.save_shard(
            features_arr, globals_arr, values_arr, values_mp_arr,
            num_players_arr, policy_indices_arr, policy_values_arr,
            move_numbers_arr, total_game_moves_arr, phases_arr,
            all_processed_ids, "incremental_export"
        )

        stats.shards_created = 1
        logger.info(f"Saved {stats.positions_extracted} positions to {shard_path}")

    stats.export_time_seconds = time.time() - start_time
    logger.info(f"Incremental export completed in {stats.export_time_seconds:.1f}s")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Incremental NPZ export - only processes new games"
    )
    parser.add_argument(
        "--board-type",
        type=str,
        required=True,
        choices=["square8", "square19", "hexagonal", "hex8"],
        help="Board type",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        required=True,
        choices=[2, 3, 4],
        help="Number of players",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=5,
        help="Sample every Nth position (default: 5)",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=3,
        help="Number of history frames (default: 3)",
    )
    parser.add_argument(
        "--encoder-version",
        type=str,
        default="v2",
        choices=["v2", "v3"],
        help="Encoder version for hex boards (default: v2)",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing shards without exporting",
    )
    parser.add_argument(
        "--merge-output",
        type=str,
        help="Output path for merged NPZ (with --merge-only)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show export statistics only",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset tracking (will re-export everything)",
    )

    args = parser.parse_args()

    exporter = get_incremental_exporter(args.board_type, args.num_players)

    if args.stats:
        stats = exporter.get_stats()
        print(f"\nExport Statistics for {args.board_type}_{args.num_players}p:")
        print(f"  Total exported games: {stats['total_exported_games']}")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Number of shards: {stats['num_shards']}")
        return

    if args.reset:
        confirm = input("Are you sure you want to reset tracking? This will re-export everything. (y/N): ")
        if confirm.lower() == "y":
            exporter.tracker.clear()
            print("Tracking data cleared.")
        return

    if args.merge_only:
        output_path = args.merge_output
        if not output_path:
            output_path = f"data/training/merged_{args.board_type}_{args.num_players}p.npz"

        samples = exporter.merge_to_npz(Path(output_path))
        print(f"Merged {samples} samples to {output_path}")
        return

    # Run incremental export
    stats = run_incremental_export(
        board_type=args.board_type,
        num_players=args.num_players,
        sample_every=args.sample_every,
        history_length=args.history_length,
        encoder_version=args.encoder_version,
    )

    print(f"\nIncremental Export Complete:")
    print(f"  Total games seen: {stats.total_games_seen}")
    print(f"  New games found: {stats.new_games_found}")
    print(f"  Games exported: {stats.games_exported}")
    print(f"  Positions extracted: {stats.positions_extracted}")
    print(f"  Export time: {stats.export_time_seconds:.1f}s")


if __name__ == "__main__":
    main()
