#!/usr/bin/env python
"""Convert JSONL selfplay data to NPZ format for neural network training.

This script converts game records from JSONL format (produced by run_self_play_soak.py)
into compressed NumPy arrays (.npz) suitable for direct neural network training.

The script replays each game from its initial_state using the moves list,
extracting proper 56-channel features at each position using NeuralNetAI's
feature extraction (matching the format expected by app.training.train).

Output NPZ format (compatible with train.py):
- features: (N, 56, H, W) float32 - Full feature channels
- globals: (N, 20) float32 - Global features
- values: (N,) float32 - Game outcomes (-1 to +1)
- policy_indices: (N,) object - Sparse action indices per sample
- policy_values: (N,) object - Sparse action probabilities per sample
- move_numbers: (N,) int32 - Move index within game
- total_game_moves: (N,) int32 - Total moves in source game
- phases: (N,) object - Game phase at each position
- values_mp: (N, 4) float32 - Multi-player value vectors
- num_players: (N,) int32 - Player count per sample

Usage:
    # Basic conversion (replays games properly)
    PYTHONPATH=. python scripts/jsonl_to_npz.py \\
        --input data/selfplay/combined_cloud.jsonl \\
        --output data/training/from_jsonl.npz \\
        --board-type square8 \\
        --num-players 2

    # With subsampling for large datasets
    PYTHONPATH=. python scripts/jsonl_to_npz.py \\
        --input-dir data/selfplay/ \\
        --output data/training/sampled.npz \\
        --max-games 1000 \\
        --sample-every 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure app.* imports resolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Force CPU to avoid MPS/OMP issues during batch conversion
os.environ.setdefault("RINGRIFT_FORCE_CPU", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from app.models import AIConfig, BoardType, GameState, Move, Position
from app.game_engine import GameEngine
from app.ai.neural_net import NeuralNetAI, INVALID_MOVE_INDEX
from app.rules.serialization import deserialize_game_state


BOARD_TYPE_MAP = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hexagonal": BoardType.HEXAGONAL,
}


def build_encoder(board_type: BoardType) -> NeuralNetAI:
    """Build a NeuralNetAI instance for feature extraction."""
    config = AIConfig(
        difficulty=5,
        think_time=0,
        randomness=0.0,
        rngSeed=None,
        heuristic_profile_id=None,
        nn_model_id=None,
        heuristic_eval_mode=None,
        use_neural_net=True,
    )
    encoder = NeuralNetAI(player_number=1, config=config)
    encoder.board_size = {
        BoardType.SQUARE8: 8,
        BoardType.SQUARE19: 19,
        BoardType.HEXAGONAL: 25,
    }.get(board_type, 8)
    return encoder


def parse_position(pos_dict: Optional[Dict[str, Any]]) -> Optional[Position]:
    """Parse position dict to Position object."""
    if pos_dict is None:
        return None
    return Position(
        x=pos_dict.get("x", 0),
        y=pos_dict.get("y", 0),
        z=pos_dict.get("z"),
    )


def parse_move(move_dict: Dict[str, Any]) -> Move:
    """Parse move dict from JSONL to Move object."""
    return Move(
        id=move_dict.get("id", "imported"),
        type=move_dict.get("type", "unknown"),
        player=move_dict.get("player", 1),
        from_pos=parse_position(move_dict.get("from_pos") or move_dict.get("from")),
        to=parse_position(move_dict.get("to")),
        capture_target=parse_position(move_dict.get("capture_target")),
        captured_stacks=move_dict.get("captured_stacks"),
        capture_chain=move_dict.get("capture_chain"),
        overtaken_rings=move_dict.get("overtaken_rings"),
        placed_on_stack=move_dict.get("placed_on_stack", False),
        placement_count=move_dict.get("placement_count"),
        stack_moved=move_dict.get("stack_moved"),
        minimum_distance=move_dict.get("minimum_distance"),
        # Required fields with defaults - use epoch time if timestamp missing
        timestamp=move_dict.get("timestamp") or "1970-01-01T00:00:00Z",
        thinkTime=move_dict.get("think_time", move_dict.get("thinkTime", 0)),
        moveNumber=move_dict.get("move_number", move_dict.get("moveNumber", 0)),
    )


def encode_state_with_history(
    encoder: NeuralNetAI,
    state: GameState,
    history_frames: List[np.ndarray],
    history_length: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode state with history frames, matching export_replay_dataset.py."""
    base_features, globals_vec = encoder._extract_features(state)

    # Stack history frames
    c, h, w = base_features.shape
    total_channels = c * (history_length + 1)
    stacked = np.zeros((total_channels, h, w), dtype=np.float32)

    # Current frame first
    stacked[:c] = base_features

    # Historical frames
    for i, hist_frame in enumerate(reversed(history_frames[-history_length:])):
        offset = c * (i + 1)
        if offset + c <= total_channels:
            stacked[offset:offset + c] = hist_frame

    return stacked, globals_vec


def value_from_final_ranking(
    final_state: GameState,
    perspective: int,
    num_players: int,
) -> float:
    """Compute value from final ranking (rank-aware for multiplayer)."""
    # Get rankings from final state
    rankings = []
    for p in final_state.players:
        score = p.eliminated_rings  # Higher eliminated = better
        rankings.append((p.player_number, score))

    # Sort by score descending
    rankings.sort(key=lambda x: -x[1])

    # Find perspective player's rank (0-indexed)
    rank = 0
    for i, (pnum, _) in enumerate(rankings):
        if pnum == perspective:
            rank = i
            break

    # Convert rank to value
    if num_players == 2:
        return 1.0 if rank == 0 else -1.0
    elif num_players == 3:
        return [1.0, 0.0, -1.0][rank]
    else:  # 4 players
        return [1.0, 0.33, -0.33, -1.0][rank]


def compute_multi_player_values(final_state: GameState, num_players: int) -> List[float]:
    """Compute value vector for all players."""
    values = []
    for p in range(1, 5):  # Always 4 slots
        if p <= num_players:
            values.append(value_from_final_ranking(final_state, p, num_players))
        else:
            values.append(0.0)
    return values


@dataclass
class ConversionStats:
    """Statistics from conversion."""
    files_processed: int = 0
    games_processed: int = 0
    games_skipped_filter: int = 0
    games_skipped_no_data: int = 0
    games_skipped_error: int = 0
    positions_extracted: int = 0


def process_jsonl_file(
    filepath: Path,
    encoder: NeuralNetAI,
    board_type: BoardType,
    board_filter: Optional[str],
    players_filter: Optional[int],
    max_games: Optional[int],
    sample_every: int,
    history_length: int,
    current_games: int,
) -> Tuple[
    List[np.ndarray],  # features
    List[np.ndarray],  # globals
    List[float],       # values
    List[np.ndarray],  # values_mp
    List[int],         # num_players
    List[np.ndarray],  # policy_indices
    List[np.ndarray],  # policy_values
    List[int],         # move_numbers
    List[int],         # total_game_moves
    List[str],         # phases
    ConversionStats,
]:
    """Process a single JSONL file and extract training data."""
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

    stats = ConversionStats()
    games_in_file = 0

    with open(filepath, "r") as f:
        for line in f:
            if max_games and (current_games + games_in_file) >= max_games:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.games_skipped_error += 1
                continue

            # Apply filters
            board_type_str = record.get("board_type", "square8")
            num_players = record.get("num_players", 2)

            if board_filter and board_type_str != board_filter:
                stats.games_skipped_filter += 1
                continue
            if players_filter and num_players != players_filter:
                stats.games_skipped_filter += 1
                continue

            # Check required data
            initial_state_dict = record.get("initial_state")
            moves_list = record.get("moves", [])

            if not initial_state_dict or not moves_list:
                stats.games_skipped_no_data += 1
                continue

            try:
                # Parse initial state
                initial_state = deserialize_game_state(initial_state_dict)

                # Parse moves
                moves = [parse_move(m) for m in moves_list]
                total_moves = len(moves)

                # Replay game and extract features
                current_state = initial_state
                history_frames: List[np.ndarray] = []

                # Compute final state for value targets
                # Replay until we hit an error, use that as "final" state
                final_state = initial_state
                moves_succeeded = 0
                for move in moves:
                    try:
                        final_state = GameEngine.apply_move(final_state, move)
                        moves_succeeded += 1
                    except Exception:
                        # Stop at first error - state is now desynced
                        break

                if moves_succeeded < 10:
                    # Need at least 10 successful moves to have meaningful data
                    raise ValueError(f"Only {moves_succeeded}/{len(moves)} moves succeeded")

                # Precompute multi-player values
                values_vec = np.array(
                    compute_multi_player_values(final_state, num_players),
                    dtype=np.float32,
                )

                # Only process up to moves_succeeded moves
                for move_idx, move in enumerate(moves[:moves_succeeded]):
                    # Sample every N moves
                    if sample_every > 1 and (move_idx % sample_every) != 0:
                        # Still need to apply move and update history
                        base_features, _ = encoder._extract_features(current_state)
                        history_frames.append(base_features)
                        if len(history_frames) > history_length + 1:
                            history_frames.pop(0)
                        current_state = GameEngine.apply_move(current_state, move)
                        continue

                    # Encode state with history
                    stacked, globals_vec = encode_state_with_history(
                        encoder, current_state, history_frames, history_length
                    )

                    # Update history
                    base_features, _ = encoder._extract_features(current_state)
                    history_frames.append(base_features)
                    if len(history_frames) > history_length + 1:
                        history_frames.pop(0)

                    # Encode action (sparse)
                    action_idx = encoder.encode_move(move, current_state.board)
                    if action_idx == INVALID_MOVE_INDEX:
                        # Skip invalid moves
                        current_state = GameEngine.apply_move(current_state, move)
                        continue

                    # Value from perspective of current player
                    value = value_from_final_ranking(
                        final_state, current_state.current_player, num_players
                    )

                    # Phase string
                    phase_str = (
                        str(current_state.current_phase.value)
                        if hasattr(current_state.current_phase, "value")
                        else str(current_state.current_phase)
                    )

                    # Store sample
                    features_list.append(stacked)
                    globals_list.append(globals_vec)
                    values_list.append(float(value))
                    values_mp_list.append(values_vec.copy())
                    num_players_list.append(num_players)
                    policy_indices_list.append(np.array([action_idx], dtype=np.int32))
                    policy_values_list.append(np.array([1.0], dtype=np.float32))
                    move_numbers_list.append(move_idx)
                    total_game_moves_list.append(moves_succeeded)
                    phases_list.append(phase_str)

                    stats.positions_extracted += 1

                    # Apply move for next iteration
                    current_state = GameEngine.apply_move(current_state, move)

                games_in_file += 1
                stats.games_processed += 1

            except Exception as e:
                stats.games_skipped_error += 1
                if stats.games_skipped_error <= 3:
                    import traceback
                    logger.warning(f"Error processing game: {e}")
                    logger.warning(traceback.format_exc())
                continue

    stats.files_processed = 1
    return (
        features_list, globals_list, values_list, values_mp_list,
        num_players_list, policy_indices_list, policy_values_list,
        move_numbers_list, total_game_moves_list, phases_list, stats
    )


def find_jsonl_files(input_path: Path, recursive: bool = True) -> List[Path]:
    """Find all JSONL files in the given path."""
    if input_path.is_file():
        return [input_path]
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    return sorted(input_path.glob(pattern))


def convert_jsonl_to_npz(
    input_paths: List[Path],
    output_path: Path,
    board_type_str: str,
    players_filter: Optional[int] = None,
    max_games: Optional[int] = None,
    sample_every: int = 1,
    history_length: int = 3,
) -> ConversionStats:
    """Convert JSONL files to NPZ training data."""
    board_type = BOARD_TYPE_MAP.get(board_type_str, BoardType.SQUARE8)
    encoder = build_encoder(board_type)

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

    total_stats = ConversionStats()

    logger.info(f"Processing {len(input_paths)} JSONL files...")
    logger.info(f"Board type: {board_type_str}, Players: {players_filter or 'any'}")

    for i, filepath in enumerate(input_paths):
        if max_games and total_stats.games_processed >= max_games:
            break

        (features, globals_vec, values, values_mp, num_players,
         policy_indices, policy_values, move_numbers, total_moves,
         phases, stats) = process_jsonl_file(
            filepath=filepath,
            encoder=encoder,
            board_type=board_type,
            board_filter=board_type_str,
            players_filter=players_filter,
            max_games=max_games,
            sample_every=sample_every,
            history_length=history_length,
            current_games=total_stats.games_processed,
        )

        all_features.extend(features)
        all_globals.extend(globals_vec)
        all_values.extend(values)
        all_values_mp.extend(values_mp)
        all_num_players.extend(num_players)
        all_policy_indices.extend(policy_indices)
        all_policy_values.extend(policy_values)
        all_move_numbers.extend(move_numbers)
        all_total_game_moves.extend(total_moves)
        all_phases.extend(phases)

        total_stats.files_processed += stats.files_processed
        total_stats.games_processed += stats.games_processed
        total_stats.games_skipped_filter += stats.games_skipped_filter
        total_stats.games_skipped_no_data += stats.games_skipped_no_data
        total_stats.games_skipped_error += stats.games_skipped_error
        total_stats.positions_extracted += stats.positions_extracted

        if (i + 1) % 10 == 0 or i == len(input_paths) - 1:
            logger.info(
                f"Processed {i + 1}/{len(input_paths)} files, "
                f"{total_stats.games_processed} games, "
                f"{total_stats.positions_extracted} positions"
            )

    if not all_features:
        logger.warning("No training data extracted!")
        return total_stats

    # Convert to numpy arrays
    logger.info("Converting to numpy arrays...")
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

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {output_path}...")

    np.savez_compressed(
        output_path,
        features=features_arr,
        globals=globals_arr,
        values=values_arr,
        policy_indices=policy_indices_arr,
        policy_values=policy_values_arr,
        move_numbers=move_numbers_arr,
        total_game_moves=total_game_moves_arr,
        phases=phases_arr,
        values_mp=values_mp_arr,
        num_players=num_players_arr,
    )

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {output_size_mb:.2f} MB")

    return total_stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL selfplay data to NPZ format for NN training"
    )
    parser.add_argument(
        "--input",
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
        type=str,
        required=True,
        help="Output NPZ file path",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        required=True,
        choices=["square8", "square19", "hexagonal"],
        help="Board type (required for proper feature encoding)",
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
        help="Maximum number of games to process",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Use every Nth move as a training sample (default: 1 = all)",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=3,
        help="Number of historical frames to stack (default: 3)",
    )

    args = parser.parse_args()

    # Collect input files
    input_paths = []
    if args.input:
        input_paths.append(Path(args.input))
    if args.input_dir:
        input_paths.extend(find_jsonl_files(Path(args.input_dir)))

    if not input_paths:
        parser.error("Must specify --input or --input-dir")

    logger.info(f"Found {len(input_paths)} JSONL files")

    stats = convert_jsonl_to_npz(
        input_paths=input_paths,
        output_path=Path(args.output),
        board_type_str=args.board_type,
        players_filter=args.num_players,
        max_games=args.max_games,
        sample_every=args.sample_every,
        history_length=args.history_length,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats.files_processed}")
    logger.info(f"Games processed: {stats.games_processed}")
    logger.info(f"Games skipped (filter): {stats.games_skipped_filter}")
    logger.info(f"Games skipped (no data): {stats.games_skipped_no_data}")
    logger.info(f"Games skipped (error): {stats.games_skipped_error}")
    logger.info(f"Positions extracted: {stats.positions_extracted}")


if __name__ == "__main__":
    main()
