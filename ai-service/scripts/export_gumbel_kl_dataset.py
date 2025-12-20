#!/usr/bin/env python3
"""Export Gumbel MCTS selfplay data to NPZ with soft policy targets for KL distillation.

This script directly exports JSONL files containing mcts_policy (visit distributions)
to NPZ format suitable for training with KL divergence loss.

The key difference from export_replay_dataset.py is that policy targets are
SOFT distributions from MCTS rather than one-hot encoded.

Usage:
    python scripts/export_gumbel_kl_dataset.py \
        --input data/gumbel_selfplay/sq8_kl_combined.jsonl \
        --output data/training/sq8_kl_distill.npz \
        --board-type square8 \
        --num-players 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.game_engine import GameEngine
from app.models import BoardType, GameState, Move, MoveType, Position
from app.training.encoding import get_encoder_for_board_type
from app.training.initial_state import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BOARD_TYPE_MAP = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hexagonal": BoardType.HEXAGONAL,
}


def parse_position(pos_dict: dict[str, Any] | None) -> Position | None:
    """Parse position dict to Position object."""
    if pos_dict is None:
        return None
    return Position(
        x=pos_dict.get("x", 0),
        y=pos_dict.get("y", 0),
        z=pos_dict.get("z"),
    )


def parse_move(move_dict: dict[str, Any], move_num: int) -> Move:
    """Parse move dict from JSONL to Move object."""
    move_type_str = move_dict.get("type", "place_ring")
    try:
        move_type = MoveType(move_type_str)
    except ValueError:
        move_type = MoveType.PLACE_RING

    return Move(
        id=f"move-{move_num}",
        type=move_type,
        player=move_dict.get("player", 1),
        from_pos=parse_position(move_dict.get("from")),
        to=parse_position(move_dict.get("to")),
        capture_target=parse_position(move_dict.get("capture_target")),
        timestamp=None,
        think_time=0,
        move_number=move_num,
    )


def value_from_winner(winner: int | None, perspective: int, num_players: int) -> float:
    """Compute value target from game outcome."""
    if winner is None:
        return 0.0
    if num_players == 2:
        return 1.0 if winner == perspective else -1.0
    # Multi-player: simplified
    return 1.0 if winner == perspective else -0.5


def export_gumbel_kl_dataset(
    input_path: str,
    output_path: str,
    board_type: BoardType,
    num_players: int,
    history_length: int = 3,
    feature_version: int = 2,
    max_games: int | None = None,
    sample_every: int = 1,
) -> dict[str, Any]:
    """Export Gumbel MCTS selfplay data to NPZ with soft policy targets.

    Returns:
        Statistics dict
    """
    encoder = get_encoder_for_board_type(board_type, feature_version=feature_version)

    features_list: list[np.ndarray] = []
    globals_list: list[np.ndarray] = []
    values_list: list[float] = []
    policy_indices_list: list[np.ndarray] = []
    policy_values_list: list[np.ndarray] = []

    games_processed = 0
    samples_created = 0
    moves_with_policy = 0
    moves_without_policy = 0

    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            if max_games and games_processed >= max_games:
                break

            line = line.strip()
            if not line:
                continue

            try:
                game = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON: {e}")
                continue

            # Filter by board type and player count
            game_board = game.get("board_type", "square8")
            game_players = game.get("num_players", 2)

            if game_board != board_type.value:
                continue
            if game_players != num_players:
                continue

            winner = game.get("winner")
            moves_data = game.get("moves", [])

            # Create initial state
            initial_state_dict = game.get("initial_state")
            if initial_state_dict:
                try:
                    state = GameState.model_validate(initial_state_dict)
                except Exception:
                    state = create_initial_state(board_type, num_players)
            else:
                state = create_initial_state(board_type, num_players)

            # History buffer for stacking
            history_buffer: list[np.ndarray] = []

            for move_idx, move_dict in enumerate(moves_data):
                mcts_policy = move_dict.get("mcts_policy")

                # Skip moves without MCTS policy
                if not mcts_policy:
                    moves_without_policy += 1
                    # Still need to apply the move to advance state
                    try:
                        move = parse_move(move_dict, move_idx + 1)
                        state = GameEngine.apply_move(state, move)
                    except Exception:
                        break
                    continue

                moves_with_policy += 1

                # Sample every N moves
                if move_idx % sample_every != 0:
                    try:
                        move = parse_move(move_dict, move_idx + 1)
                        state = GameEngine.apply_move(state, move)
                    except Exception:
                        break
                    continue

                perspective = state.current_player

                # Encode current state
                try:
                    features = encoder.encode_state(state, perspective)
                    globals_vec = encoder.encode_globals(state, perspective)
                except Exception as e:
                    logger.debug(f"Encoding failed: {e}")
                    try:
                        move = parse_move(move_dict, move_idx + 1)
                        state = GameEngine.apply_move(state, move)
                    except Exception:
                        break
                    continue

                # Update history buffer
                history_buffer.append(features)
                if len(history_buffer) > history_length:
                    history_buffer.pop(0)

                # Stack history frames
                while len(history_buffer) < history_length:
                    # Pad with zeros at start
                    history_buffer.insert(0, np.zeros_like(features))

                stacked = np.concatenate(history_buffer, axis=0)

                # Parse MCTS policy as soft targets
                indices = []
                probs = []
                for idx_str, prob in mcts_policy.items():
                    try:
                        idx = int(idx_str)
                        if prob > 1e-6:  # Skip negligible probabilities
                            indices.append(idx)
                            probs.append(float(prob))
                    except ValueError:
                        continue

                if not indices:
                    try:
                        move = parse_move(move_dict, move_idx + 1)
                        state = GameEngine.apply_move(state, move)
                    except Exception:
                        break
                    continue

                # Normalize probabilities
                total = sum(probs)
                if total > 0:
                    probs = [p / total for p in probs]

                # Compute value target
                value = value_from_winner(winner, perspective, num_players)

                # Add sample
                features_list.append(stacked)
                globals_list.append(globals_vec)
                values_list.append(value)
                policy_indices_list.append(np.array(indices, dtype=np.int32))
                policy_values_list.append(np.array(probs, dtype=np.float32))
                samples_created += 1

                # Apply move to advance state
                try:
                    move = parse_move(move_dict, move_idx + 1)
                    state = GameEngine.apply_move(state, move)
                except Exception as e:
                    logger.debug(f"Move application failed: {e}")
                    break

            games_processed += 1
            if games_processed % 100 == 0:
                logger.info(f"Processed {games_processed} games, {samples_created} samples")

    if not features_list:
        raise ValueError("No samples created - check input data and filters")

    # Stack arrays
    features_arr = np.stack(features_list, axis=0).astype(np.float32)
    globals_arr = np.stack(globals_list, axis=0).astype(np.float32)
    values_arr = np.array(values_list, dtype=np.float32)
    policy_indices_arr = np.array(policy_indices_list, dtype=object)
    policy_values_arr = np.array(policy_values_list, dtype=object)

    # Save NPZ
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        features=features_arr,
        globals=globals_arr,
        values=values_arr,
        policy_indices=policy_indices_arr,
        policy_values=policy_values_arr,
        board_type=np.asarray(board_type.value),
        board_size=np.asarray(int(features_arr.shape[-1])),
        history_length=np.asarray(int(history_length)),
        feature_version=np.asarray(int(feature_version)),
        policy_encoding=np.asarray("soft_kl"),
    )

    stats = {
        "games_processed": games_processed,
        "samples_created": samples_created,
        "moves_with_policy": moves_with_policy,
        "moves_without_policy": moves_without_policy,
        "feature_shape": features_arr.shape,
        "output_path": output_path,
    }

    logger.info("=" * 60)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Games processed: {games_processed}")
    logger.info(f"Samples created: {samples_created}")
    logger.info(f"Moves with MCTS policy: {moves_with_policy}")
    logger.info(f"Moves without MCTS policy: {moves_without_policy}")
    logger.info(f"Feature shape: {features_arr.shape}")
    logger.info(f"Output: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Export Gumbel MCTS selfplay to NPZ with soft policy targets"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input JSONL file with Gumbel MCTS games"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output NPZ file path"
    )
    parser.add_argument(
        "--board-type", "-b",
        choices=["square8", "square19", "hexagonal"],
        default="square8",
        help="Board type to filter for"
    )
    parser.add_argument(
        "--num-players", "-p",
        type=int, choices=[2, 3, 4], default=2,
        help="Number of players to filter for"
    )
    parser.add_argument(
        "--history-length",
        type=int, default=3,
        help="History frames to stack"
    )
    parser.add_argument(
        "--feature-version",
        type=int, default=2,
        help="Feature encoding version"
    )
    parser.add_argument(
        "--max-games",
        type=int, default=None,
        help="Maximum games to process"
    )
    parser.add_argument(
        "--sample-every",
        type=int, default=1,
        help="Sample every Nth move"
    )

    args = parser.parse_args()

    board_type = BOARD_TYPE_MAP[args.board_type]

    export_gumbel_kl_dataset(
        input_path=args.input,
        output_path=args.output,
        board_type=board_type,
        num_players=args.num_players,
        history_length=args.history_length,
        feature_version=args.feature_version,
        max_games=args.max_games,
        sample_every=args.sample_every,
    )


if __name__ == "__main__":
    main()
