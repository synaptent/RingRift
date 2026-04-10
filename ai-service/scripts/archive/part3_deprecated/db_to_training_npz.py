#!/usr/bin/env python
"""Export training data from SQLite game databases to NPZ format.

DEPRECATED: Use export_replay_dataset.py instead, which is faster (parallel by default)
and has more features. This script is kept for legacy compatibility.

Recommended:
    python scripts/export_replay_dataset.py \
        --db data/selfplay/hex8_policy_c/games.db \
        --output data/training/hex8_2p_export.npz \
        --board-type hex8 \
        --num-players 2

Uses GameReplayDB.get_state_at_move() to properly reconstruct states,
then encodes them for neural network training.

Legacy usage:
    python scripts/db_to_training_npz.py \
        --db data/selfplay/hex8_policy_c/games.db \
        --output data/training/hex8_2p_export.npz \
        --board-type hex8 \
        --num-players 2
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("RINGRIFT_FORCE_CPU", "1")

from app.training.canonical_sources import enforce_canonical_sources
from app.training.export_core import compute_value
from scripts.lib.cli import BOARD_TYPE_MAP
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("db_to_training_npz")


class HexEncoderWrapper:
    """Wrapper to give hex encoders a consistent interface with frame stacking.

    Supports both HexStateEncoder (v2, 10 base channels → 40 total) and
    HexStateEncoderV3 (v3, 16 base channels → 64 total) with proper frame stacking
    for compatibility with HexNeuralNet_v2 and HexNeuralNet_v3 respectively.
    """

    def __init__(self, encoder, board_size: int = 25, history_length: int = 3):
        self._encoder = encoder
        self._board_size = board_size
        self._history_length = history_length
        self._history_frames: list[np.ndarray] = []

    def encode_state(self, state):
        """Extract features with frame stacking for NN compatibility."""
        features, globals_vec = self._encoder.encode_state(state)

        # Build stacked features (current + history frames)
        hist = self._history_frames[::-1][:self._history_length]
        while len(hist) < self._history_length:
            hist.append(np.zeros_like(features))

        stacked = np.concatenate([features, *hist], axis=0)

        # Update history for next call
        self._history_frames.append(features.copy())
        if len(self._history_frames) > self._history_length + 1:
            self._history_frames.pop(0)

        return stacked.astype(np.float32), globals_vec.astype(np.float32)

    def reset_history(self):
        """Reset history frames for a new game."""
        self._history_frames = []


def get_encoder(
    board_type: str,
    num_players: int,
    hex_encoder_version: str = "v3",
    history_length: int = 3,
):
    """Get the appropriate state encoder for the board type.

    Args:
        board_type: Board type string (square8, square19, hex8, hexagonal)
        num_players: Number of players
        hex_encoder_version: For hex boards - "v2" (40ch) or "v3" (64ch, default)
        history_length: Number of history frames (default 3 for 4 total frames)

    Returns:
        Encoder with consistent interface (encode_state returning features, globals)

    Model compatibility:
        - v2 encoder (10 base × 4 frames = 40ch) → HexNeuralNet_v2
        - v3 encoder (16 base × 4 frames = 64ch) → HexNeuralNet_v3 (recommended)
    """
    from app.ai.base import AIConfig
    from app.ai.neural_net import NeuralNetAI
    from app.models import BoardType
    from app.training.encoding import (
        P_HEX,
        POLICY_SIZE_HEX8,
        HexStateEncoder,
        HexStateEncoderV3,
    )

    bt = BOARD_TYPE_MAP.get(board_type, BoardType.SQUARE8)

    if bt in (BoardType.HEXAGONAL, BoardType.HEX8):
        # Select hex encoder based on version
        if bt == BoardType.HEX8:
            board_size = 9
            policy_size = POLICY_SIZE_HEX8
        else:
            board_size = 25
            policy_size = P_HEX

        if hex_encoder_version == "v2":
            # HexStateEncoder (v2): 10 base channels → 40 total for HexNeuralNet_v2
            hex_encoder = HexStateEncoder(board_size=board_size, policy_size=policy_size)
            logger.info(f"Using HexStateEncoder v2 (10 base ch × 4 frames = 40 total)")
        else:
            # HexStateEncoderV3: 16 base channels → 64 total for HexNeuralNet_v3
            hex_encoder = HexStateEncoderV3(board_size=board_size, policy_size=policy_size)
            logger.info(f"Using HexStateEncoderV3 (16 base ch × 4 frames = 64 total)")

        return HexEncoderWrapper(hex_encoder, board_size=board_size, history_length=history_length)
    else:
        # For square boards, use NeuralNetAI which handles frame stacking internally
        config = AIConfig(
            difficulty=5,
            think_time=0,
            randomness=0.0,
            rngSeed=None,
            heuristic_profile_id=None,
            nn_model_id=None,
            use_neural_net=True,
        )
        encoder = NeuralNetAI(player_number=1, config=config)
        encoder.board_size = 8 if bt == BoardType.SQUARE8 else 19
        return encoder


def get_encoder_metadata(
    board_type: str,
    hex_encoder_version: str = "v3",
    history_length: int = 3,
) -> dict:
    """Get metadata about the encoder configuration for NPZ files.

    This metadata helps ensure training data is used with compatible models.

    Returns:
        Dict with keys:
        - encoder_type: str (e.g., "hex_v2", "hex_v3", "square")
        - base_channels: int (channels per frame before history stacking)
        - in_channels: int (total input channels = base × (history + 1))
        - board_type: str (board type string)
        - spatial_size: int (H=W dimension for the board)
        - data_schema_version: str (schema version for future changes)
    """
    from app.models import BoardType
    from app.ai.neural_net import get_spatial_size_for_board

    bt = BOARD_TYPE_MAP.get(board_type, BoardType.SQUARE8)
    frames = history_length + 1  # Current + history
    spatial_size = get_spatial_size_for_board(bt)

    base_metadata = {
        "board_type": board_type,
        "spatial_size": spatial_size,
        "data_schema_version": "v2",
    }

    if bt in (BoardType.HEXAGONAL, BoardType.HEX8):
        if hex_encoder_version == "v3":
            return {
                **base_metadata,
                "encoder_type": "hex_v3",
                "base_channels": 16,
                "in_channels": 16 * frames,
            }
        else:
            return {
                **base_metadata,
                "encoder_type": "hex_v2",
                "base_channels": 10,
                "in_channels": 10 * frames,
            }
    else:
        # Square boards use 14 base channels
        return {
            **base_metadata,
            "encoder_type": "square",
            "base_channels": 14,
            "in_channels": 14 * frames,
        }


def export_db_to_npz(
    db_path: Path,
    output_path: Path,
    board_type: str,
    num_players: int,
    sample_every: int = 3,
    max_games: int | None = None,
    max_positions: int = 500000,
    hex_encoder_version: str = "v3",
    history_length: int = 3,
    legacy_injection: bool = False,
) -> int:
    """Export training positions from a game database.

    Args:
        legacy_injection: If True, auto-inject missing bookkeeping moves
            (NO_TERRITORY_ACTION, etc.) to heal non-canonical recordings.
            WARNING: Violates RR-CANON-R075 - use only for old data recovery.

    Returns number of positions exported.
    """
    from app.db.game_replay import GameReplayDB

    logger.info(f"Loading database: {db_path}")

    # Get encoder metadata for NPZ validation
    encoder_metadata = get_encoder_metadata(
        board_type=board_type,
        hex_encoder_version=hex_encoder_version,
        history_length=history_length,
    )
    logger.info(
        f"Encoder metadata: {encoder_metadata['encoder_type']} "
        f"({encoder_metadata['in_channels']} channels = "
        f"{encoder_metadata['base_channels']} base × {history_length + 1} frames)"
    )

    # When legacy_injection is enabled, open DB with enforce_canonical_history=False
    # to allow phase injection during replay
    enforce_canonical = not legacy_injection
    replay = GameReplayDB(str(db_path), enforce_canonical_history=enforce_canonical)

    if legacy_injection:
        logger.warning(
            "LEGACY INJECTION ENABLED: Auto-injecting missing phase transitions. "
            "This violates RR-CANON-R075 - data should be marked as 'healed' in registry."
        )

    # Get game IDs - handle both old (move_count) and new (total_moves) schema
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check which column name exists
    cursor.execute("PRAGMA table_info(games)")
    columns = {row[1] for row in cursor.fetchall()}
    moves_col = "total_moves" if "total_moves" in columns else "move_count"

    # Build filter query for board type and num players
    filter_conditions = [f"winner IS NOT NULL", f"{moves_col} > 10"]
    filter_params = []

    if board_type:
        filter_conditions.append("board_type = ?")
        filter_params.append(board_type)

    if num_players:
        filter_conditions.append("num_players = ?")
        filter_params.append(num_players)

    where_clause = " AND ".join(filter_conditions)
    cursor.execute(f"""
        SELECT game_id, winner, {moves_col}
        FROM games
        WHERE {where_clause}
        ORDER BY game_id
    """, filter_params)
    games = cursor.fetchall()
    conn.close()

    if max_games:
        games = games[:max_games]

    logger.info(f"Found {len(games)} games with winners")

    encoder = get_encoder(
        board_type, num_players,
        hex_encoder_version=hex_encoder_version,
        history_length=history_length,
    )

    all_features = []
    all_globals = []
    all_values = []
    all_move_numbers = []
    all_total_moves = []
    all_num_players = []

    processed_games = 0
    for game_id, winner, total_moves in games:
        if len(all_features) >= max_positions:
            break

        # Reset history frames at start of each game (for hex encoders)
        if hasattr(encoder, 'reset_history'):
            encoder.reset_history()

        try:
            # Sample positions throughout the game
            for move_num in range(0, total_moves, sample_every):
                if len(all_features) >= max_positions:
                    break

                state = replay.get_state_at_move(game_id, move_num)
                if state is None:
                    continue

                # Encode state - handle both encoder types
                if hasattr(encoder, 'encode_state'):
                    encoded = encoder.encode_state(state)
                else:
                    # NeuralNetAI uses _extract_features
                    features, global_features = encoder._extract_features(state)
                    encoded = (features, global_features)
                if encoded is None:
                    continue

                features, global_features = encoded

                # Compute value target using consolidated function
                value = compute_value(winner, state.current_player)

                all_features.append(features)
                all_globals.append(global_features)
                all_values.append(value)
                all_move_numbers.append(move_num)
                all_total_moves.append(total_moves)
                all_num_players.append(num_players)

            processed_games += 1
            if processed_games % 100 == 0:
                logger.info(f"Processed {processed_games}/{len(games)} games, {len(all_features)} positions")

        except Exception as e:
            logger.warning(f"Error processing game {game_id}: {e}")
            continue

    if not all_features:
        logger.error("No positions extracted!")
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
    # Phase 5: Get policy size from constants for validation metadata
    from app.ai.neural_net.constants import get_policy_size_for_board
    from app.models import BoardType as BT
    bt = BT[board_type.upper()] if isinstance(board_type, str) else board_type
    policy_size = get_policy_size_for_board(bt)

    np.savez_compressed(
        output_path,
        features=features_arr,
        globals=globals_arr,
        values=values_arr,
        move_numbers=move_numbers_arr,
        total_game_moves=total_moves_arr,
        num_players=num_players_arr,
        board_type=board_type,
        source_db=str(db_path),
        history_length=np.asarray(int(history_length)),
        # Encoder metadata for model compatibility validation
        encoder_type=np.asarray(encoder_metadata["encoder_type"]),
        base_channels=np.asarray(encoder_metadata["base_channels"]),
        in_channels=np.asarray(encoder_metadata["in_channels"]),
        spatial_size=np.asarray(encoder_metadata.get("spatial_size", 0)),
        data_schema_version=np.asarray(encoder_metadata.get("data_schema_version", "v1")),
        # Phase 5 Metadata: Additional fields for training compatibility validation
        policy_size=np.asarray(int(policy_size)),
        policies_normalized=np.asarray(True),
        export_version=np.asarray("2.0"),
    )

    logger.info(f"Saved {len(all_features)} positions to {output_path}")
    logger.info(f"  Features shape: {features_arr.shape}")
    logger.info(f"  Values range: [{values_arr.min():.2f}, {values_arr.max():.2f}]")

    return len(all_features)


def main():
    import warnings
    warnings.warn(
        "db_to_training_npz.py is deprecated. Use export_replay_dataset.py instead "
        "(parallel by default, 10-20x faster).",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "DEPRECATED: Use 'python scripts/export_replay_dataset.py' instead for faster parallel exports"
    )
    parser = argparse.ArgumentParser(description="Export game DB to training NPZ")
    parser.add_argument("--db", type=str, required=True, help="Path to game database")
    parser.add_argument("--output", type=str, required=True, help="Output NPZ path")
    parser.add_argument("--board-type", type=str, required=True, help="Board type")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--sample-every", type=int, default=3, help="Sample every N moves")
    parser.add_argument("--max-games", type=int, default=None, help="Max games to process")
    parser.add_argument("--max-positions", type=int, default=500000, help="Max positions")
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help="Allow exporting from non-canonical DBs for legacy/experimental runs.",
    )
    parser.add_argument(
        "--allow-pending-gate",
        action="store_true",
        help="Allow DBs marked pending_gate in TRAINING_DATA_REGISTRY.md.",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to TRAINING_DATA_REGISTRY.md (default: repo root)",
    )
    parser.add_argument(
        "--hex-encoder-version",
        type=str,
        choices=["v2", "v3"],
        default="v3",
        help="Hex encoder version: v2 (40ch for HexNeuralNet_v2) or v3 (64ch for HexNeuralNet_v3, default)",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=3,
        help="Number of history frames to stack (default: 3 for 4 total frames)",
    )
    parser.add_argument(
        "--legacy-injection",
        action="store_true",
        help=(
            "Enable legacy phase injection for non-canonical recordings. "
            "Auto-injects missing bookkeeping moves (NO_TERRITORY_ACTION, etc.) "
            "to heal games recorded before explicit phase transitions were required. "
            "WARNING: Violates RR-CANON-R075 - use only for old data recovery."
        ),
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1

    # Use central canonical source validation
    allowed_statuses = ["canonical", "pending_gate"] if args.allow_pending_gate else ["canonical"]
    enforce_canonical_sources(
        [db_path],
        registry_path=Path(args.registry) if args.registry else None,
        allowed_statuses=allowed_statuses,
        allow_noncanonical=bool(args.allow_noncanonical),
        error_prefix="db-to-training-npz",
    )

    output_path = Path(args.output)

    count = export_db_to_npz(
        db_path=db_path,
        output_path=output_path,
        board_type=args.board_type,
        num_players=args.num_players,
        sample_every=args.sample_every,
        max_games=args.max_games,
        max_positions=args.max_positions,
        hex_encoder_version=args.hex_encoder_version,
        history_length=args.history_length,
        legacy_injection=args.legacy_injection,
    )

    return 0 if count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
