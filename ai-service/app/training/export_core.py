"""Core export utilities for training data generation.

This module consolidates common patterns across the 7+ export scripts:
- Value computation (rank-aware, binary, multiplayer)
- State encoding with history stacking
- NPZ dataset I/O operations
- Progress tracking

Usage:
    from app.training.export_core import (
        compute_value,  # Simple (winner, perspective) → float
        value_from_final_winner,  # GameState-based
        value_from_final_ranking,  # Rank-aware for multiplayer
        compute_multi_player_values,
        encode_state_with_history,
        NPZDatasetWriter,
    )
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from app.models.core import GameState

logger = logging.getLogger(__name__)


# =============================================================================
# Value Computation Functions
# =============================================================================


def compute_value(winner: int | None, perspective: int) -> float:
    """Compute scalar value from winner and perspective player.

    This is the fundamental value computation used across all export scripts.
    For GameState-based computation, use value_from_final_winner instead.

    Args:
        winner: Player number who won (1-indexed), 0 for draw, None for incomplete
        perspective: Player number (1-indexed) to compute value for

    Returns:
        +1.0 if perspective player won, -1.0 if lost, 0.0 if draw/incomplete
    """
    if winner is None or winner == 0:
        return 0.0
    if winner == perspective:
        return 1.0
    return -1.0


def value_from_final_winner(final_state: GameState, perspective: int) -> float:
    """Map final winner to a scalar value from the perspective of `perspective`.

    This is a convenience wrapper around compute_value() that extracts
    the winner from a GameState. For 2-player games.
    For multiplayer support, use value_from_final_ranking.

    Args:
        final_state: The completed game state with a winner attribute
        perspective: Player number (1-indexed) to compute value for

    Returns:
        +1.0 if perspective player won, -1.0 if lost, 0.0 if no winner
    """
    winner = getattr(final_state, "winner", None)
    return compute_value(winner, perspective)


def value_from_final_ranking(
    final_state: GameState,
    perspective: int,
    num_players: int,
) -> float:
    """Compute rank-aware value from final game state.

    Maps final ranking to a scalar value in [-1, +1] using linear interpolation:
      - 1st place: +1
      - Last place: -1
      - Intermediate positions: linearly interpolated

    Formula: value = 1 - 2 * (rank - 1) / (num_players - 1)
      - 2-player: 1st=+1, 2nd=-1
      - 3-player: 1st=+1, 2nd=0, 3rd=-1
      - 4-player: 1st=+1, 2nd=+0.333, 3rd=-0.333, 4th=-1

    Ranking is determined by eliminated_rings (more = better), with ties broken
    by territory_spaces.

    Args:
        final_state: The completed game state
        perspective: Player number (1-indexed) to compute value for
        num_players: Total number of players in the game

    Returns:
        Value in [-1, +1] representing expected outcome for this player
    """
    winner = getattr(final_state, "winner", None)

    # Handle incomplete games or draws
    if winner is None or not final_state.players:
        return 0.0

    # For 2-player games, use simple winner/loser logic
    if num_players == 2:
        if winner == perspective:
            return 1.0
        return -1.0

    # For multiplayer, compute ranking based on eliminated_rings (primary)
    # and territory_spaces (tiebreaker)
    player_scores = []
    for player in final_state.players:
        # Score: eliminated rings as primary, territory as tiebreaker
        # Higher score = better ranking
        score = (player.eliminated_rings, player.territory_spaces)
        player_scores.append((player.player_number, score))

    # Sort by score descending (best first)
    player_scores.sort(key=lambda x: x[1], reverse=True)

    # Find rank of perspective player (1-indexed)
    rank = 1
    for i, (player_num, _) in enumerate(player_scores):
        if player_num == perspective:
            rank = i + 1
            break

    # Linear interpolation: 1st=+1, last=-1, intermediate=interpolated
    # value = 1 - 2 * (rank - 1) / (num_players - 1)
    if num_players <= 1:
        return 0.0

    value = 1.0 - 2.0 * (rank - 1) / (num_players - 1)
    return float(value)


def compute_multi_player_values(
    final_state: GameState,
    num_players: int,
    max_players: int = 4,
) -> list[float]:
    """Compute value vector for all player positions.

    This is used with RingRiftCNN_MultiPlayer which outputs values for all
    players simultaneously instead of just the current player's perspective.

    Args:
        final_state: The completed game state
        num_players: Number of active players in the game (2, 3, or 4)
        max_players: Maximum players the model supports (default: 4)

    Returns:
        List of values of length max_players, where:
        - Active players (0 to num_players-1) have values in [-1, +1]
        - Inactive slots are filled with 0.0

    Examples:
        >>> # 2-player game where P1 wins
        >>> values = compute_multi_player_values(state, num_players=2)
        >>> # values = [1.0, -1.0, 0.0, 0.0]

        >>> # 3-player game ranking P2, P1, P3
        >>> values = compute_multi_player_values(state, num_players=3)
        >>> # values = [0.0, 1.0, -1.0, 0.0]  (P2=1st, P1=2nd, P3=3rd)
    """
    # Initialize with zeros for inactive slots
    values = [0.0] * max_players

    winner = getattr(final_state, "winner", None)

    # Handle incomplete games
    if winner is None or not final_state.players:
        return values

    # Compute ranking based on eliminated_rings and territory_spaces
    player_scores = []
    for player in final_state.players:
        score = (player.eliminated_rings, player.territory_spaces)
        player_scores.append((player.player_number, score))

    # Sort by score descending (best = rank 1)
    player_scores.sort(key=lambda x: x[1], reverse=True)

    # Build rank lookup: player_number -> rank (1-indexed)
    player_ranks: dict[int, int] = {}
    for rank, (player_num, _) in enumerate(player_scores, start=1):
        player_ranks[player_num] = rank

    # Compute value for each active player position
    for player in final_state.players:
        player_idx = player.player_number - 1  # 0-indexed for array
        if player_idx >= max_players:
            continue

        rank = player_ranks.get(player.player_number, num_players)

        if num_players <= 1:
            values[player_idx] = 0.0
        else:
            # Linear interpolation: 1st=+1, last=-1
            values[player_idx] = 1.0 - 2.0 * (rank - 1) / (num_players - 1)

    return values


# =============================================================================
# State Encoding Functions
# =============================================================================


def encode_state_with_history(
    encoder: Any,
    state: GameState,
    history_frames: list[np.ndarray],
    history_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a GameState + history into (stacked_features, globals_vec).

    This mirrors the stacking logic in NeuralNetAI.evaluate_batch /
    encode_state_for_model: current features followed by up to history_length
    previous feature frames, newest-first, padded with zeros as needed.

    For hex boards with attached _hex_encoder, uses the specialized encoder
    (HexStateEncoder or HexStateEncoderV3) for proper channel count.

    Args:
        encoder: NeuralNetAI instance or similar with _extract_features method
        state: Current game state to encode
        history_frames: List of previous feature frames (oldest first)
        history_length: Number of history frames to include

    Returns:
        Tuple of (stacked_features, globals_vec) as float32 arrays
    """
    # Check if we have a specialized hex encoder attached
    hex_encoder = getattr(encoder, "_hex_encoder", None)
    if hex_encoder is not None:
        # Use the hex-specific encoder
        features, globals_vec = hex_encoder.encode_state(state)
    else:
        # Use the internal feature extractor; this is stable tooling code.
        features, globals_vec = encoder._extract_features(state)

    hist = history_frames[::-1][:history_length]
    while len(hist) < history_length:
        hist.append(np.zeros_like(features))

    stacked = np.concatenate([features, *hist], axis=0)
    return stacked.astype(np.float32), globals_vec.astype(np.float32)


# =============================================================================
# NPZ Dataset I/O
# =============================================================================


class NPZDatasetWriter:
    """Unified NPZ dataset writer with consistent metadata handling.

    Provides a single interface for:
    - Writing new datasets
    - Appending to existing datasets
    - Loading existing datasets
    - Managing dataset metadata

    The output format is compatible with all training scripts.
    """

    # Standard array names in NPZ files
    STANDARD_ARRAYS = [
        "features",
        "globals",
        "values",
        "policy_indices",
        "policy_values",
    ]

    # Optional arrays that may be present
    OPTIONAL_ARRAYS = [
        "values_mp",
        "num_players",
        "move_numbers",
        "total_game_moves",
        "phases",
        "victory_types",
    ]

    # Standard metadata fields
    METADATA_FIELDS = [
        "board_type",
        "board_size",
        "history_length",
        "feature_version",
        "policy_encoding",
    ]

    @staticmethod
    def save(
        output_path: str | Path,
        features: np.ndarray,
        globals_: np.ndarray,
        values: np.ndarray,
        policy_indices: np.ndarray,
        policy_values: np.ndarray,
        *,
        board_type: str,
        board_size: int,
        history_length: int = 3,
        feature_version: int = 2,
        policy_encoding: str = "legacy_max_n",
        values_mp: np.ndarray | None = None,
        num_players: np.ndarray | None = None,
        move_numbers: np.ndarray | None = None,
        total_game_moves: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        victory_types: np.ndarray | None = None,
        archive_existing: bool = True,
    ) -> None:
        """Save a training dataset to NPZ format.

        Args:
            output_path: Destination file path
            features: Feature arrays, shape (N, C, H, W)
            globals_: Global feature vectors, shape (N, G)
            values: Value targets, shape (N,)
            policy_indices: Policy target indices
            policy_values: Policy target values
            board_type: Board type string (e.g., "square8")
            board_size: Board size (e.g., 8, 19, 25)
            history_length: Number of history frames
            feature_version: Feature encoding version
            policy_encoding: Policy encoding type
            values_mp: Multi-player values (optional)
            num_players: Number of players per sample (optional)
            move_numbers: Move number within game (optional)
            total_game_moves: Total moves in source game (optional)
            phases: Game phase strings (optional)
            victory_types: Victory type strings (optional)
            archive_existing: If True, archive existing file before overwriting
        """
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)

        # Archive existing file if present
        if archive_existing and output_path.exists():
            archive_path = output_path.with_suffix(
                f".archived_{time.strftime('%Y%m%d_%H%M%S')}.npz"
            )
            try:
                output_path.rename(archive_path)
                logger.info(f"Archived existing file to {archive_path}")
            except OSError as e:
                logger.warning(f"Failed to archive existing file: {e}")

        # Build save arguments
        save_kwargs: dict[str, Any] = {
            "features": features.astype(np.float32),
            "globals": globals_.astype(np.float32),
            "values": values.astype(np.float32),
            "policy_indices": policy_indices,
            "policy_values": policy_values,
            # Metadata
            "board_type": np.asarray(board_type),
            "board_size": np.asarray(int(board_size)),
            "history_length": np.asarray(int(history_length)),
            "feature_version": np.asarray(int(feature_version)),
            "policy_encoding": np.asarray(policy_encoding),
        }

        # Add optional arrays
        if values_mp is not None:
            save_kwargs["values_mp"] = values_mp.astype(np.float32)
        if num_players is not None:
            save_kwargs["num_players"] = num_players.astype(np.int32)
        if move_numbers is not None:
            save_kwargs["move_numbers"] = move_numbers.astype(np.int32)
        if total_game_moves is not None:
            save_kwargs["total_game_moves"] = total_game_moves.astype(np.int32)
        if phases is not None:
            save_kwargs["phases"] = phases
        if victory_types is not None:
            save_kwargs["victory_types"] = victory_types

        np.savez_compressed(str(output_path), **save_kwargs)
        logger.info(f"Saved {len(values)} samples to {output_path}")

    @staticmethod
    def append(
        output_path: str | Path,
        features: np.ndarray,
        globals_: np.ndarray,
        values: np.ndarray,
        policy_indices: np.ndarray,
        policy_values: np.ndarray,
        *,
        values_mp: np.ndarray | None = None,
        num_players: np.ndarray | None = None,
        victory_types: np.ndarray | None = None,
    ) -> int:
        """Append samples to an existing NPZ dataset.

        Args:
            output_path: Path to existing NPZ file
            features: New feature arrays to append
            globals_: New global feature vectors to append
            values: New value targets to append
            policy_indices: New policy indices to append
            policy_values: New policy values to append
            values_mp: New multi-player values to append (optional)
            num_players: New num_players array to append (optional)
            victory_types: New victory types to append (optional)

        Returns:
            Total sample count after appending

        Raises:
            FileNotFoundError: If output_path doesn't exist
        """
        output_path = Path(output_path)
        if not output_path.exists():
            raise FileNotFoundError(f"Cannot append to non-existent file: {output_path}")

        with np.load(str(output_path), allow_pickle=True) as data:
            # Load existing arrays
            existing_features = data["features"]
            existing_globals = data["globals"]
            existing_values = data["values"]
            existing_policy_indices = data["policy_indices"]
            existing_policy_values = data["policy_values"]

            # Concatenate
            new_features = np.concatenate([existing_features, features], axis=0)
            new_globals = np.concatenate([existing_globals, globals_], axis=0)
            new_values = np.concatenate([existing_values, values], axis=0)
            new_policy_indices = np.concatenate([existing_policy_indices, policy_indices], axis=0)
            new_policy_values = np.concatenate([existing_policy_values, policy_values], axis=0)

            # Preserve metadata
            save_kwargs: dict[str, Any] = {
                "features": new_features.astype(np.float32),
                "globals": new_globals.astype(np.float32),
                "values": new_values.astype(np.float32),
                "policy_indices": new_policy_indices,
                "policy_values": new_policy_values,
            }

            # Copy metadata fields
            for field in NPZDatasetWriter.METADATA_FIELDS:
                if field in data:
                    save_kwargs[field] = data[field]

            # Handle optional arrays
            has_mp = "values_mp" in data and "num_players" in data
            if has_mp and values_mp is not None and num_players is not None:
                save_kwargs["values_mp"] = np.concatenate([data["values_mp"], values_mp], axis=0)
                save_kwargs["num_players"] = np.concatenate([data["num_players"], num_players], axis=0)

            if "victory_types" in data and victory_types is not None:
                save_kwargs["victory_types"] = np.concatenate([data["victory_types"], victory_types], axis=0)

        np.savez_compressed(str(output_path), **save_kwargs)
        return len(new_values)

    @staticmethod
    def load(
        path: str | Path,
    ) -> dict[str, np.ndarray]:
        """Load an NPZ dataset.

        Args:
            path: Path to NPZ file

        Returns:
            Dictionary containing all arrays from the file
        """
        path = Path(path)
        with np.load(str(path), allow_pickle=True) as data:
            return {key: data[key] for key in data.files}

    @staticmethod
    def get_sample_count(path: str | Path) -> int:
        """Get the number of samples in an NPZ dataset.

        Args:
            path: Path to NPZ file

        Returns:
            Number of samples in the dataset
        """
        path = Path(path)
        with np.load(str(path), allow_pickle=True) as data:
            if "values" in data:
                return len(data["values"])
            if "features" in data:
                return len(data["features"])
            return 0


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Value computation
    "value_from_final_winner",
    "value_from_final_ranking",
    "compute_multi_player_values",
    # Encoding
    "encode_state_with_history",
    # NPZ I/O
    "NPZDatasetWriter",
]
