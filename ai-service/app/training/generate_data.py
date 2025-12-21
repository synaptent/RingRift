"""
Self-play data generation script for RingRift
Uses MCTS to generate high-quality training data with data augmentation.

Supports both square (8x8, 19x19) and hexagonal boards.
Hex boards use D6 symmetry augmentation (12 transformations).
"""

import argparse
import contextlib
import json
import logging
import os
import random as py_random
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from app.ai.descent_ai import DescentAI
from app.ai.gmo_ai import GMOAI, GMOConfig
from app.ai.mcts_ai import MCTSAI
from app.ai.neural_net import INVALID_MOVE_INDEX, NeuralNetAI, encode_move_for_board
from app.db import GameReplayDB, get_or_create_db, record_completed_game
from app.models import AIConfig, BoardType, GameState, GameStatus
from app.models.game_record import RecordSource
from app.training.env import (
    TrainingEnvConfig,
    get_theoretical_max_moves,
    make_env,
)
from app.training.game_record_export import build_training_game_record
from app.training.hex_augmentation import (
    HexSymmetryTransform,
    augment_hex_sample,
)
from app.utils.progress_reporter import SoakProgressReporter
from app.utils.resource_guard import LIMITS, check_disk_space, get_disk_usage

logger = logging.getLogger(__name__)


# =============================================================================
# Data Quality Tracking (Section 4.4: Self-Play Data Quality)
# =============================================================================

# Minimum unique positions per game for quality threshold
MIN_UNIQUE_POSITIONS_PER_GAME = 50


class DataQualityTracker:
    """Track data quality metrics during self-play generation.

    This tracker monitors position uniqueness to ensure training data
    diversity. Per Section 4.4 of the action plan, games with fewer than
    MIN_UNIQUE_POSITIONS_PER_GAME unique positions may indicate issues
    like repetitive play patterns or search instabilities.
    """

    def __init__(self):
        # Per-game tracking
        self._game_position_hashes: set[int] = set()
        self._game_move_count: int = 0

        # Aggregate tracking across all games
        self._all_unique_positions_counts: list[int] = []
        self._all_move_counts: list[int] = []
        self._low_uniqueness_games: int = 0
        self._total_games: int = 0

    def start_game(self) -> None:
        """Reset per-game tracking for a new game."""
        self._game_position_hashes = set()
        self._game_move_count = 0

    def record_position(self, zobrist_hash: int) -> None:
        """Record a position during self-play.

        Args:
            zobrist_hash: The zobrist hash of the current game state.
        """
        if zobrist_hash != 0:  # Ignore uninitialized hashes
            self._game_position_hashes.add(zobrist_hash)
        self._game_move_count += 1

    def finish_game(self) -> dict:
        """Finish tracking for the current game and return metrics.

        Returns:
            dict with keys:
                - unique_positions: Number of unique positions seen
                - total_moves: Total moves in the game
                - uniqueness_ratio: unique_positions / total_moves
                - below_threshold: Whether unique_positions < MIN_UNIQUE_POSITIONS_PER_GAME
        """
        unique_count = len(self._game_position_hashes)
        move_count = self._game_move_count

        # Avoid division by zero
        uniqueness_ratio = unique_count / max(move_count, 1)
        below_threshold = unique_count < MIN_UNIQUE_POSITIONS_PER_GAME

        # Update aggregate stats
        self._all_unique_positions_counts.append(unique_count)
        self._all_move_counts.append(move_count)
        self._total_games += 1
        if below_threshold:
            self._low_uniqueness_games += 1

        return {
            "unique_positions": unique_count,
            "total_moves": move_count,
            "uniqueness_ratio": uniqueness_ratio,
            "below_threshold": below_threshold,
        }

    def get_summary(self) -> dict:
        """Get summary statistics across all tracked games.

        Returns:
            dict with aggregate data quality metrics.
        """
        if not self._all_unique_positions_counts:
            return {
                "total_games": 0,
                "avg_unique_positions": 0.0,
                "min_unique_positions": 0,
                "max_unique_positions": 0,
                "avg_uniqueness_ratio": 0.0,
                "low_uniqueness_games": 0,
                "low_uniqueness_pct": 0.0,
            }

        unique_arr = np.array(self._all_unique_positions_counts)
        move_arr = np.array(self._all_move_counts)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(move_arr > 0, unique_arr / move_arr, 0.0)

        return {
            "total_games": self._total_games,
            "avg_unique_positions": float(np.mean(unique_arr)),
            "std_unique_positions": float(np.std(unique_arr)),
            "min_unique_positions": int(np.min(unique_arr)),
            "max_unique_positions": int(np.max(unique_arr)),
            "avg_uniqueness_ratio": float(np.mean(ratios)),
            "low_uniqueness_games": self._low_uniqueness_games,
            "low_uniqueness_pct": 100.0 * self._low_uniqueness_games / max(self._total_games, 1),
        }

    def log_summary(self) -> None:
        """Log a summary of data quality metrics."""
        summary = self.get_summary()
        if summary["total_games"] == 0:
            return

        logger.info(
            f"Data Quality Summary: "
            f"games={summary['total_games']}, "
            f"avg_unique_positions={summary['avg_unique_positions']:.1f}±{summary['std_unique_positions']:.1f}, "
            f"range=[{summary['min_unique_positions']}, {summary['max_unique_positions']}], "
            f"low_uniqueness_games={summary['low_uniqueness_games']} ({summary['low_uniqueness_pct']:.1f}%)"
        )

        # Warn if too many games have low uniqueness
        if summary["low_uniqueness_pct"] > 10.0:
            logger.warning(
                f"DATA_QUALITY_WARNING: {summary['low_uniqueness_pct']:.1f}% of games "
                f"have fewer than {MIN_UNIQUE_POSITIONS_PER_GAME} unique positions. "
                f"This may indicate repetitive play patterns or search issues."
            )


def extract_mcts_visit_distribution(
    ai: MCTSAI,
    state: GameState,
    encoder: NeuralNetAI | None = None,
    use_board_aware_encoding: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract MCTS visit count distribution as soft policy targets.

    This provides a richer training signal than 1-hot policy targets by
    capturing the relative value of different moves as explored by MCTS.

    Parameters
    ----------
    ai : MCTSAI
        The MCTS AI instance after a search has been performed.
    state : GameState
        The game state that was searched (for move encoding).
    encoder : Optional[NeuralNetAI]
        The neural network for encoding moves. If None, uses ai.neural_net.
        Only used when use_board_aware_encoding=False.
    use_board_aware_encoding : bool
        If True, uses encode_move_for_board() which produces compact
        board-specific policy indices (e.g., max ~7000 for square8).
        If False (default), uses the legacy encoder.encode_move() which
        always uses MAX_N=19 layout (~55000 indices).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (policy_indices, policy_values) as sparse arrays.
        Returns empty arrays if no visit distribution is available.

    Examples
    --------
    >>> ai = MCTSAI(1, config)
    >>> move = ai.select_move(state)  # Performs MCTS search
    >>> p_indices, p_values = extract_mcts_visit_distribution(ai, state)
    >>> # p_indices contains move indices, p_values contains visit probabilities
    """
    if not use_board_aware_encoding:
        if encoder is None:
            encoder = ai.neural_net
        if encoder is None:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    # Get visit distribution from MCTS (handles both legacy and incremental)
    moves, probs = ai.get_visit_distribution()

    if not moves:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    # Encode moves to policy indices
    p_indices = []
    p_values = []

    for move, prob in zip(moves, probs, strict=False):
        if use_board_aware_encoding:
            idx = encode_move_for_board(move, state.board)
        else:
            idx = encoder.encode_move(move, state.board)
        if idx != INVALID_MOVE_INDEX:
            p_indices.append(idx)
            p_values.append(prob)

    # Re-normalize in case some moves couldn't be encoded
    if p_values:
        total = sum(p_values)
        if total > 0:
            p_values = [v / total for v in p_values]

    return (
        np.array(p_indices, dtype=np.int32),
        np.array(p_values, dtype=np.float32),
    )


# Re-export create_initial_state from the lightweight module for backwards compatibility
# Using this module directly loads torch; for torch-free imports, use:
#   from app.training.initial_state import create_initial_state
from app.training.initial_state import create_initial_state


def calculate_outcome(state, player_number, depth):
    """
    Calculate detailed outcome with bonuses and discount
    Matches DescentAI logic
    """
    base_val = 0.0
    if state.winner == player_number:
        base_val = 1.0
    elif state.winner is not None:
        base_val = -1.0
    else:
        return 0.0

    # Bonuses
    territory_count = 0
    for p_id in state.board.collapsed_spaces.values():
        if p_id == player_number:
            territory_count += 1

    eliminated_count = state.board.eliminated_rings.get(str(player_number), 0)

    marker_count = 0
    for m in state.board.markers.values():
        if m.player == player_number:
            marker_count += 1

    # Normalize bonuses
    bonus = (
        (territory_count * 0.001) +
        (eliminated_count * 0.001) +
        (marker_count * 0.0001)
    )

    if base_val > 0:
        val = base_val + bonus
    else:
        val = base_val + bonus

    # Discount
    gamma = 0.99
    discounted_val = val * (gamma ** depth)

    if base_val > 0:
        return max(0.001, min(1.0, discounted_val))
    elif base_val < 0:
        return max(-1.0, min(-0.001, discounted_val))
    return 0.0


def calculate_multi_player_outcome(
    state,
    num_players: int,
    depth: int,
    max_players: int = 4,
    graded: bool = False,
) -> np.ndarray:
    """
    Calculate outcome values for all players simultaneously.

    Returns a vector of shape (max_players,) where:
    - Active player slots (0 to num_players-1) contain the outcome value
    - Inactive slots (num_players to max_players-1) are set to 0.0

    Parameters
    ----------
    state : GameState
        The final game state.
    num_players : int
        Number of active players in this game (2, 3, or 4).
    depth : int
        Number of moves remaining from this position (for discounting).
    max_players : int
        Maximum number of players to support (default: 4).
    graded : bool
        If True, use graded outcomes for multiplayer games (2nd place gets
        intermediate value instead of full loss). Default: False for backward
        compatibility.

    Returns
    -------
    np.ndarray
        Shape (max_players,) with values in [-1, +1] for active players.
    """
    values = np.zeros(max_players, dtype=np.float32)

    if not graded or num_players == 2:
        # Standard binary win/lose calculation
        for player_num in range(1, num_players + 1):
            outcome = calculate_outcome(state, player_num, depth)
            values[player_num - 1] = outcome
    else:
        # Graded outcomes for 3+ player games
        # Rank players by their final position metrics
        player_scores = []
        for player_num in range(1, num_players + 1):
            score = _calculate_player_score(state, player_num)
            player_scores.append((player_num, score))

        # Sort by score (highest = best performance)
        player_scores.sort(key=lambda x: x[1], reverse=True)

        # Assign graded values based on ranking
        # For 3 players: 1st=+1.0, 2nd=0.0, 3rd=-1.0
        # For 4 players: 1st=+1.0, 2nd=+0.33, 3rd=-0.33, 4th=-1.0
        if num_players == 3:
            graded_values = [1.0, 0.0, -1.0]
        else:  # num_players == 4
            graded_values = [1.0, 0.33, -0.33, -1.0]

        # Apply temporal discounting
        gamma = 0.99
        discount = gamma ** depth

        for rank, (player_num, score) in enumerate(player_scores):
            base_val = graded_values[rank]
            # Add small bonuses based on score margin
            bonus = score * 0.001  # Normalize large scores
            discounted_val = (base_val + bonus) * discount

            # Clamp to valid range
            if base_val > 0:
                values[player_num - 1] = max(0.001, min(1.0, discounted_val))
            elif base_val < 0:
                values[player_num - 1] = max(-1.0, min(-0.001, discounted_val))
            else:
                # Neutral position (2nd place in 3-player)
                values[player_num - 1] = max(-0.5, min(0.5, discounted_val))

    return values


def _calculate_player_score(state, player_number: int) -> float:
    """
    Calculate a composite score for ranking players in multiplayer games.

    Higher score = better performance. Used for determining placement
    in graded outcome calculation.

    Scoring components:
    - Winner bonus: +1000
    - Territory count: +10 per space
    - Rings on board: +1 per ring marker
    - Eliminated rings of opponents: +5 per ring
    """
    score = 0.0

    # Winner gets massive bonus
    if state.winner == player_number:
        score += 1000.0

    # Territory count
    for p_id in state.board.collapsed_spaces.values():
        if p_id == player_number:
            score += 10.0

    # Ring markers on board
    for m in state.board.markers.values():
        if m.player == player_number:
            score += 1.0

    # Rings eliminated from this player (penalty)
    eliminated = state.board.eliminated_rings.get(str(player_number), 0)
    score -= eliminated * 2.0

    return score


def augment_data(
    features,
    globals_vec,
    policy_indices,
    policy_values,
    neural_net,
    board_type: BoardType,
    hex_transform: HexSymmetryTransform | None = None,
    use_board_aware_encoding: bool = False,
):
    """
    Augment data by rotating and flipping.

    Returns a list of (features, globals, policy_indices, policy_values)
    tuples.

    For square boards: 8 augmentations (4 rotations × 2 flips)
    For hexagonal boards: 12 augmentations (D6 symmetry group)

    Parameters
    ----------
    use_board_aware_encoding : bool
        If True, policy indices use board-aware encoding (compact indices).
        Uses transform_policy_index_square for efficient policy transformation.
    """
    # Hex boards: use D6 symmetry augmentation (12 transformations)
    if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
        return augment_hex_data(
            features,
            globals_vec,
            policy_indices,
            policy_values,
            hex_transform,
        )

    augmented = []

    # Original sample
    augmented.append((features, globals_vec, policy_indices, policy_values))

    # Helper to transform a sparse policy using board-aware encoding
    def transform_policy_board_aware(indices, values, k_rot, flip_h):
        """Transform policy indices using transform_policy_index_square."""
        if len(indices) == 0:
            return indices, values

        from app.ai.neural_net import transform_policy_index_square

        new_indices = []
        new_values = []
        for idx, prob in zip(indices, values, strict=False):
            new_idx = transform_policy_index_square(
                int(idx), board_type, k_rot, flip_h
            )
            new_indices.append(new_idx)
            new_values.append(prob)

        return (
            np.array(new_indices, dtype=np.int32),
            np.array(new_values, dtype=np.float32),
        )

    # Helper to transform a sparse policy using legacy Move decode/encode
    def transform_policy_legacy(indices, values, k_rot, flip_h):
        if len(indices) == 0:
            return indices, values

        new_indices = []
        new_values = []
        # Get board_size from BOARD_CONFIGS (neural_net may be None for some engines)
        from app.rules.core import BOARD_CONFIGS
        board_size = BOARD_CONFIGS[board_type].size if board_type in BOARD_CONFIGS else 8

        # Create a dummy game state for decoding/encoding context.
        from app.models import (
            BoardState,
            GamePhase,
            GameState,
            GameStatus,
            Position,
            TimeControl,
        )
        dummy_state = GameState(
            id="dummy",
            boardType=board_type,
            board=BoardState(type=board_type, size=board_size),
            players=[],
            currentPhase=GamePhase.MOVEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=0,
                increment=0,
                type="blitz",
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt=datetime.now(),
            lastMoveAt=datetime.now(),
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=0,
            totalRingsEliminated=0,
            victoryThreshold=0,
            territoryVictoryThreshold=0,
            chainCaptureState=None,
            mustMoveFromStackKey=None,
        )

        for idx, prob in zip(indices, values, strict=False):
            move = neural_net.decode_move(idx, dummy_state)
            if not move:
                continue

            # Transform move coordinates in the CNN's 2D embedding.
            def rotate_point(x, y, n, k):
                """Rotate (x, y) in an n×n grid k times by 90°."""
                for _ in range(k):
                    x, y = y, n - 1 - x
                return x, y

            def flip_point(x, y, n):
                """Horizontal flip of (x, y) in an n×n grid."""
                return n - 1 - x, y

            # Transform 'to'
            tx, ty = move.to.x, move.to.y
            tx, ty = rotate_point(tx, ty, board_size, k_rot)
            if flip_h:
                tx, ty = flip_point(tx, ty, board_size)
            new_to = Position(x=tx, y=ty)

            new_from = None
            new_capture_target = None

            # Transform 'from' if exists
            if move.from_pos:
                fx, fy = move.from_pos.x, move.from_pos.y
                fx, fy = rotate_point(fx, fy, board_size, k_rot)
                if flip_h:
                    fx, fy = flip_point(fx, fy, board_size)
                new_from = Position(x=fx, y=fy)

            # Transform 'capture_target' if exists
            if move.capture_target:
                cx, cy = move.capture_target.x, move.capture_target.y
                cx, cy = rotate_point(cx, cy, board_size, k_rot)
                if flip_h:
                    cx, cy = flip_point(cx, cy, board_size)
                new_capture_target = Position(x=cx, y=cy)

            # Create new Move object with transformed coordinates
            move = move.model_copy(
                update={
                    "to": new_to,
                    "from_pos": new_from,
                    "capture_target": new_capture_target,
                }
            )

            # Re-encode using canonical coordinates derived from the dummy
            # board geometry. Moves that fall outside the fixed 19×19 policy
            # grid return INVALID_MOVE_INDEX and are skipped.
            new_idx = neural_net.encode_move(move, dummy_state.board)
            if new_idx != INVALID_MOVE_INDEX:
                new_indices.append(new_idx)
                new_values.append(prob)

        return (
            np.array(new_indices, dtype=np.int32),
            np.array(new_values, dtype=np.float32),
        )

    # Select the appropriate transform function
    transform_policy = (
        transform_policy_board_aware
        if use_board_aware_encoding
        else transform_policy_legacy
    )

    for k in range(1, 4):
        # Rotate features (C, H, W)
        rotated_features = np.rot90(features, k=k, axes=(1, 2))
        r_indices, r_values = transform_policy(
            policy_indices,
            policy_values,
            k,
            False,
        )
        augmented.append((rotated_features, globals_vec, r_indices, r_values))

    # Flip (horizontal)
    flipped_features = np.flip(features, axis=2)
    f_indices, f_values = transform_policy(
        policy_indices,
        policy_values,
        0,
        True,
    )
    augmented.append((flipped_features, globals_vec, f_indices, f_values))

    # Flip + rotations
    for k in range(1, 4):
        r_feat = np.rot90(features, k=k, axes=(1, 2))
        rf_features = np.flip(r_feat, axis=2)
        rf_indices, rf_values = transform_policy(
            policy_indices,
            policy_values,
            k,
            True,
        )
        augmented.append((rf_features, globals_vec, rf_indices, rf_values))

    return augmented


def augment_hex_data(
    features: np.ndarray,
    globals_vec: np.ndarray,
    policy_indices: np.ndarray,
    policy_values: np.ndarray,
    hex_transform: HexSymmetryTransform | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Augment hex board data using D6 symmetry transformations.

    Applies all 12 transformations from the dihedral group D6:
    - 6 rotations (0°, 60°, 120°, 180°, 240°, 300°)
    - 6 reflections (rotation + swap)

    Args:
        features: Board feature tensor (C, H, W)
        globals_vec: Global features vector
        policy_indices: Sparse policy move indices
        policy_values: Sparse policy probabilities
        hex_transform: Optional pre-created transform (efficiency)

    Returns:
        List of 12 augmented (features, globals, indices, values) tuples
    """
    if hex_transform is None:
        # Infer board size from features shape (C, H, W) where H=W=board_size
        board_size = features.shape[1] if len(features.shape) >= 2 else 25
        hex_transform = HexSymmetryTransform(board_size=board_size)

    # Convert to numpy arrays if needed
    if not isinstance(policy_indices, np.ndarray):
        policy_indices = np.array(policy_indices, dtype=np.int32)
    if not isinstance(policy_values, np.ndarray):
        policy_values = np.array(policy_values, dtype=np.float32)

    # Use the hex augmentation module
    return augment_hex_sample(
        features,
        globals_vec,
        policy_indices,
        policy_values,
        hex_transform,
    )


def generate_dataset(
    num_games: int = 10,
    output_file: str = "data/dataset.npz",
    ai1=None,
    ai2=None,
    board_type: BoardType = BoardType.SQUARE8,
    seed: int | None = None,
    max_moves: int = 10000,
    history_length: int = 3,
    feature_version: int = 2,
    batch_size: int | None = None,
    replay_db: GameReplayDB | None = None,
    num_players: int = 2,
    game_records_jsonl: str | None = None,
    engine: str = "descent",
    engine_mix: str = "single",
    engine_ratio: float = 0.5,
    nn_model_id: str | None = None,
    multi_player_values: bool = False,
    max_players: int = 4,
    graded_outcomes: bool = False,
) -> None:
    """
    Generate self-play data using DescentAI/MCTSAI and RingRiftEnv.
    Logs (state, best_move, root_value) for training.

    Parameters
    ----------
    num_games:
        Number of self-play games to generate.
    output_file:
        Path to the output NPZ file for training data.
    ai1:
        Optional pre-configured AI instance for player 1.
    ai2:
        Optional pre-configured AI instance for player 2.
    board_type:
        Board geometry to use (square8, square19, hexagonal).
    seed:
        Optional random seed for reproducibility.
    max_moves:
        Maximum moves per game before forcing termination.
    history_length:
        Number of previous feature frames to stack (default: 3).
    feature_version:
        Feature encoding version for global feature layout.
    batch_size:
        Optional batch size for buffer flushing (currently unused,
        reserved for future streaming implementation).
    replay_db:
        Optional GameReplayDB instance for recording completed games.
        If provided, all games are recorded to this database.
    num_players:
        Number of players in each game (2, 3, or 4). Default: 2.
    game_records_jsonl:
        Optional path to a JSONL file. When provided, each completed
        self-play game is also exported as a canonical GameRecord line
        suitable for downstream training and analysis pipelines.
    engine:
        Default search engine for self-play. One of:
          - "descent" (default): DescentAI-based pipeline
          - "mcts": MCTSAI-based AlphaZero-style pipeline
    engine_mix:
        Engine mixing strategy for diverse data generation. One of:
          - "single" (default): All players use the same engine type.
          - "per_game": Randomly choose engine per game (based on engine_ratio).
          - "per_player": Randomly choose engine per player within a game.
        When engine_mix != "single", the `engine` param becomes the default
        for ratio-based selection.
    engine_ratio:
        Ratio of MCTS games/players when using engine_mix != "single".
        0.0 = all Descent, 1.0 = all MCTS, 0.5 = 50/50. Default: 0.5.
    nn_model_id:
        Optional neural network model ID (e.g. "ringrift_v4_sq8_2p"). If provided,
        AI instances will use this model for evaluation. If None, engines
        use their default model loading behavior (board-aware defaults in
        `app/ai/neural_net.py`).
    multi_player_values:
        If True, store per-player value vectors of shape (N, max_players)
        instead of scalar values from the current player's perspective.
        Required for training RingRiftCNN_MultiPlayer models.
    max_players:
        Maximum number of player slots in value vectors (default: 4).
        Only used when multi_player_values=True.
    graded_outcomes:
        If True, use graded outcome values for 3+ player games where
        intermediate placements (2nd, 3rd) receive intermediate values
        instead of full loss (-1). For example, in a 4-player game:
        1st=+1.0, 2nd=+0.33, 3rd=-0.33, 4th=-1.0. Default: False.
    """
    import torch  # Import at function start for GMO device detection
    if seed is not None:
        py_random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # batch_size is accepted for future use but not currently implemented
    _ = batch_size

    # Accumulate data in separate lists
    new_features = []
    new_globals = []
    new_values = []
    new_policy_indices = []
    new_policy_values = []
    # Optional multi-player value vectors and num_players metadata (v2)
    new_values_mp: list[np.ndarray] = []
    new_num_players: list[int] = []
    # Optional per-sample metadata (aligned with replay export) used for
    # advanced sampling strategies when present.
    new_move_numbers: list[int] = []
    new_total_game_moves: list[int] = []
    new_phases: list[str] = []
    new_engines: list[str] = []

    # Validate engine parameters
    engine = engine.lower()
    engine_mix = engine_mix.lower()
    if engine not in {"descent", "mcts", "gmo"}:
        raise ValueError(f"Unsupported engine '{engine}'. Expected 'descent', 'mcts', or 'gmo'.")
    if engine_mix not in {"single", "per_game", "per_player"}:
        raise ValueError(f"Unsupported engine_mix '{engine_mix}'. Expected 'single', 'per_game', or 'per_player'.")
    if not 0.0 <= engine_ratio <= 1.0:
        raise ValueError(f"engine_ratio must be in [0.0, 1.0], got {engine_ratio}")

    feature_version = int(feature_version)

    def _apply_feature_version(ai_obj) -> None:
        """Propagate feature_version to any attached NeuralNetAI encoder."""
        nn = getattr(ai_obj, "neural_net", None)
        if nn is None:
            return
        with contextlib.suppress(Exception):
            nn.feature_version = feature_version
        hex_encoder = getattr(nn, "_hex_encoder", None)
        if hex_encoder is not None:
            with contextlib.suppress(Exception):
                hex_encoder.feature_version = feature_version

    def _make_ai(player_num: int, engine_type: str):
        """Create an AI instance for the specified player and engine type."""
        ai_config = AIConfig(
            difficulty=10,
            randomness=0.1 if engine_type == "descent" else 0.0,
            think_time=500,
            nn_model_id=nn_model_id,  # May be None for default behavior
        )
        if engine_type == "descent":
            ai = DescentAI(
                player_number=player_num,
                config=ai_config,
            )
        elif engine_type == "gmo":
            # Load GMO model if available
            gmo_config = GMOConfig(device="cuda" if torch.cuda.is_available() else "cpu")
            ai = GMOAI(
                player_number=player_num,
                config=ai_config,  # AIConfig for BaseAI
                gmo_config=gmo_config,  # GMOConfig for GMO-specific params
            )
            # Try to load trained checkpoint
            gmo_checkpoint = Path("models/gmo/gmo_best.pt")
            if gmo_checkpoint.exists():
                ai.load_checkpoint(gmo_checkpoint)
        else:
            ai = MCTSAI(
                player_number=player_num,
                config=ai_config,
            )
        _apply_feature_version(ai)
        return ai

    def _select_engine_for_player(player_num: int, game_engine: str) -> str:
        """Select engine type for a player based on mixing strategy."""
        if engine_mix == "single":
            return game_engine
        elif engine_mix == "per_player":
            return "mcts" if py_random.random() < engine_ratio else "descent"
        else:  # per_game - use the game-level engine
            return game_engine

    # For single engine mode or when using provided AI instances, build once.
    # For mixing modes, we'll rebuild per-game or use the engine selection.
    ai_players: dict = {}
    game_engine_type = engine  # Will be updated per-game in per_game mode

    if ai1 is not None:
        _apply_feature_version(ai1)
    if ai2 is not None:
        _apply_feature_version(ai2)

    if engine_mix == "single":
        # Initialize AI players once for all games
        for player_num in range(1, num_players + 1):
            if engine == "descent":
                if player_num == 1 and ai1 is not None:
                    ai_players[player_num] = ai1
                elif player_num == 2 and ai2 is not None:
                    ai_players[player_num] = ai2
                else:
                    ai_players[player_num] = _make_ai(player_num, engine)
            else:
                ai_players[player_num] = _make_ai(player_num, engine)

    # Primary AI reference for fallback when player not in ai_players
    ai_p1 = ai_players.get(1)

    # Track engine statistics for logging
    engine_stats = {"descent": 0, "mcts": 0}

    print(f"Generating {num_games} {num_players}-player games on {board_type}...")
    if engine_mix != "single":
        print(f"  Engine mixing: {engine_mix}, ratio (MCTS): {engine_ratio:.2f}")

    env_config = TrainingEnvConfig(
        board_type=board_type,
        num_players=num_players,
        max_moves=max_moves,
        reward_mode="terminal",
    )
    env = make_env(env_config)

    # Initialize progress reporter for time-based progress output (~10s intervals)
    progress_reporter = SoakProgressReporter(
        total_games=num_games,
        report_interval_sec=10.0,
        context_label=f"generate_data_{board_type.value}",
    )

    # Initialize data quality tracker (Section 4.4: Self-Play Data Quality)
    quality_tracker = DataQualityTracker()

    games_recorded = 0

    # Optional JSONL export of per-game GameRecord entries.
    jsonl_path: str | None = None
    jsonl_file = None
    if game_records_jsonl:
        if not os.path.isabs(game_records_jsonl):
            jsonl_path = os.path.join(
                os.path.dirname(__file__),
                game_records_jsonl,
            )
        else:
            jsonl_path = game_records_jsonl
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        jsonl_file = open(jsonl_path, "a", encoding="utf-8")

    for game_idx in range(num_games):
        # Circuit breaker: Check resources every 10 games
        if game_idx > 0 and game_idx % 10 == 0:
            try:
                from app.utils.resource_guard import can_proceed, wait_for_resources
                if not can_proceed(check_disk=True, check_mem=True, check_cpu_load=True):
                    print(f"Resource pressure at game {game_idx}/{num_games}, waiting...")
                    if not wait_for_resources(timeout=120.0, mem_required_gb=1.0):
                        print("Resources still constrained, continuing anyway")
            except ImportError:
                pass  # resource_guard not available

            # GPU memory cleanup to prevent OOM over long runs
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        game_start_time = time.time()
        # Set seed for each game if provided, incrementing to ensure variety
        game_seed = seed + game_idx if seed is not None else None
        state = env.reset(seed=game_seed)
        game_history = []

        # Start data quality tracking for this game
        quality_tracker.start_game()
        # Record initial position hash
        if hasattr(state, 'zobrist_hash') and state.zobrist_hash:
            quality_tracker.record_position(state.zobrist_hash)

        # Track initial state and moves for DB recording
        initial_state = state.model_copy(deep=True)
        moves_for_db: list = []

        # History buffer for this game
        # List of feature planes (10, 8, 8)
        state_history = []

        # --- Engine mixing: select/create AI players for this game ---
        player_engines: dict = {}  # Maps player_num -> engine_type string
        if engine_mix == "per_game":
            # Select engine type for this entire game
            game_engine_type = "mcts" if py_random.random() < engine_ratio else "descent"
            for pn in range(1, num_players + 1):
                ai_players[pn] = _make_ai(pn, game_engine_type)
                player_engines[pn] = game_engine_type
            ai_p1 = ai_players[1]
        elif engine_mix == "per_player":
            # Select engine type independently for each player
            for pn in range(1, num_players + 1):
                p_engine = _select_engine_for_player(pn, engine)
                ai_players[pn] = _make_ai(pn, p_engine)
                player_engines[pn] = p_engine
            ai_p1 = ai_players[1]
            game_engine_type = "mixed"  # Sentinel for logging
        else:
            # Single mode: all players use the same engine (already initialized)
            game_engine_type = engine
            for pn in range(1, num_players + 1):
                player_engines[pn] = engine

        # Track stats
        if engine_mix == "per_player":
            for _pn, eng in player_engines.items():
                engine_stats[eng] += 1
        else:
            engine_stats[game_engine_type] += num_players

        # Reset/reseed AIs per game to:
        # - keep games reproducible under a base seed, and
        # - avoid correlated RNG streams between players when running self-play.
        # We reseed even when AI instances are reused across games ("single"
        # engine mode) so that each game is reproducible in isolation.
        for pn in range(1, num_players + 1):
            ai = ai_players.get(pn)
            if ai is None:
                continue
            try:
                derived = None
                if seed is not None:
                    # Deterministic per-(game,player,engine) seed.
                    base_seed = int(seed) & 0xFFFFFFFF
                    eng = player_engines.get(pn, engine)
                    derived = (base_seed + game_idx * 1_000_003 + pn * 97_911) & 0xFFFFFFFF
                    if eng == "mcts":
                        derived ^= 0xA5A5A5A5
                ai.reset_for_new_game(rng_seed=derived)
            except AttributeError:
                # AI does not have reset_for_new_game method - acceptable for some AI types
                logger.debug(f"AI for player {pn} has no reset_for_new_game method")
            except Exception as e:
                # Other failures should be logged
                logger.warning(
                    f"Failed to reseed AI for player {pn} in game {game_idx+1}: "
                    f"{type(e).__name__}: {e}. Continuing with potentially non-reproducible RNG."
                )

        print(f"Game {game_idx+1} started (engine: {game_engine_type})")
        move_count = 0

        while (
            state.game_status == GameStatus.ACTIVE
            and move_count < env.max_moves
        ):
            current_player = state.current_player
            ai = ai_players.get(current_player, ai_p1)
            current_engine = player_engines.get(current_player, engine)

            # Select move using the chosen engine.
            # For DescentAI we retain the existing value + soft-policy logic.
            # For MCTSAI we use MCTS visit distributions as canonical soft
            # policy targets via extract_mcts_visit_distribution(...).
            move = ai.select_move(state)

            if not move:
                # No moves available, current player loses
                # Log this anomaly for debugging (games ending this early is unusual)
                logger.warning(
                    f"AI_NO_MOVE_RETURNED [game {game_idx+1}, move {move_count}]: "
                    f"engine={current_engine}, player={current_player}, "
                    f"phase={state.current_phase.value}, "
                    f"game_status={state.game_status.value}. "
                    f"This typically indicates OOM or model loading failure."
                )
                state.winner = 2 if current_player == 1 else 1
                state.game_status = GameStatus.COMPLETED
                break

            # Encode state and action based on engine type for this player
            if current_engine == "descent" and ai.neural_net:
                # Collect Tree Learning data (search log). For scalar value
                # training we retain these auxiliary samples; for multi-player
                # value training we skip them to avoid mismatched targets.
                if not multi_player_values:
                    search_data = ai.get_search_data()

                    # Process search data (features, value).
                    # These are auxiliary value-only samples gathered from the
                    # Descent tree. We attach empty sparse policies and treat
                    # them as pure value supervision.
                    for feat, val in search_data:
                        # We need to reconstruct the full input (stacked history)
                        # This is tricky because search nodes might be deep in the tree
                        # and we don't have their history easily available.
                        # However, for "Simple AlphaZero", they just use the current state features
                        # without history for the auxiliary value targets, OR they assume history is negligible.
                        # Given our architecture requires history, we might need to approximate.
                        # For now, let's use the current game history as the context,
                        # even though it's slightly incorrect for deep nodes (history should shift).
                        # But since Descent doesn't simulate opponent moves in history during search (it's just state),
                        # the history remains "what happened before search started".

                        # Construct stacked features using current game history
                        hist_list = state_history[::-1]
                        while len(hist_list) < history_length:
                            hist_list.append(np.zeros_like(feat))
                        hist_list = hist_list[:history_length]
                        stack_list = [feat, *hist_list]
                        stacked_features = np.concatenate(stack_list, axis=0)

                        # Globals need to be re-calculated for the search node state?
                        # The search log only stored features.
                        # We should probably store globals too in DescentAI.
                        # For now, let's use the root globals as approximation or skip globals update.
                        # Actually, let's just use the root globals.
                        _, root_globals = ai.neural_net._extract_features(state)

                        # Policy is unknown/irrelevant for these value-only
                        # samples; use empty sparse arrays.
                        p_indices = np.array([], dtype=np.int32)
                        p_values = np.array([], dtype=np.float32)

                        # Augment and add immediately
                        # Note: use_board_aware_encoding=True since policy indices
                        # come from encode_move_for_board (even if empty here)
                        augmented_samples = augment_data(
                            stacked_features,
                            root_globals,
                            p_indices,
                            p_values,
                            ai.neural_net,
                            state.board.type,
                            use_board_aware_encoding=True,
                        )

                        for f, g, pi, pv in augmented_samples:
                            new_features.append(f)
                            new_globals.append(g)
                            new_values.append(val)  # Use the search value directly
                            new_policy_indices.append(pi)
                            new_policy_values.append(pv)
                            # Approximate metadata for search nodes: treat them as
                            # occurring at the current move with total length
                            # bounded by the training max_moves. This keeps array
                            # shapes aligned for weighting without requiring full
                            # tree context.
                            new_move_numbers.append(move_count + 1)
                            new_total_game_moves.append(max_moves)
                            new_phases.append(state.current_phase.value)
                            new_engines.append("descent")

                # Now handle the actual root move (standard AlphaZero-style)
                features, globals_vec = ai.neural_net._extract_features(state)

                # Construct stacked features
                hist_list = state_history[::-1]
                while len(hist_list) < history_length:
                    hist_list.append(np.zeros_like(features))
                hist_list = hist_list[:history_length]
                stack_list = [features, *hist_list]
                stacked_features = np.concatenate(stack_list, axis=0)

                # Update history
                state_history.append(features)
                if len(state_history) > history_length + 1:
                    state_history.pop(0)

                # Encode policy using DescentAI search statistics (existing path)
                state_key = ai._get_state_key(state)
                p_indices: list[int] = []
                p_values: list[float] = []

                entry = ai.transposition_table.get(state_key)
                if entry is not None:
                    # Handle various TT entry formats
                    # descent_ai: (current_val, children_values, status, remaining_moves, visits)
                    # legacy: (val, children_values) or (val, children_values, status)
                    if len(entry) >= 5:
                        _, children_values, _, _, _ = entry
                    elif len(entry) == 3:
                        _, children_values, _ = entry
                    else:
                        _, children_values = entry

                    if children_values:
                        # children_values: {move_key: (move, val, prob)}
                        moves_data = []
                        for _, data in children_values.items():
                            m = data[0]
                            v = data[1]
                            moves_data.append((m, v))

                        # Sort by value descending - high value moves should always
                        # get high probability regardless of whose turn it is.
                        # The values are already from the current player's perspective,
                        # so best moves (highest value) should be rank 0.
                        #
                        # BUG FIX: Previously used is_maximizing check that inverted
                        # sorting for opponent moves, causing P2 good moves to get
                        # LOW probability in training data.
                        moves_data.sort(key=lambda x: x[1], reverse=True)

                        # Rank-based distribution (Cohen-Solal style)
                        k_rank = 1.0
                        probs = np.array(
                            [1.0 / (rank + k_rank) for rank in range(len(moves_data))],
                            dtype=np.float32,
                        )
                        probs = probs / probs.sum()

                        for i, (m, _) in enumerate(moves_data):
                            # Use board-aware encoding for compact policy indices
                            idx = encode_move_for_board(m, state.board)
                            if idx != INVALID_MOVE_INDEX:
                                p_indices.append(idx)
                                p_values.append(float(probs[i]))

                # Fallback if no search data
                if not p_indices:
                    # Use board-aware encoding for compact policy indices
                    idx = encode_move_for_board(move, state.board)
                    if idx != INVALID_MOVE_INDEX:
                        p_indices.append(idx)
                        p_values.append(1.0)

                game_history.append(
                    {
                        'features': stacked_features,
                        'globals': globals_vec,
                        'policy_indices': np.array(p_indices, dtype=np.int32),
                        'policy_values': np.array(p_values, dtype=np.float32),
                        'player': current_player,
                        'engine': current_engine,  # Track which engine generated this sample
                        'phase': state.current_phase.value,
                    }
                )

            elif current_engine == "mcts" and isinstance(ai, MCTSAI) and ai.neural_net:
                # MCTS-based soft policy targets using visit distributions.
                # Extract NN features for the root state.
                features, globals_vec = ai.neural_net._extract_features(state)

                # Construct stacked features with history
                hist_list = state_history[::-1]
                while len(hist_list) < history_length:
                    hist_list.append(np.zeros_like(features))
                hist_list = hist_list[:history_length]
                stack_list = [features, *hist_list]
                stacked_features = np.concatenate(stack_list, axis=0)

                # Update history buffer
                state_history.append(features)
                if len(state_history) > history_length + 1:
                    state_history.pop(0)

                # Extract soft policy from MCTS visits
                # Use board-aware encoding for compact policy indices
                p_indices_arr, p_values_arr = extract_mcts_visit_distribution(
                    ai,
                    state,
                    use_board_aware_encoding=True,
                )

                # Fallback: 1-hot on selected move if distribution is empty
                if p_indices_arr.size == 0:
                    # Use board-aware encoding for compact policy indices
                    idx = encode_move_for_board(move, state.board)
                    if idx != INVALID_MOVE_INDEX:
                        p_indices_arr = np.array([idx], dtype=np.int32)
                        p_values_arr = np.array([1.0], dtype=np.float32)

                game_history.append(
                    {
                        'features': stacked_features,
                        'globals': globals_vec,
                        'policy_indices': p_indices_arr,
                        'policy_values': p_values_arr,
                        'player': current_player,
                        'engine': current_engine,  # Track which engine generated this sample
                        'phase': state.current_phase.value,
                    }
                )

            # Track move for DB recording before applying
            moves_for_db.append(move)

            state, _, done, step_info = env.step(move)

            # Record position hash for data quality tracking
            if hasattr(state, 'zobrist_hash') and state.zobrist_hash:
                quality_tracker.record_position(state.zobrist_hash)

            # Include any bookkeeping moves (e.g., no_territory_action) that
            # the host/rules stack may have appended based on phase
            # requirements per RR-CANON-R075/R076. These are critical for
            # TS↔Python replay parity.
            auto_moves = step_info.get("auto_generated_moves", [])
            if auto_moves:
                moves_for_db.extend(auto_moves)

            move_count += 1

            if move_count % 10 == 0:
                print(f"  Move {move_count}")

            if done:
                break

        winner = state.winner
        print(f"Game {game_idx+1} finished. Winner: {winner}")

        # Finish data quality tracking for this game and get metrics
        quality_metrics = quality_tracker.finish_game()
        if quality_metrics["below_threshold"]:
            logger.warning(
                f"LOW_POSITION_UNIQUENESS [game {game_idx+1}]: "
                f"unique_positions={quality_metrics['unique_positions']}, "
                f"total_moves={quality_metrics['total_moves']}, "
                f"ratio={quality_metrics['uniqueness_ratio']:.2f} "
                f"(threshold: {MIN_UNIQUE_POSITIONS_PER_GAME})"
            )

        # Log error/warning for games that hit max_moves without a winner
        if move_count >= max_moves and winner is None:
            theoretical_max = get_theoretical_max_moves(
                board_type,
                num_players,
            )
            import sys
            if move_count >= theoretical_max:
                print(
                    f"ERROR: GAME_NON_TERMINATION [game {game_idx+1}] "
                    f"Game exceeded theoretical maximum moves without a winner. "
                    f"board_type={board_type.value}, move_count={move_count}, "
                    f"max_moves={max_moves}, theoretical_max={theoretical_max}, "
                    f"game_status={state.game_status.value}, winner={winner}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"WARNING: GAME_MAX_MOVES_CUTOFF [game {game_idx+1}] "
                    f"Game hit max_moves limit without a winner. "
                    f"board_type={board_type.value}, move_count={move_count}, "
                    f"max_moves={max_moves}, theoretical_max={theoretical_max}, "
                    f"game_status={state.game_status.value}, winner={winner}",
                    file=sys.stderr,
                )

        # Record game to DB if replay_db is provided
        recorded_game_id: str | None = None
        if replay_db is not None:
            try:
                # Build per-player AI type metadata for DB schema compatibility
                per_player_ai_meta = {}
                for pnum, eng_type in player_engines.items():
                    per_player_ai_meta[f"player_{pnum}_ai_type"] = eng_type

                recorded_game_id = record_completed_game(
                    db=replay_db,
                    initial_state=initial_state,
                    final_state=state,
                    moves=moves_for_db,
                    metadata={
                        "engine_mode": game_engine_type,
                        "engine_mix": engine_mix,
                        "player_engines": player_engines,
                        "seed": game_seed,
                        "source": "training_data_generation",
                        # Per-player AI type keys for DB schema compatibility
                        **per_player_ai_meta,
                    },
                )
                games_recorded += 1
            except Exception as e:
                print(f"Warning: Failed to record game to DB: {e}")

        # Assign rewards and build per-position training samples
        total_moves = len(game_history)

        for i, step in enumerate(game_history):
            moves_remaining = total_moves - i

            if multi_player_values:
                # Vector outcome for all players plus scalar view for the
                # current player for backwards compatibility.
                values_vec = calculate_multi_player_outcome(
                    state,
                    num_players=num_players,
                    depth=moves_remaining,
                    max_players=max_players,
                    graded=graded_outcomes,
                )
                outcome = float(values_vec[step['player'] - 1])
            else:
                # Scalar outcome from the perspective of the acting player.
                outcome = calculate_outcome(state, step['player'], moves_remaining)

            # Augment data for training; board_type is fixed per dataset.
            # Use board-aware encoding for policy indices (compact indices)
            augmented_samples = augment_data(
                step["features"],
                step["globals"],
                step["policy_indices"],
                step["policy_values"],
                ai_p1.neural_net,
                board_type,
                use_board_aware_encoding=True,
            )

            for feat, glob, pi, pv in augmented_samples:
                new_features.append(feat)
                new_globals.append(glob)
                new_values.append(outcome)
                new_policy_indices.append(pi)
                new_policy_values.append(pv)
                # Metadata aligned with replay-exported datasets:
                # - move_numbers: 1-based move index within the game
                # - total_game_moves: total (non-augmented) moves in the game
                # - phases: canonical GamePhase string for this root state
                # - engines: which engine produced this sample
                new_move_numbers.append(i + 1)
                new_total_game_moves.append(total_moves)
                new_phases.append(step.get('phase', state.current_phase.value))
                new_engines.append(step.get('engine', game_engine_type))
                if multi_player_values:
                    new_values_mp.append(values_vec)
                    new_num_players.append(num_players)

        # Record game completion for progress reporting
        game_duration = time.time() - game_start_time
        progress_reporter.record_game(
            moves=total_moves,
            duration_sec=game_duration,
        )

        # Optional JSONL export of a canonical GameRecord for this game.
        if jsonl_file is not None:
            game_id = recorded_game_id or f"training-{board_type.value}-{game_idx}-{uuid.uuid4().hex[:8]}"
            terminated_by_budget_only = (
                move_count >= max_moves
                and state.game_status == GameStatus.ACTIVE
                and winner is None
            )
            record = build_training_game_record(
                game_id=game_id,
                initial_state=initial_state,
                final_state=state,
                moves=moves_for_db,
                source=RecordSource.SELF_PLAY,
                rng_seed=game_seed,
                terminated_by_budget_only=terminated_by_budget_only,
                created_at=None,
                tags=["training_data_generation"],
            )
            jsonl_file.write(record.to_jsonl_line() + "\n")
            jsonl_file.flush()

    # Emit final progress summary and close JSONL (if any).
    progress_reporter.finish()
    quality_tracker.log_summary()
    if jsonl_file is not None:
        jsonl_file.close()

    # Save data with Experience Replay (Append mode)
    # Use provided output_file, ensuring directory exists
    if not os.path.isabs(output_file):
        output_path = os.path.join(os.path.dirname(__file__), output_file)
    else:
        output_path = output_file

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert new data to numpy arrays (object array for sparse policies)
    new_features = np.array(new_features, dtype=np.float32)
    new_globals = np.array(new_globals, dtype=np.float32)
    new_values = np.array(new_values, dtype=np.float32)
    # Sparse policies stored as object arrays of numpy arrays
    new_policy_indices = np.array(new_policy_indices, dtype=object)
    new_policy_values = np.array(new_policy_values, dtype=object)
    # Optional multi-player value vectors and num_players metadata
    if multi_player_values and new_values_mp:
        new_values_mp_arr = np.stack(new_values_mp, axis=0).astype(np.float32)
        new_num_players_arr = np.array(new_num_players, dtype=np.int32)
    else:
        new_values_mp_arr = None
        new_num_players_arr = None
    # Optional metadata arrays; kept simple for compatibility.
    new_move_numbers_arr = np.array(new_move_numbers, dtype=np.int32)
    new_total_game_moves_arr = np.array(new_total_game_moves, dtype=np.int32)
    new_phases_arr = np.array(new_phases, dtype=object)
    new_engines_arr = np.array(new_engines, dtype=object)

    print(f"Generated {len(new_values)} new samples")

    # Log engine usage statistics
    if engine_mix != "single":
        total_engine_samples = engine_stats["descent"] + engine_stats["mcts"]
        if total_engine_samples > 0:
            descent_pct = 100 * engine_stats["descent"] / total_engine_samples
            mcts_pct = 100 * engine_stats["mcts"] / total_engine_samples
            print(f"Engine stats: Descent={descent_pct:.1f}%, MCTS={mcts_pct:.1f}%")

    # Load existing data if available
    write_metadata = True
    write_mp = multi_player_values
    if os.path.exists(output_path):
        try:
            with np.load(output_path, allow_pickle=True) as data:
                # Check if keys exist (handling potential format changes)
                if 'features' in data:
                    existing_features = data['features']
                    existing_globals = data['globals']
                    existing_values = data['values']

                    # Handle migration from dense to sparse
                    if 'policy_indices' in data:
                        existing_policy_indices = data['policy_indices']
                        existing_policy_values = data['policy_values']
                    else:
                        print("Migrating dense policies to sparse...")
                        # Convert dense to sparse
                        dense_policies = data['policies']
                        existing_policy_indices = []
                        existing_policy_values = []
                        for p in dense_policies:
                            indices = np.nonzero(p)[0]
                            values = p[indices]
                            existing_policy_indices.append(
                                indices.astype(np.int32)
                            )
                            existing_policy_values.append(
                                values.astype(np.float32)
                            )
                        existing_policy_indices = np.array(
                            existing_policy_indices,
                            dtype=object,
                        )
                        existing_policy_values = np.array(
                            existing_policy_values,
                            dtype=object,
                        )

                    print(f"Loaded {len(existing_values)} existing samples")

                    # Optional metadata from prior runs. Only maintain
                    # per-sample metadata when the existing file already
                    # includes aligned arrays; this avoids mismatched
                    # lengths when appending to legacy datasets.
                    has_metadata = (
                        'move_numbers' in data
                        and 'total_game_moves' in data
                        and 'phases' in data
                        and 'engines' in data
                    )
                    if has_metadata:
                        existing_move_numbers = data['move_numbers']
                        existing_total_game_moves = data['total_game_moves']
                        existing_phases = data['phases']
                        existing_engines = data['engines']

                        new_move_numbers_arr = np.concatenate(
                            [existing_move_numbers, new_move_numbers_arr]
                        )
                        new_total_game_moves_arr = np.concatenate(
                            [existing_total_game_moves, new_total_game_moves_arr]
                        )
                        new_phases_arr = np.concatenate(
                            [existing_phases, new_phases_arr]
                        )
                        new_engines_arr = np.concatenate(
                            [existing_engines, new_engines_arr]
                        )
                    else:
                        # Do not attempt to write metadata when appending to
                        # older files that lack it.
                        write_metadata = False

                    # Optional multi-player targets from prior runs. Only
                    # append when the existing file already has them; when
                    # appending to legacy scalar-only datasets we disable
                    # multi-player writes to avoid inconsistent lengths.
                    if multi_player_values:
                        has_mp = (
                            'values_mp' in data
                            and 'num_players' in data
                        )
                        if has_mp and new_values_mp_arr is not None:
                            existing_values_mp = data['values_mp']
                            existing_num_players = data['num_players']

                            new_values_mp_arr = np.concatenate(
                                [existing_values_mp, new_values_mp_arr],
                                axis=0,
                            )
                            new_num_players_arr = np.concatenate(
                                [existing_num_players, new_num_players_arr],
                                axis=0,
                            )
                        else:
                            write_mp = False

                    # Concatenate core arrays
                    new_features = np.concatenate(
                        [existing_features, new_features]
                    )
                    new_globals = np.concatenate(
                        [existing_globals, new_globals]
                    )
                    new_values = np.concatenate(
                        [existing_values, new_values]
                    )
                    new_policy_indices = np.concatenate(
                        [existing_policy_indices, new_policy_indices]
                    )
                    new_policy_values = np.concatenate(
                        [existing_policy_values, new_policy_values]
                    )
        except Exception as e:
            print(f"Could not load existing data (starting fresh): {e}")

    # Limit buffer size (Experience Replay Buffer)
    MAX_BUFFER_SIZE = 50000
    if len(new_values) > MAX_BUFFER_SIZE:
        print(
            f"Buffer full ({len(new_values)}), "
            f"keeping last {MAX_BUFFER_SIZE} samples"
        )
        new_features = new_features[-MAX_BUFFER_SIZE:]
        new_globals = new_globals[-MAX_BUFFER_SIZE:]
        new_values = new_values[-MAX_BUFFER_SIZE:]
        new_policy_indices = new_policy_indices[-MAX_BUFFER_SIZE:]
        new_policy_values = new_policy_values[-MAX_BUFFER_SIZE:]

    # Save as compressed npz
    save_kwargs = {
        "features": new_features,
        "globals": new_globals,
        "values": new_values,
        "policy_indices": new_policy_indices,
        "policy_values": new_policy_values,
        "policy_encoding": np.asarray("board_aware"),
        "history_length": np.asarray(int(history_length)),
        "feature_version": np.asarray(int(feature_version)),
    }
    if write_metadata:
        save_kwargs.update(
            {
                "move_numbers": new_move_numbers_arr,
                "total_game_moves": new_total_game_moves_arr,
                "phases": new_phases_arr,
                "engines": new_engines_arr,
            }
        )
    if write_mp and (new_values_mp_arr is not None) and (new_num_players_arr is not None):
        save_kwargs.update(
            {
                "values_mp": new_values_mp_arr,
                "num_players": new_num_players_arr,
            }
        )

    # Check disk space before writing (datasets can be 100MB-2GB)
    output_dir = os.path.dirname(output_path) or '.'
    if not check_disk_space(required_gb=2.0, path=output_dir, log_warning=False):
        disk_pct, available_gb, _ = get_disk_usage(output_dir)
        raise OSError(
            f"Insufficient disk space to save dataset: "
            f"{disk_pct:.1f}% used (limit: {LIMITS.DISK_MAX_PERCENT}%), "
            f"{available_gb:.1f}GB available. Path: {output_path}"
        )

    np.savez_compressed(output_path, **save_kwargs)


def generate_dataset_gpu_parallel(
    num_games: int = 20,
    output_file: str = "data/dataset_gpu.npz",
    board_type: BoardType = BoardType.SQUARE8,
    seed: int | None = None,
    max_moves: int = 10000,
    num_players: int = 2,
    history_length: int = 3,
    feature_version: int = 2,
    gpu_batch_size: int = 20,
    heuristic_weights: dict | None = None,
) -> None:
    """Generate self-play data using GPU-accelerated parallel game simulation.

    This function runs multiple games simultaneously on GPU using the
    ParallelGameRunner, providing 5-10x speedup over sequential CPU execution.
    Move selection uses GPU heuristic evaluation (45 weights) rather than
    tree search, making this suitable for initial training bootstrapping or
    generating large volumes of diverse data quickly.

    Parameters
    ----------
    num_games:
        Total number of games to generate. Will be processed in batches
        of gpu_batch_size.
    output_file:
        Path to the output NPZ file for training data.
    board_type:
        Board geometry to use (square8, square19, hexagonal).
    seed:
        Optional random seed for reproducibility.
    max_moves:
        Maximum moves per game before forcing termination.
    num_players:
        Number of players in each game (2, 3, or 4).
    history_length:
        Number of previous feature frames to stack (default: 3).
    feature_version:
        Feature encoding version for global feature layout.
    gpu_batch_size:
        Number of games to run in parallel on GPU. Higher values use more
        GPU memory but provide better throughput. Recommended: 10-50.
    heuristic_weights:
        Optional heuristic weight dictionary for move selection. If None,
        uses BASE_V1_BALANCED_WEIGHTS.

    Notes
    -----
    This function generates training data with:
    - Board state features extracted at each move
    - 1-hot policy targets based on selected moves
    - Final game outcome propagated to all samples with depth discount

    The data format is compatible with the standard training pipeline and
    can be mixed with data from generate_dataset().
    """
    import torch

    from app.ai.gpu_parallel_games import (
        ParallelGameRunner,
    )
    from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS
    from app.ai.neural_net import NeuralNetAI, encode_move_for_board

    if seed is not None:
        py_random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    feature_version = int(feature_version)

    # Use default weights if not provided
    weights = heuristic_weights or dict(BASE_V1_BALANCED_WEIGHTS)

    # Board size mapping
    board_size_map = {
        BoardType.SQUARE8: 8,
        BoardType.SQUARE19: 19,
        BoardType.HEXAGONAL: 25,  # Hex uses 25x25 embedding
    }
    board_size = board_size_map.get(board_type, 8)
    board_type_str = {
        BoardType.SQUARE8: "square8",
        BoardType.SQUARE19: "square19",
        BoardType.HEXAGONAL: "hexagonal",
    }.get(board_type, "square8")

    # Initialize neural net for feature extraction (lazy load)
    nn_encoder = NeuralNetAI(
        player_number=1,
        config=AIConfig(difficulty=1),
    )
    nn_encoder.feature_version = feature_version
    # Force initialization
    nn_encoder._ensure_model_initialized(board_type)

    print(f"GPU Parallel: Generating {num_games} games with batch_size={gpu_batch_size}")
    print(f"  Board: {board_type_str}, Players: {num_players}, Max moves: {max_moves}")

    # Accumulate training data
    all_features = []
    all_globals = []
    all_values = []
    all_policy_indices = []
    all_policy_values = []

    # Initialize data quality tracker (Section 4.4: Self-Play Data Quality)
    quality_tracker = DataQualityTracker()

    # Process games in batches
    num_batches = (num_games + gpu_batch_size - 1) // gpu_batch_size
    games_generated = 0

    for batch_idx in range(num_batches):
        batch_start = batch_idx * gpu_batch_size
        batch_games = min(gpu_batch_size, num_games - batch_start)

        print(f"  Batch {batch_idx + 1}/{num_batches}: {batch_games} games")
        batch_start_time = time.time()

        # Initialize runner for this batch
        runner = ParallelGameRunner(
            batch_size=batch_games,
            board_size=board_size,
            num_players=num_players,
            board_type=board_type_str,
        )

        # Run games to completion
        batch_seed = seed + batch_idx * 10000 if seed is not None else None
        if batch_seed is not None:
            torch.manual_seed(batch_seed)

        results = runner.run_games(
            weights_list=[weights] * batch_games,
            max_moves=max_moves,
        )

        batch_duration = time.time() - batch_start_time
        print(f"    Completed in {batch_duration:.1f}s")

        # Extract training data from each game
        for g in range(batch_games):
            results["winners"][g]
            results["move_counts"][g]

            # Extract move history
            move_history = runner.state.extract_move_history(g)

            if not move_history:
                continue

            # Convert final game state to CPU for reference
            final_state = runner.state.to_game_state(g)

            # Replay game to extract features at each position
            # We need to reconstruct states step-by-step
            env_config = TrainingEnvConfig(
                board_type=board_type,
                num_players=num_players,
                max_moves=max_moves,
                reward_mode="terminal",
            )
            env = make_env(env_config)
            game_seed = batch_seed + g if batch_seed is not None else None
            state = env.reset(seed=game_seed)

            # Start data quality tracking for this game
            quality_tracker.start_game()
            if hasattr(state, 'zobrist_hash') and state.zobrist_hash:
                quality_tracker.record_position(state.zobrist_hash)

            game_samples = []
            history_buffer = []

            for move_idx, move_dict in enumerate(move_history):
                current_player = state.current_player

                # Extract features for current state
                try:
                    features, globals_vec = nn_encoder._extract_features(state)

                    # Build stacked features with history
                    hist_list = history_buffer[::-1]
                    while len(hist_list) < history_length:
                        hist_list.append(np.zeros_like(features))
                    hist_list = hist_list[:history_length]
                    stacked_features = np.concatenate([features, *hist_list], axis=0)

                    # Update history buffer
                    history_buffer.append(features)
                    if len(history_buffer) > history_length + 1:
                        history_buffer.pop(0)

                    # Create move object from dict
                    from app.models import Move, Position
                    move_type_str = move_dict.get("type", "place_ring")
                    to_pos = move_dict.get("to")
                    from_pos = move_dict.get("from")

                    # Map GPU move type strings to MoveType enum
                    from app.models import MoveType as CPUMoveType
                    move_type_map = {
                        "place_ring": CPUMoveType.PLACE_RING,
                        "move_stack": CPUMoveType.MOVE_STACK,
                        "overtaking_capture": CPUMoveType.OVERTAKING_CAPTURE,
                        "chain_capture": CPUMoveType.CHAIN_CAPTURE,
                        "line_formation": CPUMoveType.LINE_FORMATION,
                        "territory_claim": CPUMoveType.TERRITORY_CLAIM,
                        "skip_capture": CPUMoveType.SKIP_CAPTURE,
                        "recovery_slide": CPUMoveType.RECOVERY_SLIDE,
                    }
                    cpu_move_type = move_type_map.get(
                        move_type_str, CPUMoveType.PLACE_RING
                    )

                    # Create Move object with required fields
                    move = Move(
                        id=f"gpu-{g}-{move_idx}",
                        type=cpu_move_type,
                        player=move_dict.get("player", current_player),
                        to=Position(
                            x=to_pos["x"], y=to_pos["y"]
                        ) if to_pos else None,
                        from_pos=Position(
                            x=from_pos["x"], y=from_pos["y"]
                        ) if from_pos else None,
                        timestamp=datetime.now(),
                        think_time=0,
                        move_number=move_idx + 1,
                    )

                    # Encode move to policy index
                    policy_idx = encode_move_for_board(move, state.board)
                    if policy_idx >= 0:
                        p_indices = np.array([policy_idx], dtype=np.int32)
                        p_values = np.array([1.0], dtype=np.float32)
                    else:
                        p_indices = np.array([], dtype=np.int32)
                        p_values = np.array([], dtype=np.float32)

                    game_samples.append({
                        "features": stacked_features,
                        "globals": globals_vec,
                        "policy_indices": p_indices,
                        "policy_values": p_values,
                        "player": current_player,
                    })

                except (ValueError, KeyError, AttributeError, TypeError) as e:
                    # Log and skip samples we can't extract features for
                    logger.debug(
                        f"Skipping sample in GPU batch {batch_idx+1}, game {g+1}, "
                        f"move {move_idx+1}: {type(e).__name__}: {e}"
                    )

                # Apply move to advance state
                try:
                    state, _, done, _ = env.step(move)
                    # Record position hash for data quality tracking
                    if hasattr(state, 'zobrist_hash') and state.zobrist_hash:
                        quality_tracker.record_position(state.zobrist_hash)
                    if done:
                        break
                except (ValueError, RuntimeError) as e:
                    logger.warning(
                        f"State transition failed in GPU batch {batch_idx+1}, game {g+1}, "
                        f"move {move_idx+1}: {type(e).__name__}: {e}. Ending game early."
                    )
                    break

            # Finish data quality tracking for this game
            quality_metrics = quality_tracker.finish_game()
            if quality_metrics["below_threshold"]:
                logger.warning(
                    f"LOW_POSITION_UNIQUENESS [GPU batch {batch_idx+1}, game {g+1}]: "
                    f"unique_positions={quality_metrics['unique_positions']}, "
                    f"total_moves={quality_metrics['total_moves']}, "
                    f"ratio={quality_metrics['uniqueness_ratio']:.2f}"
                )

            # Calculate outcomes and add to dataset
            total_game_moves = len(game_samples)
            for i, sample in enumerate(game_samples):
                moves_remaining = total_game_moves - i
                outcome = calculate_outcome(
                    final_state, sample["player"], moves_remaining
                )

                # Augment data (using board-aware encoding for policy indices)
                augmented = augment_data(
                    sample["features"],
                    sample["globals"],
                    sample["policy_indices"],
                    sample["policy_values"],
                    nn_encoder,
                    board_type,
                    use_board_aware_encoding=True,
                )

                for feat, glob, pi, pv in augmented:
                    all_features.append(feat)
                    all_globals.append(glob)
                    all_values.append(outcome)
                    all_policy_indices.append(pi)
                    all_policy_values.append(pv)

            games_generated += 1

        print(f"    Total samples so far: {len(all_values)}")

    # Save to NPZ
    print(f"\nGPU Parallel: Generated {len(all_values)} total samples from {games_generated} games")
    quality_tracker.log_summary()

    output_path = output_file
    if not os.path.isabs(output_file):
        output_path = os.path.join(
            os.path.dirname(__file__),
            output_file,
        )
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    # Check disk space before writing
    if not check_disk_space(required_gb=2.0, path=output_dir, log_warning=False):
        disk_pct, available_gb, _ = get_disk_usage(output_dir)
        raise OSError(
            f"Insufficient disk space to save dataset: "
            f"{disk_pct:.1f}% used (limit: {LIMITS.DISK_MAX_PERCENT}%), "
            f"{available_gb:.1f}GB available. Path: {output_path}"
        )

    np.savez_compressed(
        output_path,
        features=np.array(all_features, dtype=np.float32),
        globals=np.array(all_globals, dtype=np.float32),
        values=np.array(all_values, dtype=np.float32),
        policy_indices=np.array(all_policy_indices, dtype=object),
        policy_values=np.array(all_policy_values, dtype=object),
        policy_encoding=np.asarray("board_aware"),
        history_length=np.asarray(int(history_length)),
        feature_version=np.asarray(int(feature_version)),
    )
    print(f"Saved to {output_path}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for data generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate self-play training data for RingRift neural network. "
            "Uses DescentAI to play games and collects (state, policy, value) "
            "tuples with data augmentation."
        ),
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of self-play games to generate (default: 100).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/training_data.npz",
        help="Output file path for training data (default: logs/training_data.npz).",
    )
    parser.add_argument(
        "--board-type",
        choices=["square8", "square19", "hex8", "hexagonal"],
        default="square8",
        help="Board type for self-play games (default: square8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=2000,
        help="Maximum moves per game before forcing termination (default: 2000).",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=3,
        help="Number of previous feature frames to stack (default: 3).",
    )
    parser.add_argument(
        "--feature-version",
        type=int,
        default=2,
        help=(
            "Feature encoding version for global feature layout (default: 2). "
            "Use 1 to keep legacy hex globals without chain/FE flags."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for buffer flushing (optional, reserved for future use).",
    )
    parser.add_argument(
        "--record-db",
        type=str,
        default="data/games/training.db",
        help=(
            "Path to SQLite database for recording game replays. "
            "Default: data/games/training.db. Use --no-record-db to disable."
        ),
    )
    parser.add_argument(
        "--no-record-db",
        action="store_true",
        help="Disable game recording to database (overrides --record-db).",
    )
    parser.add_argument(
        "--allow-noncanonical-db",
        action="store_true",
        help=(
            "Allow using a replay DB whose canonical self-play gate summary "
            "reports canonical_ok == false or fe_territory_fixtures_ok == false. "
            "Intended only for ad-hoc experiments; canonical training data "
            "must use DBs listed as canonical in ai-service/TRAINING_DATA_REGISTRY.md."
        ),
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        default=2,
        help="Number of players in each game (default: 2).",
    )
    parser.add_argument(
        "--game-records-jsonl",
        type=str,
        default=None,
        help=(
            "Optional path to write one JSONL GameRecord per completed "
            "self-play game (canonical schema for training pipelines)."
        ),
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="descent",
        choices=["descent", "mcts"],
        help=(
            "Self-play engine for data generation: 'descent' (default) "
            "or 'mcts' for NN-guided MCTS with soft policy targets."
        ),
    )
    parser.add_argument(
        "--engine-mix",
        type=str,
        default="single",
        choices=["single", "per_game", "per_player"],
        help=(
            "Engine mixing strategy: 'single' (all players use --engine), "
            "'per_game' (choose engine per game), 'per_player' (choose engine "
            "per player). See --engine-ratio to control MCTS proportion."
        ),
    )
    parser.add_argument(
        "--engine-ratio",
        type=float,
        default=0.5,
        help=(
            "When --engine-mix != 'single', ratio of MCTS usage (0.0 = all Descent, "
            "1.0 = all MCTS, 0.5 = 50/50). Default: 0.5."
        ),
    )
    parser.add_argument(
        "--nn-model-id",
        type=str,
        default=None,
        help=(
            "Neural network model ID (e.g. 'ringrift_v4_sq8_2p') for AI evaluation. "
            "If not provided, engines use their default model loading behavior."
        ),
    )
    parser.add_argument(
        "--multi-player-values",
        action="store_true",
        help=(
            "Store per-player value vectors (values_mp / num_players) in the "
            "output dataset for multi-player training."
        ),
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=4,
        help=(
            "Maximum number of player slots for multi-player value vectors "
            "(default: 4). Only used when --multi-player-values is set."
        ),
    )
    parser.add_argument(
        "--graded-outcomes",
        action="store_true",
        help=(
            "Use graded outcome values for 3+ player games. Instead of "
            "binary win/lose, intermediate placements get intermediate "
            "values (e.g., 4-player: 1st=+1, 2nd=+0.33, 3rd=-0.33, 4th=-1). "
            "Only affects --multi-player-values datasets."
        ),
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Use parallel selfplay generation for 4-8x speedup. Uses multiple "
            "worker processes, each with its own AI instances. Recommended for "
            "generating large datasets."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes for parallel generation. "
            "Default: CPU count - 1. Only used with --parallel."
        ),
    )
    return parser.parse_args()


def _board_type_from_str(name: str) -> BoardType:
    """Convert board type string to BoardType enum."""
    if name == "square8":
        return BoardType.SQUARE8
    if name == "square19":
        return BoardType.SQUARE19
    if name == "hex8":
        return BoardType.HEX8
    if name == "hexagonal":
        return BoardType.HEXAGONAL
    raise ValueError(f"Unknown board type: {name!r}")


def _resolve_db_summary_path(db_path: Path) -> Path | None:
    """Best-effort resolution of a canonical self-play gate summary JSON.

    Mirrors the conventions used by ``scripts/generate_canonical_selfplay.py``:

    - ``db_health.<stem>.json`` in the same directory as the DB
      (for example, ``db_health.canonical_square8.json`` for
      ``data/games/canonical_square8.db``).
    - ``<name>.db.summary.json`` as a generic fallback
      (for example, ``canonical_square8.db.summary.json``).

    Returns
    -------
    Optional[Path]
        The first existing summary path, or ``None`` if no candidate exists.
    """
    # Candidate 1: db_health.<stem>.json (preferred convention).
    candidate = db_path.with_name(f"db_health.{db_path.stem}.json")
    if candidate.exists():
        return candidate

    # Candidate 2: <name>.db.summary.json (generic/fallback convention).
    candidate = db_path.with_suffix(db_path.suffix + ".summary.json")
    if candidate.exists():
        return candidate

    return None


def _assert_db_is_canonical_if_summary_exists(
    db_path: Path,
    *,
    allow_noncanonical: bool = False,
) -> None:
    """Assert that a replay DB is canonical when a gate summary is present.

    This helper is intentionally **best-effort**:

    - If no summary JSON can be found, it is a no-op.
    - If a summary exists, it expects the shape written by
      ``scripts/generate_canonical_selfplay.py``, including:

      * ``canonical_ok`` (top-level boolean allowlist flag)
      * ``fe_territory_fixtures_ok`` (boolean FE/territory gate flag)

    Behaviour
    ---------
    - When ``canonical_ok`` is **True** and ``fe_territory_fixtures_ok`` is
      **True** (or absent, defaulting to True), the check is a no-op.
    - When either flag is false and ``allow_noncanonical`` is **False``, a
      :class:`ValueError` is raised to prevent silent use of non-canonical DBs.
    - When either flag is false and ``allow_noncanonical`` is **True``, a loud
      warning is printed but execution continues. This is intended only for
      ad-hoc experiments; canonical training runs should instead rely on DBs
      listed as ``canonical`` in ``ai-service/TRAINING_DATA_REGISTRY.md``.
    """
    summary_path = _resolve_db_summary_path(db_path)
    if summary_path is None or not summary_path.exists():
        return

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive guard
        print(
            "WARNING: generate_data could not read canonical gate summary "
            f"{summary_path} for DB {db_path}: {exc}. "
            "Proceeding without canonical_ok / FE-territory checks.",
            flush=True,
        )
        return

    canonical_ok = bool(data.get("canonical_ok", False))
    fe_ok = bool(data.get("fe_territory_fixtures_ok", True))

    # Fast path: canonical and FE/territory fixtures ok.
    if canonical_ok and fe_ok:
        return

    if allow_noncanonical:
        print(
            "WARNING: generate_data is using a replay DB whose canonical self-play "
            "gate summary marks it non-canonical "
            f"(canonical_ok={canonical_ok}, fe_territory_fixtures_ok={fe_ok}) "
            f"at {summary_path}. Proceeding only because "
            "--allow-noncanonical-db was supplied. For canonical training, use "
            "DBs listed as canonical in ai-service/TRAINING_DATA_REGISTRY.md.",
            flush=True,
        )
        return

    raise ValueError(
        "Refusing to use replay DB with failing canonical self-play gate. "
        f"DB path: {db_path}. Summary: {summary_path}. "
        f"canonical_ok={canonical_ok}, fe_territory_fixtures_ok={fe_ok}. "
        "Either regenerate the DB via scripts/generate_canonical_selfplay.py "
        "or pass --allow-noncanonical-db for ad-hoc experiments."
    )


def main() -> None:
    """Entry point for CLI-based data generation."""
    args = _parse_args()
    board_type = _board_type_from_str(args.board_type)

    # Validate numeric arguments
    if args.num_games < 1:
        raise ValueError(
            f"--num-games must be at least 1, got {args.num_games}"
        )
    if args.max_moves < 1:
        raise ValueError(
            f"--max-moves must be at least 1, got {args.max_moves}"
        )
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError(
            f"--batch-size must be at least 1, got {args.batch_size}"
        )

    if args.parallel and args.feature_version > 1:
        logger.warning(
            "parallel_selfplay uses legacy globals; forcing feature_version=1 "
            "(requested %d).",
            args.feature_version,
        )
        args.feature_version = 1

    # Initialize optional game recording database
    # --no-record-db flag overrides --record-db to disable recording
    record_db_path = None if args.no_record_db else args.record_db

    # Best-effort canonical gate for replay DBs when a self-play summary exists.
    # This prevents silently recording to clearly non-canonical DBs when a
    # canonical gate summary has already been generated for that path, while
    # keeping legacy/experimental flows (without summaries) untouched.
    if record_db_path:
        _assert_db_is_canonical_if_summary_exists(
            Path(record_db_path),
            allow_noncanonical=args.allow_noncanonical_db,
        )

    replay_db = get_or_create_db(record_db_path) if record_db_path else None

    # Use parallel generation if requested (4-8x speedup)
    if args.parallel:
        try:
            from app.training.parallel_selfplay import generate_dataset_parallel
            print(f"Using parallel selfplay generation with {args.num_workers or 'auto'} workers")
            generate_dataset_parallel(
                num_games=args.num_games,
                output_file=args.output,
                num_workers=args.num_workers,
                board_type=board_type,
                seed=args.seed,
                max_moves=args.max_moves,
                num_players=args.num_players,
                history_length=args.history_length,
                feature_version=args.feature_version,
                engine=args.engine,
                nn_model_id=args.nn_model_id,
                multi_player_values=args.multi_player_values,
                max_players=args.max_players,
                graded_outcomes=args.graded_outcomes,
            )
        except ImportError as e:
            print(f"Warning: parallel_selfplay module not available ({e}), falling back to sequential")
            args.parallel = False

    if not args.parallel:
        generate_dataset(
            num_games=args.num_games,
            output_file=args.output,
            board_type=board_type,
            seed=args.seed,
            max_moves=args.max_moves,
            history_length=args.history_length,
            feature_version=args.feature_version,
            batch_size=args.batch_size,
            replay_db=replay_db,
            num_players=args.num_players,
            game_records_jsonl=args.game_records_jsonl,
            engine=args.engine,
            engine_mix=args.engine_mix,
            engine_ratio=args.engine_ratio,
            nn_model_id=args.nn_model_id,
            multi_player_values=args.multi_player_values,
            max_players=args.max_players,
            graded_outcomes=args.graded_outcomes,
        )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
