"""
Parallel encoding utilities for RingRift training data.

This module provides parallelized versions of the state and move encoding
functions, using ProcessPoolExecutor for true parallelism (bypassing GIL).

Key features:
- Batch processing of games across multiple CPU cores
- Worker pool with persistent encoders to avoid re-initialization
- Support for both hex (HexStateEncoder/V3) and square boards
- Drop-in compatible with export_replay_dataset.py

Usage:
    from app.training.parallel_encoding import ParallelEncoder

    encoder = ParallelEncoder(
        board_type=BoardType.HEXAGONAL,
        num_workers=8,
        encoder_version="v3",
    )

    # Process a batch of games
    results = encoder.encode_games_batch(games_data)

    encoder.shutdown()
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from app.models import BoardType, GameState

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies and speed up worker init
_ENCODER_CACHE: dict[str, Any] = {}


def _get_encoder(
    board_type_str: str,
    encoder_version: str = "v3",
    feature_version: int = 2,
):
    """Get or create a cached encoder for the given board type."""
    cache_key = f"{board_type_str}_{encoder_version}_fv{feature_version}"
    if cache_key not in _ENCODER_CACHE:
        # Import here to avoid circular imports and speed up module load
        from app.models import BoardType
        from app.training.encoding import get_encoder_for_board_type

        board_type = BoardType(board_type_str)
        _ENCODER_CACHE[cache_key] = get_encoder_for_board_type(
            board_type,
            encoder_version,
            feature_version=feature_version,
        )

    return _ENCODER_CACHE[cache_key]


def _get_neural_net_encoder(board_type_str: str, feature_version: int = 2):
    """Get or create a cached NeuralNetAI encoder for square boards."""
    cache_key = f"nn_{board_type_str}_fv{feature_version}"
    if cache_key not in _ENCODER_CACHE:
        from app.ai.neural_net import NeuralNetAI
        from app.models import AIConfig, BoardType

        os.environ.setdefault("RINGRIFT_FORCE_CPU", "1")

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
        encoder.feature_version = int(feature_version)

        board_type = BoardType(board_type_str)
        encoder.board_size = {
            BoardType.SQUARE8: 8,
            BoardType.SQUARE19: 19,
            BoardType.HEX8: 9,
            BoardType.HEXAGONAL: 25,
        }.get(board_type, 8)

        _ENCODER_CACHE[cache_key] = encoder

    return _ENCODER_CACHE[cache_key]


# Heuristic extraction cache (lazy loaded in worker processes)
_HEURISTIC_EXTRACTOR_CACHE: dict[str, Any] = {}


def _get_heuristic_extractor(full_mode: bool = False):
    """Get cached heuristic extraction function.

    Args:
        full_mode: If True, use full 49-feature extraction. If False, use fast 21-feature.

    Returns:
        Tuple of (extractor_func, num_features)
    """
    cache_key = "full" if full_mode else "fast"
    if cache_key not in _HEURISTIC_EXTRACTOR_CACHE:
        try:
            from app.training.fast_heuristic_features import (
                extract_heuristic_features,
                extract_full_heuristic_features,
                NUM_HEURISTIC_FEATURES,
                NUM_HEURISTIC_FEATURES_FULL,
            )
            if full_mode:
                _HEURISTIC_EXTRACTOR_CACHE[cache_key] = (
                    extract_full_heuristic_features,
                    NUM_HEURISTIC_FEATURES_FULL,
                )
            else:
                _HEURISTIC_EXTRACTOR_CACHE[cache_key] = (
                    extract_heuristic_features,
                    NUM_HEURISTIC_FEATURES,
                )
        except ImportError:
            # Fallback if heuristic module not available
            _HEURISTIC_EXTRACTOR_CACHE[cache_key] = (None, 21 if not full_mode else 49)

    return _HEURISTIC_EXTRACTOR_CACHE[cache_key]


@dataclass
class EncodedSample:
    """A single encoded training sample."""
    features: np.ndarray  # (C, H, W)
    globals: np.ndarray  # (G,)
    value: float
    values_mp: np.ndarray  # (4,)
    policy_index: int
    move_number: int
    total_moves: int
    phase: str
    perspective: int
    num_players: int
    game_id: str = ""  # Added for NNUE compatibility
    move_type: str = "unknown"  # For chain-aware sample weighting
    heuristics: np.ndarray | None = None  # For v5_heavy heuristic conditioning


@dataclass
class GameEncodingResult:
    """Result of encoding a single game."""
    game_id: str
    samples: list[EncodedSample]
    error: str | None = None


def _validate_move_data(move: Any, move_idx: int, game_id: str) -> tuple[bool, str | None]:
    """Validate move data for corruption issues.

    December 2025: Added after hex8_4p data corruption incident where
    PLACE_RING moves had to=None due to phase tracking bug.

    Args:
        move: Move object to validate
        move_idx: Index in move sequence (for error reporting)
        game_id: Game ID (for error reporting)

    Returns:
        Tuple of (is_valid, error_message)
    """
    from app.models import MoveType

    move_type = getattr(move, "type", None)

    # PLACE_RING must have a valid 'to' position
    if move_type == MoveType.PLACE_RING:
        to_pos = getattr(move, "to", None)
        if to_pos is None:
            return False, f"Game {game_id} move {move_idx}: PLACE_RING has to=None (corrupted)"

    # MOVE_STACK/MOVE_RING should have both from and to positions
    if move_type in (MoveType.MOVE_STACK, MoveType.MOVE_RING):
        from_pos = getattr(move, "from_pos", None) or getattr(move, "from", None)
        to_pos = getattr(move, "to", None)
        if from_pos is None:
            return False, f"Game {game_id} move {move_idx}: {move_type.value} has from=None"
        if to_pos is None:
            return False, f"Game {game_id} move {move_idx}: {move_type.value} has to=None"

    return True, None


def _encode_single_game(
    game_data: dict[str, Any],
    board_type_str: str,
    num_players: int,
    encoder_version: str,
    feature_version: int,
    history_length: int,
    sample_every: int,
    use_board_aware_encoding: bool,
    include_heuristics: bool = False,
    full_heuristics: bool = False,
) -> GameEncodingResult:
    """
    Encode a single game into training samples.

    This function is designed to run in a worker process. It handles:
    1. Replaying the game from initial_state using moves
    2. Encoding each state into features
    3. Computing final_state for value targets (if not provided)
    4. Optionally extracting heuristic features for v5_heavy models

    Args:
        game_data: Dict containing 'initial_state', 'moves', 'game_id', 'final_state'
        board_type_str: Board type as string (e.g., "hexagonal")
        num_players: Number of players
        encoder_version: "v2" or "v3" for hex boards
        history_length: Number of history frames
        sample_every: Sample every Nth move
        use_board_aware_encoding: Use board-specific policy encoding
        include_heuristics: If True, extract heuristic features for each sample
        full_heuristics: If True, use full 49-feature extraction (slower). If False, use fast 21-feature.

    Returns:
        GameEncodingResult with encoded samples or error
    """
    from app.ai.neural_net import INVALID_MOVE_INDEX, encode_move_for_board
    from app.game_engine import GameEngine
    from app.models import BoardType

    game_id = game_data.get("game_id", "unknown")

    try:
        initial_state = game_data["initial_state"]
        moves = game_data["moves"]
        final_state = game_data.get("final_state")

        # If final_state not provided, we'll compute it during replay
        # This is the normal case for parallel encoding
        compute_final_state = final_state is None

        board_type = BoardType(board_type_str)
        is_hex = board_type in (BoardType.HEXAGONAL, BoardType.HEX8)

        # Fix board.size if incorrect (common issue with older JSONL data)
        # Hex boards should have bounding box size, not radius
        correct_sizes = {"hex8": 9, "hexagonal": 25, "square8": 8, "square19": 19}
        expected_size = correct_sizes.get(board_type_str, initial_state.board.size)
        if initial_state.board.size != expected_size:
            # Create corrected board and state
            corrected_board = initial_state.board.model_copy(update={"size": expected_size})
            initial_state = initial_state.model_copy(update={"board": corrected_board})

        # Get appropriate encoder
        if is_hex:
            encoder = _get_encoder(
                board_type_str,
                encoder_version=encoder_version,
                feature_version=feature_version,
            )
        else:
            encoder = _get_neural_net_encoder(
                board_type_str,
                feature_version=feature_version,
            )

        # Get heuristic extractor if needed
        heuristic_extractor = None
        num_heuristic_features = 0
        if include_heuristics:
            heuristic_extractor, num_heuristic_features = _get_heuristic_extractor(full_heuristics)
            if heuristic_extractor is None:
                logger.warning("Heuristic extraction requested but module not available")

        # First pass: replay game and collect samples (without values yet)
        # Tuple: (stacked, globals_vec, idx, move_idx, perspective, phase_str, num_players, move_type_str, heuristics)
        pending_samples: list[tuple[np.ndarray, np.ndarray, int, int, int, str, int, str, np.ndarray | None]] = []
        history_frames: list[np.ndarray] = []
        current_state = initial_state
        total_moves = len(moves)

        # December 2025: Validate all moves before encoding to catch corrupted data early.
        # This prevents training on games with corrupted move data (e.g., PLACE_RING with to=None)
        for move_idx, move in enumerate(moves):
            is_valid, error_msg = _validate_move_data(move, move_idx, game_id)
            if not is_valid:
                return GameEncodingResult(game_id=game_id, samples=[], error=error_msg)

        # December 2025 fix: Encode samples BEFORE apply_move to capture partial games.
        # Previously, when apply_move() failed, samples from that move were lost.
        # Now we encode first, then try to apply. This fixes hex8_4p export where
        # chain capture FSM mismatches caused 89% of games to fail mid-replay.
        replay_error_at_move: int | None = None

        for move_idx, move in enumerate(moves):
            state_before = current_state

            # Check if we should sample this move
            should_sample = sample_every == 1 or (move_idx % sample_every) == 0

            # PARTIAL SAMPLE FIX: Encode sample BEFORE applying move
            # This ensures we capture samples even if apply_move fails for this move
            if should_sample:
                # Encode state
                if is_hex:
                    features, globals_vec = encoder.encode_state(state_before)
                else:
                    features, globals_vec = encoder._extract_features(state_before)

                # Build stacked features with history
                stacked = _build_stacked_features(features, history_frames, history_length)

                # Update history for next iteration
                history_frames.append(features.copy())
                if len(history_frames) > history_length + 1:
                    history_frames.pop(0)

                # Encode move
                if use_board_aware_encoding:
                    idx = encode_move_for_board(move, state_before.board)
                elif is_hex:
                    idx = encoder.encode_move(move, state_before.board)
                else:
                    idx = encoder.encode_move(move, state_before.board)

                if idx != INVALID_MOVE_INDEX:
                    # Get phase string
                    phase_str = (
                        str(state_before.current_phase.value)
                        if hasattr(state_before.current_phase, "value")
                        else str(state_before.current_phase)
                    )

                    # Get move type for chain-aware sample weighting
                    move_type_raw = getattr(move, "type", None)
                    if hasattr(move_type_raw, "value"):
                        move_type_str = str(move_type_raw.value)
                    else:
                        move_type_str = str(move_type_raw) if move_type_raw else "unknown"

                    # Extract heuristic features if requested
                    heuristics_arr: np.ndarray | None = None
                    if heuristic_extractor is not None:
                        try:
                            # Jan 2026 fix: Pass player_number to heuristic extractors
                            # Both extract_heuristic_features and extract_full_heuristic_features
                            # require (game_state, player_number) signature
                            heuristics_arr = heuristic_extractor(
                                state_before, state_before.current_player
                            )
                        except Exception as e:
                            # Log but don't fail on heuristic extraction errors
                            logger.debug(f"Heuristic extraction failed: {e}")
                            heuristics_arr = np.zeros(num_heuristic_features, dtype=np.float32)

                    perspective = state_before.current_player
                    pending_samples.append((
                        stacked.astype(np.float32),
                        globals_vec.astype(np.float32),
                        idx,
                        move_idx,
                        perspective,
                        phase_str,
                        num_players,
                        move_type_str,
                        heuristics_arr,
                    ))

            # Now apply the move (AFTER encoding)
            try:
                current_state = GameEngine.apply_move(current_state, move, trace_mode=True)
            except (RuntimeError, ValueError):
                # Track where replay failed but keep samples collected so far
                replay_error_at_move = move_idx
                break

        # Use current_state as final_state if not provided (computed during replay)
        if compute_final_state:
            final_state = current_state

        # Set winner on final_state from game_data if available
        # (GameEngine.apply_move doesn't auto-detect game end and set winner)
        winner_from_db = game_data.get("winner")

        # December 2025 fix: Validate winner field for multiplayer games
        # Invalid winner values (outside 1..num_players range) caused corrupted training data
        # for hex8_4p and square19_3p models, contributing to Elo regression (594/409 from 1500)
        if winner_from_db is not None and num_players > 2:
            try:
                winner_int = int(winner_from_db)
                if winner_int < 1 or winner_int > num_players:
                    logger.warning(
                        f"Game {game_id}: Invalid winner {winner_int} for {num_players}p "
                        f"(must be 1-{num_players}), skipping"
                    )
                    return GameEncodingResult(game_id=game_id, samples=[])
            except (ValueError, TypeError):
                logger.warning(
                    f"Game {game_id}: Non-numeric winner '{winner_from_db}' for {num_players}p, skipping"
                )
                return GameEncodingResult(game_id=game_id, samples=[])

        if final_state is not None and winner_from_db is not None:
            # Create a copy with winner set (pydantic frozen model)
            final_state = final_state.model_copy(update={"winner": winner_from_db})

        # Skip games without valid winner - these produce value=0 which corrupts training
        if final_state is None or getattr(final_state, "winner", None) is None:
            if pending_samples:
                logger.debug(
                    f"Game {game_id}: Discarding {len(pending_samples)} samples - no valid winner"
                )
            return GameEncodingResult(game_id=game_id, samples=[])

        # Now compute values using final_state
        values_vec = np.zeros(4, dtype=np.float32)
        if final_state:
            values_vec = _compute_multi_player_values(final_state, num_players)

        # Build final samples with computed values
        samples: list[EncodedSample] = []
        for stacked, globals_vec, idx, move_idx, perspective, phase_str, n_players, move_type_str, heuristics_arr in pending_samples:
            value = _value_from_final_ranking(final_state, perspective, num_players) if final_state else 0.0

            samples.append(EncodedSample(
                features=stacked,
                globals=globals_vec,
                value=float(value),
                values_mp=values_vec,
                policy_index=idx,
                move_number=move_idx,
                total_moves=total_moves,
                phase=phase_str,
                perspective=perspective,
                num_players=n_players,
                game_id=game_id,
                move_type=move_type_str,
                heuristics=heuristics_arr,
            ))

        # Track partial games (replay error but samples collected)
        partial_info = None
        if replay_error_at_move is not None and samples:
            partial_info = f"partial:{replay_error_at_move}/{total_moves}"
            logger.debug(
                f"Game {game_id}: Partial replay - {len(samples)} samples from "
                f"{replay_error_at_move}/{total_moves} moves (error at move {replay_error_at_move})"
            )

        return GameEncodingResult(game_id=game_id, samples=samples, error=partial_info)

    except Exception as e:
        return GameEncodingResult(game_id=game_id, samples=[], error=str(e))


def _build_stacked_features(
    current: np.ndarray,
    history: list[np.ndarray],
    history_length: int,
) -> np.ndarray:
    """Build stacked features with history frames."""
    hist = list(reversed(history))[:history_length]

    # Pad with zeros if needed
    while len(hist) < history_length:
        hist.append(np.zeros_like(current))

    return np.concatenate([current, *hist], axis=0)


def _compute_multi_player_values(
    final_state: GameState,
    num_players: int,
    max_players: int = 4,
) -> np.ndarray:
    """Compute value vector for all player positions.

    IMPORTANT: This iterates over ALL expected players (1 to num_players), not
    just those remaining in final_state.players. Eliminated players are assigned
    the worst rank (last place = -1.0 value).
    """
    values = np.zeros(max_players, dtype=np.float32)

    winner = getattr(final_state, "winner", None)
    if winner is None or not final_state.players:
        return values

    # Build a map of active players (those still in final_state.players)
    active_player_nums = {p.player_number for p in final_state.players}

    # Compute ranking for active players based on eliminated_rings and territory_spaces
    player_scores = []
    for player in final_state.players:
        score = (player.eliminated_rings, player.territory_spaces)
        player_scores.append((player.player_number, score))

    player_scores.sort(key=lambda x: x[1], reverse=True)

    player_ranks: dict[int, int] = {}
    for rank, (player_num, _) in enumerate(player_scores, start=1):
        player_ranks[player_num] = rank

    # Iterate over ALL expected players, not just active ones
    for player_num in range(1, num_players + 1):
        player_idx = player_num - 1
        if player_idx >= max_players:
            continue

        if player_num in active_player_nums:
            # Active player - use computed rank
            rank = player_ranks.get(player_num, num_players)
        else:
            # Eliminated player - worst rank (last place)
            rank = num_players

        if num_players <= 1:
            values[player_idx] = 0.0
        else:
            values[player_idx] = 1.0 - 2.0 * (rank - 1) / (num_players - 1)

    return values


def _value_from_final_ranking(
    final_state: GameState,
    perspective: int,
    num_players: int,
) -> float:
    """Compute rank-aware value from final game state.

    IMPORTANT: Handles eliminated players correctly - if the perspective player
    is not in final_state.players, they were eliminated and get worst rank (-1.0).
    """
    winner = getattr(final_state, "winner", None)
    if winner is None or not final_state.players:
        return 0.0

    if num_players == 2:
        return 1.0 if winner == perspective else -1.0

    # Check if perspective player is still active
    active_player_nums = {p.player_number for p in final_state.players}
    if perspective not in active_player_nums:
        # Eliminated player - worst rank (last place = -1.0)
        return -1.0

    player_scores = []
    for player in final_state.players:
        score = (player.eliminated_rings, player.territory_spaces)
        player_scores.append((player.player_number, score))

    player_scores.sort(key=lambda x: x[1], reverse=True)

    rank = 1
    for i, (player_num, _) in enumerate(player_scores):
        if player_num == perspective:
            rank = i + 1
            break

    if num_players <= 1:
        return 0.0

    return 1.0 - 2.0 * (rank - 1) / (num_players - 1)


class ParallelEncoder:
    """
    Parallel encoder for batch processing of games.

    Uses ProcessPoolExecutor for true parallelism across CPU cores.
    """

    def __init__(
        self,
        board_type: BoardType,
        num_workers: int | None = None,
        encoder_version: str = "v3",
        feature_version: int = 2,
        history_length: int = 3,
        sample_every: int = 1,
        use_board_aware_encoding: bool = False,
        include_heuristics: bool = False,
        full_heuristics: bool = False,
    ):
        """
        Initialize the parallel encoder.

        Args:
            board_type: Board type enum
            num_workers: Number of worker processes (default: CPU count - 1)
            encoder_version: "v2" or "v3" for hex boards
            feature_version: Feature encoding version for global feature layout
            history_length: Number of history frames to stack
            sample_every: Sample every Nth move
            use_board_aware_encoding: Use board-specific policy encoding
            include_heuristics: If True, extract heuristic features for v5_heavy training
            full_heuristics: If True, use full 49-feature extraction. If False, use fast 21-feature.
        """

        self.board_type = board_type
        self.board_type_str = board_type.value
        self.encoder_version = encoder_version
        self.feature_version = int(feature_version)
        self.history_length = history_length
        self.sample_every = sample_every
        self.use_board_aware_encoding = use_board_aware_encoding
        self.include_heuristics = include_heuristics
        self.full_heuristics = full_heuristics

        # Determine worker count
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        # February 2026: Cap workers on coordinator to prevent OOM
        if os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() == "true":
            num_workers = min(num_workers, 2)
        self.num_workers = num_workers

        # Create process pool
        self._executor: ProcessPoolExecutor | None = None

    def _get_executor(self) -> ProcessPoolExecutor:
        """Get or create the process pool executor."""
        if self._executor is None:
            # Use 'fork' on Linux for better performance (no re-import overhead)
            # Fall back to 'spawn' on Windows/macOS
            import platform
            if platform.system() == "Linux":
                ctx = mp.get_context("fork")
            else:
                ctx = mp.get_context("spawn")
            self._executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=ctx,
            )
        return self._executor

    def encode_games_batch(
        self,
        games: list[dict[str, Any]],
        num_players: int,
        show_progress: bool = True,
    ) -> tuple[list[EncodedSample], list[str]]:
        """
        Encode a batch of games in parallel.

        Args:
            games: List of game dicts with 'initial_state', 'moves', 'game_id', 'final_state'
            num_players: Number of players
            show_progress: Print progress updates

        Returns:
            Tuple of (all_samples, errors)
        """
        if not games:
            return [], []

        executor = self._get_executor()

        # Create partial function with fixed args
        encode_fn = partial(
            _encode_single_game,
            board_type_str=self.board_type_str,
            num_players=num_players,
            encoder_version=self.encoder_version,
            feature_version=self.feature_version,
            history_length=self.history_length,
            sample_every=self.sample_every,
            use_board_aware_encoding=self.use_board_aware_encoding,
            include_heuristics=self.include_heuristics,
            full_heuristics=self.full_heuristics,
        )

        all_samples: list[EncodedSample] = []
        errors: list[str] = []

        # Use map for simpler, more reliable parallel processing
        # Process in fixed chunks for better cache locality (Dec 2025)
        # Fixed chunk size instead of dynamic % for consistent performance
        chunk_size = int(os.getenv("RINGRIFT_ENCODING_CHUNK_SIZE", "64"))
        total_completed = 0

        for i in range(0, len(games), chunk_size):
            chunk = games[i:i + chunk_size]

            try:
                results = list(executor.map(encode_fn, chunk, timeout=300))

                for result in results:
                    if result.error:
                        errors.append(f"Game {result.game_id}: {result.error}")
                    else:
                        all_samples.extend(result.samples)

                total_completed += len(chunk)
                if show_progress:
                    logger.info(
                        f"Encoded {total_completed}/{len(games)} games, "
                        f"{len(all_samples)} samples so far"
                    )
            except Exception as e:
                errors.append(f"Chunk error: {e}")
                logger.warning(f"Chunk {i//chunk_size} failed: {e}")

        if show_progress:
            logger.info(
                f"Completed: {len(games)} games -> {len(all_samples)} samples "
                f"({len(errors)} errors)"
            )

        return all_samples, errors

    def encode_games_streaming(
        self,
        games_iter,
        num_players: int,
        batch_size: int = 100,
        callback=None,
    ):
        """
        Encode games in streaming fashion with batched parallelism.

        Args:
            games_iter: Iterator of game dicts
            num_players: Number of players
            batch_size: Number of games per batch
            callback: Optional callback(samples, batch_idx) called after each batch

        Yields:
            EncodedSample for each sample
        """
        batch = []
        batch_idx = 0

        for game in games_iter:
            batch.append(game)

            if len(batch) >= batch_size:
                samples, _errors = self.encode_games_batch(
                    batch, num_players, show_progress=False
                )
                if callback:
                    callback(samples, batch_idx)
                yield from samples
                batch = []
                batch_idx += 1

        # Process remaining
        if batch:
            samples, _errors = self.encode_games_batch(
                batch, num_players, show_progress=False
            )
            if callback:
                callback(samples, batch_idx)
            yield from samples

    def shutdown(self):
        """Shutdown the executor pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def samples_to_arrays(
    samples: list[EncodedSample],
) -> dict[str, np.ndarray]:
    """
    Convert a list of EncodedSample to numpy arrays suitable for NPZ.

    Returns dict with keys:
        features, globals, values, values_mp, policy_indices, policy_values,
        move_numbers, total_game_moves, phases, num_players,
        player_numbers, game_ids (for NNUE compatibility),
        heuristics (if present in samples, for v5_heavy training)
    """
    if not samples:
        return {}

    features = np.stack([s.features for s in samples], axis=0)
    globals_arr = np.stack([s.globals for s in samples], axis=0)
    values = np.array([s.value for s in samples], dtype=np.float32)
    values_mp = np.stack([s.values_mp for s in samples], axis=0)
    policy_indices = np.array(
        [np.array([s.policy_index], dtype=np.int32) for s in samples],
        dtype=object,
    )
    policy_values = np.array(
        [np.array([1.0], dtype=np.float32) for s in samples],
        dtype=object,
    )
    move_numbers = np.array([s.move_number for s in samples], dtype=np.int32)
    total_game_moves = np.array([s.total_moves for s in samples], dtype=np.int32)
    phases = np.array([s.phase for s in samples], dtype=object)
    num_players = np.array([s.num_players for s in samples], dtype=np.int32)
    # NNUE-compatible fields
    player_numbers = np.array([s.perspective for s in samples], dtype=np.int32)
    game_ids = np.array([s.game_id for s in samples], dtype=object)
    # Chain-aware sample weighting
    move_types = np.array([s.move_type for s in samples], dtype=object)

    result = {
        "features": features,
        "globals": globals_arr,
        "values": values,
        "values_mp": values_mp,
        "policy_indices": policy_indices,
        "policy_values": policy_values,
        "move_numbers": move_numbers,
        "total_game_moves": total_game_moves,
        "phases": phases,
        "num_players": num_players,
        # NNUE-compatible fields
        "player_numbers": player_numbers,
        "game_ids": game_ids,
        # Chain-aware sample weighting
        "move_types": move_types,
    }

    # Add heuristics if present in samples (for v5_heavy training)
    if samples and samples[0].heuristics is not None:
        heuristics_list = [
            s.heuristics if s.heuristics is not None
            else np.zeros_like(samples[0].heuristics)
            for s in samples
        ]
        result["heuristics"] = np.stack(heuristics_list, axis=0)

    return result


# Convenience function for direct use
def parallel_encode_games(
    games: list[dict[str, Any]],
    board_type: BoardType,
    num_players: int,
    num_workers: int | None = None,
    encoder_version: str = "v3",
    feature_version: int = 2,
    history_length: int = 3,
    sample_every: int = 1,
    use_board_aware_encoding: bool = False,
    include_heuristics: bool = False,
    full_heuristics: bool = False,
) -> dict[str, np.ndarray]:
    """
    Convenience function to parallel encode games and return arrays.

    Args:
        games: List of game dicts
        board_type: Board type enum
        num_players: Number of players
        num_workers: Number of workers (default: CPU count - 1)
        encoder_version: Encoder version for hex boards
        feature_version: Feature encoding version for global feature layout
        history_length: History frame count
        sample_every: Sample every Nth move
        use_board_aware_encoding: Use board-specific encoding
        include_heuristics: If True, extract heuristic features for v5_heavy training
        full_heuristics: If True, use full 49-feature extraction. If False, use fast 21-feature.

    Returns:
        Dict of numpy arrays ready for np.savez_compressed()
    """
    with ParallelEncoder(
        board_type=board_type,
        num_workers=num_workers,
        encoder_version=encoder_version,
        feature_version=feature_version,
        history_length=history_length,
        sample_every=sample_every,
        use_board_aware_encoding=use_board_aware_encoding,
        include_heuristics=include_heuristics,
        full_heuristics=full_heuristics,
    ) as encoder:
        samples, errors = encoder.encode_games_batch(games, num_players)

        if errors:
            logger.warning(f"Encoding errors: {len(errors)}")
            for err in errors[:5]:
                logger.warning(f"  {err}")

        return samples_to_arrays(samples)
