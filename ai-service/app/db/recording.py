"""Game recording utilities for integrating GameReplayDB into scripts.

This module provides high-level helper functions for recording games to the
SQLite database. It is intended to be used by self-play, training, and
analysis scripts.

Environment Variables
---------------------
RINGRIFT_RECORD_SELFPLAY_GAMES
    Enable/disable game recording globally. Set to "false", "0", or "no" to
    disable recording. Default: enabled (True).

RINGRIFT_SELFPLAY_DB_PATH
    Default path for the selfplay games database when no explicit path is
    provided. Default: "data/games/selfplay.db"
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from app.db import GameReplayDB, GameWriter
from app.models import GameState, Move
from app.rules.fsm import validate_move_for_phase, FSMValidationResult


# -----------------------------------------------------------------------------
# Environment variable configuration
# -----------------------------------------------------------------------------

# Default database path when none is specified
DEFAULT_SELFPLAY_DB_PATH = "data/games/selfplay.db"


def _augment_metadata_with_env(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of metadata enriched with environment-provided tags.

    This centralises common recording metadata such as engine/service
    versions so that all callers (self-play soaks, training harnesses,
    optimisation scripts) benefit without having to thread these fields
    manually at every call site.
    """
    base: Dict[str, Any] = dict(metadata or {})

    env_to_key = [
        ("RINGRIFT_RULES_ENGINE_VERSION", "rules_engine_version"),
        ("RINGRIFT_TS_ENGINE_VERSION", "ts_engine_version"),
        ("RINGRIFT_AI_SERVICE_VERSION", "ai_service_version"),
    ]

    for env_var, meta_key in env_to_key:
        value = os.environ.get(env_var)
        if value and meta_key not in base:
            base[meta_key] = value

    return base


def is_recording_enabled() -> bool:
    """Check if game recording is enabled via environment variable.

    Recording is enabled by default. Set RINGRIFT_RECORD_SELFPLAY_GAMES to
    "false", "0", or "no" to disable.
    """
    env_val = os.environ.get("RINGRIFT_RECORD_SELFPLAY_GAMES", "true").lower()
    return env_val not in ("false", "0", "no", "off", "disabled")


def get_default_db_path() -> str:
    """Get the default database path from environment or fallback.

    Returns the value of RINGRIFT_SELFPLAY_DB_PATH if set, otherwise
    DEFAULT_SELFPLAY_DB_PATH.
    """
    return os.environ.get("RINGRIFT_SELFPLAY_DB_PATH", DEFAULT_SELFPLAY_DB_PATH)


def validate_move_fsm(
    state: GameState,
    move: Move,
) -> Tuple[bool, Optional[str]]:
    """Validate a move against the FSM and return validation result.

    This is a convenience function for recording FSM validation status
    in selfplay databases. It wraps the FSM validation logic and returns
    a simple (valid, error_code) tuple suitable for storage.

    Args:
        state: GameState before the move
        move: Move to validate

    Returns:
        Tuple of (is_valid, error_code) where error_code is None if valid
    """
    result = validate_move_for_phase(state.current_phase, move, state)
    if result.ok:
        return (True, None)
    return (False, result.code)


class GameRecorder:
    """Context manager for recording a single game to the database.

    Usage:
        db = GameReplayDB("data/games.db")
        with GameRecorder(db, initial_state) as recorder:
            for move in game_loop():
                recorder.add_move(move)
            recorder.finalize(final_state, {"source": "self_play"})
    """

    def __init__(
        self,
        db: GameReplayDB,
        initial_state: GameState,
        game_id: Optional[str] = None,
    ):
        self.db = db
        self.initial_state = initial_state
        self.game_id = game_id or str(uuid.uuid4())
        self._writer: Optional[GameWriter] = None
        self._finalized = False

    def __enter__(self) -> "GameRecorder":
        # For recording we want the "initial" state to represent the start
        # of the recorded sequence, not a mid-game snapshot with an existing
        # move_history. To make downstream replay/debugging consistent, we
        # clear any pre-populated move history before storing the initial
        # state in GameReplayDB.
        initial_for_recording = (
            self.initial_state.model_copy(deep=True) if self.initial_state is not None else None
        )
        if initial_for_recording is not None and getattr(
            initial_for_recording, "move_history", None
        ):
            initial_for_recording.move_history = []

        self._writer = self.db.store_game_incremental(
            self.game_id,
            initial_for_recording,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self._writer is not None:
            # Exception occurred - abort the game recording
            self._writer.abort()
        elif not self._finalized and self._writer is not None:
            # Context exited without finalizing - abort
            self._writer.abort()
        return False  # Don't suppress exceptions

    def add_move(
        self,
        move: Move,
        state_after: Optional["GameState"] = None,
        state_before: Optional["GameState"] = None,
        available_moves: Optional[List[Move]] = None,
        available_moves_count: Optional[int] = None,
        engine_eval: Optional[float] = None,
        engine_depth: Optional[int] = None,
        fsm_valid: Optional[bool] = None,
        fsm_error_code: Optional[str] = None,
    ) -> None:
        """Add a move to the game record.

        Args:
            move: The move that was played
            state_after: Optional state after the move (enables history entry storage
                         with phase transitions and state hashes for parity debugging)
            state_before: Optional state before the move (uses tracked previous state
                          if not provided)
            available_moves: Optional list of valid moves at state_before (for parity debugging)
            available_moves_count: Optional count of valid moves (lightweight alternative)
            engine_eval: Optional evaluation score from AI engine
            engine_depth: Optional search depth from AI engine
            fsm_valid: Optional FSM validation result (True = valid, False = invalid)
            fsm_error_code: Optional FSM error code if validation failed
        """
        if self._writer is None:
            raise RuntimeError("GameRecorder not entered as context manager")
        self._writer.add_move(
            move,
            state_after=state_after,
            state_before=state_before,
            available_moves=available_moves,
            available_moves_count=available_moves_count,
            engine_eval=engine_eval,
            engine_depth=engine_depth,
            fsm_valid=fsm_valid,
            fsm_error_code=fsm_error_code,
        )

    def finalize(
        self,
        final_state: GameState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Finalize the game recording with the final state and metadata."""
        if self._writer is None:
            raise RuntimeError("GameRecorder not entered as context manager")
        enriched_metadata = _augment_metadata_with_env(metadata)
        self._writer.finalize(final_state, enriched_metadata)
        self._finalized = True


def record_completed_game(
    db: GameReplayDB,
    initial_state: GameState,
    final_state: GameState,
    moves: List[Move],
    metadata: Optional[Dict[str, Any]] = None,
    game_id: Optional[str] = None,
    store_history_entries: bool = True,
    snapshot_interval: Optional[int] = None,
) -> str:
    """Record a completed game in one shot.

    This is a convenience function for scripts that collect moves in a list
    and want to store them all at once after the game ends.

    Args:
        db: The GameReplayDB instance
        initial_state: GameState at the start of the game
        final_state: GameState at the end of the game
        moves: List of all moves in the game
        metadata: Optional metadata dict (source, difficulty, etc.)
        game_id: Optional custom game ID
        store_history_entries: If True (default), store full before/after state
            snapshots for each move. Set to False for lean recording (~100x smaller)
            that still stores initial state, moves, and final state for training.
        snapshot_interval: Store snapshots every N moves for NNUE training.
            Defaults to 20 (via RINGRIFT_SNAPSHOT_INTERVAL env var).
            Set to 0 to disable periodic snapshots.

    Returns:
        The game ID that was stored
    """
    gid = game_id or str(uuid.uuid4())

    # Determine snapshot interval (default to 20 for NNUE training support)
    if snapshot_interval is None:
        snapshot_interval = int(os.environ.get("RINGRIFT_SNAPSHOT_INTERVAL", "20"))

    # As with GameRecorder, ensure the stored "initial" state does not carry
    # a pre-populated move_history from a longer game. For replay and
    # sandbox parity we treat the initial snapshot as the start of the
    # recorded sequence and rely on game_moves for the full trajectory.
    initial_for_recording = initial_state.model_copy(deep=True)
    if getattr(initial_for_recording, "move_history", None):
        initial_for_recording.move_history = []

    enriched_metadata = _augment_metadata_with_env(metadata)

    db.store_game(
        game_id=gid,
        initial_state=initial_for_recording,
        final_state=final_state,
        moves=moves,
        metadata=enriched_metadata,
        store_history_entries=store_history_entries,
        snapshot_interval=snapshot_interval,
    )
    return gid


def get_or_create_db(
    db_path: Optional[str] = None,
    default_path: Optional[str] = None,
    respect_env_disable: bool = True,
    enforce_canonical_history: bool = True,
) -> Optional[GameReplayDB]:
    """Get or create a GameReplayDB instance.

    This function respects the RINGRIFT_RECORD_SELFPLAY_GAMES and
    RINGRIFT_SELFPLAY_DB_PATH environment variables.

    Args:
        db_path: Path to the database file. If None, uses environment default.
                 Pass explicit empty string "" to disable recording for this call.
        default_path: Fallback default path if db_path is None/empty. If not
                      provided, uses RINGRIFT_SELFPLAY_DB_PATH or built-in default.
        respect_env_disable: If True (default), returns None if
                             RINGRIFT_RECORD_SELFPLAY_GAMES is set to disable.
                             Set to False to ignore the global disable flag.
        enforce_canonical_history: If True (default), validate that recorded moves
                                   match canonical phase expectations. Set to False
                                   for training data collection where phase alignment
                                   may differ from TS canonical rules.

    Returns:
        GameReplayDB instance or None if recording is disabled
    """
    # Check global env disable flag
    if respect_env_disable and not is_recording_enabled():
        return None

    # Explicit empty string means disable for this call
    if db_path == "":
        return None

    # Determine path to use
    if db_path:
        path = db_path
    elif default_path:
        path = default_path
    else:
        path = get_default_db_path()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return GameReplayDB(path, enforce_canonical_history=enforce_canonical_history)


def should_record_games(cli_no_record: bool = False) -> bool:
    """Determine if games should be recorded based on CLI and env settings.

    This is a convenience function for CLI scripts to combine the --no-record
    flag with the environment variable.

    Args:
        cli_no_record: Value of the --no-record CLI flag (True = don't record)

    Returns:
        True if games should be recorded, False otherwise
    """
    if cli_no_record:
        return False
    return is_recording_enabled()


def record_completed_game_with_parity_check(
    db: GameReplayDB,
    initial_state: GameState,
    final_state: GameState,
    moves: List[Move],
    metadata: Optional[Dict[str, Any]] = None,
    game_id: Optional[str] = None,
    parity_mode: Optional[str] = None,
    store_history_entries: bool = True,
    snapshot_interval: Optional[int] = None,
) -> str:
    """Record a completed game and optionally validate parity with TS engine.

    This is the same as record_completed_game but adds on-the-fly parity
    validation after recording. The parity check is controlled by the
    RINGRIFT_PARITY_VALIDATION environment variable or the parity_mode argument.

    Args:
        db: The GameReplayDB instance
        initial_state: GameState at the start of the game
        final_state: GameState at the end of the game
        moves: List of all moves in the game
        metadata: Optional metadata dict (source, difficulty, etc.)
        game_id: Optional custom game ID
        parity_mode: Override parity validation mode:
            - None: use RINGRIFT_PARITY_VALIDATION env var
            - "off": skip parity validation
            - "warn": log warnings on divergence
            - "strict": raise exception on divergence
        store_history_entries: If True (default), store full before/after state
            snapshots for each move. Set to False for lean recording (~100x smaller)
            that still stores initial state, moves, and final state for training.
        snapshot_interval: Store snapshots every N moves for NNUE training.
            Defaults to 20 (via RINGRIFT_SNAPSHOT_INTERVAL env var).
            Set to 0 to disable periodic snapshots.

    Returns:
        The game ID that was stored

    Raises:
        ParityValidationError: If parity_mode is 'strict' and divergence is found
    """
    # First record the game
    gid = record_completed_game(
        db=db,
        initial_state=initial_state,
        final_state=final_state,
        moves=moves,
        metadata=metadata,
        game_id=game_id,
        store_history_entries=store_history_entries,
        snapshot_interval=snapshot_interval,
    )

    # Then validate parity if enabled
    from app.db.parity_validator import (
        validate_game_parity,
        get_parity_mode,
        ParityMode,
        ParityValidationError,
    )

    effective_mode = parity_mode or get_parity_mode()
    if effective_mode == ParityMode.OFF:
        return gid

    # Get the db path from the instance and normalise to an absolute path so
    # TS replay harnesses invoked from the monorepo root can locate the file.
    raw_db_path = getattr(db, 'db_path', None) or getattr(db, '_db_path', None)
    if raw_db_path is None:
        return gid
    db_path = Path(raw_db_path).resolve()

    # Run parity validation (will raise on strict mode with divergence)
    try:
        validate_game_parity(str(db_path), gid, mode=effective_mode)
    except ParityValidationError:
        # Parity failure: remove the just-recorded game so it never lingers
        # in the DB. This keeps canonical/self-play DBs clean while still
        # allowing long soak runs to continue.
        try:
            db.delete_game(gid)
        except Exception:
            # Best-effort cleanup; swallow any deletion failure.
            pass
        # Re-raise so callers can decide whether to halt or skip.
        raise

    return gid


# -----------------------------------------------------------------------------
# NNUE Feature Caching
# -----------------------------------------------------------------------------

def cache_nnue_features_for_game(
    db: GameReplayDB,
    game_id: str,
    sample_every_n_moves: int = 1,
    skip_if_cached: bool = True,
) -> int:
    """Extract and cache NNUE features for a completed game.

    This replays the game and extracts features at each position, then
    stores them in the game_nnue_features table for instant training access.

    Args:
        db: The GameReplayDB instance
        game_id: ID of the game to process
        sample_every_n_moves: Sample positions every N moves (default 1 = every move)
        skip_if_cached: If True, skip if features already cached for this game

    Returns:
        Number of feature records cached
    """
    if skip_if_cached and db.has_nnue_features(game_id):
        return 0

    # Import here to avoid circular imports
    from app.ai.nnue import extract_features_from_gamestate
    from app.game_engine import GameEngine
    import numpy as np

    # Get game metadata
    games = db.list_games(filters={"game_id": game_id})
    if not games:
        raise ValueError(f"Game {game_id} not found")

    game = games[0]
    board_type = game.get("board_type", "hexagonal")
    num_players = game.get("num_players", 2)
    winner = game.get("winner")

    if winner is None:
        # Skip games without a winner
        return 0

    # Get initial state and moves
    initial_state = db.get_initial_state(game_id)
    moves = db.get_moves(game_id)

    if not moves:
        return 0

    # Replay the game and extract features at each position
    engine = GameEngine(initial_state)
    records = []

    for move_num, move in enumerate(moves):
        if move_num % sample_every_n_moves == 0:
            state = engine.get_state()

            # Extract features from each player's perspective
            for player in range(1, num_players + 1):
                try:
                    features = extract_features_from_gamestate(state, player)

                    # Skip if features are all zeros
                    if np.count_nonzero(features) == 0:
                        continue

                    # Compute value label: +1 if this player won, -1 if lost, 0 for draw
                    if winner == player:
                        value = 1.0
                    elif winner == 0:
                        value = 0.0  # Draw
                    else:
                        value = -1.0

                    records.append((
                        game_id,
                        move_num,
                        player,
                        features,
                        value,
                        board_type,
                    ))
                except Exception:
                    # Skip positions that fail feature extraction
                    continue

        # Apply the move
        try:
            engine.apply_move(move)
        except Exception:
            # Stop if replay fails
            break

    # Store all records in a batch
    if records:
        db.store_nnue_features_batch(records)

    return len(records)


def cache_nnue_features_batch(
    db: GameReplayDB,
    game_ids: Optional[List[str]] = None,
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
    sample_every_n_moves: int = 1,
    limit: Optional[int] = None,
    skip_if_cached: bool = True,
) -> Tuple[int, int]:
    """Batch cache NNUE features for multiple games.

    Args:
        db: The GameReplayDB instance
        game_ids: Optional list of specific game IDs to process
        board_type: Optional filter by board type
        num_players: Optional filter by number of players
        sample_every_n_moves: Sample positions every N moves
        limit: Max number of games to process
        skip_if_cached: If True, skip games with existing cached features

    Returns:
        Tuple of (games_processed, features_cached)
    """
    import logging
    logger = logging.getLogger(__name__)

    if game_ids is None:
        # Query games from database
        filters = {}
        if board_type:
            filters["board_type"] = board_type
        if num_players:
            filters["num_players"] = num_players

        games = db.list_games(filters=filters, limit=limit or 10000)
        game_ids = [g["game_id"] for g in games if g.get("winner") is not None]

    total_features = 0
    games_processed = 0

    for gid in game_ids:
        try:
            count = cache_nnue_features_for_game(
                db,
                gid,
                sample_every_n_moves=sample_every_n_moves,
                skip_if_cached=skip_if_cached,
            )
            if count > 0:
                total_features += count
                games_processed += 1
                if games_processed % 100 == 0:
                    logger.info(f"Cached features for {games_processed} games ({total_features} records)")
        except Exception as e:
            logger.warning(f"Failed to cache features for game {gid}: {e}")
            continue

    logger.info(f"Finished: cached {total_features} features from {games_processed} games")
    return games_processed, total_features


def record_completed_game_with_nnue_cache(
    db: GameReplayDB,
    initial_state: GameState,
    final_state: GameState,
    moves: List[Move],
    metadata: Optional[Dict[str, Any]] = None,
    game_id: Optional[str] = None,
    store_history_entries: bool = True,
    snapshot_interval: Optional[int] = None,
    cache_nnue_features: bool = True,
    sample_every_n_moves: int = 1,
) -> str:
    """Record a completed game and cache NNUE features in one call.

    This combines record_completed_game with automatic NNUE feature extraction,
    eliminating the need for replay during training.

    Args:
        db: The GameReplayDB instance
        initial_state: GameState at the start of the game
        final_state: GameState at the end of the game
        moves: List of all moves in the game
        metadata: Optional metadata dict
        game_id: Optional custom game ID
        store_history_entries: If True, store full before/after state snapshots
        snapshot_interval: Store snapshots every N moves
        cache_nnue_features: If True (default), extract and cache NNUE features
        sample_every_n_moves: Sample positions every N moves for NNUE features

    Returns:
        The game ID that was stored
    """
    # Record the game
    gid = record_completed_game(
        db=db,
        initial_state=initial_state,
        final_state=final_state,
        moves=moves,
        metadata=metadata,
        game_id=game_id,
        store_history_entries=store_history_entries,
        snapshot_interval=snapshot_interval,
    )

    # Cache NNUE features if requested and game has a winner
    if cache_nnue_features and final_state.winner is not None:
        try:
            cache_nnue_features_for_game(
                db,
                gid,
                sample_every_n_moves=sample_every_n_moves,
                skip_if_cached=False,  # Just recorded, so no cache yet
            )
        except Exception:
            # Don't fail the recording if feature caching fails
            pass

    return gid
