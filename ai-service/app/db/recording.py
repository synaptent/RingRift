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
from typing import Dict, List, Optional, Any

from app.db import GameReplayDB, GameWriter
from app.models import GameState, Move


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

    Returns:
        The game ID that was stored
    """
    gid = game_id or str(uuid.uuid4())

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
    )
    return gid


def get_or_create_db(
    db_path: Optional[str] = None,
    default_path: Optional[str] = None,
    respect_env_disable: bool = True,
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
    return GameReplayDB(path)


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
    )

    # Then validate parity if enabled
    from app.db.parity_validator import (
        validate_game_parity,
        is_parity_validation_enabled,
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
