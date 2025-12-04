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
from typing import Dict, List, Optional, Any

from app.db import GameReplayDB, GameWriter
from app.models import GameState, Move


# -----------------------------------------------------------------------------
# Environment variable configuration
# -----------------------------------------------------------------------------

# Default database path when none is specified
DEFAULT_SELFPLAY_DB_PATH = "data/games/selfplay.db"


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

    def add_move(self, move: Move) -> None:
        """Add a move to the game record."""
        if self._writer is None:
            raise RuntimeError("GameRecorder not entered as context manager")
        self._writer.add_move(move)

    def finalize(
        self,
        final_state: GameState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Finalize the game recording with the final state and metadata."""
        if self._writer is None:
            raise RuntimeError("GameRecorder not entered as context manager")
        self._writer.finalize(final_state, metadata)
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

    db.store_game(
        game_id=gid,
        initial_state=initial_for_recording,
        final_state=final_state,
        moves=moves,
        metadata=metadata,
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
