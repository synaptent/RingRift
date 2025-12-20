"""Unified game recording with canonical naming.

This module provides the CANONICAL interface for recording self-play games
across all sources (self-play, soak, CMA-ES, gauntlet, tournament scripts).

MERGED: All functionality from recording.py has been consolidated here (Dec 2025).

CANONICAL FORMAT:
    - Board types: "square8", "square19", "hex8", "hexagonal"
    - Config keys: "{board_type}_{num_players}p" (e.g., "square8_2p")
    - Database names: "{prefix}_{config_key}.db" (e.g., "selfplay_square8_2p.db")

Environment Variables
---------------------
RINGRIFT_RECORD_SELFPLAY_GAMES
    Enable/disable game recording globally. Set to "false", "0", or "no" to
    disable recording. Default: enabled (True).

RINGRIFT_SELFPLAY_DB_PATH
    Default path for the selfplay games database when no explicit path is
    provided. Default: "data/games/selfplay.db"

Usage:
    from app.db.unified_recording import UnifiedGameRecorder, RecordingConfig

    config = RecordingConfig(
        board_type="sq8",  # Will be normalized to "square8"
        num_players=2,
        source="self_play",
    )

    with UnifiedGameRecorder(config, initial_state) as recorder:
        for move in game_loop():
            recorder.add_move(move, state_after)
        recorder.finalize(final_state)
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.db.game_replay import GameReplayDB, GameWriter
from app.models import BoardType, GameState, Move
from app.rules.fsm import validate_move_for_phase
from app.utils.canonical_naming import (
    get_board_type_enum,
    make_config_key,
    normalize_board_type,
    normalize_database_filename,
)

# -----------------------------------------------------------------------------
# Environment variable configuration (from recording.py)
# -----------------------------------------------------------------------------

# Default database path when none is specified
DEFAULT_SELFPLAY_DB_PATH = "data/games/selfplay.db"


def _augment_metadata_with_env(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of metadata enriched with environment-provided tags.

    This centralises common recording metadata such as engine/service
    versions so that all callers (self-play soaks, training harnesses,
    optimisation scripts) benefit without having to thread these fields
    manually at every call site.
    """
    base: dict[str, Any] = dict(metadata or {})

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
) -> tuple[bool, str | None]:
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


# -----------------------------------------------------------------------------
# GameRecorder (from recording.py)
# -----------------------------------------------------------------------------


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
        game_id: str | None = None,
    ):
        self.db = db
        self.initial_state = initial_state
        self.game_id = game_id or str(uuid.uuid4())
        self._writer: GameWriter | None = None
        self._finalized = False

    def __enter__(self) -> GameRecorder:
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
        state_after: GameState | None = None,
        state_before: GameState | None = None,
        available_moves: list[Move] | None = None,
        available_moves_count: int | None = None,
        engine_eval: float | None = None,
        engine_depth: int | None = None,
        fsm_valid: bool | None = None,
        fsm_error_code: str | None = None,
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Finalize the game recording with the final state and metadata."""
        if self._writer is None:
            raise RuntimeError("GameRecorder not entered as context manager")
        enriched_metadata = _augment_metadata_with_env(metadata)
        self._writer.finalize(final_state, enriched_metadata)
        self._finalized = True


# -----------------------------------------------------------------------------
# One-shot recording functions (from recording.py)
# -----------------------------------------------------------------------------


def record_completed_game(
    db: GameReplayDB,
    initial_state: GameState,
    final_state: GameState,
    moves: list[Move],
    metadata: dict[str, Any] | None = None,
    game_id: str | None = None,
    store_history_entries: bool = True,
    snapshot_interval: int | None = None,
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
    db_path: str | None = None,
    default_path: str | None = None,
    respect_env_disable: bool = True,
    enforce_canonical_history: bool = True,
) -> GameReplayDB | None:
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
    moves: list[Move],
    metadata: dict[str, Any] | None = None,
    game_id: str | None = None,
    parity_mode: str | None = None,
    store_history_entries: bool = True,
    snapshot_interval: int | None = None,
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
        ParityMode,
        ParityValidationError,
        get_parity_mode,
        validate_game_parity,
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
# NNUE Feature Caching (from recording.py)
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
    import numpy as np

    from app.ai.nnue import extract_features_from_gamestate
    from app.game_engine import GameEngine

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
    game_ids: list[str] | None = None,
    board_type: str | None = None,
    num_players: int | None = None,
    sample_every_n_moves: int = 1,
    limit: int | None = None,
    skip_if_cached: bool = True,
) -> tuple[int, int]:
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
    moves: list[Move],
    metadata: dict[str, Any] | None = None,
    game_id: str | None = None,
    store_history_entries: bool = True,
    snapshot_interval: int | None = None,
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


# -----------------------------------------------------------------------------
# Canonical source identifiers
# -----------------------------------------------------------------------------


class RecordSource:
    """Canonical source identifiers for game records."""

    SELF_PLAY = "self_play"
    SOAK_TEST = "soak_test"
    CMAES = "cmaes"
    GAUNTLET = "gauntlet"
    TOURNAMENT = "tournament"
    TRAINING = "training"
    MANUAL = "manual"


# -----------------------------------------------------------------------------
# RecordingConfig dataclass
# -----------------------------------------------------------------------------


@dataclass
class RecordingConfig:
    """Configuration for unified game recording.

    All board type inputs are normalized to canonical values automatically.
    """

    board_type: str  # Will be normalized to canonical value
    num_players: int
    source: str = RecordSource.SELF_PLAY

    # Optional metadata
    difficulty: int | None = None
    engine_mode: str | None = None
    model_id: str | None = None
    generation: int | None = None
    candidate_id: str | None = None
    tags: list[str] = field(default_factory=list)

    # Database configuration
    db_path: str | None = None  # If None, auto-generated from board_type/num_players
    db_prefix: str = "selfplay"
    db_dir: str = "data/games"

    # Recording options
    store_history_entries: bool = True
    snapshot_interval: int | None = None  # None = use env default (20)
    parity_mode: str | None = None  # None = use env default
    fsm_validation: bool = False

    def __post_init__(self):
        # Normalize board type to canonical value
        self.board_type = normalize_board_type(self.board_type)

        # Validate num_players
        if not 2 <= self.num_players <= 4:
            raise ValueError(f"num_players must be 2-4, got {self.num_players}")

    @property
    def config_key(self) -> str:
        """Get the canonical config key (e.g., 'square8_2p')."""
        return make_config_key(self.board_type, self.num_players)

    @property
    def board_type_enum(self) -> BoardType:
        """Get the BoardType enum value."""
        return get_board_type_enum(self.board_type)

    def get_db_path(self) -> str:
        """Get the database path, auto-generating if not specified."""
        if self.db_path:
            return self.db_path

        filename = normalize_database_filename(
            self.board_type,
            self.num_players,
            prefix=self.db_prefix,
        )
        return str(Path(self.db_dir) / filename)

    def build_metadata(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build standardized metadata dict for this recording config."""
        metadata: dict[str, Any] = {
            "source": self.source,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "config_key": self.config_key,
        }

        # Add optional fields if set
        if self.difficulty is not None:
            metadata["difficulty"] = self.difficulty
        if self.engine_mode:
            metadata["engine_mode"] = self.engine_mode
        if self.model_id:
            metadata["model_id"] = self.model_id
        if self.generation is not None:
            metadata["generation"] = self.generation
        if self.candidate_id:
            metadata["candidate_id"] = self.candidate_id
        if self.tags:
            metadata["tags"] = self.tags

        # Merge extra metadata
        if extra:
            metadata.update(extra)

        return metadata


# -----------------------------------------------------------------------------
# UnifiedGameRecorder
# -----------------------------------------------------------------------------


class UnifiedGameRecorder:
    """Unified context manager for recording games with canonical naming.

    This is the RECOMMENDED way to record games across all scripts.
    It enforces canonical board type naming and consistent metadata.

    Usage:
        config = RecordingConfig(board_type="sq8", num_players=2, source="self_play")

        with UnifiedGameRecorder(config, initial_state) as recorder:
            for move in game_loop():
                recorder.add_move(move, state_after)
            recorder.finalize(final_state)
    """

    def __init__(
        self,
        config: RecordingConfig,
        initial_state: GameState,
        game_id: str | None = None,
    ):
        self.config = config
        self.initial_state = initial_state
        self.game_id = game_id or str(uuid.uuid4())
        self._db: GameReplayDB | None = None
        self._recorder: GameRecorder | None = None
        self._finalized = False

    def __enter__(self) -> UnifiedGameRecorder:
        if not is_recording_enabled():
            return self

        db_path = self.config.get_db_path()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db = GameReplayDB(db_path)

        self._recorder = GameRecorder(self._db, self.initial_state, self.game_id)
        self._recorder.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._recorder is not None:
            self._recorder.__exit__(exc_type, exc_val, exc_tb)
        return False

    def add_move(
        self,
        move: Move,
        state_after: GameState | None = None,
        state_before: GameState | None = None,
        available_moves_count: int | None = None,
        engine_eval: float | None = None,
        engine_depth: int | None = None,
    ) -> None:
        """Add a move to the game record."""
        if self._recorder is None:
            return

        fsm_valid = None
        fsm_error_code = None

        # Optional FSM validation
        if self.config.fsm_validation and state_before:
            fsm_valid, fsm_error_code = validate_move_fsm(state_before, move)

        self._recorder.add_move(
            move,
            state_after=state_after,
            state_before=state_before,
            available_moves_count=available_moves_count,
            engine_eval=engine_eval,
            engine_depth=engine_depth,
            fsm_valid=fsm_valid,
            fsm_error_code=fsm_error_code,
        )

    def finalize(
        self,
        final_state: GameState,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Finalize the game recording with standardized metadata."""
        if self._recorder is None:
            return

        metadata = self.config.build_metadata(extra_metadata)
        self._recorder.finalize(final_state, metadata)
        self._finalized = True


# -----------------------------------------------------------------------------
# Unified recording functions
# -----------------------------------------------------------------------------


def record_game_unified(
    config: RecordingConfig,
    initial_state: GameState,
    final_state: GameState,
    moves: list[Move],
    extra_metadata: dict[str, Any] | None = None,
    game_id: str | None = None,
    with_parity_check: bool = False,
) -> str | None:
    """Record a completed game with canonical naming (one-shot).

    This is the recommended function for recording completed games
    across all scripts.

    Args:
        config: RecordingConfig with board type, num_players, source, etc.
        initial_state: GameState at the start of the game
        final_state: GameState at the end of the game
        moves: List of all moves in the game
        extra_metadata: Additional metadata to merge
        game_id: Optional custom game ID
        with_parity_check: If True, validate parity with TS engine

    Returns:
        The game ID that was stored, or None if recording is disabled
    """
    if not is_recording_enabled():
        return None

    db_path = config.get_db_path()
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    db = GameReplayDB(db_path)

    metadata = config.build_metadata(extra_metadata)
    gid = game_id or str(uuid.uuid4())

    if with_parity_check:
        return record_completed_game_with_parity_check(
            db=db,
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            metadata=metadata,
            game_id=gid,
            parity_mode=config.parity_mode,
            store_history_entries=config.store_history_entries,
            snapshot_interval=config.snapshot_interval,
        )
    else:
        return record_completed_game(
            db=db,
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            metadata=metadata,
            game_id=gid,
            store_history_entries=config.store_history_entries,
            snapshot_interval=config.snapshot_interval,
        )


def get_unified_db(
    board_type: str,
    num_players: int,
    prefix: str = "selfplay",
    db_dir: str = "data/games",
) -> GameReplayDB | None:
    """Get a GameReplayDB with canonical naming.

    Args:
        board_type: Board type (will be normalized to canonical value)
        num_players: Number of players (2-4)
        prefix: Database filename prefix (default: "selfplay")
        db_dir: Directory for database files (default: "data/games")

    Returns:
        GameReplayDB instance or None if recording is disabled
    """
    if not is_recording_enabled():
        return None

    canonical_board = normalize_board_type(board_type)
    filename = normalize_database_filename(canonical_board, num_players, prefix=prefix)
    db_path = str(Path(db_dir) / filename)

    os.makedirs(db_dir, exist_ok=True)
    return GameReplayDB(db_path)


# -----------------------------------------------------------------------------
# Module exports
# -----------------------------------------------------------------------------


__all__ = [
    # Environment/config utilities
    "DEFAULT_SELFPLAY_DB_PATH",
    # GameRecorder
    "GameRecorder",
    # Unified recording (RECOMMENDED)
    "RecordSource",
    "RecordingConfig",
    "UnifiedGameRecorder",
    "_augment_metadata_with_env",
    "cache_nnue_features_batch",
    # NNUE caching
    "cache_nnue_features_for_game",
    "get_default_db_path",
    "get_or_create_db",
    "get_unified_db",
    "is_recording_enabled",
    # One-shot recording
    "record_completed_game",
    "record_completed_game_with_nnue_cache",
    "record_completed_game_with_parity_check",
    "record_game_unified",
    "should_record_games",
    "validate_move_fsm",
]
