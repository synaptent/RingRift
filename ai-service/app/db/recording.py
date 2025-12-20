"""Game recording utilities - COMPATIBILITY SHIM.

DEPRECATED: This module is maintained for backward compatibility only.
All functionality has been merged into app.db.unified_recording (Dec 2025).

Please import from app.db.unified_recording instead:
    from app.db.unified_recording import GameRecorder, record_completed_game, ...

Or use the package-level imports:
    from app.db import GameRecorder, record_completed_game, ...
"""

from __future__ import annotations

# Re-export everything from unified_recording for backward compatibility
from app.db.unified_recording import (
    # Constants
    DEFAULT_SELFPLAY_DB_PATH,
    # GameRecorder
    GameRecorder,
    # Utilities
    _augment_metadata_with_env,
    cache_nnue_features_batch,
    # NNUE caching
    cache_nnue_features_for_game,
    get_default_db_path,
    get_or_create_db,
    is_recording_enabled,
    # One-shot recording
    record_completed_game,
    record_completed_game_with_nnue_cache,
    record_completed_game_with_parity_check,
    should_record_games,
    validate_move_fsm,
)

__all__ = [
    "DEFAULT_SELFPLAY_DB_PATH",
    "GameRecorder",
    "_augment_metadata_with_env",
    "cache_nnue_features_batch",
    "cache_nnue_features_for_game",
    "get_default_db_path",
    "get_or_create_db",
    "is_recording_enabled",
    "record_completed_game",
    "record_completed_game_with_nnue_cache",
    "record_completed_game_with_parity_check",
    "should_record_games",
    "validate_move_fsm",
]
