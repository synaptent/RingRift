"""Database module for RingRift game storage and replay."""

from app.db.game_replay import GameReplayDB, GameWriter
from app.db.integrity import (
    check_and_repair_databases,
    check_database_integrity,
    get_database_stats,
    recover_corrupted_database,
)
from app.db.parity_validator import (
    ParityDivergence,
    ParityMode,
    ParityValidationError,
    get_parity_mode,
    is_parity_validation_enabled,
    validate_game_parity,
)
from app.db.recording import (
    GameRecorder,
    cache_nnue_features_batch,
    cache_nnue_features_for_game,
    get_or_create_db,
    record_completed_game,
    record_completed_game_with_nnue_cache,
    record_completed_game_with_parity_check,
)
from app.db.unified_recording import (
    RecordingConfig,
    RecordSource,
    UnifiedGameRecorder,
    get_unified_db,
    record_game_unified,
)

__all__ = [
    "GameRecorder",
    "GameReplayDB",
    "GameWriter",
    "ParityDivergence",
    "ParityMode",
    "ParityValidationError",
    "RecordSource",
    "RecordingConfig",
    # Unified recording (RECOMMENDED)
    "UnifiedGameRecorder",
    "cache_nnue_features_batch",
    "cache_nnue_features_for_game",
    "check_and_repair_databases",
    # Database integrity
    "check_database_integrity",
    "get_database_stats",
    "get_or_create_db",
    "get_parity_mode",
    "get_unified_db",
    "is_parity_validation_enabled",
    "record_completed_game",
    "record_completed_game_with_nnue_cache",
    "record_completed_game_with_parity_check",
    "record_game_unified",
    "recover_corrupted_database",
    # Parity validation
    "validate_game_parity",
]
