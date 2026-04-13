"""Database module for RingRift game storage and replay."""

from __future__ import annotations

import importlib

__all__ = [
    "DEFAULT_INTEGRITY_CHECK_TIMEOUT",
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
    # Database integrity (with timeout support)
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

_EXPORTS = {
    "GameReplayDB": "app.db.game_replay",
    "GameWriter": "app.db.game_replay",
    "DEFAULT_INTEGRITY_CHECK_TIMEOUT": "app.db.integrity",
    "check_and_repair_databases": "app.db.integrity",
    "check_database_integrity": "app.db.integrity",
    "get_database_stats": "app.db.integrity",
    "recover_corrupted_database": "app.db.integrity",
    "ParityDivergence": "app.db.parity_validator",
    "ParityMode": "app.db.parity_validator",
    "ParityValidationError": "app.db.parity_validator",
    "get_parity_mode": "app.db.parity_validator",
    "is_parity_validation_enabled": "app.db.parity_validator",
    "validate_game_parity": "app.db.parity_validator",
    "GameRecorder": "app.db.unified_recording",
    "RecordingConfig": "app.db.unified_recording",
    "RecordSource": "app.db.unified_recording",
    "UnifiedGameRecorder": "app.db.unified_recording",
    "cache_nnue_features_batch": "app.db.unified_recording",
    "cache_nnue_features_for_game": "app.db.unified_recording",
    "get_or_create_db": "app.db.unified_recording",
    "get_unified_db": "app.db.unified_recording",
    "record_completed_game": "app.db.unified_recording",
    "record_completed_game_with_nnue_cache": "app.db.unified_recording",
    "record_completed_game_with_parity_check": "app.db.unified_recording",
    "record_game_unified": "app.db.unified_recording",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)
