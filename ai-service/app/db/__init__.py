"""Database module for RingRift game storage and replay."""

from __future__ import annotations

import importlib

__all__ = [
    "GameReplayDB",
    "GameWriter",
    "ParityValidationError",
    "RecordSource",
    "get_or_create_db",
    "record_completed_game",
    "record_completed_game_with_parity_check",
    "validate_game_parity",
]

# Compatibility map for historical root-level imports. Keep the advertised
# package surface small; advanced callers should import from owning submodules.
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


def __dir__() -> list[str]:
    """Expose the declared database package surface for discoverability."""

    return sorted(set(globals()) | set(__all__))
