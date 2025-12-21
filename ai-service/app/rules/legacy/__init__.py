"""Legacy replay compatibility module for RingRift AI service.

This package contains backwards-compatible code paths for replaying games
recorded under previous versions of the rules engine. This separation ensures
that:

1. **Canonical rules** (RULES_CANONICAL_SPEC.md) are the default for all new games
2. **Legacy code paths** are isolated and clearly marked for future deprecation
3. **Old game replays** can still be executed without breaking changes

Usage:
    # For replaying games that may use legacy rules
    from app.rules.legacy import (
        replay_with_legacy_fallback,
        normalize_legacy_state,
        convert_legacy_move_type,
    )

    # Check if a game needs legacy handling
    from app.rules.legacy import requires_legacy_replay

Deprecation Plan:
    - December 2025: Module created to isolate legacy code
    - Q2 2026: Evaluate migration of remaining legacy games
    - Q4 2026: Consider removing support for schema versions < 8

For canonical rules implementation, see:
    - app/rules/validators/ - Canonical move validation
    - app/rules/mutators/ - Canonical state mutation
    - app/rules/default_engine.py - Rules orchestration
"""

from app.rules.legacy.move_type_aliases import (
    LEGACY_TO_CANONICAL_MOVE_TYPE,
    convert_legacy_move_type,
    is_legacy_move_type,
)
from app.rules.legacy.replay_compatibility import (
    replay_with_legacy_fallback,
    requires_legacy_replay,
)
from app.rules.legacy.state_normalization import (
    normalize_legacy_phase,
    normalize_legacy_state,
    normalize_legacy_status,
)

__all__ = [
    # Move type conversion
    "LEGACY_TO_CANONICAL_MOVE_TYPE",
    "convert_legacy_move_type",
    "is_legacy_move_type",
    # Replay compatibility
    "replay_with_legacy_fallback",
    "requires_legacy_replay",
    # State normalization
    "normalize_legacy_phase",
    "normalize_legacy_state",
    "normalize_legacy_status",
]
