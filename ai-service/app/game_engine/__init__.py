"""Game engine package for RingRift.

This package contains the core game engine that implements RingRift rules.
The canonical rules are defined in RULES_CANONICAL_SPEC.md.

Package structure:
- phase_requirements.py: Phase requirement types and data classes (SSoT)
- _game_engine_legacy.py: Original monolithic module (DEPRECATED - being migrated)

Public API re-exports from appropriate SSoT locations.

Migration Plan (December 2025):
    The GameEngine class is being migrated from the legacy module to a canonical
    implementation. During the migration period:

    1. **Canonical imports**: Use ``from app.game_engine import GameEngine``
       This is the stable public API that will continue to work.

    2. **Direct legacy imports**: ``from app._game_engine_legacy import ...``
       This is DEPRECATED and will emit warnings. Migrate to canonical imports.

    3. **Legacy replay functions**: Use ``app.rules.legacy`` for replaying
       pre-canonical game recordings. These violate RR-CANON-R075.

Target Removal Date: Q2 2026
"""

import warnings

# Import PhaseRequirement types from canonical SSoT module
from app.game_engine.phase_requirements import PhaseRequirement, PhaseRequirementType

# Lazy import cache to avoid circular import with _game_engine_legacy
# The legacy module imports from game_engine.phase_requirements, so we can't
# import from it at module load time without causing a circular import.
_lazy_cache = {}

def __getattr__(name):
    """Lazy attribute access for GameEngine and related symbols."""
    if name in ("GameEngine", "STRICT_NO_MOVE_INVARIANT"):
        if name not in _lazy_cache:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"app\._game_engine_legacy is deprecated.*",
                    category=DeprecationWarning,
                )
                from app._game_engine_legacy import (
                    STRICT_NO_MOVE_INVARIANT as _STRICT,
                    GameEngine as _GameEngine,
                )
            _lazy_cache["GameEngine"] = _GameEngine
            _lazy_cache["STRICT_NO_MOVE_INVARIANT"] = _STRICT
        return _lazy_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _warn_legacy_import(name: str) -> None:
    """Emit deprecation warning for direct legacy module imports."""
    warnings.warn(
        f"Direct import of '{name}' from app._game_engine_legacy is deprecated. "
        f"Use 'from app.game_engine import {name}' instead. "
        "Direct legacy imports will be removed in Q2 2026.",
        DeprecationWarning,
        stacklevel=3,
    )


# Provide a hook for detecting legacy imports (optional - for metrics)
_legacy_import_warned = False

__all__ = [
    "STRICT_NO_MOVE_INVARIANT",
    "GameEngine",
    "PhaseRequirement",
    "PhaseRequirementType",
]


def __dir__() -> list[str]:
    """Expose the intended game-engine surface for discoverability."""

    return sorted(set(globals()) | set(__all__))
