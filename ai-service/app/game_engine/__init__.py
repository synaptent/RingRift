"""Game engine package for RingRift.

This package contains the core game engine that implements RingRift rules.
The canonical rules are defined in RULES_CANONICAL_SPEC.md.

Package structure:
- phase_requirements.py: Phase requirement types and data classes
- _game_engine_legacy.py: Original monolithic module (being migrated)

For backwards compatibility, all public symbols are re-exported from
the legacy module.
"""

# Re-export from legacy module for backwards compatibility
# NOTE: PhaseRequirementType and PhaseRequirement must come from the legacy
# module to match what GameEngine.get_phase_requirement() returns. Using
# different enum classes causes comparison failures.
from app._game_engine_legacy import (
    GameEngine,
    PhaseRequirementType,
    PhaseRequirement,
    STRICT_NO_MOVE_INVARIANT,
)

__all__ = [
    "GameEngine",
    "PhaseRequirement",
    "PhaseRequirementType",
    "STRICT_NO_MOVE_INVARIANT",
]
