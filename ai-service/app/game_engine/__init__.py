"""Game engine package for RingRift.

This package contains the core game engine that implements RingRift rules.
The canonical rules are defined in RULES_CANONICAL_SPEC.md.

Package structure:
- phase_requirements.py: Phase requirement types and data classes (SSoT)
- _game_engine_legacy.py: Original monolithic module (being migrated)

Public API re-exports from appropriate SSoT locations.
"""

# Import PhaseRequirement types from canonical SSoT module
from app.game_engine.phase_requirements import PhaseRequirement, PhaseRequirementType

# Import GameEngine and constants from legacy module (still being migrated)
from app._game_engine_legacy import (
    STRICT_NO_MOVE_INVARIANT,
    GameEngine,
)

__all__ = [
    "STRICT_NO_MOVE_INVARIANT",
    "GameEngine",
    "PhaseRequirement",
    "PhaseRequirementType",
]
