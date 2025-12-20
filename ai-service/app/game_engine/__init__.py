"""Game engine package for RingRift.

This package contains the core game engine that implements RingRift rules.
The canonical rules are defined in RULES_CANONICAL_SPEC.md.

Package structure:
- phase_requirements.py: Phase requirement types and data classes
- _game_engine_legacy.py: Original monolithic module (being migrated)

For backwards compatibility, all public symbols are re-exported from
the legacy module.
"""

# Re-export from submodules
from .phase_requirements import (
    PhaseRequirementType,
    PhaseRequirement,
)

# Re-export from legacy module for backwards compatibility
from app._game_engine_legacy import (
    GameEngine,
)

__all__ = [
    "GameEngine",
    "PhaseRequirement",
    "PhaseRequirementType",
]
