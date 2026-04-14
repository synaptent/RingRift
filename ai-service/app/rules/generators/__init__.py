"""Move generators for rules canonical compliance.

This module contains move enumeration logic extracted from GameEngine
to establish single-source-of-truth (SSoT) for move generation.

Generators (Phase 1 - December 2025):
- LineGenerator: Enumerates line processing moves (RR-CANON-R076)
- TerritoryGenerator: Enumerates territory processing moves (RR-CANON-R076)

Generators (Phase 2 - December 2025):
- CaptureGenerator: Enumerates capture moves (RR-CANON-R095/R096)
- MovementGenerator: Enumerates non-capture movement moves (RR-CANON-R085)
- PlacementGenerator: Enumerates ring placement moves (RR-CANON-R050/R055/R060)

Architecture Note (2025-12):
    These generators use BoardManager (SSoT) for detection and create
    Move objects. They replace the inline move generation in GameEngine.
"""

from app.rules.generators.capture import CaptureGenerator
from app.rules.generators.line import LineGenerator
from app.rules.generators.movement import MovementGenerator
from app.rules.generators.placement import PlacementGenerator
from app.rules.generators.territory import TerritoryGenerator

__all__ = [
    "CaptureGenerator",
    "LineGenerator",
    "MovementGenerator",
    "PlacementGenerator",
    "TerritoryGenerator",
]


def __dir__() -> list[str]:
    """Expose the intended generator surface for discoverability."""

    return sorted(set(globals()) | set(__all__))
