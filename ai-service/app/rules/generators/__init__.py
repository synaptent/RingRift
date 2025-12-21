"""Move generators for rules canonical compliance.

This module contains move enumeration logic extracted from GameEngine
to establish single-source-of-truth (SSoT) for move generation.

Generators:
- LineGenerator: Enumerates line processing moves (RR-CANON-R076)
- TerritoryGenerator: Enumerates territory processing moves (RR-CANON-R076)

Architecture Note (2025-12):
    These generators use BoardManager (SSoT) for detection and create
    Move objects. They replace the inline move generation in GameEngine.
"""

from app.rules.generators.line import LineGenerator
from app.rules.generators.territory import TerritoryGenerator

__all__ = ["LineGenerator", "TerritoryGenerator"]
