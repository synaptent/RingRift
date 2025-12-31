"""Quality Scorers Package.

Provides pluggable quality scorer implementations following the QualityScorer
protocol defined in app.quality.types.

December 30, 2025: Created as part of Priority 3 consolidation effort.

Available scorers:
    - BaseQualityScorer: Abstract base class for all scorers
    - GameQualityScorer: Game-level quality scoring (future)
    - SampleQualityScorer: Training sample quality scoring (future)
"""

from app.quality.scorers.base import (
    BaseQualityScorer,
    ScorerConfig,
    ScorerStats,
)

__all__ = [
    "BaseQualityScorer",
    "ScorerConfig",
    "ScorerStats",
]
