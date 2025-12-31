"""Quality Framework Type Definitions.

This module defines the core types and protocols for the quality scoring
framework. All quality scorers should implement the QualityScorer protocol.

December 30, 2025: Created as part of Priority 3 consolidation effort.
Provides standardized interfaces for quality scoring across the system.

Usage:
    from app.quality.types import (
        QualityScorer,
        QualityResult,
        QualityLevel,
        ValidationResult,
    )

    # Check if an object implements QualityScorer
    if isinstance(my_scorer, QualityScorer):
        result = my_scorer.score(data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


__all__ = [
    "QualityLevel",
    "QualityResult",
    "QualityScorer",
    "ValidationResult",
    "BatchQualityScorer",
]


class QualityLevel(str, Enum):
    """Quality level classification.

    Aligned with QualityCategory in unified_quality.py for compatibility.
    """
    HIGH = "high"          # Score >= 0.70 (GOOD or EXCELLENT)
    MEDIUM = "medium"      # Score >= 0.50 (ADEQUATE)
    LOW = "low"            # Score >= 0.30 (POOR)
    BLOCKED = "blocked"    # Score < 0.30 (UNUSABLE) or validation failed

    @classmethod
    def from_score(cls, score: float, is_valid: bool = True) -> QualityLevel:
        """Determine level from numeric score.

        Args:
            score: Quality score in range [0, 1]
            is_valid: Whether the data passed validation

        Returns:
            Appropriate QualityLevel
        """
        if not is_valid or score < 0.30:
            return cls.BLOCKED
        elif score < 0.50:
            return cls.LOW
        elif score < 0.70:
            return cls.MEDIUM
        else:
            return cls.HIGH


@dataclass
class QualityResult:
    """Result of a quality scoring operation.

    This is a standardized container for quality scores that can be used
    across different scorer implementations. It's compatible with but
    separate from GameQuality (which is game-specific).

    Attributes:
        score: Primary quality score in range [0, 1]
        level: Categorical quality level
        components: Individual component scores that contributed
        metadata: Additional context (scorer version, timing, etc.)
        is_valid: Whether the input data was valid
        error: Error message if validation failed
    """
    score: float
    level: QualityLevel
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    error: str | None = None

    def __post_init__(self):
        """Ensure score is in valid range."""
        self.score = max(0.0, min(1.0, self.score))
        if self.level is None:
            self.level = QualityLevel.from_score(self.score, self.is_valid)

    @classmethod
    def blocked(cls, error: str) -> QualityResult:
        """Create a blocked result for validation failures."""
        return cls(
            score=0.0,
            level=QualityLevel.BLOCKED,
            is_valid=False,
            error=error,
        )

    @classmethod
    def from_score(
        cls,
        score: float,
        components: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QualityResult:
        """Create a result from a score with auto-leveling."""
        return cls(
            score=score,
            level=QualityLevel.from_score(score),
            components=components or {},
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "score": self.score,
            "level": self.level.value,
            "components": self.components,
            "metadata": self.metadata,
            "is_valid": self.is_valid,
            "error": self.error,
        }


@dataclass
class ValidationResult:
    """Result of a data validation operation.

    Used by validators to report validation status with details.
    """
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def valid(cls, metadata: dict[str, Any] | None = None) -> ValidationResult:
        """Create a successful validation result."""
        return cls(is_valid=True, metadata=metadata or {})

    @classmethod
    def invalid(cls, *errors: str) -> ValidationResult:
        """Create a failed validation result."""
        return cls(is_valid=False, errors=list(errors))

    def add_error(self, error: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(warning)


@runtime_checkable
class QualityScorer(Protocol):
    """Protocol for quality scoring implementations.

    All quality scorers should implement this interface to ensure
    consistent behavior across the system.

    Example:
        class MyScorer:
            def score(self, data: dict) -> QualityResult:
                # Compute quality score
                return QualityResult.from_score(0.85)

        scorer: QualityScorer = MyScorer()
        result = scorer.score({"game_id": "123", ...})
    """

    def score(self, data: dict[str, Any]) -> QualityResult:
        """Compute quality score for the given data.

        Args:
            data: Input data to score (format depends on scorer type)

        Returns:
            QualityResult containing score, level, and components
        """
        ...


@runtime_checkable
class BatchQualityScorer(Protocol):
    """Protocol for batch quality scoring implementations.

    Extends QualityScorer with batch processing capability for
    efficiency when scoring many items.
    """

    def score(self, data: dict[str, Any]) -> QualityResult:
        """Compute quality score for single item."""
        ...

    def score_batch(self, data_list: list[dict[str, Any]]) -> list[QualityResult]:
        """Compute quality scores for multiple items.

        Args:
            data_list: List of data items to score

        Returns:
            List of QualityResult in same order as input
        """
        ...
