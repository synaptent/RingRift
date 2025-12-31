"""Base Quality Scorer Abstract Class.

Provides the abstract base class that all quality scorers should inherit from.
This class implements the QualityScorer protocol and provides common utilities
for weight management, threshold checking, and batch processing with caching.

December 30, 2025: Created as part of Priority 3 consolidation effort.

Usage:
    from app.quality.scorers.base import BaseQualityScorer, ScorerConfig

    class MyScorer(BaseQualityScorer):
        def _compute_score(self, data: dict) -> float:
            # Custom scoring logic
            return 0.85

        def _compute_components(self, data: dict) -> dict[str, float]:
            return {"my_component": 0.9}

    scorer = MyScorer(config=ScorerConfig(cache_size=100))
    result = scorer.score({"key": "value"})
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from app.quality.types import (
    BatchQualityScorer,
    QualityLevel,
    QualityResult,
    ValidationResult,
)

__all__ = [
    "BaseQualityScorer",
    "ScorerConfig",
    "ScorerStats",
]

logger = logging.getLogger(__name__)


@dataclass
class ScorerConfig:
    """Configuration for quality scorers.

    Attributes:
        cache_size: Maximum number of results to cache (0 to disable)
        cache_ttl_seconds: Time-to-live for cached entries (0 for no expiry)
        emit_events: Whether to emit quality events
        validate_input: Whether to validate input data before scoring
        default_weights: Default component weights
        thresholds: Quality level thresholds
    """

    cache_size: int = 100
    cache_ttl_seconds: float = 300.0  # 5 minutes
    emit_events: bool = True
    validate_input: bool = True

    # Component weights (subclasses override with their own)
    default_weights: dict[str, float] = field(default_factory=dict)

    # Quality thresholds (matching QualityLevel)
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "high": 0.70,
            "medium": 0.50,
            "low": 0.30,
        }
    )

    def get_threshold(self, level: str) -> float:
        """Get threshold for a quality level."""
        return self.thresholds.get(level, 0.0)


@dataclass
class ScorerStats:
    """Statistics for scorer performance monitoring.

    Attributes:
        total_scored: Total items scored
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        avg_score_time_ms: Average scoring time in milliseconds
        score_distribution: Distribution of scores by level
    """

    total_scored: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_score_time_ms: float = 0.0
    score_distribution: dict[str, int] = field(
        default_factory=lambda: {
            "high": 0,
            "medium": 0,
            "low": 0,
            "blocked": 0,
        }
    )

    @property
    def avg_score_time_ms(self) -> float:
        """Average scoring time in milliseconds."""
        if self.total_scored == 0:
            return 0.0
        return self.total_score_time_ms / self.total_scored

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as a fraction."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def record_score(self, level: QualityLevel, time_ms: float) -> None:
        """Record a scoring operation."""
        self.total_scored += 1
        self.total_score_time_ms += time_ms
        self.score_distribution[level.value] = (
            self.score_distribution.get(level.value, 0) + 1
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_scored": self.total_scored,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "avg_score_time_ms": self.avg_score_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "score_distribution": self.score_distribution,
        }


class BaseQualityScorer(ABC):
    """Abstract base class for quality scorers.

    Implements the QualityScorer and BatchQualityScorer protocols and provides
    common infrastructure:
    - LRU cache with TTL for repeated scores
    - Component weight management
    - Threshold-based level determination
    - Input validation framework
    - Statistics tracking

    Subclasses must implement:
    - _compute_score(data) -> float: Core scoring logic
    - _compute_components(data) -> dict: Component breakdown

    Optional overrides:
    - _validate_input(data) -> ValidationResult: Custom validation
    - _get_cache_key(data) -> str: Custom cache key generation
    """

    # Class-level scorer identification
    SCORER_NAME: str = "base"
    SCORER_VERSION: str = "1.0.0"

    def __init__(
        self,
        config: ScorerConfig | None = None,
        weights: dict[str, float] | None = None,
    ):
        """Initialize the scorer.

        Args:
            config: Scorer configuration
            weights: Component weights (overrides config defaults)
        """
        self.config = config or ScorerConfig()
        self._weights = weights or dict(self.config.default_weights)
        self._stats = ScorerStats()

        # LRU cache: key -> (result, timestamp)
        self._cache: OrderedDict[str, tuple[QualityResult, float]] = OrderedDict()

    @property
    def weights(self) -> dict[str, float]:
        """Get current component weights."""
        return dict(self._weights)

    @property
    def stats(self) -> ScorerStats:
        """Get scoring statistics."""
        return self._stats

    def set_weight(self, component: str, weight: float) -> None:
        """Set weight for a component.

        Args:
            component: Component name
            weight: Weight value (should be 0-1, weights need not sum to 1)
        """
        self._weights[component] = max(0.0, min(1.0, weight))

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set multiple weights at once.

        Args:
            weights: Dictionary of component -> weight
        """
        for component, weight in weights.items():
            self.set_weight(component, weight)

    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}

    # -------------------------------------------------------------------------
    # Protocol Implementation: QualityScorer
    # -------------------------------------------------------------------------

    def score(self, data: dict[str, Any]) -> QualityResult:
        """Compute quality score for the given data.

        This is the main entry point implementing the QualityScorer protocol.
        It handles caching, validation, and statistics tracking.

        Args:
            data: Input data to score

        Returns:
            QualityResult with score, level, and components
        """
        start_time = time.perf_counter()

        # Check cache first
        cache_key = self._get_cache_key(data)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            self._stats.cache_hits += 1
            return cached

        self._stats.cache_misses += 1

        # Validate input if enabled
        if self.config.validate_input:
            validation = self._validate_input(data)
            if not validation.is_valid:
                error_msg = "; ".join(validation.errors) or "Validation failed"
                result = QualityResult.blocked(error_msg)
                self._record_result(result, start_time)
                return result

        # Compute score and components
        try:
            score = self._compute_score(data)
            components = self._compute_components(data)

            # Clamp score to valid range
            score = max(0.0, min(1.0, score))

            # Determine level
            level = QualityLevel.from_score(score)

            # Build result
            result = QualityResult(
                score=score,
                level=level,
                components=components,
                metadata={
                    "scorer": self.SCORER_NAME,
                    "version": self.SCORER_VERSION,
                    "scored_at": time.time(),
                },
            )

        except Exception as e:
            logger.warning(
                "Scoring failed for %s: %s",
                self.SCORER_NAME,
                e,
            )
            result = QualityResult.blocked(f"Scoring error: {e}")

        # Cache and record
        self._add_to_cache(cache_key, result)
        self._record_result(result, start_time)

        return result

    # -------------------------------------------------------------------------
    # Protocol Implementation: BatchQualityScorer
    # -------------------------------------------------------------------------

    def score_batch(self, data_list: list[dict[str, Any]]) -> list[QualityResult]:
        """Compute quality scores for multiple items.

        Default implementation calls score() for each item.
        Subclasses can override for more efficient batch processing.

        Args:
            data_list: List of data items to score

        Returns:
            List of QualityResult in same order as input
        """
        return [self.score(data) for data in data_list]

    # -------------------------------------------------------------------------
    # Abstract Methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    def _compute_score(self, data: dict[str, Any]) -> float:
        """Compute the primary quality score.

        Args:
            data: Validated input data

        Returns:
            Quality score in range [0, 1]
        """
        ...

    @abstractmethod
    def _compute_components(self, data: dict[str, Any]) -> dict[str, float]:
        """Compute individual component scores.

        Args:
            data: Validated input data

        Returns:
            Dictionary of component name -> score (each in [0, 1])
        """
        ...

    # -------------------------------------------------------------------------
    # Optional Overrides
    # -------------------------------------------------------------------------

    def _validate_input(self, data: dict[str, Any]) -> ValidationResult:
        """Validate input data before scoring.

        Default implementation accepts all non-empty dictionaries.
        Subclasses should override for specific validation.

        Args:
            data: Input data to validate

        Returns:
            ValidationResult indicating validity
        """
        if not isinstance(data, dict):
            return ValidationResult.invalid("Input must be a dictionary")
        if not data:
            return ValidationResult.invalid("Input dictionary is empty")
        return ValidationResult.valid()

    def _get_cache_key(self, data: dict[str, Any]) -> str:
        """Generate cache key for data.

        Default implementation uses JSON hash.
        Subclasses can override for custom key generation.

        Args:
            data: Input data

        Returns:
            Cache key string
        """
        try:
            # Sort keys for consistent hashing
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(json_str.encode()).hexdigest()
        except (TypeError, ValueError):
            # Fallback to id-based key
            return f"id_{id(data)}"

    # -------------------------------------------------------------------------
    # Threshold Utilities
    # -------------------------------------------------------------------------

    def is_high_quality(self, score: float) -> bool:
        """Check if score meets high quality threshold."""
        return score >= self.config.get_threshold("high")

    def is_acceptable(self, score: float) -> bool:
        """Check if score meets minimum acceptable threshold."""
        return score >= self.config.get_threshold("low")

    def get_level(self, score: float) -> QualityLevel:
        """Get quality level for a score."""
        return QualityLevel.from_score(score)

    def meets_threshold(self, score: float, threshold: str | float) -> bool:
        """Check if score meets a threshold.

        Args:
            score: Quality score to check
            threshold: Either a level name ("high", "medium", "low")
                      or a numeric threshold value

        Returns:
            True if score meets or exceeds threshold
        """
        if isinstance(threshold, str):
            threshold = self.config.get_threshold(threshold)
        return score >= threshold

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def _get_from_cache(self, key: str) -> QualityResult | None:
        """Get result from cache if valid."""
        if self.config.cache_size <= 0:
            return None

        entry = self._cache.get(key)
        if entry is None:
            return None

        result, timestamp = entry

        # Check TTL
        if self.config.cache_ttl_seconds > 0:
            age = time.time() - timestamp
            if age > self.config.cache_ttl_seconds:
                del self._cache[key]
                return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return result

    def _add_to_cache(self, key: str, result: QualityResult) -> None:
        """Add result to cache."""
        if self.config.cache_size <= 0:
            return

        # Evict oldest if at capacity
        while len(self._cache) >= self.config.cache_size:
            self._cache.popitem(last=False)

        self._cache[key] = (result, time.time())

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def _record_result(self, result: QualityResult, start_time: float) -> None:
        """Record scoring result for statistics."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._stats.record_score(result.level, elapsed_ms)

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = ScorerStats()

    # -------------------------------------------------------------------------
    # Weighted Score Computation
    # -------------------------------------------------------------------------

    def weighted_average(
        self,
        components: dict[str, float],
        weights: dict[str, float] | None = None,
    ) -> float:
        """Compute weighted average of component scores.

        Args:
            components: Component name -> score mapping
            weights: Optional weight overrides (uses self._weights if not provided)

        Returns:
            Weighted average score in [0, 1]
        """
        weights = weights or self._weights

        total_weight = 0.0
        weighted_sum = 0.0

        for component, score in components.items():
            weight = weights.get(component, 0.0)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight <= 0:
            return 0.0

        return weighted_sum / total_weight

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    def get_info(self) -> dict[str, Any]:
        """Get scorer information for debugging."""
        return {
            "name": self.SCORER_NAME,
            "version": self.SCORER_VERSION,
            "weights": self.weights,
            "config": {
                "cache_size": self.config.cache_size,
                "cache_ttl_seconds": self.config.cache_ttl_seconds,
                "validate_input": self.config.validate_input,
                "emit_events": self.config.emit_events,
                "thresholds": self.config.thresholds,
            },
            "stats": self._stats.to_dict(),
            "cache_size": len(self._cache),
        }


# Type alias for protocol checking
BatchQualityScorer.register(BaseQualityScorer)  # type: ignore[attr-defined]
