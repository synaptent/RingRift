"""Tests for app.quality.scorers.base module.

Tests the BaseQualityScorer abstract class and related configuration.
"""

import time
from typing import Any

import pytest

from app.quality.scorers.base import (
    BaseQualityScorer,
    ScorerConfig,
    ScorerStats,
)
from app.quality.types import QualityLevel, QualityResult, ValidationResult


# Test implementation of BaseQualityScorer
class SimpleTestScorer(BaseQualityScorer):
    """Simple scorer for testing that returns score from data."""

    SCORER_NAME = "simple_test"
    SCORER_VERSION = "1.0.0"

    def _compute_score(self, data: dict[str, Any]) -> float:
        return data.get("score", 0.5)

    def _compute_components(self, data: dict[str, Any]) -> dict[str, float]:
        return {"main": data.get("score", 0.5)}


class ComponentTestScorer(BaseQualityScorer):
    """Scorer that uses weighted components."""

    SCORER_NAME = "component_test"
    SCORER_VERSION = "1.0.0"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self._weights:
            self._weights = {"a": 0.5, "b": 0.3, "c": 0.2}

    def _compute_score(self, data: dict[str, Any]) -> float:
        components = self._compute_components(data)
        return self.weighted_average(components)

    def _compute_components(self, data: dict[str, Any]) -> dict[str, float]:
        return {
            "a": data.get("a", 0.5),
            "b": data.get("b", 0.5),
            "c": data.get("c", 0.5),
        }


class ValidatingScorer(BaseQualityScorer):
    """Scorer with custom validation."""

    SCORER_NAME = "validating"
    SCORER_VERSION = "1.0.0"

    def _compute_score(self, data: dict[str, Any]) -> float:
        return data.get("value", 0.5)

    def _compute_components(self, data: dict[str, Any]) -> dict[str, float]:
        return {"value": data.get("value", 0.5)}

    def _validate_input(self, data: dict[str, Any]) -> ValidationResult:
        if "required_field" not in data:
            return ValidationResult.invalid("Missing required_field")
        return ValidationResult.valid()


class TestScorerConfig:
    """Tests for ScorerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ScorerConfig()
        assert config.cache_size == 100
        assert config.cache_ttl_seconds == 300.0
        assert config.emit_events is True
        assert config.validate_input is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = ScorerConfig(
            cache_size=50,
            cache_ttl_seconds=60.0,
            emit_events=False,
            validate_input=False,
        )
        assert config.cache_size == 50
        assert config.cache_ttl_seconds == 60.0
        assert config.emit_events is False
        assert config.validate_input is False

    def test_default_thresholds(self):
        """Test default quality thresholds."""
        config = ScorerConfig()
        assert config.get_threshold("high") == 0.70
        assert config.get_threshold("medium") == 0.50
        assert config.get_threshold("low") == 0.30
        assert config.get_threshold("unknown") == 0.0

    def test_custom_thresholds(self):
        """Test custom thresholds."""
        config = ScorerConfig(
            thresholds={"high": 0.80, "medium": 0.60, "low": 0.40}
        )
        assert config.get_threshold("high") == 0.80
        assert config.get_threshold("medium") == 0.60
        assert config.get_threshold("low") == 0.40


class TestScorerStats:
    """Tests for ScorerStats dataclass."""

    def test_initial_values(self):
        """Test initial statistics values."""
        stats = ScorerStats()
        assert stats.total_scored == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.avg_score_time_ms == 0.0
        assert stats.cache_hit_rate == 0.0

    def test_record_score(self):
        """Test recording scores updates stats."""
        stats = ScorerStats()
        stats.record_score(QualityLevel.HIGH, 10.0)
        stats.record_score(QualityLevel.MEDIUM, 15.0)

        assert stats.total_scored == 2
        assert stats.total_score_time_ms == 25.0
        assert stats.avg_score_time_ms == 12.5
        assert stats.score_distribution["high"] == 1
        assert stats.score_distribution["medium"] == 1

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        stats = ScorerStats()
        stats.cache_hits = 3
        stats.cache_misses = 7

        assert stats.cache_hit_rate == 0.3

    def test_to_dict(self):
        """Test dictionary serialization."""
        stats = ScorerStats()
        stats.record_score(QualityLevel.HIGH, 5.0)
        stats.cache_hits = 2
        stats.cache_misses = 8

        d = stats.to_dict()
        assert d["total_scored"] == 1
        assert d["cache_hits"] == 2
        assert d["cache_misses"] == 8
        assert d["cache_hit_rate"] == 0.2
        assert "score_distribution" in d


class TestBaseQualityScorerScoring:
    """Tests for BaseQualityScorer scoring functionality."""

    def test_basic_scoring(self):
        """Test basic score computation."""
        scorer = SimpleTestScorer()
        result = scorer.score({"score": 0.75})

        assert result.score == 0.75
        assert result.level == QualityLevel.HIGH
        assert result.is_valid is True
        assert result.components == {"main": 0.75}

    def test_score_levels(self):
        """Test different score levels."""
        scorer = SimpleTestScorer()

        # High
        assert scorer.score({"score": 0.80}).level == QualityLevel.HIGH

        # Medium
        assert scorer.score({"score": 0.60}).level == QualityLevel.MEDIUM

        # Low
        assert scorer.score({"score": 0.35}).level == QualityLevel.LOW

        # Blocked (very low)
        assert scorer.score({"score": 0.10}).level == QualityLevel.BLOCKED

    def test_score_clamping(self):
        """Test score is clamped to [0, 1]."""
        scorer = SimpleTestScorer()

        # Above 1.0
        result = scorer.score({"score": 1.5})
        assert result.score == 1.0

        # Below 0.0
        result = scorer.score({"score": -0.5})
        assert result.score == 0.0

    def test_metadata_included(self):
        """Test metadata is included in result."""
        scorer = SimpleTestScorer()
        result = scorer.score({"score": 0.5})

        assert result.metadata["scorer"] == "simple_test"
        assert result.metadata["version"] == "1.0.0"
        assert "scored_at" in result.metadata

    def test_empty_dict_rejected(self):
        """Test empty dict is rejected by default validation."""
        scorer = SimpleTestScorer()
        result = scorer.score({})

        assert result.level == QualityLevel.BLOCKED
        assert result.is_valid is False
        assert "empty" in result.error.lower()

    def test_non_dict_rejected(self):
        """Test non-dict input is rejected."""
        scorer = SimpleTestScorer()
        result = scorer.score("not a dict")  # type: ignore

        assert result.level == QualityLevel.BLOCKED
        assert result.is_valid is False


class TestBaseQualityScorerValidation:
    """Tests for input validation."""

    def test_custom_validation_passes(self):
        """Test custom validation when valid."""
        scorer = ValidatingScorer()
        result = scorer.score({"required_field": "present", "value": 0.8})

        assert result.score == 0.8
        assert result.is_valid is True

    def test_custom_validation_fails(self):
        """Test custom validation when invalid."""
        scorer = ValidatingScorer()
        result = scorer.score({"value": 0.8})  # Missing required_field

        assert result.level == QualityLevel.BLOCKED
        assert result.is_valid is False
        assert "required_field" in result.error

    def test_validation_disabled(self):
        """Test validation can be disabled."""
        config = ScorerConfig(validate_input=False)
        scorer = ValidatingScorer(config=config)

        # Would fail validation, but validation is disabled
        result = scorer.score({"value": 0.8})
        assert result.score == 0.8
        assert result.is_valid is True


class TestBaseQualityScorerCaching:
    """Tests for result caching."""

    def test_cache_hit(self):
        """Test cache returns same result."""
        scorer = SimpleTestScorer()

        # First call
        result1 = scorer.score({"score": 0.7, "id": "test1"})
        # Second call with same data
        result2 = scorer.score({"score": 0.7, "id": "test1"})

        assert result1.score == result2.score
        assert scorer.stats.cache_hits == 1
        assert scorer.stats.cache_misses == 1

    def test_cache_miss_different_data(self):
        """Test different data causes cache miss."""
        scorer = SimpleTestScorer()

        scorer.score({"score": 0.7})
        scorer.score({"score": 0.8})  # Different data

        assert scorer.stats.cache_hits == 0
        assert scorer.stats.cache_misses == 2

    def test_cache_disabled(self):
        """Test caching can be disabled."""
        config = ScorerConfig(cache_size=0)
        scorer = SimpleTestScorer(config=config)

        scorer.score({"score": 0.7})
        scorer.score({"score": 0.7})

        assert scorer.stats.cache_hits == 0
        assert scorer.stats.cache_misses == 2

    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        config = ScorerConfig(cache_size=2)
        scorer = SimpleTestScorer(config=config)

        scorer.score({"id": 1, "score": 0.5})
        scorer.score({"id": 2, "score": 0.6})
        scorer.score({"id": 3, "score": 0.7})  # Evicts id=1

        # id=1 should be evicted
        scorer.score({"id": 1, "score": 0.5})
        assert scorer.stats.cache_misses == 4  # All misses

    def test_clear_cache(self):
        """Test cache clearing."""
        scorer = SimpleTestScorer()
        scorer.score({"score": 0.7})
        assert len(scorer._cache) == 1

        scorer.clear_cache()
        assert len(scorer._cache) == 0


class TestBaseQualityScorerWeights:
    """Tests for weight management."""

    def test_default_weights(self):
        """Test scorer starts with no weights by default."""
        scorer = SimpleTestScorer()
        assert scorer.weights == {}

    def test_set_weight(self):
        """Test setting individual weight."""
        scorer = SimpleTestScorer()
        scorer.set_weight("component_a", 0.6)

        assert scorer.weights["component_a"] == 0.6

    def test_weight_clamping(self):
        """Test weights are clamped to [0, 1]."""
        scorer = SimpleTestScorer()
        scorer.set_weight("a", 1.5)
        scorer.set_weight("b", -0.5)

        assert scorer.weights["a"] == 1.0
        assert scorer.weights["b"] == 0.0

    def test_set_weights_batch(self):
        """Test setting multiple weights at once."""
        scorer = SimpleTestScorer()
        scorer.set_weights({"a": 0.5, "b": 0.3, "c": 0.2})

        assert scorer.weights == {"a": 0.5, "b": 0.3, "c": 0.2}

    def test_normalize_weights(self):
        """Test weight normalization."""
        scorer = SimpleTestScorer()
        # Use values that won't be clamped (set_weight clamps to [0, 1])
        scorer._weights = {"a": 0.2, "b": 0.3, "c": 0.5}
        # These already sum to 1.0, so try unequal values
        scorer._weights = {"a": 0.4, "b": 0.6, "c": 1.0}
        scorer.normalize_weights()

        assert scorer.weights["a"] == pytest.approx(0.2)
        assert scorer.weights["b"] == pytest.approx(0.3)
        assert scorer.weights["c"] == pytest.approx(0.5)

    def test_weighted_average(self):
        """Test weighted average computation."""
        scorer = ComponentTestScorer()
        result = scorer.score({"a": 1.0, "b": 0.5, "c": 0.0})

        # Expected: 1.0*0.5 + 0.5*0.3 + 0.0*0.2 = 0.65
        assert result.score == pytest.approx(0.65)


class TestBaseQualityScorerThresholds:
    """Tests for threshold utilities."""

    def test_is_high_quality(self):
        """Test high quality threshold check."""
        scorer = SimpleTestScorer()
        assert scorer.is_high_quality(0.75) is True
        assert scorer.is_high_quality(0.70) is True
        assert scorer.is_high_quality(0.69) is False

    def test_is_acceptable(self):
        """Test acceptable threshold check."""
        scorer = SimpleTestScorer()
        assert scorer.is_acceptable(0.30) is True
        assert scorer.is_acceptable(0.29) is False

    def test_meets_threshold_string(self):
        """Test meets_threshold with string level."""
        scorer = SimpleTestScorer()
        assert scorer.meets_threshold(0.75, "high") is True
        assert scorer.meets_threshold(0.55, "medium") is True
        assert scorer.meets_threshold(0.35, "low") is True
        assert scorer.meets_threshold(0.25, "low") is False

    def test_meets_threshold_numeric(self):
        """Test meets_threshold with numeric value."""
        scorer = SimpleTestScorer()
        assert scorer.meets_threshold(0.85, 0.80) is True
        assert scorer.meets_threshold(0.75, 0.80) is False

    def test_get_level(self):
        """Test get_level utility."""
        scorer = SimpleTestScorer()
        assert scorer.get_level(0.75) == QualityLevel.HIGH
        assert scorer.get_level(0.55) == QualityLevel.MEDIUM
        assert scorer.get_level(0.35) == QualityLevel.LOW
        assert scorer.get_level(0.25) == QualityLevel.BLOCKED


class TestBaseQualityScorerBatch:
    """Tests for batch scoring."""

    def test_score_batch(self):
        """Test batch scoring."""
        scorer = SimpleTestScorer()
        data_list = [
            {"score": 0.8},
            {"score": 0.6},
            {"score": 0.4},
        ]

        results = scorer.score_batch(data_list)

        assert len(results) == 3
        assert results[0].score == 0.8
        assert results[1].score == 0.6
        assert results[2].score == 0.4

    def test_score_batch_empty(self):
        """Test batch scoring with empty list."""
        scorer = SimpleTestScorer()
        results = scorer.score_batch([])
        assert results == []


class TestBaseQualityScorerInfo:
    """Tests for scorer introspection."""

    def test_get_info(self):
        """Test get_info() returns scorer metadata."""
        scorer = SimpleTestScorer()
        scorer.score({"score": 0.7})

        info = scorer.get_info()

        assert info["name"] == "simple_test"
        assert info["version"] == "1.0.0"
        assert "config" in info
        assert "stats" in info
        assert info["cache_size"] == 1

    def test_reset_stats(self):
        """Test statistics reset."""
        scorer = SimpleTestScorer()
        scorer.score({"score": 0.7})
        assert scorer.stats.total_scored == 1

        scorer.reset_stats()
        assert scorer.stats.total_scored == 0


class TestBaseQualityScorerErrorHandling:
    """Tests for error handling during scoring."""

    def test_exception_in_compute_score(self):
        """Test exception during scoring results in blocked."""
        class FailingScorer(BaseQualityScorer):
            SCORER_NAME = "failing"
            SCORER_VERSION = "1.0.0"

            def _compute_score(self, data):
                raise ValueError("Intentional error")

            def _compute_components(self, data):
                return {}

        scorer = FailingScorer()
        result = scorer.score({"data": "test"})

        assert result.level == QualityLevel.BLOCKED
        assert result.is_valid is False
        assert "error" in result.error.lower()
