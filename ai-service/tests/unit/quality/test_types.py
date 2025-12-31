"""Tests for app.quality.types module.

Tests the QualityLevel, QualityResult, ValidationResult, and protocol definitions.
"""

import pytest

from app.quality.types import (
    BatchQualityScorer,
    QualityLevel,
    QualityResult,
    QualityScorer,
    ValidationResult,
)


class TestQualityLevel:
    """Tests for QualityLevel enum."""

    def test_enum_values(self):
        """Test enum values are correct strings."""
        assert QualityLevel.HIGH.value == "high"
        assert QualityLevel.MEDIUM.value == "medium"
        assert QualityLevel.LOW.value == "low"
        assert QualityLevel.BLOCKED.value == "blocked"

    def test_from_score_high(self):
        """Test HIGH level for scores >= 0.70."""
        assert QualityLevel.from_score(0.70) == QualityLevel.HIGH
        assert QualityLevel.from_score(0.85) == QualityLevel.HIGH
        assert QualityLevel.from_score(1.0) == QualityLevel.HIGH

    def test_from_score_medium(self):
        """Test MEDIUM level for scores >= 0.50 and < 0.70."""
        assert QualityLevel.from_score(0.50) == QualityLevel.MEDIUM
        assert QualityLevel.from_score(0.60) == QualityLevel.MEDIUM
        assert QualityLevel.from_score(0.69) == QualityLevel.MEDIUM

    def test_from_score_low(self):
        """Test LOW level for scores >= 0.30 and < 0.50."""
        assert QualityLevel.from_score(0.30) == QualityLevel.LOW
        assert QualityLevel.from_score(0.40) == QualityLevel.LOW
        assert QualityLevel.from_score(0.49) == QualityLevel.LOW

    def test_from_score_blocked(self):
        """Test BLOCKED level for scores < 0.30."""
        assert QualityLevel.from_score(0.29) == QualityLevel.BLOCKED
        assert QualityLevel.from_score(0.0) == QualityLevel.BLOCKED
        assert QualityLevel.from_score(-0.1) == QualityLevel.BLOCKED

    def test_from_score_invalid_triggers_blocked(self):
        """Test that invalid data results in BLOCKED."""
        assert QualityLevel.from_score(0.5, is_valid=False) == QualityLevel.BLOCKED
        assert QualityLevel.from_score(0.9, is_valid=False) == QualityLevel.BLOCKED


class TestQualityResult:
    """Tests for QualityResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = QualityResult(
            score=0.75,
            level=QualityLevel.HIGH,
        )
        assert result.score == 0.75
        assert result.level == QualityLevel.HIGH
        assert result.is_valid is True
        assert result.error is None
        assert result.components == {}
        assert result.metadata == {}

    def test_score_clamping_high(self):
        """Test score is clamped to max 1.0."""
        result = QualityResult(score=1.5, level=QualityLevel.HIGH)
        assert result.score == 1.0

    def test_score_clamping_low(self):
        """Test score is clamped to min 0.0."""
        result = QualityResult(score=-0.5, level=QualityLevel.LOW)
        assert result.score == 0.0

    def test_auto_level_from_score(self):
        """Test level is auto-computed if None."""
        result = QualityResult(score=0.6, level=None)  # type: ignore
        assert result.level == QualityLevel.MEDIUM

    def test_blocked_factory(self):
        """Test blocked() factory method."""
        result = QualityResult.blocked("Test error")
        assert result.score == 0.0
        assert result.level == QualityLevel.BLOCKED
        assert result.is_valid is False
        assert result.error == "Test error"

    def test_from_score_factory(self):
        """Test from_score() factory method."""
        result = QualityResult.from_score(
            0.8,
            components={"a": 0.9, "b": 0.7},
            metadata={"source": "test"},
        )
        assert result.score == 0.8
        assert result.level == QualityLevel.HIGH
        assert result.components == {"a": 0.9, "b": 0.7}
        assert result.metadata == {"source": "test"}

    def test_to_dict(self):
        """Test to_dict() serialization."""
        result = QualityResult(
            score=0.75,
            level=QualityLevel.HIGH,
            components={"test": 0.8},
            metadata={"key": "value"},
            is_valid=True,
            error=None,
        )
        d = result.to_dict()

        assert d["score"] == 0.75
        assert d["level"] == "high"
        assert d["components"] == {"test": 0.8}
        assert d["metadata"] == {"key": "value"}
        assert d["is_valid"] is True
        assert d["error"] is None


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_factory(self):
        """Test valid() factory method."""
        result = ValidationResult.valid()
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_valid_with_metadata(self):
        """Test valid() with metadata."""
        result = ValidationResult.valid(metadata={"checked": True})
        assert result.is_valid is True
        assert result.metadata == {"checked": True}

    def test_invalid_factory(self):
        """Test invalid() factory method."""
        result = ValidationResult.invalid("Error 1", "Error 2")
        assert result.is_valid is False
        assert result.errors == ["Error 1", "Error 2"]

    def test_add_error(self):
        """Test add_error() method."""
        result = ValidationResult(is_valid=True)
        result.add_error("New error")
        assert result.is_valid is False
        assert "New error" in result.errors

    def test_add_warning(self):
        """Test add_warning() method."""
        result = ValidationResult(is_valid=True)
        result.add_warning("Warning message")
        assert result.is_valid is True  # Warnings don't affect validity
        assert "Warning message" in result.warnings


class TestQualityScorerProtocol:
    """Tests for QualityScorer protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that QualityScorer can be used with isinstance."""
        # Create a class that implements the protocol
        class TestScorer:
            def score(self, data: dict) -> QualityResult:
                return QualityResult.from_score(0.5)

        scorer = TestScorer()
        assert isinstance(scorer, QualityScorer)

    def test_non_implementor_fails_check(self):
        """Test that non-implementors fail isinstance check."""
        class NotAScorer:
            def something_else(self):
                pass

        obj = NotAScorer()
        assert not isinstance(obj, QualityScorer)


class TestBatchQualityScorerProtocol:
    """Tests for BatchQualityScorer protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that BatchQualityScorer can be used with isinstance."""
        class TestBatchScorer:
            def score(self, data: dict) -> QualityResult:
                return QualityResult.from_score(0.5)

            def score_batch(self, data_list: list) -> list[QualityResult]:
                return [self.score(d) for d in data_list]

        scorer = TestBatchScorer()
        assert isinstance(scorer, BatchQualityScorer)

    def test_partial_implementation_fails(self):
        """Test that partial implementations fail isinstance check."""
        class PartialScorer:
            def score(self, data: dict) -> QualityResult:
                return QualityResult.from_score(0.5)
            # Missing score_batch

        obj = PartialScorer()
        assert not isinstance(obj, BatchQualityScorer)
