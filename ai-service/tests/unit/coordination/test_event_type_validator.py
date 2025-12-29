"""Unit tests for EventTypeValidator.

P0 December 2025: Tests for event type validation and normalization enforcement.
"""

import os
import pytest
from unittest.mock import patch

from app.coordination.event_normalization import (
    CANONICAL_EVENT_NAMES,
    CANONICAL_EVENT_TYPES,
    EventTypeValidator,
    UnknownEventTypeError,
    get_event_validator,
    normalize_event_type,
    validate_event_type,
    _ensure_data_event_types_loaded,
)


class TestCanonicalEventTypes:
    """Tests for CANONICAL_EVENT_TYPES set."""

    def test_canonical_set_not_empty(self):
        """Test that canonical set has entries."""
        _ensure_data_event_types_loaded()
        assert len(CANONICAL_EVENT_TYPES) > 50  # We have 150+ event types

    def test_canonical_set_contains_data_event_type_names(self):
        """Test that canonical set contains DataEventType enum names."""
        _ensure_data_event_types_loaded()

        # Some key event type names should be present
        assert "TRAINING_COMPLETED" in CANONICAL_EVENT_TYPES
        assert "DATA_SYNC_COMPLETED" in CANONICAL_EVENT_TYPES
        assert "MODEL_PROMOTED" in CANONICAL_EVENT_TYPES

    def test_canonical_set_contains_data_event_type_values(self):
        """Test that canonical set contains DataEventType enum values."""
        _ensure_data_event_types_loaded()

        # Values (lowercase) should also be present
        assert "training_completed" in CANONICAL_EVENT_TYPES
        assert "sync_completed" in CANONICAL_EVENT_TYPES
        assert "model_promoted" in CANONICAL_EVENT_TYPES

    def test_canonical_set_contains_normalized_names(self):
        """Test that canonical set contains normalized canonical names."""
        _ensure_data_event_types_loaded()

        # All canonical names from CANONICAL_EVENT_NAMES should be in the set
        for canonical in CANONICAL_EVENT_NAMES.values():
            assert canonical in CANONICAL_EVENT_TYPES


class TestEventTypeValidator:
    """Tests for EventTypeValidator class."""

    def setup_method(self):
        """Reset unknown events tracker before each test."""
        EventTypeValidator.reset_unknown_events()

    def test_validator_default_warn_mode(self):
        """Test that validator defaults to warn mode."""
        validator = EventTypeValidator()
        assert validator.strict is False

    def test_validator_strict_mode_explicit(self):
        """Test that validator can be set to strict mode."""
        validator = EventTypeValidator(strict=True)
        assert validator.strict is True

    def test_validator_strict_mode_from_env(self):
        """Test that validator reads strict mode from environment."""
        with patch.dict(os.environ, {"RINGRIFT_EVENT_VALIDATION_STRICT": "true"}):
            validator = EventTypeValidator(strict=None)
            assert validator.strict is True

    def test_validate_canonical_event(self):
        """Test validation of canonical event type."""
        validator = EventTypeValidator(strict=False)

        is_valid, msg = validator.validate("TRAINING_COMPLETED")

        assert is_valid is True
        assert "Valid" in msg or "canonical" in msg.lower()

    def test_validate_normalized_event(self):
        """Test validation of event that gets normalized."""
        validator = EventTypeValidator(strict=False)

        # SYNC_COMPLETE normalizes to DATA_SYNC_COMPLETED
        is_valid, msg = validator.validate("SYNC_COMPLETE")

        assert is_valid is True
        assert "Normalized" in msg
        assert "DATA_SYNC_COMPLETED" in msg

    def test_validate_unknown_event_warn_mode(self):
        """Test validation of unknown event in warn mode."""
        validator = EventTypeValidator(strict=False)

        is_valid, msg = validator.validate("COMPLETELY_UNKNOWN_EVENT")

        assert is_valid is False
        assert "Unknown" in msg

    def test_validate_unknown_event_strict_mode(self):
        """Test validation of unknown event in strict mode raises."""
        validator = EventTypeValidator(strict=True)

        with pytest.raises(UnknownEventTypeError) as exc_info:
            validator.validate("COMPLETELY_UNKNOWN_EVENT")

        assert "COMPLETELY_UNKNOWN_EVENT" in str(exc_info.value)

    def test_validate_unknown_event_suggestions(self):
        """Test that unknown events get similar suggestions."""
        validator = EventTypeValidator(strict=False)

        # TRAINING_COMPLETE is close to TRAINING_COMPLETED
        is_valid, msg = validator.validate("TRAINING_COMPLETEDD")  # Typo

        assert is_valid is False
        # Should suggest TRAINING_COMPLETED or similar
        assert "TRAINING" in msg

    def test_is_valid_method(self):
        """Test is_valid() method doesn't raise in strict mode."""
        validator = EventTypeValidator(strict=True)

        # is_valid should not raise, just return bool
        assert validator.is_valid("TRAINING_COMPLETED") is True
        assert validator.is_valid("COMPLETELY_UNKNOWN_EVENT") is False

    def test_unknown_events_tracking(self):
        """Test that unknown events are tracked."""
        validator = EventTypeValidator(strict=False)

        validator.validate("UNKNOWN_EVENT_ONE")
        validator.validate("UNKNOWN_EVENT_ONE")
        validator.validate("UNKNOWN_EVENT_TWO")

        unknown = EventTypeValidator.get_unknown_events()

        assert "UNKNOWN_EVENT_ONE" in unknown
        assert unknown["UNKNOWN_EVENT_ONE"] == 2
        assert "UNKNOWN_EVENT_TWO" in unknown
        assert unknown["UNKNOWN_EVENT_TWO"] == 1

    def test_reset_unknown_events(self):
        """Test resetting unknown events tracker."""
        validator = EventTypeValidator(strict=False)

        validator.validate("UNKNOWN_FOR_RESET_TEST")
        assert len(EventTypeValidator.get_unknown_events()) > 0

        EventTypeValidator.reset_unknown_events()

        assert len(EventTypeValidator.get_unknown_events()) == 0


class TestUnknownEventTypeError:
    """Tests for UnknownEventTypeError exception."""

    def test_error_message_contains_event_type(self):
        """Test that error message contains the event type."""
        error = UnknownEventTypeError("MY_BAD_EVENT")

        assert "MY_BAD_EVENT" in str(error)

    def test_error_message_contains_suggestions(self):
        """Test that error message contains suggestions when provided."""
        error = UnknownEventTypeError(
            "MY_BAD_EVENT",
            suggestions=["SIMILAR_EVENT", "ANOTHER_EVENT"],
        )

        assert "MY_BAD_EVENT" in str(error)
        assert "Did you mean" in str(error)
        assert "SIMILAR_EVENT" in str(error)

    def test_error_properties(self):
        """Test that error has correct properties."""
        error = UnknownEventTypeError(
            "MY_BAD_EVENT",
            suggestions=["SUGGESTION_1", "SUGGESTION_2"],
        )

        assert error.event_type == "MY_BAD_EVENT"
        assert error.suggestions == ["SUGGESTION_1", "SUGGESTION_2"]


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Reset state before each test."""
        EventTypeValidator.reset_unknown_events()

    def test_validate_event_type_function(self):
        """Test validate_event_type() module function."""
        is_valid, msg = validate_event_type("TRAINING_COMPLETED")

        assert is_valid is True

    def test_get_event_validator_singleton(self):
        """Test get_event_validator() returns consistent instance."""
        v1 = get_event_validator()
        v2 = get_event_validator()

        assert v1 is v2


class TestFindSimilar:
    """Tests for similar event type suggestions."""

    def test_find_similar_substring_match(self):
        """Test finding similar events by substring."""
        validator = EventTypeValidator(strict=False)

        # TRAINING is a substring of many events
        similar = validator._find_similar("TRAINING_X")

        # Should find TRAINING_COMPLETED, TRAINING_STARTED, etc.
        assert any("TRAINING" in s for s in similar)

    def test_find_similar_word_overlap(self):
        """Test finding similar events by word overlap."""
        validator = EventTypeValidator(strict=False)

        # SYNC and COMPLETED both appear in DATA_SYNC_COMPLETED
        similar = validator._find_similar("SYNC_COMPLETED")

        # Should suggest DATA_SYNC_COMPLETED
        assert any("SYNC" in s for s in similar)

    def test_find_similar_max_results(self):
        """Test that find_similar respects max_results."""
        validator = EventTypeValidator(strict=False)

        similar = validator._find_similar("TRAINING", max_results=2)

        assert len(similar) <= 2

    def test_find_similar_no_match(self):
        """Test find_similar with completely unrelated string."""
        validator = EventTypeValidator(strict=False)

        similar = validator._find_similar("XYZZYQWERTY12345")

        # May return some low-score matches or empty
        assert isinstance(similar, list)


class TestIntegrationWithNormalization:
    """Tests for integration between validation and normalization."""

    def test_validate_after_normalization(self):
        """Test that normalized events are validated correctly."""
        validator = EventTypeValidator(strict=False)

        # These variants should normalize and then validate
        variants = [
            "sync_complete",
            "SYNC_COMPLETE",
            "training_complete",
            "TRAINING_COMPLETE",
            "model_promoted",
            "MODEL_PROMOTED",
        ]

        for variant in variants:
            is_valid, msg = validator.validate(variant)
            assert is_valid is True, f"Failed for variant: {variant}"

    def test_normalization_before_validation(self):
        """Test that normalization happens before validation check."""
        # normalize_event_type should be called first
        normalized = normalize_event_type("SYNC_COMPLETE")
        assert normalized == "DATA_SYNC_COMPLETED"

        # Then validation should succeed on the normalized form
        validator = EventTypeValidator(strict=False)
        is_valid, _ = validator.validate("SYNC_COMPLETE")
        assert is_valid is True
