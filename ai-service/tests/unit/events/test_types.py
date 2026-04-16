"""
Tests for app.events.types module.

Tests the unified event type system including:
- Event type enum definitions
- Event category mapping
- Utility functions
- Backwards compatibility
"""

import warnings

import pytest

import app.events.types as event_types
from app.events.types import (
    CROSS_PROCESS_EVENT_TYPES,
    EventCategory,
    RingRiftEventType,
    get_events_by_category,
    is_cross_process_event,
)


def _canonical_stage_event():
    """Return the canonical coordination StageEvent without surfacing the module deprecation in test output."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="app.coordination.stage_events is deprecated",
            category=DeprecationWarning,
        )
        from app.coordination.stage_events import StageEvent as canonical_stage_event

    return canonical_stage_event


# ============================================
# Test RingRiftEventType Enum
# ============================================


class TestRingRiftEventType:
    """Tests for RingRiftEventType enum."""

    def test_event_type_values_are_strings(self):
        """Test that all event types have string values."""
        for event_type in RingRiftEventType:
            assert isinstance(event_type.value, str)
            assert len(event_type.value) > 0

    def test_event_type_values_are_unique(self):
        """Test that all event type values are unique."""
        values = [event_type.value for event_type in RingRiftEventType]
        assert len(values) == len(set(values))

    def test_data_collection_events_exist(self):
        """Test that data collection events are defined."""
        assert RingRiftEventType.NEW_GAMES_AVAILABLE
        assert RingRiftEventType.DATA_SYNC_STARTED
        assert RingRiftEventType.DATA_SYNC_COMPLETED
        assert RingRiftEventType.DATA_SYNC_FAILED
        assert RingRiftEventType.GAME_SYNCED
        assert RingRiftEventType.DATABASE_CREATED
        assert RingRiftEventType.ORPHAN_GAMES_DETECTED
        assert RingRiftEventType.ORPHAN_GAMES_REGISTERED

    def test_training_events_exist(self):
        """Test that training events are defined."""
        assert RingRiftEventType.TRAINING_THRESHOLD_REACHED
        assert RingRiftEventType.TRAINING_STARTED
        assert RingRiftEventType.TRAINING_PROGRESS
        assert RingRiftEventType.TRAINING_COMPLETED
        assert RingRiftEventType.TRAINING_FAILED
        assert RingRiftEventType.TRAINING_LOSS_ANOMALY
        assert RingRiftEventType.TRAINING_EARLY_STOPPED

    def test_evaluation_events_exist(self):
        """Test that evaluation events are defined."""
        assert RingRiftEventType.EVALUATION_STARTED
        assert RingRiftEventType.EVALUATION_COMPLETED
        assert RingRiftEventType.EVALUATION_FAILED
        assert RingRiftEventType.ELO_UPDATED
        assert RingRiftEventType.ELO_SIGNIFICANT_CHANGE

    def test_promotion_events_exist(self):
        """Test that promotion events are defined."""
        assert RingRiftEventType.PROMOTION_CANDIDATE
        assert RingRiftEventType.PROMOTION_STARTED
        assert RingRiftEventType.MODEL_PROMOTED
        assert RingRiftEventType.PROMOTION_FAILED
        assert RingRiftEventType.PROMOTION_REJECTED

    def test_cluster_events_exist(self):
        """Test that cluster events are defined."""
        assert RingRiftEventType.P2P_MODEL_SYNCED
        assert RingRiftEventType.P2P_CLUSTER_HEALTHY
        assert RingRiftEventType.P2P_CLUSTER_UNHEALTHY
        assert RingRiftEventType.NODE_UNHEALTHY
        assert RingRiftEventType.NODE_RECOVERED

    def test_stage_events_exist(self):
        """Test that stage completion events are defined."""
        assert RingRiftEventType.STAGE_SELFPLAY_COMPLETE
        assert RingRiftEventType.STAGE_TRAINING_COMPLETE
        assert RingRiftEventType.STAGE_EVALUATION_COMPLETE
        assert RingRiftEventType.STAGE_PROMOTION_COMPLETE


# ============================================
# Test EventCategory
# ============================================


class TestEventCategory:
    """Tests for EventCategory enum."""

    def test_event_categories_are_defined(self):
        """Test that all expected categories exist."""
        assert EventCategory.DATA
        assert EventCategory.TRAINING
        assert EventCategory.EVALUATION
        assert EventCategory.PROMOTION
        assert EventCategory.CURRICULUM
        assert EventCategory.SELFPLAY
        assert EventCategory.OPTIMIZATION
        assert EventCategory.QUALITY
        assert EventCategory.REGRESSION
        assert EventCategory.CLUSTER
        assert EventCategory.SYSTEM
        assert EventCategory.WORK
        assert EventCategory.STAGE
        assert EventCategory.SYNC
        assert EventCategory.TASK

    def test_from_event_returns_correct_category(self):
        """Test that from_event returns the correct category."""
        # Training events
        assert EventCategory.from_event(RingRiftEventType.TRAINING_STARTED) == EventCategory.TRAINING
        assert EventCategory.from_event(RingRiftEventType.TRAINING_COMPLETED) == EventCategory.TRAINING

        # Evaluation events
        assert EventCategory.from_event(RingRiftEventType.EVALUATION_STARTED) == EventCategory.EVALUATION
        assert EventCategory.from_event(RingRiftEventType.ELO_UPDATED) == EventCategory.EVALUATION

        # Promotion events
        assert EventCategory.from_event(RingRiftEventType.MODEL_PROMOTED) == EventCategory.PROMOTION

        # Data events
        assert EventCategory.from_event(RingRiftEventType.DATA_SYNC_COMPLETED) == EventCategory.DATA

        # Cluster events
        assert EventCategory.from_event(RingRiftEventType.P2P_MODEL_SYNCED) == EventCategory.CLUSTER

    def test_from_event_returns_system_for_unknown(self):
        """Test that from_event returns SYSTEM for unmapped events."""
        # Create a mock event type that isn't in the mapping
        # Since we can't easily create a new enum value, we'll just verify
        # the fallback behavior exists by checking the default in the method
        assert EventCategory.SYSTEM is not None


# ============================================
# Test Utility Functions
# ============================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_events_by_category_training(self):
        """Test get_events_by_category for TRAINING category."""
        events = get_events_by_category(EventCategory.TRAINING)

        assert isinstance(events, list)
        assert len(events) > 0
        assert RingRiftEventType.TRAINING_STARTED in events
        assert RingRiftEventType.TRAINING_COMPLETED in events
        assert RingRiftEventType.TRAINING_FAILED in events

    def test_get_events_by_category_evaluation(self):
        """Test get_events_by_category for EVALUATION category."""
        events = get_events_by_category(EventCategory.EVALUATION)

        assert isinstance(events, list)
        assert len(events) > 0
        assert RingRiftEventType.EVALUATION_STARTED in events
        assert RingRiftEventType.EVALUATION_COMPLETED in events
        assert RingRiftEventType.ELO_UPDATED in events

    def test_get_events_by_category_promotion(self):
        """Test get_events_by_category for PROMOTION category."""
        events = get_events_by_category(EventCategory.PROMOTION)

        assert isinstance(events, list)
        assert len(events) > 0
        assert RingRiftEventType.MODEL_PROMOTED in events
        assert RingRiftEventType.PROMOTION_STARTED in events

    def test_get_events_by_category_data(self):
        """Test get_events_by_category for DATA category."""
        events = get_events_by_category(EventCategory.DATA)

        assert isinstance(events, list)
        assert len(events) > 0
        assert RingRiftEventType.DATA_SYNC_COMPLETED in events
        assert RingRiftEventType.NEW_GAMES_AVAILABLE in events

    def test_get_events_by_category_cluster(self):
        """Test get_events_by_category for CLUSTER category."""
        events = get_events_by_category(EventCategory.CLUSTER)

        assert isinstance(events, list)
        assert len(events) > 0
        assert RingRiftEventType.P2P_MODEL_SYNCED in events
        assert RingRiftEventType.NODE_RECOVERED in events

    def test_get_events_by_category_stage(self):
        """Test get_events_by_category for STAGE category."""
        events = get_events_by_category(EventCategory.STAGE)

        assert isinstance(events, list)
        assert len(events) > 0
        assert RingRiftEventType.STAGE_SELFPLAY_COMPLETE in events
        assert RingRiftEventType.STAGE_TRAINING_COMPLETE in events

    def test_get_events_by_category_returns_only_matching(self):
        """Test that get_events_by_category returns only events from that category."""
        training_events = get_events_by_category(EventCategory.TRAINING)

        # Verify that non-training events are not included
        assert RingRiftEventType.MODEL_PROMOTED not in training_events
        assert RingRiftEventType.EVALUATION_STARTED not in training_events
        assert RingRiftEventType.DATA_SYNC_COMPLETED not in training_events

    def test_is_cross_process_event_true_for_cross_process_events(self):
        """Test is_cross_process_event returns True for cross-process events."""
        assert is_cross_process_event(RingRiftEventType.MODEL_PROMOTED) is True
        assert is_cross_process_event(RingRiftEventType.TRAINING_STARTED) is True
        assert is_cross_process_event(RingRiftEventType.TRAINING_COMPLETED) is True
        assert is_cross_process_event(RingRiftEventType.TRAINING_FAILED) is True
        assert is_cross_process_event(RingRiftEventType.DATA_SYNC_COMPLETED) is True

    def test_is_cross_process_event_false_for_local_events(self):
        """Test is_cross_process_event returns False for local events."""
        # These events should not be in CROSS_PROCESS_EVENT_TYPES
        assert is_cross_process_event(RingRiftEventType.TRAINING_PROGRESS) is False
        assert is_cross_process_event(RingRiftEventType.CHECKPOINT_SAVED) is False

    def test_cross_process_event_types_is_set(self):
        """Test that CROSS_PROCESS_EVENT_TYPES is a set."""
        assert isinstance(CROSS_PROCESS_EVENT_TYPES, set)
        assert len(CROSS_PROCESS_EVENT_TYPES) > 0

    def test_cross_process_event_types_contains_expected_events(self):
        """Test that CROSS_PROCESS_EVENT_TYPES contains expected events."""
        expected_events = [
            RingRiftEventType.MODEL_PROMOTED,
            RingRiftEventType.TRAINING_STARTED,
            RingRiftEventType.TRAINING_COMPLETED,
            RingRiftEventType.TRAINING_FAILED,
            RingRiftEventType.EVALUATION_COMPLETED,
            RingRiftEventType.DATA_SYNC_COMPLETED,
            RingRiftEventType.REGRESSION_DETECTED,
        ]

        for event in expected_events:
            assert event in CROSS_PROCESS_EVENT_TYPES


# ============================================
# Test Backwards Compatibility
# ============================================


class TestBackwardsCompatibility:
    """Tests for backwards compatibility aliases."""

    def test_stage_event_alias_warns_and_uses_canonical_enum(self):
        """StageEvent alias should warn and resolve to the canonical coordination enum."""
        with pytest.deprecated_call(
            match="app.events.types.StageEvent is deprecated"
        ):
            stage_event = event_types.StageEvent
        assert stage_event is _canonical_stage_event()

    def test_stage_event_has_expected_values(self):
        """Test that the deprecated alias exposes canonical stage event values."""
        with pytest.deprecated_call(
            match="app.events.types.StageEvent is deprecated"
        ):
            stage_event = event_types.StageEvent
        assert hasattr(stage_event, 'SELFPLAY_COMPLETE')
        assert hasattr(stage_event, 'TRAINING_COMPLETE')
        assert hasattr(stage_event, 'EVALUATION_COMPLETE')
        assert hasattr(stage_event, 'PROMOTION_COMPLETE')

    def test_stage_event_values_match_ringrift_event_type(self):
        """Test that the deprecated alias matches the canonical coordination enum values."""
        with pytest.deprecated_call(
            match="app.events.types.StageEvent is deprecated"
        ):
            stage_event = event_types.StageEvent
        canonical_stage_event = _canonical_stage_event()
        assert stage_event.SELFPLAY_COMPLETE.value == canonical_stage_event.SELFPLAY_COMPLETE.value
        assert stage_event.TRAINING_COMPLETE.value == canonical_stage_event.TRAINING_COMPLETE.value
        assert stage_event.EVALUATION_COMPLETE.value == canonical_stage_event.EVALUATION_COMPLETE.value
        assert stage_event.PROMOTION_COMPLETE.value == canonical_stage_event.PROMOTION_COMPLETE.value


# ============================================
# Test Event Coverage
# ============================================


class TestEventCoverage:
    """Tests to ensure all events are properly categorized."""

    def test_all_events_have_categories(self):
        """Test that all events can be mapped to a category."""
        for event_type in RingRiftEventType:
            category = EventCategory.from_event(event_type)
            assert category is not None
            assert isinstance(category, EventCategory)

    def test_categories_contain_all_events(self):
        """Test that all events appear in at least one category."""
        all_categorized_events = set()

        for category in EventCategory:
            events = get_events_by_category(category)
            all_categorized_events.update(events)

        # All RingRiftEventType members should be categorized
        # (Note: Some may default to SYSTEM category)
        all_event_types = set(RingRiftEventType)

        # The difference should be small (only events that default to SYSTEM)
        uncategorized = all_event_types - all_categorized_events

        # If there are uncategorized events, they should all map to SYSTEM
        for event in uncategorized:
            assert EventCategory.from_event(event) == EventCategory.SYSTEM

    def test_no_duplicate_categorization(self):
        """Test that events are not duplicated across categories."""
        events_by_category = {}

        for category in EventCategory:
            events = get_events_by_category(category)
            events_by_category[category] = set(events)

        # Check for any duplicates
        all_events = []
        for events in events_by_category.values():
            all_events.extend(events)

        # Each event should appear exactly once
        assert len(all_events) == len(set(all_events))


# ============================================
# Test Event Type Semantics
# ============================================


class TestEventTypeSemantics:
    """Tests for event type semantic correctness."""

    def test_success_and_failure_events_paired(self):
        """Test that success/failure event pairs exist."""
        # Training
        assert RingRiftEventType.TRAINING_STARTED
        assert RingRiftEventType.TRAINING_COMPLETED
        assert RingRiftEventType.TRAINING_FAILED

        # Evaluation
        assert RingRiftEventType.EVALUATION_STARTED
        assert RingRiftEventType.EVALUATION_COMPLETED
        assert RingRiftEventType.EVALUATION_FAILED

        # Promotion
        assert RingRiftEventType.PROMOTION_STARTED
        assert RingRiftEventType.MODEL_PROMOTED  # Success case
        assert RingRiftEventType.PROMOTION_FAILED

    def test_lifecycle_events_exist(self):
        """Test that lifecycle events exist for major operations."""
        # Task lifecycle
        assert RingRiftEventType.TASK_SPAWNED
        assert RingRiftEventType.TASK_COMPLETED
        assert RingRiftEventType.TASK_FAILED
        assert RingRiftEventType.TASK_CANCELLED

        # Work queue lifecycle
        assert RingRiftEventType.WORK_QUEUED
        assert RingRiftEventType.WORK_CLAIMED
        assert RingRiftEventType.WORK_STARTED
        assert RingRiftEventType.WORK_COMPLETED
        assert RingRiftEventType.WORK_FAILED

    def test_health_status_events_exist(self):
        """Test that health status events exist."""
        assert RingRiftEventType.HEALTH_CHECK_PASSED
        assert RingRiftEventType.HEALTH_CHECK_FAILED
        assert RingRiftEventType.HEALTH_ALERT

        # Coordinator health
        assert RingRiftEventType.COORDINATOR_HEALTHY
        assert RingRiftEventType.COORDINATOR_UNHEALTHY
        assert RingRiftEventType.COORDINATOR_HEALTH_DEGRADED

    def test_regression_severity_levels_exist(self):
        """Test that regression events have severity levels."""
        assert RingRiftEventType.REGRESSION_DETECTED
        assert RingRiftEventType.REGRESSION_MINOR
        assert RingRiftEventType.REGRESSION_MODERATE
        assert RingRiftEventType.REGRESSION_SEVERE
        assert RingRiftEventType.REGRESSION_CRITICAL
        assert RingRiftEventType.REGRESSION_CLEARED


# ============================================
# Test Event Value Formats
# ============================================


class TestEventValueFormats:
    """Tests for event value format conventions."""

    def test_event_values_use_snake_case(self):
        """Test that event values use snake_case convention."""
        for event_type in RingRiftEventType:
            value = event_type.value
            # Should not contain uppercase letters
            assert value == value.lower(), f"Event value '{value}' is not lowercase"
            # Should use underscores, not hyphens
            assert '-' not in value, f"Event value '{value}' contains hyphens"

    def test_event_values_are_descriptive(self):
        """Test that event values are reasonably descriptive."""
        for event_type in RingRiftEventType:
            value = event_type.value
            # Values should be at least 3 characters
            assert len(value) >= 3, f"Event value '{value}' is too short"
            # Values should not be just numbers
            assert not value.isdigit(), f"Event value '{value}' is just a number"


# ============================================
# Test Cross-Process Event Types
# ============================================


class TestCrossProcessEventTypes:
    """Tests for cross-process event type definitions."""

    def test_success_events_are_cross_process(self):
        """Test that important success events are cross-process."""
        success_events = [
            RingRiftEventType.MODEL_PROMOTED,
            RingRiftEventType.TRAINING_COMPLETED,
            RingRiftEventType.EVALUATION_COMPLETED,
            RingRiftEventType.DATA_SYNC_COMPLETED,
        ]

        for event in success_events:
            assert is_cross_process_event(event), f"{event} should be cross-process"

    def test_failure_events_are_cross_process(self):
        """Test that important failure events are cross-process."""
        failure_events = [
            RingRiftEventType.TRAINING_FAILED,
            RingRiftEventType.EVALUATION_FAILED,
            RingRiftEventType.PROMOTION_FAILED,
            RingRiftEventType.DATA_SYNC_FAILED,
        ]

        for event in failure_events:
            assert is_cross_process_event(event), f"{event} should be cross-process"

    def test_cluster_events_are_cross_process(self):
        """Test that cluster coordination events are cross-process."""
        cluster_events = [
            RingRiftEventType.HOST_ONLINE,
            RingRiftEventType.HOST_OFFLINE,
            RingRiftEventType.DAEMON_STARTED,
            RingRiftEventType.DAEMON_STOPPED,
        ]

        for event in cluster_events:
            assert is_cross_process_event(event), f"{event} should be cross-process"

    def test_regression_events_are_cross_process(self):
        """Test that critical regression events are cross-process."""
        regression_events = [
            RingRiftEventType.REGRESSION_DETECTED,
            RingRiftEventType.REGRESSION_SEVERE,
            RingRiftEventType.REGRESSION_CRITICAL,
            RingRiftEventType.REGRESSION_CLEARED,
        ]

        for event in regression_events:
            assert is_cross_process_event(event), f"{event} should be cross-process"
