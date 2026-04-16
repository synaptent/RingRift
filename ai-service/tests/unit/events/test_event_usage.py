"""
Tests for event usage patterns and best practices.

Tests that events are used correctly throughout the codebase.
"""

import pytest

from app.events import (
    CROSS_PROCESS_EVENT_TYPES,
    EventCategory,
    RingRiftEventType,
    get_events_by_category,
    is_cross_process_event,
)


# ============================================
# Test Event Usage Patterns
# ============================================


class TestEventUsagePatterns:
    """Tests for correct event usage patterns."""

    def test_import_from_events_package(self):
        """Test that events can be imported from app.events package."""
        # This test verifies the package __init__.py exports
        from app.events import RingRiftEventType as EventTypeImport
        from app.events import EventCategory as CategoryImport

        assert EventTypeImport is RingRiftEventType
        assert CategoryImport is EventCategory

    def test_event_type_can_be_used_in_equality_checks(self):
        """Test that event types work in equality checks."""
        event1 = RingRiftEventType.TRAINING_STARTED
        event2 = RingRiftEventType.TRAINING_STARTED
        event3 = RingRiftEventType.TRAINING_COMPLETED

        assert event1 == event2
        assert event1 != event3

    def test_event_type_can_be_used_in_dictionary_keys(self):
        """Test that event types can be used as dictionary keys."""
        event_handlers = {
            RingRiftEventType.TRAINING_STARTED: "handle_training_start",
            RingRiftEventType.TRAINING_COMPLETED: "handle_training_complete",
        }

        assert event_handlers[RingRiftEventType.TRAINING_STARTED] == "handle_training_start"
        assert event_handlers[RingRiftEventType.TRAINING_COMPLETED] == "handle_training_complete"

    def test_event_type_can_be_used_in_sets(self):
        """Test that event types can be used in sets."""
        event_set = {
            RingRiftEventType.TRAINING_STARTED,
            RingRiftEventType.TRAINING_COMPLETED,
            RingRiftEventType.TRAINING_STARTED,  # Duplicate
        }

        # Set should deduplicate
        assert len(event_set) == 2
        assert RingRiftEventType.TRAINING_STARTED in event_set
        assert RingRiftEventType.TRAINING_COMPLETED in event_set

    def test_event_category_can_be_used_in_if_statements(self):
        """Test that event categories work in conditional logic."""
        category = EventCategory.from_event(RingRiftEventType.TRAINING_STARTED)

        if category == EventCategory.TRAINING:
            result = "training"
        elif category == EventCategory.EVALUATION:
            result = "evaluation"
        else:
            result = "other"

        assert result == "training"


# ============================================
# Test Event Filtering and Routing
# ============================================


class TestEventFilteringAndRouting:
    """Tests for event filtering and routing patterns."""

    def test_filter_events_by_category(self):
        """Test filtering events by category."""
        all_events = [
            RingRiftEventType.TRAINING_STARTED,
            RingRiftEventType.EVALUATION_COMPLETED,
            RingRiftEventType.MODEL_PROMOTED,
            RingRiftEventType.TRAINING_COMPLETED,
        ]

        training_events = [
            e for e in all_events
            if EventCategory.from_event(e) == EventCategory.TRAINING
        ]

        assert len(training_events) == 2
        assert RingRiftEventType.TRAINING_STARTED in training_events
        assert RingRiftEventType.TRAINING_COMPLETED in training_events

    def test_filter_cross_process_events(self):
        """Test filtering cross-process events."""
        all_events = [
            RingRiftEventType.TRAINING_STARTED,
            RingRiftEventType.TRAINING_PROGRESS,
            RingRiftEventType.TRAINING_COMPLETED,
            RingRiftEventType.CHECKPOINT_SAVED,
        ]

        cross_process = [e for e in all_events if is_cross_process_event(e)]

        # TRAINING_STARTED and TRAINING_COMPLETED should be cross-process
        # TRAINING_PROGRESS and CHECKPOINT_SAVED should not be
        assert RingRiftEventType.TRAINING_STARTED in cross_process
        assert RingRiftEventType.TRAINING_COMPLETED in cross_process
        assert RingRiftEventType.TRAINING_PROGRESS not in cross_process

    def test_route_events_by_category(self):
        """Test routing events to different handlers by category."""
        handlers_called = []

        def route_event(event_type):
            category = EventCategory.from_event(event_type)

            if category == EventCategory.TRAINING:
                handlers_called.append("training_handler")
            elif category == EventCategory.EVALUATION:
                handlers_called.append("evaluation_handler")
            elif category == EventCategory.PROMOTION:
                handlers_called.append("promotion_handler")

        route_event(RingRiftEventType.TRAINING_STARTED)
        route_event(RingRiftEventType.EVALUATION_COMPLETED)
        route_event(RingRiftEventType.MODEL_PROMOTED)

        assert handlers_called == [
            "training_handler",
            "evaluation_handler",
            "promotion_handler",
        ]


# ============================================
# Test Event Subscription Patterns
# ============================================


class TestEventSubscriptionPatterns:
    """Tests for event subscription patterns."""

    def test_subscribe_to_category(self):
        """Test subscribing to all events in a category."""
        # Simulate subscribing to all training events
        training_events = get_events_by_category(EventCategory.TRAINING)

        subscriptions = {event: True for event in training_events}

        # Should be subscribed to training events
        assert subscriptions[RingRiftEventType.TRAINING_STARTED] is True
        assert subscriptions[RingRiftEventType.TRAINING_COMPLETED] is True

        # Should not be subscribed to non-training events
        assert RingRiftEventType.MODEL_PROMOTED not in subscriptions
        assert RingRiftEventType.EVALUATION_COMPLETED not in subscriptions

    def test_subscribe_to_multiple_categories(self):
        """Test subscribing to multiple categories."""
        training_events = get_events_by_category(EventCategory.TRAINING)
        evaluation_events = get_events_by_category(EventCategory.EVALUATION)

        all_subscribed = set(training_events) | set(evaluation_events)

        assert RingRiftEventType.TRAINING_STARTED in all_subscribed
        assert RingRiftEventType.EVALUATION_COMPLETED in all_subscribed
        assert RingRiftEventType.MODEL_PROMOTED not in all_subscribed

    def test_subscribe_to_cross_process_events_only(self):
        """Test subscribing only to cross-process events."""
        # In a distributed system, might only want cross-process events
        cross_process_subscriptions = {
            event: True for event in CROSS_PROCESS_EVENT_TYPES
        }

        assert RingRiftEventType.MODEL_PROMOTED in cross_process_subscriptions
        assert RingRiftEventType.TRAINING_COMPLETED in cross_process_subscriptions

        # Local events should not be subscribed
        assert RingRiftEventType.TRAINING_PROGRESS not in cross_process_subscriptions


# ============================================
# Test Event Value Access
# ============================================


class TestEventValueAccess:
    """Tests for accessing event values."""

    def test_event_name_attribute(self):
        """Test accessing event enum name."""
        event = RingRiftEventType.TRAINING_STARTED
        assert event.name == "TRAINING_STARTED"

    def test_event_value_attribute(self):
        """Test accessing event enum value."""
        event = RingRiftEventType.TRAINING_STARTED
        assert event.value == "training_started"

    def test_get_event_by_value(self):
        """Test getting event by its value."""
        event = RingRiftEventType("training_started")
        assert event == RingRiftEventType.TRAINING_STARTED

    def test_get_event_by_name(self):
        """Test getting event by its name."""
        event = RingRiftEventType["TRAINING_STARTED"]
        assert event == RingRiftEventType.TRAINING_STARTED

    def test_invalid_event_value_raises_error(self):
        """Test that invalid event value raises ValueError."""
        with pytest.raises(ValueError):
            RingRiftEventType("nonexistent_event")

    def test_invalid_event_name_raises_error(self):
        """Test that invalid event name raises KeyError."""
        with pytest.raises(KeyError):
            RingRiftEventType["NONEXISTENT_EVENT"]


# ============================================
# Test Event Iteration
# ============================================


class TestEventIteration:
    """Tests for iterating over events."""

    def test_iterate_all_events(self):
        """Test iterating over all events."""
        events = list(RingRiftEventType)
        assert len(events) > 0
        assert all(isinstance(e, RingRiftEventType) for e in events)

    def test_iterate_events_by_category(self):
        """Test iterating over events in a specific category."""
        training_events = get_events_by_category(EventCategory.TRAINING)

        for event in training_events:
            assert isinstance(event, RingRiftEventType)
            assert EventCategory.from_event(event) == EventCategory.TRAINING

    def test_count_events_by_category(self):
        """Test counting events in each category."""
        counts = {}

        for category in EventCategory:
            events = get_events_by_category(category)
            counts[category] = len(events)

        # All categories should have at least one event
        # (except possibly some that default to SYSTEM)
        assert counts[EventCategory.TRAINING] > 0
        assert counts[EventCategory.EVALUATION] > 0
        assert counts[EventCategory.PROMOTION] > 0
        assert counts[EventCategory.DATA] > 0


# ============================================
# Test Event String Representation
# ============================================


class TestEventStringRepresentation:
    """Tests for event string representation."""

    def test_event_str_representation(self):
        """Test string representation of events."""
        event = RingRiftEventType.TRAINING_STARTED
        event_str = str(event)

        # Should contain both name and value information
        assert "RingRiftEventType" in event_str
        assert "TRAINING_STARTED" in event_str

    def test_event_repr_representation(self):
        """Test repr representation of events."""
        event = RingRiftEventType.TRAINING_STARTED
        event_repr = repr(event)

        # Should be a valid Python expression
        assert "RingRiftEventType" in event_repr
        assert "TRAINING_STARTED" in event_repr

    def test_category_str_representation(self):
        """Test string representation of categories."""
        category = EventCategory.TRAINING
        category_str = str(category)

        assert "EventCategory" in category_str
        assert "TRAINING" in category_str


# ============================================
# Test Backwards Compatibility Usage
# ============================================


class TestBackwardsCompatibilityUsage:
    """Tests for using backwards compatibility aliases."""

    def test_stage_event_can_be_compared(self):
        """Test that StageEvent values can be compared."""
        with pytest.deprecated_call(match="app.events.StageEvent is deprecated"):
            from app.events import StageEvent

        event1 = StageEvent.TRAINING_COMPLETE
        event2 = StageEvent.TRAINING_COMPLETE

        assert event1 == event2

    def test_stage_event_value_matches_ringrift_type(self):
        """Test that StageEvent values match RingRiftEventType."""
        with pytest.deprecated_call(match="app.events.StageEvent is deprecated"):
            from app.events import StageEvent

        assert StageEvent.SELFPLAY_COMPLETE.value == "selfplay_complete"
        assert StageEvent.TRAINING_COMPLETE.value == "training_complete"


# ============================================
# Test Event Type Safety
# ============================================


class TestEventTypeSafety:
    """Tests for event type safety."""

    def test_event_type_is_enum(self):
        """Test that RingRiftEventType is an Enum."""
        from enum import Enum

        assert issubclass(RingRiftEventType, Enum)

    def test_category_type_is_enum(self):
        """Test that EventCategory is an Enum."""
        from enum import Enum

        assert issubclass(EventCategory, Enum)

    def test_events_are_immutable(self):
        """Test that event enum members are immutable."""
        event = RingRiftEventType.TRAINING_STARTED

        # Cannot reassign enum values
        with pytest.raises(AttributeError):
            event.value = "new_value"

    def test_cannot_add_new_events_at_runtime(self):
        """Test that new events cannot be added at runtime."""
        # Enums are immutable - cannot add new members
        # Note: In Python 3.10+, setting attributes on enum class doesn't raise AttributeError
        # but the new attribute is not treated as an enum member
        initial_count = len(list(RingRiftEventType))

        # Try to add a new attribute
        RingRiftEventType.NEW_EVENT = "new_event"

        # The count of enum members should not change
        assert len(list(RingRiftEventType)) == initial_count

        # And it should not be a valid enum member
        assert not hasattr(RingRiftEventType, '__members__') or \
               'NEW_EVENT' not in RingRiftEventType.__members__
