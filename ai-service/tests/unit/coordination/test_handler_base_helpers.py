"""Tests for HandlerBase helper methods added in Sprint 17.2.

Tests for:
- Event payload normalization (_normalize_event_payload, _extract_event_fields)
- Thread-safe queue operations (_append_to_queue, _pop_queue_copy, _get_queue_length)
- Staleness checks (_is_stale, _get_staleness_ratio, _get_age_seconds, _get_age_hours)
"""

import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.handler_base import HandlerBase


# =============================================================================
# Test Fixtures
# =============================================================================


class TestHandler(HandlerBase):
    """Concrete handler for testing."""

    __test__ = False

    def __init__(self):
        super().__init__(name="test_handler", cycle_interval=60.0)

    async def _run_cycle(self) -> None:
        pass


@pytest.fixture
def handler():
    """Create a test handler instance."""
    TestHandler.reset_instance()
    return TestHandler()


# =============================================================================
# Event Payload Normalization Tests
# =============================================================================


class TestNormalizeEventPayload:
    """Tests for _normalize_event_payload method."""

    def test_extracts_payload_attribute(self, handler):
        """Test extraction from .payload attribute."""
        event = MagicMock()
        event.payload = {"key": "value", "count": 42}
        result = handler._normalize_event_payload(event)
        assert result == {"key": "value", "count": 42}

    def test_extracts_data_attribute(self, handler):
        """Test extraction from .data attribute."""
        event = MagicMock()
        event.payload = None  # payload not a dict
        event.data = {"data_key": "data_value"}
        result = handler._normalize_event_payload(event)
        assert result == {"data_key": "data_value"}

    def test_extracts_metadata_attribute(self, handler):
        """Test extraction from .metadata attribute."""
        event = MagicMock()
        event.payload = None
        event.data = None
        event.metadata = {"meta_key": "meta_value"}
        result = handler._normalize_event_payload(event)
        assert result == {"meta_key": "meta_value"}

    def test_handles_plain_dict(self, handler):
        """Test handling of plain dictionary input."""
        event = {"config_key": "hex8_2p", "model_path": "/path/to/model.pth"}
        result = handler._normalize_event_payload(event)
        assert result == event

    def test_handles_dataclass(self, handler):
        """Test handling of dataclass instances."""

        @dataclass
        class TestEvent:
            config_key: str
            elo: int

        event = TestEvent(config_key="square8_2p", elo=1500)
        result = handler._normalize_event_payload(event)
        assert result == {"config_key": "square8_2p", "elo": 1500}

    def test_returns_empty_dict_for_unknown_type(self, handler):
        """Test fallback to empty dict for unsupported types."""
        result = handler._normalize_event_payload("just a string")
        assert result == {}
        result = handler._normalize_event_payload(12345)
        assert result == {}
        result = handler._normalize_event_payload(None)
        assert result == {}

    def test_prefers_payload_over_data(self, handler):
        """Test that .payload takes precedence over .data."""
        event = MagicMock()
        event.payload = {"source": "payload"}
        event.data = {"source": "data"}
        result = handler._normalize_event_payload(event)
        assert result["source"] == "payload"


class TestExtractEventFields:
    """Tests for _extract_event_fields method."""

    def test_extracts_existing_fields(self, handler):
        """Test extraction of fields that exist in payload."""
        event = {"config_key": "hex8_2p", "elo": 1500, "model_path": "/models/test.pth"}
        result = handler._extract_event_fields(event, ["config_key", "elo"])
        assert result == {"config_key": "hex8_2p", "elo": 1500}

    def test_applies_defaults_for_missing_fields(self, handler):
        """Test that defaults are applied for missing fields."""
        event = {"config_key": "hex8_2p"}
        result = handler._extract_event_fields(
            event,
            ["config_key", "elo", "win_rate"],
            defaults={"elo": 1200, "win_rate": 0.5},
        )
        assert result == {"config_key": "hex8_2p", "elo": 1200, "win_rate": 0.5}

    def test_returns_none_for_missing_without_default(self, handler):
        """Test that None is returned for fields without defaults."""
        event = {"config_key": "hex8_2p"}
        result = handler._extract_event_fields(event, ["config_key", "missing_field"])
        assert result == {"config_key": "hex8_2p", "missing_field": None}

    def test_handles_empty_field_list(self, handler):
        """Test handling of empty field list."""
        event = {"config_key": "hex8_2p", "elo": 1500}
        result = handler._extract_event_fields(event, [])
        assert result == {}

    def test_works_with_event_objects(self, handler):
        """Test that extraction works with event objects, not just dicts."""
        event = MagicMock()
        event.payload = {"config_key": "square8_4p", "games": 100}
        result = handler._extract_event_fields(event, ["config_key", "games"])
        assert result == {"config_key": "square8_4p", "games": 100}


# =============================================================================
# Thread-Safe Queue Helper Tests
# =============================================================================


class TestAppendToQueue:
    """Tests for _append_to_queue method."""

    def test_appends_without_lock(self, handler):
        """Test basic append without lock."""
        queue = []
        handler._append_to_queue(queue, "item1")
        handler._append_to_queue(queue, "item2")
        assert queue == ["item1", "item2"]

    def test_appends_with_lock(self, handler):
        """Test append with lock."""
        queue = []
        lock = threading.Lock()
        handler._append_to_queue(queue, "item1", lock)
        handler._append_to_queue(queue, "item2", lock)
        assert queue == ["item1", "item2"]

    def test_thread_safety(self, handler):
        """Test that appending is thread-safe with lock."""
        queue = []
        lock = threading.Lock()
        results = []

        def append_items(start):
            for i in range(100):
                handler._append_to_queue(queue, start + i, lock)
                results.append(start + i)

        threads = [
            threading.Thread(target=append_items, args=(0,)),
            threading.Thread(target=append_items, args=(1000,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(queue) == 200


class TestPopQueueCopy:
    """Tests for _pop_queue_copy method."""

    def test_copies_and_clears_by_default(self, handler):
        """Test that queue is copied and cleared by default."""
        queue = [1, 2, 3]
        result = handler._pop_queue_copy(queue)
        assert result == [1, 2, 3]
        assert queue == []

    def test_copies_without_clearing(self, handler):
        """Test copy without clearing when clear=False."""
        queue = [1, 2, 3]
        result = handler._pop_queue_copy(queue, clear=False)
        assert result == [1, 2, 3]
        assert queue == [1, 2, 3]

    def test_with_lock(self, handler):
        """Test operation with lock."""
        queue = ["a", "b", "c"]
        lock = threading.Lock()
        result = handler._pop_queue_copy(queue, lock)
        assert result == ["a", "b", "c"]
        assert queue == []

    def test_returns_independent_copy(self, handler):
        """Test that returned list is independent of original."""
        queue = [1, 2, 3]
        result = handler._pop_queue_copy(queue, clear=False)
        result.append(4)
        assert queue == [1, 2, 3]  # Original unchanged


class TestGetQueueLength:
    """Tests for _get_queue_length method."""

    def test_returns_correct_length(self, handler):
        """Test that correct length is returned."""
        queue = [1, 2, 3, 4, 5]
        assert handler._get_queue_length(queue) == 5

    def test_handles_empty_queue(self, handler):
        """Test handling of empty queue."""
        queue = []
        assert handler._get_queue_length(queue) == 0

    def test_with_lock(self, handler):
        """Test operation with lock."""
        queue = [1, 2, 3]
        lock = threading.Lock()
        assert handler._get_queue_length(queue, lock) == 3


# =============================================================================
# Staleness Check Helper Tests
# =============================================================================


class TestIsStale:
    """Tests for _is_stale method."""

    def test_fresh_timestamp_is_not_stale(self, handler):
        """Test that recent timestamp is not stale."""
        now = time.time()
        assert handler._is_stale(now - 30, 60) is False

    def test_old_timestamp_is_stale(self, handler):
        """Test that old timestamp is stale."""
        now = time.time()
        assert handler._is_stale(now - 120, 60) is True

    def test_edge_case_at_threshold(self, handler):
        """Test behavior exactly at threshold."""
        now = time.time()
        # Slightly over threshold
        assert handler._is_stale(now - 60.1, 60) is True
        # Slightly under threshold
        assert handler._is_stale(now - 59.9, 60) is False


class TestGetStalenessRatio:
    """Tests for _get_staleness_ratio method."""

    def test_ratio_below_one_is_fresh(self, handler):
        """Test that ratio < 1.0 indicates fresh."""
        now = time.time()
        ratio = handler._get_staleness_ratio(now - 30, 60)
        assert 0.4 < ratio < 0.6  # Around 0.5

    def test_ratio_above_one_is_stale(self, handler):
        """Test that ratio > 1.0 indicates stale."""
        now = time.time()
        ratio = handler._get_staleness_ratio(now - 120, 60)
        assert ratio > 1.5  # Around 2.0

    def test_handles_zero_threshold(self, handler):
        """Test handling of zero threshold (edge case)."""
        now = time.time()
        ratio = handler._get_staleness_ratio(now - 30, 0)
        assert ratio == 0.0

    def test_handles_negative_age(self, handler):
        """Test handling of future timestamp (should return 0)."""
        future = time.time() + 60
        ratio = handler._get_staleness_ratio(future, 60)
        assert ratio == 0.0


class TestGetAgeSeconds:
    """Tests for _get_age_seconds method."""

    def test_returns_correct_age(self, handler):
        """Test that correct age in seconds is returned."""
        now = time.time()
        age = handler._get_age_seconds(now - 45)
        assert 44 < age < 46  # Allow small timing variance


class TestGetAgeHours:
    """Tests for _get_age_hours method."""

    def test_returns_correct_age_in_hours(self, handler):
        """Test that correct age in hours is returned."""
        now = time.time()
        one_hour_ago = now - 3600
        age = handler._get_age_hours(one_hour_ago)
        assert 0.99 < age < 1.01

    def test_handles_fractional_hours(self, handler):
        """Test handling of partial hours."""
        now = time.time()
        thirty_mins_ago = now - 1800
        age = handler._get_age_hours(thirty_mins_ago)
        assert 0.49 < age < 0.51
