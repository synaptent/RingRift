"""Unit tests for app.coordination.utils (December 27, 2025).

Tests for coordination utility classes:
- BoundedHistory: Fixed-size history collection
- MetricsAccumulator: Numeric metrics with windowed statistics
- CallbackRegistry: Type-safe callback management
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock

from app.coordination.utils import (
    BoundedHistory,
    CallbackRegistry,
    HistoryEntry,
    MetricsAccumulator,
    MetricsSnapshot,
)


# =============================================================================
# BoundedHistory Tests
# =============================================================================


class TestBoundedHistory:
    """Tests for BoundedHistory class."""

    def test_initialization(self):
        """Test default initialization."""
        history = BoundedHistory[int](max_size=10)
        assert history.max_size == 10
        assert history.track_timestamps is True
        assert len(history) == 0
        assert history.oldest is None
        assert history.newest is None

    def test_append_single_item(self):
        """Test appending a single item."""
        history = BoundedHistory[str](max_size=5)
        history.append("item1")

        assert len(history) == 1
        assert history.oldest == "item1"
        assert history.newest == "item1"
        assert history.total_added == 1

    def test_append_multiple_items(self):
        """Test appending multiple items."""
        history = BoundedHistory[int](max_size=10)
        for i in range(5):
            history.append(i)

        assert len(history) == 5
        assert history.oldest == 0
        assert history.newest == 4
        assert history.total_added == 5

    def test_bounded_eviction(self):
        """Test that old items are evicted when capacity is reached."""
        history = BoundedHistory[int](max_size=3)
        for i in range(5):
            history.append(i)

        assert len(history) == 3
        assert history.oldest == 2
        assert history.newest == 4
        assert history.total_added == 5
        assert history.is_full is True

    def test_extend(self):
        """Test extending with multiple items."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3, 4, 5])

        assert len(history) == 5
        assert history.get_all() == [1, 2, 3, 4, 5]

    def test_clear(self):
        """Test clearing history."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3])
        history.clear()

        assert len(history) == 0
        assert history.oldest is None
        assert history.newest is None

    def test_get_recent(self):
        """Test getting recent items."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3, 4, 5])

        assert history.get_recent(3) == [3, 4, 5]
        assert history.get_recent(10) == [1, 2, 3, 4, 5]
        # Note: get_recent(0) returns all items due to Python slice behavior [-0:]
        assert history.get_recent(0) == [1, 2, 3, 4, 5]

    def test_get_oldest(self):
        """Test getting oldest items."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3, 4, 5])

        assert history.get_oldest(3) == [1, 2, 3]
        assert history.get_oldest(10) == [1, 2, 3, 4, 5]

    def test_filter(self):
        """Test filtering items by predicate."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3, 4, 5, 6])

        evens = history.filter(lambda x: x % 2 == 0)
        assert evens == [2, 4, 6]

    def test_count_matching(self):
        """Test counting matching items."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3, 4, 5, 6])

        count = history.count_matching(lambda x: x > 3)
        assert count == 3

    def test_filter_by_time(self):
        """Test filtering by timestamp."""
        history = BoundedHistory[int](max_size=10)

        # Add items with controlled timestamps
        base_time = time.time()
        history.append(1)
        time.sleep(0.01)
        history.append(2)
        time.sleep(0.01)
        history.append(3)

        # Get all
        all_items = history.filter_by_time()
        assert len(all_items) == 3

        # Filter by since
        entries = history.get_entries()
        mid_time = entries[1].timestamp
        recent = history.filter_by_time(since=mid_time)
        assert len(recent) >= 1

    def test_metadata(self):
        """Test appending with metadata."""
        history = BoundedHistory[str](max_size=10)
        history.append("item1", source="test", priority=1)

        entries = history.get_entries()
        assert len(entries) == 1
        assert entries[0].value == "item1"
        assert entries[0].metadata["source"] == "test"
        assert entries[0].metadata["priority"] == 1

    def test_timestamps(self):
        """Test timestamp tracking."""
        history = BoundedHistory[int](max_size=10)
        history.append(1)

        assert history.oldest_timestamp is not None
        assert history.newest_timestamp is not None
        assert history.oldest_timestamp == history.newest_timestamp

    def test_no_timestamps(self):
        """Test with timestamp tracking disabled."""
        history = BoundedHistory[int](max_size=10, track_timestamps=False)
        history.append(1)

        entries = history.get_entries()
        assert entries[0].timestamp == 0.0

    def test_iteration(self):
        """Test iteration over history."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3])

        items = list(history)
        assert items == [1, 2, 3]

    def test_contains(self):
        """Test membership checking."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3])

        assert 2 in history
        assert 5 not in history

    def test_bool(self):
        """Test boolean conversion."""
        history = BoundedHistory[int](max_size=10)
        assert not history

        history.append(1)
        assert history

    def test_generic_type(self):
        """Test with custom type."""
        from dataclasses import dataclass

        @dataclass
        class Event:
            name: str
            value: int

        history = BoundedHistory[Event](max_size=5)
        history.append(Event("start", 1))
        history.append(Event("end", 2))

        assert len(history) == 2
        assert history.newest.name == "end"


# =============================================================================
# MetricsAccumulator Tests
# =============================================================================


class TestMetricsAccumulator:
    """Tests for MetricsAccumulator class."""

    def test_initialization(self):
        """Test default initialization."""
        metrics = MetricsAccumulator(window_size=50)

        assert metrics.window_size == 50
        assert metrics.count == 0
        assert metrics.mean == 0.0
        assert metrics.current is None

    def test_add_single_value(self):
        """Test adding a single value."""
        metrics = MetricsAccumulator()
        metrics.add(0.5)

        assert metrics.count == 1
        assert metrics.window_count == 1
        assert metrics.mean == 0.5
        assert metrics.current == 0.5

    def test_add_multiple_values(self):
        """Test adding multiple values."""
        metrics = MetricsAccumulator()
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            metrics.add(v)

        assert metrics.count == 5
        assert metrics.mean == 0.3
        assert metrics.min_value == 0.1
        assert metrics.max_value == 0.5

    def test_add_batch(self):
        """Test adding batch of values."""
        metrics = MetricsAccumulator()
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])

        assert metrics.count == 5
        assert metrics.mean == 3.0

    def test_window_statistics(self):
        """Test windowed statistics."""
        metrics = MetricsAccumulator(window_size=3)
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])

        # Window should only have last 3 values
        assert metrics.window_count == 3
        assert metrics.window_mean == 4.0  # (3+4+5)/3

    def test_total_accumulation(self):
        """Test total is accumulated correctly."""
        metrics = MetricsAccumulator()
        metrics.add_batch([10.0, 20.0, 30.0])

        assert metrics.total == 60.0

    def test_std_dev(self):
        """Test standard deviation calculation."""
        metrics = MetricsAccumulator()
        metrics.add_batch([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

        # Std dev should be approximately 2.0-2.2 (sample std dev is ~2.138)
        assert 1.9 < metrics.std_dev < 2.2

    def test_variance(self):
        """Test variance calculation."""
        metrics = MetricsAccumulator()
        metrics.add_batch([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

        assert metrics.variance > 0

    def test_trend_increasing(self):
        """Test trend detection for increasing values."""
        metrics = MetricsAccumulator()
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])

        assert metrics.trend > 0

    def test_trend_decreasing(self):
        """Test trend detection for decreasing values."""
        metrics = MetricsAccumulator()
        metrics.add_batch([5.0, 4.0, 3.0, 2.0, 1.0])

        assert metrics.trend < 0

    def test_trend_flat(self):
        """Test trend detection for flat values."""
        metrics = MetricsAccumulator()
        metrics.add_batch([3.0, 3.0, 3.0, 3.0, 3.0])

        assert metrics.trend == 0.0

    def test_is_improving_higher_is_better(self):
        """Test is_improving when higher is better."""
        metrics = MetricsAccumulator(higher_is_better=True)
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])

        assert metrics.is_improving is True

    def test_is_improving_lower_is_better(self):
        """Test is_improving when lower is better."""
        metrics = MetricsAccumulator(higher_is_better=False)
        metrics.add_batch([5.0, 4.0, 3.0, 2.0, 1.0])

        assert metrics.is_improving is True

    def test_best_value_higher_is_better(self):
        """Test best value tracking when higher is better."""
        metrics = MetricsAccumulator(higher_is_better=True)
        metrics.add_batch([3.0, 5.0, 2.0, 4.0, 1.0])

        assert metrics.best_value == 5.0

    def test_best_value_lower_is_better(self):
        """Test best value tracking when lower is better."""
        metrics = MetricsAccumulator(higher_is_better=False)
        metrics.add_batch([3.0, 5.0, 2.0, 4.0, 1.0])

        assert metrics.best_value == 1.0

    def test_reset(self):
        """Test full reset."""
        metrics = MetricsAccumulator()
        metrics.add_batch([1.0, 2.0, 3.0])
        metrics.reset()

        assert metrics.count == 0
        assert metrics.window_count == 0
        assert metrics.min_value is None
        assert metrics.max_value is None

    def test_reset_window(self):
        """Test window-only reset."""
        metrics = MetricsAccumulator()
        metrics.add_batch([1.0, 2.0, 3.0])
        metrics.reset_window()

        # All-time stats preserved
        assert metrics.count == 3
        assert metrics.total == 6.0
        # Window cleared
        assert metrics.window_count == 0

    def test_get_recent(self):
        """Test getting recent values."""
        metrics = MetricsAccumulator()
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])

        recent = metrics.get_recent(3)
        assert recent == [3.0, 4.0, 5.0]

    def test_get_snapshot(self):
        """Test getting metrics snapshot."""
        metrics = MetricsAccumulator()
        metrics.add_batch([1.0, 2.0, 3.0])

        snapshot = metrics.get_snapshot()
        assert isinstance(snapshot, MetricsSnapshot)
        assert snapshot.count == 3
        assert snapshot.mean == 2.0

    def test_to_dict(self):
        """Test dictionary serialization."""
        metrics = MetricsAccumulator(name="loss")
        metrics.add_batch([0.5, 0.4, 0.3])

        d = metrics.to_dict()
        assert d["name"] == "loss"
        assert d["count"] == 3
        assert "mean" in d
        assert "trend" in d


# =============================================================================
# CallbackRegistry Tests
# =============================================================================


class TestCallbackRegistry:
    """Tests for CallbackRegistry class."""

    def test_initialization(self):
        """Test default initialization."""
        registry = CallbackRegistry[str](name="test")

        assert registry.name == "test"
        assert registry.count == 0
        assert registry.invocation_count == 0
        assert registry.error_count == 0

    def test_register_callback(self):
        """Test registering a callback."""
        registry = CallbackRegistry[str]()

        def handler(data: str) -> None:
            pass

        registry.register(handler)
        assert registry.count == 1

    def test_register_duplicate(self):
        """Test that duplicate registration is prevented."""
        registry = CallbackRegistry[str]()

        def handler(data: str) -> None:
            pass

        registry.register(handler)
        registry.register(handler)  # Should not add again
        assert registry.count == 1

    def test_unregister_callback(self):
        """Test unregistering a callback."""
        registry = CallbackRegistry[str]()

        def handler(data: str) -> None:
            pass

        registry.register(handler)
        result = registry.unregister(handler)

        assert result is True
        assert registry.count == 0

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent callback."""
        registry = CallbackRegistry[str]()

        def handler(data: str) -> None:
            pass

        result = registry.unregister(handler)
        assert result is False

    def test_clear(self):
        """Test clearing all callbacks."""
        registry = CallbackRegistry[str]()

        registry.register(lambda x: None)
        registry.register(lambda x: None)
        registry.clear()

        assert registry.count == 0

    @pytest.mark.asyncio
    async def test_invoke_all_sync_callbacks(self):
        """Test invoking sync callbacks."""
        registry = CallbackRegistry[str]()
        results = []

        def handler1(data: str) -> None:
            results.append(f"h1:{data}")

        def handler2(data: str) -> None:
            results.append(f"h2:{data}")

        registry.register(handler1)
        registry.register(handler2)

        errors = await registry.invoke_all("test")

        assert len(errors) == 0
        assert results == ["h1:test", "h2:test"]
        assert registry.invocation_count == 2

    @pytest.mark.asyncio
    async def test_invoke_all_async_callbacks(self):
        """Test invoking async callbacks."""
        registry = CallbackRegistry[str]()
        results = []

        async def async_handler(data: str) -> None:
            await asyncio.sleep(0.001)
            results.append(f"async:{data}")

        registry.register(async_handler)
        errors = await registry.invoke_all("test")

        assert len(errors) == 0
        assert results == ["async:test"]

    @pytest.mark.asyncio
    async def test_invoke_all_mixed_callbacks(self):
        """Test invoking mixed sync and async callbacks."""
        registry = CallbackRegistry[str]()
        results = []

        def sync_handler(data: str) -> None:
            results.append(f"sync:{data}")

        async def async_handler(data: str) -> None:
            results.append(f"async:{data}")

        registry.register(sync_handler)
        registry.register(async_handler)

        errors = await registry.invoke_all("test")

        assert len(errors) == 0
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_invoke_all_error_isolation(self):
        """Test that callback errors are isolated."""
        registry = CallbackRegistry[str]()
        results = []

        def good_handler(data: str) -> None:
            results.append("good")

        def bad_handler(data: str) -> None:
            raise ValueError("intentional error")

        registry.register(bad_handler)
        registry.register(good_handler)

        errors = await registry.invoke_all("test")

        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        assert "good" in results
        assert registry.error_count == 1

    def test_invoke_all_sync(self):
        """Test sync-only invocation."""
        registry = CallbackRegistry[str]()
        results = []

        def sync_handler(data: str) -> None:
            results.append(f"sync:{data}")

        async def async_handler(data: str) -> None:
            results.append(f"async:{data}")

        registry.register(sync_handler)
        registry.register(async_handler)

        errors = registry.invoke_all_sync("test")

        assert len(errors) == 0
        assert results == ["sync:test"]  # Async handler skipped

    def test_error_rate(self):
        """Test error rate calculation."""
        registry = CallbackRegistry[str]()

        assert registry.error_rate == 0.0

        # Manually set counts for testing
        registry._invocation_count = 10
        registry._error_count = 2

        assert registry.error_rate == 0.2

    def test_get_stats(self):
        """Test getting registry stats."""
        registry = CallbackRegistry[str](name="events")
        registry.register(lambda x: None)
        registry._invocation_count = 100
        registry._error_count = 5

        stats = registry.get_stats()

        assert stats["name"] == "events"
        assert stats["callback_count"] == 1
        assert stats["invocation_count"] == 100
        assert stats["error_count"] == 5
        assert stats["error_rate"] == 0.05


# =============================================================================
# HistoryEntry Tests
# =============================================================================


class TestHistoryEntry:
    """Tests for HistoryEntry dataclass."""

    def test_default_timestamp(self):
        """Test that timestamp defaults to current time."""
        before = time.time()
        entry = HistoryEntry(value="test")
        after = time.time()

        assert before <= entry.timestamp <= after

    def test_custom_timestamp(self):
        """Test custom timestamp."""
        entry = HistoryEntry(value="test", timestamp=1000.0)
        assert entry.timestamp == 1000.0

    def test_metadata(self):
        """Test metadata storage."""
        entry = HistoryEntry(
            value="test",
            metadata={"key": "value", "count": 42}
        )

        assert entry.metadata["key"] == "value"
        assert entry.metadata["count"] == 42


# =============================================================================
# MetricsSnapshot Tests
# =============================================================================


class TestMetricsSnapshot:
    """Tests for MetricsSnapshot dataclass."""

    def test_creation(self):
        """Test snapshot creation."""
        snapshot = MetricsSnapshot(
            count=100,
            total=500.0,
            mean=5.0,
            min_value=1.0,
            max_value=10.0,
            std_dev=2.5,
            recent_mean=5.5,
            trend=0.1,
        )

        assert snapshot.count == 100
        assert snapshot.mean == 5.0
        assert snapshot.trend == 0.1

    def test_default_timestamp(self):
        """Test that timestamp defaults to current time."""
        before = time.time()
        snapshot = MetricsSnapshot(
            count=0, total=0, mean=0, min_value=0,
            max_value=0, std_dev=0, recent_mean=0, trend=0
        )
        after = time.time()

        assert before <= snapshot.timestamp <= after
