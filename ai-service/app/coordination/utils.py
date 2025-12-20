"""Coordination Utilities (December 2025).

This module provides reusable utility classes for coordinators:
- BoundedHistory: Fixed-size history collection with statistics
- MetricsAccumulator: Accumulator for numeric metrics with windowed stats
- CallbackRegistry: Type-safe callback registration and invocation

Usage:
    from app.coordination.utils import (
        BoundedHistory,
        MetricsAccumulator,
        CallbackRegistry,
    )

    # Bounded history for tracking events
    history = BoundedHistory[TaskEvent](max_size=100)
    history.append(event)
    recent = history.get_recent(10)

    # Metrics accumulator for numeric values
    metrics = MetricsAccumulator(window_size=50)
    metrics.add(0.95)
    print(f"Mean: {metrics.mean}, Std: {metrics.std_dev}")

    # Callback registry for event handlers
    callbacks = CallbackRegistry[str]()
    callbacks.register(my_handler)
    await callbacks.invoke_all("event_data")
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
V = TypeVar("V")


# =============================================================================
# BoundedHistory - Fixed-size history collection
# =============================================================================


@dataclass
class HistoryEntry(Generic[T]):
    """Entry in bounded history with timestamp."""

    value: T
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class BoundedHistory(Generic[T]):
    """Fixed-size history collection with statistics and filtering.

    Maintains a bounded collection of items, automatically evicting oldest
    items when capacity is reached. Provides convenient access patterns
    and statistical operations.

    Type Parameters:
        T: Type of items stored in history

    Example:
        history = BoundedHistory[float](max_size=100)
        history.append(0.95)
        history.append(0.92)
        history.append(0.88)

        print(f"Recent 5: {history.get_recent(5)}")
        print(f"Count: {len(history)}")
        print(f"Oldest: {history.oldest}")
        print(f"Newest: {history.newest}")
    """

    def __init__(
        self,
        max_size: int = 100,
        track_timestamps: bool = True,
    ):
        """Initialize BoundedHistory.

        Args:
            max_size: Maximum number of items to retain
            track_timestamps: Whether to track timestamps for each entry
        """
        self.max_size = max_size
        self.track_timestamps = track_timestamps
        self._entries: deque[HistoryEntry[T]] = deque(maxlen=max_size)
        self._total_added: int = 0

    def append(self, value: T, **metadata) -> None:
        """Add an item to history.

        Args:
            value: Item to add
            **metadata: Optional metadata to attach
        """
        entry = HistoryEntry(
            value=value,
            timestamp=time.time() if self.track_timestamps else 0.0,
            metadata=metadata,
        )
        self._entries.append(entry)
        self._total_added += 1

    def extend(self, values: list[T]) -> None:
        """Add multiple items to history.

        Args:
            values: Items to add
        """
        for value in values:
            self.append(value)

    def clear(self) -> None:
        """Clear all history."""
        self._entries.clear()

    def get_recent(self, n: int) -> list[T]:
        """Get the n most recent items.

        Args:
            n: Number of items to return

        Returns:
            List of most recent items (newest last)
        """
        entries = list(self._entries)[-n:]
        return [e.value for e in entries]

    def get_oldest(self, n: int) -> list[T]:
        """Get the n oldest items.

        Args:
            n: Number of items to return

        Returns:
            List of oldest items (oldest first)
        """
        entries = list(self._entries)[:n]
        return [e.value for e in entries]

    def get_all(self) -> list[T]:
        """Get all items as a list.

        Returns:
            All items (oldest first)
        """
        return [e.value for e in self._entries]

    def get_entries(self) -> list[HistoryEntry[T]]:
        """Get all entries with timestamps and metadata.

        Returns:
            All entries (oldest first)
        """
        return list(self._entries)

    def filter(self, predicate: Callable[[T], bool]) -> list[T]:
        """Filter items by predicate.

        Args:
            predicate: Function returning True for items to include

        Returns:
            Filtered list of items
        """
        return [e.value for e in self._entries if predicate(e.value)]

    def filter_by_time(
        self,
        since: float | None = None,
        until: float | None = None,
    ) -> list[T]:
        """Filter items by timestamp range.

        Args:
            since: Include items after this timestamp
            until: Include items before this timestamp

        Returns:
            Filtered list of items
        """
        result = []
        for entry in self._entries:
            if since is not None and entry.timestamp < since:
                continue
            if until is not None and entry.timestamp > until:
                continue
            result.append(entry.value)
        return result

    def count_matching(self, predicate: Callable[[T], bool]) -> int:
        """Count items matching predicate.

        Args:
            predicate: Function returning True for items to count

        Returns:
            Number of matching items
        """
        return sum(1 for e in self._entries if predicate(e.value))

    @property
    def oldest(self) -> T | None:
        """Get oldest item, or None if empty."""
        if self._entries:
            return self._entries[0].value
        return None

    @property
    def newest(self) -> T | None:
        """Get newest item, or None if empty."""
        if self._entries:
            return self._entries[-1].value
        return None

    @property
    def oldest_timestamp(self) -> float | None:
        """Get timestamp of oldest item."""
        if self._entries:
            return self._entries[0].timestamp
        return None

    @property
    def newest_timestamp(self) -> float | None:
        """Get timestamp of newest item."""
        if self._entries:
            return self._entries[-1].timestamp
        return None

    @property
    def total_added(self) -> int:
        """Total number of items ever added (including evicted)."""
        return self._total_added

    @property
    def is_full(self) -> bool:
        """Whether history is at capacity."""
        return len(self._entries) >= self.max_size

    def __len__(self) -> int:
        """Number of items currently in history."""
        return len(self._entries)

    def __iter__(self) -> Iterator[T]:
        """Iterate over items (oldest first)."""
        for entry in self._entries:
            yield entry.value

    def __bool__(self) -> bool:
        """True if history is non-empty."""
        return len(self._entries) > 0

    def __contains__(self, item: T) -> bool:
        """Check if item is in history."""
        return any(e.value == item for e in self._entries)


# =============================================================================
# MetricsAccumulator - Numeric metrics with windowed statistics
# =============================================================================


@dataclass
class MetricsSnapshot:
    """Snapshot of accumulated metrics statistics."""

    count: int
    total: float
    mean: float
    min_value: float
    max_value: float
    std_dev: float
    recent_mean: float  # Mean of recent window
    trend: float  # Change rate
    timestamp: float = field(default_factory=time.time)


class MetricsAccumulator:
    """Accumulator for numeric metrics with windowed statistics.

    Tracks numeric values over time, maintaining both all-time and
    windowed statistics. Useful for tracking loss, accuracy, throughput,
    and other numeric metrics.

    Example:
        metrics = MetricsAccumulator(window_size=50)
        for loss in training_losses:
            metrics.add(loss)

        print(f"Mean loss: {metrics.mean:.4f}")
        print(f"Recent trend: {metrics.trend:.6f}")
        print(f"Improving: {metrics.is_improving}")
    """

    def __init__(
        self,
        window_size: int = 100,
        name: str = "metric",
        higher_is_better: bool = False,
    ):
        """Initialize MetricsAccumulator.

        Args:
            window_size: Size of sliding window for recent stats
            name: Name of the metric (for logging)
            higher_is_better: If True, increasing values are improvements
        """
        self.window_size = window_size
        self.name = name
        self.higher_is_better = higher_is_better

        self._values: deque[float] = deque(maxlen=window_size)
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._total: float = 0.0
        self._count: int = 0
        self._min: float | None = None
        self._max: float | None = None
        self._best: float | None = None

    def add(self, value: float, timestamp: float | None = None) -> None:
        """Add a value to the accumulator.

        Args:
            value: Numeric value to add
            timestamp: Optional timestamp (defaults to now)
        """
        self._values.append(value)
        self._timestamps.append(timestamp or time.time())
        self._total += value
        self._count += 1

        # Update min/max
        if self._min is None or value < self._min:
            self._min = value
        if self._max is None or value > self._max:
            self._max = value

        # Update best
        if self._best is None or (self.higher_is_better and value > self._best) or (not self.higher_is_better and value < self._best):
            self._best = value

    def add_batch(self, values: list[float]) -> None:
        """Add multiple values at once.

        Args:
            values: List of values to add
        """
        for value in values:
            self.add(value)

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self._values.clear()
        self._timestamps.clear()
        self._total = 0.0
        self._count = 0
        self._min = None
        self._max = None
        self._best = None

    def reset_window(self) -> None:
        """Reset only the sliding window, keeping all-time stats."""
        self._values.clear()
        self._timestamps.clear()

    @property
    def count(self) -> int:
        """Total number of values added."""
        return self._count

    @property
    def window_count(self) -> int:
        """Number of values in current window."""
        return len(self._values)

    @property
    def total(self) -> float:
        """Sum of all values."""
        return self._total

    @property
    def mean(self) -> float:
        """Mean of all values."""
        if self._count == 0:
            return 0.0
        return self._total / self._count

    @property
    def window_mean(self) -> float:
        """Mean of values in current window."""
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def min_value(self) -> float | None:
        """Minimum value seen."""
        return self._min

    @property
    def max_value(self) -> float | None:
        """Maximum value seen."""
        return self._max

    @property
    def best_value(self) -> float | None:
        """Best value seen (based on higher_is_better)."""
        return self._best

    @property
    def std_dev(self) -> float:
        """Standard deviation of window values."""
        if len(self._values) < 2:
            return 0.0
        return statistics.stdev(self._values)

    @property
    def variance(self) -> float:
        """Variance of window values."""
        if len(self._values) < 2:
            return 0.0
        return statistics.variance(self._values)

    @property
    def trend(self) -> float:
        """Trend (rate of change) over the window.

        Returns:
            Positive if increasing, negative if decreasing
        """
        if len(self._values) < 2:
            return 0.0

        values = list(self._values)
        n = len(values)
        x = list(range(n))
        y = values

        # Linear regression slope
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_xx = sum(xi ** 2 for xi in x)

        denominator = n * sum_xx - sum_x ** 2
        if denominator == 0:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denominator

    @property
    def is_improving(self) -> bool:
        """Whether metric is improving based on trend."""
        if self.higher_is_better:
            return self.trend > 0
        return self.trend < 0

    @property
    def current(self) -> float | None:
        """Most recent value."""
        if self._values:
            return self._values[-1]
        return None

    def get_recent(self, n: int) -> list[float]:
        """Get n most recent values."""
        return list(self._values)[-n:]

    def get_snapshot(self) -> MetricsSnapshot:
        """Get a snapshot of current statistics."""
        return MetricsSnapshot(
            count=self._count,
            total=self._total,
            mean=self.mean,
            min_value=self._min or 0.0,
            max_value=self._max or 0.0,
            std_dev=self.std_dev,
            recent_mean=self.window_mean,
            trend=self.trend,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "count": self._count,
            "window_count": len(self._values),
            "mean": round(self.mean, 6),
            "window_mean": round(self.window_mean, 6),
            "min": self._min,
            "max": self._max,
            "best": self._best,
            "std_dev": round(self.std_dev, 6),
            "trend": round(self.trend, 8),
            "is_improving": self.is_improving,
            "current": self.current,
        }


# =============================================================================
# CallbackRegistry - Type-safe callback management
# =============================================================================


class CallbackRegistry(Generic[T]):
    """Type-safe registry for callbacks with async support.

    Manages registration and invocation of callbacks, supporting both
    sync and async handlers. Provides error isolation so one failing
    handler doesn't affect others.

    Type Parameters:
        T: Type of data passed to callbacks

    Example:
        registry = CallbackRegistry[str]()

        def sync_handler(data: str) -> None:
            print(f"Received: {data}")

        async def async_handler(data: str) -> None:
            await some_async_operation(data)

        registry.register(sync_handler)
        registry.register(async_handler)

        # Invoke all handlers
        await registry.invoke_all("event_data")
    """

    def __init__(self, name: str = "callbacks"):
        """Initialize CallbackRegistry.

        Args:
            name: Name for logging
        """
        self.name = name
        self._callbacks: list[Callable[[T], Any]] = []
        self._invocation_count: int = 0
        self._error_count: int = 0

    def register(self, callback: Callable[[T], Any]) -> None:
        """Register a callback.

        Args:
            callback: Function to call (sync or async)
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            logger.debug(f"[{self.name}] Registered callback: {callback.__name__}")

    def unregister(self, callback: Callable[[T], Any]) -> bool:
        """Unregister a callback.

        Args:
            callback: Function to remove

        Returns:
            True if callback was found and removed
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug(f"[{self.name}] Unregistered callback: {callback.__name__}")
            return True
        return False

    def clear(self) -> None:
        """Clear all registered callbacks."""
        self._callbacks.clear()

    async def invoke_all(self, data: T) -> list[Exception]:
        """Invoke all callbacks with data.

        Handles both sync and async callbacks. Errors are caught and
        collected rather than propagated.

        Args:
            data: Data to pass to callbacks

        Returns:
            List of exceptions from failed callbacks
        """
        errors: list[Exception] = []

        for callback in self._callbacks:
            self._invocation_count += 1
            try:
                result = callback(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self._error_count += 1
                errors.append(e)
                logger.warning(
                    f"[{self.name}] Callback {callback.__name__} failed: {e}"
                )

        return errors

    def invoke_all_sync(self, data: T) -> list[Exception]:
        """Invoke all sync callbacks (ignores async ones).

        Args:
            data: Data to pass to callbacks

        Returns:
            List of exceptions from failed callbacks
        """
        errors: list[Exception] = []

        for callback in self._callbacks:
            if asyncio.iscoroutinefunction(callback):
                continue

            self._invocation_count += 1
            try:
                callback(data)
            except Exception as e:
                self._error_count += 1
                errors.append(e)
                logger.warning(
                    f"[{self.name}] Callback {callback.__name__} failed: {e}"
                )

        return errors

    @property
    def count(self) -> int:
        """Number of registered callbacks."""
        return len(self._callbacks)

    @property
    def invocation_count(self) -> int:
        """Total number of callback invocations."""
        return self._invocation_count

    @property
    def error_count(self) -> int:
        """Total number of callback errors."""
        return self._error_count

    @property
    def error_rate(self) -> float:
        """Error rate (errors / invocations)."""
        if self._invocation_count == 0:
            return 0.0
        return self._error_count / self._invocation_count

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        return {
            "name": self.name,
            "callback_count": len(self._callbacks),
            "invocation_count": self._invocation_count,
            "error_count": self._error_count,
            "error_rate": round(self.error_rate, 4),
        }


__all__ = [
    "BoundedHistory",
    "CallbackRegistry",
    "HistoryEntry",
    "MetricsAccumulator",
    "MetricsSnapshot",
]
