"""Metrics utilities for training and monitoring scripts.

Provides common patterns for:
- Timing statistics (min/max/avg/total with context manager)
- Rate calculations (throughput, games/sec, moves/sec)
- Counters with computed rates (wins/losses/draws)
- Progress tracking with ETA estimation
- Running statistics (online mean/variance)

Usage:
    from scripts.lib.metrics import (
        TimingStats,
        RateCalculator,
        Counter,
        WinLossCounter,
        ProgressTracker,
        RunningStats,
    )

    # Time a code section
    stats = TimingStats()
    with stats.time():
        do_work()
    print(f"Avg time: {stats.avg_time_ms:.2f}ms")

    # Calculate throughput
    rate = RateCalculator()
    rate.record(games=100, elapsed=10.0)
    print(f"Rate: {rate.rate_per_second:.1f} games/sec")

    # Track wins/losses
    counter = WinLossCounter()
    counter.record_win()
    counter.record_loss()
    print(f"Win rate: {counter.win_rate:.1%}")

    # Track progress
    progress = ProgressTracker(total=1000)
    progress.update(100)
    print(f"Progress: {progress.percentage:.1f}% ETA: {progress.eta_str}")
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from collections.abc import Iterator

from scripts.lib.datetime_utils import format_elapsed_time


# =============================================================================
# Timing Statistics
# =============================================================================


@dataclass
class TimingStats:
    """Accumulated timing statistics for a code section.

    Provides min/max/average timing with a context manager for easy measurement.

    Example:
        stats = TimingStats("database_query")
        with stats.time():
            result = db.query(...)

        print(f"Query stats: {stats}")
        print(f"Avg: {stats.avg_time_ms:.2f}ms over {stats.count} calls")
    """

    name: str = ""
    total_time: float = 0.0
    count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0
    _start_time: float = field(default=0.0, repr=False)

    def record(self, elapsed: float) -> None:
        """Record an elapsed time measurement.

        Args:
            elapsed: Time in seconds
        """
        self.total_time += elapsed
        self.count += 1
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)

    @contextmanager
    def time(self) -> Iterator[None]:
        """Context manager to time a code section.

        Example:
            with stats.time():
                do_work()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(time.perf_counter() - start)

    @property
    def avg_time(self) -> float:
        """Average time in seconds."""
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def avg_time_ms(self) -> float:
        """Average time in milliseconds."""
        return self.avg_time * 1000

    @property
    def total_time_ms(self) -> float:
        """Total time in milliseconds."""
        return self.total_time * 1000

    @property
    def min_time_ms(self) -> float:
        """Minimum time in milliseconds."""
        if self.min_time == float("inf"):
            return 0.0
        return self.min_time * 1000

    @property
    def max_time_ms(self) -> float:
        """Maximum time in milliseconds."""
        return self.max_time * 1000

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_time = 0.0
        self.count = 0
        self.min_time = float("inf")
        self.max_time = 0.0

    def merge(self, other: "TimingStats") -> None:
        """Merge statistics from another TimingStats.

        Args:
            other: Another TimingStats to merge in
        """
        self.total_time += other.total_time
        self.count += other.count
        self.min_time = min(self.min_time, other.min_time)
        self.max_time = max(self.max_time, other.max_time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "count": self.count,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
        }

    def __str__(self) -> str:
        if self.count == 0:
            return f"{self.name}: no data" if self.name else "no data"

        parts = []
        if self.name:
            parts.append(f"{self.name}:")

        parts.append(f"{self.count} calls")
        parts.append(f"avg={self.avg_time_ms:.2f}ms")
        parts.append(f"total={self.total_time_ms:.1f}ms")

        if self.count > 1:
            parts.append(f"min={self.min_time_ms:.2f}ms")
            parts.append(f"max={self.max_time_ms:.2f}ms")

        return " ".join(parts)


# =============================================================================
# Rate Calculations
# =============================================================================


@dataclass
class RateCalculator:
    """Calculate throughput/rate statistics.

    Example:
        rate = RateCalculator()
        rate.start()
        for batch in batches:
            process(batch)
            rate.record(len(batch))

        print(f"Processed {rate.total_items} at {rate.rate_per_second:.1f}/sec")
        print(f"Projected hourly: {rate.rate_per_hour:,.0f}")
    """

    total_items: int = 0
    total_time: float = 0.0
    _start_time: float | None = field(default=None, repr=False)

    def start(self) -> None:
        """Start the rate timer."""
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the timer and record elapsed time."""
        if self._start_time is not None:
            self.total_time += time.perf_counter() - self._start_time
            self._start_time = None

    def record(self, items: int = 1, elapsed: float | None = None) -> None:
        """Record processed items.

        Args:
            items: Number of items processed
            elapsed: Optional explicit elapsed time (uses running timer if None)
        """
        self.total_items += items
        if elapsed is not None:
            self.total_time += elapsed
        elif self._start_time is not None:
            self.total_time = time.perf_counter() - self._start_time

    @property
    def elapsed(self) -> float:
        """Current elapsed time in seconds."""
        if self._start_time is not None:
            return time.perf_counter() - self._start_time
        return self.total_time

    @property
    def rate_per_second(self) -> float:
        """Items per second."""
        elapsed = self.elapsed
        if elapsed <= 0:
            return 0.0
        return self.total_items / elapsed

    @property
    def rate_per_minute(self) -> float:
        """Items per minute."""
        return self.rate_per_second * 60

    @property
    def rate_per_hour(self) -> float:
        """Items per hour."""
        return self.rate_per_second * 3600

    def format_rate(self, unit: str = "items") -> str:
        """Format rate as human-readable string.

        Args:
            unit: Unit name (e.g., "games", "moves", "samples")
        """
        rate = self.rate_per_second
        if rate >= 1000:
            return f"{rate/1000:.1f}k {unit}/sec"
        elif rate >= 1:
            return f"{rate:.1f} {unit}/sec"
        else:
            return f"{rate*60:.1f} {unit}/min"

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_items = 0
        self.total_time = 0.0
        self._start_time = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_items": self.total_items,
            "total_time": self.total_time,
            "rate_per_second": self.rate_per_second,
            "rate_per_hour": self.rate_per_hour,
        }


# =============================================================================
# Counters
# =============================================================================


@dataclass
class Counter:
    """Simple counter with rate calculation.

    Example:
        counter = Counter("processed_games")
        for game in games:
            process(game)
            counter.increment()

        print(f"Processed {counter.value} games")
    """

    name: str = ""
    value: int = 0

    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value."""
        self.value += amount
        return self.value

    def reset(self) -> None:
        """Reset counter to zero."""
        self.value = 0

    def __str__(self) -> str:
        if self.name:
            return f"{self.name}: {self.value}"
        return str(self.value)


@dataclass
class WinLossCounter:
    """Counter for wins/losses/draws with computed statistics.

    Example:
        counter = WinLossCounter()
        for result in game_results:
            if result == "win":
                counter.record_win()
            elif result == "loss":
                counter.record_loss()
            else:
                counter.record_draw()

        print(f"Win rate: {counter.win_rate:.1%}")
        print(f"Record: {counter}")  # "10-5-2 (W-L-D)"
    """

    wins: int = 0
    losses: int = 0
    draws: int = 0

    def record_win(self, count: int = 1) -> None:
        """Record win(s)."""
        self.wins += count

    def record_loss(self, count: int = 1) -> None:
        """Record loss(es)."""
        self.losses += count

    def record_draw(self, count: int = 1) -> None:
        """Record draw(s)."""
        self.draws += count

    def record_result(self, result: str) -> None:
        """Record a result by string.

        Args:
            result: "win", "loss", or "draw"
        """
        result = result.lower()
        if result == "win":
            self.record_win()
        elif result == "loss":
            self.record_loss()
        elif result == "draw":
            self.record_draw()

    @property
    def total_games(self) -> int:
        """Total number of games played."""
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        """Win rate as a fraction (0.0 to 1.0).

        Note: Draws count as 0.5 wins for ELO-style calculations.
        """
        total = self.total_games
        if total == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / total

    @property
    def win_rate_strict(self) -> float:
        """Strict win rate (draws don't count as partial wins)."""
        total = self.total_games
        if total == 0:
            return 0.0
        return self.wins / total

    @property
    def loss_rate(self) -> float:
        """Loss rate as a fraction."""
        total = self.total_games
        if total == 0:
            return 0.0
        return self.losses / total

    @property
    def draw_rate(self) -> float:
        """Draw rate as a fraction."""
        total = self.total_games
        if total == 0:
            return 0.0
        return self.draws / total

    def reset(self) -> None:
        """Reset all counters."""
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def merge(self, other: "WinLossCounter") -> None:
        """Merge another counter into this one."""
        self.wins += other.wins
        self.losses += other.losses
        self.draws += other.draws

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "total_games": self.total_games,
            "win_rate": self.win_rate,
        }

    def __str__(self) -> str:
        return f"{self.wins}-{self.losses}-{self.draws} (W-L-D)"


# =============================================================================
# Progress Tracking
# =============================================================================


@dataclass
class ProgressTracker:
    """Track progress with percentage and ETA calculation.

    Example:
        progress = ProgressTracker(total=1000)
        for item in items:
            process(item)
            progress.update(1)
            if progress.current % 100 == 0:
                print(f"{progress.percentage:.1f}% ETA: {progress.eta_str}")

        print(f"Completed in {progress.elapsed_str}")
    """

    total: int
    current: int = 0
    _start_time: float = field(default_factory=time.perf_counter, repr=False)

    def update(self, amount: int = 1) -> None:
        """Update progress by amount."""
        self.current += amount

    def set(self, value: int) -> None:
        """Set progress to specific value."""
        self.current = value

    def reset(self, total: int | None = None) -> None:
        """Reset progress and optionally update total."""
        self.current = 0
        if total is not None:
            self.total = total
        self._start_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return time.perf_counter() - self._start_time

    @property
    def elapsed_str(self) -> str:
        """Formatted elapsed time."""
        return format_elapsed_time(self.elapsed)

    @property
    def remaining(self) -> int:
        """Remaining items."""
        return max(0, self.total - self.current)

    @property
    def percentage(self) -> float:
        """Progress percentage (0-100)."""
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100)

    @property
    def fraction(self) -> float:
        """Progress fraction (0.0-1.0)."""
        if self.total <= 0:
            return 0.0
        return min(1.0, self.current / self.total)

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time remaining in seconds."""
        if self.current <= 0:
            return None
        rate = self.current / self.elapsed
        if rate <= 0:
            return None
        return self.remaining / rate

    @property
    def eta_str(self) -> str:
        """Formatted ETA string."""
        eta = self.eta_seconds
        if eta is None:
            return "calculating..."
        return format_elapsed_time(eta)

    @property
    def rate_per_second(self) -> float:
        """Processing rate in items per second."""
        if self.elapsed <= 0:
            return 0.0
        return self.current / self.elapsed

    @property
    def is_complete(self) -> bool:
        """Whether progress is complete."""
        return self.current >= self.total

    def format_status(self) -> str:
        """Format a status line."""
        return (
            f"{self.current}/{self.total} "
            f"({self.percentage:.1f}%) "
            f"ETA: {self.eta_str}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current": self.current,
            "total": self.total,
            "percentage": self.percentage,
            "elapsed": self.elapsed,
            "eta_seconds": self.eta_seconds,
            "rate_per_second": self.rate_per_second,
        }


# =============================================================================
# Running Statistics (Online Algorithm)
# =============================================================================


@dataclass
class RunningStats:
    """Online algorithm for computing mean and variance.

    Uses Welford's online algorithm for numerically stable
    computation of running mean and variance without storing
    all values.

    Example:
        stats = RunningStats("latency_ms")
        for request in requests:
            latency = measure_latency(request)
            stats.update(latency)

        print(f"Mean: {stats.mean:.2f}ms, Stddev: {stats.stddev:.2f}ms")
    """

    name: str = ""
    count: int = 0
    _mean: float = 0.0
    _m2: float = 0.0  # Sum of squared differences
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def update(self, value: float) -> None:
        """Update statistics with a new value.

        Uses Welford's online algorithm for numerical stability.
        """
        self.count += 1
        delta = value - self._mean
        self._mean += delta / self.count
        delta2 = value - self._mean
        self._m2 += delta * delta2

        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    @property
    def mean(self) -> float:
        """Running mean."""
        return self._mean if self.count > 0 else 0.0

    @property
    def variance(self) -> float:
        """Sample variance."""
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def stddev(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.variance)

    @property
    def population_variance(self) -> float:
        """Population variance."""
        if self.count < 1:
            return 0.0
        return self._m2 / self.count

    @property
    def population_stddev(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.population_variance)

    def reset(self) -> None:
        """Reset all statistics."""
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self.min_value = float("inf")
        self.max_value = float("-inf")

    def merge(self, other: "RunningStats") -> None:
        """Merge another RunningStats into this one.

        Uses parallel algorithm for combining statistics.
        """
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self._mean = other._mean
            self._m2 = other._m2
            self.min_value = other.min_value
            self.max_value = other.max_value
            return

        # Chan's parallel algorithm
        combined_count = self.count + other.count
        delta = other._mean - self._mean
        combined_mean = (
            (self.count * self._mean + other.count * other._mean) / combined_count
        )
        combined_m2 = (
            self._m2
            + other._m2
            + delta * delta * self.count * other.count / combined_count
        )

        self.count = combined_count
        self._mean = combined_mean
        self._m2 = combined_m2
        self.min_value = min(self.min_value, other.min_value)
        self.max_value = max(self.max_value, other.max_value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "count": self.count,
            "mean": self.mean,
            "stddev": self.stddev,
            "min": self.min_value if self.count > 0 else None,
            "max": self.max_value if self.count > 0 else None,
        }

    def __str__(self) -> str:
        if self.count == 0:
            return f"{self.name}: no data" if self.name else "no data"

        parts = []
        if self.name:
            parts.append(f"{self.name}:")

        parts.append(f"n={self.count}")
        parts.append(f"mean={self.mean:.2f}")
        parts.append(f"std={self.stddev:.2f}")

        if self.count > 1:
            if self.min_value != float("inf"):
                parts.append(f"min={self.min_value:.2f}")
            if self.max_value != float("-inf"):
                parts.append(f"max={self.max_value:.2f}")

        return " ".join(parts)


# =============================================================================
# Aggregate Statistics Container
# =============================================================================


@dataclass
class MetricsCollection:
    """Container for multiple named metrics.

    Example:
        metrics = MetricsCollection()
        metrics.timing("db_query").record(0.05)
        metrics.counter("games_processed").increment()
        metrics.stats("move_count").update(42)

        print(metrics.summary())
    """

    _timings: dict[str, TimingStats] = field(default_factory=dict)
    _counters: dict[str, Counter] = field(default_factory=dict)
    _stats: dict[str, RunningStats] = field(default_factory=dict)

    def timing(self, name: str) -> TimingStats:
        """Get or create a TimingStats by name."""
        if name not in self._timings:
            self._timings[name] = TimingStats(name=name)
        return self._timings[name]

    def counter(self, name: str) -> Counter:
        """Get or create a Counter by name."""
        if name not in self._counters:
            self._counters[name] = Counter(name=name)
        return self._counters[name]

    def stats(self, name: str) -> RunningStats:
        """Get or create a RunningStats by name."""
        if name not in self._stats:
            self._stats[name] = RunningStats(name=name)
        return self._stats[name]

    def reset(self) -> None:
        """Reset all metrics."""
        for t in self._timings.values():
            t.reset()
        for c in self._counters.values():
            c.reset()
        for s in self._stats.values():
            s.reset()

    def summary(self) -> str:
        """Generate a summary of all metrics."""
        lines = []

        if self._timings:
            lines.append("Timings:")
            for name, timing in sorted(self._timings.items()):
                lines.append(f"  {timing}")

        if self._counters:
            lines.append("Counters:")
            for name, counter in sorted(self._counters.items()):
                lines.append(f"  {counter}")

        if self._stats:
            lines.append("Statistics:")
            for name, stats in sorted(self._stats.items()):
                lines.append(f"  {stats}")

        return "\n".join(lines) if lines else "No metrics recorded"

    def to_dict(self) -> dict[str, Any]:
        """Convert all metrics to dictionary for serialization."""
        return {
            "timings": {k: v.to_dict() for k, v in self._timings.items()},
            "counters": {k: v.value for k, v in self._counters.items()},
            "stats": {k: v.to_dict() for k, v in self._stats.items()},
        }


__all__ = [
    # Timing
    "TimingStats",
    # Rates
    "RateCalculator",
    # Counters
    "Counter",
    "WinLossCounter",
    # Progress
    "ProgressTracker",
    # Running statistics
    "RunningStats",
    # Collection
    "MetricsCollection",
]
