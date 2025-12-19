"""Tests for scripts.lib.metrics module.

Tests metric utilities including:
- TimingStats for timing measurements
- RateCalculator for throughput calculation
- Counter and WinLossCounter
- ProgressTracker with ETA
- RunningStats for online statistics
- MetricsCollection for aggregate metrics
"""

import math
import time

import pytest

from scripts.lib.metrics import (
    Counter,
    MetricsCollection,
    ProgressTracker,
    RateCalculator,
    RunningStats,
    TimingStats,
    WinLossCounter,
)


class TestTimingStats:
    """Tests for TimingStats class."""

    def test_empty_stats(self):
        """Test newly created stats."""
        stats = TimingStats(name="test")

        assert stats.name == "test"
        assert stats.count == 0
        assert stats.total_time == 0.0
        assert stats.avg_time == 0.0

    def test_record_single(self):
        """Test recording a single measurement."""
        stats = TimingStats()
        stats.record(0.1)

        assert stats.count == 1
        assert stats.total_time == pytest.approx(0.1)
        assert stats.avg_time == pytest.approx(0.1)

    def test_record_multiple(self):
        """Test recording multiple measurements."""
        stats = TimingStats()
        stats.record(0.1)
        stats.record(0.2)
        stats.record(0.3)

        assert stats.count == 3
        assert stats.total_time == pytest.approx(0.6)
        assert stats.avg_time == pytest.approx(0.2)
        assert stats.min_time == pytest.approx(0.1)
        assert stats.max_time == pytest.approx(0.3)

    def test_time_context_manager(self):
        """Test timing with context manager."""
        stats = TimingStats()

        with stats.time():
            time.sleep(0.01)  # Sleep 10ms

        assert stats.count == 1
        assert stats.total_time >= 0.01  # At least 10ms

    def test_millisecond_properties(self):
        """Test millisecond conversion properties."""
        stats = TimingStats()
        stats.record(0.5)  # 500ms

        assert stats.avg_time_ms == pytest.approx(500.0)
        assert stats.total_time_ms == pytest.approx(500.0)

    def test_reset(self):
        """Test resetting stats."""
        stats = TimingStats()
        stats.record(0.1)
        stats.record(0.2)

        stats.reset()

        assert stats.count == 0
        assert stats.total_time == 0.0

    def test_merge(self):
        """Test merging two TimingStats."""
        stats1 = TimingStats()
        stats1.record(0.1)
        stats1.record(0.2)

        stats2 = TimingStats()
        stats2.record(0.05)
        stats2.record(0.3)

        stats1.merge(stats2)

        assert stats1.count == 4
        assert stats1.total_time == pytest.approx(0.65)
        assert stats1.min_time == pytest.approx(0.05)
        assert stats1.max_time == pytest.approx(0.3)

    def test_to_dict(self):
        """Test dictionary serialization."""
        stats = TimingStats(name="test")
        stats.record(0.1)

        d = stats.to_dict()

        assert d["name"] == "test"
        assert d["count"] == 1
        assert "avg_time_ms" in d

    def test_str_empty(self):
        """Test string representation with no data."""
        stats = TimingStats(name="empty")
        assert "no data" in str(stats)

    def test_str_with_data(self):
        """Test string representation with data."""
        stats = TimingStats(name="test")
        stats.record(0.1)

        s = str(stats)
        assert "test:" in s
        assert "1 calls" in s


class TestRateCalculator:
    """Tests for RateCalculator class."""

    def test_empty_rate(self):
        """Test rate with no data."""
        rate = RateCalculator()

        assert rate.total_items == 0
        assert rate.rate_per_second == 0.0

    def test_record_with_explicit_time(self):
        """Test recording with explicit elapsed time."""
        rate = RateCalculator()
        rate.record(items=100, elapsed=2.0)

        assert rate.total_items == 100
        assert rate.total_time == 2.0
        assert rate.rate_per_second == pytest.approx(50.0)

    def test_rate_calculations(self):
        """Test rate per minute/hour calculations."""
        rate = RateCalculator()
        rate.record(items=60, elapsed=1.0)

        assert rate.rate_per_second == pytest.approx(60.0)
        assert rate.rate_per_minute == pytest.approx(3600.0)
        assert rate.rate_per_hour == pytest.approx(216000.0)

    def test_start_stop_timer(self):
        """Test start/stop timer functionality."""
        rate = RateCalculator()
        rate.start()
        time.sleep(0.01)
        rate.record(items=10)
        rate.stop()

        assert rate.total_items == 10
        assert rate.total_time >= 0.01

    def test_format_rate(self):
        """Test rate formatting."""
        rate = RateCalculator()

        # Test k format
        rate.record(items=5000, elapsed=1.0)
        assert "5.0k" in rate.format_rate("games")

        # Reset and test normal format
        rate.reset()
        rate.record(items=50, elapsed=1.0)
        assert "50.0 items/sec" in rate.format_rate()

    def test_reset(self):
        """Test resetting rate calculator."""
        rate = RateCalculator()
        rate.record(items=100, elapsed=1.0)
        rate.reset()

        assert rate.total_items == 0
        assert rate.total_time == 0.0


class TestCounter:
    """Tests for Counter class."""

    def test_empty_counter(self):
        """Test new counter."""
        counter = Counter(name="test")

        assert counter.name == "test"
        assert counter.value == 0

    def test_increment(self):
        """Test incrementing counter."""
        counter = Counter()
        counter.increment()
        counter.increment()

        assert counter.value == 2

    def test_increment_by_amount(self):
        """Test incrementing by specific amount."""
        counter = Counter()
        result = counter.increment(5)

        assert counter.value == 5
        assert result == 5

    def test_reset(self):
        """Test resetting counter."""
        counter = Counter()
        counter.increment(10)
        counter.reset()

        assert counter.value == 0


class TestWinLossCounter:
    """Tests for WinLossCounter class."""

    def test_empty_counter(self):
        """Test new counter."""
        counter = WinLossCounter()

        assert counter.wins == 0
        assert counter.losses == 0
        assert counter.draws == 0
        assert counter.total_games == 0
        assert counter.win_rate == 0.0

    def test_record_wins_losses_draws(self):
        """Test recording wins, losses, and draws."""
        counter = WinLossCounter()
        counter.record_win()
        counter.record_loss()
        counter.record_draw()

        assert counter.wins == 1
        assert counter.losses == 1
        assert counter.draws == 1
        assert counter.total_games == 3

    def test_record_result_string(self):
        """Test recording by string."""
        counter = WinLossCounter()
        counter.record_result("win")
        counter.record_result("WIN")  # Case insensitive
        counter.record_result("loss")
        counter.record_result("draw")

        assert counter.wins == 2
        assert counter.losses == 1
        assert counter.draws == 1

    def test_win_rate_calculation(self):
        """Test win rate calculation (draws count as 0.5)."""
        counter = WinLossCounter()
        counter.record_win(7)
        counter.record_loss(2)
        counter.record_draw(1)

        # win_rate = (7 + 0.5*1) / 10 = 7.5/10 = 0.75
        assert counter.win_rate == pytest.approx(0.75)
        # win_rate_strict = 7/10 = 0.7
        assert counter.win_rate_strict == pytest.approx(0.7)

    def test_loss_and_draw_rate(self):
        """Test loss and draw rate calculations."""
        counter = WinLossCounter()
        counter.record_win(5)
        counter.record_loss(3)
        counter.record_draw(2)

        assert counter.loss_rate == pytest.approx(0.3)
        assert counter.draw_rate == pytest.approx(0.2)

    def test_merge(self):
        """Test merging counters."""
        c1 = WinLossCounter()
        c1.record_win(5)
        c1.record_loss(3)

        c2 = WinLossCounter()
        c2.record_win(2)
        c2.record_draw(1)

        c1.merge(c2)

        assert c1.wins == 7
        assert c1.losses == 3
        assert c1.draws == 1

    def test_str_format(self):
        """Test string representation."""
        counter = WinLossCounter()
        counter.record_win(10)
        counter.record_loss(5)
        counter.record_draw(2)

        assert str(counter) == "10-5-2 (W-L-D)"


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_initial_state(self):
        """Test initial progress state."""
        progress = ProgressTracker(total=100)

        assert progress.total == 100
        assert progress.current == 0
        assert progress.percentage == 0.0
        assert progress.remaining == 100

    def test_update(self):
        """Test updating progress."""
        progress = ProgressTracker(total=100)
        progress.update(25)
        progress.update(25)

        assert progress.current == 50
        assert progress.percentage == pytest.approx(50.0)
        assert progress.remaining == 50

    def test_set(self):
        """Test setting progress directly."""
        progress = ProgressTracker(total=100)
        progress.set(75)

        assert progress.current == 75
        assert progress.percentage == pytest.approx(75.0)

    def test_fraction(self):
        """Test fraction property."""
        progress = ProgressTracker(total=200)
        progress.update(50)

        assert progress.fraction == pytest.approx(0.25)

    def test_is_complete(self):
        """Test completion check."""
        progress = ProgressTracker(total=100)
        assert not progress.is_complete

        progress.update(100)
        assert progress.is_complete

    def test_eta_calculation(self):
        """Test ETA calculation."""
        progress = ProgressTracker(total=100)

        # Initially no ETA
        assert progress.eta_seconds is None

        # After some progress, ETA should be calculable
        time.sleep(0.01)
        progress.update(50)
        eta = progress.eta_seconds

        # Should be roughly the same time remaining
        assert eta is not None
        assert eta > 0

    def test_reset(self):
        """Test resetting progress."""
        progress = ProgressTracker(total=100)
        progress.update(50)

        progress.reset(total=200)

        assert progress.current == 0
        assert progress.total == 200

    def test_format_status(self):
        """Test status string formatting."""
        progress = ProgressTracker(total=100)
        progress._start_time = time.perf_counter() - 1.0  # Set fake start time
        progress.update(50)

        status = progress.format_status()

        assert "50/100" in status
        assert "50.0%" in status
        assert "ETA:" in status


class TestRunningStats:
    """Tests for RunningStats class."""

    def test_empty_stats(self):
        """Test empty statistics."""
        stats = RunningStats(name="test")

        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.variance == 0.0
        assert stats.stddev == 0.0

    def test_single_value(self):
        """Test with single value."""
        stats = RunningStats()
        stats.update(10.0)

        assert stats.count == 1
        assert stats.mean == pytest.approx(10.0)
        assert stats.variance == 0.0  # Need at least 2 for sample variance

    def test_multiple_values(self):
        """Test with multiple values."""
        stats = RunningStats()
        values = [2.0, 4.0, 6.0, 8.0, 10.0]
        for v in values:
            stats.update(v)

        assert stats.count == 5
        assert stats.mean == pytest.approx(6.0)
        assert stats.min_value == pytest.approx(2.0)
        assert stats.max_value == pytest.approx(10.0)

    def test_variance_calculation(self):
        """Test variance and stddev calculation."""
        stats = RunningStats()
        # Standard normal sample
        values = [1, 2, 3, 4, 5]
        for v in values:
            stats.update(float(v))

        # Sample variance of [1,2,3,4,5] is 2.5
        assert stats.variance == pytest.approx(2.5)
        assert stats.stddev == pytest.approx(math.sqrt(2.5))

    def test_population_statistics(self):
        """Test population variance vs sample variance."""
        stats = RunningStats()
        values = [1, 2, 3, 4, 5]
        for v in values:
            stats.update(float(v))

        # Population variance = 2.0, sample variance = 2.5
        assert stats.population_variance == pytest.approx(2.0)
        assert stats.population_stddev == pytest.approx(math.sqrt(2.0))

    def test_merge_empty(self):
        """Test merging with empty stats."""
        stats1 = RunningStats()
        stats1.update(1.0)
        stats1.update(2.0)

        stats2 = RunningStats()  # Empty

        stats1.merge(stats2)

        assert stats1.count == 2
        assert stats1.mean == pytest.approx(1.5)

    def test_merge_into_empty(self):
        """Test merging into empty stats."""
        stats1 = RunningStats()  # Empty

        stats2 = RunningStats()
        stats2.update(1.0)
        stats2.update(2.0)

        stats1.merge(stats2)

        assert stats1.count == 2
        assert stats1.mean == pytest.approx(1.5)

    def test_merge_two_populated(self):
        """Test merging two populated stats."""
        stats1 = RunningStats()
        for v in [1.0, 2.0, 3.0]:
            stats1.update(v)

        stats2 = RunningStats()
        for v in [4.0, 5.0, 6.0]:
            stats2.update(v)

        stats1.merge(stats2)

        assert stats1.count == 6
        assert stats1.mean == pytest.approx(3.5)
        assert stats1.min_value == pytest.approx(1.0)
        assert stats1.max_value == pytest.approx(6.0)

    def test_reset(self):
        """Test resetting stats."""
        stats = RunningStats()
        stats.update(1.0)
        stats.update(2.0)
        stats.reset()

        assert stats.count == 0
        assert stats.mean == 0.0

    def test_str_empty(self):
        """Test string with no data."""
        stats = RunningStats(name="test")
        assert "no data" in str(stats)

    def test_str_with_data(self):
        """Test string with data."""
        stats = RunningStats(name="latency")
        stats.update(10.0)
        stats.update(20.0)

        s = str(stats)
        assert "latency:" in s
        assert "n=2" in s
        assert "mean=" in s


class TestMetricsCollection:
    """Tests for MetricsCollection class."""

    def test_timing_get_or_create(self):
        """Test getting/creating timing metrics."""
        metrics = MetricsCollection()

        timing1 = metrics.timing("query")
        timing2 = metrics.timing("query")

        assert timing1 is timing2
        assert timing1.name == "query"

    def test_counter_get_or_create(self):
        """Test getting/creating counters."""
        metrics = MetricsCollection()

        counter1 = metrics.counter("requests")
        counter2 = metrics.counter("requests")

        assert counter1 is counter2
        assert counter1.name == "requests"

    def test_stats_get_or_create(self):
        """Test getting/creating running stats."""
        metrics = MetricsCollection()

        stats1 = metrics.stats("latency")
        stats2 = metrics.stats("latency")

        assert stats1 is stats2
        assert stats1.name == "latency"

    def test_reset_all(self):
        """Test resetting all metrics."""
        metrics = MetricsCollection()

        metrics.timing("query").record(0.1)
        metrics.counter("count").increment(5)
        metrics.stats("latency").update(10.0)

        metrics.reset()

        assert metrics.timing("query").count == 0
        assert metrics.counter("count").value == 0
        assert metrics.stats("latency").count == 0

    def test_summary(self):
        """Test summary generation."""
        metrics = MetricsCollection()

        metrics.timing("query").record(0.1)
        metrics.counter("games").increment(5)
        metrics.stats("moves").update(42)

        summary = metrics.summary()

        assert "Timings:" in summary
        assert "Counters:" in summary
        assert "Statistics:" in summary

    def test_to_dict(self):
        """Test dictionary serialization."""
        metrics = MetricsCollection()

        metrics.timing("query").record(0.1)
        metrics.counter("count").increment(5)
        metrics.stats("latency").update(10.0)

        d = metrics.to_dict()

        assert "timings" in d
        assert "counters" in d
        assert "stats" in d
        assert d["counters"]["count"] == 5
