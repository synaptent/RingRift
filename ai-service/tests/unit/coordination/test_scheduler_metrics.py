"""Tests for SchedulerMetricsCollector.

December 30, 2025: Unit tests for the extracted SchedulerMetricsCollector class.
"""

import time
from unittest.mock import patch

import pytest

from app.coordination.scheduler_metrics import (
    AllocationRecord,
    SchedulerMetrics,
    SchedulerMetricsCollector,
    create_metrics_collector,
)


class TestAllocationRecord:
    """Tests for AllocationRecord dataclass."""

    def test_create_record(self):
        """Test creating an allocation record."""
        record = AllocationRecord(timestamp=1000.0, games=50)
        assert record.timestamp == 1000.0
        assert record.games == 50


class TestSchedulerMetrics:
    """Tests for SchedulerMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = SchedulerMetrics()
        assert metrics.games_allocated_total == 0
        assert metrics.games_allocated_window == 0
        assert metrics.games_per_hour == 0.0
        assert metrics.window_seconds == 3600.0
        assert metrics.allocation_count == 0
        assert metrics.last_allocation_time is None

    def test_custom_values(self):
        """Test metrics with custom values."""
        metrics = SchedulerMetrics(
            games_allocated_total=1000,
            games_allocated_window=500,
            games_per_hour=500.0,
            window_seconds=3600.0,
            allocation_count=10,
            last_allocation_time=1000.0,
        )
        assert metrics.games_allocated_total == 1000
        assert metrics.games_per_hour == 500.0


class TestSchedulerMetricsCollector:
    """Tests for SchedulerMetricsCollector class."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        collector = SchedulerMetricsCollector()
        assert collector.window_seconds == 3600.0
        assert collector._games_allocated_total == 0

    def test_init_custom_window(self):
        """Test initialization with custom window."""
        collector = SchedulerMetricsCollector(window_seconds=1800.0)
        assert collector.window_seconds == 1800.0

    def test_record_allocation(self):
        """Test recording an allocation."""
        collector = SchedulerMetricsCollector()
        collector.record_allocation(50)

        assert collector._games_allocated_total == 50
        assert collector._allocation_events_total == 1
        assert len(collector._allocation_history) == 1

    def test_record_allocation_multiple(self):
        """Test recording multiple allocations."""
        collector = SchedulerMetricsCollector()
        collector.record_allocation(50)
        collector.record_allocation(30)
        collector.record_allocation(20)

        assert collector._games_allocated_total == 100
        assert collector._allocation_events_total == 3
        assert len(collector._allocation_history) == 3

    def test_record_allocation_zero_ignored(self):
        """Test zero allocation is ignored."""
        collector = SchedulerMetricsCollector()
        collector.record_allocation(0)

        assert collector._games_allocated_total == 0
        assert len(collector._allocation_history) == 0

    def test_record_allocation_negative_ignored(self):
        """Test negative allocation is ignored."""
        collector = SchedulerMetricsCollector()
        collector.record_allocation(-10)

        assert collector._games_allocated_total == 0
        assert len(collector._allocation_history) == 0

    def test_get_games_in_window(self):
        """Test getting games in current window."""
        collector = SchedulerMetricsCollector()
        collector.record_allocation(50)
        collector.record_allocation(30)

        assert collector.get_games_in_window() == 80

    def test_get_games_in_window_empty(self):
        """Test getting games with empty history."""
        collector = SchedulerMetricsCollector()
        assert collector.get_games_in_window() == 0

    def test_get_games_per_hour(self):
        """Test computing games per hour."""
        collector = SchedulerMetricsCollector(window_seconds=3600.0)
        collector.record_allocation(500)

        # With 500 games in a 1-hour window
        assert collector.get_games_per_hour() == 500.0

    def test_get_games_per_hour_half_window(self):
        """Test games per hour with 30-minute window."""
        collector = SchedulerMetricsCollector(window_seconds=1800.0)
        collector.record_allocation(250)

        # 250 games in 30 minutes = 500/hour
        assert collector.get_games_per_hour() == 500.0

    def test_get_games_per_hour_empty(self):
        """Test games per hour with no allocations."""
        collector = SchedulerMetricsCollector()
        assert collector.get_games_per_hour() == 0.0

    def test_get_allocation_rate(self):
        """Test allocation rate calculation."""
        collector = SchedulerMetricsCollector(window_seconds=3600.0)
        collector.record_allocation(10)
        collector.record_allocation(20)
        collector.record_allocation(30)

        # 3 events in 1 hour = 3 events/hour
        assert collector.get_allocation_rate() == 3.0

    def test_get_average_allocation_size(self):
        """Test average allocation size."""
        collector = SchedulerMetricsCollector()
        collector.record_allocation(10)
        collector.record_allocation(20)
        collector.record_allocation(30)

        # (10 + 20 + 30) / 3 = 20
        assert collector.get_average_allocation_size() == 20.0

    def test_get_average_allocation_size_empty(self):
        """Test average allocation size with no allocations."""
        collector = SchedulerMetricsCollector()
        assert collector.get_average_allocation_size() == 0.0

    def test_get_last_allocation_time(self):
        """Test getting last allocation timestamp."""
        collector = SchedulerMetricsCollector()

        before = time.time()
        collector.record_allocation(50)
        after = time.time()

        last_time = collector.get_last_allocation_time()
        assert before <= last_time <= after

    def test_get_last_allocation_time_empty(self):
        """Test last allocation time with no allocations."""
        collector = SchedulerMetricsCollector()
        assert collector.get_last_allocation_time() is None

    def test_pruning_old_entries(self):
        """Test old entries are pruned."""
        collector = SchedulerMetricsCollector(window_seconds=1.0)

        collector.record_allocation(50)
        time.sleep(1.1)
        collector.record_allocation(30)

        # After pruning, only recent allocation should remain
        assert collector.get_games_in_window() == 30
        assert len(collector._allocation_history) == 1

    def test_window_setter_prunes_entries(self):
        """Test changing window prunes old entries."""
        collector = SchedulerMetricsCollector(window_seconds=3600.0)
        collector.record_allocation(50)

        # Change to very short window
        collector.window_seconds = 0.001

        # Old entry should be pruned
        time.sleep(0.01)
        assert collector.get_games_in_window() == 0

    def test_get_metrics(self):
        """Test getting full metrics dict."""
        collector = SchedulerMetricsCollector()
        collector.record_allocation(50)
        collector.record_allocation(30)

        metrics = collector.get_metrics()

        assert metrics["games_allocated_total"] == 80
        assert metrics["games_allocated_last_hour"] == 80
        assert metrics["allocation_events_total"] == 2
        assert metrics["allocation_events_in_window"] == 2
        assert metrics["average_allocation_size"] == 40.0
        assert metrics["allocation_window_seconds"] == 3600.0
        assert "games_per_hour" in metrics
        assert "last_allocation_time" in metrics

    def test_get_summary(self):
        """Test getting summary as dataclass."""
        collector = SchedulerMetricsCollector()
        collector.record_allocation(50)

        summary = collector.get_summary()

        assert isinstance(summary, SchedulerMetrics)
        assert summary.games_allocated_total == 50
        assert summary.games_allocated_window == 50
        assert summary.allocation_count == 1

    def test_get_status(self):
        """Test getting full status."""
        collector = SchedulerMetricsCollector()
        collector.record_allocation(50)

        status = collector.get_status()

        assert "metrics" in status
        assert "health" in status
        assert status["metrics"]["games_allocated_total"] == 50

    def test_get_status_with_scheduler_state(self):
        """Test getting status with scheduler state."""
        collector = SchedulerMetricsCollector()
        scheduler_state = {"running": True, "priority_configs": ["hex8_2p"]}

        status = collector.get_status(scheduler_state=scheduler_state)

        assert "scheduler" in status
        assert status["scheduler"]["running"] is True

    def test_health_status_active(self):
        """Test health status when active."""
        collector = SchedulerMetricsCollector()
        collector.record_allocation(100)

        health = collector._compute_health_status()

        assert health["is_active"] is True
        assert health["is_producing"] is True

    def test_health_status_inactive(self):
        """Test health status when inactive."""
        collector = SchedulerMetricsCollector()
        # No allocations

        health = collector._compute_health_status()

        assert health["is_active"] is False
        assert health["is_producing"] is False
        assert health["throughput_level"] == "none"

    def test_health_status_throughput_levels(self):
        """Test different throughput levels."""
        # Use a 1-hour window for predictable math
        collector = SchedulerMetricsCollector(window_seconds=3600.0)

        # Low throughput (1 game in 1 hour = 1 game/hour)
        collector.record_allocation(1)
        health = collector._compute_health_status()
        assert health["throughput_level"] == "low"

        collector.reset()

        # Medium throughput (500 games in 1 hour = 500 games/hour)
        collector.record_allocation(500)
        health = collector._compute_health_status()
        assert health["throughput_level"] == "medium"

        collector.reset()

        # High throughput (>1000/hour)
        collector.record_allocation(1500)
        health = collector._compute_health_status()
        assert health["throughput_level"] == "high"

    def test_reset(self):
        """Test resetting metrics."""
        collector = SchedulerMetricsCollector()
        collector.record_allocation(50)
        collector.record_allocation(30)

        collector.reset()

        assert collector._games_allocated_total == 0
        assert collector._allocation_events_total == 0
        assert len(collector._allocation_history) == 0

    def test_max_history_size(self):
        """Test history size is limited."""
        collector = SchedulerMetricsCollector(max_history_size=10)

        # Record more than max
        for i in range(15):
            collector.record_allocation(10)

        # Should only keep max entries
        assert len(collector._allocation_history) <= 10

    def test_history_deque_maxlen(self):
        """Test deque maxlen enforces limit."""
        collector = SchedulerMetricsCollector(max_history_size=5)

        for i in range(10):
            collector.record_allocation(1)

        assert len(collector._allocation_history) == 5


class TestCreateMetricsCollector:
    """Tests for create_metrics_collector factory function."""

    def test_create_with_defaults(self):
        """Test creating collector with defaults."""
        collector = create_metrics_collector()
        assert collector.window_seconds == 3600.0

    def test_create_with_custom_window(self):
        """Test creating collector with custom window."""
        collector = create_metrics_collector(window_seconds=1800.0)
        assert collector.window_seconds == 1800.0


class TestMetricsIntegration:
    """Integration tests for metrics collector."""

    def test_realistic_allocation_pattern(self):
        """Test with realistic allocation pattern."""
        collector = SchedulerMetricsCollector()

        # Simulate allocations over time
        for _ in range(10):
            collector.record_allocation(50)  # 50 games per allocation

        metrics = collector.get_metrics()

        assert metrics["games_allocated_total"] == 500
        assert metrics["allocation_events_total"] == 10
        assert metrics["average_allocation_size"] == 50.0

    def test_bursty_allocation_pattern(self):
        """Test with bursty allocation pattern."""
        collector = SchedulerMetricsCollector()

        # Burst of small allocations
        for _ in range(100):
            collector.record_allocation(5)

        # Single large allocation
        collector.record_allocation(500)

        metrics = collector.get_metrics()

        assert metrics["games_allocated_total"] == 1000
        assert metrics["allocation_events_total"] == 101

    def test_metrics_accuracy_over_time(self):
        """Test metrics remain accurate over time."""
        collector = SchedulerMetricsCollector(window_seconds=1.0)

        # First batch
        collector.record_allocation(100)

        # Wait for window to expire
        time.sleep(1.1)

        # Second batch
        collector.record_allocation(50)

        # Window should only have second batch
        metrics = collector.get_metrics()
        assert metrics["games_allocated_last_hour"] == 50

        # Total should have both
        assert metrics["games_allocated_total"] == 150
