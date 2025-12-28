"""Tests for MonitorBase class.

December 2025 - Phase 3 consolidation tests.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.monitor_base import (
    MonitorBase,
    MonitorConfig,
    MonitorStats,
    get_monitor,
    reset_all_monitors,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass(kw_only=True)
class TestMonitorConfig(MonitorConfig):
    """Test configuration."""
    test_setting: str = "test_value"


class TestMonitor(MonitorBase[TestMonitorConfig]):
    """Concrete test monitor implementation."""

    def __init__(self, config: TestMonitorConfig | None = None):
        super().__init__(config)
        self.run_cycle_called = 0
        self.events_received: list[dict] = []
        self._fail_on_cycle = False

    async def _run_cycle(self) -> None:
        self.run_cycle_called += 1
        self.record_cycle()
        if self._fail_on_cycle:
            raise RuntimeError("Test cycle failure")

    def _get_default_config(self) -> TestMonitorConfig:
        return TestMonitorConfig(check_interval_seconds=1)

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        return {
            "test_event": self._on_test_event,
        }

    async def _on_test_event(self, event: dict) -> None:
        if not self._is_duplicate_event(event):
            self.events_received.append(event)
            self.record_event()


class NoEventsMonitor(MonitorBase[MonitorConfig]):
    """Monitor with no event subscriptions."""

    async def _run_cycle(self) -> None:
        pass

    def _get_default_config(self) -> MonitorConfig:
        return MonitorConfig(check_interval_seconds=1)


@pytest.fixture(autouse=True)
def reset_monitors():
    """Reset all monitor singletons before each test."""
    reset_all_monitors()
    yield
    reset_all_monitors()


# =============================================================================
# MonitorStats Tests
# =============================================================================


class TestMonitorStats:
    """Tests for MonitorStats dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        stats = MonitorStats()
        assert stats.events_processed == 0
        assert stats.errors_count == 0
        assert stats.cycles_completed == 0

    def test_error_rate_zero_cycles(self):
        """Test error rate with zero cycles."""
        stats = MonitorStats()
        assert stats.error_rate == 0.0

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        stats = MonitorStats()
        stats.cycles_completed = 100
        stats.errors_count = 25
        assert stats.error_rate == 0.25

    def test_error_rate_capped_at_one(self):
        """Test error rate doesn't exceed 1.0."""
        stats = MonitorStats()
        stats.cycles_completed = 10
        stats.errors_count = 20
        assert stats.error_rate == 1.0

    def test_to_dict(self):
        """Test dictionary serialization."""
        stats = MonitorStats()
        stats.events_processed = 42
        stats.cycles_completed = 10
        stats.custom = {"my_key": "my_value"}

        d = stats.to_dict()
        assert d["events_processed"] == 42
        assert d["cycles_completed"] == 10
        assert d["my_key"] == "my_value"


# =============================================================================
# MonitorConfig Tests
# =============================================================================


class TestMonitorConfig:
    """Tests for MonitorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MonitorConfig()
        assert config.enabled is True
        assert config.check_interval_seconds == 300
        assert config.unhealthy_error_threshold == 0.5
        assert config.degraded_error_threshold == 0.1
        assert config.stale_threshold_seconds == 1800.0
        assert config.subscribe_to_events is True
        assert config.dedup_enabled is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = TestMonitorConfig(
            check_interval_seconds=60,
            unhealthy_error_threshold=0.3,
            test_setting="custom",
        )
        assert config.check_interval_seconds == 60
        assert config.unhealthy_error_threshold == 0.3
        assert config.test_setting == "custom"


# =============================================================================
# MonitorBase Singleton Tests
# =============================================================================


class TestMonitorBaseSingleton:
    """Tests for singleton pattern."""

    def test_get_instance_creates_singleton(self):
        """Test singleton creation."""
        m1 = TestMonitor.get_instance()
        m2 = TestMonitor.get_instance()
        assert m1 is m2

    def test_different_classes_different_instances(self):
        """Test different monitor classes have different singletons."""
        m1 = TestMonitor.get_instance()
        m2 = NoEventsMonitor.get_instance()
        assert m1 is not m2

    def test_reset_instance(self):
        """Test singleton reset."""
        m1 = TestMonitor.get_instance()
        TestMonitor.reset_instance()
        m2 = TestMonitor.get_instance()
        assert m1 is not m2

    def test_has_instance(self):
        """Test singleton existence check."""
        assert not TestMonitor.has_instance()
        TestMonitor.get_instance()
        assert TestMonitor.has_instance()

    def test_get_instance_with_config(self):
        """Test singleton creation with config."""
        config = TestMonitorConfig(test_setting="custom_value")
        m = TestMonitor.get_instance(config)
        assert m.config.test_setting == "custom_value"


# =============================================================================
# MonitorBase Lifecycle Tests
# =============================================================================


class TestMonitorBaseLifecycle:
    """Tests for lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test basic start/stop."""
        monitor = TestMonitor()
        assert not monitor.is_running

        await monitor.start()
        assert monitor.is_running

        await monitor.stop()
        assert not monitor.is_running

    @pytest.mark.asyncio
    async def test_double_start(self):
        """Test starting already running monitor."""
        monitor = TestMonitor()
        await monitor.start()
        await monitor.start()  # Should not error
        assert monitor.is_running
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_double_stop(self):
        """Test stopping already stopped monitor."""
        monitor = TestMonitor()
        await monitor.stop()  # Should not error
        assert not monitor.is_running

    @pytest.mark.asyncio
    async def test_disabled_config(self):
        """Test monitor with disabled config."""
        config = TestMonitorConfig(enabled=False)
        monitor = TestMonitor(config)
        await monitor.start()
        assert not monitor.is_running

    @pytest.mark.asyncio
    async def test_uptime_tracking(self):
        """Test uptime tracking."""
        monitor = TestMonitor()
        assert monitor.uptime_seconds == 0.0

        await monitor.start()
        await asyncio.sleep(0.1)
        assert monitor.uptime_seconds > 0.0
        await monitor.stop()


# =============================================================================
# MonitorBase Event Subscription Tests
# =============================================================================


class TestMonitorBaseEventSubscription:
    """Tests for event subscription."""

    @pytest.mark.asyncio
    async def test_subscribe_to_events(self):
        """Test event subscription on start."""
        with patch("app.coordination.monitor_base.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            monitor = TestMonitor()
            await monitor.start()

            # Should have subscribed to test_event
            mock_router.subscribe.assert_called()
            assert monitor._event_subscribed

            await monitor.stop()

            # Should have unsubscribed
            mock_router.unsubscribe.assert_called()

    @pytest.mark.asyncio
    async def test_subscribe_disabled(self):
        """Test subscription disabled via config."""
        config = TestMonitorConfig(subscribe_to_events=False)
        monitor = TestMonitor(config)

        with patch("app.coordination.monitor_base.get_router") as mock_get_router:
            await monitor.start()
            mock_get_router.assert_not_called()
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_subscribe_import_error(self):
        """Test graceful handling of import error."""
        with patch(
            "app.coordination.monitor_base.get_router",
            side_effect=ImportError("No router"),
        ):
            monitor = TestMonitor()
            await monitor.start()
            assert not monitor._event_subscribed
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_no_subscriptions_defined(self):
        """Test monitor with no event subscriptions."""
        monitor = NoEventsMonitor()
        await monitor.start()
        assert len(monitor._event_subscriptions) == 0
        await monitor.stop()


# =============================================================================
# MonitorBase Deduplication Tests
# =============================================================================


class TestMonitorBaseDeduplication:
    """Tests for event deduplication."""

    def test_duplicate_detection(self):
        """Test duplicate event detection."""
        monitor = TestMonitor()
        event = {"id": "123", "type": "test"}

        # First time - not a duplicate
        assert not monitor._is_duplicate_event(event)

        # Second time - is a duplicate
        assert monitor._is_duplicate_event(event)

    def test_different_events_not_duplicates(self):
        """Test different events are not duplicates."""
        monitor = TestMonitor()

        assert not monitor._is_duplicate_event({"id": "1"})
        assert not monitor._is_duplicate_event({"id": "2"})

    def test_dedup_key_fields(self):
        """Test deduplication with specific key fields."""
        monitor = TestMonitor()
        event1 = {"id": "1", "timestamp": 100}
        event2 = {"id": "1", "timestamp": 200}  # Different timestamp

        # With all fields - different
        assert not monitor._is_duplicate_event(event1)
        assert not monitor._is_duplicate_event(event2)

        # Reset
        monitor._seen_events.clear()

        # With only id field - duplicate
        assert not monitor._is_duplicate_event(event1, key_fields=["id"])
        assert monitor._is_duplicate_event(event2, key_fields=["id"])

    def test_dedup_disabled(self):
        """Test deduplication can be disabled."""
        config = TestMonitorConfig(dedup_enabled=False)
        monitor = TestMonitor(config)
        event = {"id": "123"}

        assert not monitor._is_duplicate_event(event)
        assert not monitor._is_duplicate_event(event)

    def test_dedup_ttl_expiry(self):
        """Test deduplication entries expire."""
        config = TestMonitorConfig(dedup_ttl_seconds=0.1)
        monitor = TestMonitor(config)
        event = {"id": "123"}

        assert not monitor._is_duplicate_event(event)

        # Wait for TTL
        time.sleep(0.15)

        # Should not be a duplicate anymore
        assert not monitor._is_duplicate_event(event)

    def test_dedup_size_limit(self):
        """Test deduplication size limit."""
        config = TestMonitorConfig(dedup_max_size=5)
        monitor = TestMonitor(config)

        # Add more than max size
        for i in range(10):
            monitor._is_duplicate_event({"id": str(i)})

        # Size should be limited
        assert len(monitor._seen_events) <= 5


# =============================================================================
# MonitorBase Health Check Tests
# =============================================================================


class TestMonitorBaseHealthCheck:
    """Tests for health check."""

    def test_health_check_stopped(self):
        """Test health check when stopped."""
        monitor = TestMonitor()
        result = monitor.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_health_check_running(self):
        """Test health check when running."""
        monitor = TestMonitor()
        await monitor.start()

        result = monitor.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_health_check_degraded(self):
        """Test health check with elevated error rate."""
        config = TestMonitorConfig(
            degraded_error_threshold=0.1,
            unhealthy_error_threshold=0.5,
        )
        monitor = TestMonitor(config)
        await monitor.start()

        # Simulate 15% error rate
        monitor._monitor_stats.cycles_completed = 100
        monitor._monitor_stats.errors_count = 15

        result = monitor.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.DEGRADED
        assert "15.0%" in result.message

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test health check with high error rate."""
        config = TestMonitorConfig(unhealthy_error_threshold=0.5)
        monitor = TestMonitor(config)
        await monitor.start()

        # Simulate 60% error rate
        monitor._monitor_stats.cycles_completed = 100
        monitor._monitor_stats.errors_count = 60

        result = monitor.health_check()
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert "60.0%" in result.message

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_health_check_stale(self):
        """Test health check with stale activity."""
        config = TestMonitorConfig(stale_threshold_seconds=0.1)
        monitor = TestMonitor(config)
        await monitor.start()

        # Set last activity to long ago
        monitor._monitor_stats.last_activity = time.time() - 1.0

        result = monitor.health_check()
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert "Stale" in result.message

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_health_check_details(self):
        """Test health check includes details."""
        monitor = TestMonitor()
        await monitor.start()

        # Record some activity
        monitor.record_cycle()
        monitor.record_event()

        result = monitor.health_check()
        assert "cycles_completed" in result.details
        assert result.details["cycles_completed"] == 1
        assert result.details["events_processed"] == 1

        await monitor.stop()


# =============================================================================
# MonitorBase Status and Recording Tests
# =============================================================================


class TestMonitorBaseStatus:
    """Tests for status and recording methods."""

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test get_status returns expected fields."""
        monitor = TestMonitor()
        await monitor.start()

        status = monitor.get_status()
        assert "name" in status
        assert "running" in status
        assert status["running"] is True
        assert "uptime_seconds" in status
        assert "stats" in status

        await monitor.stop()

    def test_record_event(self):
        """Test event recording."""
        monitor = TestMonitor()
        initial = monitor._monitor_stats.events_processed

        monitor.record_event()
        assert monitor._monitor_stats.events_processed == initial + 1
        assert monitor._monitor_stats.last_activity > 0

    def test_record_error(self):
        """Test error recording."""
        monitor = TestMonitor()

        monitor.record_error("Test error")
        assert monitor._monitor_stats.errors_count == 1
        assert monitor._monitor_stats.last_error == "Test error"
        assert monitor._monitor_stats.last_error_time > 0

    def test_record_cycle(self):
        """Test cycle recording."""
        monitor = TestMonitor()
        initial = monitor._monitor_stats.cycles_completed

        monitor.record_cycle()
        assert monitor._monitor_stats.cycles_completed == initial + 1


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_monitor(self):
        """Test get_monitor factory function."""
        monitor = get_monitor(TestMonitor)
        assert isinstance(monitor, TestMonitor)

        # Should return same instance
        monitor2 = get_monitor(TestMonitor)
        assert monitor is monitor2

    def test_reset_all_monitors(self):
        """Test reset_all_monitors function."""
        # Create some instances
        TestMonitor.get_instance()
        NoEventsMonitor.get_instance()

        assert TestMonitor.has_instance()
        assert NoEventsMonitor.has_instance()

        reset_all_monitors()

        assert not TestMonitor.has_instance()
        assert not NoEventsMonitor.has_instance()


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestMonitorBaseIntegration:
    """Integration-style tests."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test full monitor lifecycle."""
        monitor = TestMonitor.get_instance()

        # Start
        await monitor.start()
        assert monitor.is_running
        assert monitor._event_subscribed or not monitor.config.subscribe_to_events

        # Simulate some activity
        for _ in range(5):
            monitor.record_cycle()
            monitor.record_event()

        monitor.record_error("Test error")

        # Check status
        status = monitor.get_status()
        assert status["stats"]["cycles_completed"] == 5
        assert status["stats"]["events_processed"] == 5
        assert status["stats"]["errors_count"] == 1

        # Check health
        health = monitor.health_check()
        assert health.healthy is True  # Error rate is 20%

        # Stop
        await monitor.stop()
        assert not monitor.is_running

    @pytest.mark.asyncio
    async def test_event_handling_flow(self):
        """Test event handling with deduplication."""
        monitor = TestMonitor()

        # Manually call event handler
        event = {"id": "123", "data": "test"}
        await monitor._on_test_event(event)
        assert len(monitor.events_received) == 1

        # Duplicate should be filtered
        await monitor._on_test_event(event)
        assert len(monitor.events_received) == 1

        # Different event should pass
        await monitor._on_test_event({"id": "456", "data": "test2"})
        assert len(monitor.events_received) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
