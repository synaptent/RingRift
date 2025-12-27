"""Tests for HealthCheckMixin - Standard health check implementation.

Tests cover:
- Basic health check functionality
- Attribute name flexibility (BaseDaemon, HandlerBase patterns)
- Error rate calculation
- Status determination (healthy, degraded, unhealthy)
- Stale activity detection
- Threshold customization
- Convenience methods (is_healthy, get_health_status)
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.mixins.health_check_mixin import HealthCheckMixin
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult


class MockDaemonWithMixin(HealthCheckMixin):
    """Mock daemon using BaseDaemon attribute pattern."""

    def __init__(self):
        self._running = True
        self._cycles_completed = 0
        self._errors_count = 0
        self._start_time = time.time()
        self._last_activity_time = time.time()


class MockHandlerWithMixin(HealthCheckMixin):
    """Mock handler using HandlerBase attribute pattern."""

    def __init__(self):
        self._running = True
        self._cycle_count = 0
        self._error_count = 0
        self._start_time = time.time()
        self._last_activity_time = time.time()


class MockCoordinatorWithMixin(HealthCheckMixin):
    """Mock coordinator with custom thresholds."""

    UNHEALTHY_THRESHOLD = 0.3  # 30% instead of 50%
    DEGRADED_THRESHOLD = 0.05  # 5% instead of 10%

    def __init__(self):
        self._running = True
        self._cycles_completed = 100
        self._errors_count = 0
        self._start_time = time.time()


class TestHealthCheckMixinBasics:
    """Basic health check functionality tests."""

    def test_healthy_when_running_no_errors(self):
        """Test healthy status when running with no errors."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 100
        daemon._errors_count = 0

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert "Healthy" in result.message

    def test_stopped_when_not_running(self):
        """Test stopped status when not running."""
        daemon = MockDaemonWithMixin()
        daemon._running = False

        result = daemon.health_check()

        assert result.healthy is True  # Stopped is not unhealthy
        assert result.status == CoordinatorStatus.STOPPED
        assert "stopped" in result.message.lower()

    def test_result_includes_details(self):
        """Test that health check includes detailed metrics."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 50
        daemon._errors_count = 5

        result = daemon.health_check()

        assert "details" in dir(result)
        details = result.details
        assert details["cycles"] == 50
        assert details["errors"] == 5
        assert details["error_rate"] == 0.1
        assert "uptime_seconds" in details


class TestAttributePatterns:
    """Tests for different attribute naming patterns."""

    def test_basedaemon_pattern(self):
        """Test mixin works with BaseDaemon attributes."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 100
        daemon._errors_count = 10

        result = daemon.health_check()

        # Should read from _cycles_completed and _errors_count
        assert result.details["cycles"] == 100
        assert result.details["errors"] == 10

    def test_handlerbase_pattern(self):
        """Test mixin works with HandlerBase attributes."""
        handler = MockHandlerWithMixin()
        handler._cycle_count = 200
        handler._error_count = 20

        result = handler.health_check()

        # Should read from _cycle_count and _error_count
        assert result.details["cycles"] == 200
        assert result.details["errors"] == 20

    def test_generic_pattern_total_runs(self):
        """Test mixin works with total_runs attribute."""

        class GenericMixin(HealthCheckMixin):
            def __init__(self):
                self._running = True
                self.total_runs = 150
                self.failed_runs = 15

        obj = GenericMixin()
        result = obj.health_check()

        assert result.details["cycles"] == 150
        assert result.details["errors"] == 15

    def test_missing_attributes_default_to_zero(self):
        """Test that missing attributes default to zero."""

        class MinimalMixin(HealthCheckMixin):
            def __init__(self):
                self._running = True

        obj = MinimalMixin()
        result = obj.health_check()

        assert result.details["cycles"] == 0
        assert result.details["errors"] == 0
        assert result.details["error_rate"] == 0.0


class TestErrorRateCalculation:
    """Tests for error rate calculation and status determination."""

    def test_healthy_below_degraded_threshold(self):
        """Test healthy when error rate < 10% (default degraded threshold)."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 100
        daemon._errors_count = 5  # 5% error rate

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING

    def test_degraded_above_threshold(self):
        """Test degraded when error rate > 10% but < 50%."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 100
        daemon._errors_count = 20  # 20% error rate

        result = daemon.health_check()

        assert result.healthy is True  # Degraded is still functional
        assert result.status == CoordinatorStatus.DEGRADED
        assert "Elevated error rate" in result.message

    def test_unhealthy_above_high_threshold(self):
        """Test unhealthy when error rate > 50%."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 100
        daemon._errors_count = 60  # 60% error rate

        result = daemon.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert "High error rate" in result.message

    def test_error_rate_calculation_accuracy(self):
        """Test that error rate is calculated correctly."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 200
        daemon._errors_count = 50  # 25% error rate

        result = daemon.health_check()

        assert result.details["error_rate"] == 0.25

    def test_divide_by_zero_protection(self):
        """Test that zero cycles doesn't cause divide by zero."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 0
        daemon._errors_count = 0

        result = daemon.health_check()

        assert result.details["error_rate"] == 0.0  # No crash


class TestStaleActivityDetection:
    """Tests for stale/inactive state detection."""

    def test_degraded_when_stale(self):
        """Test degraded status when no recent activity."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 100
        daemon._errors_count = 0
        daemon._last_activity_time = time.time() - 3600  # 1 hour ago

        result = daemon.health_check()

        assert result.status == CoordinatorStatus.DEGRADED
        assert "No activity" in result.message

    def test_healthy_with_recent_activity(self):
        """Test healthy when activity is recent."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 100
        daemon._errors_count = 0
        daemon._last_activity_time = time.time() - 60  # 1 minute ago

        result = daemon.health_check()

        assert result.status == CoordinatorStatus.RUNNING
        assert "Healthy" in result.message

    def test_no_activity_time_tracked(self):
        """Test behavior when no activity time is tracked."""
        daemon = MockDaemonWithMixin()
        daemon._last_activity_time = 0  # Not tracked

        result = daemon.health_check()

        # Should not trigger stale detection
        assert result.status == CoordinatorStatus.RUNNING


class TestThresholdCustomization:
    """Tests for customizing thresholds in subclasses."""

    def test_custom_unhealthy_threshold(self):
        """Test custom UNHEALTHY_THRESHOLD is respected."""
        coordinator = MockCoordinatorWithMixin()
        coordinator._errors_count = 35  # 35% error rate

        result = coordinator.health_check()

        # With 30% threshold, 35% should be unhealthy
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR

    def test_custom_degraded_threshold(self):
        """Test custom DEGRADED_THRESHOLD is respected."""
        coordinator = MockCoordinatorWithMixin()
        coordinator._errors_count = 8  # 8% error rate

        result = coordinator.health_check()

        # With 5% threshold, 8% should be degraded
        assert result.status == CoordinatorStatus.DEGRADED

    def test_custom_threshold_healthy(self):
        """Test healthy with custom thresholds."""
        coordinator = MockCoordinatorWithMixin()
        coordinator._errors_count = 3  # 3% error rate

        result = coordinator.health_check()

        # Below 5% degraded threshold
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING


class TestUptimeTracking:
    """Tests for uptime calculation."""

    def test_uptime_calculated_from_start_time(self):
        """Test uptime is calculated from _start_time."""
        daemon = MockDaemonWithMixin()
        daemon._start_time = time.time() - 300  # Started 5 minutes ago

        result = daemon.health_check()

        assert 299 <= result.details["uptime_seconds"] <= 301

    def test_uptime_zero_when_no_start_time(self):
        """Test uptime is zero when no start time tracked."""

        class NoStartTimeMixin(HealthCheckMixin):
            def __init__(self):
                self._running = True

        obj = NoStartTimeMixin()
        result = obj.health_check()

        assert result.details["uptime_seconds"] == 0.0


class TestConvenienceMethods:
    """Tests for convenience methods is_healthy() and get_health_status()."""

    def test_is_healthy_true(self):
        """Test is_healthy() returns True when healthy."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 100
        daemon._errors_count = 0

        assert daemon.is_healthy() is True

    def test_is_healthy_false(self):
        """Test is_healthy() returns False when unhealthy."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 100
        daemon._errors_count = 60  # 60% error rate

        assert daemon.is_healthy() is False

    def test_get_health_status_running(self):
        """Test get_health_status() returns 'running' when healthy."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 100
        daemon._errors_count = 0

        status = daemon.get_health_status()

        assert status == "running"

    def test_get_health_status_degraded(self):
        """Test get_health_status() returns 'degraded'."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 100
        daemon._errors_count = 20  # 20% error rate

        status = daemon.get_health_status()

        assert status == "degraded"

    def test_get_health_status_error(self):
        """Test get_health_status() returns 'error'."""
        daemon = MockDaemonWithMixin()
        daemon._cycles_completed = 100
        daemon._errors_count = 60  # 60% error rate

        status = daemon.get_health_status()

        assert status == "error"


class TestLastActivityTimePatterns:
    """Tests for different last activity time attribute patterns."""

    def test_last_activity_time_pattern(self):
        """Test _last_activity_time attribute."""
        daemon = MockDaemonWithMixin()
        daemon._last_activity_time = time.time() - 100

        result = daemon.health_check()

        assert "last_activity" in result.details
        assert 99 <= result.details["seconds_since_activity"] <= 101

    def test_last_sync_execution_pattern(self):
        """Test _last_sync_execution attribute."""

        class SyncMixin(HealthCheckMixin):
            def __init__(self):
                self._running = True
                self._cycles_completed = 100
                self._last_sync_execution = time.time() - 200

        obj = SyncMixin()
        result = obj.health_check()

        assert "last_activity" in result.details
        assert 199 <= result.details["seconds_since_activity"] <= 201

    def test_last_run_time_pattern(self):
        """Test last_run_time attribute."""

        class RunnerMixin(HealthCheckMixin):
            def __init__(self):
                self._running = True
                self._cycles_completed = 100
                self.last_run_time = time.time() - 50

        obj = RunnerMixin()
        result = obj.health_check()

        assert "last_activity" in result.details
        assert 49 <= result.details["seconds_since_activity"] <= 51
