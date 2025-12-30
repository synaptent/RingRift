"""Tests for health_check_mixin.py.

December 30, 2025: Created comprehensive tests for the health check mixin
used by 76+ coordinators.

Tests cover:
- HealthCheckMixin class attributes
- health_check() method status determination
- Error rate thresholds (unhealthy, degraded)
- Stale activity detection
- Various attribute name patterns
- is_healthy() and get_health_status() convenience methods
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.coordination.contracts import CoordinatorStatus
from app.coordination.mixins.health_check_mixin import HealthCheckMixin


# =============================================================================
# Test Fixtures
# =============================================================================


class MockComponent(HealthCheckMixin):
    """Mock component for testing HealthCheckMixin."""

    def __init__(
        self,
        running: bool = True,
        cycle_count: int = 0,
        error_count: int = 0,
        start_time: float | None = None,
        last_activity_time: float = 0.0,
    ):
        self._running = running
        self._cycle_count = cycle_count
        self._error_count = error_count
        self._start_time = start_time
        self._last_activity_time = last_activity_time


class MockComponentBaseDaemonStyle(HealthCheckMixin):
    """Mock component using BaseDaemon-style attribute names."""

    def __init__(self, cycles: int = 0, errors: int = 0):
        self._cycles_completed = cycles
        self._errors_count = errors
        self._running = True


class MockComponentGenericStyle(HealthCheckMixin):
    """Mock component using generic attribute names."""

    def __init__(self, total: int = 0, failed: int = 0):
        self.total_runs = total
        self.failed_runs = failed
        self._running = True


class MockComponentCustomThresholds(HealthCheckMixin):
    """Mock component with custom thresholds."""

    UNHEALTHY_THRESHOLD = 0.3  # 30% error rate
    DEGRADED_THRESHOLD = 0.05  # 5% error rate
    HEALTH_CHECK_WINDOW_SECONDS = 600.0  # 10 minutes

    def __init__(self, cycles: int = 0, errors: int = 0):
        self._cycle_count = cycles
        self._error_count = errors
        self._running = True


# =============================================================================
# Class Attribute Tests
# =============================================================================


class TestHealthCheckMixinDefaults:
    """Tests for default class attributes."""

    def test_default_unhealthy_threshold(self):
        """Test default unhealthy threshold is 50%."""
        assert HealthCheckMixin.UNHEALTHY_THRESHOLD == 0.5

    def test_default_degraded_threshold(self):
        """Test default degraded threshold is 10%."""
        assert HealthCheckMixin.DEGRADED_THRESHOLD == 0.1

    def test_default_health_check_window(self):
        """Test default health check window is 30 minutes."""
        assert HealthCheckMixin.HEALTH_CHECK_WINDOW_SECONDS == 1800.0

    def test_custom_thresholds_override(self):
        """Test that custom thresholds override defaults."""
        component = MockComponentCustomThresholds()
        assert component.UNHEALTHY_THRESHOLD == 0.3
        assert component.DEGRADED_THRESHOLD == 0.05
        assert component.HEALTH_CHECK_WINDOW_SECONDS == 600.0


# =============================================================================
# health_check() Method Tests
# =============================================================================


class TestHealthCheckMethod:
    """Tests for health_check() method."""

    def test_stopped_component(self):
        """Test health check for stopped component."""
        component = MockComponent(running=False)
        result = component.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.STOPPED
        assert "stopped" in result.message.lower()

    def test_healthy_running_component(self):
        """Test health check for healthy running component."""
        component = MockComponent(
            running=True,
            cycle_count=100,
            error_count=5,  # 5% error rate
            start_time=time.time() - 3600,
            last_activity_time=time.time(),
        )
        result = component.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert "healthy" in result.message.lower()

    def test_degraded_error_rate(self):
        """Test degraded status when error rate exceeds threshold."""
        component = MockComponent(
            running=True,
            cycle_count=100,
            error_count=20,  # 20% error rate (> 10% degraded threshold)
            last_activity_time=time.time(),
        )
        result = component.health_check()

        assert result.healthy is True  # Still functional
        assert result.status == CoordinatorStatus.DEGRADED
        assert "elevated error rate" in result.message.lower()

    def test_unhealthy_error_rate(self):
        """Test unhealthy status when error rate exceeds unhealthy threshold."""
        component = MockComponent(
            running=True,
            cycle_count=100,
            error_count=60,  # 60% error rate (> 50% unhealthy threshold)
            last_activity_time=time.time(),
        )
        result = component.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert "high error rate" in result.message.lower()

    def test_stale_activity(self):
        """Test degraded status when activity is stale."""
        component = MockComponent(
            running=True,
            cycle_count=100,
            error_count=1,  # Low error rate
            last_activity_time=time.time() - 2000,  # 33+ minutes ago
        )
        result = component.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.DEGRADED
        assert "no activity" in result.message.lower()

    def test_no_activity_time_is_ok(self):
        """Test that zero last_activity_time doesn't trigger stale warning."""
        component = MockComponent(
            running=True,
            cycle_count=100,
            error_count=1,
            last_activity_time=0.0,  # No activity time set
        )
        result = component.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING

    def test_details_include_metrics(self):
        """Test that details include all expected metrics."""
        component = MockComponent(
            running=True,
            cycle_count=50,
            error_count=5,
            start_time=time.time() - 120,
            last_activity_time=time.time() - 10,
        )
        result = component.health_check()

        assert result.details is not None
        assert result.details["cycles"] == 50
        assert result.details["errors"] == 5
        assert result.details["error_rate"] == 0.1
        assert result.details["uptime_seconds"] >= 119
        assert "last_activity" in result.details
        assert "seconds_since_activity" in result.details


# =============================================================================
# Attribute Detection Tests
# =============================================================================


class TestAttributeDetection:
    """Tests for attribute detection across different naming patterns."""

    def test_handler_base_style_attributes(self):
        """Test detection of HandlerBase-style attributes (_cycle_count, _error_count)."""
        component = MockComponent(cycle_count=100, error_count=10)
        assert component._get_cycle_count() == 100
        assert component._get_error_count() == 10

    def test_base_daemon_style_attributes(self):
        """Test detection of BaseDaemon-style attributes (_cycles_completed, _errors_count)."""
        component = MockComponentBaseDaemonStyle(cycles=200, errors=20)
        assert component._get_cycle_count() == 200
        assert component._get_error_count() == 20

    def test_generic_style_attributes(self):
        """Test detection of generic attributes (total_runs, failed_runs)."""
        component = MockComponentGenericStyle(total=150, failed=15)
        assert component._get_cycle_count() == 150
        assert component._get_error_count() == 15

    def test_running_state_default(self):
        """Test running state defaults to True if not set."""

        class NoRunningAttr(HealthCheckMixin):
            pass

        component = NoRunningAttr()
        assert component._get_running_state() is True

    def test_running_state_explicit(self):
        """Test explicit running state detection."""
        component = MockComponent(running=False)
        assert component._get_running_state() is False


# =============================================================================
# Error Rate Calculation Tests
# =============================================================================


class TestErrorRateCalculation:
    """Tests for error rate calculation."""

    def test_zero_cycles_no_division_error(self):
        """Test that zero cycles doesn't cause division by zero."""
        component = MockComponent(cycle_count=0, error_count=0)
        assert component._get_error_rate() == 0.0

    def test_zero_errors(self):
        """Test error rate with zero errors."""
        component = MockComponent(cycle_count=100, error_count=0)
        assert component._get_error_rate() == 0.0

    def test_50_percent_error_rate(self):
        """Test 50% error rate calculation."""
        component = MockComponent(cycle_count=100, error_count=50)
        assert component._get_error_rate() == 0.5

    def test_100_percent_error_rate(self):
        """Test 100% error rate calculation."""
        component = MockComponent(cycle_count=100, error_count=100)
        assert component._get_error_rate() == 1.0


# =============================================================================
# Uptime Calculation Tests
# =============================================================================


class TestUptimeCalculation:
    """Tests for uptime calculation."""

    def test_no_start_time(self):
        """Test uptime with no start time set."""
        component = MockComponent(start_time=None)
        assert component._get_uptime() == 0.0

    def test_uptime_calculation(self):
        """Test uptime calculation from start time."""
        component = MockComponent(start_time=time.time() - 100)
        uptime = component._get_uptime()
        assert 99 <= uptime <= 101  # Allow 1 second tolerance


# =============================================================================
# Convenience Method Tests
# =============================================================================


class TestConvenienceMethods:
    """Tests for convenience methods."""

    def test_is_healthy_true(self):
        """Test is_healthy() returns True for healthy component."""
        component = MockComponent(
            running=True,
            cycle_count=100,
            error_count=1,
            last_activity_time=time.time(),
        )
        assert component.is_healthy() is True

    def test_is_healthy_false(self):
        """Test is_healthy() returns False for unhealthy component."""
        component = MockComponent(
            running=True,
            cycle_count=100,
            error_count=60,  # 60% error rate
        )
        assert component.is_healthy() is False

    def test_get_health_status_running(self):
        """Test get_health_status() returns 'running' for healthy component."""
        component = MockComponent(
            running=True,
            cycle_count=100,
            error_count=1,
            last_activity_time=time.time(),
        )
        assert component.get_health_status() == "running"

    def test_get_health_status_stopped(self):
        """Test get_health_status() returns 'stopped' for stopped component."""
        component = MockComponent(running=False)
        assert component.get_health_status() == "stopped"

    def test_get_health_status_degraded(self):
        """Test get_health_status() returns 'degraded' for degraded component."""
        component = MockComponent(
            running=True,
            cycle_count=100,
            error_count=20,  # 20% error rate
            last_activity_time=time.time(),
        )
        assert component.get_health_status() == "degraded"

    def test_get_health_status_error(self):
        """Test get_health_status() returns 'error' for unhealthy component."""
        component = MockComponent(
            running=True,
            cycle_count=100,
            error_count=60,  # 60% error rate
        )
        assert component.get_health_status() == "error"


# =============================================================================
# Custom Threshold Tests
# =============================================================================


class TestCustomThresholds:
    """Tests for custom threshold behavior."""

    def test_custom_unhealthy_threshold(self):
        """Test custom unhealthy threshold (30%)."""
        component = MockComponentCustomThresholds(cycles=100, errors=35)
        result = component.health_check()
        assert result.status == CoordinatorStatus.ERROR  # 35% > 30%

    def test_custom_degraded_threshold(self):
        """Test custom degraded threshold (5%)."""
        component = MockComponentCustomThresholds(cycles=100, errors=8)
        result = component.health_check()
        assert result.status == CoordinatorStatus.DEGRADED  # 8% > 5%

    def test_custom_stale_window(self):
        """Test custom health check window (10 minutes)."""
        component = MockComponentCustomThresholds(cycles=100, errors=1)
        component._last_activity_time = time.time() - 700  # 11+ minutes ago

        result = component.health_check()
        assert result.status == CoordinatorStatus.DEGRADED


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_exactly_at_unhealthy_threshold(self):
        """Test behavior at exactly unhealthy threshold."""
        component = MockComponent(
            running=True,
            cycle_count=100,
            error_count=50,  # Exactly 50%
            last_activity_time=time.time(),
        )
        result = component.health_check()
        # At threshold should not trigger (> not >=)
        assert result.status == CoordinatorStatus.DEGRADED

    def test_exactly_at_degraded_threshold(self):
        """Test behavior at exactly degraded threshold."""
        component = MockComponent(
            running=True,
            cycle_count=100,
            error_count=10,  # Exactly 10%
            last_activity_time=time.time(),
        )
        result = component.health_check()
        # At threshold should not trigger (> not >=)
        assert result.status == CoordinatorStatus.RUNNING

    def test_very_high_cycle_count(self):
        """Test with very high cycle count."""
        component = MockComponent(
            running=True,
            cycle_count=1_000_000,
            error_count=1,  # Very low error rate
            last_activity_time=time.time(),
        )
        result = component.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING

    def test_all_zeros(self):
        """Test with all zero metrics."""
        component = MockComponent(
            running=True,
            cycle_count=0,
            error_count=0,
            start_time=None,
            last_activity_time=0.0,
        )
        result = component.health_check()
        assert result.healthy is True
        # Zero cycles gives 0% error rate which is healthy

    def test_component_with_no_attributes(self):
        """Test component with minimal attributes."""

        class MinimalComponent(HealthCheckMixin):
            pass

        component = MinimalComponent()
        result = component.health_check()

        # Should still work with defaults
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING


# =============================================================================
# Last Activity Detection Tests
# =============================================================================


class TestLastActivityDetection:
    """Tests for last activity time detection."""

    def test_last_activity_time_attribute(self):
        """Test detection of _last_activity_time."""
        component = MockComponent(last_activity_time=12345.0)
        assert component._get_last_activity_time() == 12345.0

    def test_last_sync_execution_attribute(self):
        """Test detection of _last_sync_execution."""

        class SyncComponent(HealthCheckMixin):
            def __init__(self):
                self._last_sync_execution = 54321.0

        component = SyncComponent()
        assert component._get_last_activity_time() == 54321.0

    def test_last_run_time_attribute(self):
        """Test detection of last_run_time."""

        class RunnerComponent(HealthCheckMixin):
            def __init__(self):
                self.last_run_time = 98765.0

        component = RunnerComponent()
        assert component._get_last_activity_time() == 98765.0

    def test_no_activity_attribute(self):
        """Test when no activity attribute exists."""

        class NoActivityComponent(HealthCheckMixin):
            pass

        component = NoActivityComponent()
        assert component._get_last_activity_time() == 0.0
