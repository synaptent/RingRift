"""Tests for HealthCheckHelper - Reusable health check utility methods.

Created: December 28, 2025
Purpose: Test the HealthCheckHelper class used by 34+ coordination modules

Tests cover:
- check_error_rate() boundary conditions
- check_uptime_grace() timing behavior
- check_recent_errors() windowed counting
- check_queue_depth() threshold handling
- check_last_activity() staleness detection
- check_connection_health() connection state
- build_details() filtering
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.coordination.health_check_helper import HealthCheckHelper


class TestCheckErrorRate:
    """Tests for HealthCheckHelper.check_error_rate()."""

    def test_zero_cycles_returns_healthy(self):
        """Zero cycles should return healthy with 'no cycles' message."""
        is_healthy, message = HealthCheckHelper.check_error_rate(errors=0, cycles=0)
        assert is_healthy is True
        assert "No cycles completed yet" in message

    def test_zero_cycles_with_errors_returns_healthy(self):
        """Zero cycles should be healthy even with errors recorded."""
        is_healthy, message = HealthCheckHelper.check_error_rate(errors=5, cycles=0)
        assert is_healthy is True

    def test_zero_errors_is_healthy(self):
        """No errors should always be healthy."""
        is_healthy, message = HealthCheckHelper.check_error_rate(errors=0, cycles=100)
        assert is_healthy is True
        assert "0.0%" in message or "0%" in message

    def test_error_rate_below_threshold_is_healthy(self):
        """Error rate below threshold should be healthy."""
        # 10% error rate, 50% threshold
        is_healthy, message = HealthCheckHelper.check_error_rate(
            errors=10, cycles=100, threshold=0.5
        )
        assert is_healthy is True
        assert "10" in message  # Should show error count

    def test_error_rate_at_threshold_is_unhealthy(self):
        """Error rate at threshold should be unhealthy."""
        # 50% error rate, 50% threshold
        is_healthy, message = HealthCheckHelper.check_error_rate(
            errors=50, cycles=100, threshold=0.5
        )
        assert is_healthy is False
        assert "too high" in message.lower()

    def test_error_rate_above_threshold_is_unhealthy(self):
        """Error rate above threshold should be unhealthy."""
        # 60% error rate, 50% threshold
        is_healthy, message = HealthCheckHelper.check_error_rate(
            errors=60, cycles=100, threshold=0.5
        )
        assert is_healthy is False
        assert "60" in message

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        # 20% error rate, 10% threshold -> unhealthy
        is_healthy, _ = HealthCheckHelper.check_error_rate(
            errors=20, cycles=100, threshold=0.1
        )
        assert is_healthy is False

        # 20% error rate, 30% threshold -> healthy
        is_healthy, _ = HealthCheckHelper.check_error_rate(
            errors=20, cycles=100, threshold=0.3
        )
        assert is_healthy is True

    def test_message_contains_cycle_counts(self):
        """Message should contain error and cycle counts."""
        _, message = HealthCheckHelper.check_error_rate(errors=5, cycles=50)
        assert "5" in message
        assert "50" in message


class TestCheckUptimeGrace:
    """Tests for HealthCheckHelper.check_uptime_grace()."""

    def test_zero_start_time_is_in_grace(self):
        """Zero start time should indicate not started yet."""
        in_grace, message = HealthCheckHelper.check_uptime_grace(start_time=0)
        assert in_grace is True
        assert "Not started" in message

    def test_negative_start_time_is_in_grace(self):
        """Negative start time should indicate not started."""
        in_grace, message = HealthCheckHelper.check_uptime_grace(start_time=-1)
        assert in_grace is True

    def test_recent_start_is_in_grace(self):
        """Recent start should be in grace period."""
        # Started 5 seconds ago, 30 second grace period
        start_time = time.time() - 5
        in_grace, message = HealthCheckHelper.check_uptime_grace(
            start_time=start_time, grace_period=30
        )
        assert in_grace is True
        assert "grace period" in message.lower()

    def test_old_start_not_in_grace(self):
        """Old start should not be in grace period."""
        # Started 60 seconds ago, 30 second grace period
        start_time = time.time() - 60
        in_grace, message = HealthCheckHelper.check_uptime_grace(
            start_time=start_time, grace_period=30
        )
        assert in_grace is False
        assert "uptime" in message.lower()

    def test_custom_grace_period(self):
        """Custom grace period should be respected."""
        start_time = time.time() - 45

        # 30s grace period -> not in grace
        in_grace, _ = HealthCheckHelper.check_uptime_grace(
            start_time=start_time, grace_period=30
        )
        assert in_grace is False

        # 60s grace period -> in grace
        in_grace, _ = HealthCheckHelper.check_uptime_grace(
            start_time=start_time, grace_period=60
        )
        assert in_grace is True


class TestCheckRecentErrors:
    """Tests for HealthCheckHelper.check_recent_errors()."""

    def test_empty_error_log_is_healthy(self):
        """Empty error log should be healthy."""
        is_healthy, message = HealthCheckHelper.check_recent_errors(error_log=[])
        assert is_healthy is True
        assert "No errors" in message

    def test_old_errors_not_counted(self):
        """Errors outside the window should not be counted."""
        now = time.time()
        # Errors from 10 minutes ago (outside 5-minute window)
        old_errors = [now - 600, now - 650, now - 700]
        is_healthy, message = HealthCheckHelper.check_recent_errors(
            error_log=old_errors, window_seconds=300, limit=5
        )
        assert is_healthy is True
        assert "0" in message  # Zero recent errors

    def test_recent_errors_counted(self):
        """Recent errors within window should be counted."""
        now = time.time()
        # 3 errors in last minute
        recent_errors = [now - 10, now - 30, now - 50]
        is_healthy, message = HealthCheckHelper.check_recent_errors(
            error_log=recent_errors, window_seconds=300, limit=5
        )
        assert is_healthy is True
        assert "3" in message

    def test_too_many_recent_errors_unhealthy(self):
        """Too many recent errors should be unhealthy."""
        now = time.time()
        # 6 errors in last minute, limit is 5
        errors = [now - i * 10 for i in range(6)]
        is_healthy, message = HealthCheckHelper.check_recent_errors(
            error_log=errors, window_seconds=300, limit=5
        )
        assert is_healthy is False
        assert "Too many" in message

    def test_mixed_old_and_new_errors(self):
        """Only recent errors should count, not old ones."""
        now = time.time()
        # 2 recent + 10 old errors
        errors = [now - 10, now - 20]  # Recent
        errors.extend([now - 400 - i for i in range(10)])  # Old (outside 5min window)
        is_healthy, message = HealthCheckHelper.check_recent_errors(
            error_log=errors, window_seconds=300, limit=5
        )
        assert is_healthy is True


class TestCheckQueueDepth:
    """Tests for HealthCheckHelper.check_queue_depth()."""

    def test_empty_queue_is_healthy(self):
        """Empty queue should be healthy."""
        is_healthy, message = HealthCheckHelper.check_queue_depth(
            queue_size=0, max_depth=1000
        )
        assert is_healthy is True
        assert "0%" in message or "0/" in message

    def test_partial_queue_is_healthy(self):
        """Partially full queue should be healthy."""
        is_healthy, message = HealthCheckHelper.check_queue_depth(
            queue_size=500, max_depth=1000
        )
        assert is_healthy is True
        assert "50%" in message

    def test_full_queue_is_unhealthy(self):
        """Full queue should be unhealthy."""
        is_healthy, message = HealthCheckHelper.check_queue_depth(
            queue_size=1000, max_depth=1000
        )
        assert is_healthy is False
        assert "too high" in message.lower()

    def test_overfull_queue_is_unhealthy(self):
        """Overfull queue should be unhealthy."""
        is_healthy, message = HealthCheckHelper.check_queue_depth(
            queue_size=1500, max_depth=1000
        )
        assert is_healthy is False

    def test_zero_max_depth(self):
        """Zero max depth should be treated as no limit."""
        is_healthy, message = HealthCheckHelper.check_queue_depth(
            queue_size=0, max_depth=0
        )
        # max_depth=0 means no limit, so always healthy
        assert is_healthy is True
        assert "no limit" in message


class TestCheckLastActivity:
    """Tests for HealthCheckHelper.check_last_activity()."""

    def test_zero_last_activity_is_healthy(self):
        """Zero last activity (not started) should be healthy."""
        is_healthy, message = HealthCheckHelper.check_last_activity(last_activity=0)
        assert is_healthy is True
        assert "No activity recorded" in message

    def test_negative_last_activity_is_healthy(self):
        """Negative last activity should be handled."""
        is_healthy, message = HealthCheckHelper.check_last_activity(last_activity=-1)
        assert is_healthy is True

    def test_recent_activity_is_healthy(self):
        """Recent activity should be healthy."""
        # Activity 5 minutes ago, threshold is 1 hour
        last_activity = time.time() - 300
        is_healthy, message = HealthCheckHelper.check_last_activity(
            last_activity=last_activity, stale_threshold=3600
        )
        assert is_healthy is True
        assert "ago" in message.lower()

    def test_stale_activity_is_unhealthy(self):
        """Stale activity should be unhealthy."""
        # Activity 2 hours ago, threshold is 1 hour
        last_activity = time.time() - 7200
        is_healthy, message = HealthCheckHelper.check_last_activity(
            last_activity=last_activity, stale_threshold=3600
        )
        assert is_healthy is False
        assert "Stale" in message

    def test_custom_stale_threshold(self):
        """Custom stale threshold should be respected."""
        last_activity = time.time() - 600  # 10 minutes ago

        # 5 minute threshold -> stale
        is_healthy, _ = HealthCheckHelper.check_last_activity(
            last_activity=last_activity, stale_threshold=300
        )
        assert is_healthy is False

        # 15 minute threshold -> healthy
        is_healthy, _ = HealthCheckHelper.check_last_activity(
            last_activity=last_activity, stale_threshold=900
        )
        assert is_healthy is True


class TestCheckConnectionHealth:
    """Tests for HealthCheckHelper.check_connection_health()."""

    def test_connected_is_healthy(self):
        """Connected state should always be healthy."""
        is_healthy, message = HealthCheckHelper.check_connection_health(
            connected=True, last_success=0
        )
        assert is_healthy is True
        assert "Connected" in message

    def test_never_connected_is_unhealthy(self):
        """Never connected should be unhealthy."""
        is_healthy, message = HealthCheckHelper.check_connection_health(
            connected=False, last_success=0
        )
        assert is_healthy is False
        assert "Never connected" in message

    def test_recent_disconnect_is_healthy(self):
        """Recent disconnect within tolerance should be healthy."""
        # Disconnected 30 seconds ago, threshold is 60 seconds
        last_success = time.time() - 30
        is_healthy, message = HealthCheckHelper.check_connection_health(
            connected=False, last_success=last_success, max_disconnect_time=60
        )
        assert is_healthy is True
        assert "tolerance" in message.lower()

    def test_long_disconnect_is_unhealthy(self):
        """Long disconnect should be unhealthy."""
        # Disconnected 120 seconds ago, threshold is 60 seconds
        last_success = time.time() - 120
        is_healthy, message = HealthCheckHelper.check_connection_health(
            connected=False, last_success=last_success, max_disconnect_time=60
        )
        assert is_healthy is False
        assert "Disconnected" in message


class TestBuildDetails:
    """Tests for HealthCheckHelper.build_details()."""

    def test_empty_returns_empty(self):
        """No args should return empty dict."""
        details = HealthCheckHelper.build_details()
        assert details == {}

    def test_filters_none_values(self):
        """None values should be filtered out."""
        details = HealthCheckHelper.build_details(
            a=1, b=None, c="test", d=None
        )
        assert details == {"a": 1, "c": "test"}

    def test_keeps_falsy_non_none_values(self):
        """Falsy but non-None values should be kept."""
        details = HealthCheckHelper.build_details(
            zero=0, empty_string="", empty_list=[], false=False, none_val=None
        )
        assert "zero" in details
        assert "empty_string" in details
        assert "empty_list" in details
        assert "false" in details
        assert "none_val" not in details

    def test_preserves_all_types(self):
        """All value types should be preserved."""
        details = HealthCheckHelper.build_details(
            int_val=42,
            float_val=3.14,
            str_val="hello",
            list_val=[1, 2, 3],
            dict_val={"nested": "dict"},
        )
        assert details["int_val"] == 42
        assert details["float_val"] == 3.14
        assert details["str_val"] == "hello"
        assert details["list_val"] == [1, 2, 3]
        assert details["dict_val"] == {"nested": "dict"}


class TestCheckHandlerFailures:
    """Tests for HealthCheckHelper.check_handler_failures()."""

    def test_no_handlers_is_healthy(self):
        """No handlers should be healthy."""
        is_healthy, message = HealthCheckHelper.check_handler_failures(
            failed_handlers=0, total_handlers=0
        )
        assert is_healthy is True
        assert "No handlers registered" in message

    def test_no_failures_is_healthy(self):
        """No failures should be healthy."""
        is_healthy, message = HealthCheckHelper.check_handler_failures(
            failed_handlers=0, total_handlers=10
        )
        assert is_healthy is True
        assert "0.0%" in message or "0%" in message

    def test_low_failure_rate_is_healthy(self):
        """Low failure rate should be healthy."""
        is_healthy, message = HealthCheckHelper.check_handler_failures(
            failed_handlers=1, total_handlers=10, threshold=0.2
        )
        assert is_healthy is True
        assert "1/10" in message  # Check for fraction in message

    def test_high_failure_rate_is_unhealthy(self):
        """High failure rate should be unhealthy."""
        is_healthy, message = HealthCheckHelper.check_handler_failures(
            failed_handlers=5, total_handlers=10, threshold=0.2
        )
        assert is_healthy is False
        assert "too high" in message.lower()

    def test_threshold_boundary(self):
        """At threshold should be unhealthy."""
        is_healthy, message = HealthCheckHelper.check_handler_failures(
            failed_handlers=2, total_handlers=10, threshold=0.2
        )
        assert is_healthy is False


class TestCheckSubscriptionHealth:
    """Tests for HealthCheckHelper.check_subscription_health()."""

    def test_not_subscribed_is_unhealthy(self):
        """Not subscribed should be unhealthy."""
        is_healthy, message = HealthCheckHelper.check_subscription_health(
            subscribed=False
        )
        assert is_healthy is False
        assert "not registered" in message

    def test_subscribed_with_no_min_is_healthy(self):
        """Subscribed with no minimum requirement is healthy."""
        is_healthy, message = HealthCheckHelper.check_subscription_health(
            subscribed=True, events_received=0, min_events=0
        )
        assert is_healthy is True
        assert "Subscribed" in message

    def test_subscribed_below_min_is_unhealthy(self):
        """Subscribed but below minimum events is unhealthy."""
        is_healthy, message = HealthCheckHelper.check_subscription_health(
            subscribed=True, events_received=5, min_events=10
        )
        assert is_healthy is False
        assert "Too few events" in message

    def test_subscribed_above_min_is_healthy(self):
        """Subscribed with enough events is healthy."""
        is_healthy, message = HealthCheckHelper.check_subscription_health(
            subscribed=True, events_received=15, min_events=10
        )
        assert is_healthy is True


class TestCheckDataStaleness:
    """Tests for HealthCheckHelper.check_data_staleness()."""

    def test_no_sync_recorded_is_healthy(self):
        """No sync recorded yet is healthy."""
        is_healthy, message = HealthCheckHelper.check_data_staleness(last_sync=0)
        assert is_healthy is True
        assert "No sync recorded" in message

    def test_recent_sync_is_healthy(self):
        """Recent sync is healthy."""
        last_sync = time.time() - 300  # 5 minutes ago
        is_healthy, message = HealthCheckHelper.check_data_staleness(
            last_sync=last_sync, max_age_seconds=3600
        )
        assert is_healthy is True
        assert "fresh" in message.lower()

    def test_stale_sync_is_unhealthy(self):
        """Stale sync is unhealthy."""
        last_sync = time.time() - 7200  # 2 hours ago
        is_healthy, message = HealthCheckHelper.check_data_staleness(
            last_sync=last_sync, max_age_seconds=3600
        )
        assert is_healthy is False
        assert "stale" in message.lower()


class TestCheckPendingItems:
    """Tests for HealthCheckHelper.check_pending_items()."""

    def test_no_pending_is_healthy(self):
        """No pending items is healthy."""
        is_healthy, message = HealthCheckHelper.check_pending_items(
            pending_count=0, max_pending=100
        )
        assert is_healthy is True
        assert "0%" in message or "0/" in message

    def test_some_pending_is_healthy(self):
        """Some pending items within limit is healthy."""
        is_healthy, message = HealthCheckHelper.check_pending_items(
            pending_count=50, max_pending=100
        )
        assert is_healthy is True
        assert "50%" in message

    def test_too_many_pending_is_unhealthy(self):
        """Too many pending items is unhealthy."""
        is_healthy, message = HealthCheckHelper.check_pending_items(
            pending_count=150, max_pending=100
        )
        assert is_healthy is False
        assert "Too many" in message

    def test_no_limit_is_always_healthy(self):
        """No limit (0) is always healthy."""
        is_healthy, message = HealthCheckHelper.check_pending_items(
            pending_count=10000, max_pending=0
        )
        assert is_healthy is True
        assert "no limit" in message


class TestCheckDependencyHealth:
    """Tests for HealthCheckHelper.check_dependency_health()."""

    def test_no_dependencies_is_healthy(self):
        """No dependencies is healthy."""
        is_healthy, message = HealthCheckHelper.check_dependency_health(
            dependencies={}
        )
        assert is_healthy is True
        assert "No dependencies" in message

    def test_all_healthy_is_healthy(self):
        """All healthy dependencies is healthy."""
        is_healthy, message = HealthCheckHelper.check_dependency_health(
            dependencies={"db": True, "cache": True, "api": True}
        )
        assert is_healthy is True
        assert "3 dependencies healthy" in message

    def test_some_unhealthy_is_unhealthy(self):
        """Some unhealthy dependencies is unhealthy."""
        is_healthy, message = HealthCheckHelper.check_dependency_health(
            dependencies={"db": True, "cache": False, "api": True}
        )
        assert is_healthy is False
        assert "cache" in message
        assert "2/3" in message

    def test_all_unhealthy(self):
        """All unhealthy dependencies is unhealthy."""
        is_healthy, message = HealthCheckHelper.check_dependency_health(
            dependencies={"db": False, "cache": False}
        )
        assert is_healthy is False
        assert "db" in message
        assert "cache" in message


class TestCheckMemoryUsage:
    """Tests for HealthCheckHelper.check_memory_usage()."""

    def test_low_memory_is_healthy(self):
        """Low memory usage is healthy."""
        is_healthy, message = HealthCheckHelper.check_memory_usage(
            current_mb=256, threshold_mb=1024
        )
        assert is_healthy is True
        assert "25%" in message

    def test_high_memory_is_unhealthy(self):
        """High memory usage is unhealthy."""
        is_healthy, message = HealthCheckHelper.check_memory_usage(
            current_mb=1500, threshold_mb=1024
        )
        assert is_healthy is False
        assert "too high" in message.lower()

    def test_at_threshold_is_unhealthy(self):
        """At threshold is unhealthy."""
        is_healthy, message = HealthCheckHelper.check_memory_usage(
            current_mb=1024, threshold_mb=1024
        )
        assert is_healthy is False

    def test_no_limit_is_always_healthy(self):
        """No limit (0) is always healthy."""
        is_healthy, message = HealthCheckHelper.check_memory_usage(
            current_mb=10000, threshold_mb=0
        )
        assert is_healthy is True
        assert "no limit" in message


class TestIntegration:
    """Integration tests for typical health check patterns."""

    def test_typical_daemon_health_check(self):
        """Test a typical daemon health check pattern."""
        # Simulate a daemon that has been running
        start_time = time.time() - 120  # Started 2 minutes ago
        errors = 2
        cycles = 100

        # Check error rate first
        is_healthy, msg = HealthCheckHelper.check_error_rate(
            errors=errors, cycles=cycles, threshold=0.5
        )
        assert is_healthy is True

        # Check if past grace period
        in_grace, grace_msg = HealthCheckHelper.check_uptime_grace(
            start_time=start_time, grace_period=30
        )
        assert in_grace is False

        # Build details
        details = HealthCheckHelper.build_details(
            uptime_seconds=time.time() - start_time,
            error_rate=errors / cycles if cycles > 0 else None,
            cycles=cycles,
        )
        assert "uptime_seconds" in details
        assert "error_rate" in details
        assert details["cycles"] == 100

    def test_unhealthy_daemon_detection(self):
        """Test detection of unhealthy daemon."""
        # Simulate daemon with high error rate
        is_healthy, msg = HealthCheckHelper.check_error_rate(
            errors=60, cycles=100, threshold=0.5
        )
        assert is_healthy is False

        # Verify message is descriptive
        assert "60" in msg
        assert "100" in msg
        assert "too high" in msg.lower() or "failed" in msg.lower()

    def test_full_health_check_with_new_methods(self):
        """Test combining new helper methods in a health check."""
        # Check handler failures
        is_healthy, _ = HealthCheckHelper.check_handler_failures(
            failed_handlers=1, total_handlers=10, threshold=0.2
        )
        assert is_healthy is True

        # Check subscriptions
        is_healthy, _ = HealthCheckHelper.check_subscription_health(
            subscribed=True, events_received=50, min_events=10
        )
        assert is_healthy is True

        # Check data staleness
        is_healthy, _ = HealthCheckHelper.check_data_staleness(
            last_sync=time.time() - 300, max_age_seconds=3600
        )
        assert is_healthy is True

        # Check pending items
        is_healthy, _ = HealthCheckHelper.check_pending_items(
            pending_count=25, max_pending=100
        )
        assert is_healthy is True

        # Check dependencies
        is_healthy, _ = HealthCheckHelper.check_dependency_health(
            dependencies={"db": True, "cache": True}
        )
        assert is_healthy is True

        # Check memory
        is_healthy, _ = HealthCheckHelper.check_memory_usage(
            current_mb=512, threshold_mb=1024
        )
        assert is_healthy is True
