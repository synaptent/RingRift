"""Health Check Helper - Reusable health check utility methods.

This module provides static helper methods for common health check patterns
used across coordination modules. Reduces duplication (~350-400 LOC savings)
by centralizing common health check logic.

Created: December 27, 2025
Purpose: Consolidate duplicated health check logic across 34+ modules

Usage:
    from app.coordination.health_check_helper import HealthCheckHelper

    # In your health_check() method:
    def health_check(self) -> HealthCheckResult:
        # Check error rate
        is_healthy, msg = HealthCheckHelper.check_error_rate(
            errors=self._errors,
            cycles=self._cycles,
            threshold=0.5,
        )
        if not is_healthy:
            return HealthCheckResult.unhealthy(msg)

        # Check startup grace period
        in_grace, grace_msg = HealthCheckHelper.check_uptime_grace(
            start_time=self._start_time,
            grace_period=30,
        )
        if in_grace:
            return HealthCheckResult(healthy=True, message=grace_msg)

        # All checks passed
        return HealthCheckResult(healthy=True, message="Healthy")
"""

from __future__ import annotations

import time
from typing import Any

__all__ = [
    "HealthCheckHelper",
]


class HealthCheckHelper:
    """Static helper methods for common health check patterns."""

    @staticmethod
    def check_error_rate(
        errors: int,
        cycles: int,
        threshold: float = 0.5,
    ) -> tuple[bool, str]:
        """Check if error rate is below threshold.

        Args:
            errors: Number of errors encountered
            cycles: Total number of cycles/operations
            threshold: Maximum acceptable error rate (0.0-1.0)

        Returns:
            Tuple of (is_healthy, message)
        """
        if cycles == 0:
            return True, "No cycles completed yet"

        error_rate = errors / cycles
        if error_rate >= threshold:
            return (
                False,
                f"Error rate too high: {error_rate:.1%} >= {threshold:.0%} "
                f"({errors}/{cycles} cycles failed)",
            )
        return True, f"Error rate: {error_rate:.1%} ({errors}/{cycles})"

    @staticmethod
    def check_uptime_grace(
        start_time: float,
        grace_period: int = 30,
    ) -> tuple[bool, str]:
        """Check if still in startup grace period.

        Args:
            start_time: Unix timestamp when component started
            grace_period: Seconds of grace period after startup

        Returns:
            Tuple of (in_grace_period, message)
        """
        if start_time <= 0:
            return True, "Not started yet"

        uptime = time.time() - start_time
        if uptime < grace_period:
            return True, f"In startup grace period ({uptime:.0f}/{grace_period}s)"
        return False, f"Uptime: {uptime:.0f}s"

    @staticmethod
    def check_recent_errors(
        error_log: list[float],
        window_seconds: int = 300,
        limit: int = 5,
    ) -> tuple[bool, str]:
        """Check if recent error count is below limit.

        Args:
            error_log: List of error timestamps (Unix time)
            window_seconds: Time window to consider (default 5 minutes)
            limit: Maximum errors allowed in window

        Returns:
            Tuple of (is_healthy, message)
        """
        if not error_log:
            return True, "No errors recorded"

        cutoff = time.time() - window_seconds
        recent_errors = sum(1 for ts in error_log if ts > cutoff)

        if recent_errors >= limit:
            return (
                False,
                f"Too many recent errors: {recent_errors} in last "
                f"{window_seconds // 60}min (limit: {limit})",
            )
        return True, f"Recent errors: {recent_errors} in last {window_seconds // 60}min"

    @staticmethod
    def check_queue_depth(
        queue_size: int,
        max_depth: int = 1000,
    ) -> tuple[bool, str]:
        """Check if queue depth is below maximum.

        Args:
            queue_size: Current queue size
            max_depth: Maximum acceptable queue depth (0 means no limit)

        Returns:
            Tuple of (is_healthy, message)
        """
        # max_depth <= 0 means no limit
        if max_depth <= 0:
            return True, f"Queue depth: {queue_size} (no limit)"
        if queue_size >= max_depth:
            return False, f"Queue depth too high: {queue_size} >= {max_depth}"
        fill_percent = (queue_size / max_depth) * 100
        return True, f"Queue depth: {queue_size}/{max_depth} ({fill_percent:.0f}%)"

    @staticmethod
    def check_last_activity(
        last_activity: float,
        stale_threshold: int = 3600,
    ) -> tuple[bool, str]:
        """Check if last activity was recent enough.

        Args:
            last_activity: Unix timestamp of last activity
            stale_threshold: Seconds after which activity is considered stale

        Returns:
            Tuple of (is_healthy, message)
        """
        if last_activity <= 0:
            return True, "No activity recorded yet"

        age = time.time() - last_activity
        if age > stale_threshold:
            return (
                False,
                f"Stale: no activity for {age / 60:.0f}min "
                f"(threshold: {stale_threshold // 60}min)",
            )
        return True, f"Last activity: {age:.0f}s ago"

    @staticmethod
    def check_connection_health(
        connected: bool,
        last_success: float,
        max_disconnect_time: int = 60,
    ) -> tuple[bool, str]:
        """Check connection health status.

        Args:
            connected: Whether currently connected
            last_success: Unix timestamp of last successful connection
            max_disconnect_time: Max seconds disconnected before unhealthy

        Returns:
            Tuple of (is_healthy, message)
        """
        if connected:
            return True, "Connected"

        if last_success <= 0:
            return False, "Never connected"

        disconnect_time = time.time() - last_success
        if disconnect_time > max_disconnect_time:
            return (
                False,
                f"Disconnected for {disconnect_time:.0f}s "
                f"(threshold: {max_disconnect_time}s)",
            )
        return True, f"Disconnected {disconnect_time:.0f}s (within tolerance)"

    @staticmethod
    def check_handler_failures(
        failed_handlers: int,
        total_handlers: int,
        threshold: float = 0.2,
    ) -> tuple[bool, str]:
        """Check if event handler failure rate is acceptable.

        Args:
            failed_handlers: Number of handlers that failed
            total_handlers: Total number of handlers registered
            threshold: Maximum acceptable failure rate (0.0-1.0)

        Returns:
            Tuple of (is_healthy, message)
        """
        if total_handlers == 0:
            return True, "No handlers registered"

        failure_rate = failed_handlers / total_handlers
        if failure_rate >= threshold:
            return (
                False,
                f"Handler failure rate too high: {failure_rate:.1%} >= {threshold:.0%} "
                f"({failed_handlers}/{total_handlers} handlers failed)",
            )
        return True, f"Handler failure rate: {failure_rate:.1%} ({failed_handlers}/{total_handlers})"

    @staticmethod
    def check_subscription_health(
        subscribed: bool,
        events_received: int = 0,
        min_events: int = 0,
    ) -> tuple[bool, str]:
        """Check if event subscriptions are active and receiving.

        Args:
            subscribed: Whether subscriptions are registered
            events_received: Number of events received
            min_events: Minimum events expected (0 means no check)

        Returns:
            Tuple of (is_healthy, message)
        """
        if not subscribed:
            return False, "Event subscriptions not registered"

        if min_events > 0 and events_received < min_events:
            return (
                False,
                f"Too few events received: {events_received} < {min_events} expected",
            )
        return True, f"Subscribed, {events_received} events received"

    @staticmethod
    def check_data_staleness(
        last_sync: float,
        max_age_seconds: float = 3600.0,
    ) -> tuple[bool, str]:
        """Check if data is within acceptable age.

        Args:
            last_sync: Unix timestamp of last sync/update
            max_age_seconds: Maximum acceptable age in seconds

        Returns:
            Tuple of (is_healthy, message)
        """
        if last_sync <= 0:
            return True, "No sync recorded yet"

        age = time.time() - last_sync
        if age > max_age_seconds:
            age_mins = age / 60
            max_mins = max_age_seconds / 60
            return (
                False,
                f"Data stale: last sync {age_mins:.0f}min ago (max: {max_mins:.0f}min)",
            )
        return True, f"Data fresh: synced {age:.0f}s ago"

    @staticmethod
    def check_pending_items(
        pending_count: int,
        max_pending: int = 100,
    ) -> tuple[bool, str]:
        """Check if pending work queue is manageable.

        Args:
            pending_count: Number of pending items
            max_pending: Maximum acceptable pending items (0 means no limit)

        Returns:
            Tuple of (is_healthy, message)
        """
        if max_pending <= 0:
            return True, f"Pending items: {pending_count} (no limit)"

        if pending_count >= max_pending:
            return (
                False,
                f"Too many pending items: {pending_count} >= {max_pending}",
            )
        fill_percent = (pending_count / max_pending) * 100
        return True, f"Pending items: {pending_count}/{max_pending} ({fill_percent:.0f}%)"

    @staticmethod
    def check_dependency_health(
        dependencies: dict[str, bool],
    ) -> tuple[bool, str]:
        """Aggregate health from multiple dependencies.

        Args:
            dependencies: Dict of dependency_name -> is_healthy

        Returns:
            Tuple of (is_healthy, message)
        """
        if not dependencies:
            return True, "No dependencies"

        unhealthy = [name for name, healthy in dependencies.items() if not healthy]
        total = len(dependencies)
        healthy_count = total - len(unhealthy)

        if unhealthy:
            return (
                False,
                f"Unhealthy dependencies ({healthy_count}/{total}): {', '.join(unhealthy)}",
            )
        return True, f"All {total} dependencies healthy"

    @staticmethod
    def check_memory_usage(
        current_mb: float,
        threshold_mb: float = 1024.0,
    ) -> tuple[bool, str]:
        """Check memory usage against threshold.

        Args:
            current_mb: Current memory usage in MB
            threshold_mb: Maximum acceptable memory in MB

        Returns:
            Tuple of (is_healthy, message)
        """
        if threshold_mb <= 0:
            return True, f"Memory usage: {current_mb:.0f}MB (no limit)"

        if current_mb >= threshold_mb:
            return (
                False,
                f"Memory usage too high: {current_mb:.0f}MB >= {threshold_mb:.0f}MB",
            )
        usage_percent = (current_mb / threshold_mb) * 100
        return True, f"Memory usage: {current_mb:.0f}/{threshold_mb:.0f}MB ({usage_percent:.0f}%)"

    @staticmethod
    def build_details(
        **metrics: Any,
    ) -> dict[str, Any]:
        """Build a details dict for HealthCheckResult, filtering None values.

        Args:
            **metrics: Key-value pairs to include in details

        Returns:
            Dict with None values filtered out
        """
        return {k: v for k, v in metrics.items() if v is not None}
