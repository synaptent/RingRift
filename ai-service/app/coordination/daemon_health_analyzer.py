"""Daemon Health Analyzer - Unified failure classification system.

Analyzes HealthCheckResult outputs to classify failures into categories
(TRANSIENT, PERSISTENT, CRITICAL, DEGRADED) and track failure patterns.

Integrates with:
- DaemonManager health loop (consumer of classifications)
- Circuit breaker infrastructure (NodeCircuitBreaker, DaemonStatusCircuitBreaker)
- Event router (emits DAEMON_FAILURE_CLASSIFIED events)

Sprint 17.9+ (Jan 5, 2026): Initial implementation.

Usage:
    from app.coordination.daemon_health_analyzer import (
        get_daemon_health_analyzer,
        DaemonHealthAnalyzer,
        reset_daemon_health_analyzer,
    )

    analyzer = get_daemon_health_analyzer()
    result = analyzer.analyze(daemon_name, health_check_result)
    print(f"Category: {result.category}, Action: {result.recommended_action}")
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from app.coordination.daemon_health_types import (
    AnalysisResult,
    AnalyzerConfig,
    DaemonFailurePattern,
    FailureCategory,
)
from app.coordination.singleton_mixin import SingletonMixin

if TYPE_CHECKING:
    from app.coordination.contracts import HealthCheckResult

__all__ = [
    "DaemonHealthAnalyzer",
    "get_daemon_health_analyzer",
    "reset_daemon_health_analyzer",
]

logger = logging.getLogger(__name__)


class DaemonHealthAnalyzer(SingletonMixin):
    """Unified failure classification system for daemon health checks.

    Analyzes HealthCheckResult outputs and classifies failures based on:
    - Consecutive failure count
    - Failure frequency (failures per time window)
    - Status field from HealthCheckResult
    - Historical patterns

    Thread-safe via internal lock for pattern state updates.

    Example:
        >>> analyzer = DaemonHealthAnalyzer.get_instance()
        >>> result = analyzer.analyze("my_daemon", health_result)
        >>> if result.category == FailureCategory.CRITICAL:
        ...     await trigger_recovery(result.daemon_name)
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        """Initialize the analyzer.

        Args:
            config: Optional configuration. Defaults to env-based config.
        """
        self._config = config or AnalyzerConfig.from_env()
        self._patterns: dict[str, DaemonFailurePattern] = {}
        self._lock = threading.RLock()
        self._initialized = True

    def analyze(
        self,
        daemon_name: str,
        health_result: dict[str, Any] | "HealthCheckResult",
    ) -> AnalysisResult:
        """Analyze a health check result and classify the failure.

        Args:
            daemon_name: Name of the daemon being checked
            health_result: Result from daemon's health_check() method.
                          Can be a dict or HealthCheckResult instance.

        Returns:
            AnalysisResult with category, pattern, and recommended action
        """
        with self._lock:
            pattern = self._get_or_create_pattern(daemon_name)

            # Normalize health result to dict
            if hasattr(health_result, "healthy"):
                healthy = health_result.healthy
                message = getattr(health_result, "message", "")
                status = getattr(health_result, "status", None)
                if status is not None:
                    status_value = status.value if hasattr(status, "value") else str(status)
                else:
                    status_value = None
            else:
                healthy = health_result.get("healthy", True)
                message = health_result.get("message", "")
                status_value = health_result.get("status")

            # Record the check
            pattern.record_check(healthy, message)

            if healthy:
                # Success - check if this is a recovery
                was_failing = pattern.last_category in (
                    FailureCategory.PERSISTENT,
                    FailureCategory.CRITICAL,
                )
                category = FailureCategory.TRANSIENT
                action = "none"
                should_emit = was_failing  # Emit recovery event

                if was_failing:
                    logger.info(
                        f"[HealthAnalyzer] {daemon_name} recovered from {pattern.last_category.value}"
                    )
            else:
                # Failure - classify based on patterns
                category = self._classify_failure(pattern, status_value)
                action = self._get_recommended_action(category, pattern)
                should_emit = self._should_emit_event(category, pattern)

            pattern.last_category = category

            return AnalysisResult(
                daemon_name=daemon_name,
                category=category,
                pattern=pattern,
                recommended_action=action,
                should_emit_event=should_emit,
                details={
                    "consecutive_failures": pattern.consecutive_failures,
                    "failure_rate": round(pattern.failure_rate, 3),
                    "recent_failures": pattern.recent_failure_count(300),
                    "status": status_value,
                    "message": message[:100] if message else "",
                },
            )

    def _get_or_create_pattern(self, daemon_name: str) -> DaemonFailurePattern:
        """Get or create failure pattern for a daemon."""
        if daemon_name not in self._patterns:
            self._patterns[daemon_name] = DaemonFailurePattern(daemon_name=daemon_name)
        return self._patterns[daemon_name]

    def _classify_failure(
        self,
        pattern: DaemonFailurePattern,
        status_value: str | None,
    ) -> FailureCategory:
        """Classify a failure based on patterns and health result.

        Classification logic:
        1. Check status field first (ERROR/STOPPED = CRITICAL)
        2. Check consecutive failures (escalating severity)
        3. Check failure frequency (high frequency = PERSISTENT)
        4. Default to TRANSIENT for single/new failures
        """
        config = self._config

        # 1. Status-based classification (highest priority)
        if status_value:
            status_lower = status_value.lower()
            if status_lower in config.critical_statuses:
                return FailureCategory.CRITICAL
            if status_lower in config.degraded_statuses:
                return FailureCategory.DEGRADED

        # 2. Consecutive failures
        if pattern.consecutive_failures >= config.critical_threshold:
            return FailureCategory.CRITICAL
        if pattern.consecutive_failures >= config.persistent_threshold:
            return FailureCategory.PERSISTENT

        # 3. High frequency failures
        recent = pattern.recent_failure_count(config.persistent_window)
        if recent >= config.high_frequency_threshold:
            return FailureCategory.PERSISTENT

        # 4. Default: transient
        return FailureCategory.TRANSIENT

    def _get_recommended_action(
        self,
        category: FailureCategory,
        pattern: DaemonFailurePattern,
    ) -> str:
        """Get recommended action based on failure category."""
        actions = {
            FailureCategory.TRANSIENT: "monitor",
            FailureCategory.DEGRADED: "log_warning",
            FailureCategory.PERSISTENT: "restart_daemon",
            FailureCategory.CRITICAL: "escalate_to_recovery",
        }
        return actions.get(category, "monitor")

    def _should_emit_event(
        self,
        category: FailureCategory,
        pattern: DaemonFailurePattern,
    ) -> bool:
        """Determine if an event should be emitted for this failure.

        Emit events when:
        - Category changes from previous check
        - First CRITICAL failure
        - Transitioning from healthy to PERSISTENT
        """
        # Always emit for CRITICAL
        if category == FailureCategory.CRITICAL:
            return True

        # Emit on category escalation
        if pattern.last_category is not None and pattern.last_category != category:
            # Compare severity (enum order is TRANSIENT < DEGRADED < PERSISTENT < CRITICAL)
            category_order = list(FailureCategory)
            if category_order.index(category) > category_order.index(pattern.last_category):
                return True

        # Emit on first PERSISTENT
        if category == FailureCategory.PERSISTENT:
            if pattern.consecutive_failures == self._config.persistent_threshold:
                return True

        return False

    def get_pattern(self, daemon_name: str) -> DaemonFailurePattern | None:
        """Get the failure pattern for a daemon (read-only).

        Args:
            daemon_name: Name of the daemon

        Returns:
            DaemonFailurePattern if exists, None otherwise
        """
        with self._lock:
            return self._patterns.get(daemon_name)

    def get_all_patterns(self) -> dict[str, DaemonFailurePattern]:
        """Get all failure patterns (read-only copy).

        Returns:
            Dict of daemon_name -> DaemonFailurePattern
        """
        with self._lock:
            return dict(self._patterns)

    def reset_pattern(self, daemon_name: str) -> None:
        """Reset the failure pattern for a daemon.

        Args:
            daemon_name: Name of the daemon to reset
        """
        with self._lock:
            if daemon_name in self._patterns:
                del self._patterns[daemon_name]

    def clear_all_patterns(self) -> None:
        """Clear all failure patterns."""
        with self._lock:
            self._patterns.clear()

    def get_failing_daemons(
        self,
        min_category: FailureCategory = FailureCategory.PERSISTENT,
    ) -> list[tuple[str, DaemonFailurePattern]]:
        """Get daemons with failures at or above the specified category.

        Args:
            min_category: Minimum failure category to include

        Returns:
            List of (daemon_name, pattern) tuples
        """
        category_order = list(FailureCategory)
        min_index = category_order.index(min_category)

        with self._lock:
            result = []
            for name, pattern in self._patterns.items():
                if pattern.last_category is not None:
                    cat_index = category_order.index(pattern.last_category)
                    if cat_index >= min_index:
                        result.append((name, pattern))
            return result

    def get_critical_daemons(self) -> list[str]:
        """Get names of daemons currently in CRITICAL state.

        Returns:
            List of daemon names with CRITICAL failures
        """
        with self._lock:
            return [
                name
                for name, pattern in self._patterns.items()
                if pattern.last_category == FailureCategory.CRITICAL
            ]


# Module-level singleton accessor
_instance: DaemonHealthAnalyzer | None = None
_instance_lock = threading.Lock()


def get_daemon_health_analyzer(
    config: AnalyzerConfig | None = None,
) -> DaemonHealthAnalyzer:
    """Get the singleton DaemonHealthAnalyzer instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The singleton DaemonHealthAnalyzer instance
    """
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = DaemonHealthAnalyzer(config)
        return _instance


def reset_daemon_health_analyzer() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance
    with _instance_lock:
        _instance = None
