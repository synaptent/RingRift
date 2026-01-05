"""Daemon health analysis types.

Provides unified failure classification for daemon health checks.
Integrates with HealthCheckResult without modifying the existing interface.

Sprint 17.9+ (Jan 5, 2026): Initial implementation.

Usage:
    from app.coordination.daemon_health_types import (
        FailureCategory,
        DaemonFailurePattern,
        AnalyzerConfig,
        AnalysisResult,
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "FailureCategory",
    "DaemonFailurePattern",
    "AnalyzerConfig",
    "AnalysisResult",
]


class FailureCategory(Enum):
    """Classification of daemon health check failures.

    Categories are ordered by severity (TRANSIENT < DEGRADED < PERSISTENT < CRITICAL).
    Used by DaemonHealthAnalyzer to classify failures based on patterns.
    """

    TRANSIENT = "transient"  # Single failure, likely to self-recover
    DEGRADED = "degraded"  # Partial functionality, can continue with reduced capacity
    PERSISTENT = "persistent"  # Repeated failures (3+), needs intervention
    CRITICAL = "critical"  # Complete failure (5+) or ERROR status, immediate action needed


@dataclass
class DaemonFailurePattern:
    """Tracks failure patterns for a single daemon.

    Used by DaemonHealthAnalyzer to classify failures based on history.
    Thread-safe updates are handled by the analyzer.
    """

    daemon_name: str
    consecutive_failures: int = 0
    total_failures: int = 0
    total_checks: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    failure_messages: list[str] = field(default_factory=list)
    recovery_attempts: int = 0
    last_category: FailureCategory | None = None

    # Sliding window for frequency calculation (timestamps of recent failures)
    failure_timestamps: list[float] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Failure rate as percentage (0.0-1.0)."""
        if self.total_checks == 0:
            return 0.0
        return self.total_failures / self.total_checks

    @property
    def time_since_last_failure(self) -> float:
        """Seconds since last failure."""
        if self.last_failure_time == 0:
            return float("inf")
        return time.time() - self.last_failure_time

    @property
    def time_since_last_success(self) -> float:
        """Seconds since last success."""
        if self.last_success_time == 0:
            return float("inf")
        return time.time() - self.last_success_time

    def recent_failure_count(self, window_seconds: float = 300.0) -> int:
        """Count failures in the last N seconds (default 5 minutes)."""
        cutoff = time.time() - window_seconds
        return sum(1 for ts in self.failure_timestamps if ts >= cutoff)

    def record_check(self, healthy: bool, message: str = "") -> None:
        """Record a health check result.

        Args:
            healthy: Whether the health check passed
            message: Optional failure message for unhealthy checks
        """
        self.total_checks += 1
        now = time.time()

        if healthy:
            self.consecutive_failures = 0
            self.last_success_time = now
        else:
            self.consecutive_failures += 1
            self.total_failures += 1
            self.last_failure_time = now
            self.failure_timestamps.append(now)
            if message:
                # Keep last 10 messages
                self.failure_messages = (self.failure_messages + [message])[-10:]

        # Prune old timestamps (keep last hour)
        cutoff = now - 3600
        self.failure_timestamps = [ts for ts in self.failure_timestamps if ts >= cutoff]

    def reset(self) -> None:
        """Reset pattern state (e.g., after daemon restart)."""
        self.consecutive_failures = 0
        self.total_failures = 0
        self.total_checks = 0
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
        self.failure_messages.clear()
        self.recovery_attempts = 0
        self.last_category = None
        self.failure_timestamps.clear()


@dataclass
class AnalyzerConfig:
    """Configuration for DaemonHealthAnalyzer thresholds.

    Can be loaded from environment variables via from_env().
    """

    # Consecutive failures to escalate categories
    transient_threshold: int = 1  # 1 failure = transient
    persistent_threshold: int = 3  # 3+ consecutive = persistent
    critical_threshold: int = 5  # 5+ consecutive = critical

    # Time-based thresholds (seconds)
    transient_window: float = 60.0  # Failures within 1 min may be transient
    persistent_window: float = 300.0  # Failures spanning 5 min = persistent

    # Frequency thresholds
    high_frequency_threshold: int = 5  # 5+ failures in 5 min = concerning

    # Status-based classification keywords
    degraded_statuses: tuple[str, ...] = ("degraded", "draining", "paused")
    critical_statuses: tuple[str, ...] = ("error", "stopped")

    @classmethod
    def from_env(cls) -> AnalyzerConfig:
        """Create config from environment variables."""
        import os

        return cls(
            transient_threshold=int(
                os.environ.get("RINGRIFT_FAILURE_TRANSIENT_THRESHOLD", 1)
            ),
            persistent_threshold=int(
                os.environ.get("RINGRIFT_FAILURE_PERSISTENT_THRESHOLD", 3)
            ),
            critical_threshold=int(
                os.environ.get("RINGRIFT_FAILURE_CRITICAL_THRESHOLD", 5)
            ),
            transient_window=float(
                os.environ.get("RINGRIFT_FAILURE_TRANSIENT_WINDOW", 60.0)
            ),
            persistent_window=float(
                os.environ.get("RINGRIFT_FAILURE_PERSISTENT_WINDOW", 300.0)
            ),
            high_frequency_threshold=int(
                os.environ.get("RINGRIFT_FAILURE_HIGH_FREQUENCY_THRESHOLD", 5)
            ),
        )


@dataclass
class AnalysisResult:
    """Result of analyzing a health check.

    Returned by DaemonHealthAnalyzer.analyze() with classification and recommendations.
    """

    daemon_name: str
    category: FailureCategory
    pattern: DaemonFailurePattern
    recommended_action: str
    should_emit_event: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Whether the last check was healthy (category is TRANSIENT with 0 failures)."""
        return self.pattern.consecutive_failures == 0

    @property
    def needs_intervention(self) -> bool:
        """Whether manual intervention may be needed."""
        return self.category in (FailureCategory.PERSISTENT, FailureCategory.CRITICAL)
