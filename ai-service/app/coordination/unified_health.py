"""Unified Health Monitor - Correlated failure tracking across all layers.

January 2026: Created as part of distributed architecture Phase 3.

Problem:
    Three independent health check systems don't share context:
    - Gossip (15s heartbeat)
    - Circuit breaker (60s timeout)
    - Staleness (1hr threshold)

    When a failure occurs, it takes multiple systems 20+ minutes to detect
    and correlate the issue. Cascade detection is delayed.

Solution:
    Unified failure tracking that correlates events across all systems.
    - Central failure context repository
    - Cascade detection (5+ correlated failures = cascade)
    - Periodic health assessment with correlation scoring
    - Unified alerts and recovery recommendations

Usage:
    from app.coordination.unified_health import (
        UnifiedHealthMonitor,
        get_unified_health_monitor,
        FailureContext,
    )

    monitor = get_unified_health_monitor()

    # Record a failure
    await monitor.record_failure(FailureContext(
        node_id="lambda-gh200-1",
        transport="tailscale",
        operation="heartbeat",
        error_type="timeout",
    ))

    # Get cluster health assessment
    assessment = await monitor.assess_cluster_health()
    if assessment.severity == "critical":
        await monitor.recommend_recovery_actions(assessment)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Failure severity levels."""

    TRANSIENT = 1  # Temporary, likely to self-recover
    DEGRADATION = 2  # Service degraded but operational
    CASCADE = 3  # Multiple correlated failures


class HealthLevel(Enum):
    """Overall cluster health level."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class FailureContext:
    """Unified failure tracking context."""

    node_id: str
    transport: str  # tailscale, ssh, http, relay
    operation: str  # heartbeat, sync, training, etc.
    error_type: str  # timeout, connection_refused, etc.
    timestamp: float = field(default_factory=time.time)
    severity: Severity = Severity.TRANSIENT
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Get age of this failure in seconds."""
        return time.time() - self.timestamp

    def is_correlated_with(self, other: FailureContext) -> bool:
        """Check if this failure is correlated with another.

        Failures are correlated if they:
        - Happened within 60 seconds of each other
        - Share the same transport or operation
        """
        time_diff = abs(self.timestamp - other.timestamp)
        if time_diff > 60:
            return False

        return self.transport == other.transport or self.operation == other.operation


@dataclass
class HealthAssessment:
    """Result of cluster health assessment."""

    healthy: bool
    level: HealthLevel
    severity: str
    failures: list[FailureContext]
    correlation_score: float  # 0.0 - 1.0, higher = more correlated failures
    summary: str
    recommendations: list[str] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)


@dataclass
class UnifiedHealthConfig:
    """Configuration for unified health monitor."""

    # Failure history settings
    max_history_size: int = 1000
    history_ttl_seconds: float = 3600.0  # 1 hour

    # Cascade detection
    cascade_threshold: int = 5  # 5 correlated failures = cascade
    cascade_window_seconds: float = 60.0

    # Health check settings
    check_interval_seconds: float = 30.0
    include_p2p_health: bool = True
    include_sync_health: bool = True
    include_training_health: bool = True
    include_resource_health: bool = True


class UnifiedHealthMonitor:
    """Correlated health monitoring across all layers.

    Centralizes failure tracking to detect cascades faster than
    independent health check systems.
    """

    def __init__(self, config: UnifiedHealthConfig | None = None):
        """Initialize unified health monitor."""
        self.config = config or UnifiedHealthConfig()

        # Failure history
        self._failure_history: deque[FailureContext] = deque(maxlen=self.config.max_history_size)

        # Cascade tracking
        self._cascade_alerts: list[dict[str, Any]] = []
        self._last_cascade_alert: float = 0.0

        # Assessment cache
        self._last_assessment: HealthAssessment | None = None
        self._last_assessment_time: float = 0.0

        # Node-level tracking
        self._node_failure_counts: dict[str, int] = {}
        self._node_last_failure: dict[str, float] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def record_failure(self, ctx: FailureContext) -> None:
        """Record failure with correlation analysis.

        Args:
            ctx: Failure context to record
        """
        async with self._lock:
            self._failure_history.append(ctx)

            # Update node-level tracking
            self._node_failure_counts[ctx.node_id] = (
                self._node_failure_counts.get(ctx.node_id, 0) + 1
            )
            self._node_last_failure[ctx.node_id] = ctx.timestamp

            # Check for cascade
            await self._check_for_cascade(ctx)

            logger.debug(
                f"Recorded failure: {ctx.node_id}/{ctx.transport}/{ctx.operation} "
                f"({ctx.error_type})"
            )

    async def _check_for_cascade(self, new_failure: FailureContext) -> None:
        """Check if recent failures indicate a cascade."""
        recent = [
            f
            for f in self._failure_history
            if f.age_seconds < self.config.cascade_window_seconds
        ]

        if len(recent) >= self.config.cascade_threshold:
            # Check correlation
            correlated = [
                f for f in recent if f.is_correlated_with(new_failure)
            ]

            if len(correlated) >= self.config.cascade_threshold:
                # Cascade detected!
                new_failure.severity = Severity.CASCADE
                await self._emit_cascade_alert(correlated)

    async def _emit_cascade_alert(self, failures: list[FailureContext]) -> None:
        """Emit cascade failure alert."""
        # Rate limit alerts
        if time.time() - self._last_cascade_alert < 60:
            return

        self._last_cascade_alert = time.time()

        # Group failures
        nodes = set(f.node_id for f in failures)
        transports = set(f.transport for f in failures)
        operations = set(f.operation for f in failures)

        alert = {
            "type": "cascade",
            "failure_count": len(failures),
            "nodes_affected": list(nodes),
            "transports_affected": list(transports),
            "operations_affected": list(operations),
            "timestamp": time.time(),
        }

        self._cascade_alerts.append(alert)

        logger.warning(
            f"CASCADE DETECTED: {len(failures)} correlated failures across "
            f"{len(nodes)} nodes, {len(transports)} transports"
        )

        # Emit event
        try:
            from app.coordination.event_router import emit_event
            from app.distributed.data_events import DataEventType

            emit_event(
                DataEventType.CLUSTER_DEGRADED,
                {
                    "reason": "cascade_failure",
                    "failure_count": len(failures),
                    "nodes_affected": list(nodes),
                    "source": "unified_health",
                },
            )
        except ImportError:
            pass

    async def assess_cluster_health(self) -> HealthAssessment:
        """Perform correlated health assessment across all systems.

        Returns:
            Health assessment with severity and recommendations
        """
        checks = await asyncio.gather(
            self._check_p2p_health(),
            self._check_sync_health(),
            self._check_training_health(),
            self._check_resource_health(),
            return_exceptions=True,
        )

        failures: list[FailureContext] = []
        for check in checks:
            if isinstance(check, Exception):
                failures.append(
                    FailureContext(
                        node_id="local",
                        transport="internal",
                        operation="health_check",
                        error_type=str(type(check).__name__),
                    )
                )
            elif isinstance(check, list):
                failures.extend(check)

        # Calculate correlation score
        correlation_score = self._calculate_correlation_score(failures)

        # Determine severity
        if len(failures) == 0:
            level = HealthLevel.HEALTHY
            severity = "ok"
        elif len(failures) >= 5 or correlation_score > 0.5:
            level = HealthLevel.CRITICAL
            severity = "critical"
        elif len(failures) >= 2:
            level = HealthLevel.WARNING
            severity = "warning"
        else:
            level = HealthLevel.WARNING
            severity = "warning"

        # Build summary
        if level == HealthLevel.HEALTHY:
            summary = "Cluster is healthy"
        else:
            nodes_affected = len(set(f.node_id for f in failures))
            summary = f"{len(failures)} issues across {nodes_affected} nodes"

        # Build recommendations
        recommendations = self._get_recommendations(failures, correlation_score)

        assessment = HealthAssessment(
            healthy=level == HealthLevel.HEALTHY,
            level=level,
            severity=severity,
            failures=failures,
            correlation_score=correlation_score,
            summary=summary,
            recommendations=recommendations,
        )

        self._last_assessment = assessment
        self._last_assessment_time = time.time()

        return assessment

    def _calculate_correlation_score(self, failures: list[FailureContext]) -> float:
        """Calculate correlation score for failures.

        Returns:
            Score 0.0-1.0 where higher means more correlated
        """
        if len(failures) < 2:
            return 0.0

        # Count correlated pairs
        correlated_pairs = 0
        total_pairs = 0

        for i, f1 in enumerate(failures):
            for f2 in failures[i + 1 :]:
                total_pairs += 1
                if f1.is_correlated_with(f2):
                    correlated_pairs += 1

        if total_pairs == 0:
            return 0.0

        return correlated_pairs / total_pairs

    async def _check_p2p_health(self) -> list[FailureContext]:
        """Check P2P cluster health."""
        failures: list[FailureContext] = []

        if not self.config.include_p2p_health:
            return failures

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:8770/status",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        failures.append(
                            FailureContext(
                                node_id="local",
                                transport="http",
                                operation="p2p_status",
                                error_type="http_error",
                                details={"status_code": resp.status},
                            )
                        )
                    else:
                        data = await resp.json()
                        # Check for unhealthy conditions
                        quorum_ok = data.get("quorum_ok", False)
                        alive_peers = data.get("alive_peers", 0)

                        if not quorum_ok:
                            failures.append(
                                FailureContext(
                                    node_id="cluster",
                                    transport="p2p",
                                    operation="quorum",
                                    error_type="quorum_lost",
                                    severity=Severity.CASCADE,
                                    details={"alive_peers": alive_peers},
                                )
                            )

        except asyncio.TimeoutError:
            failures.append(
                FailureContext(
                    node_id="local",
                    transport="http",
                    operation="p2p_status",
                    error_type="timeout",
                )
            )
        except Exception as e:
            failures.append(
                FailureContext(
                    node_id="local",
                    transport="http",
                    operation="p2p_status",
                    error_type=str(type(e).__name__),
                )
            )

        return failures

    async def _check_sync_health(self) -> list[FailureContext]:
        """Check data sync health."""
        failures: list[FailureContext] = []

        if not self.config.include_sync_health:
            return failures

        # Check recent sync failures from circuit breaker
        try:
            from app.coordination.transport_circuit_breaker import (
                get_transport_circuit_breaker,
            )

            breaker = get_transport_circuit_breaker()
            summary = breaker.get_summary()

            if summary["nodes_excluded"] > 0:
                for node_id in summary["excluded_nodes"]:
                    failures.append(
                        FailureContext(
                            node_id=node_id,
                            transport="multiple",
                            operation="sync",
                            error_type="all_transports_failed",
                            severity=Severity.DEGRADATION,
                        )
                    )

        except ImportError:
            pass

        return failures

    async def _check_training_health(self) -> list[FailureContext]:
        """Check training pipeline health."""
        failures: list[FailureContext] = []

        if not self.config.include_training_health:
            return failures

        # Check for stalled training
        try:
            from app.coordination.daemon_manager import get_daemon_manager
            from app.coordination.daemon_types import DaemonType

            dm = get_daemon_manager()
            health = dm.get_daemon_health(DaemonType.TRAINING_COORDINATOR)

            if health and not health.get("healthy", True):
                failures.append(
                    FailureContext(
                        node_id="local",
                        transport="internal",
                        operation="training",
                        error_type="daemon_unhealthy",
                        details=health,
                    )
                )

        except ImportError:
            pass

        return failures

    async def _check_resource_health(self) -> list[FailureContext]:
        """Check resource usage health."""
        failures: list[FailureContext] = []

        if not self.config.include_resource_health:
            return failures

        try:
            import psutil

            # Check disk space
            disk = psutil.disk_usage("/")
            if disk.percent > 90:  # DISK_CRITICAL_PERCENT from app.config.thresholds
                failures.append(
                    FailureContext(
                        node_id="local",
                        transport="internal",
                        operation="disk",
                        error_type="disk_full",
                        severity=Severity.DEGRADATION,
                        details={"percent_used": disk.percent},
                    )
                )

            # Check memory
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                failures.append(
                    FailureContext(
                        node_id="local",
                        transport="internal",
                        operation="memory",
                        error_type="memory_pressure",
                        severity=Severity.DEGRADATION,
                        details={"percent_used": mem.percent},
                    )
                )

        except ImportError:
            pass

        return failures

    def _get_recommendations(
        self, failures: list[FailureContext], correlation_score: float
    ) -> list[str]:
        """Generate recovery recommendations based on failures."""
        recommendations: list[str] = []

        if not failures:
            return recommendations

        # Check for cascade
        if correlation_score > 0.5:
            recommendations.append(
                "CASCADE DETECTED: Check network connectivity and restart P2P orchestrator"
            )

        # Check for transport-specific issues
        transport_counts: dict[str, int] = {}
        for f in failures:
            transport_counts[f.transport] = transport_counts.get(f.transport, 0) + 1

        for transport, count in transport_counts.items():
            if count >= 3:
                if transport == "tailscale":
                    recommendations.append(
                        "Multiple Tailscale failures: Check tailscaled status and VPN connectivity"
                    )
                elif transport == "ssh":
                    recommendations.append(
                        "Multiple SSH failures: Check SSH keys and firewall rules"
                    )

        # Check for node-specific issues
        node_counts: dict[str, int] = {}
        for f in failures:
            node_counts[f.node_id] = node_counts.get(f.node_id, 0) + 1

        problem_nodes = [n for n, c in node_counts.items() if c >= 3]
        if problem_nodes:
            recommendations.append(
                f"Problem nodes ({len(problem_nodes)}): Consider removing from work queue: "
                + ", ".join(problem_nodes[:5])
            )

        # Resource recommendations
        resource_failures = [f for f in failures if f.operation in ("disk", "memory")]
        if resource_failures:
            recommendations.append("Resource pressure detected: Run cleanup or scale up")

        return recommendations

    def get_recent_failures(
        self, window_seconds: float = 300.0
    ) -> list[FailureContext]:
        """Get recent failures within time window."""
        cutoff = time.time() - window_seconds
        return [f for f in self._failure_history if f.timestamp > cutoff]

    def get_node_health(self, node_id: str) -> dict[str, Any]:
        """Get health summary for a specific node."""
        failure_count = self._node_failure_counts.get(node_id, 0)
        last_failure = self._node_last_failure.get(node_id, 0.0)

        recent_failures = [
            f
            for f in self._failure_history
            if f.node_id == node_id and f.age_seconds < 300
        ]

        return {
            "node_id": node_id,
            "total_failures": failure_count,
            "recent_failures": len(recent_failures),
            "last_failure_age_seconds": (
                time.time() - last_failure if last_failure > 0 else None
            ),
            "failure_details": [
                {
                    "transport": f.transport,
                    "operation": f.operation,
                    "error_type": f.error_type,
                    "age_seconds": round(f.age_seconds, 1),
                }
                for f in recent_failures
            ],
        }

    def health_check(self) -> dict[str, Any]:
        """Return health check result for this monitor."""
        recent_failures = len(self.get_recent_failures(300))
        cascade_count = len(self._cascade_alerts)

        return {
            "healthy": recent_failures < 10 and cascade_count == 0,
            "status": "healthy" if recent_failures < 10 else "degraded",
            "details": {
                "recent_failures": recent_failures,
                "cascade_alerts": cascade_count,
                "nodes_with_failures": len(self._node_failure_counts),
                "last_assessment": (
                    {
                        "severity": self._last_assessment.severity,
                        "age_seconds": time.time() - self._last_assessment_time,
                    }
                    if self._last_assessment
                    else None
                ),
            },
        }


# Singleton instance
_instance: UnifiedHealthMonitor | None = None


def get_unified_health_monitor() -> UnifiedHealthMonitor:
    """Get the singleton unified health monitor."""
    global _instance
    if _instance is None:
        _instance = UnifiedHealthMonitor()
    return _instance


def reset_unified_health_monitor() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None
