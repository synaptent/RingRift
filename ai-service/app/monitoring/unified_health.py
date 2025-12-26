"""Unified Health Orchestrator for RingRift AI (December 2025).

Consolidates all health monitoring implementations into a single orchestrator:
- Component health (training, sync, evaluation, promotion)
- Cluster health (nodes, disk, network)
- Training health (model staleness, training progress)

Usage:
    from app.monitoring.unified_health import (
        UnifiedHealthOrchestrator,
        get_health_orchestrator,
        check_system_health,
    )

    # Get unified health status
    orchestrator = get_health_orchestrator()
    health = orchestrator.check_all_health()

    # Or use convenience function
    health = check_system_health()
    if not health["healthy"]:
        print(f"Issues: {health['issues']}")
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Phase 9 (Dec 2025): Import from canonical source in monitoring.base
# Keep local alias for backward compatibility
from app.monitoring.base import HealthStatus


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    component: str
    status: HealthStatus
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    check_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "check_time": self.check_time,
        }


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""
    overall_status: HealthStatus
    healthy: bool
    checks: list[HealthCheckResult]
    issues: list[str]
    timestamp: float
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "healthy": self.healthy,
            "checks": [c.to_dict() for c in self.checks],
            "issues": self.issues,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


class UnifiedHealthOrchestrator:
    """Unified orchestrator for all health monitoring.

    Consolidates health checks from multiple sources:
    - app/distributed/health_checks.py
    - app/monitoring/cluster_monitor.py
    - app/training/training_health.py
    - app/coordination/orchestrator_registry.py
    """

    def __init__(self):
        self._health_checks: dict[str, Callable[[], HealthCheckResult]] = {}
        self._last_report: SystemHealthReport | None = None
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health check functions."""

        # Component health (from distributed/health_checks.py)
        def check_training_scheduler() -> HealthCheckResult:
            try:
                from app.distributed.health_checks import TrainingSchedulerHealth
                health = TrainingSchedulerHealth()
                status = HealthStatus.HEALTHY if health.is_healthy() else HealthStatus.UNHEALTHY
                return HealthCheckResult(
                    component="training_scheduler",
                    status=status,
                    message=health.get_status_message() if hasattr(health, 'get_status_message') else "",
                    details=health.get_details() if hasattr(health, 'get_details') else {},
                )
            except ImportError:
                return HealthCheckResult(
                    component="training_scheduler",
                    status=HealthStatus.UNKNOWN,
                    message="Health check module not available",
                )
            except Exception as e:
                return HealthCheckResult(
                    component="training_scheduler",
                    status=HealthStatus.UNKNOWN,
                    message=str(e),
                )

        def check_data_sync() -> HealthCheckResult:
            try:
                from app.distributed.health_checks import DataSyncHealth
                health = DataSyncHealth()
                status = HealthStatus.HEALTHY if health.is_healthy() else HealthStatus.UNHEALTHY
                return HealthCheckResult(
                    component="data_sync",
                    status=status,
                    message=health.get_status_message() if hasattr(health, 'get_status_message') else "",
                )
            except ImportError:
                return HealthCheckResult(
                    component="data_sync",
                    status=HealthStatus.UNKNOWN,
                    message="Health check module not available",
                )
            except Exception as e:
                return HealthCheckResult(
                    component="data_sync",
                    status=HealthStatus.UNKNOWN,
                    message=str(e),
                )

        # Resource health
        def check_resources() -> HealthCheckResult:
            try:
                import psutil

                from app.config.thresholds import (
                    CPU_CRITICAL_PERCENT,
                    DISK_CRITICAL_PERCENT,
                    MEMORY_CRITICAL_PERCENT,
                )

                cpu = psutil.cpu_percent()
                memory = psutil.virtual_memory().percent
                disk = psutil.disk_usage('/').percent

                issues = []
                if cpu > CPU_CRITICAL_PERCENT:
                    issues.append(f"CPU at {cpu}% (threshold: {CPU_CRITICAL_PERCENT}%)")
                if memory > MEMORY_CRITICAL_PERCENT:
                    issues.append(f"Memory at {memory}% (threshold: {MEMORY_CRITICAL_PERCENT}%)")
                if disk > DISK_CRITICAL_PERCENT:
                    issues.append(f"Disk at {disk}% (threshold: {DISK_CRITICAL_PERCENT}%)")

                if issues:
                    status = HealthStatus.UNHEALTHY
                    message = "; ".join(issues)
                else:
                    status = HealthStatus.HEALTHY
                    message = f"CPU: {cpu}%, Memory: {memory}%, Disk: {disk}%"

                return HealthCheckResult(
                    component="resources",
                    status=status,
                    message=message,
                    details={"cpu": cpu, "memory": memory, "disk": disk},
                )
            except ImportError:
                return HealthCheckResult(
                    component="resources",
                    status=HealthStatus.UNKNOWN,
                    message="psutil not available",
                )
            except Exception as e:
                return HealthCheckResult(
                    component="resources",
                    status=HealthStatus.UNKNOWN,
                    message=str(e),
                )

        # Coordinator health
        def check_coordinators() -> HealthCheckResult:
            try:
                from app.coordination.orchestrator_registry import (
                    check_cluster_health,
                )
                health = check_cluster_health()
                is_healthy = health.get("overall_healthy", True)
                status = HealthStatus.HEALTHY if is_healthy else HealthStatus.DEGRADED

                return HealthCheckResult(
                    component="coordinators",
                    status=status,
                    message=f"{health.get('healthy_count', 0)}/{health.get('total_count', 0)} healthy",
                    details=health,
                )
            except ImportError:
                return HealthCheckResult(
                    component="coordinators",
                    status=HealthStatus.UNKNOWN,
                    message="Orchestrator registry not available",
                )
            except Exception as e:
                return HealthCheckResult(
                    component="coordinators",
                    status=HealthStatus.UNKNOWN,
                    message=str(e),
                )

        # Event router health
        def check_event_bus() -> HealthCheckResult:
            try:
                from app.coordination.event_router import get_router
                router = get_router()
                stats = router.get_stats()

                total_subs = stats.get("total_subscriptions", 0)
                total_events = stats.get("total_events_published", 0)

                if total_subs == 0:
                    status = HealthStatus.DEGRADED
                    message = "No event subscribers registered"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"{total_events} events published, {total_subs} subscriptions"

                return HealthCheckResult(
                    component="event_bus",
                    status=status,
                    message=message,
                    details=stats,
                )
            except Exception as e:
                return HealthCheckResult(
                    component="event_bus",
                    status=HealthStatus.UNKNOWN,
                    message=str(e),
                )

        # Register all checks
        self._health_checks = {
            "training_scheduler": check_training_scheduler,
            "data_sync": check_data_sync,
            "resources": check_resources,
            "coordinators": check_coordinators,
            "event_bus": check_event_bus,
        }

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], HealthCheckResult],
    ) -> None:
        """Register a custom health check.

        Args:
            name: Name of the health check
            check_fn: Function that returns HealthCheckResult
        """
        self._health_checks[name] = check_fn
        logger.debug(f"[UnifiedHealthOrchestrator] Registered check: {name}")

    def unregister_check(self, name: str) -> bool:
        """Unregister a health check.

        Returns:
            True if check was removed
        """
        if name in self._health_checks:
            del self._health_checks[name]
            return True
        return False

    def check_all_health(self) -> SystemHealthReport:
        """Run all health checks and generate a report.

        Returns:
            SystemHealthReport with all check results
        """
        start_time = time.time()
        checks: list[HealthCheckResult] = []
        issues: list[str] = []

        for name, check_fn in self._health_checks.items():
            check_start = time.time()
            try:
                result = check_fn()
                result.check_time = (time.time() - check_start) * 1000
                checks.append(result)

                if result.status == HealthStatus.UNHEALTHY:
                    issues.append(f"{result.component}: {result.message}")
                elif result.status == HealthStatus.DEGRADED:
                    issues.append(f"{result.component} (degraded): {result.message}")

            except Exception as e:
                checks.append(HealthCheckResult(
                    component=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {e}",
                    check_time=(time.time() - check_start) * 1000,
                ))

        # Determine overall status
        statuses = [c.status for c in checks]
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        else:
            overall = HealthStatus.DEGRADED

        report = SystemHealthReport(
            overall_status=overall,
            healthy=overall == HealthStatus.HEALTHY,
            checks=checks,
            issues=issues,
            timestamp=time.time(),
            duration_ms=(time.time() - start_time) * 1000,
        )

        self._last_report = report
        return report

    def check_component(self, name: str) -> HealthCheckResult | None:
        """Check health of a specific component.

        Args:
            name: Name of the component

        Returns:
            HealthCheckResult or None if component not found
        """
        if name not in self._health_checks:
            return None

        check_fn = self._health_checks[name]
        start_time = time.time()
        try:
            result = check_fn()
            result.check_time = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            return HealthCheckResult(
                component=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {e}",
                check_time=(time.time() - start_time) * 1000,
            )

    def get_last_report(self) -> SystemHealthReport | None:
        """Get the last health report."""
        return self._last_report

    def get_registered_checks(self) -> list[str]:
        """Get list of registered health check names."""
        return list(self._health_checks.keys())


# Singleton instance
_health_orchestrator: UnifiedHealthOrchestrator | None = None


def get_health_orchestrator() -> UnifiedHealthOrchestrator:
    """Get the global health orchestrator singleton."""
    global _health_orchestrator
    if _health_orchestrator is None:
        _health_orchestrator = UnifiedHealthOrchestrator()
    return _health_orchestrator


def check_system_health() -> dict[str, Any]:
    """Convenience function to check system health.

    Returns:
        Dict with health report
    """
    return get_health_orchestrator().check_all_health().to_dict()


def is_system_healthy() -> bool:
    """Quick check if system is healthy.

    Returns:
        True if system is healthy
    """
    return get_health_orchestrator().check_all_health().healthy
