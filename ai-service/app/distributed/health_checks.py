#!/usr/bin/env python3
"""Component health checks for the RingRift AI improvement pipeline.

This module provides health monitoring for all pipeline components:
- Data sync daemon
- Training scheduler
- Evaluation/tournament service
- Model promotion service
- Cluster coordinator

Usage:
    from app.distributed.health_checks import HealthChecker, get_health_summary

    checker = HealthChecker()
    summary = checker.check_all()
    if not summary.healthy:
        for issue in summary.issues:
            print(f"UNHEALTHY: {issue}")
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)

# Path setup
from app.utils.paths import AI_SERVICE_ROOT

# Unified resource checking utilities (80% max utilization)
try:
    from app.utils.resource_guard import LIMITS as RESOURCE_LIMITS
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    RESOURCE_LIMITS = None

# Coordinator metrics for monitoring
try:
    from app.metrics.coordinator import (
        collect_all_coordinator_metrics_sync,
        update_coordinator_status,
        update_coordinator_uptime,
    )
    HAS_COORDINATOR_METRICS = True
except ImportError:
    collect_all_coordinator_metrics_sync = None
    update_coordinator_status = None
    update_coordinator_uptime = None
    HAS_COORDINATOR_METRICS = False

# Import centralized thresholds (single source of truth)
try:
    from app.config.thresholds import (
        CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        CPU_CRITICAL_PERCENT,
        CPU_WARNING_PERCENT,
        DISK_CRITICAL_PERCENT,
        DISK_WARNING_PERCENT,
        MEMORY_CRITICAL_PERCENT,
        MEMORY_WARNING_PERCENT,
    )
    # Use centralized thresholds
    MEMORY_WARNING_THRESHOLD = MEMORY_WARNING_PERCENT
    MEMORY_CRITICAL_THRESHOLD = MEMORY_CRITICAL_PERCENT
    DISK_WARNING_THRESHOLD = DISK_WARNING_PERCENT
    DISK_CRITICAL_THRESHOLD = DISK_CRITICAL_PERCENT
    CPU_WARNING_THRESHOLD = CPU_WARNING_PERCENT
    CPU_CRITICAL_THRESHOLD = CPU_CRITICAL_PERCENT
    RECOVERY_COOLDOWN = CIRCUIT_BREAKER_RECOVERY_TIMEOUT
except ImportError:
    # Fallback defaults matching thresholds.py
    MEMORY_WARNING_THRESHOLD = 70.0
    MEMORY_CRITICAL_THRESHOLD = 80.0
    DISK_WARNING_THRESHOLD = 65.0
    DISK_CRITICAL_THRESHOLD = 70.0
    CPU_WARNING_THRESHOLD = 70.0
    CPU_CRITICAL_THRESHOLD = 80.0
    RECOVERY_COOLDOWN = 300

# Event router for health/recovery events (Phase 10 consolidation)
try:
    from app.coordination.event_router import get_router
    from app.coordination.event_router import DataEventType
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False
    get_router = None


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    healthy: bool
    status: str  # "ok", "warning", "error", "unknown"
    message: str = ""
    last_activity: float | None = None  # Unix timestamp
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthSummary:
    """Overall health summary of all components."""
    healthy: bool
    timestamp: str
    components: list[ComponentHealth]
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def component_status(self) -> dict[str, str]:
        return {c.name: c.status for c in self.components}


class HealthChecker:
    """Checks health of all pipeline components."""

    # Thresholds for health checks
    DATA_SYNC_STALE_THRESHOLD = 3600  # 1 hour
    TRAINING_STALE_THRESHOLD = 14400  # 4 hours
    EVALUATION_STALE_THRESHOLD = 7200  # 2 hours
    COORDINATOR_STALE_THRESHOLD = 86400  # 24 hours

    def __init__(self, merged_db_path: Path | None = None):
        self.elo_db_path = AI_SERVICE_ROOT / "data" / "unified_elo.db"
        # Default to selfplay.db which is the actual merged training database
        self.merged_db_path = merged_db_path or AI_SERVICE_ROOT / "data" / "games" / "selfplay.db"
        self.coordinator_db_path = AI_SERVICE_ROOT / "data" / "coordination" / "task_registry.db"
        self.state_path = AI_SERVICE_ROOT / "logs" / "unified_loop" / "unified_loop_state.json"

    def check_all(self) -> HealthSummary:
        """Run all health checks and return summary."""
        components = [
            self.check_data_sync(),
            self.check_training(),
            self.check_evaluation(),
            self.check_coordinator(),
            self.check_coordinators(),  # New: coordinator metrics
            self.check_resources(),
        ]

        issues = []
        warnings = []

        for c in components:
            if c.status == "error":
                issues.append(f"[{c.name}] {c.message}")
            elif c.status == "warning":
                warnings.append(f"[{c.name}] {c.message}")

        healthy = len(issues) == 0

        return HealthSummary(
            healthy=healthy,
            timestamp=datetime.now(timezone.utc).isoformat(),
            components=components,
            issues=issues,
            warnings=warnings,
        )

    def check_data_sync(self) -> ComponentHealth:
        """Check data sync daemon health."""
        name = "data_sync"

        # Check if merged database exists and is recent
        if not self.merged_db_path.exists():
            return ComponentHealth(
                name=name,
                healthy=False,
                status="error",
                message="Merged training database not found",
            )

        mtime = self.merged_db_path.stat().st_mtime
        age = time.time() - mtime

        if age > self.DATA_SYNC_STALE_THRESHOLD:
            return ComponentHealth(
                name=name,
                healthy=False,
                status="warning",
                message=f"Data sync stale ({age/3600:.1f}h since last update)",
                last_activity=mtime,
                details={"age_seconds": age},
            )

        # Check game count
        try:
            conn = sqlite3.connect(str(self.merged_db_path), timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM games")
            game_count = cursor.fetchone()[0]
            conn.close()

            return ComponentHealth(
                name=name,
                healthy=True,
                status="ok",
                message=f"{game_count} games synced",
                last_activity=mtime,
                details={"game_count": game_count, "age_seconds": age},
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                healthy=False,
                status="error",
                message=f"Failed to read merged database: {e}",
            )

    def check_training(self) -> ComponentHealth:
        """Check training component health."""
        name = "training"

        # Check for recent training runs
        runs_dir = AI_SERVICE_ROOT / "logs" / "unified_training"
        if not runs_dir.exists():
            return ComponentHealth(
                name=name,
                healthy=True,  # No training yet is OK
                status="ok",
                message="No training runs yet",
            )

        # Find most recent run
        runs = list(runs_dir.iterdir())
        if not runs:
            return ComponentHealth(
                name=name,
                healthy=True,
                status="ok",
                message="No training runs yet",
            )

        latest_run = max(runs, key=lambda p: p.stat().st_mtime)
        mtime = latest_run.stat().st_mtime
        age = time.time() - mtime

        # Check for training report
        report_path = latest_run / "training_report.json"
        if report_path.exists():
            import json
            try:
                with open(report_path) as f:
                    report = json.load(f)
                success = report.get("success", False)
                status = "ok" if success else "warning"
                message = f"Last run: {latest_run.name} ({'success' if success else 'failed'})"
            except (json.JSONDecodeError, OSError, ValueError):
                status = "warning"
                message = f"Last run: {latest_run.name} (report unreadable)"
        else:
            status = "warning" if age < 3600 else "ok"  # In progress if recent
            message = f"Last run: {latest_run.name} (in progress or no report)"

        return ComponentHealth(
            name=name,
            healthy=status != "error",
            status=status,
            message=message,
            last_activity=mtime,
            details={"latest_run": latest_run.name, "age_seconds": age},
        )

    def check_evaluation(self) -> ComponentHealth:
        """Check evaluation/tournament health."""
        name = "evaluation"

        if not self.elo_db_path.exists():
            return ComponentHealth(
                name=name,
                healthy=False,
                status="warning",
                message="Elo database not found",
            )

        try:
            conn = sqlite3.connect(str(self.elo_db_path), timeout=5)
            cursor = conn.cursor()

            # Check for recent matches
            cursor.execute("""
                SELECT MAX(timestamp) FROM match_history
            """)
            row = cursor.fetchone()

            if row[0] is None:
                conn.close()
                return ComponentHealth(
                    name=name,
                    healthy=True,
                    status="ok",
                    message="No evaluations yet",
                )

            # Parse timestamp
            last_match = row[0]
            if isinstance(last_match, str):
                try:
                    dt = datetime.fromisoformat(last_match.replace("Z", "+00:00"))
                    last_match_ts = dt.timestamp()
                except ValueError:
                    last_match_ts = 0
            else:
                last_match_ts = last_match

            age = time.time() - last_match_ts

            # Count total matches
            cursor.execute("SELECT COUNT(*) FROM match_history")
            match_count = cursor.fetchone()[0]
            conn.close()

            if age > self.EVALUATION_STALE_THRESHOLD:
                return ComponentHealth(
                    name=name,
                    healthy=False,
                    status="warning",
                    message=f"Evaluation stale ({age/3600:.1f}h since last match)",
                    last_activity=last_match_ts,
                    details={"match_count": match_count, "age_seconds": age},
                )

            return ComponentHealth(
                name=name,
                healthy=True,
                status="ok",
                message=f"{match_count} matches recorded",
                last_activity=last_match_ts,
                details={"match_count": match_count, "age_seconds": age},
            )

        except Exception as e:
            return ComponentHealth(
                name=name,
                healthy=False,
                status="error",
                message=f"Failed to read Elo database: {e}",
            )

    def check_coordinator(self) -> ComponentHealth:
        """Check cluster coordinator health."""
        name = "coordinator"

        if not self.coordinator_db_path.exists():
            return ComponentHealth(
                name=name,
                healthy=True,
                status="ok",
                message="No coordination database (standalone mode)",
            )

        try:
            conn = sqlite3.connect(str(self.coordinator_db_path), timeout=5)
            cursor = conn.cursor()

            # Check for active tasks
            cursor.execute("SELECT COUNT(*) FROM tasks")
            task_count = cursor.fetchone()[0]

            # Check for stale tasks
            cursor.execute("""
                SELECT COUNT(*) FROM tasks
                WHERE last_heartbeat < datetime('now', '-1 hour')
            """)
            stale_count = cursor.fetchone()[0]
            conn.close()

            if stale_count > 0:
                return ComponentHealth(
                    name=name,
                    healthy=False,
                    status="warning",
                    message=f"{stale_count} stale tasks detected",
                    details={"task_count": task_count, "stale_count": stale_count},
                )

            return ComponentHealth(
                name=name,
                healthy=True,
                status="ok",
                message=f"{task_count} active tasks",
                details={"task_count": task_count},
            )

        except Exception as e:
            return ComponentHealth(
                name=name,
                healthy=False,
                status="error",
                message=f"Failed to read coordinator database: {e}",
            )

    def check_coordinators(self) -> ComponentHealth:
        """Check coordinator manager health via metrics collection.

        This collects metrics from RecoveryManager, BandwidthManager, and
        SyncCoordinator, updating Prometheus metrics and returning health status.
        """
        name = "coordinator_managers"

        if not HAS_COORDINATOR_METRICS or collect_all_coordinator_metrics_sync is None:
            return ComponentHealth(
                name=name,
                healthy=True,
                status="ok",
                message="Coordinator metrics not available (optional)",
            )

        try:
            # Collect metrics from all coordinators (sync version for health checks)
            metrics = collect_all_coordinator_metrics_sync()
            coordinators = metrics.get("coordinators", {})

            if not coordinators:
                return ComponentHealth(
                    name=name,
                    healthy=True,
                    status="ok",
                    message="No coordinators active (standalone mode)",
                )

            # Check status of each coordinator
            error_coordinators = []
            running_coordinators = []

            for coord_name, stats in coordinators.items():
                status = stats.get("status", "unknown")
                if status == "error":
                    error_coordinators.append(coord_name)
                elif status in ("running", "ready"):
                    running_coordinators.append(coord_name)

            details = {
                "coordinators": list(coordinators.keys()),
                "running": running_coordinators,
                "errors": error_coordinators,
            }

            if error_coordinators:
                return ComponentHealth(
                    name=name,
                    healthy=False,
                    status="error",
                    message=f"Coordinators in error: {', '.join(error_coordinators)}",
                    details=details,
                )

            if running_coordinators:
                return ComponentHealth(
                    name=name,
                    healthy=True,
                    status="ok",
                    message=f"{len(running_coordinators)} coordinators running",
                    details=details,
                )

            return ComponentHealth(
                name=name,
                healthy=True,
                status="ok",
                message=f"{len(coordinators)} coordinators tracked",
                details=details,
            )

        except Exception as e:
            return ComponentHealth(
                name=name,
                healthy=False,
                status="warning",
                message=f"Failed to collect coordinator metrics: {e}",
            )

    def check_resources(self) -> ComponentHealth:
        """Check system resource health.

        Uses thresholds aligned with 80% max utilization policy (70% for disk).
        """
        name = "resources"

        try:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage(str(AI_SERVICE_ROOT))
            cpu_percent = psutil.cpu_percent(interval=0.1)

            issues = []

            # Memory thresholds - 80% max
            if mem.percent > MEMORY_CRITICAL_THRESHOLD:
                issues.append(f"Memory critical: {mem.percent:.1f}%")
            elif mem.percent > MEMORY_WARNING_THRESHOLD:
                issues.append(f"Memory high: {mem.percent:.1f}%")

            # Disk thresholds - 70% max (tighter because cleanup takes time)
            if disk.percent > DISK_CRITICAL_THRESHOLD:
                issues.append(f"Disk critical: {disk.percent:.1f}%")
            elif disk.percent > DISK_WARNING_THRESHOLD:
                issues.append(f"Disk high: {disk.percent:.1f}%")

            # CPU thresholds - 80% max
            if cpu_percent > CPU_CRITICAL_THRESHOLD:
                issues.append(f"CPU critical: {cpu_percent:.1f}%")
            elif cpu_percent > CPU_WARNING_THRESHOLD:
                issues.append(f"CPU high: {cpu_percent:.1f}%")

            details = {
                "memory_percent": mem.percent,
                "memory_available_gb": mem.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "cpu_percent": cpu_percent,
            }

            if issues:
                return ComponentHealth(
                    name=name,
                    healthy=False,
                    status="warning" if len(issues) == 1 else "error",
                    message="; ".join(issues),
                    details=details,
                )

            return ComponentHealth(
                name=name,
                healthy=True,
                status="ok",
                message=f"Memory: {mem.percent:.0f}%, Disk: {disk.percent:.0f}%, CPU: {cpu_percent:.0f}%",
                details=details,
            )

        except Exception as e:
            return ComponentHealth(
                name=name,
                healthy=False,
                status="error",
                message=f"Failed to check resources: {e}",
            )


def get_health_summary() -> HealthSummary:
    """Convenience function to get health summary."""
    checker = HealthChecker()
    return checker.check_all()


def format_health_report(summary: HealthSummary) -> str:
    """Format health summary as human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("HEALTH CHECK REPORT")
    lines.append(f"Timestamp: {summary.timestamp}")
    lines.append(f"Overall: {'HEALTHY' if summary.healthy else 'UNHEALTHY'}")
    lines.append("=" * 60)

    for component in summary.components:
        status_icon = {
            "ok": "✓",
            "warning": "⚠",
            "error": "✗",
            "unknown": "?",
        }.get(component.status, "?")

        lines.append(f"\n[{status_icon}] {component.name.upper()}")
        lines.append(f"    Status: {component.status}")
        lines.append(f"    Message: {component.message}")
        if component.last_activity:
            age = time.time() - component.last_activity
            lines.append(f"    Last activity: {age/60:.0f} minutes ago")

    if summary.issues:
        lines.append("\n" + "-" * 60)
        lines.append("ISSUES:")
        for issue in summary.issues:
            lines.append(f"  - {issue}")

    if summary.warnings:
        lines.append("\n" + "-" * 60)
        lines.append("WARNINGS:")
        for warning in summary.warnings:
            lines.append(f"  - {warning}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


class HealthRecoveryIntegration:
    """
    Integrates health checks with automatic recovery.

    When health checks detect issues:
    - Stale jobs → attempt job recovery
    - Unhealthy nodes → attempt node recovery
    - Persistent failures → escalate to human operators

    Usage:
        from app.distributed.health_checks import HealthRecoveryIntegration

        integration = HealthRecoveryIntegration(recovery_manager)
        integration.start_monitoring(check_interval=60)
    """

    def __init__(
        self,
        recovery_manager=None,
        notifier=None,
        auto_recover: bool = True,
        check_interval: int = 60,
    ):
        """
        Initialize health-recovery integration.

        Args:
            recovery_manager: RecoveryManager instance for triggering recovery
            notifier: Optional notifier for alerts
            auto_recover: Whether to automatically trigger recovery
            check_interval: Seconds between health checks
        """
        self.recovery_manager = recovery_manager
        self.notifier = notifier
        self.auto_recover = auto_recover
        self.check_interval = check_interval
        self.checker = HealthChecker()
        self._running = False
        self._consecutive_failures: dict[str, int] = {}
        self._last_recovery_attempt: dict[str, float] = {}
        self._recovery_cooldown = RECOVERY_COOLDOWN  # From thresholds.py

    async def check_and_recover(self) -> HealthSummary:
        """
        Run health check and trigger recovery if needed.

        Returns:
            HealthSummary from the check
        """
        summary = self.checker.check_all()

        if summary.healthy:
            # Reset failure counters on healthy check
            self._consecutive_failures.clear()
            return summary

        # Process each unhealthy component
        for component in summary.components:
            if component.status in ("error", "warning"):
                await self._handle_unhealthy_component(component)

        return summary

    async def _handle_unhealthy_component(self, component: ComponentHealth) -> None:
        """Handle an unhealthy component."""
        component_name = component.name

        # Track consecutive failures
        self._consecutive_failures[component_name] = \
            self._consecutive_failures.get(component_name, 0) + 1

        failures = self._consecutive_failures[component_name]
        logger.warning(
            f"[Health→Recovery] {component_name} unhealthy "
            f"(consecutive={failures}): {component.message}"
        )

        # Check recovery cooldown
        last_attempt = self._last_recovery_attempt.get(component_name, 0)
        if time.time() - last_attempt < self._recovery_cooldown:
            logger.debug(f"[Health→Recovery] {component_name} in recovery cooldown")
            return

        # Trigger recovery based on component and failure count
        if self.auto_recover and self.recovery_manager:
            await self._trigger_recovery(component, failures)

    async def _trigger_recovery(
        self,
        component: ComponentHealth,
        failure_count: int
    ) -> None:
        """Trigger appropriate recovery action."""
        component_name = component.name
        self._last_recovery_attempt[component_name] = time.time()
        recovery_action = None
        recovery_success = False

        try:
            # Emit RECOVERY_INITIATED event
            await self._emit_recovery_event(
                DataEventType.RECOVERY_INITIATED if HAS_EVENT_BUS else None,
                component_name, failure_count, component.message
            )

            if component_name == "coordinator" and failure_count >= 2:
                # Stale tasks - try to recover stuck jobs
                details = component.details or {}
                stale_count = details.get("stale_count", 0)
                if stale_count > 0:
                    logger.info(f"[Health→Recovery] Attempting to recover {stale_count} stale jobs")
                    recovery_action = "cleanup_stale_jobs"
                    if hasattr(self.recovery_manager, 'cleanup_stale_jobs'):
                        await self.recovery_manager.cleanup_stale_jobs()
                        recovery_success = True

            elif component_name == "data_sync" and failure_count >= 3:
                # Data sync stale - restart sync daemon
                logger.info("[Health→Recovery] Data sync stale, triggering sync restart")
                recovery_action = "restart_data_sync"
                if hasattr(self.recovery_manager, 'restart_data_sync'):
                    await self.recovery_manager.restart_data_sync()
                    recovery_success = True

            elif component_name == "resources":
                # Resource issues - log for manual intervention
                logger.warning(f"[Health→Recovery] Resource issue: {component.message}")
                if self.notifier and failure_count >= 5:
                    recovery_action = "notify_resource_issue"
                    await self._notify_resource_issue(component)
                    recovery_success = True

                # Emit RESOURCE_CONSTRAINT event
                await self._emit_recovery_event(
                    DataEventType.RESOURCE_CONSTRAINT if HAS_EVENT_BUS else None,
                    component_name,
                    failure_count,
                    component.message,
                    details=component.details,
                )

            elif failure_count >= 5:
                # Persistent failures - escalate
                logger.error(
                    f"[Health→Recovery] Persistent failures for {component_name}, "
                    f"escalating to human operator"
                )
                recovery_action = "escalate_to_human"
                if self.notifier:
                    await self._notify_escalation(component, failure_count)

            # Emit RECOVERY_COMPLETED or RECOVERY_FAILED
            if recovery_action and HAS_EVENT_BUS:
                event_type = (
                    DataEventType.RECOVERY_COMPLETED if recovery_success
                    else DataEventType.RECOVERY_FAILED
                )
                await self._emit_recovery_event(
                    event_type, component_name, failure_count,
                    f"Recovery action: {recovery_action}",
                    action=recovery_action, success=recovery_success
                )

        except Exception as e:
            logger.error(f"[Health→Recovery] Recovery failed for {component_name}: {e}")
            # Emit RECOVERY_FAILED event
            if HAS_EVENT_BUS:
                await self._emit_recovery_event(
                    DataEventType.RECOVERY_FAILED,
                    component_name, failure_count, str(e)
                )

    async def _emit_recovery_event(
        self,
        event_type,
        component: str,
        failure_count: int,
        message: str,
        **kwargs
    ) -> None:
        """Emit a recovery-related event."""
        if not HAS_EVENT_BUS or event_type is None:
            return

        event_router = get_router()
        await event_router.publish(
            event_type,
            payload={
                "component": component,
                "failure_count": failure_count,
                "message": message,
                **kwargs,
            },
            source="health_recovery_integration",
        )

    async def _notify_resource_issue(self, component: ComponentHealth) -> None:
        """Notify about resource issues."""
        if self.notifier and hasattr(self.notifier, 'send_alert'):
            await self.notifier.send_alert(
                level="warning",
                title="Resource Issue Detected",
                message=component.message,
                details=component.details
            )

    async def _notify_escalation(
        self,
        component: ComponentHealth,
        failure_count: int
    ) -> None:
        """Notify about escalation to human operator."""
        if self.notifier and hasattr(self.notifier, 'send_alert'):
            await self.notifier.send_alert(
                level="critical",
                title=f"Persistent Failure: {component.name}",
                message=f"Component has failed {failure_count} consecutive times. "
                        f"Manual intervention may be required.",
                details={
                    "component": component.name,
                    "status": component.status,
                    "message": component.message,
                    "failure_count": failure_count,
                    **component.details
                }
            )

    async def start_monitoring(self) -> None:
        """Start background health monitoring loop."""
        import asyncio

        self._running = True
        logger.info(
            f"[Health→Recovery] Starting health monitoring "
            f"(interval={self.check_interval}s, auto_recover={self.auto_recover})"
        )

        while self._running:
            try:
                summary = await self.check_and_recover()
                if not summary.healthy:
                    logger.info(
                        f"[Health→Recovery] Check complete: "
                        f"{len(summary.issues)} issues, {len(summary.warnings)} warnings"
                    )
            except Exception as e:
                logger.error(f"[Health→Recovery] Health check error: {e}")

            await asyncio.sleep(self.check_interval)

    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._running = False
        logger.info("[Health→Recovery] Health monitoring stopped")

    def get_status(self) -> dict[str, Any]:
        """Get integration status."""
        return {
            "running": self._running,
            "auto_recover": self.auto_recover,
            "check_interval": self.check_interval,
            "consecutive_failures": dict(self._consecutive_failures),
            "recovery_cooldown": self._recovery_cooldown,
        }


def integrate_health_with_recovery(
    recovery_manager=None,
    notifier=None,
    auto_recover: bool = True,
) -> HealthRecoveryIntegration:
    """
    Create health-recovery integration.

    Usage:
        from app.distributed.health_checks import integrate_health_with_recovery

        integration = integrate_health_with_recovery(
            recovery_manager=recovery_manager,
            notifier=slack_notifier,
            auto_recover=True
        )

        # Start monitoring (in async context)
        await integration.start_monitoring()

    Args:
        recovery_manager: RecoveryManager for triggering recovery
        notifier: Optional notification service
        auto_recover: Whether to automatically trigger recovery

    Returns:
        Configured HealthRecoveryIntegration
    """
    integration = HealthRecoveryIntegration(
        recovery_manager=recovery_manager,
        notifier=notifier,
        auto_recover=auto_recover
    )
    logger.info("[Health→Recovery] Integration created")
    return integration


if __name__ == "__main__":
    summary = get_health_summary()
    print(format_health_report(summary))
