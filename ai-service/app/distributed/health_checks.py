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

import os
import psutil
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path setup
AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]

# Unified resource checking utilities (80% max utilization)
try:
    from app.utils.resource_guard import LIMITS as RESOURCE_LIMITS
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    RESOURCE_LIMITS = None

# Resource thresholds - aligned with 80% max utilization (2025-12-16)
MEMORY_WARNING_THRESHOLD = 70.0  # Below 80% limit
MEMORY_CRITICAL_THRESHOLD = 80.0  # At limit
DISK_WARNING_THRESHOLD = 65.0  # Below 70% limit
DISK_CRITICAL_THRESHOLD = 70.0  # At disk limit (tighter than other resources)
CPU_WARNING_THRESHOLD = 70.0  # Below 80% limit
CPU_CRITICAL_THRESHOLD = 80.0  # At limit


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    healthy: bool
    status: str  # "ok", "warning", "error", "unknown"
    message: str = ""
    last_activity: Optional[float] = None  # Unix timestamp
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthSummary:
    """Overall health summary of all components."""
    healthy: bool
    timestamp: str
    components: List[ComponentHealth]
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def component_status(self) -> Dict[str, str]:
        return {c.name: c.status for c in self.components}


class HealthChecker:
    """Checks health of all pipeline components."""

    # Thresholds for health checks
    DATA_SYNC_STALE_THRESHOLD = 3600  # 1 hour
    TRAINING_STALE_THRESHOLD = 14400  # 4 hours
    EVALUATION_STALE_THRESHOLD = 7200  # 2 hours
    COORDINATOR_STALE_THRESHOLD = 86400  # 24 hours

    def __init__(self, merged_db_path: Optional[Path] = None):
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
            except Exception:
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


if __name__ == "__main__":
    summary = get_health_summary()
    print(format_health_report(summary))
