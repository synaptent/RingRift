#!/usr/bin/env python3
"""Autonomous Operation Monitor for 48-Hour Unattended Runs.

This script provides a live dashboard for monitoring long-running autonomous
training sessions. It aggregates metrics from multiple sources:

1. Daemon health (DaemonManager)
2. Cluster health (ClusterHealthDashboard)
3. Selfplay stats (P2P orchestrator)
4. Training progress (Elo tracking)
5. Disk usage (coordinator and training nodes)
6. Active alerts

Usage:
    # Live dashboard (updates every 60 seconds)
    python scripts/autonomous_monitor.py

    # Custom update interval
    python scripts/autonomous_monitor.py --interval 30

    # Single snapshot
    python scripts/autonomous_monitor.py --once

    # JSON output (for automation)
    python scripts/autonomous_monitor.py --json

Created: December 29, 2025
Part of: 48-Hour Autonomous Operation Plan (Phase 4.1)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Terminal colors
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
DIM = "\033[2m"


@dataclass
class DaemonHealth:
    """Health status of a single daemon."""

    name: str
    running: bool = False
    healthy: bool = False
    last_cycle: datetime | None = None
    errors_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfplayStats:
    """Selfplay job statistics."""

    jobs_dispatched_total: int = 0
    jobs_active: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    games_generated: int = 0
    dispatch_success_rate: float = 0.0


@dataclass
class TrainingStats:
    """Training progress statistics."""

    training_active: bool = False
    active_configs: list[str] = field(default_factory=list)
    last_training_config: str = ""
    last_elo_change: dict[str, float] = field(default_factory=dict)
    models_promoted: int = 0


@dataclass
class DiskStats:
    """Disk usage statistics."""

    coordinator_percent: float = 0.0
    coordinator_free_gb: float = 0.0
    avg_training_node_percent: float = 0.0
    nodes_above_80_percent: int = 0


@dataclass
class Alert:
    """System alert."""

    level: str  # critical, warning, info
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""


@dataclass
class AutonomousStatus:
    """Complete status for autonomous operation."""

    # Meta
    timestamp: datetime = field(default_factory=datetime.now)
    uptime_hours: float = 0.0
    start_time: datetime | None = None

    # Health
    system_health_score: float = 0.0
    system_health_level: str = "unknown"
    daemons_healthy: int = 0
    daemons_total: int = 0
    daemon_details: list[DaemonHealth] = field(default_factory=list)

    # Cluster
    nodes_alive: int = 0
    nodes_total: int = 0
    leader_id: str = ""

    # Selfplay
    selfplay: SelfplayStats = field(default_factory=SelfplayStats)

    # Training
    training: TrainingStats = field(default_factory=TrainingStats)

    # Disk
    disk: DiskStats = field(default_factory=DiskStats)

    # Alerts
    alerts: list[Alert] = field(default_factory=list)

    # Errors during collection
    collection_errors: list[str] = field(default_factory=list)


class AutonomousMonitor:
    """Monitor for autonomous 48-hour operations."""

    # Critical daemons that must be healthy
    CRITICAL_DAEMONS = [
        "event_router",
        "auto_sync",
        "data_pipeline",
        "feedback_loop",
        "selfplay_coordinator",
    ]

    def __init__(self, start_time: datetime | None = None):
        """Initialize monitor.

        Args:
            start_time: When the autonomous run started. If None, uses now.
        """
        self.start_time = start_time or datetime.now()
        self._p2p_status_cache: dict[str, Any] | None = None
        self._p2p_cache_time: float = 0

    def get_uptime_hours(self) -> float:
        """Get hours since start of autonomous run."""
        delta = datetime.now() - self.start_time
        return delta.total_seconds() / 3600

    async def get_status(self) -> AutonomousStatus:
        """Collect complete autonomous operation status."""
        status = AutonomousStatus(
            start_time=self.start_time,
            uptime_hours=self.get_uptime_hours(),
        )

        # Collect all metrics concurrently where possible
        try:
            # Get P2P status first (many metrics depend on it)
            p2p_status = await self._get_p2p_status()

            # Parallel collection
            daemon_task = asyncio.create_task(self._get_daemon_health())
            health_task = asyncio.create_task(self._get_system_health())
            disk_task = asyncio.create_task(self._get_disk_stats(p2p_status))

            daemon_health = await daemon_task
            system_health = await health_task
            disk_stats = await disk_task

            # Extract cluster info from P2P
            if p2p_status:
                status.nodes_alive = p2p_status.get("alive_peers", 0)
                status.nodes_total = p2p_status.get("total_peers", 0)
                status.leader_id = p2p_status.get("leader_id", "")

                # Extract selfplay stats
                status.selfplay = self._extract_selfplay_stats(p2p_status)

                # Extract training stats
                status.training = self._extract_training_stats(p2p_status)

            # Apply daemon health
            status.daemon_details = daemon_health
            status.daemons_healthy = sum(1 for d in daemon_health if d.healthy)
            status.daemons_total = len(daemon_health)

            # Apply system health
            status.system_health_score = system_health.get("score", 0.0)
            status.system_health_level = system_health.get("level", "unknown")

            # Apply disk stats
            status.disk = disk_stats

            # Generate alerts
            status.alerts = self._generate_alerts(status)

        except Exception as e:
            status.collection_errors.append(f"Collection error: {e}")
            logger.exception("Error collecting status")

        return status

    async def _get_p2p_status(self) -> dict[str, Any]:
        """Get P2P cluster status."""
        # Cache for 5 seconds
        now = time.time()
        if self._p2p_status_cache and (now - self._p2p_cache_time) < 5:
            return self._p2p_status_cache

        try:
            proc = await asyncio.create_subprocess_exec(
                "curl",
                "-s",
                "--connect-timeout",
                "5",
                "http://localhost:8770/status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            if stdout:
                self._p2p_status_cache = json.loads(stdout.decode())
                self._p2p_cache_time = now
                return self._p2p_status_cache
        except Exception as e:
            logger.debug(f"P2P status fetch failed: {e}")

        return {}

    async def _get_daemon_health(self) -> list[DaemonHealth]:
        """Get health of all managed daemons."""
        results: list[DaemonHealth] = []

        try:
            from app.coordination.daemon_manager import get_daemon_manager

            dm = get_daemon_manager()
            all_health = dm.get_all_daemon_health()

            for daemon_name, health_info in all_health.items():
                results.append(
                    DaemonHealth(
                        name=daemon_name,
                        running=health_info.get("running", False),
                        healthy=health_info.get("healthy", False),
                        errors_count=health_info.get("errors_count", 0),
                        details=health_info,
                    )
                )

        except Exception as e:
            logger.debug(f"Daemon health fetch failed: {e}")
            # Add placeholder for critical daemons
            for daemon in self.CRITICAL_DAEMONS:
                results.append(DaemonHealth(name=daemon, running=False, healthy=False))

        return results

    async def _get_system_health(self) -> dict[str, Any]:
        """Get system-level health score."""
        try:
            from app.coordination.health_facade import (
                get_system_health_score,
            )

            score_obj = get_system_health_score()
            if score_obj:
                return {
                    "score": score_obj.score,
                    "level": score_obj.level.value if hasattr(score_obj.level, "value") else str(score_obj.level),
                }
        except Exception as e:
            logger.debug(f"System health fetch failed: {e}")

        return {"score": 0.0, "level": "unknown"}

    async def _get_disk_stats(self, p2p_status: dict[str, Any]) -> DiskStats:
        """Get disk usage statistics."""
        stats = DiskStats()

        # Get coordinator disk usage
        try:
            import shutil

            total, used, free = shutil.disk_usage("/")
            stats.coordinator_percent = (used / total) * 100
            stats.coordinator_free_gb = free / (1024**3)
        except Exception as e:
            logger.debug(f"Local disk check failed: {e}")

        # Get training node stats from P2P if available
        try:
            node_statuses = p2p_status.get("node_statuses", {})
            disk_percents = []
            above_80 = 0

            for node_id, node_info in node_statuses.items():
                if isinstance(node_info, dict):
                    disk_pct = node_info.get("disk_usage_percent", 0)
                    if disk_pct > 0:
                        disk_percents.append(disk_pct)
                        if disk_pct > 80:
                            above_80 += 1

            if disk_percents:
                stats.avg_training_node_percent = sum(disk_percents) / len(disk_percents)
                stats.nodes_above_80_percent = above_80

        except Exception as e:
            logger.debug(f"Node disk stats failed: {e}")

        return stats

    def _extract_selfplay_stats(self, p2p_status: dict[str, Any]) -> SelfplayStats:
        """Extract selfplay statistics from P2P status."""
        stats = SelfplayStats()

        try:
            job_stats = p2p_status.get("job_stats", {})
            stats.jobs_dispatched_total = job_stats.get("total_dispatched", 0)
            stats.jobs_active = job_stats.get("active", 0)
            stats.jobs_completed = job_stats.get("completed", 0)
            stats.jobs_failed = job_stats.get("failed", 0)

            # Calculate success rate
            total = stats.jobs_completed + stats.jobs_failed
            if total > 0:
                stats.dispatch_success_rate = stats.jobs_completed / total

            # Games count
            stats.games_generated = p2p_status.get("total_games", 0)

        except Exception as e:
            logger.debug(f"Selfplay stats extraction failed: {e}")

        return stats

    def _extract_training_stats(self, p2p_status: dict[str, Any]) -> TrainingStats:
        """Extract training statistics from P2P status."""
        stats = TrainingStats()

        try:
            training_info = p2p_status.get("training", {})
            stats.training_active = training_info.get("active", False)
            stats.active_configs = training_info.get("active_configs", [])
            stats.last_training_config = training_info.get("last_config", "")

            # Elo changes
            elo_data = p2p_status.get("elo_deltas", {})
            stats.last_elo_change = elo_data

            # Promotions count
            stats.models_promoted = p2p_status.get("promotions_count", 0)

        except Exception as e:
            logger.debug(f"Training stats extraction failed: {e}")

        return stats

    def _generate_alerts(self, status: AutonomousStatus) -> list[Alert]:
        """Generate alerts based on status."""
        alerts: list[Alert] = []

        # Critical daemon alerts
        critical_down = []
        for daemon in status.daemon_details:
            if daemon.name in self.CRITICAL_DAEMONS and not daemon.healthy:
                critical_down.append(daemon.name)

        if critical_down:
            alerts.append(
                Alert(
                    level="critical",
                    message=f"Critical daemons unhealthy: {', '.join(critical_down)}",
                    source="daemon_health",
                )
            )

        # System health alerts
        if status.system_health_score < 50:
            alerts.append(
                Alert(
                    level="critical",
                    message=f"System health critical: {status.system_health_score:.0f}%",
                    source="system_health",
                )
            )
        elif status.system_health_score < 70:
            alerts.append(
                Alert(
                    level="warning",
                    message=f"System health degraded: {status.system_health_score:.0f}%",
                    source="system_health",
                )
            )

        # Disk alerts
        if status.disk.coordinator_percent > 90:
            alerts.append(
                Alert(
                    level="critical",
                    message=f"Coordinator disk critical: {status.disk.coordinator_percent:.0f}%",
                    source="disk",
                )
            )
        elif status.disk.coordinator_percent > 80:
            alerts.append(
                Alert(
                    level="warning",
                    message=f"Coordinator disk high: {status.disk.coordinator_percent:.0f}%",
                    source="disk",
                )
            )

        if status.disk.nodes_above_80_percent > 0:
            alerts.append(
                Alert(
                    level="warning",
                    message=f"{status.disk.nodes_above_80_percent} nodes above 80% disk usage",
                    source="disk",
                )
            )

        # Cluster health
        if status.nodes_total > 0:
            alive_pct = (status.nodes_alive / status.nodes_total) * 100
            if alive_pct < 50:
                alerts.append(
                    Alert(
                        level="critical",
                        message=f"Less than 50% of nodes alive: {status.nodes_alive}/{status.nodes_total}",
                        source="cluster",
                    )
                )
            elif alive_pct < 70:
                alerts.append(
                    Alert(
                        level="warning",
                        message=f"Node availability below 70%: {status.nodes_alive}/{status.nodes_total}",
                        source="cluster",
                    )
                )

        # No leader
        if not status.leader_id:
            alerts.append(
                Alert(
                    level="critical",
                    message="No P2P leader elected",
                    source="cluster",
                )
            )

        # Selfplay dispatch issues
        if status.selfplay.dispatch_success_rate < 0.8 and status.selfplay.jobs_dispatched_total > 10:
            alerts.append(
                Alert(
                    level="warning",
                    message=f"Dispatch success rate low: {status.selfplay.dispatch_success_rate:.0%}",
                    source="selfplay",
                )
            )

        return alerts


def format_uptime(hours: float) -> str:
    """Format uptime as human-readable string."""
    if hours < 1:
        return f"{int(hours * 60)}m"
    elif hours < 24:
        h = int(hours)
        m = int((hours - h) * 60)
        return f"{h}h {m}m"
    else:
        d = int(hours / 24)
        h = int(hours % 24)
        return f"{d}d {h}h"


def get_health_color(score: float) -> str:
    """Get terminal color for health score."""
    if score >= 80:
        return GREEN
    elif score >= 60:
        return YELLOW
    else:
        return RED


def get_alert_color(level: str) -> str:
    """Get terminal color for alert level."""
    if level == "critical":
        return RED
    elif level == "warning":
        return YELLOW
    else:
        return CYAN


def print_dashboard(status: AutonomousStatus) -> None:
    """Print formatted dashboard to terminal."""
    # Clear screen
    print("\033[2J\033[H", end="")

    # Header
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  RINGRIFT AUTONOMOUS OPERATION MONITOR{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")
    print()

    # Uptime and timestamp
    print(f"{BOLD}Uptime:{RESET} {format_uptime(status.uptime_hours)} | "
          f"{BOLD}Target:{RESET} 48h | "
          f"{DIM}Updated: {status.timestamp.strftime('%H:%M:%S')}{RESET}")
    print()

    # System Health
    health_color = get_health_color(status.system_health_score)
    print(f"{BOLD}System Health:{RESET} "
          f"{health_color}{status.system_health_score:.0f}%{RESET} "
          f"({status.system_health_level})")
    print()

    # Daemons
    print(f"{BOLD}Daemons:{RESET} {status.daemons_healthy}/{status.daemons_total} healthy")
    unhealthy = [d for d in status.daemon_details if not d.healthy]
    if unhealthy:
        for d in unhealthy[:5]:  # Show first 5
            print(f"  {RED}x{RESET} {d.name}")
        if len(unhealthy) > 5:
            print(f"  {DIM}... and {len(unhealthy) - 5} more{RESET}")
    print()

    # Cluster
    if status.nodes_total > 0:
        alive_pct = (status.nodes_alive / status.nodes_total) * 100
        cluster_color = GREEN if alive_pct >= 80 else (YELLOW if alive_pct >= 60 else RED)
        print(f"{BOLD}Cluster:{RESET} "
              f"{cluster_color}{status.nodes_alive}/{status.nodes_total}{RESET} nodes alive "
              f"({alive_pct:.0f}%)")
        if status.leader_id:
            print(f"  Leader: {status.leader_id}")
    else:
        print(f"{BOLD}Cluster:{RESET} {RED}No P2P status{RESET}")
    print()

    # Selfplay
    print(f"{BOLD}Selfplay:{RESET}")
    print(f"  Active jobs: {status.selfplay.jobs_active} | "
          f"Completed: {status.selfplay.jobs_completed} | "
          f"Failed: {status.selfplay.jobs_failed}")
    print(f"  Games generated: {status.selfplay.games_generated:,}")
    if status.selfplay.jobs_completed + status.selfplay.jobs_failed > 0:
        rate_color = GREEN if status.selfplay.dispatch_success_rate >= 0.9 else (
            YELLOW if status.selfplay.dispatch_success_rate >= 0.7 else RED
        )
        print(f"  Success rate: {rate_color}{status.selfplay.dispatch_success_rate:.0%}{RESET}")
    print()

    # Training
    print(f"{BOLD}Training:{RESET}")
    if status.training.training_active:
        print(f"  {GREEN}ACTIVE{RESET} on: {', '.join(status.training.active_configs) or 'unknown'}")
    else:
        print(f"  {DIM}No active training{RESET}")
    if status.training.last_elo_change:
        print(f"  Recent Elo changes:")
        for config, delta in list(status.training.last_elo_change.items())[:3]:
            delta_color = GREEN if delta > 0 else (RED if delta < 0 else DIM)
            print(f"    {config}: {delta_color}{delta:+.0f}{RESET}")
    print(f"  Models promoted: {status.training.models_promoted}")
    print()

    # Disk
    print(f"{BOLD}Disk Usage:{RESET}")
    coord_color = GREEN if status.disk.coordinator_percent < 70 else (
        YELLOW if status.disk.coordinator_percent < 85 else RED
    )
    print(f"  Coordinator: {coord_color}{status.disk.coordinator_percent:.0f}%{RESET} "
          f"({status.disk.coordinator_free_gb:.1f} GB free)")
    if status.disk.avg_training_node_percent > 0:
        print(f"  Training nodes avg: {status.disk.avg_training_node_percent:.0f}%")
    if status.disk.nodes_above_80_percent > 0:
        print(f"  {YELLOW}Nodes >80%: {status.disk.nodes_above_80_percent}{RESET}")
    print()

    # Alerts
    if status.alerts:
        print(f"{BOLD}Alerts:{RESET} ({len(status.alerts)})")
        for alert in status.alerts:
            color = get_alert_color(alert.level)
            print(f"  {color}[{alert.level.upper()}]{RESET} {alert.message}")
    else:
        print(f"{BOLD}Alerts:{RESET} {GREEN}None{RESET}")
    print()

    # Collection errors
    if status.collection_errors:
        print(f"{DIM}Collection errors: {len(status.collection_errors)}{RESET}")

    # Footer
    print(f"{DIM}{'=' * 60}{RESET}")
    print(f"{DIM}Press Ctrl+C to exit{RESET}")


async def run_monitor(args: argparse.Namespace) -> None:
    """Run the monitor loop."""
    monitor = AutonomousMonitor()

    while True:
        status = await monitor.get_status()

        if args.json:
            # JSON output for automation
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif hasattr(obj, "__dataclass_fields__"):
                    return asdict(obj)
                return str(obj)

            output = {
                "timestamp": status.timestamp.isoformat(),
                "uptime_hours": status.uptime_hours,
                "system_health_score": status.system_health_score,
                "system_health_level": status.system_health_level,
                "daemons_healthy": status.daemons_healthy,
                "daemons_total": status.daemons_total,
                "nodes_alive": status.nodes_alive,
                "nodes_total": status.nodes_total,
                "leader_id": status.leader_id,
                "selfplay": asdict(status.selfplay),
                "training": asdict(status.training),
                "disk": asdict(status.disk),
                "alerts_count": len(status.alerts),
                "alerts": [asdict(a) for a in status.alerts],
            }
            print(json.dumps(output, default=json_serializer, indent=2))
        else:
            print_dashboard(status)

        if args.once:
            break

        await asyncio.sleep(args.interval)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor autonomous 48-hour training operations"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Update interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Show single snapshot and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (for automation)",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Start time for uptime calculation (ISO format)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        asyncio.run(run_monitor(args))
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")


if __name__ == "__main__":
    main()
