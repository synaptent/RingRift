#!/usr/bin/env python3
"""Continuous cluster monitoring script for RingRift AI training loop.

Monitors every 20 seconds:
- Cluster health (active nodes, leader status)
- Selfplay job counts
- Training job status (running, failed)
- Resource utilization (CPU, memory, GPU)
- Data sync status

Usage:
    python scripts/cluster_monitor.py [--interval 20] [--leader-url http://150.136.65.197:8770]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)

from scripts.lib.logging_config import setup_script_logging, get_metrics_logger
from scripts.lib.alerts import (
    Alert,
    AlertSeverity,
    AlertType,
    AlertThresholds,
    AlertManager,
    create_alert,
)

# Alias for backwards compatibility
MonitorAlert = Alert


@dataclass
class HealthStatus:
    """Health status response from leader."""
    is_healthy: bool = False
    error: Optional[str] = None
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    selfplay_jobs: int = 0
    cluster_utilization: Dict[str, Any] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "HealthStatus":
        """Create from API response."""
        if "error" in data:
            return cls(is_healthy=False, error=data["error"], raw_response=data)
        return cls(
            is_healthy=True,
            cpu_percent=data.get("cpu_percent", 0),
            memory_percent=data.get("memory_percent", 0),
            disk_percent=data.get("disk_percent", 0),
            selfplay_jobs=data.get("selfplay_jobs", 0),
            cluster_utilization=data.get("cluster_utilization", {}),
            raw_response=data,
        )


@dataclass
class LeaderStatus:
    """Leader status response."""
    node_id: str = "unknown"
    role: str = "unknown"
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "LeaderStatus":
        """Create from API response."""
        return cls(
            node_id=data.get("node_id", "unknown"),
            role=data.get("role", "unknown"),
            raw_response=data,
        )


@dataclass
class TrainingJob:
    """A training job."""
    job_id: str = ""
    job_type: str = ""
    status: str = ""
    board_type: str = ""
    num_players: int = 0
    error_message: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingJob":
        """Create from job dict."""
        return cls(
            job_id=data.get("job_id", ""),
            job_type=data.get("job_type", ""),
            status=data.get("status", ""),
            board_type=data.get("board_type", ""),
            num_players=data.get("num_players", 0),
            error_message=data.get("error_message"),
        )


@dataclass
class TrainingStatus:
    """Training status response."""
    jobs: List[TrainingJob] = field(default_factory=list)
    auto_nnue_enabled: bool = False
    auto_cmaes_enabled: bool = False
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "TrainingStatus":
        """Create from API response."""
        jobs = [TrainingJob.from_dict(j) for j in data.get("jobs", [])]
        thresholds = data.get("thresholds", {})
        return cls(
            jobs=jobs,
            auto_nnue_enabled=thresholds.get("auto_nnue_enabled", False),
            auto_cmaes_enabled=thresholds.get("auto_cmaes_enabled", False),
            raw_response=data,
        )

    @property
    def running_nnue_count(self) -> int:
        """Count of running NNUE jobs."""
        return len([j for j in self.jobs if j.status == "running" and j.job_type == "nnue"])

    @property
    def running_cmaes_count(self) -> int:
        """Count of running CMA-ES jobs."""
        return len([j for j in self.jobs if j.status == "running" and j.job_type == "cmaes"])

    @property
    def failed_jobs(self) -> List[TrainingJob]:
        """List of failed jobs."""
        return [j for j in self.jobs if j.status == "failed"]


@dataclass
class MonitorConfig:
    """Configuration for the cluster monitor."""
    leader_url: str
    interval: int = 20
    request_timeout: int = 10
    disk_warning_threshold: float = 70.0  # % - enforced 2025-12-15
    memory_warning_threshold: float = 95.0  # %


class ClusterMonitor:
    """Continuous cluster health monitor."""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.leader_url = config.leader_url.rstrip("/")
        self.check_count = 0
        self.alerts_history: List[Alert] = []

        # Set up logging
        self.logger = setup_script_logging("cluster_monitor")
        self.metrics = get_metrics_logger("cluster_monitor")

    async def fetch_json(self, endpoint: str) -> Dict[str, Any]:
        """Fetch JSON from an endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.leader_url}{endpoint}",
                    timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {"error": f"HTTP {resp.status}"}
        except aiohttp.ClientError as e:
            self.logger.warning(f"HTTP error fetching {endpoint}: {e}")
            return {"error": str(e)}
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching {endpoint}")
            return {"error": "Request timed out"}
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {endpoint}: {e}")
            return {"error": str(e)}

    async def check_health(self) -> HealthStatus:
        """Check cluster health via leader."""
        data = await self.fetch_json("/health")
        return HealthStatus.from_response(data)

    async def check_status(self) -> LeaderStatus:
        """Get leader status."""
        data = await self.fetch_json("/status")
        return LeaderStatus.from_response(data)

    async def check_training(self) -> TrainingStatus:
        """Get training job status."""
        data = await self.fetch_json("/training/status")
        return TrainingStatus.from_response(data)

    def generate_alerts(
        self,
        health: HealthStatus,
        status: LeaderStatus,
        training: TrainingStatus,
    ) -> List[Alert]:
        """Generate alerts based on current status."""
        alerts: List[Alert] = []

        # Check if leader is responsive
        if health.error:
            alerts.append(create_alert(
                AlertSeverity.CRITICAL,
                AlertType.LEADER_UNREACHABLE,
                f"Leader unreachable - {health.error}",
                source="cluster_monitor",
            ))
            return alerts

        # Check disk usage (70% limit enforced 2025-12-15)
        if health.disk_percent > self.config.disk_warning_threshold:
            alerts.append(create_alert(
                AlertSeverity.WARNING,
                AlertType.HIGH_DISK_USAGE,
                f"Disk usage {health.disk_percent:.1f}% on leader",
                details={"disk_percent": health.disk_percent},
                source="cluster_monitor",
            ))

        # Check memory usage
        if health.memory_percent > self.config.memory_warning_threshold:
            alerts.append(create_alert(
                AlertSeverity.WARNING,
                AlertType.HIGH_MEMORY_USAGE,
                f"Memory usage {health.memory_percent:.1f}% on leader",
                details={"memory_percent": health.memory_percent},
                source="cluster_monitor",
            ))

        # Check for failed training jobs
        for job in training.failed_jobs[-3:]:  # Report last 3 failures
            alerts.append(create_alert(
                AlertSeverity.CRITICAL,
                AlertType.TRAINING_FAILED,
                f"{job.job_type} {job.board_type}_{job.num_players}p - {job.error_message or 'unknown'}",
                details={"job_id": job.job_id, "job_type": job.job_type},
                source="cluster_monitor",
            ))

        # Check if leader role is as expected
        if status.role != "leader":
            alerts.append(create_alert(
                AlertSeverity.WARNING,
                AlertType.ROLE_MISMATCH,
                f"Node {status.node_id} is {status.role}, not leader",
                source="cluster_monitor",
            ))

        return alerts

    def format_timestamp(self) -> str:
        """Format current timestamp."""
        return datetime.now().strftime("%H:%M:%S")

    def print_status(
        self,
        health: HealthStatus,
        status: LeaderStatus,
        training: TrainingStatus,
        alerts: List[Alert],
    ) -> None:
        """Print formatted status."""
        timestamp = self.format_timestamp()
        selfplay_rate = health.cluster_utilization.get("selfplay_rate", 0)

        # Print status lines
        print(f"\n[{timestamp}] === Cluster Status (check #{self.check_count}) ===")
        print(f"  Leader: {status.node_id} ({status.role})")
        print(f"  Selfplay: {health.selfplay_jobs} jobs | Rate: {selfplay_rate}/hr")
        print(f"  Training: {training.running_nnue_count} NNUE + {training.running_cmaes_count} CMA-ES running | {len(training.failed_jobs)} failed")
        print(f"  Resources: CPU {health.cpu_percent:.0f}% | Mem {health.memory_percent:.1f}% | Disk {health.disk_percent:.1f}%")
        print(f"  Auto-train: NNUE={'ON' if training.auto_nnue_enabled else 'OFF'} | CMA-ES={'ON' if training.auto_cmaes_enabled else 'OFF'}")

        # Print alerts
        if alerts:
            print("  ALERTS:")
            for alert in alerts:
                print(f"    ! {alert}")
        else:
            print("  Status: OK")

    def update_metrics(
        self,
        health: HealthStatus,
        training: TrainingStatus,
        alerts: List[Alert],
    ) -> None:
        """Update Prometheus-style metrics."""
        self.metrics.set("cpu_percent", health.cpu_percent)
        self.metrics.set("memory_percent", health.memory_percent)
        self.metrics.set("disk_percent", health.disk_percent)
        self.metrics.set("selfplay_jobs", health.selfplay_jobs)
        self.metrics.set("training_nnue_running", training.running_nnue_count)
        self.metrics.set("training_cmaes_running", training.running_cmaes_count)
        self.metrics.set("training_failed", len(training.failed_jobs))
        self.metrics.set("alert_count", len(alerts))
        self.metrics.increment("checks_total")

    async def run_once(self) -> tuple[HealthStatus, LeaderStatus, TrainingStatus, List[Alert]]:
        """Run a single monitoring check."""
        self.check_count += 1

        # Fetch all status info in parallel
        health_task = asyncio.create_task(self.check_health())
        status_task = asyncio.create_task(self.check_status())
        training_task = asyncio.create_task(self.check_training())

        health = await health_task
        status = await status_task
        training = await training_task

        # Generate alerts
        alerts = self.generate_alerts(health, status, training)
        self.alerts_history.extend(alerts)

        # Update metrics
        self.update_metrics(health, training, alerts)

        # Print status
        self.print_status(health, status, training, alerts)

        # Log structured data
        self.logger.info(
            "Cluster check completed",
            extra={
                "check_number": self.check_count,
                "leader_healthy": health.is_healthy,
                "selfplay_jobs": health.selfplay_jobs,
                "training_running": training.running_nnue_count + training.running_cmaes_count,
                "alert_count": len(alerts),
            }
        )

        return health, status, training, alerts

    async def run(self) -> None:
        """Run continuous monitoring."""
        self.logger.info(f"Starting cluster monitoring (interval: {self.config.interval}s)")
        self.logger.info(f"Leader URL: {self.leader_url}")
        print(f"Starting cluster monitoring (interval: {self.config.interval}s)")
        print(f"Leader URL: {self.leader_url}")
        print("-" * 60)

        while True:
            try:
                await self.run_once()
            except Exception as e:
                self.logger.error(f"Monitor error: {e}", exc_info=True)
                print(f"\n[{self.format_timestamp()}] Monitor error: {e}")

            await asyncio.sleep(self.config.interval)


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor RingRift cluster")
    parser.add_argument(
        "--leader-url",
        default="http://150.136.65.197:8770",
        help="URL of the leader node"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=20,
        help="Check interval in seconds (default: 20)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit"
    )
    parser.add_argument(
        "--disk-threshold",
        type=float,
        default=70.0,
        help="Disk usage warning threshold %% (default: 70)"
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=95.0,
        help="Memory usage warning threshold %% (default: 95)"
    )

    args = parser.parse_args()

    config = MonitorConfig(
        leader_url=args.leader_url,
        interval=args.interval,
        disk_warning_threshold=args.disk_threshold,
        memory_warning_threshold=args.memory_threshold,
    )

    monitor = ClusterMonitor(config)

    if args.once:
        await monitor.run_once()
        return 0
    else:
        await monitor.run()
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
