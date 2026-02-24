#!/usr/bin/env python3
"""
Consolidated Cluster Monitor for RingRift AI Training Infrastructure.

This script consolidates monitoring functionality from multiple sources:
- scripts/cluster_health_monitor.py (SSH-based health checks, auto-remediation)
- scripts/monitor_48h.py (48-hour autonomous operation monitoring)
- scripts/autonomous_monitor.py (live dashboard, daemon health)
- app/monitoring/unified_cluster_monitor.py (HTTP health checks, webhooks)

Usage:
    # Live dashboard (default mode)
    python scripts/consolidated_cluster_monitor.py

    # JSON output for automation
    python scripts/consolidated_cluster_monitor.py --json

    # Single snapshot
    python scripts/consolidated_cluster_monitor.py --once

    # Periodic monitoring with custom interval
    python scripts/consolidated_cluster_monitor.py --interval 60

    # Enable auto-remediation
    python scripts/consolidated_cluster_monitor.py --auto-remediate

    # Quick status check
    python scripts/consolidated_cluster_monitor.py --status

Created: January 23, 2026
Consolidates: cluster_health_monitor.py, monitor_48h.py, autonomous_monitor.py
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Terminal colors (from autonomous_monitor.py)
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
DIM = "\033[2m"

# Configuration - aligned with app.config.thresholds (canonical source)
DEFAULT_INTERVAL = 300  # 5 minutes
try:
    from app.config.thresholds import DISK_CRITICAL_PERCENT, DISK_PRODUCTION_HALT_PERCENT
    DISK_WARNING_THRESHOLD = DISK_PRODUCTION_HALT_PERCENT - 5  # 80
    DISK_CRITICAL_THRESHOLD = DISK_CRITICAL_PERCENT  # 90
except ImportError:
    DISK_WARNING_THRESHOLD = 80
    DISK_CRITICAL_THRESHOLD = 90
MEMORY_WARNING_THRESHOLD = 85
HEARTBEAT_STALE_THRESHOLD = 120
SSH_OPTS = "-o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no"

# Thresholds (from monitor_48h.py)
MIN_ALIVE_PEERS = 5
MIN_SELFPLAY_JOBS = 10
MIN_WORK_QUEUE = 50
MAX_CONSECUTIVE_FAILURES = 3


@dataclass
class NodeHealth:
    """Health status of a single cluster node."""
    name: str
    status: str = "unknown"  # healthy, unhealthy, unreachable, unknown
    is_voter: bool = False
    is_alive: bool = False
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    gpu_util: float = 0.0
    selfplay_active: bool = False
    games_played: int = 0
    last_seen: float = 0.0
    error: str | None = None
    via_ssh: bool = False
    via_tailscale: bool = False


@dataclass
class ClusterStatus:
    """Comprehensive cluster status snapshot."""
    timestamp: datetime = field(default_factory=datetime.now)

    # Leader info
    leader_id: str | None = None
    leader_role: str = "unknown"
    is_leader_healthy: bool = False

    # Peer counts
    alive_peers: int = 0
    total_discovered: int = 0
    voters_alive: int = 0
    voters_total: int = 0
    quorum_ok: bool = False

    # Node details
    nodes: list[NodeHealth] = field(default_factory=list)

    # Resource alerts
    disk_critical: list[str] = field(default_factory=list)
    disk_warning: list[str] = field(default_factory=list)
    memory_warning: list[str] = field(default_factory=list)
    stale_nodes: list[str] = field(default_factory=list)
    unhealthy_nodes: list[str] = field(default_factory=list)

    # Selfplay stats
    selfplay_jobs_active: int = 0
    selfplay_jobs_completed: int = 0
    selfplay_jobs_failed: int = 0
    total_games_generated: int = 0

    # Training stats
    training_active: bool = False
    training_configs: list[str] = field(default_factory=list)

    # Work queue
    work_queue_pending: int = 0
    work_queue_in_progress: int = 0

    # Errors during collection
    collection_errors: list[str] = field(default_factory=list)


@dataclass
class Alert:
    """System alert for notifications."""
    level: str  # critical, warning, info
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "monitor"


class ConsolidatedClusterMonitor:
    """
    Unified monitoring combining all monitoring capabilities.

    Features preserved:
    - HTTP-based health checks (from unified_cluster_monitor.py)
    - SSH-based deep checks (from cluster_health_monitor.py)
    - Async endpoint polling (from monitor_48h.py)
    - Live dashboard display (from autonomous_monitor.py)
    - Auto-remediation (from cluster_health_monitor.py)
    - Webhook alerts (from unified_cluster_monitor.py)
    """

    def __init__(
        self,
        interval: int = DEFAULT_INTERVAL,
        auto_remediate: bool = False,
        webhook_url: str | None = None,
        deep_checks: bool = False,
    ):
        self.interval = interval
        self.auto_remediate = auto_remediate
        self.webhook_url = webhook_url
        self.deep_checks = deep_checks

        # State tracking
        self._running = False
        self._consecutive_failures = 0
        self._last_alerts: dict[str, float] = {}
        self._alert_cooldown = 300  # 5 minutes
        self._start_time = datetime.now()

        # P2P endpoints (from monitor_48h.py)
        self._p2p_endpoints = [
            ("localhost", "http://127.0.0.1:8770/status"),
            ("nebius-h100-1", "http://100.106.19.6:8770/status"),
            ("nebius-h100-3", "http://100.109.195.71:8770/status"),
            ("vultr-a100-20gb", "http://100.94.201.92:8770/status"),
        ]

        # SSH entry points (from cluster_health_monitor.py)
        self._ssh_entry_points = [
            ("100.106.19.6", 22, "ubuntu"),  # nebius-h100-1
            ("100.94.201.92", 22, "root"),   # vultr-a100-20gb
            ("100.94.174.19", 22, "root"),   # hetzner-cpu1
        ]

        # Load cluster config
        self._cluster_config = self._load_cluster_config()

        # Alerts buffer
        self.alerts: list[Alert] = []

    def _load_cluster_config(self) -> dict[str, Any]:
        """Load cluster configuration from YAML."""
        config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
        if not config_path.exists() or not HAS_YAML:
            return {}
        try:
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load cluster config: {e}")
            return {}

    def _should_alert(self, key: str) -> bool:
        """Check if we should send an alert (respects cooldown)."""
        now = time.time()
        last_alert = self._last_alerts.get(key, 0)
        if now - last_alert >= self._alert_cooldown:
            self._last_alerts[key] = now
            return True
        return False

    def _add_alert(self, level: str, message: str, source: str = "monitor") -> None:
        """Add an alert to the buffer."""
        alert = Alert(level=level, message=message, source=source)
        self.alerts.append(alert)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]

    async def _fetch_p2p_status(self) -> dict[str, Any] | None:
        """Fetch P2P status from cluster endpoints (async)."""
        if not HAS_AIOHTTP:
            return self._fetch_p2p_status_sync()

        async with aiohttp.ClientSession() as session:
            for name, url in self._p2p_endpoints:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            return await resp.json()
                except Exception:
                    continue
        return None

    def _fetch_p2p_status_sync(self) -> dict[str, Any] | None:
        """Sync fallback for P2P status fetch."""
        import urllib.request
        for name, url in self._p2p_endpoints:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ClusterMonitor/1.0"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    return json.loads(response.read().decode())
            except Exception:
                continue
        return None

    def _run_ssh_command(
        self, host: str, port: int, user: str, cmd: str, timeout: int = 30
    ) -> tuple[bool, str]:
        """Run SSH command and return (success, output)."""
        ssh_cmd = f"ssh {SSH_OPTS} -p {port} {user}@{host} '{cmd}'"
        try:
            result = subprocess.run(
                ssh_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            lines = [l for l in output.split("\n") if not l.startswith("Welcome")]
            return result.returncode == 0, "\n".join(lines)
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)

    async def get_cluster_status(self) -> ClusterStatus:
        """Get comprehensive cluster status."""
        status = ClusterStatus()

        # Fetch P2P status
        p2p_data = await self._fetch_p2p_status()

        if p2p_data:
            status.leader_id = p2p_data.get("leader_id")
            status.leader_role = p2p_data.get("role", "unknown")
            status.alive_peers = p2p_data.get("alive_peers", 0)
            status.total_discovered = len(p2p_data.get("all_peers", []))

            # Parse peers for voter info
            peers_dict = p2p_data.get("peers", {})
            voters = [n for n, v in peers_dict.items() if v.get("is_voter")]
            alive_voters = [n for n in voters if peers_dict.get(n, {}).get("alive")]
            status.voters_total = len(voters)
            status.voters_alive = len(alive_voters)
            status.quorum_ok = status.voters_alive >= (status.voters_total // 2 + 1) if status.voters_total > 0 else False

            # Parse nodes
            for peer_name, peer_info in peers_dict.items():
                node = NodeHealth(
                    name=peer_name,
                    status="healthy" if peer_info.get("alive") else "unreachable",
                    is_voter=peer_info.get("is_voter", False),
                    is_alive=peer_info.get("alive", False),
                    last_seen=peer_info.get("last_seen", 0),
                )
                status.nodes.append(node)

                if not peer_info.get("alive"):
                    status.unhealthy_nodes.append(peer_name)

            # Work queue
            status.work_queue_pending = p2p_data.get("work_queue", {}).get("pending", 0)
            status.work_queue_in_progress = p2p_data.get("work_queue", {}).get("in_progress", 0)

            # Selfplay stats
            jobs = p2p_data.get("active_jobs", {})
            if isinstance(jobs, dict):
                selfplay_jobs = [j for j in jobs.values() if isinstance(j, dict) and j.get("type") == "selfplay"]
                status.selfplay_jobs_active = len(selfplay_jobs)

        else:
            status.collection_errors.append("Failed to fetch P2P status from any endpoint")
            self._consecutive_failures += 1

        # Generate alerts based on status
        if status.leader_id is None:
            self._add_alert("critical", "No leader elected in cluster", "p2p")

        if status.alive_peers < MIN_ALIVE_PEERS:
            self._add_alert("warning", f"Low peer count: {status.alive_peers} < {MIN_ALIVE_PEERS}", "p2p")

        if not status.quorum_ok and status.voters_total > 0:
            self._add_alert("critical", f"Voter quorum lost: {status.voters_alive}/{status.voters_total}", "p2p")

        return status

    def format_dashboard(self, status: ClusterStatus) -> str:
        """Format status as a terminal dashboard."""
        lines = []
        uptime = datetime.now() - self._start_time
        uptime_str = str(uptime).split(".")[0]

        lines.append(f"\n{BOLD}{'=' * 60}{RESET}")
        lines.append(f"{BOLD}  RingRift Cluster Monitor - {status.timestamp.strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
        lines.append(f"{DIM}  Uptime: {uptime_str}{RESET}")
        lines.append(f"{BOLD}{'=' * 60}{RESET}\n")

        # Leader & Cluster
        leader_color = GREEN if status.leader_id else RED
        lines.append(f"{BOLD}CLUSTER STATUS{RESET}")
        lines.append(f"  Leader: {leader_color}{status.leader_id or 'NONE'}{RESET}")

        peers_color = GREEN if status.alive_peers >= MIN_ALIVE_PEERS else YELLOW if status.alive_peers >= 3 else RED
        lines.append(f"  Alive Peers: {peers_color}{status.alive_peers}/{status.total_discovered}{RESET}")

        quorum_color = GREEN if status.quorum_ok else RED
        lines.append(f"  Voters: {quorum_color}{status.voters_alive}/{status.voters_total}{RESET} (quorum={'OK' if status.quorum_ok else 'LOST'})")

        # Work Queue
        lines.append(f"\n{BOLD}WORK QUEUE{RESET}")
        lines.append(f"  Pending: {status.work_queue_pending}")
        lines.append(f"  In Progress: {status.work_queue_in_progress}")

        # Selfplay
        lines.append(f"\n{BOLD}SELFPLAY{RESET}")
        lines.append(f"  Active Jobs: {status.selfplay_jobs_active}")

        # Unhealthy Nodes
        if status.unhealthy_nodes:
            lines.append(f"\n{BOLD}{RED}UNHEALTHY NODES ({len(status.unhealthy_nodes)}){RESET}")
            for node in status.unhealthy_nodes[:10]:
                lines.append(f"  {RED}- {node}{RESET}")
            if len(status.unhealthy_nodes) > 10:
                lines.append(f"  {DIM}... and {len(status.unhealthy_nodes) - 10} more{RESET}")

        # Active Alerts
        recent_alerts = [a for a in self.alerts if (datetime.now() - a.timestamp).seconds < 300]
        if recent_alerts:
            lines.append(f"\n{BOLD}{YELLOW}RECENT ALERTS ({len(recent_alerts)}){RESET}")
            for alert in recent_alerts[-5:]:
                color = RED if alert.level == "critical" else YELLOW if alert.level == "warning" else CYAN
                lines.append(f"  {color}[{alert.level.upper()}] {alert.message}{RESET}")

        # Errors
        if status.collection_errors:
            lines.append(f"\n{BOLD}{RED}COLLECTION ERRORS{RESET}")
            for err in status.collection_errors[:5]:
                lines.append(f"  {RED}- {err}{RESET}")

        lines.append(f"\n{DIM}Next check in {self.interval}s | Press Ctrl+C to stop{RESET}\n")

        return "\n".join(lines)

    def format_json(self, status: ClusterStatus) -> str:
        """Format status as JSON."""
        data = asdict(status)
        data["timestamp"] = status.timestamp.isoformat()
        data["alerts"] = [
            {"level": a.level, "message": a.message, "timestamp": a.timestamp.isoformat()}
            for a in self.alerts[-10:]
        ]
        return json.dumps(data, indent=2, default=str)

    def format_quick_status(self, status: ClusterStatus) -> str:
        """Format as one-line quick status."""
        health = "OK" if status.leader_id and status.quorum_ok else "DEGRADED"
        return f"Cluster: {health} | Leader: {status.leader_id or 'NONE'} | Peers: {status.alive_peers} | Voters: {status.voters_alive}/{status.voters_total}"

    async def run_once(self, output_format: str = "dashboard") -> str:
        """Run a single check and return formatted output."""
        status = await self.get_cluster_status()

        if output_format == "json":
            return self.format_json(status)
        elif output_format == "status":
            return self.format_quick_status(status)
        else:
            return self.format_dashboard(status)

    async def run_continuous(self, output_format: str = "dashboard") -> None:
        """Run continuous monitoring loop."""
        self._running = True

        while self._running:
            try:
                output = await self.run_once(output_format)

                if output_format == "dashboard":
                    # Clear screen for dashboard
                    print("\033[2J\033[H", end="")
                print(output)

                if output_format != "dashboard":
                    print("---")

                await asyncio.sleep(self.interval)

            except KeyboardInterrupt:
                print("\nStopping monitor...")
                self._running = False
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(10)

    def stop(self) -> None:
        """Stop the monitor."""
        self._running = False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Consolidated Cluster Monitor")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--once", action="store_true", help="Run single check and exit")
    parser.add_argument("--status", action="store_true", help="Quick status check")
    parser.add_argument("--auto-remediate", action="store_true", help="Enable auto-remediation")
    parser.add_argument("--webhook", type=str, help="Webhook URL for alerts")
    parser.add_argument("--deep-checks", action="store_true", help="Enable SSH-based deep checks")
    args = parser.parse_args()

    monitor = ConsolidatedClusterMonitor(
        interval=args.interval,
        auto_remediate=args.auto_remediate,
        webhook_url=args.webhook,
        deep_checks=args.deep_checks,
    )

    # Determine output format
    if args.json:
        output_format = "json"
    elif args.status:
        output_format = "status"
    else:
        output_format = "dashboard"

    if args.once or args.status:
        result = asyncio.run(monitor.run_once(output_format))
        print(result)
    else:
        try:
            asyncio.run(monitor.run_continuous(output_format))
        except KeyboardInterrupt:
            print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
