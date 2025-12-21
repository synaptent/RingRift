#!/usr/bin/env python3
"""Idle Node Alerter - Monitors cluster for idle GPU nodes.

Detects when GPU nodes are idle (no training/selfplay running) and can:
1. Log warnings for operator review
2. Send alerts via Slack webhook (if configured)
3. Auto-recover by submitting selfplay jobs (if enabled)

Usage:
    # Check and alert once
    python scripts/idle_node_alerter.py

    # Run as daemon with 5-minute interval
    python scripts/idle_node_alerter.py --daemon --interval 300

    # Enable auto-recovery (submit selfplay jobs to idle nodes)
    python scripts/idle_node_alerter.py --auto-recover

    # With Slack alerts
    SLACK_WEBHOOK_URL=https://hooks.slack.com/... python scripts/idle_node_alerter.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Cluster backend imports
try:
    from app.coordination.slurm_backend import (
        SlurmBackend,
        SlurmJob,
        SlurmPartition,
        get_slurm_backend,
    )
    HAS_SLURM = True
except ImportError:
    HAS_SLURM = False

# Optional Slack integration
try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# SSH options for direct node access
SSH_OPTS = "-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"

# Known cluster nodes (can be overridden via env)
GH200_NODES = os.environ.get("GH200_NODES", "").split(",") if os.environ.get("GH200_NODES") else [
    "lambda-gh200-a",
    "lambda-gh200-c",
    "lambda-gh200-d",
    "lambda-gh200-e",
    "lambda-gh200-f",
    "lambda-gh200-g",
    "lambda-gh200-h",
    "lambda-gh200-i",
    "lambda-gh200-k",
    "lambda-gh200-l",
]

H100_NODES = os.environ.get("H100_NODES", "").split(",") if os.environ.get("H100_NODES") else [
    "lambda-h100",
    "lambda-2xh100",
]

VAST_NODES = os.environ.get("VAST_NODES", "").split(",") if os.environ.get("VAST_NODES") else []

# Thresholds
GPU_IDLE_THRESHOLD = 10  # GPU utilization below this % is considered idle
IDLE_TIME_THRESHOLD = 300  # Node must be idle for this many seconds before alerting


@dataclass
class NodeStatus:
    """Status of a cluster node."""
    hostname: str
    reachable: bool = False
    gpu_utilization: float = 0.0
    gpu_memory_used_mb: int = 0
    gpu_memory_total_mb: int = 0
    training_processes: int = 0
    selfplay_processes: int = 0
    last_check: float = 0.0
    consecutive_idle_checks: int = 0

    @property
    def is_idle(self) -> bool:
        """Check if node is idle (no work running, low GPU util)."""
        return (
            self.reachable
            and self.gpu_utilization < GPU_IDLE_THRESHOLD
            and self.training_processes == 0
            and self.selfplay_processes == 0
        )

    @property
    def has_gpu_work(self) -> bool:
        """Check if node has active GPU work."""
        return self.gpu_utilization >= GPU_IDLE_THRESHOLD


@dataclass
class AlerterState:
    """State tracking for the alerter."""
    nodes: dict[str, NodeStatus] = field(default_factory=dict)
    alerts_sent: int = 0
    last_alert_time: float = 0.0
    recoveries_triggered: int = 0
    check_count: int = 0


class IdleNodeAlerter:
    """Monitors cluster nodes for idle GPU resources."""

    def __init__(
        self,
        auto_recover: bool = False,
        slack_webhook: str | None = None,
        alert_cooldown: int = 1800,  # 30 minutes between alerts
    ):
        self.auto_recover = auto_recover
        self.slack_webhook = slack_webhook or os.environ.get("SLACK_WEBHOOK_URL")
        self.alert_cooldown = alert_cooldown
        self.state = AlerterState()

        # SSH user by node type
        self.ssh_users = {
            "lambda": "ubuntu",
            "vast": "root",
        }

    def _get_ssh_user(self, hostname: str) -> str:
        """Get SSH user for a hostname."""
        if hostname.startswith("lambda") or hostname.startswith("100."):
            return "ubuntu"
        return "root"

    async def _ssh_command(self, hostname: str, command: str) -> tuple[int, str, str]:
        """Execute command on remote host via SSH."""
        user = self._get_ssh_user(hostname)
        ssh_cmd = f"ssh {SSH_OPTS} {user}@{hostname} '{command}'"

        try:
            proc = await asyncio.create_subprocess_shell(
                ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=15.0
            )
            return (
                proc.returncode or 0,
                stdout.decode().strip(),
                stderr.decode().strip(),
            )
        except asyncio.TimeoutError:
            return (-1, "", "SSH timeout")
        except Exception as e:
            return (-1, "", str(e))

    async def check_node(self, hostname: str) -> NodeStatus:
        """Check status of a single node."""
        status = self.state.nodes.get(hostname, NodeStatus(hostname=hostname))
        status.last_check = time.time()

        # Check reachability
        rc, stdout, _ = await self._ssh_command(hostname, "echo OK")
        if rc != 0 or stdout != "OK":
            status.reachable = False
            self.state.nodes[hostname] = status
            return status

        status.reachable = True

        # Get GPU utilization
        rc, stdout, _ = await self._ssh_command(
            hostname,
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader,nounits 2>/dev/null | head -1"
        )
        if rc == 0 and stdout:
            try:
                parts = stdout.split(",")
                if len(parts) >= 3:
                    status.gpu_utilization = float(parts[0].strip())
                    status.gpu_memory_used_mb = int(parts[1].strip())
                    status.gpu_memory_total_mb = int(parts[2].strip())
            except (ValueError, IndexError):
                pass

        # Count training/selfplay processes
        rc, stdout, _ = await self._ssh_command(
            hostname,
            "ps aux | grep -E 'train_nnue|train_policy|train.py' | grep -v grep | wc -l"
        )
        if rc == 0:
            try:
                status.training_processes = int(stdout)
            except ValueError:
                pass

        rc, stdout, _ = await self._ssh_command(
            hostname,
            "ps aux | grep -E 'selfplay|unified_ai_loop|run_distributed' | grep -v grep | wc -l"
        )
        if rc == 0:
            try:
                status.selfplay_processes = int(stdout)
            except ValueError:
                pass

        # Track consecutive idle checks
        if status.is_idle:
            status.consecutive_idle_checks += 1
        else:
            status.consecutive_idle_checks = 0

        self.state.nodes[hostname] = status
        return status

    async def check_all_nodes(self) -> list[NodeStatus]:
        """Check all known cluster nodes in parallel."""
        all_nodes = GH200_NODES + H100_NODES + VAST_NODES
        all_nodes = [n for n in all_nodes if n]  # Filter empty

        if not all_nodes:
            logger.warning("No cluster nodes configured")
            return []

        # Check nodes in parallel
        tasks = [self.check_node(hostname) for hostname in all_nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        statuses = []
        for result in results:
            if isinstance(result, NodeStatus):
                statuses.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Node check failed: {result}")

        self.state.check_count += 1
        return statuses

    def get_idle_nodes(self) -> list[NodeStatus]:
        """Get list of nodes that have been idle for multiple checks."""
        idle = []
        for status in self.state.nodes.values():
            # Require at least 2 consecutive idle checks (10+ minutes)
            if status.is_idle and status.consecutive_idle_checks >= 2:
                idle.append(status)
        return idle

    def send_slack_alert(self, message: str, color: str = "warning") -> bool:
        """Send alert to Slack webhook."""
        if not self.slack_webhook or not HAS_URLLIB:
            return False

        try:
            payload = {
                "attachments": [{
                    "color": color,
                    "text": message,
                    "footer": "RingRift Cluster Alerter",
                    "ts": int(time.time()),
                }]
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.slack_webhook,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")
            return False

    async def auto_recover_node(self, status: NodeStatus) -> bool:
        """Attempt to auto-recover an idle node by starting selfplay."""
        if not self.auto_recover:
            return False

        logger.info(f"[AutoRecover] Starting selfplay on {status.hostname}")

        # Determine board type based on node (GH200 = all configs, H100 = large boards)
        if "h100" in status.hostname.lower():
            board_type = "square19"
            num_players = 2
        else:
            # Round-robin through configs
            configs = [
                ("square8", 2), ("square8", 3), ("square8", 4),
                ("hex8", 2), ("hex8", 3), ("hex8", 4),
            ]
            idx = self.state.recoveries_triggered % len(configs)
            board_type, num_players = configs[idx]

        # Build selfplay command
        command = (
            f"cd ~/ringrift/ai-service && source venv/bin/activate && "
            f"PYTHONPATH=~/ringrift/ai-service nohup python scripts/unified_ai_loop.py "
            f"--board-type {board_type} --num-players {num_players} "
            f"> logs/auto_recover_{board_type}_{num_players}p.log 2>&1 &"
        )

        rc, stdout, stderr = await self._ssh_command(status.hostname, command)

        if rc == 0:
            logger.info(f"[AutoRecover] Started selfplay on {status.hostname}: {board_type}_{num_players}p")
            self.state.recoveries_triggered += 1
            return True
        else:
            logger.error(f"[AutoRecover] Failed on {status.hostname}: {stderr}")
            return False

    async def run_check_cycle(self) -> None:
        """Run a single check cycle."""
        logger.info("=== Cluster Health Check ===")

        # Check all nodes
        statuses = await self.check_all_nodes()

        # Summary stats
        reachable = sum(1 for s in statuses if s.reachable)
        active = sum(1 for s in statuses if s.has_gpu_work or s.training_processes > 0 or s.selfplay_processes > 0)
        idle = self.get_idle_nodes()

        logger.info(f"Nodes: {reachable}/{len(statuses)} reachable, {active} active, {len(idle)} idle")

        # Log each node status
        for status in sorted(statuses, key=lambda s: s.hostname):
            state = "IDLE" if status.is_idle else ("ACTIVE" if status.reachable else "OFFLINE")
            if status.reachable:
                logger.info(
                    f"  {status.hostname}: {state} "
                    f"(GPU: {status.gpu_utilization:.0f}%, "
                    f"train: {status.training_processes}, "
                    f"selfplay: {status.selfplay_processes})"
                )
            else:
                logger.info(f"  {status.hostname}: OFFLINE")

        # Handle idle nodes
        if idle:
            now = time.time()
            should_alert = (now - self.state.last_alert_time) > self.alert_cooldown

            idle_names = [s.hostname for s in idle]
            message = (
                f":warning: *{len(idle)} Idle GPU Nodes Detected*\n"
                f"Nodes: {', '.join(idle_names)}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            if should_alert:
                logger.warning(f"ALERT: {len(idle)} idle nodes: {idle_names}")
                if self.slack_webhook:
                    if self.send_slack_alert(message):
                        self.state.alerts_sent += 1
                        self.state.last_alert_time = now

            # Auto-recover idle nodes
            if self.auto_recover:
                for status in idle:
                    await self.auto_recover_node(status)

    async def run_daemon(self, interval: int = 300) -> None:
        """Run as daemon, checking at regular intervals."""
        logger.info(f"Starting idle node alerter daemon (interval: {interval}s)")

        while True:
            try:
                await self.run_check_cycle()
            except Exception as e:
                logger.error(f"Check cycle failed: {e}")

            await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor cluster for idle GPU nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon with periodic checks",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--auto-recover",
        action="store_true",
        help="Automatically start selfplay on idle nodes",
    )
    parser.add_argument(
        "--slack-webhook",
        type=str,
        help="Slack webhook URL for alerts (or set SLACK_WEBHOOK_URL env)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=1800,
        help="Cooldown between alerts in seconds (default: 1800)",
    )

    args = parser.parse_args()

    alerter = IdleNodeAlerter(
        auto_recover=args.auto_recover,
        slack_webhook=args.slack_webhook,
        alert_cooldown=args.cooldown,
    )

    if args.daemon:
        asyncio.run(alerter.run_daemon(interval=args.interval))
    else:
        asyncio.run(alerter.run_check_cycle())


if __name__ == "__main__":
    main()
