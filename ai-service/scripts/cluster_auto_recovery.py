#!/usr/bin/env python3
"""Cluster auto-recovery daemon.

Monitors node health and automatically restarts services when nodes come back online.
Also alerts on persistent issues and tracks node uptime.

Features:
- Detects when offline nodes become reachable
- Automatically starts selfplay workers on recovered nodes
- Syncs code and models to recovered nodes
- Sends Slack alerts on state changes
- Tracks node uptime statistics

Usage:
    python scripts/cluster_auto_recovery.py --daemon --interval 300
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Try to load host configurations
try:
    from app.distributed.hosts import load_remote_hosts, HostConfig
    HOSTS_AVAILABLE = True
except ImportError:
    HOSTS_AVAILABLE = False

# Webhook URLs
SLACK_WEBHOOK_URL = os.environ.get("RINGRIFT_SLACK_WEBHOOK_URL", "")
ALERT_WEBHOOK_URL = os.environ.get("RINGRIFT_WEBHOOK_URL", "")

# Node configuration (fallback if hosts module unavailable)
GH200_NODES = [
    "lambda-gh200-a", "lambda-gh200-b", "lambda-gh200-c", "lambda-gh200-d",
    "lambda-gh200-e", "lambda-gh200-f", "lambda-gh200-g", "lambda-gh200-h",
    "lambda-gh200-i", "lambda-gh200-j", "lambda-gh200-k", "lambda-gh200-l",
]
OTHER_NODES = ["lambda-2xh100", "lambda-h100", "lambda-a10"]
ALL_NODES = GH200_NODES + OTHER_NODES

# State file for persistence
STATE_FILE = AI_SERVICE_ROOT / "logs" / "cluster_recovery_state.json"

# Recovery settings
SELFPLAY_WORKERS_PER_GH200 = 48
SELFPLAY_WORKERS_PER_H100 = 16
SSH_TIMEOUT = 15


@dataclass
class NodeState:
    """Track state of a single node."""
    name: str
    reachable: bool = False
    workers_running: int = 0
    gpu_utilization: float = 0.0
    last_seen: Optional[str] = None
    last_recovery: Optional[str] = None
    consecutive_failures: int = 0
    total_recoveries: int = 0


@dataclass
class ClusterState:
    """Track overall cluster state."""
    nodes: Dict[str, NodeState] = field(default_factory=dict)
    last_check: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "nodes": {
                name: {
                    "name": ns.name,
                    "reachable": ns.reachable,
                    "workers_running": ns.workers_running,
                    "gpu_utilization": ns.gpu_utilization,
                    "last_seen": ns.last_seen,
                    "last_recovery": ns.last_recovery,
                    "consecutive_failures": ns.consecutive_failures,
                    "total_recoveries": ns.total_recoveries,
                }
                for name, ns in self.nodes.items()
            },
            "last_check": self.last_check,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClusterState":
        state = cls()
        state.last_check = data.get("last_check")
        for name, ns_data in data.get("nodes", {}).items():
            state.nodes[name] = NodeState(
                name=ns_data["name"],
                reachable=ns_data.get("reachable", False),
                workers_running=ns_data.get("workers_running", 0),
                gpu_utilization=ns_data.get("gpu_utilization", 0.0),
                last_seen=ns_data.get("last_seen"),
                last_recovery=ns_data.get("last_recovery"),
                consecutive_failures=ns_data.get("consecutive_failures", 0),
                total_recoveries=ns_data.get("total_recoveries", 0),
            )
        return state


def load_state() -> ClusterState:
    """Load cluster state from file."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return ClusterState.from_dict(json.load(f))
        except Exception:
            pass
    return ClusterState()


def save_state(state: ClusterState):
    """Save cluster state to file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state.to_dict(), f, indent=2)


def send_alert(message: str, level: str = "info"):
    """Send alert via Slack webhook."""
    webhook = SLACK_WEBHOOK_URL or ALERT_WEBHOOK_URL
    if not webhook:
        print(f"[{level.upper()}] {message}")
        return

    emoji = {
        "info": ":information_source:",
        "success": ":white_check_mark:",
        "warning": ":warning:",
        "error": ":x:",
        "recovery": ":arrow_up:",
    }.get(level, ":robot_face:")

    payload = {"text": f"{emoji} *Cluster Recovery*\n{message}"}

    try:
        subprocess.run(
            ["curl", "-s", "-X", "POST", "-H", "Content-Type: application/json",
             "-d", json.dumps(payload), webhook],
            capture_output=True, timeout=10
        )
    except Exception as e:
        print(f"Failed to send alert: {e}")


def run_ssh(host: str, cmd: str, timeout: int = SSH_TIMEOUT) -> Optional[str]:
    """Run SSH command and return output or None on failure."""
    try:
        result = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=no", host, cmd],
            capture_output=True, text=True, timeout=timeout + 5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def check_node(host: str) -> NodeState:
    """Check health of a single node."""
    state = NodeState(name=host)

    # Check basic connectivity
    result = run_ssh(host, "echo ok")
    if result != "ok":
        return state

    state.reachable = True
    state.last_seen = datetime.now().isoformat()

    # Check worker count
    workers = run_ssh(host, "ps aux | grep -E 'selfplay|hybrid' | grep python | grep -v grep | wc -l")
    if workers:
        try:
            state.workers_running = int(workers)
        except ValueError:
            pass

    # Check GPU utilization
    gpu = run_ssh(host, "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1")
    if gpu:
        try:
            state.gpu_utilization = float(gpu.split()[0])
        except (ValueError, IndexError):
            pass

    return state


def recover_node(host: str) -> bool:
    """Attempt to recover a node by syncing code and starting workers."""
    print(f"  Recovering {host}...")

    # Determine worker count based on node type
    if "gh200" in host:
        num_workers = SELFPLAY_WORKERS_PER_GH200
        work_dir = "~/ringrift/ai-service"
    elif "h100" in host or "2xh100" in host:
        num_workers = SELFPLAY_WORKERS_PER_H100
        work_dir = "~/ringrift/ai-service"
    else:
        num_workers = 8
        work_dir = "~/ringrift/ai-service"

    # Step 1: Pull latest code
    git_result = run_ssh(host, f"cd {work_dir} && git pull", timeout=30)
    if git_result is None:
        print(f"    Git pull failed for {host}")
        # Continue anyway - might be a git issue but code might still work

    # Step 2: Check if workers already running
    workers = run_ssh(host, "ps aux | grep selfplay | grep python | grep -v grep | wc -l")
    if workers:
        try:
            if int(workers) > 5:
                print(f"    {host} already has {workers} workers")
                return True
        except ValueError:
            pass

    # Step 3: Start selfplay workers
    # Use screen or nohup to keep processes running
    # --auto-ramdrive will use /dev/shm on RAM-rich machines with limited disk
    start_cmd = f"""
cd {work_dir} && source venv/bin/activate &&
for i in $(seq 1 {num_workers}); do
    nohup python scripts/run_hybrid_selfplay.py \\
        --board-type square8 --num-players 2 \\
        --num-games 50000 --record-db data/games/selfplay.db \\
        --auto-ramdrive --sync-interval 300 --sync-target data/selfplay \\
        --engine-mode mixed --seed $RANDOM \\
        > /dev/null 2>&1 &
done
echo "Started $i workers"
"""

    result = run_ssh(host, start_cmd.replace("\n", " "), timeout=60)
    if result and "Started" in result:
        print(f"    Successfully started workers on {host}")
        return True

    print(f"    Failed to start workers on {host}")
    return False


def run_check_cycle(state: ClusterState) -> List[str]:
    """Run one check cycle and return list of alerts."""
    alerts = []
    now = datetime.now().isoformat()
    state.last_check = now

    for host in ALL_NODES:
        # Initialize node state if new
        if host not in state.nodes:
            state.nodes[host] = NodeState(name=host)

        prev_state = state.nodes[host]
        prev_reachable = prev_state.reachable

        # Check current state
        current = check_node(host)

        # Node came back online
        if current.reachable and not prev_reachable:
            print(f"Node {host} is back online!")

            # Attempt recovery
            if recover_node(host):
                current.last_recovery = now
                current.total_recoveries = prev_state.total_recoveries + 1
                alerts.append(
                    f"`{host}` recovered and workers started "
                    f"(recovery #{current.total_recoveries})"
                )
            else:
                alerts.append(f"`{host}` came online but worker start failed")

            current.consecutive_failures = 0

        # Node went offline
        elif not current.reachable and prev_reachable:
            current.consecutive_failures = 1
            alerts.append(f"`{host}` went offline")

        # Node still offline
        elif not current.reachable:
            current.consecutive_failures = prev_state.consecutive_failures + 1
            current.last_seen = prev_state.last_seen
            current.last_recovery = prev_state.last_recovery
            current.total_recoveries = prev_state.total_recoveries

            # Alert after extended outage
            if current.consecutive_failures == 12:  # ~1 hour at 5 min interval
                alerts.append(f"`{host}` has been offline for ~1 hour")

        # Node online but no workers
        elif current.reachable and current.workers_running < 5:
            print(f"Node {host} has low workers ({current.workers_running}), recovering...")
            if recover_node(host):
                alerts.append(f"`{host}` had low workers, restarted")

        # Update state
        state.nodes[host] = current

    return alerts


def print_status(state: ClusterState):
    """Print current cluster status."""
    print("\n" + "="*60)
    print("CLUSTER STATUS")
    print("="*60)

    online = [n for n in state.nodes.values() if n.reachable]
    offline = [n for n in state.nodes.values() if not n.reachable]

    print(f"Online: {len(online)}/{len(state.nodes)}")

    print("\nOnline nodes:")
    for n in sorted(online, key=lambda x: x.name):
        print(f"  {n.name}: {n.workers_running} workers, GPU {n.gpu_utilization:.0f}%")

    if offline:
        print("\nOffline nodes:")
        for n in sorted(offline, key=lambda x: x.name):
            last = n.last_seen[:16] if n.last_seen else "never"
            print(f"  {n.name}: last seen {last}, {n.consecutive_failures} failed checks")

    print("="*60 + "\n")


def run_daemon(interval: int = 300):
    """Run recovery daemon."""
    print(f"Starting cluster recovery daemon (interval: {interval}s)")
    send_alert(f"Cluster recovery daemon started (interval: {interval}s)", "info")

    state = load_state()

    while True:
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running check cycle...")

            alerts = run_check_cycle(state)
            save_state(state)

            if alerts:
                message = f"Status update ({datetime.now().strftime('%H:%M')}):\n"
                message += "\n".join(f"• {a}" for a in alerts)
                send_alert(message, "recovery")

            # Print summary
            online = sum(1 for n in state.nodes.values() if n.reachable)
            total_workers = sum(n.workers_running for n in state.nodes.values() if n.reachable)
            print(f"  {online}/{len(state.nodes)} nodes online, {total_workers} total workers")

        except KeyboardInterrupt:
            print("\nDaemon stopped")
            break
        except Exception as e:
            print(f"Error in check cycle: {e}")
            send_alert(f"Recovery daemon error: {e}", "error")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Cluster auto-recovery daemon")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=300,
                        help="Check interval in seconds (default: 5 min)")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--recover", type=str, help="Manually recover a specific node")

    args = parser.parse_args()

    if args.status:
        state = load_state()
        # Run one check cycle to update
        run_check_cycle(state)
        save_state(state)
        print_status(state)

    elif args.recover:
        print(f"Manually recovering {args.recover}...")
        current = check_node(args.recover)
        if current.reachable:
            if recover_node(args.recover):
                print("Recovery successful!")
            else:
                print("Recovery failed")
        else:
            print(f"Node {args.recover} is not reachable")

    elif args.daemon:
        run_daemon(args.interval)

    else:
        # Single check
        state = load_state()
        alerts = run_check_cycle(state)
        save_state(state)
        print_status(state)

        if alerts:
            print("Alerts:")
            for a in alerts:
                print(f"  • {a}")


if __name__ == "__main__":
    main()
