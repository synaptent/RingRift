#!/usr/bin/env python3
"""Cluster Speed-Up Script - Diagnose and fix cluster issues to maximize throughput.

This script:
1. Checks all nodes for connectivity, P2P status, GPU utilization
2. Identifies bottlenecks and idle resources
3. Provides actionable recommendations
4. Can automatically fix common issues

Usage:
    python scripts/cluster_speedup.py              # Full diagnosis
    python scripts/cluster_speedup.py --fix        # Diagnose and fix issues
    python scripts/cluster_speedup.py --restart-p2p  # Restart P2P on all nodes
    python scripts/cluster_speedup.py --deploy     # Deploy latest code to cluster
    python scripts/cluster_speedup.py --start-jobs # Start selfplay on idle nodes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"

# SSH keys
SSH_KEYS = {
    "~/.ssh/id_ed25519": os.path.expanduser("~/.ssh/id_ed25519"),
    "~/.ssh/id_cluster": os.path.expanduser("~/.ssh/id_cluster"),
    "~/.runpod/ssh/RunPod-Key-Go": os.path.expanduser("~/.runpod/ssh/RunPod-Key-Go"),
}


@dataclass
class NodeStatus:
    """Status of a single node."""
    name: str
    host: str
    port: int = 22

    # Connectivity
    ssh_ok: bool = False
    ssh_error: str = ""
    p2p_ok: bool = False
    p2p_error: str = ""

    # P2P details
    p2p_role: str = ""
    p2p_leader: str = ""
    p2p_peers: int = 0

    # Resources
    gpu_name: str = ""
    gpu_util: float = 0.0
    gpu_mem_used: float = 0.0
    gpu_mem_total: float = 0.0
    cpu_percent: float = 0.0
    disk_percent: float = 0.0

    # Jobs
    selfplay_jobs: int = 0
    training_jobs: int = 0
    screen_sessions: list[str] = field(default_factory=list)

    # Config
    role: str = ""
    p2p_voter: bool = False
    p2p_enabled: bool = False

    @property
    def is_idle(self) -> bool:
        """Node has GPU but isn't running jobs."""
        return self.gpu_name and self.gpu_util < 20 and self.selfplay_jobs == 0 and self.training_jobs == 0

    @property
    def status_icon(self) -> str:
        if not self.ssh_ok:
            return "❌"
        elif self.is_idle and self.p2p_enabled:
            return "😴"  # Idle
        elif not self.p2p_ok and self.p2p_enabled:
            return "⚠️"
        elif self.selfplay_jobs > 0 or self.training_jobs > 0:
            return "✅"
        else:
            return "🔵"


def load_config() -> dict:
    """Load cluster configuration."""
    if not CONFIG_PATH.exists():
        print(f"Error: Config not found at {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_ssh_cmd(host_config: dict) -> list[str]:
    """Build SSH command for a node."""
    host = host_config.get("ssh_host", "")
    user = host_config.get("ssh_user", "root")
    port = host_config.get("ssh_port", 22)
    key = host_config.get("ssh_key", "~/.ssh/id_cluster")
    key_path = os.path.expanduser(key)

    cmd = [
        "ssh", "-i", key_path,
        "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-o", "LogLevel=ERROR",
    ]
    if port != 22:
        cmd.extend(["-p", str(port)])
    cmd.append(f"{user}@{host}")
    return cmd


def check_node(name: str, config: dict) -> NodeStatus:
    """Check a single node's status."""
    status = NodeStatus(
        name=name,
        host=config.get("ssh_host", ""),
        port=config.get("ssh_port", 22),
        role=config.get("role", ""),
        p2p_voter=config.get("p2p_voter", False),
        p2p_enabled=config.get("p2p_enabled", False),
    )

    ssh_cmd = get_ssh_cmd(config)

    # Check SSH connectivity
    try:
        result = subprocess.run(
            ssh_cmd + ["echo", "ok"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            status.ssh_ok = True
        else:
            status.ssh_error = result.stderr.strip()[:80]
            return status
    except subprocess.TimeoutExpired:
        status.ssh_error = "SSH timeout"
        return status
    except Exception as e:
        status.ssh_error = str(e)[:80]
        return status

    # Check P2P status via HTTP
    if status.p2p_enabled:
        try:
            result = subprocess.run(
                ssh_cmd + ["curl", "-s", "--connect-timeout", "3", "http://localhost:8770/status"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    p2p_data = json.loads(result.stdout)
                    status.p2p_ok = True
                    status.p2p_role = p2p_data.get("role", "")
                    status.p2p_leader = p2p_data.get("leader_id", "")
                    status.p2p_peers = p2p_data.get("alive_peers", 0)
                    status.selfplay_jobs = p2p_data.get("local_selfplay_jobs", 0)
                    status.training_jobs = p2p_data.get("local_training_jobs", 0)
                except json.JSONDecodeError:
                    status.p2p_error = "Invalid JSON"
            else:
                status.p2p_error = "P2P not responding"
        except Exception as e:
            status.p2p_error = str(e)[:50]

    # Check GPU status
    try:
        result = subprocess.run(
            ssh_cmd + ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                       "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                status.gpu_name = parts[0].strip()
                status.gpu_util = float(parts[1])
                status.gpu_mem_used = float(parts[2])
                status.gpu_mem_total = float(parts[3])
    except (ValueError, IndexError):
        pass

    # Check screen sessions
    try:
        result = subprocess.run(
            ssh_cmd + ["screen", "-ls"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split("\n"):
            if "." in line and ("Detached" in line or "Attached" in line):
                parts = line.strip().split(".")
                if len(parts) >= 2:
                    session_name = parts[1].split("\t")[0]
                    status.screen_sessions.append(session_name)
    except (ValueError, IndexError):
        pass

    # Check disk usage
    try:
        result = subprocess.run(
            ssh_cmd + ["df", "-h", "/"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 5:
                    status.disk_percent = float(parts[4].rstrip("%"))
    except (ValueError, IndexError):
        pass

    return status


def run_ssh_command(name: str, config: dict, command: str, timeout: int = 30) -> tuple[bool, str]:
    """Run a command on a node via SSH."""
    ssh_cmd = get_ssh_cmd(config)
    try:
        result = subprocess.run(
            ssh_cmd + ["bash", "-c", command],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def restart_p2p(name: str, config: dict) -> bool:
    """Restart P2P orchestrator on a node."""
    print(f"  Restarting P2P on {name}...")

    ringrift_path = config.get("ringrift_path", "~/ringrift/ai-service")
    venv_activate = config.get("venv_activate", f"source {ringrift_path}/venv/bin/activate")

    # Kill existing P2P
    run_ssh_command(name, config, "pkill -f p2p_orchestrator || true")
    time.sleep(1)

    # Start new P2P
    start_cmd = f"""
cd {ringrift_path} && {venv_activate} && \
screen -dmS p2p bash -c 'PYTHONPATH=. python scripts/p2p_orchestrator.py 2>&1 | tee logs/p2p.log'
"""
    ok, output = run_ssh_command(name, config, start_cmd)

    if ok:
        print(f"    ✅ P2P restarted on {name}")
        return True
    else:
        print(f"    ❌ Failed to restart P2P on {name}: {output[:100]}")
        return False


def deploy_code(name: str, config: dict) -> bool:
    """Deploy latest code to a node."""
    print(f"  Deploying code to {name}...")

    ringrift_path = config.get("ringrift_path", "~/ringrift/ai-service")

    deploy_cmd = f"""
cd {ringrift_path}/.. && \
git fetch origin && \
git reset --hard origin/main && \
echo "Deployed: $(git log -1 --oneline)"
"""
    ok, output = run_ssh_command(name, config, deploy_cmd, timeout=60)

    if ok and "Deployed:" in output:
        print(f"    ✅ {name}: {output.split('Deployed:')[1].strip()[:60]}")
        return True
    else:
        print(f"    ❌ Failed on {name}: {output[:100]}")
        return False


def start_selfplay(name: str, config: dict, board_type: str = "hex8", num_players: int = 2) -> bool:
    """Start selfplay jobs on a node."""
    print(f"  Starting selfplay on {name}...")

    ringrift_path = config.get("ringrift_path", "~/ringrift/ai-service")
    venv_activate = config.get("venv_activate", f"source {ringrift_path}/venv/bin/activate")

    start_cmd = f"""
cd {ringrift_path} && {venv_activate} && \
screen -dmS selfplay bash -c 'PYTHONPATH=. python scripts/selfplay.py --board {board_type} --num-players {num_players} --engine gumbel --num-games 10000 2>&1 | tee logs/selfplay.log'
"""
    ok, output = run_ssh_command(name, config, start_cmd)

    if ok:
        print(f"    ✅ Selfplay started on {name} ({board_type}_{num_players}p)")
        return True
    else:
        print(f"    ❌ Failed on {name}: {output[:100]}")
        return False


def print_report(nodes: list[NodeStatus]) -> dict[str, Any]:
    """Print cluster status report and return summary."""
    print("\n" + "=" * 100)
    print("RINGRIFT CLUSTER STATUS REPORT")
    print("=" * 100 + "\n")

    # Summary counts
    total = len(nodes)
    ssh_ok = sum(1 for n in nodes if n.ssh_ok)
    p2p_ok = sum(1 for n in nodes if n.p2p_ok)
    p2p_enabled = sum(1 for n in nodes if n.p2p_enabled)
    idle_gpu = [n for n in nodes if n.is_idle]
    running_selfplay = sum(1 for n in nodes if n.selfplay_jobs > 0)
    running_training = sum(1 for n in nodes if n.training_jobs > 0)
    total_selfplay = sum(n.selfplay_jobs for n in nodes)
    total_training = sum(n.training_jobs for n in nodes)

    # Find leader
    leaders = set(n.p2p_leader for n in nodes if n.p2p_leader)
    leader = list(leaders)[0] if len(leaders) == 1 else ("NONE" if not leaders else f"SPLIT: {leaders}")

    print(f"SSH Connectivity:  {ssh_ok}/{total} nodes reachable")
    print(f"P2P Network:       {p2p_ok}/{p2p_enabled} P2P-enabled nodes online")
    print(f"Leader:            {leader}")
    print(f"Selfplay:          {running_selfplay} nodes running {total_selfplay} jobs")
    print(f"Training:          {running_training} nodes running {total_training} jobs")
    print(f"Idle GPUs:         {len(idle_gpu)} nodes with idle GPUs")
    print()

    # Detailed table
    print(f"{'Node':<22} {'St':<3} {'P2P':<6} {'GPU':<16} {'Util':<6} {'Jobs':<12} {'Screens'}")
    print("-" * 100)

    for n in sorted(nodes, key=lambda x: (not x.ssh_ok, not x.p2p_ok, -x.selfplay_jobs, x.name)):
        p2p_str = n.p2p_role[:5] if n.p2p_ok else ("OFF" if not n.p2p_enabled else "DOWN")
        gpu_str = n.gpu_name[:15] if n.gpu_name else "-"
        util_str = f"{n.gpu_util:.0f}%" if n.gpu_name else "-"
        jobs_str = f"S:{n.selfplay_jobs} T:{n.training_jobs}"
        screens_str = ", ".join(n.screen_sessions[:3]) if n.screen_sessions else "-"
        if len(n.screen_sessions) > 3:
            screens_str += f" +{len(n.screen_sessions) - 3}"

        print(f"{n.name:<22} {n.status_icon:<3} {p2p_str:<6} {gpu_str:<16} {util_str:<6} {jobs_str:<12} {screens_str}")

    # Issues and recommendations
    print("\n" + "-" * 100)
    print("ISSUES & RECOMMENDATIONS:")

    issues = []

    # Unreachable nodes
    unreachable = [n for n in nodes if not n.ssh_ok]
    if unreachable:
        issues.append(f"❌ {len(unreachable)} nodes unreachable via SSH: {', '.join(n.name for n in unreachable)}")

    # P2P down
    p2p_down = [n for n in nodes if n.ssh_ok and n.p2p_enabled and not n.p2p_ok]
    if p2p_down:
        issues.append(f"⚠️  {len(p2p_down)} nodes have P2P down: {', '.join(n.name for n in p2p_down)}")
        issues.append(f"   → Run: python scripts/cluster_speedup.py --restart-p2p")

    # No leader
    if leader == "NONE" or leader.startswith("SPLIT"):
        issues.append(f"🔴 CRITICAL: No P2P leader elected! Cluster coordination disabled.")
        issues.append(f"   → Run: curl -X POST http://<any-node>:8770/election/reset")

    # Idle GPUs
    if idle_gpu:
        issues.append(f"😴 {len(idle_gpu)} idle GPU nodes: {', '.join(n.name for n in idle_gpu)}")
        issues.append(f"   → Run: python scripts/cluster_speedup.py --start-jobs")

    # Low job counts
    if total_selfplay < 20 and len([n for n in nodes if n.gpu_name]) > 5:
        issues.append(f"📉 Low selfplay throughput: only {total_selfplay} jobs on {len([n for n in nodes if n.gpu_name])} GPU nodes")

    if not issues:
        issues.append("✅ No issues detected - cluster is operating normally")

    for issue in issues:
        print(f"  {issue}")

    print()

    return {
        "total": total,
        "ssh_ok": ssh_ok,
        "p2p_ok": p2p_ok,
        "leader": leader,
        "total_selfplay": total_selfplay,
        "total_training": total_training,
        "idle_gpu": len(idle_gpu),
        "issues": len([i for i in issues if i.startswith("❌") or i.startswith("🔴")]),
    }


def main():
    parser = argparse.ArgumentParser(description="Cluster speed-up and diagnostics")
    parser.add_argument("--fix", action="store_true", help="Automatically fix issues")
    parser.add_argument("--restart-p2p", action="store_true", help="Restart P2P on all nodes")
    parser.add_argument("--deploy", action="store_true", help="Deploy latest code to all nodes")
    parser.add_argument("--start-jobs", action="store_true", help="Start selfplay on idle nodes")
    parser.add_argument("--nodes", nargs="+", help="Specific nodes to check/fix")
    parser.add_argument("--parallel", type=int, default=12, help="Parallel checks")
    parser.add_argument("--board", default="hex8", help="Board type for selfplay")
    parser.add_argument("--players", type=int, default=2, help="Number of players for selfplay")
    args = parser.parse_args()

    config = load_config()
    hosts = config.get("hosts", {})

    # Filter to active nodes with P2P or GPU
    active_hosts = {
        name: cfg for name, cfg in hosts.items()
        if cfg.get("status") == "ready"
        and cfg.get("role") not in ("proxy", "coordinator")
        and (cfg.get("p2p_enabled") or cfg.get("gpu"))
    }

    # Filter to specific nodes if requested
    if args.nodes:
        active_hosts = {k: v for k, v in active_hosts.items() if k in args.nodes}

    print(f"Checking {len(active_hosts)} nodes...")

    # Check all nodes in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(check_node, name, cfg): name
            for name, cfg in active_hosts.items()
        }
        for future in as_completed(futures):
            results.append(future.result())

    # Print report
    summary = print_report(results)

    # Handle fix actions
    if args.restart_p2p or (args.fix and summary.get("p2p_ok", 0) < len([r for r in results if r.p2p_enabled])):
        print("\n" + "=" * 50)
        print("RESTARTING P2P ON NODES...")
        print("=" * 50)

        targets = [n for n in results if n.ssh_ok and n.p2p_enabled and (args.restart_p2p or not n.p2p_ok)]
        for node in targets:
            restart_p2p(node.name, active_hosts[node.name])

    if args.deploy:
        print("\n" + "=" * 50)
        print("DEPLOYING CODE TO NODES...")
        print("=" * 50)

        targets = [n for n in results if n.ssh_ok]
        for node in targets:
            deploy_code(node.name, active_hosts[node.name])

    if args.start_jobs or (args.fix and summary.get("idle_gpu", 0) > 0):
        print("\n" + "=" * 50)
        print("STARTING SELFPLAY ON IDLE NODES...")
        print("=" * 50)

        targets = [n for n in results if n.is_idle]
        for node in targets:
            start_selfplay(node.name, active_hosts[node.name], args.board, args.players)

    # Return exit code
    if summary.get("issues", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
