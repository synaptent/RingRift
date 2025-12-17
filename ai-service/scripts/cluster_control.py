#!/usr/bin/env python
"""
RingRift Cluster Control - Unified management for distributed training infrastructure.

This script provides a single entry point to:
- View status of all hosts, improvement loops, and GPU selfplay
- Start/stop improvement loops on GPU hosts
- Start/stop GPU selfplay on Vast instances
- Sync training data between hosts

Usage:
    # Show full cluster status
    python scripts/cluster_control.py status

    # Show compact status
    python scripts/cluster_control.py status --compact

    # Start all improvement loops
    python scripts/cluster_control.py loops start

    # Stop all improvement loops
    python scripts/cluster_control.py loops stop

    # Start GPU selfplay on all Vast instances
    python scripts/cluster_control.py selfplay start

    # Sync training data from all hosts to local
    python scripts/cluster_control.py sync

    # Monitor cluster in real-time
    python scripts/cluster_control.py monitor
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# Allow imports from app/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.distributed.hosts import (
    HostConfig,
    SSHExecutor,
    load_remote_hosts,
)
from scripts.cluster_manager import (
    ImprovementLoopManager,
    ImprovementLoopStatus,
)


@dataclass
class HostStatus:
    """Status of a single host."""
    name: str
    reachable: bool
    gpu_info: Optional[Dict[str, Any]] = None
    improvement_loops: List[ImprovementLoopStatus] = field(default_factory=list)
    selfplay_running: bool = False
    selfplay_board: Optional[str] = None
    selfplay_progress: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ClusterStatus:
    """Full cluster status."""
    timestamp: datetime
    hosts: List[HostStatus] = field(default_factory=list)
    total_loops_running: int = 0
    total_selfplay_running: int = 0
    total_gpus: int = 0
    total_gpu_memory_gb: float = 0


def check_host_status(host_config: HostConfig) -> HostStatus:
    """Check status of a single host."""
    status = HostStatus(name=host_config.name, reachable=False)
    executor = SSHExecutor(host_config)

    try:
        # Check if reachable
        result = executor.run("echo ok", timeout=10)
        if result.returncode != 0 or "ok" not in result.stdout:
            status.error = "Host unreachable"
            return status
        status.reachable = True

        # Check GPU info
        gpu_result = executor.run(
            "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader,nounits 2>/dev/null || echo 'no-gpu'",
            timeout=15,
        )
        if gpu_result.returncode == 0 and "no-gpu" not in gpu_result.stdout:
            gpus = []
            for line in gpu_result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "utilization": int(parts[2]),
                            "memory_used_mb": int(parts[3]),
                            "memory_total_mb": int(parts[4]),
                        })
            status.gpu_info = {"gpus": gpus}

        # Check selfplay process
        selfplay_result = executor.run(
            "ps aux | grep run_gpu_selfplay | grep -v grep | head -1",
            timeout=10,
        )
        if selfplay_result.returncode == 0 and selfplay_result.stdout.strip():
            status.selfplay_running = True
            # Extract board type from command line
            cmd_line = selfplay_result.stdout
            if "--board" in cmd_line:
                for part in cmd_line.split():
                    if part in ["square8", "square19", "hexagonal", "hex"]:
                        status.selfplay_board = part
                        break

            # Get progress from log
            progress_result = executor.run(
                "tail -1 logs/gpu_selfplay.log 2>/dev/null | grep -E 'Game|Progress'",
                timeout=10,
            )
            if progress_result.returncode == 0 and progress_result.stdout.strip():
                status.selfplay_progress = progress_result.stdout.strip()[:100]

    except Exception as e:
        status.error = str(e)

    return status


def get_cluster_status(hosts: Optional[List[str]] = None) -> ClusterStatus:
    """Get full cluster status."""
    all_hosts = load_remote_hosts()

    if hosts:
        filtered = {name: cfg for name, cfg in all_hosts.items() if name in hosts}
    else:
        filtered = all_hosts

    status = ClusterStatus(timestamp=datetime.now())

    # Check host status in parallel
    with ThreadPoolExecutor(max_workers=min(len(filtered), 8)) as pool:
        futures = {pool.submit(check_host_status, cfg): name for name, cfg in filtered.items()}

        for future in as_completed(futures):
            host_status = future.result()
            status.hosts.append(host_status)

            if host_status.gpu_info:
                for gpu in host_status.gpu_info.get("gpus", []):
                    status.total_gpus += 1
                    status.total_gpu_memory_gb += gpu.get("memory_total_mb", 0) / 1024

            if host_status.selfplay_running:
                status.total_selfplay_running += 1

    # Check improvement loops on GPU hosts
    gpu_hosts = [
        name for name, cfg in filtered.items()
        if cfg.properties.get("gpu") or cfg.properties.get("role", "").startswith("nn_")
    ]
    if gpu_hosts:
        loop_mgr = ImprovementLoopManager(hosts=gpu_hosts)
        loop_statuses = loop_mgr.get_cluster_loop_status()

        for host_status in status.hosts:
            if host_status.name in loop_statuses:
                host_status.improvement_loops = loop_statuses[host_status.name]
                status.total_loops_running += sum(
                    1 for ls in host_status.improvement_loops if ls.is_running
                )

    return status


def print_status(status: ClusterStatus, compact: bool = False) -> None:
    """Print cluster status to console."""
    print(f"\n{'=' * 80}")
    print(f"RingRift Cluster Status - {status.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")
    print(f"Total GPUs: {status.total_gpus} ({status.total_gpu_memory_gb:.0f} GB VRAM)")
    print(f"Improvement Loops Running: {status.total_loops_running}")
    print(f"GPU Selfplay Running: {status.total_selfplay_running}")

    # Sort hosts: reachable first, then by name
    sorted_hosts = sorted(status.hosts, key=lambda h: (not h.reachable, h.name))

    for host in sorted_hosts:
        print(f"\n{'-' * 40}")
        reachable_str = "ONLINE" if host.reachable else "OFFLINE"
        print(f"{host.name:20} [{reachable_str}]")

        if not host.reachable:
            if host.error:
                print(f"  Error: {host.error}")
            continue

        # GPU info
        if host.gpu_info and host.gpu_info.get("gpus"):
            for gpu in host.gpu_info["gpus"]:
                mem_pct = (gpu["memory_used_mb"] / gpu["memory_total_mb"] * 100) if gpu["memory_total_mb"] else 0
                print(f"  GPU {gpu['index']}: {gpu['name'][:20]:20} "
                      f"Util: {gpu['utilization']:3}% "
                      f"Mem: {gpu['memory_used_mb']:5}MB/{gpu['memory_total_mb']:5}MB ({mem_pct:.0f}%)")

        # Improvement loops
        if host.improvement_loops:
            print("  Improvement Loops:")
            for loop in host.improvement_loops:
                state = "RUNNING" if loop.is_running else "stopped"
                pid_str = f"PID={loop.pid}" if loop.pid else ""
                print(f"    {loop.board_type:5} {state:8} {pid_str}")
                if loop.progress and not compact:
                    prog_short = loop.progress[:60] + "..." if len(loop.progress) > 60 else loop.progress
                    print(f"          {prog_short}")

        # Selfplay
        if host.selfplay_running:
            print(f"  GPU Selfplay: RUNNING ({host.selfplay_board or 'unknown board'})")
            if host.selfplay_progress and not compact:
                print(f"          {host.selfplay_progress}")
        elif host.gpu_info:  # Has GPU but no selfplay
            print("  GPU Selfplay: stopped")

    print()


def start_selfplay_on_vast(host_config: HostConfig, board: str, num_games: int) -> bool:
    """Start GPU selfplay on a Vast instance."""
    executor = SSHExecutor(host_config)

    try:
        # Check if already running
        result = executor.run("pgrep -f run_gpu_selfplay | head -1", timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            print(f"  {host_config.name}: Selfplay already running")
            return True

        # Start selfplay with auto-ramdrive detection
        executor.run("mkdir -p logs data/games", timeout=10)
        cmd = f"python scripts/run_gpu_selfplay.py --board {board} --num-games {num_games} --auto-ramdrive --sync-interval 300 --sync-target data/games"
        result = executor.run_async(cmd, "logs/gpu_selfplay.log")

        time.sleep(3)
        check = executor.run("pgrep -f run_gpu_selfplay | head -1", timeout=10)
        if check.returncode == 0 and check.stdout.strip():
            print(f"  {host_config.name}: Started selfplay ({board})")
            return True
        else:
            print(f"  {host_config.name}: Failed to start selfplay")
            return False

    except Exception as e:
        print(f"  {host_config.name}: Error - {e}")
        return False


def stop_selfplay_on_host(host_config: HostConfig) -> bool:
    """Stop GPU selfplay on a host."""
    executor = SSHExecutor(host_config)

    try:
        executor.run("pkill -f run_gpu_selfplay", timeout=10)
        print(f"  {host_config.name}: Stopped selfplay")
        return True
    except Exception as e:
        print(f"  {host_config.name}: Error - {e}")
        return False


def cmd_status(args) -> None:
    """Show cluster status."""
    hosts = args.hosts.split(",") if args.hosts else None
    status = get_cluster_status(hosts)

    if args.json:
        output = {
            "timestamp": status.timestamp.isoformat(),
            "total_gpus": status.total_gpus,
            "total_gpu_memory_gb": status.total_gpu_memory_gb,
            "total_loops_running": status.total_loops_running,
            "total_selfplay_running": status.total_selfplay_running,
            "hosts": [
                {
                    "name": h.name,
                    "reachable": h.reachable,
                    "gpu_info": h.gpu_info,
                    "selfplay_running": h.selfplay_running,
                    "selfplay_board": h.selfplay_board,
                    "improvement_loops": [
                        {"board": l.board_type, "running": l.is_running, "pid": l.pid}
                        for l in h.improvement_loops
                    ],
                    "error": h.error,
                }
                for h in status.hosts
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print_status(status, compact=args.compact)


def cmd_loops(args) -> None:
    """Manage improvement loops."""
    hosts = args.hosts.split(",") if args.hosts else None
    mgr = ImprovementLoopManager(hosts=hosts)

    if not mgr.host_names:
        print("No GPU hosts found")
        return

    boards = args.boards.split(",") if args.boards else mgr.BOARD_TYPES

    if args.action == "start":
        print(f"Starting improvement loops on {len(mgr.host_names)} host(s)...")
        total = 0
        for host in mgr.host_names:
            for board in boards:
                if mgr.start_loop(host, board, args.games, args.eval_games):
                    total += 1
        print(f"Started {total} loop(s)")

    elif args.action == "stop":
        print(f"Stopping improvement loops on {len(mgr.host_names)} host(s)...")
        total = 0
        for host in mgr.host_names:
            for board in boards:
                if mgr.stop_loop(host, board):
                    total += 1
        print(f"Stopped {total} loop(s)")

    elif args.action == "status":
        statuses = mgr.get_cluster_loop_status()
        from scripts.cluster_manager import print_loop_status
        print_loop_status(statuses)


def cmd_selfplay(args) -> None:
    """Manage GPU selfplay on Vast instances."""
    all_hosts = load_remote_hosts()

    # Filter to Vast instances or specified hosts
    if args.hosts:
        host_names = args.hosts.split(",")
        hosts = {name: cfg for name, cfg in all_hosts.items() if name in host_names}
    else:
        # Default to Vast instances
        hosts = {
            name: cfg for name, cfg in all_hosts.items()
            if name.startswith("vast-") and cfg.properties.get("gpu")
        }

    if not hosts:
        print("No Vast instances found")
        return

    if args.action == "start":
        print(f"Starting GPU selfplay on {len(hosts)} host(s)...")
        board = args.board or "square8"
        games = args.games or 1000
        for name, cfg in hosts.items():
            start_selfplay_on_vast(cfg, board, games)

    elif args.action == "stop":
        print(f"Stopping GPU selfplay on {len(hosts)} host(s)...")
        for name, cfg in hosts.items():
            stop_selfplay_on_host(cfg)

    elif args.action == "status":
        print(f"Checking selfplay on {len(hosts)} host(s)...")
        for name, cfg in hosts.items():
            status = check_host_status(cfg)
            if status.selfplay_running:
                print(f"  {name}: RUNNING ({status.selfplay_board or 'unknown'})")
            else:
                print(f"  {name}: stopped")


def cmd_monitor(args) -> None:
    """Monitor cluster in real-time."""
    hosts = args.hosts.split(",") if args.hosts else None
    interval = args.interval

    print(f"Monitoring cluster (interval={interval}s, Ctrl+C to stop)...")

    try:
        while True:
            # Clear screen
            print("\033[2J\033[H", end="")

            status = get_cluster_status(hosts)
            print_status(status, compact=True)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RingRift Cluster Control - Unified management for distributed training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # status command
    p_status = subparsers.add_parser("status", help="Show cluster status")
    p_status.add_argument("--hosts", help="Comma-separated list of hosts")
    p_status.add_argument("--json", action="store_true", help="Output as JSON")
    p_status.add_argument("--compact", action="store_true", help="Compact output")

    # loops command
    p_loops = subparsers.add_parser("loops", help="Manage improvement loops")
    p_loops.add_argument("action", choices=["start", "stop", "status"], help="Action to perform")
    p_loops.add_argument("--hosts", help="Comma-separated list of hosts")
    p_loops.add_argument("--boards", help="Comma-separated boards (sq8,sq19,hex)")
    p_loops.add_argument("--games", type=int, default=200, help="Games per iteration")
    p_loops.add_argument("--eval-games", type=int, default=50, help="Evaluation games")

    # selfplay command
    p_selfplay = subparsers.add_parser("selfplay", help="Manage GPU selfplay")
    p_selfplay.add_argument("action", choices=["start", "stop", "status"], help="Action to perform")
    p_selfplay.add_argument("--hosts", help="Comma-separated list of hosts")
    p_selfplay.add_argument("--board", help="Board type for selfplay")
    p_selfplay.add_argument("--games", type=int, help="Number of games")

    # monitor command
    p_monitor = subparsers.add_parser("monitor", help="Monitor cluster in real-time")
    p_monitor.add_argument("--hosts", help="Comma-separated list of hosts")
    p_monitor.add_argument("--interval", type=float, default=30.0, help="Refresh interval")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "loops":
        cmd_loops(args)
    elif args.command == "selfplay":
        cmd_selfplay(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
