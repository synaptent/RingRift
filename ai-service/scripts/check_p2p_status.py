#!/usr/bin/env python3
"""Check P2P status on all cluster nodes.

This script:
1. Reads distributed_hosts.yaml to get the host list
2. SSHes to each node with p2p_enabled: true
3. Checks if P2P is running (curl localhost:8770/status or ps aux | grep p2p)
4. Reports which nodes have P2P running vs not running
5. For nodes where P2P is not running, checks for recent crash logs

Usage:
    python scripts/check_p2p_status.py

    # With verbose output
    python scripts/check_p2p_status.py --verbose

    # Check specific nodes only
    python scripts/check_p2p_status.py --nodes nebius-h100-1 vast-29118471

    # Show only nodes with issues
    python scripts/check_p2p_status.py --issues-only

    # Output as JSON for scripting
    python scripts/check_p2p_status.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add ai-service to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.lib.hosts import get_hosts, get_hosts_manager, HostConfig
from app.core.ssh import SSHClient, SSHConfig

logger = logging.getLogger(__name__)

# P2P default port
P2P_PORT = 8770
# SSH timeout for quick checks
SSH_TIMEOUT = 15
# Max concurrent SSH connections
MAX_CONCURRENT = 10


@dataclass
class P2PStatus:
    """P2P status for a node."""
    node_name: str
    ssh_reachable: bool = False
    p2p_running: bool = False
    p2p_role: str = ""
    leader_id: str | None = None
    uptime_seconds: int = 0
    alive_peers: int = 0
    selfplay_jobs: int = 0
    gpu_name: str = ""
    gpu_percent: float = 0.0
    error: str | None = None
    crash_log: str | None = None


@dataclass
class ClusterP2PReport:
    """Report of P2P status across the cluster."""
    timestamp: str
    total_nodes: int = 0
    ssh_reachable: int = 0
    p2p_running: int = 0
    p2p_not_running: int = 0
    ssh_unreachable: int = 0
    nodes: list[P2PStatus] = field(default_factory=list)

    @property
    def leader_node(self) -> str | None:
        """Get the current leader node ID."""
        for node in self.nodes:
            if node.p2p_role == "leader":
                return node.node_name
            if node.leader_id:
                return node.leader_id
        return None


def get_p2p_enabled_hosts() -> list[HostConfig]:
    """Get all hosts with p2p_enabled: true from config."""
    manager = get_hosts_manager()
    config = manager._load_config()
    hosts_data = config.get("hosts", {})

    p2p_hosts = []
    for name, data in hosts_data.items():
        if data and data.get("p2p_enabled", False):
            host = manager._parse_host(name, data)
            p2p_hosts.append(host)

    return p2p_hosts


async def check_node_p2p_status(host: HostConfig, verbose: bool = False) -> P2PStatus:
    """Check P2P status on a single node.

    Args:
        host: Host configuration
        verbose: Enable verbose logging

    Returns:
        P2PStatus with results
    """
    status = P2PStatus(node_name=host.name)

    # Build SSH config and client (using app.core.ssh)
    ssh_config = SSHConfig(
        host=host.ssh_host,
        port=host.ssh_port,
        user=host.ssh_user,
        key_path=host.ssh_key,
        connect_timeout=SSH_TIMEOUT,
    )
    client = SSHClient(ssh_config)

    if verbose:
        logger.info(f"Checking {host.name} ({host.ssh_host}:{host.ssh_port})...")

    # First, check SSH reachability with a simple command
    result = await client.run_async("echo ok", timeout=SSH_TIMEOUT)

    if not result.success:
        status.error = f"SSH unreachable: {result.output}"
        return status

    status.ssh_reachable = True

    # Check P2P status via curl
    curl_cmd = f"curl -s --max-time 5 http://localhost:{P2P_PORT}/status 2>/dev/null || echo 'CURL_FAILED'"
    result = await client.run_async(curl_cmd, timeout=SSH_TIMEOUT)

    if result.success and result.output and result.output != "CURL_FAILED":
        try:
            data = json.loads(result.output)
            status.p2p_running = True
            status.p2p_role = data.get("role", "unknown")
            status.leader_id = data.get("leader_id") or data.get("effective_leader_id")
            status.uptime_seconds = data.get("uptime_seconds", 0)
            status.alive_peers = data.get("alive_peers", 0)

            # Get self metrics
            self_info = data.get("self", {})
            status.selfplay_jobs = self_info.get("selfplay_jobs", 0)
            status.gpu_name = self_info.get("gpu_name", "")
            status.gpu_percent = self_info.get("gpu_percent", 0)

            return status
        except json.JSONDecodeError:
            # Not valid JSON, P2P might not be running
            pass

    # P2P not responding to HTTP, check if process is running
    ps_cmd = "ps aux | grep -E 'p2p_orchestrator|p2p.main' | grep -v grep | head -1"
    result = await client.run_async(ps_cmd, timeout=SSH_TIMEOUT)

    if result.success and result.output:
        # Process exists but HTTP not responding
        status.error = "P2P process exists but HTTP not responding (possibly starting up)"
        status.p2p_running = False  # Not fully operational
    else:
        status.p2p_running = False
        status.error = "P2P not running"

        # Check for crash logs
        await check_crash_logs(client, status, host)

    return status


async def check_crash_logs(client: SSHClient, status: P2PStatus, host: HostConfig) -> None:
    """Check for recent P2P crash logs on a node.

    Args:
        client: SSH client to use
        status: P2PStatus to update with crash log info
        host: Host configuration for path info
    """
    # Common log locations to check
    log_paths = [
        f"{host.ringrift_path}/logs/p2p.log",
        f"{host.ringrift_path}/p2p.log",
        f"{host.ringrift_path}/logs/p2p_orchestrator.log",
        "~/p2p.log",
        "/var/log/ringrift/p2p.log",
    ]

    for log_path in log_paths:
        # Get last 20 lines of the log if it exists
        check_cmd = f"test -f {log_path} && tail -20 {log_path} 2>/dev/null || true"
        result = await client.run_async(check_cmd, timeout=SSH_TIMEOUT)

        if result.success and result.output:
            # Look for error indicators
            error_indicators = ["ERROR", "Exception", "Traceback", "FATAL", "crashed", "killed"]
            lines = result.output.split("\n")
            error_lines = [
                line for line in lines
                if any(indicator in line for indicator in error_indicators)
            ]

            if error_lines:
                status.crash_log = "\n".join(error_lines[-5:])  # Last 5 error lines
                break


async def check_all_nodes(
    hosts: list[HostConfig],
    verbose: bool = False,
    max_concurrent: int = MAX_CONCURRENT,
) -> list[P2PStatus]:
    """Check P2P status on all nodes concurrently.

    Args:
        hosts: List of hosts to check
        verbose: Enable verbose logging
        max_concurrent: Maximum concurrent SSH connections

    Returns:
        List of P2PStatus results
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def check_with_semaphore(host: HostConfig) -> P2PStatus:
        async with semaphore:
            return await check_node_p2p_status(host, verbose)

    tasks = [check_with_semaphore(host) for host in hosts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to error statuses
    statuses = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            statuses.append(P2PStatus(
                node_name=hosts[i].name,
                error=f"Check failed: {result}"
            ))
        else:
            statuses.append(result)

    return statuses


def format_uptime(seconds: int) -> str:
    """Format uptime in human-readable format."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m"
    elif seconds < 86400:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h{mins}m"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        return f"{days}d{hours}h"


def print_summary_table(report: ClusterP2PReport, issues_only: bool = False) -> None:
    """Print a summary table of P2P status.

    Args:
        report: Cluster P2P report
        issues_only: Only show nodes with issues
    """
    print("\n" + "=" * 100)
    print(f"P2P Cluster Status - {report.timestamp}")
    print("=" * 100)

    # Summary stats
    print(f"\nSummary:")
    print(f"  Total nodes with p2p_enabled: {report.total_nodes}")
    print(f"  SSH reachable:                {report.ssh_reachable}")
    print(f"  P2P running:                  {report.p2p_running}")
    print(f"  P2P not running:              {report.p2p_not_running}")
    print(f"  SSH unreachable:              {report.ssh_unreachable}")

    if report.leader_node:
        print(f"  Current leader:               {report.leader_node}")

    # Node table
    print("\n" + "-" * 100)
    print(f"{'Node':<25} {'SSH':<5} {'P2P':<8} {'Role':<10} {'Uptime':<10} {'Peers':<6} {'Jobs':<5} {'GPU':<20}")
    print("-" * 100)

    # Sort nodes: running first, then by name
    sorted_nodes = sorted(
        report.nodes,
        key=lambda n: (not n.p2p_running, not n.ssh_reachable, n.node_name)
    )

    for node in sorted_nodes:
        # Skip if issues_only and node is running fine
        if issues_only and node.p2p_running and node.ssh_reachable:
            continue

        ssh_status = "OK" if node.ssh_reachable else "FAIL"
        p2p_status = "Running" if node.p2p_running else "Down"
        role = node.p2p_role or "-"
        uptime = format_uptime(node.uptime_seconds) if node.uptime_seconds else "-"
        peers = str(node.alive_peers) if node.alive_peers else "-"
        jobs = str(node.selfplay_jobs) if node.selfplay_jobs else "-"
        gpu = node.gpu_name[:18] + ".." if len(node.gpu_name) > 20 else node.gpu_name or "-"

        # Color coding (ANSI)
        if not node.ssh_reachable:
            line_color = "\033[91m"  # Red
        elif not node.p2p_running:
            line_color = "\033[93m"  # Yellow
        elif node.p2p_role == "leader":
            line_color = "\033[92m"  # Green
        else:
            line_color = ""

        end_color = "\033[0m" if line_color else ""

        print(f"{line_color}{node.node_name:<25} {ssh_status:<5} {p2p_status:<8} {role:<10} {uptime:<10} {peers:<6} {jobs:<5} {gpu:<20}{end_color}")

    print("-" * 100)

    # Show errors and crash logs
    nodes_with_issues = [n for n in report.nodes if n.error or n.crash_log]
    if nodes_with_issues and not issues_only:
        print("\n" + "=" * 100)
        print("Issues & Crash Logs:")
        print("=" * 100)

        for node in nodes_with_issues:
            print(f"\n{node.node_name}:")
            if node.error:
                print(f"  Error: {node.error}")
            if node.crash_log:
                print(f"  Recent log errors:")
                for line in node.crash_log.split("\n"):
                    print(f"    {line}")


def print_json_output(report: ClusterP2PReport) -> None:
    """Print report as JSON.

    Args:
        report: Cluster P2P report
    """
    output = {
        "timestamp": report.timestamp,
        "summary": {
            "total_nodes": report.total_nodes,
            "ssh_reachable": report.ssh_reachable,
            "p2p_running": report.p2p_running,
            "p2p_not_running": report.p2p_not_running,
            "ssh_unreachable": report.ssh_unreachable,
            "leader_node": report.leader_node,
        },
        "nodes": [
            {
                "name": n.node_name,
                "ssh_reachable": n.ssh_reachable,
                "p2p_running": n.p2p_running,
                "role": n.p2p_role,
                "leader_id": n.leader_id,
                "uptime_seconds": n.uptime_seconds,
                "alive_peers": n.alive_peers,
                "selfplay_jobs": n.selfplay_jobs,
                "gpu_name": n.gpu_name,
                "gpu_percent": n.gpu_percent,
                "error": n.error,
                "crash_log": n.crash_log,
            }
            for n in report.nodes
        ]
    }
    print(json.dumps(output, indent=2))


async def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 if any nodes have issues)
    """
    parser = argparse.ArgumentParser(
        description="Check P2P status on all cluster nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--nodes",
        nargs="*",
        help="Check only specific nodes (space-separated names)"
    )
    parser.add_argument(
        "--issues-only",
        action="store_true",
        help="Only show nodes with issues"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Maximum concurrent SSH connections (default: {MAX_CONCURRENT})"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Get hosts with P2P enabled
    all_hosts = get_p2p_enabled_hosts()

    if not all_hosts:
        print("No hosts found with p2p_enabled: true")
        return 1

    # Filter if specific nodes requested
    if args.nodes:
        hosts = [h for h in all_hosts if h.name in args.nodes]
        if not hosts:
            print(f"No matching hosts found. Available: {[h.name for h in all_hosts]}")
            return 1
    else:
        hosts = all_hosts

    if not args.json:
        print(f"Checking P2P status on {len(hosts)} nodes...")

    # Check all nodes
    statuses = await check_all_nodes(hosts, args.verbose, args.max_concurrent)

    # Build report
    report = ClusterP2PReport(
        timestamp=datetime.now().isoformat(),
        total_nodes=len(hosts),
        nodes=statuses,
    )

    # Calculate summary stats
    for status in statuses:
        if status.ssh_reachable:
            report.ssh_reachable += 1
            if status.p2p_running:
                report.p2p_running += 1
            else:
                report.p2p_not_running += 1
        else:
            report.ssh_unreachable += 1

    # Output
    if args.json:
        print_json_output(report)
    else:
        print_summary_table(report, args.issues_only)

    # Return non-zero if any nodes have issues
    has_issues = report.p2p_not_running > 0 or report.ssh_unreachable > 0
    return 1 if has_issues else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
