#!/usr/bin/env python3
"""P2P Cluster Monitoring Script.

Checks cluster health every N minutes and logs issues for analysis.
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_cluster_status(base_url: str = "http://localhost:8770") -> dict[str, Any]:
    """Fetch current cluster status."""
    try:
        # Get health
        health_resp = requests.get(f"{base_url}/health", timeout=5)
        health = health_resp.json() if health_resp.ok else {}

        # Get status for more details
        status_resp = requests.get(f"{base_url}/status", timeout=5)
        status = status_resp.json() if status_resp.ok else {}

        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "healthy": health.get("healthy", False),
            "node_id": health.get("node_id"),
            "role": health.get("role"),
            "leader_id": health.get("leader_id"),
            "active_peers": health.get("active_peers", 0),
            "total_peers": health.get("total_peers", 0),
            "uptime_seconds": health.get("uptime_seconds", 0),
            "cpu_percent": health.get("cpu_percent", 0),
            "memory_percent": health.get("memory_percent", 0),
            "voters_alive": status.get("voter_health", {}).get("voters_alive", 0),
            "voters_total": status.get("voter_health", {}).get("voters_total", 0),
            "quorum_ok": status.get("voter_health", {}).get("quorum_ok", False),
            "suspect_peers": status.get("suspect_peers", 0),
            "dead_peers": status.get("dead_peers", 0),
        }
    except Exception as e:
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "error": str(e),
            "healthy": False,
        }


def get_peer_details(base_url: str = "http://localhost:8770") -> list[dict]:
    """Get detailed peer information."""
    try:
        resp = requests.get(f"{base_url}/status", timeout=5)
        if not resp.ok:
            return []

        status = resp.json()
        peers = status.get("all_peers", {})

        result = []
        for node_id, info in peers.items():
            if isinstance(info, dict):
                result.append({
                    "node_id": node_id,
                    "state": info.get("state", "unknown"),
                    "last_seen": info.get("last_seen_seconds", None),
                })
            else:
                result.append({
                    "node_id": node_id,
                    "state": str(info),
                })

        return sorted(result, key=lambda x: x["node_id"])
    except Exception:
        return []


def analyze_issues(current: dict, previous: dict | None) -> list[str]:
    """Analyze issues between current and previous status."""
    issues = []

    if not current.get("healthy"):
        issues.append("CRITICAL: Cluster is unhealthy")

    if current.get("active_peers", 0) < 20:
        issues.append(f"WARNING: Only {current.get('active_peers', 0)} active peers (target: 20+)")

    if not current.get("quorum_ok"):
        issues.append("CRITICAL: Quorum lost!")

    if current.get("voters_alive", 0) < current.get("voters_total", 0):
        offline = current.get("voters_total", 0) - current.get("voters_alive", 0)
        issues.append(f"WARNING: {offline} voters offline")

    if previous:
        prev_peers = previous.get("active_peers", 0)
        curr_peers = current.get("active_peers", 0)
        if curr_peers < prev_peers - 2:
            issues.append(f"ALERT: Peer count dropped from {prev_peers} to {curr_peers}")

        if previous.get("leader_id") != current.get("leader_id"):
            issues.append(f"INFO: Leader changed from {previous.get('leader_id')} to {current.get('leader_id')}")

    if current.get("error"):
        issues.append(f"ERROR: {current.get('error')}")

    return issues


def print_status(status: dict, issues: list[str]) -> None:
    """Print formatted status."""
    print("\n" + "=" * 70)
    print(f"CHECK TIME: {status.get('timestamp', 'unknown')}")
    print("=" * 70)

    if status.get("error"):
        print(f"ERROR: {status['error']}")
        return

    print(f"Node: {status.get('node_id')} ({status.get('role')})")
    print(f"Leader: {status.get('leader_id')}")
    print(f"Healthy: {status.get('healthy')}")
    print(f"Active Peers: {status.get('active_peers')}/{status.get('total_peers')}")
    print(f"Voters: {status.get('voters_alive')}/{status.get('voters_total')} (quorum={'OK' if status.get('quorum_ok') else 'LOST'})")
    print(f"Suspect Peers: {status.get('suspect_peers', 0)}")
    print(f"Dead Peers: {status.get('dead_peers', 0)}")
    print(f"Uptime: {status.get('uptime_seconds', 0):.0f}s")
    print(f"CPU: {status.get('cpu_percent', 0):.1f}%, Memory: {status.get('memory_percent', 0):.1f}%")

    if issues:
        print("\n--- ISSUES DETECTED ---")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n--- NO ISSUES ---")

    print("=" * 70)


def save_to_log(log_file: Path, status: dict, issues: list[str]) -> None:
    """Append status to log file."""
    with open(log_file, "a") as f:
        entry = {
            **status,
            "issues": issues,
        }
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor P2P cluster health")
    parser.add_argument("--interval", type=int, default=600, help="Check interval in seconds (default: 600)")
    parser.add_argument("--duration", type=int, default=3600, help="Total duration in seconds (default: 3600)")
    parser.add_argument("--url", default="http://localhost:8770", help="P2P orchestrator URL")
    parser.add_argument("--log", default="cluster_health_log.jsonl", help="Log file path")
    parser.add_argument("--once", action="store_true", help="Run once and exit")

    args = parser.parse_args()

    log_file = Path(args.log)
    start_time = time.time()
    previous_status = None
    check_count = 0

    print(f"Starting P2P cluster monitoring")
    print(f"Interval: {args.interval}s, Duration: {args.duration}s")
    print(f"Log file: {log_file}")

    while True:
        check_count += 1
        elapsed = time.time() - start_time

        if elapsed >= args.duration and not args.once:
            print(f"\nMonitoring complete. {check_count} checks performed over {elapsed:.0f}s")
            break

        # Get status
        status = get_cluster_status(args.url)
        issues = analyze_issues(status, previous_status)

        # Print and log
        print_status(status, issues)
        save_to_log(log_file, status, issues)

        previous_status = status

        if args.once:
            break

        # Wait for next check
        remaining = args.duration - elapsed
        sleep_time = min(args.interval, remaining)
        if sleep_time > 0:
            print(f"\nNext check in {sleep_time:.0f}s...")
            time.sleep(sleep_time)

    # Summary
    print("\n" + "=" * 70)
    print("MONITORING SUMMARY")
    print("=" * 70)
    print(f"Total checks: {check_count}")
    print(f"Duration: {elapsed:.0f}s")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
