#!/usr/bin/env python3
"""Cluster Health Monitor - Checks every 10 minutes for 60 minutes.

Records problems and determines root causes.
Created for the 4+ hour stability verification (Jan 24, 2026).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

LOG_FILE = Path(__file__).parent.parent / "cluster_health_log.md"
JSONL_FILE = Path(__file__).parent.parent / "cluster_health_log.jsonl"


def get_p2p_status(host: str = "localhost", port: int = 8770) -> dict:
    """Fetch P2P status from the orchestrator."""
    try:
        url = f"http://{host}:{port}/status"
        with urlopen(url, timeout=15) as response:
            return json.loads(response.read().decode())
    except (URLError, json.JSONDecodeError, OSError) as e:
        return {"error": str(e), "timestamp": time.time()}


def analyze_cluster(status: dict) -> dict:
    """Analyze cluster status and identify issues."""
    issues = []
    recommendations = []

    if "error" in status:
        issues.append(f"P2P not responding: {status['error']}")
        return {
            "healthy": False,
            "issues": issues,
            "recommendations": ["Restart P2P orchestrator"],
            "leader": None,
            "alive_peers": 0,
            "total_peers": 0,
        }

    leader = status.get("leader_id")
    alive_peers = status.get("alive_peers", 0)
    peers = status.get("peers", {})
    total_peers = len(peers)

    # Check for issues
    if not leader:
        issues.append("No leader elected")
        recommendations.append("Check voter quorum and network connectivity")

    if alive_peers < 20:
        issues.append(f"Below target of 20 peers: {alive_peers}")
        if alive_peers < 10:
            recommendations.append("Check network connectivity to cluster nodes")
            recommendations.append("Verify Tailscale is running on all nodes")

    if alive_peers < 5:
        issues.append(f"CRITICAL: Low peer count: {alive_peers}")
        recommendations.append("Immediate investigation required")

    # Check for dead peers
    dead_peers = []
    suspect_peers = []
    for name, info in peers.items():
        peer_status = info.get("status", "unknown")
        if peer_status == "dead":
            dead_peers.append(name)
        elif peer_status == "suspect":
            suspect_peers.append(name)

    if len(dead_peers) > 5:
        issues.append(f"High dead peer count: {len(dead_peers)}")
        recommendations.append("Check for network partition or widespread outage")

    if len(suspect_peers) > 3:
        issues.append(f"Multiple suspect peers: {len(suspect_peers)}")
        recommendations.append("Possible network latency issues")

    return {
        "healthy": len(issues) == 0 or (alive_peers >= 10 and leader),
        "issues": issues,
        "recommendations": recommendations,
        "leader": leader,
        "alive_peers": alive_peers,
        "total_peers": total_peers,
        "dead_peers": len(dead_peers),
        "suspect_peers": len(suspect_peers),
        "uptime_hours": round(status.get("uptime", 0) / 3600, 2),
    }


def print_status(timestamp: datetime, analysis: dict, elapsed_minutes: int, check_num: int) -> None:
    """Print status to console."""
    status_icon = "OK" if analysis["healthy"] else "ISSUE"
    print(f"\n[Check {check_num}] [T+{elapsed_minutes:02d}min] [{status_icon}] {timestamp.strftime('%H:%M:%S')}")
    print(f"  Leader: {analysis['leader'] or 'None'}")
    print(f"  Alive peers: {analysis['alive_peers']} / Target: 20+")
    print(f"  Dead: {analysis.get('dead_peers', 0)}, Suspect: {analysis.get('suspect_peers', 0)}")

    if analysis["issues"]:
        print("  Issues:")
        for issue in analysis["issues"]:
            print(f"    - {issue}")


def main():
    parser = argparse.ArgumentParser(description="Monitor P2P cluster health")
    parser.add_argument("--duration", type=int, default=60, help="Duration in minutes")
    parser.add_argument("--interval", type=int, default=10, help="Check interval in minutes")
    parser.add_argument("--host", default="localhost", help="P2P host")
    parser.add_argument("--port", type=int, default=8770, help="P2P port")
    args = parser.parse_args()

    total_checks = args.duration // args.interval + 1
    print(f"Starting Cluster Health Monitor")
    print(f"Duration: {args.duration} minutes, Interval: {args.interval} minutes")
    print("-" * 60)

    start_time = time.time()
    end_time = start_time + (args.duration * 60)
    check_count = 0
    issues_count = 0
    all_checks = []

    # Initial check
    status = get_p2p_status(args.host, args.port)
    analysis = analyze_cluster(status)
    timestamp = datetime.now()
    print_status(timestamp, analysis, 0, 1)
    check_count += 1
    all_checks.append(analysis)
    if not analysis["healthy"]:
        issues_count += 1

    # Save to JSONL
    with open(JSONL_FILE, "a") as f:
        f.write(json.dumps({"timestamp": timestamp.isoformat(), **analysis}) + "\n")

    # Periodic checks
    while time.time() < end_time:
        time.sleep(args.interval * 60)
        elapsed = (time.time() - start_time) / 60
        status = get_p2p_status(args.host, args.port)
        analysis = analyze_cluster(status)
        timestamp = datetime.now()
        check_count += 1
        print_status(timestamp, analysis, int(elapsed), check_count)
        all_checks.append(analysis)
        if not analysis["healthy"]:
            issues_count += 1
        with open(JSONL_FILE, "a") as f:
            f.write(json.dumps({"timestamp": timestamp.isoformat(), **analysis}) + "\n")

    # Summary
    print("\n" + "=" * 60)
    print("MONITORING COMPLETE")
    print("=" * 60)
    print(f"Total checks: {check_count}")
    print(f"Healthy checks: {check_count - issues_count}")
    print(f"Stability rate: {((check_count - issues_count) / check_count * 100):.1f}%")
    avg_peers = sum(c.get("alive_peers", 0) for c in all_checks) / len(all_checks) if all_checks else 0
    print(f"Average alive peers: {avg_peers:.1f}")


if __name__ == "__main__":
    main()
