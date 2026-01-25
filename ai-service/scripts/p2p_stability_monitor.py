#!/usr/bin/env python3
"""P2P Cluster Stability Monitor.

Monitors the P2P cluster status every interval and logs issues.
Created for the 60-minute stability verification (Jan 24, 2026).

Usage:
    python scripts/p2p_stability_monitor.py --duration 60 --interval 10
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

LOG_FILE = Path(__file__).parent.parent / "cluster_stability_log.jsonl"


def get_p2p_status(host: str = "localhost", port: int = 8770) -> dict:
    """Fetch P2P status from the orchestrator."""
    try:
        url = f"http://{host}:{port}/status"
        with urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode())
    except (URLError, json.JSONDecodeError, OSError) as e:
        return {"error": str(e), "timestamp": time.time()}


def analyze_status(status: dict) -> dict:
    """Analyze status and identify issues."""
    issues = []

    if "error" in status:
        issues.append(f"P2P not responding: {status['error']}")
        return {
            "healthy": False,
            "issues": issues,
            "leader": None,
            "alive_peers": 0,
        }

    leader = status.get("leader_id")
    alive_peers = status.get("alive_peers", 0)

    # Check for issues
    if not leader:
        issues.append("No leader elected")

    if alive_peers < 5:
        issues.append(f"Low peer count: {alive_peers}")

    if alive_peers < 20:
        issues.append(f"Below target of 20 peers: {alive_peers}")

    return {
        "healthy": len(issues) == 0 or (alive_peers >= 5 and leader),
        "issues": issues,
        "leader": leader,
        "alive_peers": alive_peers,
        "uptime_hours": round(status.get("uptime", 0) / 3600, 2),
    }


def log_status(timestamp: datetime, analysis: dict, raw_status: dict) -> None:
    """Log status to JSONL file."""
    entry = {
        "timestamp": timestamp.isoformat(),
        "healthy": analysis["healthy"],
        "leader": analysis["leader"],
        "alive_peers": analysis["alive_peers"],
        "issues": analysis["issues"],
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def print_status(timestamp: datetime, analysis: dict, elapsed_minutes: int) -> None:
    """Print status to console."""
    status_icon = "OK" if analysis["healthy"] else "ISSUE"
    print(f"\n[T+{elapsed_minutes:02d}min] [{status_icon}] {timestamp.strftime('%H:%M:%S')}")
    print(f"  Leader: {analysis['leader'] or 'None'}")
    print(f"  Alive peers: {analysis['alive_peers']}")

    if analysis["issues"]:
        print("  Issues:")
        for issue in analysis["issues"]:
            print(f"    - {issue}")


def main():
    parser = argparse.ArgumentParser(description="Monitor P2P cluster stability")
    parser.add_argument("--duration", type=int, default=60, help="Duration in minutes")
    parser.add_argument("--interval", type=int, default=10, help="Check interval in minutes")
    parser.add_argument("--host", default="localhost", help="P2P host")
    parser.add_argument("--port", type=int, default=8770, help="P2P port")
    args = parser.parse_args()

    print(f"Starting P2P stability monitor")
    print(f"Duration: {args.duration} minutes, Interval: {args.interval} minutes")
    print(f"Log file: {LOG_FILE}")
    print("-" * 60)

    start_time = time.time()
    end_time = start_time + (args.duration * 60)
    check_count = 0
    issues_count = 0

    # Initial check
    status = get_p2p_status(args.host, args.port)
    analysis = analyze_status(status)
    timestamp = datetime.now()
    log_status(timestamp, analysis, status)
    print_status(timestamp, analysis, 0)
    check_count += 1
    if not analysis["healthy"]:
        issues_count += 1

    # Periodic checks
    while time.time() < end_time:
        time.sleep(args.interval * 60)

        elapsed = (time.time() - start_time) / 60
        status = get_p2p_status(args.host, args.port)
        analysis = analyze_status(status)
        timestamp = datetime.now()

        log_status(timestamp, analysis, status)
        print_status(timestamp, analysis, int(elapsed))

        check_count += 1
        if not analysis["healthy"]:
            issues_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("MONITORING SUMMARY")
    print("=" * 60)
    print(f"Total checks: {check_count}")
    print(f"Healthy checks: {check_count - issues_count}")
    print(f"Issues detected: {issues_count}")
    print(f"Stability rate: {((check_count - issues_count) / check_count * 100):.1f}%")
    print(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()
