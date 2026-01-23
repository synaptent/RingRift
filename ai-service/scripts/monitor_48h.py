#!/usr/bin/env python3
"""
48-Hour Autonomous Operation Monitor

Monitors the P2P cluster health and training progress for unattended operation.
Logs status every 5 minutes and alerts on critical issues.

Usage:
    python scripts/monitor_48h.py                    # Run in foreground
    python scripts/monitor_48h.py --daemon           # Run as background daemon
    python scripts/monitor_48h.py --status           # Quick status check
    python scripts/monitor_48h.py --summary          # Show summary of recent logs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import aiohttp

# Configuration
CLUSTER_ENDPOINTS = [
    ("nebius-h100-1", "http://100.106.19.6:8770/status"),
    ("nebius-h100-3", "http://100.109.195.71:8770/status"),
    ("lambda-gh200-1", "http://100.71.89.91:8770/status"),
    ("lambda-gh200-2", "http://100.110.143.119:8770/status"),
    ("lambda-gh200-10", "http://100.100.19.96:8770/status"),
]

LOG_DIR = Path(__file__).parent.parent / "logs" / "monitor_48h"
LOG_FILE = LOG_DIR / "cluster_status.jsonl"
SUMMARY_FILE = LOG_DIR / "summary.json"

# Thresholds
MIN_ALIVE_PEERS = 5
MIN_SELFPLAY_JOBS = 10
MIN_WORK_QUEUE = 50
MAX_CONSECUTIVE_FAILURES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def fetch_status(session: aiohttp.ClientSession, name: str, url: str) -> dict[str, Any] | None:
    """Fetch status from a cluster endpoint."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {"name": name, "url": url, "status": "ok", "data": data}
    except Exception as e:
        logger.debug(f"Failed to fetch from {name}: {e}")
    return {"name": name, "url": url, "status": "failed", "data": None}


async def get_cluster_status() -> dict[str, Any]:
    """Get comprehensive cluster status from all endpoints."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_status(session, name, url) for name, url in CLUSTER_ENDPOINTS]
        results = await asyncio.gather(*tasks)

    # Find the best responding node (preferring leader)
    best_data = None
    leader_id = None

    for result in results:
        if result["status"] == "ok" and result["data"]:
            data = result["data"]
            if data.get("is_leader"):
                best_data = data
                leader_id = data.get("node_id")
                break
            elif best_data is None:
                best_data = data
                leader_id = data.get("leader_id")

    # Extract key metrics
    status = {
        "timestamp": datetime.now().isoformat(),
        "cluster_reachable": any(r["status"] == "ok" for r in results),
        "endpoints_responding": sum(1 for r in results if r["status"] == "ok"),
        "total_endpoints": len(results),
    }

    if best_data:
        status.update({
            "leader_id": best_data.get("leader_id"),
            "alive_peers": best_data.get("alive_peers", 0),
            "voter_quorum_ok": best_data.get("voter_quorum_ok", False),
            "voters_alive": best_data.get("voters_alive", 0),
            "selfplay_jobs": best_data.get("selfplay_jobs", 0),
            "training_jobs": best_data.get("training_jobs", 0),
            "work_queue_size": best_data.get("work_queue_size", 0),
            "diversity_metrics": best_data.get("diversity_metrics", {}),
        })
    else:
        status.update({
            "leader_id": None,
            "alive_peers": 0,
            "voter_quorum_ok": False,
            "voters_alive": 0,
            "selfplay_jobs": 0,
            "training_jobs": 0,
            "work_queue_size": 0,
        })

    return status


def check_alerts(status: dict[str, Any]) -> list[str]:
    """Check for critical issues that need attention."""
    alerts = []

    if not status.get("cluster_reachable"):
        alerts.append("CRITICAL: Cluster not reachable from any endpoint")

    if status.get("alive_peers", 0) < MIN_ALIVE_PEERS:
        alerts.append(f"WARNING: Only {status.get('alive_peers')} alive peers (min: {MIN_ALIVE_PEERS})")

    if not status.get("voter_quorum_ok"):
        alerts.append("CRITICAL: Voter quorum lost")

    if status.get("selfplay_jobs", 0) < MIN_SELFPLAY_JOBS:
        alerts.append(f"WARNING: Only {status.get('selfplay_jobs')} selfplay jobs (min: {MIN_SELFPLAY_JOBS})")

    if status.get("work_queue_size", 0) < MIN_WORK_QUEUE:
        alerts.append(f"WARNING: Work queue low ({status.get('work_queue_size')} items)")

    return alerts


def log_status(status: dict[str, Any], alerts: list[str]) -> None:
    """Log status to file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Append to JSONL log
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(status) + "\n")

    # Update summary
    summary = {
        "last_update": status["timestamp"],
        "last_status": status,
        "alerts": alerts,
    }

    # Load existing summary to track history
    if SUMMARY_FILE.exists():
        try:
            with open(SUMMARY_FILE) as f:
                existing = json.load(f)
            summary["total_checks"] = existing.get("total_checks", 0) + 1
            summary["total_alerts"] = existing.get("total_alerts", 0) + len(alerts)
            summary["started_at"] = existing.get("started_at", status["timestamp"])
        except Exception:
            summary["total_checks"] = 1
            summary["total_alerts"] = len(alerts)
            summary["started_at"] = status["timestamp"]
    else:
        summary["total_checks"] = 1
        summary["total_alerts"] = len(alerts)
        summary["started_at"] = status["timestamp"]

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)


def print_status(status: dict[str, Any], alerts: list[str]) -> None:
    """Print formatted status to console."""
    print("\n" + "=" * 60)
    print(f"Cluster Status - {status['timestamp']}")
    print("=" * 60)
    print(f"Endpoints Responding: {status['endpoints_responding']}/{status['total_endpoints']}")
    print(f"Leader: {status.get('leader_id', 'Unknown')}")
    print(f"Alive Peers: {status.get('alive_peers', 0)}")
    print(f"Voter Quorum: {'OK' if status.get('voter_quorum_ok') else 'LOST'} ({status.get('voters_alive', 0)} voters)")
    print(f"Selfplay Jobs: {status.get('selfplay_jobs', 0)}")
    print(f"Training Jobs: {status.get('training_jobs', 0)}")
    print(f"Work Queue: {status.get('work_queue_size', 0)} items")

    if alerts:
        print("\n" + "-" * 60)
        print("ALERTS:")
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("\n[All systems nominal]")

    print("=" * 60)


async def run_monitor(interval: int = 300, daemon: bool = False) -> None:
    """Run the monitoring loop."""
    logger.info(f"Starting 48h monitor (interval: {interval}s, daemon: {daemon})")

    consecutive_failures = 0

    while True:
        try:
            status = await get_cluster_status()
            alerts = check_alerts(status)

            if not daemon:
                print_status(status, alerts)

            log_status(status, alerts)

            if alerts:
                for alert in alerts:
                    logger.warning(alert)
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                logger.info(f"Cluster healthy: {status.get('alive_peers')} peers, "
                          f"{status.get('selfplay_jobs')} selfplay, {status.get('training_jobs')} training")

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(f"CRITICAL: {consecutive_failures} consecutive check failures")

        except Exception as e:
            logger.error(f"Monitor error: {e}")

        await asyncio.sleep(interval)


def show_summary() -> None:
    """Show summary of recent monitoring."""
    if not SUMMARY_FILE.exists():
        print("No monitoring data found. Run the monitor first.")
        return

    with open(SUMMARY_FILE) as f:
        summary = json.load(f)

    print("\n" + "=" * 60)
    print("48h Monitor Summary")
    print("=" * 60)
    print(f"Started: {summary.get('started_at', 'Unknown')}")
    print(f"Last Update: {summary.get('last_update', 'Unknown')}")
    print(f"Total Checks: {summary.get('total_checks', 0)}")
    print(f"Total Alerts: {summary.get('total_alerts', 0)}")

    if summary.get("alerts"):
        print("\nCurrent Alerts:")
        for alert in summary["alerts"]:
            print(f"  - {alert}")
    else:
        print("\n[No current alerts]")

    last = summary.get("last_status", {})
    print(f"\nLast Status:")
    print(f"  Leader: {last.get('leader_id', 'Unknown')}")
    print(f"  Alive Peers: {last.get('alive_peers', 0)}")
    print(f"  Selfplay Jobs: {last.get('selfplay_jobs', 0)}")
    print(f"  Training Jobs: {last.get('training_jobs', 0)}")
    print("=" * 60)


async def quick_status() -> None:
    """Do a quick status check and exit."""
    status = await get_cluster_status()
    alerts = check_alerts(status)
    print_status(status, alerts)


def main():
    parser = argparse.ArgumentParser(description="48-Hour Autonomous Operation Monitor")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon (no console output)")
    parser.add_argument("--status", action="store_true", help="Quick status check and exit")
    parser.add_argument("--summary", action="store_true", help="Show summary of recent monitoring")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds (default: 300)")
    args = parser.parse_args()

    if args.summary:
        show_summary()
    elif args.status:
        asyncio.run(quick_status())
    else:
        try:
            asyncio.run(run_monitor(interval=args.interval, daemon=args.daemon))
        except KeyboardInterrupt:
            logger.info("Monitor stopped by user")


if __name__ == "__main__":
    main()
