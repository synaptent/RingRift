#!/usr/bin/env python3
"""Continuous cluster monitoring script for RingRift AI training loop.

Monitors every 20 seconds:
- Cluster health (active nodes, leader status)
- Selfplay job counts
- Training job status (running, failed)
- Resource utilization (CPU, memory, GPU)
- Data sync status

Usage:
    python scripts/cluster_monitor.py [--interval 20] [--leader-url http://150.136.65.197:8770]
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)


class ClusterMonitor:
    def __init__(self, leader_url: str, interval: int = 20):
        self.leader_url = leader_url.rstrip('/')
        self.interval = interval
        self.last_status: Dict[str, Any] = {}
        self.alerts: List[str] = []
        self.start_time = time.time()
        self.check_count = 0

    async def fetch_json(self, endpoint: str, timeout: int = 10) -> Optional[Dict]:
        """Fetch JSON from an endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.leader_url}{endpoint}",
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {"error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def check_health(self) -> Dict[str, Any]:
        """Check cluster health via leader."""
        return await self.fetch_json("/health") or {}

    async def check_status(self) -> Dict[str, Any]:
        """Get leader status."""
        return await self.fetch_json("/status") or {}

    async def check_training(self) -> Dict[str, Any]:
        """Get training job status."""
        return await self.fetch_json("/training/status") or {}

    def format_timestamp(self) -> str:
        """Format current timestamp."""
        return datetime.now().strftime("%H:%M:%S")

    def check_for_issues(self, health: Dict, status: Dict, training: Dict) -> List[str]:
        """Check for issues and return alerts."""
        alerts = []

        # Check if leader is responsive
        if health.get("error"):
            alerts.append(f"CRITICAL: Leader unreachable - {health.get('error')}")
            return alerts

        # Check disk usage
        disk_pct = health.get("disk_percent", 0)
        if disk_pct > 90:
            alerts.append(f"WARNING: Disk usage {disk_pct:.1f}% on leader")

        # Check memory usage
        mem_pct = health.get("memory_percent", 0)
        if mem_pct > 95:
            alerts.append(f"WARNING: Memory usage {mem_pct:.1f}% on leader")

        # Check for failed training jobs
        training_jobs = training.get("jobs", [])
        failed_jobs = [j for j in training_jobs if j.get("status") == "failed"]
        if failed_jobs:
            for job in failed_jobs[-3:]:  # Report last 3 failures
                alerts.append(
                    f"FAILED: {job.get('job_type')} {job.get('board_type')}_{job.get('num_players')}p - {job.get('error_message', 'unknown')}"
                )

        # Check if leader role is as expected
        if status.get("role") != "leader":
            alerts.append(f"WARNING: Node {status.get('node_id')} is {status.get('role')}, not leader")

        return alerts

    def print_status(self, health: Dict, status: Dict, training: Dict):
        """Print formatted status."""
        timestamp = self.format_timestamp()

        # Count training jobs by status
        jobs = training.get("jobs", [])
        running_nnue = len([j for j in jobs if j.get("status") == "running" and j.get("job_type") == "nnue"])
        running_cmaes = len([j for j in jobs if j.get("status") == "running" and j.get("job_type") == "cmaes"])
        failed = len([j for j in jobs if j.get("status") == "failed"])

        # Build status line
        node_id = status.get("node_id", "unknown")
        role = status.get("role", "unknown")
        selfplay = health.get("selfplay_jobs", 0)
        cpu = health.get("cpu_percent", 0)
        mem = health.get("memory_percent", 0)
        disk = health.get("disk_percent", 0)

        cluster_util = health.get("cluster_utilization", {})
        selfplay_rate = cluster_util.get("selfplay_rate", 0)

        # Print status line
        print(f"\n[{timestamp}] === Cluster Status (check #{self.check_count}) ===")
        print(f"  Leader: {node_id} ({role})")
        print(f"  Selfplay: {selfplay} jobs | Rate: {selfplay_rate}/hr")
        print(f"  Training: {running_nnue} NNUE + {running_cmaes} CMA-ES running | {failed} failed")
        print(f"  Resources: CPU {cpu:.0f}% | Mem {mem:.1f}% | Disk {disk:.1f}%")

        # Print auto-training status
        thresholds = training.get("thresholds", {})
        auto_nnue = thresholds.get("auto_nnue_enabled", False)
        auto_cmaes = thresholds.get("auto_cmaes_enabled", False)
        print(f"  Auto-train: NNUE={'ON' if auto_nnue else 'OFF'} | CMA-ES={'ON' if auto_cmaes else 'OFF'}")

        # Print alerts
        alerts = self.check_for_issues(health, status, training)
        if alerts:
            print("  ALERTS:")
            for alert in alerts:
                print(f"    ! {alert}")
        else:
            print("  Status: OK")

    async def run_once(self):
        """Run a single monitoring check."""
        self.check_count += 1

        # Fetch all status info in parallel
        health_task = asyncio.create_task(self.check_health())
        status_task = asyncio.create_task(self.check_status())
        training_task = asyncio.create_task(self.check_training())

        health = await health_task
        status = await status_task
        training = await training_task

        # Print status
        self.print_status(health, status, training)

        return health, status, training

    async def run(self):
        """Run continuous monitoring."""
        print(f"Starting cluster monitoring (interval: {self.interval}s)")
        print(f"Leader URL: {self.leader_url}")
        print("-" * 60)

        while True:
            try:
                await self.run_once()
            except Exception as e:
                print(f"\n[{self.format_timestamp()}] Monitor error: {e}")

            await asyncio.sleep(self.interval)


async def main():
    parser = argparse.ArgumentParser(description="Monitor RingRift cluster")
    parser.add_argument(
        "--leader-url",
        default="http://150.136.65.197:8770",
        help="URL of the leader node"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=20,
        help="Check interval in seconds (default: 20)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit"
    )

    args = parser.parse_args()

    monitor = ClusterMonitor(args.leader_url, args.interval)

    if args.once:
        await monitor.run_once()
    else:
        await monitor.run()


if __name__ == "__main__":
    asyncio.run(main())
