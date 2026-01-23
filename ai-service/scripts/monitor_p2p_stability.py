#!/usr/bin/env python3
"""Monitor P2P cluster stability over time.

Logs peer counts and stability metrics to track whether 20+ nodes
remain connected for 4+ hours.

Usage:
    python scripts/monitor_p2p_stability.py [--interval 60] [--output stability.log]
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
import urllib.request
from pathlib import Path


def get_p2p_status(url: str = "http://localhost:8770/status") -> dict | None:
    """Fetch P2P status from local orchestrator."""
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def log_status(status: dict, output_file: Path | None, start_time: float) -> None:
    """Log status to console and optionally to file."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    uptime_hours = (time.time() - start_time) / 3600

    if "error" in status:
        line = f"{now} | UPTIME: {uptime_hours:.2f}h | ERROR: {status['error']}"
    else:
        alive = status.get("alive_peers", 0)
        leader = status.get("leader_id", "none")
        voters = status.get("voter_health", {}).get("voters_alive", 0)
        all_peers = len(status.get("all_peers", {}))

        # Check stability criteria
        stable = "YES" if alive >= 20 else "NO"

        line = f"{now} | UPTIME: {uptime_hours:.2f}h | ALIVE: {alive:2d} | ALL_PEERS: {all_peers:2d} | VOTERS: {voters} | LEADER: {leader:20s} | 20+ STABLE: {stable}"

    print(line)
    sys.stdout.flush()

    if output_file:
        with open(output_file, "a") as f:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor P2P cluster stability")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--output", type=str, default="stability_monitor.log", help="Output log file")
    parser.add_argument("--target-hours", type=float, default=4.0, help="Target stability hours")
    args = parser.parse_args()

    output_file = Path(args.output)
    start_time = time.time()
    target_seconds = args.target_hours * 3600

    print(f"P2P Cluster Stability Monitor")
    print(f"Target: 20+ nodes for {args.target_hours} hours")
    print(f"Logging to: {output_file}")
    print(f"Check interval: {args.interval}s")
    print("-" * 100)

    # Track stability
    consecutive_stable = 0
    stable_start = None

    while True:
        status = get_p2p_status()
        log_status(status, output_file, start_time)

        # Track stability periods
        if status and "error" not in status:
            alive = status.get("alive_peers", 0)
            if alive >= 20:
                if stable_start is None:
                    stable_start = time.time()
                consecutive_stable += 1
                stable_duration = time.time() - stable_start

                if stable_duration >= target_seconds:
                    print(f"\n{'='*80}")
                    print(f"SUCCESS! 20+ nodes stable for {stable_duration/3600:.2f} hours!")
                    print(f"{'='*80}")
            else:
                if stable_start is not None:
                    stable_duration = time.time() - stable_start
                    print(f"[STABILITY LOST] Was stable for {stable_duration/60:.1f} minutes")
                stable_start = None
                consecutive_stable = 0

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
