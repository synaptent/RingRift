#!/usr/bin/env python3
"""Lightweight P2P health monitor using /health endpoints instead of /status.

The /status endpoint blocks when event loop is overloaded.
This script uses the lighter /health endpoint for monitoring.
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path
import subprocess

LOG_FILE = Path("/Users/armand/Development/RingRift/ai-service/logs/p2p_health_monitor.log")

# Key nodes to check
NODES = {
    "local-mac": "http://localhost:8770",
    "hetzner-cpu1": "http://100.94.174.19:8770",
    "hetzner-cpu2": "http://100.67.131.72:8770",
    "hetzner-cpu3": "http://100.126.21.102:8770",
    "vultr-a100-20gb": "http://100.109.195.71:8770",
    "nebius-h100-1": "http://100.106.19.6:8770",
    "lambda-gh200-1": "http://100.71.89.91:8770",
    "lambda-gh200-3": "http://100.77.77.122:8770",
}


def check_node(name: str, url: str) -> dict:
    """Check a single node's health."""
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "5", f"{url}/health"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            return {
                "status": "alive",
                "healthy": data.get("healthy", False),
                "leader": data.get("leader_id"),
                "role": data.get("role"),
                "active_peers": data.get("active_peers", 0),
            }
        return {"status": "unreachable", "error": "no response"}
    except json.JSONDecodeError:
        return {"status": "error", "error": "invalid json"}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "curl timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def log(msg: str):
    """Log to file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} - {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_check():
    """Run a single health check cycle."""
    log("--- Health Check ---")
    alive_count = 0
    leaders = []

    for name, url in NODES.items():
        result = check_node(name, url)
        status = result["status"]

        if status == "alive":
            alive_count += 1
            leader = result.get("leader")
            if leader:
                leaders.append(leader)
            log(f"  {name}: {status} (role={result.get('role')}, leader={leader}, peers={result.get('active_peers')})")
        else:
            log(f"  {name}: {status} ({result.get('error', 'unknown')})")

    # Assess stability
    leader_consensus = len(set(leaders)) <= 1 if leaders else False
    common_leader = leaders[0] if leader_consensus and leaders else "none"

    log(f"Summary: {alive_count}/{len(NODES)} alive, leader consensus: {leader_consensus} ({common_leader})")

    return {
        "alive_count": alive_count,
        "total_nodes": len(NODES),
        "leader_consensus": leader_consensus,
        "common_leader": common_leader,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=6, help="Number of checks")
    parser.add_argument("--interval", type=float, default=10.0, help="Minutes between checks")
    args = parser.parse_args()

    log(f"=== P2P Health Monitor ===")
    log(f"Cycles: {args.cycles}, Interval: {args.interval} min")

    results = []
    for i in range(args.cycles):
        log(f"\n=== Cycle {i+1}/{args.cycles} ===")
        result = run_check()
        results.append(result)

        if i < args.cycles - 1:
            log(f"Sleeping {args.interval} minutes...")
            time.sleep(args.interval * 60)

    # Summary
    log("\n=== Final Summary ===")
    avg_alive = sum(r["alive_count"] for r in results) / len(results)
    consensus_count = sum(1 for r in results if r["leader_consensus"])
    log(f"Average alive: {avg_alive:.1f}/{len(NODES)}")
    log(f"Leader consensus: {consensus_count}/{len(results)} cycles")

    # Assess overall stability
    stable = avg_alive >= 5 and consensus_count >= len(results) * 0.8
    log(f"Stability: {'GOOD' if stable else 'UNSTABLE'}")

    return 0 if stable else 1


if __name__ == "__main__":
    sys.exit(main())
