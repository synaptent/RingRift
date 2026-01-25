#!/usr/bin/env python3
"""P2P Network Monitoring Script.

Records P2P network status for analysis. Run periodically to track stability.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

LOG_FILE = Path(__file__).parent.parent / "logs" / "p2p_monitoring.log"
P2P_URL = "http://localhost:8770/status"
TIMEOUT = 10


def get_p2p_status() -> dict | None:
    """Fetch P2P status from local orchestrator."""
    try:
        req = Request(P2P_URL, headers={"Accept": "application/json"})
        with urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except (URLError, json.JSONDecodeError, Exception) as e:
        return {"error": str(e)}


def analyze_status(status: dict) -> dict:
    """Extract key metrics from P2P status."""
    if "error" in status:
        return {"error": status["error"]}

    all_peers = status.get("all_peers", {})
    alive = [n for n, i in all_peers.items() if i.get("is_alive")]
    dead = [n for n, i in all_peers.items() if not i.get("is_alive")]

    voter_health = status.get("voter_health", {})

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "leader": status.get("leader_id"),
        "role": status.get("role"),
        "alive_peers": len(alive),
        "dead_peers": len(dead),
        "alive_nodes": sorted(alive),
        "dead_nodes": sorted(dead),
        "voter_quorum_ok": voter_health.get("quorum_ok", False),
        "voters_alive": voter_health.get("voters_alive", 0),
        "voters_total": voter_health.get("voters_total", 0),
        "work_queue_size": status.get("work_queue_size", 0),
        "active_jobs": status.get("active_jobs", 0),
        "selfplay_jobs": status.get("selfplay_jobs", 0),
    }


def log_status(metrics: dict):
    """Append status to monitoring log."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "a") as f:
        f.write(f"\n## Check: {metrics.get('timestamp', 'unknown')}\n")

        if "error" in metrics:
            f.write(f"ERROR: {metrics['error']}\n")
            return

        f.write(f"- Leader: {metrics['leader']}\n")
        f.write(f"- Alive peers: {metrics['alive_peers']}\n")
        f.write(f"- Dead peers: {metrics['dead_peers']}\n")
        f.write(f"- Voter quorum: {'OK' if metrics['voter_quorum_ok'] else 'LOST'} ({metrics['voters_alive']}/{metrics['voters_total']})\n")
        f.write(f"- Work queue: {metrics['work_queue_size']}\n")
        f.write(f"- Active jobs: {metrics['active_jobs']} (selfplay: {metrics['selfplay_jobs']})\n")
        f.write(f"- Alive: {', '.join(metrics['alive_nodes'][:15])}\n")
        if len(metrics['dead_nodes']) > 0:
            f.write(f"- Dead: {', '.join(metrics['dead_nodes'][:10])}\n")


def main():
    """Run status check and log results."""
    status = get_p2p_status()
    metrics = analyze_status(status)
    log_status(metrics)

    # Print summary to stdout
    if "error" in metrics:
        print(f"ERROR: {metrics['error']}")
        sys.exit(1)

    print(f"[{metrics['timestamp']}] Leader: {metrics['leader']} | "
          f"Alive: {metrics['alive_peers']} | "
          f"Dead: {metrics['dead_peers']} | "
          f"Quorum: {'OK' if metrics['voter_quorum_ok'] else 'LOST'}")

    # Return appropriate exit code
    if metrics['alive_peers'] >= 20:
        print("✓ Target of 20+ peers achieved!")
        sys.exit(0)
    elif metrics['alive_peers'] >= 10:
        print(f"! Making progress: {metrics['alive_peers']}/20 peers")
        sys.exit(0)
    else:
        print(f"✗ Below target: only {metrics['alive_peers']} peers alive")
        sys.exit(1)


if __name__ == "__main__":
    main()
