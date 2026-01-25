#!/usr/bin/env python3
"""Simple P2P stability monitor using status file instead of HTTP.

Avoids HTTP timeouts when event loop is under load.
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter

STATUS_FILE = Path("/tmp/p2p_status.json")
LOG_FILE = Path("/Users/armand/Development/RingRift/ai-service/logs/p2p_file_monitor.log")


def get_status():
    """Read status from file."""
    if not STATUS_FILE.exists():
        return None
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def assess_stability(status) -> tuple[str, dict]:
    """Assess cluster stability."""
    if not status:
        return "ERROR", {"reason": "No status file"}

    alive = status.get("alive_peers", 0)
    leader = status.get("leader_id") or status.get("effective_leader_id")
    voters = status.get("voters_alive", 0)
    quorum = status.get("voter_quorum_ok", status.get("quorum_ok", voters >= 2))

    # Check leader agreement from leaders_reported field
    leaders_reported = status.get("leaders_reported", [])
    leader_votes = Counter(leaders_reported)

    total_reports = len(leaders_reported) if leaders_reported else alive  # Fallback to alive count
    agreement_pct = 0
    if leader and total_reports > 0:
        # If leaders_reported is non-empty, use it; otherwise assume all alive agree
        if leaders_reported:
            agreement_pct = leader_votes.get(leader, 0) / total_reports * 100
        else:
            agreement_pct = 100 if alive > 0 else 0  # Assume consensus if no disagreement data

    details = {
        "leader": leader,
        "alive": alive,
        "voters": voters,
        "quorum": quorum,
        "agreement_pct": agreement_pct,
        "leader_votes": dict(leader_votes),
    }

    # Stability assessment
    if not quorum:
        return "CRITICAL", details
    if not leader:
        return "UNSTABLE", details
    if alive < 10:
        return "DEGRADED", details
    if agreement_pct < 50:
        return "UNSTABLE", details
    if alive >= 20 and agreement_pct >= 80:
        return "GOOD", details
    return "MODERATE", details


def log(msg):
    """Log to file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} - {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=6, help="Number of checks")
    parser.add_argument("--interval", type=float, default=10.0, help="Minutes between checks")
    args = parser.parse_args()

    log(f"=== P2P File-Based Monitor ===")
    log(f"Cycles: {args.cycles}, Interval: {args.interval} min")

    results = []
    for i in range(args.cycles):
        status = get_status()
        stability, details = assess_stability(status)
        results.append(stability)

        log(f"--- Check {i+1}/{args.cycles} ---")
        log(f"  Status: {stability}")
        log(f"  Leader: {details.get('leader', 'None')}")
        log(f"  Alive: {details.get('alive', 0)}")
        log(f"  Voters: {details.get('voters', 0)}/7")
        log(f"  Agreement: {details.get('agreement_pct', 0):.0f}%")

        if i < args.cycles - 1:
            time.sleep(args.interval * 60)

    # Summary
    log("=== Summary ===")
    status_counts = Counter(results)
    for status, count in status_counts.items():
        log(f"  {status}: {count}")

    # Count DEGRADED as partially stable (has leader, has quorum, just low peer count)
    good_count = status_counts.get("GOOD", 0) + status_counts.get("MODERATE", 0) + status_counts.get("DEGRADED", 0)
    stability_pct = good_count / len(results) * 100
    log(f"Stability Score: {stability_pct:.0f}%")

    return 0 if stability_pct >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
