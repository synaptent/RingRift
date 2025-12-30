#!/usr/bin/env python3
"""10-hour cluster monitoring with automatic problem detection and fixing.

This script monitors the P2P cluster and takes corrective action when issues are detected.

Features:
- Checks P2P cluster status every 5 minutes
- Detects stalled training, low selfplay, quorum issues
- Automatically restarts local P2P if isolated
- Logs all activity with timestamps
- Runs for 10 hours then exits

Usage:
    python scripts/monitor_cluster_10h.py                    # Full 10-hour monitoring
    python scripts/monitor_cluster_10h.py --once             # Single check
    python scripts/monitor_cluster_10h.py --duration 2       # 2-hour monitoring
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
P2P_LEADER_IP = "100.94.201.92"  # vultr-a100-20gb
P2P_PORT = 8770
CHECK_INTERVAL = 300  # 5 minutes
DEFAULT_DURATION_HOURS = 10

# Thresholds for problem detection
MIN_SELFPLAY_JOBS = 50
MIN_TRAINING_JOBS = 1
MIN_ALIVE_NODES = 5
MIN_VOTERS_ALIVE = 3

LOG_FILE = Path(__file__).parent.parent / "logs" / "monitor_cluster_10h.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str, level: str = "INFO"):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_cluster_status(timeout: int = 30) -> dict | None:
    """Fetch cluster status from P2P leader."""
    try:
        url = f"http://{P2P_LEADER_IP}:{P2P_PORT}/status"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as e:
        log(f"Failed to get cluster status: {e}", "ERROR")
        return None


def get_local_p2p_status(timeout: int = 10) -> dict | None:
    """Fetch local P2P status."""
    try:
        url = f"http://localhost:{P2P_PORT}/status"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def analyze_cluster(status: dict) -> dict:
    """Analyze cluster status and identify issues."""
    issues = []
    metrics = {}

    # Count jobs
    peers = status.get("peers", {})
    self_info = status.get("self", {})

    total_selfplay = self_info.get("selfplay_jobs", 0)
    total_training = self_info.get("training_jobs", 0)
    alive_count = 1
    training_nodes = []

    for name, info in peers.items():
        if isinstance(info, dict) and info.get("is_alive"):
            alive_count += 1
            total_selfplay += info.get("selfplay_jobs", 0)
            total_training += info.get("training_jobs", 0)
            if info.get("training_jobs", 0) > 0:
                training_nodes.append(name)

    metrics["selfplay_jobs"] = total_selfplay
    metrics["training_jobs"] = total_training
    metrics["alive_nodes"] = alive_count
    metrics["training_nodes"] = training_nodes
    metrics["voter_quorum_ok"] = status.get("voter_quorum_ok", False)
    metrics["voters_alive"] = status.get("voters_alive", 0)
    metrics["leader"] = status.get("leader_id", "unknown")

    # Check for issues
    if not metrics["voter_quorum_ok"]:
        issues.append(f"CRITICAL: Voter quorum not met ({metrics['voters_alive']}/{MIN_VOTERS_ALIVE})")

    if total_selfplay < MIN_SELFPLAY_JOBS:
        issues.append(f"WARNING: Low selfplay jobs ({total_selfplay} < {MIN_SELFPLAY_JOBS})")

    if total_training < MIN_TRAINING_JOBS:
        issues.append(f"WARNING: No training jobs running")

    if alive_count < MIN_ALIVE_NODES:
        issues.append(f"WARNING: Low alive nodes ({alive_count} < {MIN_ALIVE_NODES})")

    return {"metrics": metrics, "issues": issues}


def check_local_p2p_health() -> dict:
    """Check if local P2P is healthy and connected."""
    status = get_local_p2p_status()

    if status is None:
        return {"healthy": False, "reason": "Local P2P not responding"}

    alive_peers = status.get("alive_peers", 0)
    if alive_peers < 3:
        return {"healthy": False, "reason": f"Local P2P isolated ({alive_peers} peers)"}

    voter_quorum = status.get("voter_quorum_ok", False)
    if not voter_quorum:
        return {"healthy": False, "reason": "Local P2P has no quorum"}

    return {
        "healthy": True,
        "node_id": status.get("node_id"),
        "leader": status.get("leader_id"),
        "peers": alive_peers
    }


def restart_local_p2p():
    """Restart the local P2P orchestrator."""
    log("Attempting to restart local P2P...", "ACTION")

    # Kill existing
    try:
        subprocess.run(["pkill", "-f", "p2p_orchestrator"], timeout=5)
        time.sleep(2)
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        pass

    # Start new
    try:
        script_path = Path(__file__).parent / "p2p_orchestrator.py"
        if script_path.exists():
            cmd = [
                "python3", str(script_path),
                "--node-id", "local-mac",
                "--port", str(P2P_PORT),
                "--peers", f"{P2P_LEADER_IP}:{P2P_PORT}"
            ]

            log_file = open("/tmp/p2p_restart.log", "a")
            subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                cwd=str(Path(__file__).parent.parent),
                start_new_session=True
            )
            log("Local P2P restart initiated", "ACTION")
            return True
    except Exception as e:
        log(f"Failed to restart local P2P: {e}", "ERROR")

    return False


def take_corrective_action(issues: list[str], metrics: dict) -> list[str]:
    """Take corrective action based on detected issues."""
    actions_taken = []

    # Check if local P2P needs restart
    local_health = check_local_p2p_health()
    if not local_health.get("healthy"):
        reason = local_health.get("reason", "unknown")
        log(f"Local P2P unhealthy: {reason}", "WARNING")

        if "isolated" in reason.lower() or "not responding" in reason.lower():
            if restart_local_p2p():
                actions_taken.append("Restarted local P2P (was isolated)")
                time.sleep(10)  # Wait for startup

    # Log severe issues that need manual intervention
    for issue in issues:
        if "CRITICAL" in issue:
            log(f"Manual intervention may be needed: {issue}", "ALERT")

    return actions_taken


def print_status_summary(metrics: dict, issues: list[str]):
    """Print a summary of cluster status."""
    log("=" * 60)
    log("CLUSTER STATUS SUMMARY")
    log("=" * 60)
    log(f"Leader: {metrics.get('leader', 'unknown')}")
    log(f"Alive nodes: {metrics.get('alive_nodes', 0)}")
    log(f"Voter quorum: {'OK' if metrics.get('voter_quorum_ok') else 'FAILED'} ({metrics.get('voters_alive', 0)}/3)")
    log(f"Selfplay jobs: {metrics.get('selfplay_jobs', 0)}")
    log(f"Training jobs: {metrics.get('training_jobs', 0)}")

    if metrics.get('training_nodes'):
        log(f"Training on: {', '.join(metrics['training_nodes'][:5])}")

    if issues:
        log("-" * 40)
        log("ISSUES DETECTED:")
        for issue in issues:
            log(f"  - {issue}")
    else:
        log("-" * 40)
        log("No issues detected - cluster healthy!")
    log("=" * 60)


def run_single_check() -> bool:
    """Run a single monitoring check. Returns True if cluster is healthy."""
    status = get_cluster_status()

    if status is None:
        log("Cannot reach cluster leader - checking local P2P", "WARNING")
        local_health = check_local_p2p_health()
        if not local_health.get("healthy"):
            restart_local_p2p()
        return False

    analysis = analyze_cluster(status)
    metrics = analysis["metrics"]
    issues = analysis["issues"]

    print_status_summary(metrics, issues)

    if issues:
        actions = take_corrective_action(issues, metrics)
        if actions:
            log(f"Actions taken: {', '.join(actions)}")

    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="10-hour cluster monitoring")
    parser.add_argument("--once", action="store_true", help="Single check only")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_HOURS,
                        help=f"Monitoring duration in hours (default: {DEFAULT_DURATION_HOURS})")
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL,
                        help=f"Check interval in seconds (default: {CHECK_INTERVAL})")
    args = parser.parse_args()

    log("=" * 60)
    log("CLUSTER MONITORING STARTED")
    log(f"Duration: {args.duration} hours")
    log(f"Check interval: {args.interval} seconds")
    log("=" * 60)

    if args.once:
        run_single_check()
        return

    # Continuous monitoring
    end_time = datetime.now() + timedelta(hours=args.duration)
    check_count = 0
    healthy_count = 0

    try:
        while datetime.now() < end_time:
            check_count += 1
            log(f"\n--- Check #{check_count} ---")

            if run_single_check():
                healthy_count += 1

            remaining = end_time - datetime.now()
            if remaining.total_seconds() > 0:
                next_check = min(args.interval, remaining.total_seconds())
                log(f"Next check in {int(next_check)} seconds... ({remaining.total_seconds()/3600:.1f}h remaining)")
                time.sleep(next_check)

    except KeyboardInterrupt:
        log("\nMonitoring stopped by user")

    # Final summary
    log("\n" + "=" * 60)
    log("MONITORING COMPLETE")
    log(f"Total checks: {check_count}")
    log(f"Healthy checks: {healthy_count} ({100*healthy_count/check_count:.1f}%)" if check_count > 0 else "No checks completed")
    log("=" * 60)


if __name__ == "__main__":
    main()
