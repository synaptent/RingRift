#!/usr/bin/env python3
"""P2P Orchestrator Watchdog - Ensures P2P orchestrator stays running and connected.

This script should be run periodically via cron to ensure the P2P orchestrator
stays healthy and connected to the cluster.

Features:
- Checks if P2P process is running and responsive
- Uses systemctl for managed services, falls back to direct process control
- Prevents restart loops by requiring minimum process uptime before restart
- Configurable peer count requirements

Usage:
    # Check and restart if needed
    python scripts/p2p_watchdog.py --node-id lambda-h100 --peers http://100.97.104.89:8770

    # Cron entry (every 5 minutes - recommended to allow service to stabilize):
    */5 * * * * cd /home/ubuntu/ringrift/ai-service && python3 scripts/p2p_watchdog.py --node-id lambda-h100 --peers http://100.97.104.89:8770,http://100.91.25.13:8770 >> /tmp/p2p_watchdog.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from typing import Optional, List


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def check_p2p_health(port: int = 8770, timeout: int = 30) -> Optional[dict]:
    """Check if local P2P orchestrator is healthy."""
    try:
        url = f"http://localhost:{port}/health"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except Exception:
        return None


def check_cluster_peers(port: int = 8770) -> int:
    """Get number of online peers in cluster."""
    try:
        url = f"http://localhost:{port}/api/cluster/status"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            peers = data.get("peers", [])
            online = sum(1 for p in peers if p.get("status") == "online")
            return online
    except Exception:
        return 0


def is_p2p_running() -> bool:
    """Check if p2p_orchestrator process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "p2p_orchestrator"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def get_p2p_uptime_seconds() -> float:
    """Get how long the P2P process has been running (in seconds).

    Returns 0 if process not found or error.
    """
    try:
        # Get PID of p2p_orchestrator
        result = subprocess.run(
            ["pgrep", "-f", "p2p_orchestrator"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return 0.0

        pid = result.stdout.strip().split('\n')[0]
        if not pid:
            return 0.0

        # Get process start time using ps
        ps_result = subprocess.run(
            ["ps", "-p", pid, "-o", "etimes="],
            capture_output=True,
            text=True
        )
        if ps_result.returncode != 0:
            return 0.0

        uptime_str = ps_result.stdout.strip()
        return float(uptime_str) if uptime_str else 0.0
    except Exception:
        return 0.0


# Minimum uptime before considering a restart (prevents restart loops)
MIN_UPTIME_BEFORE_RESTART = 180  # 3 minutes


def is_systemd_service_available() -> bool:
    """Check if ringrift-p2p.service is available."""
    try:
        result = subprocess.run(
            ["systemctl", "is-enabled", "ringrift-p2p.service"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def stop_p2p():
    """Stop any running P2P orchestrator."""
    try:
        if is_systemd_service_available():
            # Use systemctl for systemd-managed service
            subprocess.run(["sudo", "systemctl", "stop", "ringrift-p2p.service"], timeout=30)
            time.sleep(2)
        else:
            # Fallback to pkill
            subprocess.run(["pkill", "-f", "p2p_orchestrator"], timeout=5)
            time.sleep(2)
            if is_p2p_running():
                subprocess.run(["pkill", "-9", "-f", "p2p_orchestrator"], timeout=5)
                time.sleep(1)
    except Exception as e:
        log(f"Warning: Error stopping P2P: {e}")


def start_p2p(node_id: str, port: int, peers: List[str], ringrift_path: str):
    """Start the P2P orchestrator."""
    if is_systemd_service_available():
        # Use systemctl for systemd-managed service
        log("Starting P2P via systemctl")
        try:
            subprocess.run(["sudo", "systemctl", "start", "ringrift-p2p.service"], timeout=30)
            return True
        except Exception as e:
            log(f"Error starting P2P via systemctl: {e}")
            return False

    # Fallback to direct start
    script_path = os.path.join(ringrift_path, "ai-service", "scripts", "p2p_orchestrator.py")

    if not os.path.exists(script_path):
        log(f"Error: P2P script not found at {script_path}")
        return False

    cmd = [
        "python3", script_path,
        "--node-id", node_id,
        "--port", str(port),
    ]

    if peers:
        cmd.extend(["--peers", ",".join(peers)])

    log(f"Starting P2P: {' '.join(cmd)}")

    try:
        # Start in background
        log_file = open("/tmp/p2p.log", "a")
        subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=os.path.join(ringrift_path, "ai-service"),
            start_new_session=True
        )
        return True
    except Exception as e:
        log(f"Error starting P2P: {e}")
        return False


def find_ringrift_path() -> str:
    """Find the RingRift installation path."""
    # Common locations
    candidates = [
        os.path.expanduser("~/ringrift"),
        os.path.expanduser("~/Development/RingRift"),
        "/home/ubuntu/ringrift",
        "/root/ringrift",
    ]

    for path in candidates:
        if os.path.exists(os.path.join(path, "ai-service", "scripts", "p2p_orchestrator.py")):
            return path

    # Try to find from current directory
    cwd = os.getcwd()
    if "ringrift" in cwd.lower():
        # Walk up to find ringrift root
        parts = cwd.split(os.sep)
        for i, part in enumerate(parts):
            if "ringrift" in part.lower():
                candidate = os.sep.join(parts[:i+1])
                if os.path.exists(os.path.join(candidate, "ai-service", "scripts", "p2p_orchestrator.py")):
                    return candidate

    return os.path.expanduser("~/ringrift")


def main():
    parser = argparse.ArgumentParser(description="P2P Orchestrator Watchdog")
    parser.add_argument("--node-id", required=True, help="Node identifier")
    parser.add_argument("--port", type=int, default=8770, help="P2P port (default: 8770)")
    parser.add_argument("--peers", help="Comma-separated list of peer URLs")
    parser.add_argument("--min-peers", type=int, default=0, help="Minimum required online peers (default: 0)")
    parser.add_argument("--ringrift-path", help="Path to RingRift installation")
    parser.add_argument("--force-restart", action="store_true", help="Force restart even if healthy")

    args = parser.parse_args()

    peers = args.peers.split(",") if args.peers else []
    ringrift_path = args.ringrift_path or find_ringrift_path()

    log(f"Watchdog check for {args.node_id}")

    # Check if P2P is running
    if not is_p2p_running():
        log("P2P orchestrator not running, starting...")
        if start_p2p(args.node_id, args.port, peers, ringrift_path):
            time.sleep(5)
            health = check_p2p_health(args.port)
            if health:
                log(f"P2P started successfully: {health.get('node_id')} as {health.get('role')}")
            else:
                log("P2P started but not responding yet")
        else:
            log("Failed to start P2P")
            sys.exit(1)
        return

    # Check health
    health = check_p2p_health(args.port)
    if not health:
        # Check process uptime before restarting to prevent restart loops
        uptime = get_p2p_uptime_seconds()
        if uptime > 0 and uptime < MIN_UPTIME_BEFORE_RESTART:
            log(f"P2P not responding but only {uptime:.0f}s old (< {MIN_UPTIME_BEFORE_RESTART}s), skipping restart")
            return
        log(f"P2P running but not responding (uptime: {uptime:.0f}s), restarting...")
        stop_p2p()
        time.sleep(2)
        start_p2p(args.node_id, args.port, peers, ringrift_path)
        time.sleep(5)
        log("Restarted P2P orchestrator")
        return

    # Check if healthy
    if not health.get("healthy", False):
        log(f"P2P unhealthy (disk={health.get('disk_percent'):.1f}%, mem={health.get('memory_percent'):.1f}%), may need attention")

    # Check peer count if required
    if args.min_peers > 0:
        online_peers = check_cluster_peers(args.port)
        if online_peers < args.min_peers:
            log(f"Only {online_peers} peers online (need {args.min_peers}), restarting...")
            stop_p2p()
            time.sleep(2)
            start_p2p(args.node_id, args.port, peers, ringrift_path)
            time.sleep(5)
            return

    # Force restart if requested
    if args.force_restart:
        log("Force restart requested")
        stop_p2p()
        time.sleep(2)
        start_p2p(args.node_id, args.port, peers, ringrift_path)
        time.sleep(5)
        return

    # All good
    log(f"P2P healthy: {health.get('node_id')} as {health.get('role')}, {health.get('selfplay_jobs', 0)} selfplay jobs")


if __name__ == "__main__":
    main()
