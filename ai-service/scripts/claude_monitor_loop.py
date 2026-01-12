#!/usr/bin/env python3
"""Claude monitoring loop for RingRift infrastructure.

Runs every 5 minutes to check:
1. P2P cluster health
2. Master loop status
3. Active jobs
4. Disk space
5. Any errors in logs

Takes corrective action when issues are detected.
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup PYTHONPATH for app imports
_ai_service_root = Path(__file__).parent.parent
sys.path.insert(0, str(_ai_service_root))

from app.config.ports import get_local_p2p_status_url

# Configuration
CHECK_INTERVAL = 300  # 5 minutes
LOG_FILE = Path("/tmp/claude_monitor.log")
P2P_URL = get_local_p2p_status_url()

def log(msg: str, level: str = "INFO"):
    """Log message with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def run_cmd(cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def check_p2p_health() -> dict:
    """Check P2P cluster health."""
    code, stdout, stderr = run_cmd(f"curl -s {P2P_URL}", timeout=10)
    if code != 0 or not stdout:
        return {"healthy": False, "error": "P2P unreachable"}

    try:
        data = json.loads(stdout)
        alive = data.get("alive_peers", 0)
        leader = data.get("leader_id", "unknown")
        return {
            "healthy": alive >= 5,
            "alive_peers": alive,
            "leader": leader,
            "node_id": data.get("node_id", "unknown"),
        }
    except json.JSONDecodeError:
        return {"healthy": False, "error": "Invalid JSON from P2P"}

def check_master_loop() -> dict:
    """Check master_loop.py status."""
    code, stdout, _ = run_cmd("pgrep -f 'master_loop.py' | wc -l")
    count = int(stdout.strip()) if stdout.strip().isdigit() else 0

    if count == 0:
        return {"healthy": False, "status": "not_running", "count": 0}
    elif count > 1:
        return {"healthy": False, "status": "multiple_instances", "count": count}
    else:
        return {"healthy": True, "status": "running", "count": 1}

def check_disk_space() -> dict:
    """Check disk space on key directories."""
    code, stdout, _ = run_cmd("df -h /Users/armand/Development/RingRift/ai-service | tail -1")
    if code != 0:
        return {"healthy": False, "error": "df failed"}

    parts = stdout.split()
    if len(parts) >= 5:
        use_pct = int(parts[4].replace("%", ""))
        return {
            "healthy": use_pct < 85,
            "use_percent": use_pct,
            "available": parts[3],
        }
    return {"healthy": True, "use_percent": 0}

def check_active_jobs() -> dict:
    """Check active gauntlet/training jobs."""
    code, stdout, _ = run_cmd("pgrep -f 'auto_promote|train.py|selfplay' | wc -l")
    count = int(stdout.strip()) if stdout.strip().isdigit() else 0
    return {"active_jobs": count}

def fix_duplicate_master_loops():
    """Kill duplicate master_loop instances, keeping the oldest."""
    code, stdout, _ = run_cmd("pgrep -f 'master_loop.py'")
    pids = [p.strip() for p in stdout.strip().split("\n") if p.strip()]

    if len(pids) > 1:
        log(f"Found {len(pids)} master_loop instances: {pids}", "WARN")
        # Kill all but the first (oldest) - use SIGKILL for stubborn processes
        for pid in pids[1:]:
            log(f"Force killing duplicate master_loop PID {pid}", "ACTION")
            run_cmd(f"kill -9 {pid}")
        return True
    return False

def restart_p2p_if_needed(p2p_status: dict) -> bool:
    """Attempt to restart P2P if unhealthy."""
    if p2p_status.get("healthy"):
        return False

    # Check if P2P process exists
    code, stdout, _ = run_cmd("pgrep -f 'p2p_orchestrator'")
    if not stdout.strip():
        log("P2P orchestrator not running - consider manual restart", "WARN")
        return False

    # If running but unhealthy, log but don't auto-restart (too risky)
    log(f"P2P unhealthy: {p2p_status.get('error', 'unknown')}", "WARN")
    return False

def run_check_cycle():
    """Run one monitoring cycle."""
    log("=" * 60)
    log("Starting health check cycle")

    # 1. P2P Health
    p2p = check_p2p_health()
    if p2p.get("healthy"):
        log(f"P2P: OK - {p2p.get('alive_peers')} peers, leader={p2p.get('leader')}")
    else:
        log(f"P2P: UNHEALTHY - {p2p.get('error', 'unknown')}", "WARN")
        restart_p2p_if_needed(p2p)

    # 2. Master Loop
    ml = check_master_loop()
    if ml.get("healthy"):
        log(f"Master Loop: OK - running")
    elif ml.get("status") == "multiple_instances":
        log(f"Master Loop: {ml.get('count')} instances running", "WARN")
        if fix_duplicate_master_loops():
            log("Fixed duplicate master_loop instances", "ACTION")
    else:
        log("Master Loop: NOT RUNNING", "WARN")

    # 3. Disk Space
    disk = check_disk_space()
    if disk.get("healthy"):
        log(f"Disk: OK - {disk.get('use_percent')}% used, {disk.get('available')} available")
    else:
        log(f"Disk: LOW SPACE - {disk.get('use_percent')}% used", "WARN")

    # 4. Active Jobs
    jobs = check_active_jobs()
    log(f"Active Jobs: {jobs.get('active_jobs')} training/eval processes")

    log("Health check cycle complete")
    return {
        "p2p": p2p,
        "master_loop": ml,
        "disk": disk,
        "jobs": jobs,
    }

def main():
    """Main monitoring loop."""
    log("Claude Monitor Loop starting")
    log(f"Check interval: {CHECK_INTERVAL} seconds")
    log(f"Log file: {LOG_FILE}")

    # Handle graceful shutdown
    def handle_signal(signum, frame):
        log("Received shutdown signal, exiting")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    cycle = 0
    while True:
        cycle += 1
        log(f"Cycle {cycle}")

        try:
            run_check_cycle()
        except Exception as e:
            log(f"Error in check cycle: {e}", "ERROR")

        log(f"Sleeping for {CHECK_INTERVAL} seconds...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
