#!/usr/bin/env python3
"""
Cluster Monitor Daemon - Ensures all nodes are active and utilized.

Runs for a specified duration, checking cluster health every 3-5 minutes
and taking corrective actions to restore optimal utilization.

Usage:
    python scripts/cluster_monitor_daemon.py --duration-hours 10
"""

import argparse
import datetime
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = PROJECT_ROOT / "logs" / "cluster_monitor_daemon.log"
STATE_FILE = PROJECT_ROOT / "logs" / "cluster_monitor_state.json"

# Check intervals (randomized between 3-5 minutes)
MIN_INTERVAL_SECONDS = 180  # 3 minutes
MAX_INTERVAL_SECONDS = 300  # 5 minutes


def log(message: str, level: str = "INFO"):
    """Log message to file and stdout."""
    timestamp = datetime.datetime.now().isoformat()
    log_line = f"[{timestamp}] [{level}] {message}"
    print(log_line)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")


def run_command(cmd: list[str], timeout: int = 120, env_extra: dict = None) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        if env_extra:
            env.update(env_extra)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT / "ai-service"),
            env=env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


# Key cluster nodes to check via SSH
CLUSTER_NODES = [
    {"name": "lambda-gh200-e", "host": "ubuntu@lambda-gh200-e", "check": "squeue"},
    {"name": "A40", "host": "root@ssh8.vast.ai", "port": "38742", "check": "nvidia-smi"},
    {"name": "5070", "host": "root@ssh2.vast.ai", "port": "10042", "check": "nvidia-smi"},
]


def check_node_health_ssh() -> dict:
    """Check node health via SSH for key cluster nodes."""
    log("Checking cluster node health via SSH...")

    results = {}
    healthy_count = 0

    for node in CLUSTER_NODES:
        name = node["name"]
        host = node["host"]
        check_cmd = node.get("check", "uptime")
        port = node.get("port")

        ssh_cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no"]
        if port:
            ssh_cmd.extend(["-p", port])
        ssh_cmd.extend([host, check_cmd])

        rc, stdout, stderr = run_command(ssh_cmd, timeout=30)

        if rc == 0:
            healthy_count += 1
            results[name] = {"healthy": True, "output": stdout[:200]}
            log(f"  {name}: OK")
        else:
            results[name] = {"healthy": False, "error": stderr[:200]}
            log(f"  {name}: FAILED - {stderr[:100]}", "WARN")

    log(f"SSH health: {healthy_count}/{len(CLUSTER_NODES)} nodes healthy")
    return {"nodes": results, "healthy_count": healthy_count, "total": len(CLUSTER_NODES)}


def check_slurm_status() -> dict:
    """Check Slurm cluster status and return summary."""
    log("Checking Slurm cluster status...")

    rc, stdout, stderr = run_command([
        "python", "scripts/cluster_submit.py", "status"
    ])

    if rc != 0:
        log(f"Slurm status check failed: {stderr}", "ERROR")
        return {"healthy": False, "error": stderr}

    # Parse output to count idle/busy nodes
    lines = stdout.strip().split("\n")
    idle_count = 0
    busy_count = 0
    total_count = 0

    for line in lines:
        if "idle" in line.lower():
            idle_count += 1
            total_count += 1
        elif "running" in line.lower() or "busy" in line.lower() or "allocated" in line.lower():
            busy_count += 1
            total_count += 1

    log(f"Slurm: {busy_count} busy, {idle_count} idle, {total_count} total")

    return {
        "healthy": True,
        "idle": idle_count,
        "busy": busy_count,
        "total": total_count,
        "raw_output": stdout[:1000]  # Truncate for state file
    }


def fill_idle_slurm_nodes() -> dict:
    """Fill idle Slurm nodes with selfplay jobs."""
    log("Filling idle Slurm nodes with selfplay jobs...")

    # Rotate between different configurations
    configs = [
        ("square19", "3"),
        ("square19", "2"),
        ("hexagonal", "3"),
        ("hexagonal", "2"),
    ]
    board_type, player_count = random.choice(configs)

    rc, stdout, stderr = run_command([
        "python", "scripts/cluster_submit.py", "fill-idle",
        "--board", board_type,
        "--players", player_count,
        "--games", "100",
    ], timeout=180)

    if rc != 0:
        log(f"Fill-idle failed: {stderr}", "WARN")
        return {"success": False, "error": stderr}

    log(f"Fill-idle result: {stdout[:500]}")
    return {"success": True, "config": f"{board_type}_{player_count}p", "output": stdout[:500]}


def check_tailscale_status() -> dict:
    """Check Tailscale mesh connectivity."""
    log("Checking Tailscale mesh status...")

    rc, stdout, stderr = run_command(["tailscale", "status", "--json"], timeout=30)

    if rc != 0:
        log(f"Tailscale check failed: {stderr}", "WARN")
        return {"healthy": False, "error": stderr}

    try:
        data = json.loads(stdout)
        peers = data.get("Peer", {})
        online = sum(1 for p in peers.values() if p.get("Online", False))
        total = len(peers)
        log(f"Tailscale: {online}/{total} peers online")
        return {"healthy": True, "online": online, "total": total}
    except json.JSONDecodeError:
        log("Failed to parse Tailscale JSON output", "WARN")
        return {"healthy": False, "error": "JSON parse error"}


def check_vast_instances() -> dict:
    """Check Vast.ai instance status using project scripts."""
    log("Checking Vast.ai instances...")

    # Use project's vast_lifecycle.py script for status
    rc, stdout, stderr = run_command([
        "python", "scripts/vast_lifecycle.py", "--status"
    ], timeout=60)

    if rc != 0:
        # Fall back to trying vastai Python module directly
        try:
            from vastai import api as vast_api
            instances = vast_api.show_instances()
            if instances:
                running = sum(1 for i in instances if i.get("actual_status") == "running")
                total = len(instances)
                log(f"Vast.ai: {running}/{total} instances running")
                return {"healthy": True, "running": running, "total": total}
        except Exception as e:
            log(f"Vast check: {e}", "INFO")
            return {"healthy": True, "note": "Vast.ai not configured or no instances"}

    log(f"Vast.ai: {stdout[:300]}")
    return {"healthy": True, "output": stdout[:500]}


def check_lambda_instances() -> dict:
    """Check Lambda Cloud instance status."""
    log("Checking Lambda Cloud instances...")

    rc, stdout, stderr = run_command([
        "python", "scripts/lambda_cli.py", "list"
    ], timeout=60)

    if rc != 0:
        # Lambda CLI might not be configured
        log(f"Lambda check: {stderr or 'No instances or not configured'}", "INFO")
        return {"healthy": True, "note": "Lambda CLI not configured or no instances"}

    log(f"Lambda: {stdout[:300]}")
    return {"healthy": True, "output": stdout[:500]}


def run_vast_autoscaler() -> dict:
    """Run Vast.ai autoscaler to ensure optimal instance count."""
    log("Running Vast.ai autoscaler...")

    rc, stdout, stderr = run_command([
        "python", "scripts/vast_autoscaler.py", "--check"
    ], timeout=120)

    if rc != 0:
        log(f"Vast autoscaler check: {stderr}", "INFO")
        return {"success": False, "note": stderr}

    log(f"Vast autoscaler: {stdout[:300]}")
    return {"success": True, "output": stdout[:500]}


def save_state(state: dict):
    """Save monitoring state to file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def run_monitoring_cycle() -> dict:
    """Run a complete monitoring cycle and return results."""
    cycle_start = datetime.datetime.now()
    log("=" * 60)
    log(f"Starting monitoring cycle at {cycle_start.isoformat()}")

    results = {
        "cycle_start": cycle_start.isoformat(),
        "checks": {},
        "actions": {},
    }

    # 1. Check Slurm status
    slurm_status = check_slurm_status()
    results["checks"]["slurm"] = slurm_status

    # 2. If idle nodes found, fill them
    if slurm_status.get("healthy") and slurm_status.get("idle", 0) > 0:
        fill_result = fill_idle_slurm_nodes()
        results["actions"]["fill_idle"] = fill_result

    # 3. Check Tailscale mesh
    tailscale_status = check_tailscale_status()
    results["checks"]["tailscale"] = tailscale_status

    # 4. Check Vast.ai instances
    vast_status = check_vast_instances()
    results["checks"]["vast"] = vast_status

    # 5. Run Vast autoscaler if instances are low
    if vast_status.get("healthy") and vast_status.get("total", 0) > 0:
        autoscaler_result = run_vast_autoscaler()
        results["actions"]["vast_autoscaler"] = autoscaler_result

    # 6. Check Lambda instances
    lambda_status = check_lambda_instances()
    results["checks"]["lambda"] = lambda_status

    # 7. Check key cluster nodes via SSH
    ssh_status = check_node_health_ssh()
    results["checks"]["ssh_nodes"] = ssh_status

    cycle_end = datetime.datetime.now()
    results["cycle_end"] = cycle_end.isoformat()
    results["cycle_duration_seconds"] = (cycle_end - cycle_start).total_seconds()

    log(f"Monitoring cycle complete in {results['cycle_duration_seconds']:.1f}s")
    log("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Cluster Monitor Daemon")
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=10,
        help="How long to run the monitor (hours)",
    )
    parser.add_argument(
        "--single-cycle",
        action="store_true",
        help="Run a single monitoring cycle and exit",
    )
    args = parser.parse_args()

    log(f"Cluster Monitor Daemon starting")
    log(f"Duration: {args.duration_hours} hours")
    log(f"Check interval: {MIN_INTERVAL_SECONDS}-{MAX_INTERVAL_SECONDS} seconds")
    log(f"Log file: {LOG_FILE}")
    log(f"State file: {STATE_FILE}")

    if args.single_cycle:
        results = run_monitoring_cycle()
        save_state({"last_cycle": results, "cycles_completed": 1})
        return

    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(hours=args.duration_hours)
    cycles_completed = 0

    state = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "cycles_completed": 0,
        "last_cycle": None,
    }

    log(f"Will run until {end_time.isoformat()}")

    while datetime.datetime.now() < end_time:
        try:
            # Run monitoring cycle
            results = run_monitoring_cycle()
            cycles_completed += 1

            # Update state
            state["cycles_completed"] = cycles_completed
            state["last_cycle"] = results
            save_state(state)

            # Check if we should continue
            if datetime.datetime.now() >= end_time:
                break

            # Sleep for random interval (3-5 minutes)
            sleep_seconds = random.randint(MIN_INTERVAL_SECONDS, MAX_INTERVAL_SECONDS)
            next_check = datetime.datetime.now() + datetime.timedelta(seconds=sleep_seconds)
            log(f"Sleeping {sleep_seconds}s until next check at {next_check.strftime('%H:%M:%S')}")
            time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            log("Received interrupt, shutting down gracefully", "WARN")
            break
        except Exception as e:
            log(f"Error in monitoring cycle: {e}", "ERROR")
            # Sleep a bit before retrying
            time.sleep(60)

    log(f"Cluster Monitor Daemon finished after {cycles_completed} cycles")
    state["finished"] = datetime.datetime.now().isoformat()
    state["cycles_completed"] = cycles_completed
    save_state(state)


if __name__ == "__main__":
    main()
