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

# Thresholds for waste detection
GPU_IDLE_THRESHOLD_PERCENT = 20  # GPU usage below this is considered idle
JOB_STALL_THRESHOLD_MINUTES = 30  # Job with no progress for this long is stalled
MAX_CONSECUTIVE_FAILURES = 3  # Max failures before escalating

# Priority job configurations (higher priority = more compute value)
PRIORITY_CONFIGS = [
    # (board_type, players, priority, description)
    # Boosted hex priorities to scale up hexagonal selfplay (Dec 2025)
    ("hexagonal", "2", 10, "hex 2p - priority data collection"),
    ("hexagonal", "3", 8, "hex 3p - priority data collection"),
    ("hexagonal", "4", 6, "hex 4p - needed for model training"),
    ("square19", "2", 3, "sq19 2p - needed for model training"),
    ("square19", "3", 2, "sq19 3p - data expansion"),
    ("square8", "2", 1, "sq8 2p - baseline data"),
    ("square8", "3", 1, "sq8 3p - low priority"),
]


def log(message: str, level: str = "INFO"):
    """Log message to file and stdout."""
    timestamp = datetime.datetime.now().isoformat()
    log_line = f"[{timestamp}] [{level}] {message}"
    print(log_line)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")


def run_command(cmd: list[str], timeout: int = 120, env_extra: dict | None = None) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        env = os.environ.copy()
        # Set PYTHONPATH to ai-service directory
        ai_service_dir = PROJECT_ROOT / "ai-service"
        env["PYTHONPATH"] = str(ai_service_dir)
        if env_extra:
            env.update(env_extra)

        # Fix python command to use python3 or specific version
        if cmd and cmd[0] == "python":
            # Use python3.11 on mac-studio
            cmd = ["/opt/homebrew/bin/python3.11"] + cmd[1:]

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


def sync_job_states() -> dict:
    """Sync unified job states before status checks."""
    rc, stdout, stderr = run_command([
        "python", "scripts/cluster_submit.py", "sync-jobs", "--once"
    ], timeout=60)

    if rc != 0:
        log(f"Job state sync failed: {stderr}", "WARN")
        return {"success": False, "error": stderr}

    log(f"Job state sync: {stdout.strip()[:200]}")
    return {"success": True, "output": stdout.strip()[:200]}


# Key cluster nodes to check via SSH - loaded from config/cluster.yaml
def load_cluster_nodes_from_yaml() -> list[dict]:
    """Load all cluster nodes from config/cluster.yaml."""
    import yaml

    config_path = PROJECT_ROOT / "ai-service" / "config" / "cluster.yaml"
    if not config_path.exists():
        log(f"Cluster config not found at {config_path}, using fallback", "WARN")
        return [
            {"name": "lambda-gh200-e", "host": "ubuntu@lambda-gh200-e", "check": "uptime"},
        ]

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        nodes = []
        for node_name, node_config in config.get("nodes", {}).items():
            if node_config.get("status") != "active":
                continue

            host = node_config.get("host", node_name)
            ssh_user = node_config.get("ssh_user", "ubuntu")
            ssh_key = node_config.get("ssh_key")
            tailscale_ip = node_config.get("tailscale_ip")
            gpu_type = node_config.get("gpu_type", "unknown")

            # Determine check command based on GPU type
            if gpu_type == "none":
                check_cmd = "uptime"
            else:
                check_cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null || uptime"

            # Build node entry - prefer tailscale IP if available
            node_entry = {
                "name": node_name,
                "host": f"{ssh_user}@{tailscale_ip or host}",
                "check": check_cmd,
                "gpu_type": gpu_type,
            }

            if ssh_key:
                node_entry["ssh_key"] = ssh_key

            nodes.append(node_entry)

        log(f"Loaded {len(nodes)} active nodes from cluster.yaml")
        return nodes
    except Exception as e:
        log(f"Failed to load cluster.yaml: {e}", "ERROR")
        return [
            {"name": "lambda-gh200-e", "host": "ubuntu@lambda-gh200-e", "check": "uptime"},
        ]

# Load nodes at startup
CLUSTER_NODES = load_cluster_nodes_from_yaml()


def check_node_health_ssh() -> dict:
    """Check node health via SSH for all cluster nodes."""
    log(f"Checking cluster node health via SSH ({len(CLUSTER_NODES)} nodes)...")

    results = {}
    healthy_count = 0

    for node in CLUSTER_NODES:
        name = node["name"]
        host = node["host"]
        check_cmd = node.get("check", "uptime")
        port = node.get("port")
        ssh_key = node.get("ssh_key")

        ssh_cmd = ["ssh", "-T", "-n", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=no"]
        if ssh_key:
            ssh_cmd.extend(["-i", os.path.expanduser(f"~/.ssh/{ssh_key}")])
        if port:
            ssh_cmd.extend(["-p", port])
        ssh_cmd.extend([host, check_cmd])

        rc, stdout, stderr = run_command(ssh_cmd, timeout=90)

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
        "python", "scripts/cluster_submit.py", "status", "--json"
    ])

    if rc != 0:
        log(f"Slurm status check failed: {stderr}", "ERROR")
        return {"healthy": False, "error": stderr}

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        log("Failed to parse Slurm status JSON output", "WARN")
        return {"healthy": False, "error": "JSON parse error"}

    slurm = data.get("slurm", {})
    jobs = data.get("jobs", {})
    idle_count = int(slurm.get("idle_nodes", 0) or 0)
    running_jobs = int(slurm.get("jobs_running", 0) or 0)
    pending_jobs = int(slurm.get("jobs_pending", 0) or 0)
    total_nodes = int(slurm.get("nodes", 0) or 0)

    log(
        "Slurm: "
        f"{running_jobs} running, {pending_jobs} pending, "
        f"{idle_count} idle, {total_nodes} nodes "
        f"(db_running={jobs.get('running', 0)}, db_pending={jobs.get('pending', 0)})"
    )

    return {
        "healthy": True,
        "idle": idle_count,
        "running_jobs": running_jobs,
        "pending_jobs": pending_jobs,
        "total": total_nodes,
        "db_running": jobs.get("running", 0),
        "db_pending": jobs.get("pending", 0),
        "raw_output": stdout[:1000]  # Truncate for state file
    }


def check_gpu_utilization(node_host: str, port: str | None = None) -> dict:
    """Check GPU utilization on a remote node via SSH."""
    ssh_cmd = ["ssh", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=no"]
    if port:
        ssh_cmd.extend(["-p", port])

    # Get GPU utilization with nvidia-smi
    nvidia_cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
    ssh_cmd.extend([node_host, nvidia_cmd])

    rc, stdout, stderr = run_command(ssh_cmd, timeout=30)

    if rc != 0:
        return {"healthy": False, "error": stderr[:100]}

    gpus = []
    for line in stdout.strip().split("\n"):
        if line:
            parts = line.split(",")
            if len(parts) >= 3:
                gpus.append({
                    "util_percent": int(parts[0].strip()),
                    "mem_used_mb": int(parts[1].strip()),
                    "mem_total_mb": int(parts[2].strip()),
                })

    avg_util = sum(g["util_percent"] for g in gpus) / len(gpus) if gpus else 0
    idle_gpus = sum(1 for g in gpus if g["util_percent"] < GPU_IDLE_THRESHOLD_PERCENT)

    return {
        "healthy": True,
        "gpus": gpus,
        "avg_utilization": avg_util,
        "idle_gpus": idle_gpus,
        "total_gpus": len(gpus),
    }


def check_stalled_jobs() -> dict:
    """Check for jobs that appear to be stalled (no recent output)."""
    log("Checking for stalled jobs...")

    # Check job state files for recent updates
    rc, stdout, stderr = run_command([
        "python", "scripts/cluster_submit.py", "status", "--json"
    ], timeout=60)

    if rc != 0:
        return {"healthy": False, "error": stderr}

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {"healthy": False, "error": "JSON parse error"}

    stalled_jobs = []
    running_jobs = data.get("running_jobs", [])

    for job in running_jobs:
        # Check if job has a last_update timestamp
        last_update = job.get("last_update")
        if last_update:
            try:
                update_time = datetime.datetime.fromisoformat(last_update)
                age_minutes = (datetime.datetime.now() - update_time).total_seconds() / 60
                if age_minutes > JOB_STALL_THRESHOLD_MINUTES:
                    stalled_jobs.append({
                        "job_id": job.get("job_id"),
                        "node": job.get("node"),
                        "minutes_since_update": int(age_minutes),
                    })
            except (ValueError, TypeError):
                pass

    if stalled_jobs:
        log(f"Found {len(stalled_jobs)} potentially stalled jobs", "WARN")
        for job in stalled_jobs:
            log(f"  Stalled: {job['job_id']} on {job['node']} ({job['minutes_since_update']}min)", "WARN")

    return {"stalled_jobs": stalled_jobs, "count": len(stalled_jobs)}


def cancel_stalled_job(job_id: str) -> dict:
    """Cancel a stalled job and optionally restart it."""
    log(f"Canceling stalled job {job_id}...")

    rc, stdout, stderr = run_command([
        "python", "scripts/cluster_submit.py", "cancel", job_id
    ], timeout=60)

    if rc != 0:
        log(f"Failed to cancel job {job_id}: {stderr}", "ERROR")
        return {"success": False, "error": stderr}

    log(f"Canceled stalled job {job_id}")
    return {"success": True, "job_id": job_id}


def fill_idle_slurm_nodes() -> dict:
    """Fill idle Slurm nodes with selfplay jobs."""
    log("Filling idle Slurm nodes with selfplay jobs...")

    # Use priority-weighted selection (higher priority = more likely)
    weights = [cfg[2] for cfg in PRIORITY_CONFIGS]
    total_weight = sum(weights)
    rand_val = random.random() * total_weight

    cumulative = 0
    selected = PRIORITY_CONFIGS[0]
    for cfg in PRIORITY_CONFIGS:
        cumulative += cfg[2]
        if rand_val <= cumulative:
            selected = cfg
            break

    board_type, player_count = selected[0], selected[1]
    log(f"Selected config: {selected[3]}")

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
    rc, stdout, _stderr = run_command([
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


def log_cycle_summary(results: dict):
    """Log a summary of the monitoring cycle."""
    checks = results.get("checks", {})
    actions = results.get("actions", {})

    summary_parts = []

    # Slurm summary
    slurm = checks.get("slurm", {})
    if slurm.get("healthy"):
        summary_parts.append(f"Slurm: {slurm.get('running_jobs', 0)}R/{slurm.get('idle', 0)}I")

    # SSH health summary
    ssh = checks.get("ssh_nodes", {})
    if ssh.get("healthy_count") is not None:
        summary_parts.append(f"SSH: {ssh['healthy_count']}/{ssh['total']}")

    # Actions taken
    action_count = len(actions)
    if action_count > 0:
        summary_parts.append(f"Actions: {action_count}")

    # Stalled jobs
    stalled = checks.get("stalled_jobs", {})
    if stalled.get("count", 0) > 0:
        summary_parts.append(f"⚠️ {stalled['count']} stalled")

    log("Summary: " + " | ".join(summary_parts))


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

    # 0. Sync job states for accurate status
    results["actions"]["sync_jobs"] = sync_job_states()

    # 1. Check Slurm status
    slurm_status = check_slurm_status()
    results["checks"]["slurm"] = slurm_status

    # 2. If idle nodes found, fill them
    if slurm_status.get("healthy") and slurm_status.get("idle", 0) > 0:
        fill_result = fill_idle_slurm_nodes()
        results["actions"]["fill_idle"] = fill_result

    # 2b. Check for stalled jobs
    stalled_result = check_stalled_jobs()
    results["checks"]["stalled_jobs"] = stalled_result

    # 2c. Cancel stalled jobs if found (more than 45 min without update)
    if stalled_result.get("stalled_jobs"):
        for stalled in stalled_result["stalled_jobs"]:
            if stalled.get("minutes_since_update", 0) > 45:
                cancel_result = cancel_stalled_job(stalled["job_id"])
                results["actions"].setdefault("canceled_jobs", []).append(cancel_result)

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

    # 8. Check GPU utilization on key nodes
    gpu_checks = {}
    for node in CLUSTER_NODES:
        if node.get("check") == "nvidia-smi":
            util = check_gpu_utilization(node["host"], node.get("port"))
            gpu_checks[node["name"]] = util
            if util.get("healthy") and util.get("idle_gpus", 0) > 0:
                log(f"  {node['name']}: {util['idle_gpus']}/{util['total_gpus']} GPUs idle", "WARN")
    results["checks"]["gpu_utilization"] = gpu_checks

    # 9. Log cycle summary
    log_cycle_summary(results)

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

    log("Cluster Monitor Daemon starting")
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
