#!/usr/bin/env python3
"""
10-hour cluster health monitoring daemon.

Checks cluster health every 3-5 minutes and takes corrective actions.
"""

import json
import logging
import random
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
LOG_FILE = Path(__file__).parent.parent / "data" / "cluster_monitor.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CHECK_INTERVAL_MIN = 180  # 3 minutes
CHECK_INTERVAL_MAX = 300  # 5 minutes
TOTAL_DURATION_HOURS = 10

# Known nodes for SSH health checks
VAST_NODES = [
    ("vast-a40", None),
    ("vast-5090", None),
    ("vast-5080-new", None),
    ("vast-5070-new", None),
    ("vast-512-norway", None),
    ("vast-384-poland", None),
    ("vast-384-taiwan", None),
]

SSH_NODES_BY_PORT = [
    ("ssh2.vast.ai", 16314, "vast-512-norway"),
    ("ssh6.vast.ai", 16316, "vast-384-poland"),
    ("ssh7.vast.ai", 16316, "vast-384-taiwan"),
]


def run_cmd(cmd: str, timeout: int = 60) -> tuple[bool, str]:
    """Run a command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def check_vast_instances() -> dict:
    """Check Vast.ai instance status."""
    logger.info("Checking Vast.ai instances...")
    success, output = run_cmd("vastai show instances --raw", timeout=30)

    if not success:
        logger.warning(f"Failed to get Vast instances: {output}")
        return {"error": output, "instances": []}

    try:
        instances = json.loads(output)
        running = [i for i in instances if i.get("actual_status") == "running"]
        total_cost = sum(i.get("dph_total", 0) for i in instances)

        logger.info(f"Vast: {len(running)}/{len(instances)} running, ${total_cost:.2f}/hr")
        return {
            "total": len(instances),
            "running": len(running),
            "hourly_cost": total_cost,
            "instances": instances
        }
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse Vast output: {output[:200]}")
        return {"error": "parse_error", "instances": []}


def check_tailscale_mesh() -> dict:
    """Check Tailscale mesh status."""
    logger.info("Checking Tailscale mesh...")
    success, output = run_cmd("tailscale status", timeout=30)

    if not success:
        logger.warning(f"Failed to get Tailscale status: {output}")
        return {"error": output}

    lines = output.strip().split("\n")
    online = len([l for l in lines if l and "offline" not in l.lower() and not l.startswith("#")])
    offline = len([l for l in lines if "offline" in l.lower()])

    logger.info(f"Tailscale: {online} online, {offline} offline")
    return {"online": online, "offline": offline}


def check_node_jobs(host: str, port: int | None = None) -> dict:
    """Check jobs running on a specific node."""
    if port:
        ssh_cmd = f"ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p {port} root@{host}"
    else:
        ssh_cmd = f"ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no root@{host}"

    # Check Python processes
    success, output = run_cmd(f"{ssh_cmd} 'ps aux | grep python | grep -v grep | wc -l'", timeout=30)
    python_procs = int(output) if success and output.isdigit() else 0

    # Check GPU utilization if available
    success, gpu_output = run_cmd(f"{ssh_cmd} 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1'", timeout=30)
    gpu_util = int(gpu_output) if success and gpu_output.isdigit() else -1

    return {"python_procs": python_procs, "gpu_util": gpu_util}


def check_node_health(node_name: str, host: str | None = None, port: int | None = None) -> dict:
    """Check health of a specific node."""
    if host is None:
        host = node_name

    if port:
        ssh_cmd = f"ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p {port} root@{host}"
    else:
        ssh_cmd = f"ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no root@{host}"

    # Quick connectivity check
    success, _ = run_cmd(f"{ssh_cmd} 'echo ok'", timeout=20)

    if not success:
        return {"name": node_name, "reachable": False}

    # Check orchestrator
    success, orch_output = run_cmd(f"{ssh_cmd} 'ps aux | grep p2p_orchestrator | grep -v grep | wc -l'", timeout=20)
    has_orchestrator = success and orch_output.strip() == "1"

    # Check selfplay jobs
    success, jobs_output = run_cmd(f"{ssh_cmd} 'ps aux | grep -E \"selfplay|run_self\" | grep -v grep | wc -l'", timeout=20)
    selfplay_jobs = int(jobs_output) if success and jobs_output.isdigit() else 0

    return {
        "name": node_name,
        "reachable": True,
        "has_orchestrator": has_orchestrator,
        "selfplay_jobs": selfplay_jobs
    }


def restart_orchestrator(host: str, port: int, node_id: str) -> bool:
    """Restart orchestrator on a node."""
    logger.info(f"Restarting orchestrator on {node_id}...")

    if port:
        ssh_cmd = f"ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p {port} root@{host}"
    else:
        ssh_cmd = f"ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no root@{host}"

    # Kill existing and start new (try both RingRift and ringrift paths)
    cmd = f"{ssh_cmd} 'pkill -f p2p_orchestrator 2>/dev/null; cd ~/RingRift/ai-service 2>/dev/null || cd ~/ringrift/ai-service && screen -dmS p2p bash -c \"export PYTHONPATH=\\$PWD && python scripts/p2p_orchestrator.py --node-id {node_id} --port 8770 2>&1 | tee /tmp/p2p.log\"'"

    success, output = run_cmd(cmd, timeout=60)
    if success:
        logger.info(f"Orchestrator restarted on {node_id}")
    else:
        logger.warning(f"Failed to restart orchestrator on {node_id}: {output}")
    return success


def check_training_jobs() -> dict:
    """Check status of training jobs on A40."""
    logger.info("Checking training jobs on A40...")

    ssh_cmd = "ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no root@vast-a40"

    # Check screens
    success, screens = run_cmd(f"{ssh_cmd} 'screen -ls'", timeout=20)

    # Check training log
    success, train_log = run_cmd(f"{ssh_cmd} 'tail -5 /tmp/train_v2.log 2>/dev/null'", timeout=20)

    # Check square19 log
    success, sq19_log = run_cmd(f"{ssh_cmd} 'tail -3 /tmp/square19.log 2>/dev/null'", timeout=20)

    return {
        "screens": screens if success else "unknown",
        "train_log": train_log[:500] if train_log else "no log",
        "square19_log": sq19_log[:300] if sq19_log else "no log"
    }


def get_cluster_summary() -> dict:
    """Get overall cluster summary."""
    vast = check_vast_instances()
    tailscale = check_tailscale_mesh()

    # Sample a few nodes for detailed health
    node_health = []
    for host, port, name in SSH_NODES_BY_PORT[:2]:  # Check first 2 new nodes
        health = check_node_health(name, host, port)
        node_health.append(health)

    # Check A40 specifically
    a40_health = check_node_health("vast-a40")
    node_health.append(a40_health)

    training = check_training_jobs()

    return {
        "timestamp": datetime.now().isoformat(),
        "vast": vast,
        "tailscale": tailscale,
        "node_samples": node_health,
        "training": training
    }


def assess_and_fix(summary: dict) -> list[str]:
    """Assess cluster health and take corrective actions."""
    actions = []

    # Check if nodes need orchestrator restart
    for node in summary.get("node_samples", []):
        if node.get("reachable") and not node.get("has_orchestrator"):
            # Find connection info
            for host, port, name in SSH_NODES_BY_PORT:
                if name == node["name"]:
                    if restart_orchestrator(host, port, name.replace("-", "_")):
                        actions.append(f"Restarted orchestrator on {name}")
                    break

    # Check vast instances
    vast = summary.get("vast", {})
    if vast.get("running", 0) < vast.get("total", 0):
        stopped = vast.get("total", 0) - vast.get("running", 0)
        logger.warning(f"{stopped} Vast instances not running")
        actions.append(f"Warning: {stopped} Vast instances not running")

    # Check tailscale
    tailscale = summary.get("tailscale", {})
    if tailscale.get("offline", 0) > 10:
        logger.warning(f"Many nodes offline in Tailscale mesh: {tailscale.get('offline')}")
        actions.append(f"Warning: {tailscale.get('offline')} nodes offline in mesh")

    return actions


def monitoring_loop():
    """Main monitoring loop."""
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=TOTAL_DURATION_HOURS)
    iteration = 0

    logger.info(f"Starting {TOTAL_DURATION_HOURS}-hour monitoring loop")
    logger.info(f"Will run until {end_time.isoformat()}")

    while datetime.now() < end_time:
        iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration} at {datetime.now().isoformat()}")
        logger.info(f"{'='*60}")

        try:
            # Get cluster summary
            summary = get_cluster_summary()

            # Log summary
            vast = summary.get("vast", {})
            logger.info(f"Vast: {vast.get('running', '?')}/{vast.get('total', '?')} instances, ${vast.get('hourly_cost', 0):.2f}/hr")

            tailscale = summary.get("tailscale", {})
            logger.info(f"Tailscale: {tailscale.get('online', '?')} online, {tailscale.get('offline', '?')} offline")

            for node in summary.get("node_samples", []):
                status = "OK" if node.get("reachable") and node.get("has_orchestrator") else "NEEDS ATTENTION"
                logger.info(f"Node {node.get('name')}: {status}, jobs={node.get('selfplay_jobs', 0)}")

            # Training status
            training = summary.get("training", {})
            if "Epoch" in str(training.get("train_log", "")):
                logger.info("Training: Active")
            else:
                logger.info(f"Training log: {training.get('train_log', 'unknown')[:100]}")

            # Assess and fix
            actions = assess_and_fix(summary)
            if actions:
                logger.info(f"Actions taken: {actions}")
            else:
                logger.info("No corrective actions needed")

        except Exception as e:
            logger.error(f"Error in monitoring iteration: {e}")

        # Calculate time remaining
        remaining = end_time - datetime.now()
        remaining_hours = remaining.total_seconds() / 3600
        logger.info(f"Time remaining: {remaining_hours:.1f} hours")

        # Sleep with random jitter
        sleep_time = random.randint(CHECK_INTERVAL_MIN, CHECK_INTERVAL_MAX)
        logger.info(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)

    logger.info(f"\n{'='*60}")
    logger.info(f"Monitoring complete after {TOTAL_DURATION_HOURS} hours")
    logger.info(f"Total iterations: {iteration}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    monitoring_loop()
