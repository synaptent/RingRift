#!/usr/bin/env python3
"""
Robust cluster health monitor with automatic recovery.

Key improvements over previous version:
1. Uses correct SSH ports for each node (not Tailscale hostnames for new nodes)
2. Verifies data is actually being saved (game counts increasing)
3. Automatically restarts dead jobs with proper error handling
4. Tracks game counts over time to detect stalled jobs
5. Handles connection failures gracefully
"""

import json
import logging
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Setup logging
LOG_FILE = Path(__file__).parent.parent / "data" / "robust_monitor.log"
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
STALL_THRESHOLD_CHECKS = 3  # If no progress in 3 checks, consider stalled

# Node configurations with CORRECT SSH access
@dataclass
class NodeConfig:
    name: str
    ssh_cmd: str  # Full SSH command prefix
    node_id: str  # Node ID for orchestrator
    is_gpu: bool = False
    repo_path: str = "~/RingRift/ai-service"


NODES = [
    # New CPU nodes - use direct SSH ports
    NodeConfig(
        name="norway-512",
        ssh_cmd="ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 16314 root@ssh2.vast.ai",
        node_id="vast_512_norway",
        is_gpu=False
    ),
    NodeConfig(
        name="poland-384",
        ssh_cmd="ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 16316 root@ssh6.vast.ai",
        node_id="vast_384_poland",
        is_gpu=False
    ),
    NodeConfig(
        name="taiwan-384",
        ssh_cmd="ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 16316 root@ssh7.vast.ai",
        node_id="vast_384_taiwan",
        is_gpu=False
    ),
    # GPU nodes - use Tailscale hostnames
    NodeConfig(
        name="a40",
        ssh_cmd="ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no root@vast-a40",
        node_id="vast_a40",
        is_gpu=True,
        repo_path="~/ringrift/ai-service"  # lowercase on A40
    ),
]


@dataclass
class NodeState:
    """Track state for a node across checks."""
    last_game_count: int = 0
    stall_count: int = 0
    last_check_time: datetime | None = None
    is_healthy: bool = False


# Global state tracking
node_states: dict[str, NodeState] = {}


def run_ssh(node: NodeConfig, cmd: str, timeout: int = 60) -> tuple[bool, str]:
    """Run a command on a node via SSH."""
    full_cmd = f"{node.ssh_cmd} '{cmd}'"
    try:
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout.strip() or result.stderr.strip()
        # Vast.ai sometimes returns 255 even on success
        success = result.returncode == 0 or ("Welcome to vast.ai" in output and len(output) > 100)
        return success, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def check_orchestrator_running(node: NodeConfig) -> bool:
    """Check if orchestrator is running on a node."""
    success, output = run_ssh(node, "ps aux | grep p2p_orchestrator | grep -v grep | wc -l", timeout=20)
    if not success:
        logger.warning(f"[{node.name}] Failed to check orchestrator: {output[:100]}")
        return False
    try:
        count = int(output.split('\n')[-1].strip())
        return count > 0
    except (ValueError, IndexError):
        return False


def start_orchestrator(node: NodeConfig) -> bool:
    """Start orchestrator on a node."""
    logger.info(f"[{node.name}] Starting orchestrator...")

    # Kill any existing, then start fresh
    cmd = f'''cd {node.repo_path} && \
        pkill -f p2p_orchestrator 2>/dev/null; \
        screen -dmS p2p bash -c "export PYTHONPATH=$PWD && python scripts/p2p_orchestrator.py --node-id {node.node_id} --port 8770 2>&1 | tee /tmp/p2p.log"'''

    success, output = run_ssh(node, cmd, timeout=45)
    if success:
        # Verify it started
        time.sleep(3)
        if check_orchestrator_running(node):
            logger.info(f"[{node.name}] Orchestrator started successfully")
            return True
        else:
            logger.warning(f"[{node.name}] Orchestrator failed to start")
            return False
    else:
        logger.warning(f"[{node.name}] Failed to start orchestrator: {output[:200]}")
        return False


def get_total_game_count(node: NodeConfig) -> int:
    """Get total game count across all DB files on a node."""
    cmd = '''find ~/RingRift/ai-service/data/selfplay -name "games.db" -exec sqlite3 {} "SELECT COUNT(*) FROM games" \\; 2>/dev/null | awk '{sum+=$1} END {print sum+0}' '''
    success, output = run_ssh(node, cmd, timeout=45)
    if not success:
        logger.warning(f"[{node.name}] Failed to get game count: {output[:100]}")
        return -1
    try:
        # Get the last number in output
        lines = output.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line.isdigit():
                return int(line)
        return 0
    except (ValueError, IndexError):
        return 0


def get_active_selfplay_jobs(node: NodeConfig) -> int:
    """Get count of active selfplay jobs."""
    cmd = 'ps aux | grep -E "run_hybrid_selfplay|run_self" | grep -v grep | wc -l'
    success, output = run_ssh(node, cmd, timeout=20)
    if not success:
        return -1
    try:
        return int(output.split('\n')[-1].strip())
    except (ValueError, IndexError):
        return 0


def check_node_health(node: NodeConfig) -> dict:
    """Comprehensive health check for a node."""
    state = node_states.get(node.name, NodeState())

    result = {
        "name": node.name,
        "reachable": False,
        "has_orchestrator": False,
        "game_count": 0,
        "selfplay_jobs": 0,
        "is_stalled": False,
        "action_taken": None
    }

    # Check connectivity
    success, _ = run_ssh(node, "echo ok", timeout=15)
    if not success:
        logger.warning(f"[{node.name}] Node unreachable")
        return result

    result["reachable"] = True

    # Check orchestrator
    result["has_orchestrator"] = check_orchestrator_running(node)

    # Get game count
    result["game_count"] = get_total_game_count(node)

    # Get active jobs
    result["selfplay_jobs"] = get_active_selfplay_jobs(node)

    # Check for stall (game count not increasing)
    if result["game_count"] > 0 and result["game_count"] == state.last_game_count:
        state.stall_count += 1
        if state.stall_count >= STALL_THRESHOLD_CHECKS:
            result["is_stalled"] = True
            logger.warning(f"[{node.name}] STALLED: Game count stuck at {result['game_count']} for {state.stall_count} checks")
    else:
        state.stall_count = 0

    state.last_game_count = result["game_count"]
    state.last_check_time = datetime.now()
    node_states[node.name] = state

    return result


def take_corrective_action(node: NodeConfig, health: dict) -> str:
    """Take corrective action based on health check results."""
    actions = []

    # Start orchestrator if not running
    if health["reachable"] and not health["has_orchestrator"]:
        if start_orchestrator(node):
            actions.append("started_orchestrator")
        else:
            actions.append("failed_to_start_orchestrator")

    # Restart orchestrator if stalled
    if health["is_stalled"] and health["has_orchestrator"]:
        logger.info(f"[{node.name}] Restarting stalled orchestrator...")
        if start_orchestrator(node):
            actions.append("restarted_stalled_orchestrator")
            # Reset stall counter
            if node.name in node_states:
                node_states[node.name].stall_count = 0

    return ", ".join(actions) if actions else None


def check_vast_instances() -> dict:
    """Check Vast.ai instance status."""
    try:
        result = subprocess.run(
            "vastai show instances --raw",
            shell=True, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return {"error": result.stderr, "running": 0, "total": 0}

        instances = json.loads(result.stdout)
        running = len([i for i in instances if i.get("actual_status") == "running"])
        total_cost = sum(i.get("dph_total", 0) for i in instances)

        return {
            "total": len(instances),
            "running": running,
            "hourly_cost": total_cost
        }
    except Exception as e:
        return {"error": str(e), "running": 0, "total": 0}


def monitoring_iteration(iteration: int) -> dict:
    """Run a single monitoring iteration."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Iteration {iteration} at {datetime.now().isoformat()}")
    logger.info(f"{'='*60}")

    summary = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "nodes": [],
        "actions_taken": [],
        "total_games": 0
    }

    # Check Vast instances
    vast = check_vast_instances()
    logger.info(f"Vast: {vast.get('running', 0)}/{vast.get('total', 0)} instances, ${vast.get('hourly_cost', 0):.2f}/hr")
    summary["vast"] = vast

    # Check each node
    for node in NODES:
        health = check_node_health(node)
        summary["nodes"].append(health)
        summary["total_games"] += max(0, health.get("game_count", 0))

        # Log status
        status_parts = []
        if health["reachable"]:
            status_parts.append("reachable")
        else:
            status_parts.append("UNREACHABLE")

        if health["has_orchestrator"]:
            status_parts.append("orchestrator OK")
        else:
            status_parts.append("NO ORCHESTRATOR")

        status_parts.append(f"games={health['game_count']}")
        status_parts.append(f"jobs={health['selfplay_jobs']}")

        if health["is_stalled"]:
            status_parts.append("STALLED")

        logger.info(f"[{node.name}] {', '.join(status_parts)}")

        # Take action if needed
        action = take_corrective_action(node, health)
        if action:
            health["action_taken"] = action
            summary["actions_taken"].append(f"{node.name}: {action}")
            logger.info(f"[{node.name}] Action: {action}")

    # Summary
    logger.info(f"Total games across cluster: {summary['total_games']}")
    if summary["actions_taken"]:
        logger.info(f"Actions taken: {summary['actions_taken']}")
    else:
        logger.info("No corrective actions needed")

    return summary


def monitoring_loop():
    """Main monitoring loop."""
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=TOTAL_DURATION_HOURS)
    iteration = 0

    logger.info(f"Starting {TOTAL_DURATION_HOURS}-hour robust monitoring")
    logger.info(f"Will run until {end_time.isoformat()}")
    logger.info(f"Checking nodes: {[n.name for n in NODES]}")

    while datetime.now() < end_time:
        iteration += 1

        try:
            monitoring_iteration(iteration)
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
    logger.info(f"Final game counts: {json.dumps({n.name: node_states.get(n.name, NodeState()).last_game_count for n in NODES})}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    monitoring_loop()
