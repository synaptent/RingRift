#!/usr/bin/env python3
"""Resource-Aware Work Router - Smart work assignment with CPU-first fallback.

Key improvements over smart_work_router.py:
1. Loads comprehensive node registry from cluster_nodes.yaml
2. Tries to wake up CPU-rich nodes BEFORE falling back to GPU nodes for CPU work
3. Uses GPU-accelerated CMA-ES (EvoTorch) on GPU nodes when optimization is needed
4. Better tracking of what work is running where

Work Routing Rules:
- GPU-heavy work (training, tournaments, gauntlets) -> GPU nodes only
- CPU-heavy work (CPU CMA-ES, data processing) -> CPU nodes FIRST, then GPU fallback
- GPU CMA-ES (optimization) -> GPU nodes with EvoTorch for 10-100x speedup

Usage:
    # Check status and show routing recommendations
    python scripts/resource_aware_router.py --status

    # Rebalance with CPU-first logic
    python scripts/resource_aware_router.py --rebalance

    # Run as daemon
    python scripts/resource_aware_router.py --daemon --interval 120
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from scripts.lib.paths import AI_SERVICE_ROOT, CONFIG_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

P2P_PORT = int(os.environ.get("RINGRIFT_P2P_PORT", "8770"))

# =============================================================================
# Work Type Classification
# =============================================================================

class WorkType(Enum):
    """Types of work that can be assigned to nodes."""
    # GPU-heavy work
    TRAINING = "training"
    GPU_CMAES = "gpu_cmaes"       # EvoTorch GPU CMA-ES
    TOURNAMENT = "tournament"
    GAUNTLET = "gauntlet"
    GPU_SELFPLAY = "gpu_selfplay"

    # CPU-heavy work
    CPU_CMAES = "cpu_cmaes"
    HYBRID_SELFPLAY = "hybrid_selfplay"
    DATA_MERGE = "data_merge"
    DATA_EXPORT = "data_export"


GPU_WORK_TYPES = {
    WorkType.TRAINING, WorkType.GPU_CMAES, WorkType.TOURNAMENT,
    WorkType.GAUNTLET, WorkType.GPU_SELFPLAY
}

CPU_WORK_TYPES = {
    WorkType.CPU_CMAES, WorkType.HYBRID_SELFPLAY,
    WorkType.DATA_MERGE, WorkType.DATA_EXPORT
}


# =============================================================================
# Node Classification
# =============================================================================

@dataclass
class NodeConfig:
    """Static node configuration from cluster_nodes.yaml."""
    name: str
    host: str
    user: str = "ubuntu"
    gpu: str = ""
    gpu_memory_gb: int = 0
    cpu_cores: int = 0
    memory_gb: int = 0
    description: str = ""
    priority: int = 5
    wake_method: str = "ssh"
    is_optional: bool = False
    node_type: str = "unknown"  # gpu_heavy, gpu_medium, cpu_rich


@dataclass
class NodeState:
    """Dynamic node state (current utilization)."""
    config: NodeConfig
    is_reachable: bool = False
    gpu_percent: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    # Running work
    training_running: bool = False
    cmaes_running: bool = False
    tournament_running: bool = False
    gauntlet_running: bool = False
    selfplay_jobs: int = 0
    # Derived
    is_gpu_idle: bool = False
    is_cpu_idle: bool = False
    recommended_work: List[WorkType] = field(default_factory=list)


# =============================================================================
# Configuration Loading
# =============================================================================

def load_cluster_config() -> Dict[str, List[NodeConfig]]:
    """Load cluster configuration from YAML."""
    config_path = CONFIG_DIR / "cluster_nodes.yaml"

    if not config_path.exists():
        logger.warning(f"Cluster config not found: {config_path}")
        return {"gpu_heavy": [], "gpu_medium": [], "cpu_rich": []}

    if not YAML_AVAILABLE:
        logger.warning("PyYAML not installed, using fallback config")
        return _get_fallback_config()

    with open(config_path) as f:
        data = yaml.safe_load(f)

    nodes = {"gpu_heavy": [], "gpu_medium": [], "cpu_rich": []}

    for node_type in ["gpu_heavy", "gpu_medium", "cpu_rich"]:
        for node_data in data.get(node_type, []):
            node = NodeConfig(
                name=node_data.get("name", "unknown"),
                host=node_data.get("host", ""),
                user=node_data.get("user", "ubuntu"),
                gpu=node_data.get("gpu", ""),
                gpu_memory_gb=node_data.get("gpu_memory_gb", 0),
                cpu_cores=node_data.get("cpu_cores", 0),
                memory_gb=node_data.get("memory_gb", 0),
                description=node_data.get("description", ""),
                priority=node_data.get("priority", 5),
                wake_method=node_data.get("wake_method", "ssh"),
                is_optional=node_data.get("is_optional", False),
                node_type=node_type,
            )
            nodes[node_type].append(node)

    logger.info(f"Loaded cluster config: {len(nodes['gpu_heavy'])} GPU-heavy, "
                f"{len(nodes['gpu_medium'])} GPU-medium, {len(nodes['cpu_rich'])} CPU-rich")

    return nodes


def _get_fallback_config() -> Dict[str, List[NodeConfig]]:
    """Return empty config when YAML not available.

    Node configuration should only come from config/cluster_nodes.yaml.
    Copy config/cluster_nodes.yaml.example and fill in your node details.
    """
    logger.error("No cluster configuration found!")
    logger.error("Please copy config/cluster_nodes.yaml.example to config/cluster_nodes.yaml")
    logger.error("and fill in your actual node IPs and hostnames.")
    return {"gpu_heavy": [], "gpu_medium": [], "cpu_rich": []}


# =============================================================================
# Node Wake-Up Logic
# =============================================================================

def wake_node_ssh(host: str, user: str, timeout: int = 10) -> bool:
    """Try to wake/reach a node via SSH."""
    try:
        result = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=no",
             f"{user}@{host}", "echo alive"],
            capture_output=True, text=True, timeout=timeout + 5
        )
        return result.returncode == 0 and "alive" in result.stdout
    except Exception as e:
        logger.debug(f"SSH wake failed for {host}: {e}")
        return False


def wake_node_tailscale(host: str, timeout: int = 5) -> bool:
    """Try to reach a node via Tailscale ping."""
    try:
        result = subprocess.run(
            ["tailscale", "ping", f"--timeout={timeout}s", host],
            capture_output=True, text=True, timeout=timeout + 5
        )
        return result.returncode == 0
    except Exception as e:
        logger.debug(f"Tailscale ping failed for {host}: {e}")
        return False


def wake_node_hcloud(name: str) -> bool:
    """Try to start a Hetzner Cloud server."""
    try:
        # Check if server exists and is stopped
        result = subprocess.run(
            ["hcloud", "server", "describe", name, "-o", "json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return False

        data = json.loads(result.stdout)
        status = data.get("status", "")

        if status == "running":
            return True

        if status == "off":
            logger.info(f"Starting Hetzner server {name}...")
            subprocess.run(
                ["hcloud", "server", "poweron", name],
                capture_output=True, timeout=60
            )
            time.sleep(30)  # Wait for boot
            return True

        return False
    except Exception as e:
        logger.debug(f"Hcloud wake failed for {name}: {e}")
        return False


def try_wake_node(config: NodeConfig) -> bool:
    """Try to wake a node using its configured method."""
    logger.info(f"Attempting to wake {config.name} ({config.host}) via {config.wake_method}")

    if config.wake_method == "hcloud":
        if wake_node_hcloud(config.name):
            return wake_node_ssh(config.host, config.user)
        return False

    if config.wake_method == "tailscale":
        if wake_node_tailscale(config.host):
            return wake_node_ssh(config.host, config.user)
        return False

    # Default: SSH
    return wake_node_ssh(config.host, config.user)


def wake_cpu_nodes(cpu_nodes: List[NodeConfig], max_attempts: int = 2) -> List[NodeConfig]:
    """Try to wake all CPU-rich nodes, return those that are reachable."""
    reachable = []

    for node in cpu_nodes:
        if node.is_optional:
            # Optional nodes get fewer attempts
            attempts = 1
        else:
            attempts = max_attempts

        for attempt in range(attempts):
            if try_wake_node(node):
                reachable.append(node)
                break
            if attempt < attempts - 1:
                time.sleep(5)

    logger.info(f"Woke {len(reachable)}/{len(cpu_nodes)} CPU nodes")
    return reachable


# =============================================================================
# Node State Checking
# =============================================================================

def check_node_state(config: NodeConfig) -> NodeState:
    """Check current state of a node."""
    state = NodeState(config=config)

    # Check reachability
    state.is_reachable = wake_node_ssh(config.host, config.user, timeout=5)

    if not state.is_reachable:
        return state

    # Get resource usage and running processes
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"{config.user}@{config.host}",
             """
             # GPU
             gpu_util=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0,0,1")

             # Processes
             training=$(pgrep -c -f 'train_nnue|run_training' 2>/dev/null || echo 0)
             cmaes=$(pgrep -c -f 'cmaes|HeuristicAI.*json|evotorch' 2>/dev/null || echo 0)
             tournament=$(pgrep -c -f 'run_model_elo_tournament' 2>/dev/null || echo 0)
             gauntlet=$(pgrep -c -f 'baseline_gauntlet|two_stage_gauntlet' 2>/dev/null || echo 0)
             selfplay=$(pgrep -c -f 'selfplay|hybrid' 2>/dev/null || echo 0)

             echo "$gpu_util|$training|$cmaes|$tournament|$gauntlet|$selfplay"
             """],
            capture_output=True, text=True, timeout=15
        )

        if result.returncode == 0:
            parts = result.stdout.strip().split("|")
            if len(parts) >= 6:
                gpu_parts = parts[0].split(",")
                if len(gpu_parts) >= 3:
                    state.gpu_percent = float(gpu_parts[0].strip())
                    mem_used = float(gpu_parts[1].strip())
                    mem_total = float(gpu_parts[2].strip())
                    state.gpu_memory_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0

                state.training_running = int(parts[1]) > 0
                state.cmaes_running = int(parts[2]) > 0
                state.tournament_running = int(parts[3]) > 0
                state.gauntlet_running = int(parts[4]) > 0
                state.selfplay_jobs = int(parts[5])

        # Determine if idle
        state.is_gpu_idle = state.gpu_percent < 30 and state.gpu_memory_percent < 80
        state.is_cpu_idle = state.selfplay_jobs < 5 and not state.cmaes_running

    except Exception as e:
        logger.debug(f"Failed to get state for {config.name}: {e}")

    return state


def get_all_node_states(cluster_config: Dict[str, List[NodeConfig]]) -> List[NodeState]:
    """Get state of all nodes in the cluster."""
    states = []

    for node_type, nodes in cluster_config.items():
        for config in nodes:
            state = check_node_state(config)
            states.append(state)

    return states


# =============================================================================
# Work Assignment
# =============================================================================

def start_work_on_node(config: NodeConfig, work_type: WorkType, board: str = "square8", players: int = 2) -> bool:
    """Start a specific type of work on a node."""
    ai_root = "~/ringrift/ai-service"

    if work_type == WorkType.TRAINING:
        cmd = f"cd {ai_root} && nohup python3 scripts/train_nnue.py --board {board} --players {players} >> logs/training_{board}_{players}p.log 2>&1 &"

    elif work_type == WorkType.GPU_CMAES:
        # Use EvoTorch GPU CMA-ES for GPU nodes
        cmd = f"cd {ai_root} && nohup python3 scripts/run_evotorch_cmaes.py --board {board} --num-players {players} --generations 100 --population-size 64 >> logs/evotorch_cmaes_{board}_{players}p.log 2>&1 &"

    elif work_type == WorkType.CPU_CMAES:
        # Use standard CPU CMA-ES
        cmd = f"cd {ai_root} && nohup python3 scripts/run_iterative_cmaes.py --board {board} --players {players} >> logs/cpu_cmaes_{board}_{players}p.log 2>&1 &"

    elif work_type == WorkType.TOURNAMENT:
        cmd = f"cd {ai_root} && nohup python3 scripts/run_model_elo_tournament.py --board {board} --players {players} --games 20 --quick >> logs/tournament_{board}_{players}p.log 2>&1 &"

    elif work_type == WorkType.GAUNTLET:
        cmd = f"cd {ai_root} && nohup python3 scripts/two_stage_gauntlet.py --run --board {board} --players {players} --difficulty 3 --parallel 4 >> logs/gauntlet_{board}_{players}p.log 2>&1 &"

    elif work_type == WorkType.HYBRID_SELFPLAY:
        cmd = f"cd {ai_root} && nohup python3 scripts/run_hybrid_selfplay.py --board-type {board} --num-players {players} --num-games 500 >> logs/selfplay_{board}_{players}p.log 2>&1 &"

    elif work_type == WorkType.DATA_MERGE:
        cmd = f"cd {ai_root} && nohup python3 scripts/merge_game_dbs.py >> logs/data_merge.log 2>&1 &"

    else:
        logger.warning(f"Unknown work type: {work_type}")
        return False

    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
             f"{config.user}@{config.host}", cmd],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            logger.info(f"Started {work_type.value} on {config.name}")
            return True
        else:
            logger.warning(f"Failed to start {work_type.value} on {config.name}: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error starting {work_type.value} on {config.name}: {e}")
        return False


def stop_cpu_work_on_gpu_node(config: NodeConfig) -> bool:
    """Stop CPU-only work on a GPU node to free it for GPU work."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
             f"{config.user}@{config.host}",
             "pkill -f 'run_iterative_cmaes|HeuristicAI.*json' 2>/dev/null; echo done"],
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to stop CPU work on {config.name}: {e}")
        return False


# =============================================================================
# Rebalancing Logic
# =============================================================================

def rebalance_cluster(
    cluster_config: Dict[str, List[NodeConfig]],
    dry_run: bool = False
) -> int:
    """Rebalance work across the cluster with CPU-first logic.

    Strategy:
    1. Identify GPU nodes running CPU-only work (misrouted)
    2. Try to wake up CPU-rich nodes
    3. Move CPU work from GPU nodes to CPU nodes
    4. Start GPU work on freed GPU nodes
    5. If no CPU nodes available, use GPU CMA-ES instead of CPU CMA-ES
    """
    changes = 0

    # Get current state of all nodes
    states = get_all_node_states(cluster_config)

    # Categorize by state
    gpu_heavy_idle = [s for s in states
                      if s.config.node_type == "gpu_heavy"
                      and s.is_reachable
                      and s.is_gpu_idle
                      and not s.training_running]

    gpu_heavy_misrouted = [s for s in states
                           if s.config.node_type == "gpu_heavy"
                           and s.is_reachable
                           and s.is_gpu_idle
                           and s.cmaes_running
                           and not s.training_running]

    cpu_nodes = cluster_config.get("cpu_rich", [])

    logger.info(f"Cluster state: {len(gpu_heavy_idle)} idle GPU nodes, "
                f"{len(gpu_heavy_misrouted)} misrouted (CPU work on GPU), "
                f"{len(cpu_nodes)} CPU nodes configured")

    # Board configurations to cycle through
    configs = [
        ("square8", 2), ("square8", 3), ("hexagonal", 2),
        ("hexagonal", 3), ("square19", 2), ("hex8", 2),
    ]
    config_idx = 0

    # STEP 1: Try to wake CPU nodes BEFORE touching GPU nodes
    if gpu_heavy_misrouted and cpu_nodes:
        logger.info("Attempting to wake CPU-rich nodes to offload CPU work from GPU nodes...")

        awake_cpu = wake_cpu_nodes(cpu_nodes)

        # Start CPU work on awake CPU nodes
        for cpu_config in awake_cpu[:len(gpu_heavy_misrouted)]:
            board, players = configs[config_idx % len(configs)]
            config_idx += 1

            if dry_run:
                logger.info(f"[DRY RUN] Would start CPU CMA-ES on {cpu_config.name}")
                changes += 1
            else:
                if start_work_on_node(cpu_config, WorkType.CPU_CMAES, board, players):
                    changes += 1

        # Stop CPU work on GPU nodes
        for state in gpu_heavy_misrouted[:len(awake_cpu)]:
            if dry_run:
                logger.info(f"[DRY RUN] Would stop CPU CMA-ES on {state.config.name}")
                changes += 1
            else:
                stop_cpu_work_on_gpu_node(state.config)
                changes += 1

    # STEP 2: Start GPU work on idle GPU nodes
    for state in gpu_heavy_idle:
        board, players = configs[config_idx % len(configs)]
        config_idx += 1

        # Priority 1: Training
        if not state.training_running:
            if dry_run:
                logger.info(f"[DRY RUN] Would start training on {state.config.name}")
                changes += 1
            else:
                if start_work_on_node(state.config, WorkType.TRAINING, board, players):
                    changes += 1
                    continue

        # Priority 2: GPU CMA-ES (if still idle)
        if state.is_gpu_idle and not state.cmaes_running:
            if dry_run:
                logger.info(f"[DRY RUN] Would start GPU CMA-ES on {state.config.name}")
                changes += 1
            else:
                if start_work_on_node(state.config, WorkType.GPU_CMAES, board, players):
                    changes += 1
                    continue

        # Priority 3: Tournament or Gauntlet
        if state.is_gpu_idle:
            if dry_run:
                logger.info(f"[DRY RUN] Would start tournament on {state.config.name}")
                changes += 1
            else:
                if start_work_on_node(state.config, WorkType.TOURNAMENT, board, players):
                    changes += 1

    # STEP 3: If GPU nodes still need optimization work but no CPU nodes available,
    # use GPU CMA-ES instead (10-100x faster than CPU anyway)
    remaining_misrouted = [s for s in gpu_heavy_misrouted
                           if s.cmaes_running and s.is_gpu_idle]

    if remaining_misrouted:
        logger.info(f"{len(remaining_misrouted)} GPU nodes still running CPU CMA-ES. "
                    f"Converting to GPU CMA-ES for 10-100x speedup...")

        for state in remaining_misrouted:
            board, players = configs[config_idx % len(configs)]
            config_idx += 1

            if dry_run:
                logger.info(f"[DRY RUN] Would convert {state.config.name} from CPU to GPU CMA-ES")
                changes += 1
            else:
                # Stop CPU CMA-ES
                stop_cpu_work_on_gpu_node(state.config)
                time.sleep(2)
                # Start GPU CMA-ES
                if start_work_on_node(state.config, WorkType.GPU_CMAES, board, players):
                    changes += 1

    return changes


def generate_status_report(cluster_config: Dict[str, List[NodeConfig]]) -> str:
    """Generate comprehensive status report."""
    lines = []
    lines.append("=" * 90)
    lines.append("RESOURCE-AWARE CLUSTER STATUS REPORT")
    lines.append("=" * 90)
    lines.append("")

    states = get_all_node_states(cluster_config)

    for node_type in ["gpu_heavy", "gpu_medium", "cpu_rich"]:
        type_states = [s for s in states if s.config.node_type == node_type]
        if not type_states:
            continue

        lines.append(f"\n{node_type.upper().replace('_', ' ')} NODES ({len(type_states)}):")
        lines.append("-" * 90)

        for state in sorted(type_states, key=lambda x: x.config.priority):
            status = "ONLINE" if state.is_reachable else "OFFLINE"

            work = []
            if state.training_running:
                work.append("TRAIN")
            if state.cmaes_running:
                work.append("CMAES")
            if state.tournament_running:
                work.append("TOURN")
            if state.gauntlet_running:
                work.append("GAUNT")
            if state.selfplay_jobs > 0:
                work.append(f"self({state.selfplay_jobs})")

            work_str = ", ".join(work) if work else "IDLE"

            if state.is_reachable:
                lines.append(
                    f"  {state.config.name:<15} [{status:^7}] "
                    f"GPU:{state.gpu_percent:5.1f}% MEM:{state.gpu_memory_percent:5.1f}% | {work_str}"
                )
            else:
                lines.append(f"  {state.config.name:<15} [{status:^7}]")

    # Summary
    lines.append("")
    lines.append("=" * 90)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 90)

    gpu_idle = [s for s in states if s.config.node_type == "gpu_heavy"
                and s.is_reachable and s.is_gpu_idle]
    gpu_misrouted = [s for s in states if s.config.node_type == "gpu_heavy"
                     and s.is_reachable and s.is_gpu_idle and s.cmaes_running]

    if gpu_idle:
        lines.append(f"\n  - {len(gpu_idle)} GPU-heavy nodes have idle GPUs")
        lines.append("    Action: Start training or GPU CMA-ES on these nodes")

    if gpu_misrouted:
        lines.append(f"\n  - {len(gpu_misrouted)} GPU nodes running CPU-only CMA-ES (inefficient)")
        lines.append("    Action: Run --rebalance to wake CPU nodes and move CPU work")
        lines.append("           Or convert to GPU CMA-ES for 10-100x speedup")

    cpu_configs = cluster_config.get("cpu_rich", [])
    if cpu_configs and gpu_misrouted:
        lines.append(f"\n  - {len(cpu_configs)} CPU-rich nodes configured but may be offline")
        lines.append("    Action: --rebalance will attempt to wake them")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Resource-Aware Work Router")
    parser.add_argument("--status", action="store_true", help="Show cluster status")
    parser.add_argument("--rebalance", action="store_true", help="Rebalance work across cluster")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300, help="Daemon interval (seconds)")
    args = parser.parse_args()

    cluster_config = load_cluster_config()

    if args.daemon:
        logger.info(f"Starting Resource-Aware Router daemon (interval: {args.interval}s)")
        while True:
            try:
                changes = rebalance_cluster(cluster_config)
                if changes > 0:
                    logger.info(f"Made {changes} rebalancing changes")
            except Exception as e:
                logger.error(f"Daemon error: {e}")
            time.sleep(args.interval)

    elif args.rebalance:
        changes = rebalance_cluster(cluster_config, dry_run=args.dry_run)
        logger.info(f"Rebalancing complete: {changes} changes")

    else:
        # Default: show status
        report = generate_status_report(cluster_config)
        print(report)


if __name__ == "__main__":
    main()
