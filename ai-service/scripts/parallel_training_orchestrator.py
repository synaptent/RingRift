#!/usr/bin/env python3
"""Parallel Training Orchestrator - Distribute training across cluster nodes.

This script provides a lightweight orchestrator that:
1. Discovers reachable GPU nodes
2. Distributes training jobs for different configs
3. Monitors progress and collects results
4. Handles node failures gracefully

Unlike master_loop.py (which orchestrates everything including selfplay),
this focuses specifically on parallelizing the training phase.

Usage:
    # Train all 12 configs in parallel across cluster
    python scripts/parallel_training_orchestrator.py --all-configs

    # Train specific configs
    python scripts/parallel_training_orchestrator.py --configs hex8_2p,square8_2p

    # Dry run (show what would happen)
    python scripts/parallel_training_orchestrator.py --dry-run

    # Just check cluster readiness
    python scripts/parallel_training_orchestrator.py --check-only

December 2025: Created for maximizing cluster training throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("parallel_training")


# =============================================================================
# Configuration
# =============================================================================

ALL_CONFIGS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]

# GPU memory requirements per config (approximate, in GB)
CONFIG_GPU_MEMORY = {
    "hex8_2p": 8, "hex8_3p": 8, "hex8_4p": 8,
    "square8_2p": 8, "square8_3p": 8, "square8_4p": 10,
    "square19_2p": 24, "square19_3p": 24, "square19_4p": 24,
    "hexagonal_2p": 32, "hexagonal_3p": 32, "hexagonal_4p": 32,
}

# Direct SSH configurations (bypassing potentially broken Tailscale)
CLUSTER_NODES = [
    {
        "name": "runpod-a100-1",
        "ssh": "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p 33085 -i ~/.ssh/id_ed25519 root@38.128.233.145",
        "gpu_memory": 80,
        "ringrift_path": "~/ringrift/ai-service",
    },
    {
        "name": "runpod-a100-2",
        "ssh": "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p 11681 -i ~/.ssh/id_ed25519 root@104.255.9.187",
        "gpu_memory": 80,
        "ringrift_path": "~/ringrift/ai-service",
    },
    {
        "name": "vultr-a100-20gb",
        "ssh": "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 root@208.167.249.164",
        "gpu_memory": 20,
        "ringrift_path": "~/ringrift/ai-service",
    },
    {
        "name": "runpod-h100",
        "ssh": "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p 30178 -i ~/.ssh/id_ed25519 root@102.210.171.65",
        "gpu_memory": 80,
        "ringrift_path": "~/ringrift/ai-service",
    },
    {
        "name": "nebius-backbone-1",
        "ssh": "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/id_cluster ubuntu@89.169.112.47",
        "gpu_memory": 48,
        "ringrift_path": "~/ringrift/ai-service",
    },
    {
        "name": "vast-29129529",
        "ssh": "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p 19528 -i ~/.ssh/id_cluster root@ssh6.vast.ai",
        "gpu_memory": 24,  # 8x RTX 4090 but limited VRAM per GPU
        "ringrift_path": "~/ringrift/ai-service",
    },
]


@dataclass
class NodeStatus:
    """Status of a cluster node."""
    name: str
    ssh_cmd: str
    gpu_memory: int
    ringrift_path: str
    is_reachable: bool = False
    gpu_utilization: float = 0.0
    is_training: bool = False
    python_processes: int = 0
    error: str = ""

    @property
    def is_available(self) -> bool:
        """Check if node is available for training."""
        return (
            self.is_reachable
            and not self.is_training
            and self.gpu_utilization < 50
        )


@dataclass
class TrainingJob:
    """A training job to run on a node."""
    config: str
    board_type: str
    num_players: int
    node: Optional[NodeStatus] = None
    status: str = "pending"  # pending, running, completed, failed
    start_time: float = 0.0
    end_time: float = 0.0
    error: str = ""

    @classmethod
    def from_config(cls, config: str) -> "TrainingJob":
        """Create job from config string like 'hex8_2p'."""
        parts = config.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))
        return cls(config=config, board_type=board_type, num_players=num_players)


def run_ssh_command(ssh_cmd: str, command: str, timeout: int = 30) -> tuple[bool, str]:
    """Run a command via SSH and return success status and output.

    Uses shlex.split to avoid shell=True for security.
    """
    import shlex
    try:
        ssh_parts = shlex.split(ssh_cmd)
        result = subprocess.run(
            ssh_parts + [command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def check_node_status(node_config: dict) -> NodeStatus:
    """Check the status of a cluster node."""
    node = NodeStatus(
        name=node_config["name"],
        ssh_cmd=node_config["ssh"],
        gpu_memory=node_config["gpu_memory"],
        ringrift_path=node_config["ringrift_path"],
    )

    # Check if node is reachable
    success, output = run_ssh_command(node.ssh_cmd, "echo 'ok'", timeout=10)
    if not success:
        node.is_reachable = False
        node.error = f"SSH failed: {output}"
        return node

    node.is_reachable = True

    # Check GPU utilization
    success, output = run_ssh_command(
        node.ssh_cmd,
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1",
        timeout=15,
    )
    if success and output:
        try:
            node.gpu_utilization = float(output.replace(" %", "").replace("%", "").strip())
        except ValueError:
            pass

    # Check if training is running (look for train.py processes)
    success, output = run_ssh_command(
        node.ssh_cmd,
        "pgrep -f 'app.training.train' | wc -l",
        timeout=10,
    )
    if success and output:
        try:
            count = int(output.strip())
            node.is_training = count > 0
        except ValueError:
            pass

    # Count Python processes
    success, output = run_ssh_command(node.ssh_cmd, "pgrep -c python 2>/dev/null || echo 0", timeout=10)
    if success and output:
        try:
            node.python_processes = int(output.strip())
        except ValueError:
            pass

    return node


def discover_available_nodes() -> list[NodeStatus]:
    """Discover all available cluster nodes in parallel."""
    logger.info("Discovering cluster nodes...")

    available_nodes = []

    with ThreadPoolExecutor(max_workers=len(CLUSTER_NODES)) as executor:
        futures = {executor.submit(check_node_status, node): node for node in CLUSTER_NODES}
        for future in as_completed(futures):
            node = future.result()
            if node.is_reachable:
                logger.info(
                    f"  {node.name}: GPU {node.gpu_utilization:.0f}%, "
                    f"training={node.is_training}, procs={node.python_processes}"
                )
                available_nodes.append(node)
            else:
                logger.warning(f"  {node.name}: UNREACHABLE - {node.error}")

    return available_nodes


def assign_jobs_to_nodes(
    jobs: list[TrainingJob],
    nodes: list[NodeStatus],
) -> list[TrainingJob]:
    """Assign training jobs to available nodes based on GPU memory."""
    # Sort nodes by GPU memory (descending) - best nodes get hardest jobs
    available_nodes = [n for n in nodes if n.is_available]
    available_nodes.sort(key=lambda n: n.gpu_memory, reverse=True)

    # Sort jobs by GPU memory requirement (descending)
    pending_jobs = [j for j in jobs if j.status == "pending"]
    pending_jobs.sort(key=lambda j: CONFIG_GPU_MEMORY.get(j.config, 8), reverse=True)

    assigned = []
    used_nodes = set()

    for job in pending_jobs:
        required_memory = CONFIG_GPU_MEMORY.get(job.config, 8)

        for node in available_nodes:
            if node.name in used_nodes:
                continue
            if node.gpu_memory >= required_memory:
                job.node = node
                job.status = "assigned"
                used_nodes.add(node.name)
                assigned.append(job)
                logger.info(f"  Assigned {job.config} to {node.name} (requires {required_memory}GB, has {node.gpu_memory}GB)")
                break
        else:
            logger.warning(f"  No suitable node for {job.config} (requires {required_memory}GB)")

    return assigned


def start_training_job(job: TrainingJob, epochs: int = 30, batch_size: int = 512) -> bool:
    """Start a training job on the assigned node."""
    if not job.node:
        logger.error(f"Job {job.config} has no assigned node")
        return False

    node = job.node

    # Build the training command
    train_cmd = (
        f"cd {node.ringrift_path} && "
        f"mkdir -p logs && "
        f"PYTHONPATH=. nohup python -m app.training.train "
        f"--board-type {job.board_type} --num-players {job.num_players} "
        f"--use-discovery --epochs {epochs} --batch-size {batch_size} "
        f"> logs/train_{job.config}.log 2>&1 &"
    )

    logger.info(f"Starting training on {node.name}: {job.config}")
    success, output = run_ssh_command(node.ssh_cmd, train_cmd, timeout=30)

    if success:
        job.status = "running"
        job.start_time = time.time()
        logger.info(f"  Started {job.config} on {node.name}")
        return True
    else:
        job.status = "failed"
        job.error = output
        logger.error(f"  Failed to start {job.config} on {node.name}: {output}")
        return False


def check_job_status(job: TrainingJob) -> str:
    """Check if a training job is still running."""
    if not job.node or job.status != "running":
        return job.status

    success, output = run_ssh_command(
        job.node.ssh_cmd,
        f"pgrep -f 'train.*{job.board_type}.*{job.num_players}' | wc -l",
        timeout=10,
    )

    if success:
        try:
            count = int(output.strip())
            if count == 0:
                # Training finished - check if model was created
                success, output = run_ssh_command(
                    job.node.ssh_cmd,
                    f"ls -la {job.node.ringrift_path}/models/*{job.board_type}*{job.num_players}p*.pth 2>/dev/null | tail -1",
                    timeout=10,
                )
                if "pth" in output:
                    job.status = "completed"
                    job.end_time = time.time()
                else:
                    job.status = "failed"
                    job.error = "No model output found"
        except ValueError:
            pass

    return job.status


def print_summary(jobs: list[TrainingJob], nodes: list[NodeStatus]) -> None:
    """Print a summary of the orchestration state."""
    print("\n" + "=" * 60)
    print("PARALLEL TRAINING ORCHESTRATOR SUMMARY")
    print("=" * 60)

    print(f"\nNodes ({len(nodes)} total):")
    for node in nodes:
        status = "READY" if node.is_available else ("BUSY" if node.is_training else "UNAVAILABLE")
        print(f"  {node.name}: {status} (GPU: {node.gpu_utilization:.0f}%, {node.gpu_memory}GB)")

    print(f"\nJobs ({len(jobs)} total):")
    for status in ["completed", "running", "assigned", "pending", "failed"]:
        status_jobs = [j for j in jobs if j.status == status]
        if status_jobs:
            print(f"  {status.upper()}: {', '.join(j.config for j in status_jobs)}")

    print("=" * 60 + "\n")


async def run_orchestrator(
    configs: list[str],
    epochs: int = 30,
    batch_size: int = 512,
    dry_run: bool = False,
    check_only: bool = False,
) -> None:
    """Main orchestrator loop."""
    # Discover available nodes
    nodes = discover_available_nodes()

    if not nodes:
        logger.error("No reachable nodes found!")
        return

    logger.info(f"Found {len(nodes)} reachable nodes")

    if check_only:
        print_summary([], nodes)
        return

    # Create training jobs
    jobs = [TrainingJob.from_config(config) for config in configs]
    logger.info(f"Created {len(jobs)} training jobs")

    # Assign jobs to nodes
    assigned_jobs = assign_jobs_to_nodes(jobs, nodes)

    if dry_run:
        print_summary(jobs, nodes)
        logger.info("DRY RUN - no jobs started")
        return

    if not assigned_jobs:
        logger.warning("No jobs could be assigned to available nodes")
        print_summary(jobs, nodes)
        return

    # Start assigned jobs
    for job in assigned_jobs:
        start_training_job(job, epochs=epochs, batch_size=batch_size)
        await asyncio.sleep(2)  # Small delay between starts

    print_summary(jobs, nodes)

    # Monitor until completion
    logger.info("Monitoring training progress...")
    running_jobs = [j for j in jobs if j.status == "running"]

    while running_jobs:
        await asyncio.sleep(60)  # Check every minute

        for job in running_jobs:
            check_job_status(job)

        running_jobs = [j for j in jobs if j.status == "running"]
        completed = [j for j in jobs if j.status == "completed"]
        failed = [j for j in jobs if j.status == "failed"]

        logger.info(
            f"Progress: {len(completed)} completed, {len(running_jobs)} running, {len(failed)} failed"
        )

    print_summary(jobs, nodes)
    logger.info("Orchestration complete!")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parallel Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--configs", "-c",
        type=str,
        default=None,
        help="Comma-separated list of configs (e.g., hex8_2p,square8_2p)",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Train all 12 configurations",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=512,
        help="Training batch size (default: 512)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without starting jobs",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check cluster readiness",
    )

    args = parser.parse_args()

    if args.all_configs:
        configs = ALL_CONFIGS
    elif args.configs:
        configs = [c.strip() for c in args.configs.split(",")]
    else:
        # Default: prioritize configs that need more games
        configs = ["hex8_2p", "hex8_3p", "square8_2p", "square8_3p"]

    # Validate configs
    for config in configs:
        if config not in ALL_CONFIGS:
            logger.error(f"Invalid config: {config}")
            logger.info(f"Valid configs: {', '.join(ALL_CONFIGS)}")
            sys.exit(1)

    logger.info(f"Target configs: {', '.join(configs)}")

    asyncio.run(run_orchestrator(
        configs=configs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        check_only=args.check_only,
    ))


if __name__ == "__main__":
    main()
