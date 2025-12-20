#!/usr/bin/env python3
"""Launch distributed NAS across Vast.ai instances.

This script:
1. Discovers available Vast instances
2. Syncs code to each instance
3. Launches NAS workers in parallel
4. Coordinates results collection

Usage:
    # Launch NAS on all available Vast instances
    python scripts/launch_distributed_nas.py --strategy evolutionary --generations 30

    # Launch on specific instances
    python scripts/launch_distributed_nas.py --instances 28844398,28889942 --strategy bayesian --trials 100

    # Dry run (show what would be executed)
    python scripts/launch_distributed_nas.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))


@dataclass
class VastInstance:
    """Vast.ai instance info."""
    instance_id: str
    ssh_host: str
    ssh_port: int
    gpu_model: str
    gpu_count: int
    cost_per_hour: float
    status: str


def get_vast_instances() -> list[VastInstance]:
    """Get list of running Vast instances."""
    try:
        result = subprocess.run(
            ["vastai", "show", "instances", "--raw"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"Error getting instances: {result.stderr}")
            return []

        instances = json.loads(result.stdout)
        return [
            VastInstance(
                instance_id=str(inst.get("id", "")),
                ssh_host=inst.get("ssh_host", ""),
                ssh_port=inst.get("ssh_port", 22),
                gpu_model=inst.get("gpu_name", "unknown"),
                gpu_count=inst.get("num_gpus", 1),
                cost_per_hour=inst.get("dph_total", 0),
                status=inst.get("actual_status", "unknown"),
            )
            for inst in instances
            if inst.get("actual_status") == "running"
        ]
    except Exception as e:
        print(f"Error: {e}")
        return []


def sync_code_to_instance(inst: VastInstance, dry_run: bool = False) -> bool:
    """Sync code to a Vast instance."""
    ssh_target = f"root@{inst.ssh_host}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(inst.ssh_port)]

    # Check if RingRift directory exists
    check_cmd = ["ssh", *ssh_opts, ssh_target, "test -d /root/RingRift && echo exists"]
    if dry_run:
        print(f"  Would check: {' '.join(check_cmd)}")
        return True

    try:
        result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=30)
        if "exists" not in result.stdout:
            print(f"  RingRift not found on {inst.instance_id}, needs setup")
            return False
    except Exception as e:
        print(f"  SSH check failed for {inst.instance_id}: {e}")
        return False

    # Sync critical files
    rsync_cmd = [
        "rsync", "-avz", "--delete",
        "-e", f"ssh {' '.join(ssh_opts)}",
        f"{AI_SERVICE_ROOT}/scripts/",
        f"{ssh_target}:/root/RingRift/ai-service/scripts/",
    ]
    if dry_run:
        print(f"  Would sync: {' '.join(rsync_cmd)}")
        return True

    try:
        result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  Sync failed for {inst.instance_id}: {result.stderr[:200]}")
            return False
        return True
    except Exception as e:
        print(f"  Sync error for {inst.instance_id}: {e}")
        return False


def launch_nas_worker(
    inst: VastInstance,
    strategy: str,
    iterations: int,
    board: str,
    players: int,
    run_id: str,
    dry_run: bool = False,
) -> int | None:
    """Launch NAS worker on a Vast instance. Returns PID if successful."""
    ssh_target = f"root@{inst.ssh_host}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(inst.ssh_port)]

    # Calculate iterations per GPU
    gpu_iterations = iterations // inst.gpu_count

    worker_cmd = f"""
cd /root/RingRift/ai-service && \
RINGRIFT_NAS_REAL_TRAINING=1 \
RINGRIFT_NAS_BOARD={board} \
RINGRIFT_NAS_PLAYERS={players} \
PYTHONPATH=/root/RingRift/ai-service \
nohup python scripts/neural_architecture_search.py \
    --strategy {strategy} \
    --{'generations' if strategy == 'evolutionary' else 'trials'} {gpu_iterations} \
    --board {board} \
    --players {players} \
    --output-dir /root/RingRift/ai-service/logs/nas/{run_id} \
    > /tmp/nas_{run_id}.log 2>&1 &
echo $!
"""

    full_cmd = ["ssh", *ssh_opts, ssh_target, worker_cmd]

    if dry_run:
        print(f"  Would launch on {inst.instance_id} ({inst.gpu_count}x {inst.gpu_model}):")
        print(f"    {worker_cmd.strip()}")
        return 0

    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            pid = result.stdout.strip().split()[-1]
            print(f"  Started NAS on {inst.instance_id} (PID: {pid})")
            return int(pid)
        else:
            print(f"  Failed to start on {inst.instance_id}: {result.stderr[:200]}")
            return None
    except Exception as e:
        print(f"  Error launching on {inst.instance_id}: {e}")
        return None


def check_worker_status(inst: VastInstance, pid: int) -> str:
    """Check if worker is still running."""
    ssh_target = f"root@{inst.ssh_host}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(inst.ssh_port)]

    check_cmd = ["ssh", *ssh_opts, ssh_target, f"ps -p {pid} > /dev/null 2>&1 && echo running || echo stopped"]

    try:
        result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=15)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def collect_results(inst: VastInstance, run_id: str, output_dir: Path) -> bool:
    """Collect NAS results from an instance."""
    ssh_target = f"root@{inst.ssh_host}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(inst.ssh_port)]

    output_dir.mkdir(parents=True, exist_ok=True)

    rsync_cmd = [
        "rsync", "-avz",
        "-e", f"ssh {' '.join(ssh_opts)}",
        f"{ssh_target}:/root/RingRift/ai-service/logs/nas/{run_id}/",
        f"{output_dir}/{inst.instance_id}/",
    ]

    try:
        result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Launch distributed NAS on Vast.ai")
    parser.add_argument("--strategy", choices=["evolutionary", "bayesian", "random"],
                       default="evolutionary", help="NAS strategy")
    parser.add_argument("--generations", type=int, default=30,
                       help="Generations (evolutionary) or trials (bayesian/random)")
    parser.add_argument("--board", default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--instances", help="Comma-separated instance IDs (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be executed")
    parser.add_argument("--sync-only", action="store_true", help="Only sync code, don't launch")
    parser.add_argument("--collect", help="Collect results from run ID")
    parser.add_argument("--output-dir", default=str(AI_SERVICE_ROOT / "logs" / "nas_distributed"),
                       help="Local output directory")

    args = parser.parse_args()

    print("=" * 60)
    print("Distributed Neural Architecture Search Launcher")
    print("=" * 60)

    # Get instances
    print("\nDiscovering Vast.ai instances...")
    instances = get_vast_instances()

    if not instances:
        print("No running Vast instances found!")
        return 1

    # Filter by instance IDs if specified
    if args.instances:
        target_ids = set(args.instances.split(","))
        instances = [i for i in instances if i.instance_id in target_ids]

    print(f"\nFound {len(instances)} running instances:")
    total_gpus = 0
    total_cost = 0
    for inst in instances:
        print(f"  {inst.instance_id}: {inst.gpu_count}x {inst.gpu_model} (${inst.cost_per_hour:.2f}/hr)")
        total_gpus += inst.gpu_count
        total_cost += inst.cost_per_hour

    print(f"\nTotal: {total_gpus} GPUs, ${total_cost:.2f}/hr")

    # Collect mode
    if args.collect:
        print(f"\nCollecting results for run {args.collect}...")
        output_dir = Path(args.output_dir)
        for inst in instances:
            print(f"  Collecting from {inst.instance_id}...")
            if collect_results(inst, args.collect, output_dir):
                print("    Success")
            else:
                print("    Failed")
        return 0

    # Sync code
    print("\nSyncing code to instances...")
    for inst in instances:
        print(f"  Syncing to {inst.instance_id}...")
        if not sync_code_to_instance(inst, args.dry_run):
            print("    Warning: sync failed, skipping instance")
            instances.remove(inst)

    if args.sync_only:
        print("\nSync complete (--sync-only mode)")
        return 0

    # Launch NAS
    run_id = f"nas_dist_{int(time.time())}"
    iterations = args.generations

    print("\nLaunching distributed NAS...")
    print(f"  Strategy: {args.strategy}")
    print(f"  Iterations: {iterations}")
    print(f"  Board: {args.board}_{args.players}p")
    print(f"  Run ID: {run_id}")

    workers = {}
    for inst in instances:
        pid = launch_nas_worker(
            inst, args.strategy, iterations,
            args.board, args.players, run_id,
            args.dry_run,
        )
        if pid is not None:
            workers[inst.instance_id] = (inst, pid)

    if args.dry_run:
        print("\nDry run complete. No workers launched.")
        return 0

    print(f"\nLaunched {len(workers)} workers")
    print(f"Run ID: {run_id}")
    print(f"\nTo monitor: ssh root@<host> -p <port> tail -f /tmp/nas_{run_id}.log")
    print(f"To collect results: python {__file__} --collect {run_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
