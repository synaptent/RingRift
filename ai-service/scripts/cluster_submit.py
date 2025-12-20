#!/usr/bin/env python3
"""Unified Cluster Job Submission CLI.

Submit jobs across Slurm, Vast.ai, and P2P backends from a single interface.

Usage:
    # Submit selfplay to best available node
    python scripts/cluster_submit.py selfplay --board square8 --players 2 --games 1000

    # Submit to specific node
    python scripts/cluster_submit.py selfplay --node lambda-gh200-f --games 2000

    # Submit GPU selfplay
    python scripts/cluster_submit.py gpu-selfplay --node lambda-2xh100 --games 5000

    # Submit training
    python scripts/cluster_submit.py training --data data/training/latest --epochs 100

    # List all jobs
    python scripts/cluster_submit.py list

    # Show cluster status
    python scripts/cluster_submit.py status

    # Cancel a job
    python scripts/cluster_submit.py cancel <job_id>
"""

import argparse
import asyncio
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.coordination.unified_scheduler import (
    UnifiedJob,
    JobType,
    Backend,
    JobState,
    get_scheduler,
    submit_selfplay,
    submit_gpu_selfplay,
    submit_training,
)
from app.coordination.slurm_backend import (
    get_slurm_backend,
    SlurmPartition,
)


def print_table(headers: list[str], rows: list[list[str]], colors: bool = True) -> None:
    """Print a formatted table."""
    if not rows:
        print("No data to display.")
        return

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        print(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


async def cmd_selfplay(args) -> int:
    """Submit a selfplay job."""
    job_id = await submit_selfplay(
        board_type=args.board,
        num_players=args.players,
        num_games=args.games,
        target_node=args.node,
        engine_mode=args.engine,
    )
    print(f"✅ Submitted selfplay job: {job_id}")
    return 0


async def cmd_gpu_selfplay(args) -> int:
    """Submit a GPU selfplay job."""
    job_id = await submit_gpu_selfplay(
        board_type=args.board,
        num_players=args.players,
        num_games=args.games,
        target_node=args.node,
    )
    print(f"✅ Submitted GPU selfplay job: {job_id}")
    return 0


async def cmd_training(args) -> int:
    """Submit a training job."""
    job_id = await submit_training(
        data_path=args.data,
        model_name=args.model,
        epochs=args.epochs,
        target_node=args.node,
    )
    print(f"✅ Submitted training job: {job_id}")
    return 0


async def cmd_list(args) -> int:
    """List jobs."""
    scheduler = get_scheduler()

    backend = None
    if args.backend:
        backend = Backend(args.backend)

    state = None
    if args.state:
        state = JobState(args.state)

    jobs = await scheduler.list_jobs(backend=backend, state=state, limit=args.limit)

    if args.json:
        print(json.dumps([
            {
                "id": j.unified_id,
                "backend_id": j.job_id,
                "backend": j.backend.value,
                "state": j.state.value,
                "node": j.node,
            }
            for j in jobs
        ], indent=2))
    else:
        headers = ["ID", "Backend", "Backend ID", "State", "Node"]
        rows = [
            [j.unified_id, j.backend.value, j.job_id, j.state.value, j.node or "-"]
            for j in jobs
        ]
        print_table(headers, rows)

    return 0


async def cmd_status(args) -> int:
    """Show cluster status."""
    scheduler = get_scheduler()
    status = await scheduler.get_cluster_status()

    if args.json:
        print(json.dumps(status, indent=2))
        return 0

    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║               UNIFIED CLUSTER STATUS                          ║")
    print("╠═══════════════════════════════════════════════════════════════╣")

    # Slurm
    slurm = status.get("slurm", {})
    if slurm.get("enabled"):
        print(f"║ SLURM:  {slurm.get('nodes', 0):3} nodes, {slurm.get('idle_nodes', 0):3} idle, {slurm.get('jobs', 0):3} jobs     ║")
    else:
        print("║ SLURM:  Disabled                                              ║")

    # Vast
    vast = status.get("vast", {})
    if vast.get("enabled"):
        print(f"║ VAST:   {vast.get('instances', 0):3} instances, {vast.get('running', 0):3} running                   ║")
    else:
        print("║ VAST:   Disabled                                              ║")

    # P2P
    p2p = status.get("p2p", {})
    if p2p.get("enabled"):
        print(f"║ P2P:    {p2p.get('nodes', 0):3} nodes                                          ║")
    else:
        print("║ P2P:    Disabled                                              ║")

    print("╠═══════════════════════════════════════════════════════════════╣")

    # Jobs
    jobs = status.get("jobs", {})
    print(f"║ JOBS:   {jobs.get('total', 0):3} total, {jobs.get('running', 0):3} running, {jobs.get('pending', 0):3} pending       ║")

    print("╚═══════════════════════════════════════════════════════════════╝\n")

    # Detailed Slurm node status if requested
    if args.detailed and slurm.get("enabled"):
        print("SLURM NODES:")
        backend = get_slurm_backend()
        nodes = await backend.get_nodes(refresh=True)

        headers = ["Node", "Partition", "State", "CPUs", "Memory", "GPU"]
        rows = [
            [
                n.name,
                n.partition,
                n.state,
                str(n.cpus),
                f"{n.memory_mb // 1024}GB",
                n.gpu_type,
            ]
            for n in sorted(nodes.values(), key=lambda x: x.name)
        ]
        print_table(headers, rows)
        print()

    return 0


async def cmd_slurm_queue(args) -> int:
    """Show Slurm queue."""
    backend = get_slurm_backend()
    jobs = await backend.get_jobs(refresh=True)

    if args.json:
        print(json.dumps([
            {
                "job_id": j.job_id,
                "name": j.name,
                "state": j.state.value,
                "partition": j.partition,
                "node": j.node,
                "runtime": j.run_time,
            }
            for j in jobs.values()
        ], indent=2))
    else:
        headers = ["Job ID", "Name", "State", "Partition", "Node", "Runtime"]
        rows = [
            [
                str(j.job_id),
                j.name[:30],
                j.state.value,
                j.partition,
                j.node or "-",
                j.run_time or "-",
            ]
            for j in sorted(jobs.values(), key=lambda x: x.job_id)
        ]
        print_table(headers, rows)

    return 0


async def cmd_slurm_submit(args) -> int:
    """Submit directly to Slurm."""
    from app.coordination.slurm_backend import SlurmJob

    partition = SlurmPartition(args.partition)

    job = SlurmJob(
        name=args.name,
        partition=partition,
        command=args.command,
        nodelist=args.node,
        cpus_per_task=args.cpus,
        memory_gb=args.memory,
        gpus=args.gpus,
        time_limit=args.time,
    )

    backend = get_slurm_backend()
    job_id = await backend.submit_job(job)

    if job_id:
        print(f"✅ Submitted Slurm job {job_id}: {args.name}")
        return 0
    else:
        print("❌ Failed to submit job")
        return 1


async def cmd_cancel(args) -> int:
    """Cancel a job."""
    scheduler = get_scheduler()
    success = await scheduler.cancel(args.job_id)

    if success:
        print(f"✅ Cancelled job: {args.job_id}")
        return 0
    else:
        print(f"❌ Failed to cancel job: {args.job_id}")
        return 1


async def cmd_fill_idle(args) -> int:
    """Fill idle nodes with selfplay jobs."""
    backend = get_slurm_backend()
    scheduler = get_scheduler()

    # Get idle nodes - optionally filter by partition if explicitly specified
    partition_filter = None
    if args.partition and args.partition != "all":
        try:
            partition_filter = SlurmPartition(args.partition)
        except ValueError:
            pass  # Unknown partition, don't filter

    idle_nodes = await backend.get_idle_nodes(partition=partition_filter)

    if not idle_nodes:
        print("No idle nodes found.")
        return 0

    print(f"Found {len(idle_nodes)} idle nodes:")
    for node in idle_nodes:
        print(f"  - {node.name} ({node.gpu_type})")

    if args.dry_run:
        print("\n[DRY RUN] Would submit jobs to these nodes.")
        return 0

    submitted = 0
    for node in idle_nodes[:args.max_jobs]:
        job = UnifiedJob(
            name=f"fill-{node.name}-{args.board}",
            job_type=JobType.GPU_SELFPLAY if args.gpu else JobType.SELFPLAY,
            target_node=node.name,
            config={
                "board_type": args.board,
                "num_players": args.players,
                "num_games": args.games,
            },
        )
        job_id = await scheduler.submit(job)
        print(f"✅ Submitted to {node.name}: {job_id}")
        submitted += 1

    print(f"\nSubmitted {submitted} jobs to idle nodes.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified Cluster Job Submission CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # selfplay
    sp_parser = subparsers.add_parser("selfplay", help="Submit selfplay job")
    sp_parser.add_argument("--board", "-b", default="square8", help="Board type")
    sp_parser.add_argument("--players", "-p", type=int, default=2, help="Number of players")
    sp_parser.add_argument("--games", "-g", type=int, default=1000, help="Number of games")
    sp_parser.add_argument("--node", "-n", help="Target node")
    sp_parser.add_argument("--engine", "-e", default="mixed", help="Engine mode")

    # gpu-selfplay
    gsp_parser = subparsers.add_parser("gpu-selfplay", help="Submit GPU selfplay job")
    gsp_parser.add_argument("--board", "-b", default="square8", help="Board type")
    gsp_parser.add_argument("--players", "-p", type=int, default=2, help="Number of players")
    gsp_parser.add_argument("--games", "-g", type=int, default=2000, help="Number of games")
    gsp_parser.add_argument("--node", "-n", help="Target node")

    # training
    train_parser = subparsers.add_parser("training", help="Submit training job")
    train_parser.add_argument("--data", "-d", required=True, help="Data path")
    train_parser.add_argument("--model", "-m", help="Model name")
    train_parser.add_argument("--epochs", "-e", type=int, default=100, help="Epochs")
    train_parser.add_argument("--node", "-n", help="Target node")

    # list
    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.add_argument("--backend", choices=["slurm", "vast", "p2p"], help="Filter by backend")
    list_parser.add_argument("--state", choices=["pending", "running", "completed", "failed"], help="Filter by state")
    list_parser.add_argument("--limit", type=int, default=50, help="Max jobs to show")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # status
    status_parser = subparsers.add_parser("status", help="Show cluster status")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")
    status_parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed node info")

    # slurm-queue
    sq_parser = subparsers.add_parser("slurm-queue", help="Show Slurm queue")
    sq_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # slurm-submit
    ss_parser = subparsers.add_parser("slurm-submit", help="Submit directly to Slurm")
    ss_parser.add_argument("--name", "-n", required=True, help="Job name")
    ss_parser.add_argument("--command", "-c", required=True, help="Command to run")
    ss_parser.add_argument("--partition", "-p", default="gpu-selfplay", help="Slurm partition")
    ss_parser.add_argument("--node", help="Target node")
    ss_parser.add_argument("--cpus", type=int, default=16, help="CPUs per task")
    ss_parser.add_argument("--memory", type=int, default=64, help="Memory in GB")
    ss_parser.add_argument("--gpus", type=int, default=1, help="GPUs")
    ss_parser.add_argument("--time", default="8:00:00", help="Time limit")

    # cancel
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("job_id", help="Job ID to cancel")

    # fill-idle
    fill_parser = subparsers.add_parser("fill-idle", help="Fill idle nodes with jobs")
    fill_parser.add_argument("--board", "-b", default="square8", help="Board type")
    fill_parser.add_argument("--players", "-p", type=int, default=2, help="Number of players")
    fill_parser.add_argument("--games", "-g", type=int, default=2000, help="Number of games")
    fill_parser.add_argument("--gpu", action="store_true", help="Use GPU selfplay")
    fill_parser.add_argument("--partition", default="all", help="Partition to fill (default: all)")
    fill_parser.add_argument("--max-jobs", type=int, default=10, help="Max jobs to submit")
    fill_parser.add_argument("--dry-run", action="store_true", help="Don't actually submit")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Map commands to functions
    cmd_map = {
        "selfplay": cmd_selfplay,
        "gpu-selfplay": cmd_gpu_selfplay,
        "training": cmd_training,
        "list": cmd_list,
        "status": cmd_status,
        "slurm-queue": cmd_slurm_queue,
        "slurm-submit": cmd_slurm_submit,
        "cancel": cmd_cancel,
        "fill-idle": cmd_fill_idle,
    }

    cmd_func = cmd_map.get(args.command)
    if cmd_func:
        return asyncio.run(cmd_func(args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
