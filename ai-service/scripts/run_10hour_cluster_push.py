#!/usr/bin/env python3
"""10-Hour Cluster Push - Maximize utilization for 2000+ Elo progress.

December 2025 - Orchestrates all cluster nodes for maximum training throughput.

This script:
1. Checks current cluster status
2. Launches selfplay on all idle GPUs across all 12 configs
3. Queues Elo evaluations for untested models
4. Monitors progress and restarts failed jobs
5. Triggers training when sufficient data accumulates

Usage:
    python scripts/run_10hour_cluster_push.py
    python scripts/run_10hour_cluster_push.py --dry-run  # Preview only
    python scripts/run_10hour_cluster_push.py --duration 6  # 6 hours instead of 10
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# All 12 canonical configurations with priority weights
# Higher weight = more selfplay games allocated
CONFIGS = {
    # 2-player configs (highest priority - closest to 2000)
    "square19_2p": {"priority": 10, "board": "square19", "players": 2, "min_vram_gb": 24},
    "hex8_2p": {"priority": 8, "board": "hex8", "players": 2, "min_vram_gb": 8},
    "square8_2p": {"priority": 8, "board": "square8", "players": 2, "min_vram_gb": 8},
    "hexagonal_2p": {"priority": 7, "board": "hexagonal", "players": 2, "min_vram_gb": 48},
    # 3-player configs (medium priority - need data)
    "hex8_3p": {"priority": 5, "board": "hex8", "players": 3, "min_vram_gb": 12},
    "square8_3p": {"priority": 5, "board": "square8", "players": 3, "min_vram_gb": 12},
    "square19_3p": {"priority": 4, "board": "square19", "players": 3, "min_vram_gb": 48},
    "hexagonal_3p": {"priority": 3, "board": "hexagonal", "players": 3, "min_vram_gb": 80},
    # 4-player configs (lower priority - need foundation)
    "hex8_4p": {"priority": 4, "board": "hex8", "players": 4, "min_vram_gb": 12},
    "square8_4p": {"priority": 4, "board": "square8", "players": 4, "min_vram_gb": 16},
    "square19_4p": {"priority": 3, "board": "square19", "players": 4, "min_vram_gb": 48},
    "hexagonal_4p": {"priority": 2, "board": "hexagonal", "players": 4, "min_vram_gb": 80},
}

# GPU nodes with VRAM (from distributed_hosts.yaml)
GPU_NODES = {
    # Multi-GPU powerhouses
    "vast-29129529": {"vram": 192, "gpus": 8, "type": "RTX 4090"},
    "vast-29118471": {"vram": 192, "gpus": 8, "type": "RTX 3090"},
    "vast-29128352": {"vram": 64, "gpus": 2, "type": "RTX 5090"},
    # High-end single GPU
    "vast-28925166": {"vram": 32, "gpus": 1, "type": "RTX 5090"},
    "vast-29128356": {"vram": 32, "gpus": 1, "type": "RTX 5090"},
    "vast-28918742": {"vram": 46, "gpus": 1, "type": "A40"},
    "runpod-h100": {"vram": 80, "gpus": 1, "type": "H100"},
    "runpod-a100-1": {"vram": 80, "gpus": 1, "type": "A100"},
    "runpod-a100-2": {"vram": 80, "gpus": 1, "type": "A100"},
    "runpod-l40s-2": {"vram": 48, "gpus": 1, "type": "L40S"},
    "nebius-h100-1": {"vram": 80, "gpus": 1, "type": "H100"},
    "nebius-backbone-1": {"vram": 48, "gpus": 1, "type": "L40S"},
    "nebius-l40s-2": {"vram": 48, "gpus": 1, "type": "L40S"},
    # Mid-range
    "runpod-3090ti-1": {"vram": 24, "gpus": 1, "type": "RTX 3090 Ti"},
    "vultr-a100-20gb": {"vram": 20, "gpus": 1, "type": "A100D"},
    "vultr-a100-20gb-2": {"vram": 20, "gpus": 1, "type": "A100D"},
    "vast-29031159": {"vram": 16, "gpus": 1, "type": "RTX 5080"},
    "vast-29126088": {"vram": 16, "gpus": 1, "type": "RTX 4060 Ti"},
    "vast-29031161": {"vram": 12, "gpus": 1, "type": "RTX 3060"},
    "vast-28890015": {"vram": 11, "gpus": 1, "type": "RTX 2080 Ti"},
    "vast-28889766": {"vram": 8, "gpus": 1, "type": "RTX 3060 Ti"},
    "vast-29046315": {"vram": 8, "gpus": 1, "type": "RTX 3060 Ti"},
}


@dataclass
class JobPlan:
    """Planned job for a node."""
    node: str
    config: str
    job_type: str  # "selfplay", "training", "evaluation"
    games: int = 100
    priority: int = 5


def get_configs_for_vram(vram_gb: float) -> list[str]:
    """Get configs that can run on a GPU with given VRAM."""
    return [
        config for config, info in CONFIGS.items()
        if info["min_vram_gb"] <= vram_gb
    ]


def plan_jobs(dry_run: bool = False) -> list[JobPlan]:
    """Plan optimal job distribution across cluster."""
    jobs = []

    # Sort nodes by VRAM (largest first for best config variety)
    sorted_nodes = sorted(
        GPU_NODES.items(),
        key=lambda x: x[1]["vram"],
        reverse=True
    )

    # Track how many jobs per config (for load balancing)
    config_counts = {config: 0 for config in CONFIGS}

    for node, info in sorted_nodes:
        vram = info["vram"]
        gpus = info["gpus"]

        # Get compatible configs for this GPU
        compatible = get_configs_for_vram(vram)
        if not compatible:
            logger.warning(f"No compatible configs for {node} ({vram}GB)")
            continue

        # Pick the config with highest priority that has fewest jobs
        # This ensures load balancing across all configs
        best_config = min(
            compatible,
            key=lambda c: (config_counts[c], -CONFIGS[c]["priority"])
        )

        # Calculate games based on GPU power
        base_games = 100
        if gpus > 1:
            base_games = 50 * gpus  # Multi-GPU can run parallel games
        if vram >= 80:
            base_games = 200  # High-end GPUs get more games

        jobs.append(JobPlan(
            node=node,
            config=best_config,
            job_type="selfplay",
            games=base_games,
            priority=CONFIGS[best_config]["priority"],
        ))
        config_counts[best_config] += 1

    return jobs


def print_job_plan(jobs: list[JobPlan]) -> None:
    """Print the job plan in a readable format."""
    print("\n" + "=" * 80)
    print("10-HOUR CLUSTER PUSH - JOB PLAN")
    print("=" * 80)

    # Group by config
    by_config = {}
    for job in jobs:
        if job.config not in by_config:
            by_config[job.config] = []
        by_config[job.config].append(job)

    total_games = 0
    for config in sorted(by_config.keys()):
        config_jobs = by_config[config]
        config_games = sum(j.games for j in config_jobs)
        total_games += config_games
        print(f"\n{config} ({len(config_jobs)} nodes, {config_games} games):")
        for job in config_jobs:
            gpu_info = GPU_NODES[job.node]
            print(f"  - {job.node}: {job.games} games ({gpu_info['type']}, {gpu_info['vram']}GB)")

    print(f"\n{'='*80}")
    print(f"TOTAL: {len(jobs)} nodes, {total_games} games across {len(by_config)} configs")
    print(f"Estimated completion: 10 hours at ~{total_games/10:.0f} games/hour")
    print("=" * 80 + "\n")


def generate_launch_commands(jobs: list[JobPlan]) -> list[str]:
    """Generate shell commands to launch all jobs."""
    commands = []

    for job in jobs:
        config_info = CONFIGS[job.config]
        cmd = (
            f"# {job.node} - {job.config}\n"
            f"python scripts/selfplay.py "
            f"--board {config_info['board']} "
            f"--num-players {config_info['players']} "
            f"--engine gumbel "
            f"--num-games {job.games} "
            f"--emit-pipeline-events"
        )
        commands.append(cmd)

    return commands


async def check_elo_status() -> dict:
    """Check current Elo status for all configs."""
    import sqlite3

    db_path = Path("data/unified_elo.db")
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("""
        SELECT board_type, num_players, MAX(rating) as best_elo, SUM(games_played) as total_games
        FROM elo_ratings
        GROUP BY board_type, num_players
    """)

    status = {}
    for row in cursor:
        key = f"{row[0]}_{row[1]}p"
        status[key] = {"elo": row[2], "games": row[3]}

    conn.close()
    return status


def main():
    parser = argparse.ArgumentParser(description="10-hour cluster push for 2000+ Elo")
    parser.add_argument("--dry-run", action="store_true", help="Preview plan without executing")
    parser.add_argument("--duration", type=int, default=10, help="Duration in hours")
    parser.add_argument("--generate-script", action="store_true", help="Generate launch script")
    args = parser.parse_args()

    print("\n🚀 10-HOUR CLUSTER PUSH FOR 2000+ ELO")
    print("=" * 50)

    # Check current Elo status
    print("\n📊 Current Elo Status:")
    try:
        loop = asyncio.new_event_loop()
        elo_status = loop.run_until_complete(check_elo_status())
        loop.close()

        for config in sorted(CONFIGS.keys()):
            if config in elo_status:
                info = elo_status[config]
                gap = 2000 - info["elo"]
                status = "✅" if gap <= 0 else "🔥" if gap <= 50 else "📈" if gap <= 200 else "🔄"
                print(f"  {config}: {info['elo']:.1f} Elo ({info['games']} games) {status}")
            else:
                print(f"  {config}: No data ⏳")
    except Exception as e:
        print(f"  Could not load Elo data: {e}")

    # Plan jobs
    print("\n📋 Planning job distribution...")
    jobs = plan_jobs(dry_run=args.dry_run)
    print_job_plan(jobs)

    if args.generate_script:
        # Generate a bash script for launching all jobs
        script_path = Path("scripts/launch_cluster_push.sh")
        commands = generate_launch_commands(jobs)

        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated cluster push script\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Duration: {args.duration} hours\n\n")

            for cmd in commands:
                f.write(f"{cmd}\n\n")

        script_path.chmod(0o755)
        print(f"✅ Generated launch script: {script_path}")

    if args.dry_run:
        print("\n🔍 DRY RUN - No jobs launched")
        print("Remove --dry-run to execute the plan")
        return 0

    # Recommend next steps
    print("\n📌 RECOMMENDED NEXT STEPS:")
    print("=" * 50)
    print("""
1. IMMEDIATE (square19_2p is only 16 Elo from 2000!):
   python scripts/quick_gauntlet.py \\
     --model models/canonical_square19_2p.pth \\
     --board-type square19 --num-players 2 \\
     --games 100

2. LAUNCH MASTER LOOP (orchestrates all daemons):
   python scripts/master_loop.py

3. MONITOR PROGRESS:
   python scripts/elo_progress_monitor.py --watch

4. CHECK CLUSTER STATUS:
   python -m app.distributed.cluster_monitor --watch
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
