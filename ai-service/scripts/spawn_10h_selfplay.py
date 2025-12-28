#!/usr/bin/env python3
"""Spawn long-running selfplay on all GPU nodes for 10+ hour data generation.

.. note:: December 2025 - SSH Migration
    This script uses direct subprocess.run for SSH operations (line 108).
    Consider migrating to the canonical SSH client for better error handling:
        from app.core.ssh import get_ssh_client, SSHClient
    See app/core/ssh.py for migration guide.

Prioritizes data-starved configurations for maximum training data generation.

Usage:
    python scripts/spawn_10h_selfplay.py              # Deploy to all nodes
    python scripts/spawn_10h_selfplay.py --dry-run    # Show what would be done
    python scripts/spawn_10h_selfplay.py --status     # Check running jobs
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Node:
    name: str
    target: str
    port: int
    key: str
    tier: int  # 1=best, 2=mid, 3=entry
    work_dir: str = "~/ringrift/ai-service"


# GPU nodes by tier
NODES = [
    # Tier 1 - High-end (H100, A100 80GB) - large boards
    Node("nebius-h100-1", "ubuntu@89.169.111.139", 22, "~/.ssh/id_cluster", 1),
    Node("runpod-h100", "root@102.210.171.65", 30755, "~/.runpod/ssh/RunPod-Key-Go", 1, "/workspace/ringrift/ai-service"),
    Node("runpod-a100-1", "root@38.128.233.145", 33085, "~/.runpod/ssh/RunPod-Key-Go", 1, "/workspace/ringrift/ai-service"),
    Node("runpod-a100-2", "root@104.255.9.187", 11681, "~/.runpod/ssh/RunPod-Key-Go", 1, "/workspace/ringrift/ai-service"),

    # Tier 2 - Mid-tier (A100 20GB, L40S, A40, RTX 5090/5080)
    Node("vultr-a100-20gb", "root@208.167.249.164", 22, "~/.ssh/id_ed25519", 2),
    Node("vultr-a100-20gb-2", "root@140.82.15.69", 22, "~/.ssh/id_ed25519", 2),
    Node("nebius-backbone-1", "ubuntu@89.169.112.47", 22, "~/.ssh/id_cluster", 2),
    Node("nebius-l40s-2", "ubuntu@89.169.108.182", 22, "~/.ssh/id_cluster", 2),
    Node("runpod-l40s-2", "root@193.183.22.62", 1630, "~/.runpod/ssh/RunPod-Key-Go", 2, "/workspace/ringrift/ai-service"),
    Node("vast-28918742", "root@ssh8.vast.ai", 38742, "~/.ssh/id_cluster", 2),
    Node("vast-28925166", "root@ssh1.vast.ai", 15166, "~/.ssh/id_cluster", 2),
    Node("vast-29128356", "root@ssh7.vast.ai", 18356, "~/.ssh/id_cluster", 2),
    Node("vast-29031159", "root@ssh5.vast.ai", 31158, "~/.ssh/id_cluster", 2),

    # Tier 3 - Entry (RTX 4060 Ti, 3060 Ti, 3060) - small boards only
    Node("vast-29126088", "root@ssh5.vast.ai", 16088, "~/.ssh/id_cluster", 3),
    Node("vast-28889766", "root@ssh3.vast.ai", 19766, "~/.ssh/id_cluster", 3),
    Node("vast-29046315", "root@ssh2.vast.ai", 16314, "~/.ssh/id_cluster", 3),
    Node("vast-29031161", "root@ssh2.vast.ai", 31160, "~/.ssh/id_cluster", 3),
]

# Config assignments by tier
# Format: (board, players, games)
TIER1_CONFIGS = [
    ("hexagonal", 4, 5000),  # 11 games - CRITICAL
    ("hexagonal", 2, 5000),  # 24 games - CRITICAL
    ("square19", 4, 3000),   # 26 games - CRITICAL
    ("square19", 2, 3000),   # 81 games - CRITICAL
]

TIER2_CONFIGS = [
    ("hexagonal", 3, 3000),  # 209 games
    ("square19", 3, 3000),   # 156 games
    ("hexagonal", 4, 2000),  # duplicate for more coverage
    ("square19", 4, 2000),   # duplicate for more coverage
    ("hex8", 2, 5000),       # 264 games
    ("square8", 3, 5000),    # 494 games
]

TIER3_CONFIGS = [
    ("hex8", 2, 8000),       # 264 games
    ("square8", 3, 8000),    # 494 games
    ("hex8", 4, 6000),       # supplement
    ("square8", 4, 6000),    # supplement
]


def get_config_for_node(tier: int, index: int) -> tuple:
    """Get config assignment for a node based on tier and index."""
    if tier == 1:
        return TIER1_CONFIGS[index % len(TIER1_CONFIGS)]
    elif tier == 2:
        return TIER2_CONFIGS[index % len(TIER2_CONFIGS)]
    else:
        return TIER3_CONFIGS[index % len(TIER3_CONFIGS)]


def run_ssh(node: Node, cmd: str, timeout: int = 30) -> tuple:
    """Run SSH command and return (success, output)."""
    key = node.key.replace("~", str(Path.home()))
    ssh_cmd = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-i", key,
        "-p", str(node.port),
        node.target,
        cmd
    ]
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def spawn_selfplay(node: Node, board: str, players: int, games: int, dry_run: bool = False) -> bool:
    """Spawn selfplay job on a node."""
    cmd = f"""cd {node.work_dir} && \\
source venv/bin/activate 2>/dev/null || true && \\
pkill -f 'selfplay.py.*--board {board}.*--num-players {players}' 2>/dev/null || true && \\
sleep 1 && \\
nohup python scripts/selfplay.py --board {board} --num-players {players} --engine gumbel --num-games {games} --output-dir data/games > /tmp/selfplay_{board}_{players}p_10h.log 2>&1 &"""

    if dry_run:
        print(f"  [DRY-RUN] {node.name}: {board} {players}p x {games} games")
        return True

    success, output = run_ssh(node, cmd, timeout=30)
    return success


def check_status(node: Node) -> tuple:
    """Check selfplay status on a node."""
    cmd = "ps aux | grep 'selfplay.py' | grep -v grep | wc -l"
    success, output = run_ssh(node, cmd, timeout=15)
    if success:
        try:
            return True, int(output.strip())
        except (ValueError, AttributeError):
            return True, 0
    return False, 0


def main():
    parser = argparse.ArgumentParser(description="Spawn 10-hour selfplay on all GPU nodes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--status", action="store_true", help="Check running jobs")
    args = parser.parse_args()

    if args.status:
        print("Checking selfplay status on all nodes...")
        print("-" * 50)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(check_status, node): node for node in NODES}
            for future in as_completed(futures):
                node = futures[future]
                success, count = future.result()
                if not success:
                    print(f"\033[33m[WARN]\033[0m {node.name}: unreachable")
                elif count == 0:
                    print(f"\033[31m[IDLE]\033[0m {node.name}: no selfplay running")
                else:
                    print(f"\033[32m[OK]\033[0m {node.name}: {count} selfplay job(s)")
        return

    print("=" * 60)
    print("10-Hour Selfplay Deployment")
    print("=" * 60)
    if args.dry_run:
        print("\033[33m[DRY RUN MODE - no jobs will be spawned]\033[0m\n")

    # Count nodes by tier
    tier_counts = {1: 0, 2: 0, 3: 0}
    tier_indices = {1: 0, 2: 0, 3: 0}

    for node in NODES:
        tier_counts[node.tier] += 1

    print(f"Tier 1 (H100/A100 80GB): {tier_counts[1]} nodes -> large boards")
    print(f"Tier 2 (A100 20GB/L40S/A40/RTX 50xx): {tier_counts[2]} nodes -> mixed boards")
    print(f"Tier 3 (RTX 3060/4060): {tier_counts[3]} nodes -> small boards")
    print("-" * 60)

    success_count = 0
    fail_count = 0

    for node in NODES:
        board, players, games = get_config_for_node(node.tier, tier_indices[node.tier])
        tier_indices[node.tier] += 1

        print(f"Spawning on {node.name}: {board} {players}p x {games} games... ", end="", flush=True)

        if spawn_selfplay(node, board, players, games, args.dry_run):
            print("\033[32mOK\033[0m")
            success_count += 1
        else:
            print("\033[31mFAILED\033[0m")
            fail_count += 1

    print("-" * 60)
    print(f"Deployment complete: {success_count} succeeded, {fail_count} failed")
    print("\nExpected output per config (10 hours):")
    print("  - hexagonal_4p: ~500-1000 games (slow, complex)")
    print("  - hexagonal_2p: ~1000-2000 games")
    print("  - square19_*: ~1000-3000 games")
    print("  - hex8_*: ~3000-5000 games")
    print("  - square8_*: ~5000-8000 games")
    print("\nMonitor with: python scripts/spawn_10h_selfplay.py --status")


if __name__ == "__main__":
    main()
