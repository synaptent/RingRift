#!/usr/bin/env python3
"""Smart Work Router - Routes work to appropriate nodes based on capabilities.

This script ensures:
- GPU-heavy nodes (GH200, H100) prioritize GPU work (training, tournaments)
- CPU-rich nodes prioritize CPU work (CMA-ES, hybrid selfplay)
- Work is balanced across the cluster

Key insight: GH200 nodes should NOT run CMA-ES when their GPUs are idle.
Instead, they should run training jobs, and CMA-ES should run on CPU-only nodes.

Usage:
    # Run once to rebalance cluster
    python scripts/smart_work_router.py --rebalance

    # Run as daemon to continuously optimize
    python scripts/smart_work_router.py --daemon

    # Check current utilization report
    python scripts/smart_work_router.py --report
"""

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
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import node policies (graceful fallback if not available)
try:
    from app.coordination.node_policies import is_work_allowed, get_best_work_type
    HAS_POLICIES = True
except ImportError:
    HAS_POLICIES = False
    def is_work_allowed(node_id: str, work_type: str) -> bool:
        return True  # Allow all if policies not available
    def get_best_work_type(node_id: str, available: list[str]) -> str | None:
        return available[0] if available else None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

P2P_PORT = int(os.environ.get("RINGRIFT_P2P_PORT", "8770"))

# Node classification thresholds
GPU_HEAVY_TAGS = ['gh200', 'h100', 'h200', 'a100', '5090', '4090']
GPU_MEDIUM_TAGS = ['a10', '3090', '4080', '3080', '5080', '5070']
GPU_IDLE_THRESHOLD = 30.0   # GPU is idle if < 30%
CPU_BUSY_THRESHOLD = 70.0   # CPU is busy if > 70%


@dataclass
class NodeCapabilities:
    """Node capabilities and current state."""
    node_id: str
    host: str
    has_gpu: bool = False
    gpu_name: str = ""
    gpu_tier: str = "none"  # "heavy", "medium", "light", "none"
    cpu_count: int = 0
    memory_gb: int = 0
    # Current utilization
    gpu_percent: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    # Current work
    selfplay_jobs: int = 0
    training_jobs: int = 0
    # Detected external work (not tracked by P2P)
    cmaes_running: bool = False
    tournament_running: bool = False
    gauntlet_running: bool = False
    data_merge_running: bool = False
    # Classification
    is_gpu_idle: bool = False
    is_cpu_busy: bool = False
    recommended_work: list[str] = field(default_factory=list)


def http_get(url: str, timeout: int = 15) -> dict | None:
    """Make HTTP GET request."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.debug(f"HTTP GET failed for {url}: {e}")
        return None


def http_post(url: str, data: dict, timeout: int = 30) -> dict | None:
    """Make HTTP POST request."""
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.debug(f"HTTP POST failed for {url}: {e}")
        return None


def classify_gpu_tier(gpu_name: str) -> str:
    """Classify GPU into tiers."""
    gpu_lower = gpu_name.lower()
    if any(tag in gpu_lower for tag in GPU_HEAVY_TAGS):
        return "heavy"
    if any(tag in gpu_lower for tag in GPU_MEDIUM_TAGS):
        return "medium"
    if 'mps' in gpu_lower or 'apple' in gpu_lower:
        return "apple"
    if gpu_name:
        return "light"
    return "none"


def detect_external_work(host: str) -> dict[str, bool]:
    """Detect work running outside P2P tracking via SSH."""
    work = {
        'cmaes_running': False,
        'tournament_running': False,
        'gauntlet_running': False,
        'data_merge_running': False,
    }

    try:
        # Check for CMA-ES
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"ubuntu@{host}", "pgrep -c -f 'HeuristicAI.*json|cmaes' 2>/dev/null || echo 0"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and int(result.stdout.strip() or 0) > 0:
            work['cmaes_running'] = True

        # Check for tournaments
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"ubuntu@{host}", "pgrep -c -f 'run_model_elo_tournament' 2>/dev/null || echo 0"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and int(result.stdout.strip() or 0) > 0:
            work['tournament_running'] = True

        # Check for gauntlet (both baseline and two-stage)
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"ubuntu@{host}", "pgrep -c -f 'baseline_gauntlet|two_stage_gauntlet' 2>/dev/null || echo 0"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and int(result.stdout.strip() or 0) > 0:
            work['gauntlet_running'] = True

        # Check for data merge
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"ubuntu@{host}", "pgrep -c -f 'merge_game_dbs|export_training' 2>/dev/null || echo 0"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and int(result.stdout.strip() or 0) > 0:
            work['data_merge_running'] = True

    except Exception as e:
        logger.debug(f"Failed to detect external work on {host}: {e}")

    return work


def get_cluster_status() -> list[NodeCapabilities]:
    """Get comprehensive cluster status."""
    nodes = []

    # Get P2P status
    status = http_get(f"http://localhost:{P2P_PORT}/status")
    if not status:
        logger.error("Cannot get P2P status")
        return nodes

    # Process self
    self_info = status.get('self', {})
    if self_info:
        node = NodeCapabilities(
            node_id=self_info.get('node_id', 'self'),
            host=self_info.get('host', 'localhost'),
            has_gpu=self_info.get('has_gpu', False),
            gpu_name=self_info.get('gpu_name', ''),
            cpu_count=int(self_info.get('cpu_count', 0) or 0),
            memory_gb=int(self_info.get('memory_gb', 0) or 0),
            gpu_percent=float(self_info.get('gpu_percent', 0) or 0),
            cpu_percent=float(self_info.get('cpu_percent', 0) or 0),
            memory_percent=float(self_info.get('memory_percent', 0) or 0),
            selfplay_jobs=int(self_info.get('selfplay_jobs', 0) or 0),
            training_jobs=int(self_info.get('training_jobs', 0) or 0),
        )
        node.gpu_tier = classify_gpu_tier(node.gpu_name)
        node.is_gpu_idle = node.has_gpu and node.gpu_percent < GPU_IDLE_THRESHOLD
        node.is_cpu_busy = node.cpu_percent > CPU_BUSY_THRESHOLD
        nodes.append(node)

    # Process peers
    for peer_id, peer in status.get('peers', {}).items():
        if peer.get('retired'):
            continue

        node = NodeCapabilities(
            node_id=peer_id,
            host=peer.get('host', ''),
            has_gpu=peer.get('has_gpu', False),
            gpu_name=peer.get('gpu_name', ''),
            cpu_count=int(peer.get('cpu_count', 0) or 0),
            memory_gb=int(peer.get('memory_gb', 0) or 0),
            gpu_percent=float(peer.get('gpu_percent', 0) or 0),
            cpu_percent=float(peer.get('cpu_percent', 0) or 0),
            memory_percent=float(peer.get('memory_percent', 0) or 0),
            selfplay_jobs=int(peer.get('selfplay_jobs', 0) or 0),
            training_jobs=int(peer.get('training_jobs', 0) or 0),
        )
        node.gpu_tier = classify_gpu_tier(node.gpu_name)
        node.is_gpu_idle = node.has_gpu and node.gpu_percent < GPU_IDLE_THRESHOLD
        node.is_cpu_busy = node.cpu_percent > CPU_BUSY_THRESHOLD
        nodes.append(node)

    return nodes


def generate_utilization_report(nodes: list[NodeCapabilities], detect_external: bool = False) -> str:
    """Generate accurate utilization report."""
    lines = []
    lines.append("=" * 100)
    lines.append("CLUSTER UTILIZATION REPORT (Accurate)")
    lines.append("=" * 100)
    lines.append("")

    # Detect external work if requested
    if detect_external:
        lines.append("Detecting external work (CMA-ES, tournaments, etc.)...")
        for node in nodes:
            if node.host and node.host not in ('localhost', '127.0.0.1'):
                external = detect_external_work(node.host)
                node.cmaes_running = external['cmaes_running']
                node.tournament_running = external['tournament_running']
                node.gauntlet_running = external['gauntlet_running']
                node.data_merge_running = external['data_merge_running']
        lines.append("")

    # Categorize
    gpu_heavy = [n for n in nodes if n.gpu_tier == 'heavy']
    gpu_medium = [n for n in nodes if n.gpu_tier in ('medium', 'light')]
    cpu_only = [n for n in nodes if n.gpu_tier == 'none']
    apple = [n for n in nodes if n.gpu_tier == 'apple']

    def format_node_line(n: NodeCapabilities) -> str:
        work_types = []
        if n.training_jobs > 0:
            work_types.append(f"TRAIN({n.training_jobs})")
        if n.selfplay_jobs > 0:
            work_types.append(f"self({n.selfplay_jobs})")
        if n.cmaes_running:
            work_types.append("CMAES")
        if n.tournament_running:
            work_types.append("TOURN")
        if n.gauntlet_running:
            work_types.append("GAUNT")
        if n.data_merge_running:
            work_types.append("MERGE")

        work_str = ", ".join(work_types) if work_types else "IDLE"
        status = "OK" if (n.gpu_percent > 30 or not n.has_gpu) else "LOW GPU"

        return f"{n.node_id:<22} GPU:{n.gpu_percent:5.1f}% CPU:{n.cpu_percent:5.1f}% | {work_str:<35} | {status}"

    def print_section(title: str, node_list: list[NodeCapabilities]):
        if not node_list:
            return
        lines.append(f"\n{title} ({len(node_list)} nodes):")
        lines.append("-" * 100)

        # Sort by GPU utilization (low first to highlight problems)
        for n in sorted(node_list, key=lambda x: x.gpu_percent):
            lines.append(format_node_line(n))

    print_section("GPU-HEAVY (GH200/H100/A100) - Should prioritize TRAINING", gpu_heavy)
    print_section("GPU-MEDIUM (A10/Consumer) - Mixed workloads", gpu_medium)
    print_section("CPU-ONLY - Should run CMA-ES, Selfplay", cpu_only)
    print_section("APPLE SILICON - Light training, selfplay", apple)

    # Summary
    lines.append("")
    lines.append("=" * 100)
    lines.append("SUMMARY")
    lines.append("=" * 100)

    gpu_idle_heavy = [n for n in gpu_heavy if n.is_gpu_idle]
    gpu_busy_heavy = [n for n in gpu_heavy if not n.is_gpu_idle]

    lines.append(f"GPU-Heavy nodes with idle GPU: {len(gpu_idle_heavy)}/{len(gpu_heavy)} (PROBLEM if >0)")
    lines.append(f"GPU-Heavy nodes properly utilized: {len(gpu_busy_heavy)}/{len(gpu_heavy)}")

    # Identify misrouted work
    misrouted = [n for n in gpu_heavy if n.is_gpu_idle and n.is_cpu_busy]
    if misrouted:
        lines.append("")
        lines.append("MISROUTED WORK (GPU-heavy nodes running CPU work):")
        for n in misrouted:
            lines.append(f"  {n.node_id}: GPU idle ({n.gpu_percent:.0f}%) but CPU busy ({n.cpu_percent:.0f}%)")
            lines.append(f"    -> Should stop CMA-ES and start TRAINING instead")

    return "\n".join(lines)


def get_training_configs() -> list[tuple[str, int]]:
    """Get available training configurations."""
    configs = [
        ("square8", 2),
        ("square8", 3),
        ("square8", 4),
        ("hexagonal", 2),
        ("hexagonal", 3),
        ("hexagonal", 4),
        ("hex8", 2),
        ("hex8", 4),
        ("square19", 2),
        ("square19", 3),
    ]
    return configs


def trigger_training_on_node(node_id: str, board_type: str, num_players: int) -> bool:
    """Trigger training job targeting specific node."""
    # The P2P scheduler will route to appropriate node
    result = http_post(
        f"http://localhost:{P2P_PORT}/training/start",
        {
            "board_type": board_type,
            "num_players": num_players,
            "model_type": "nnue",
            "preferred_node": node_id,  # Hint to scheduler
        }
    )
    if result and result.get('success'):
        worker = result.get('worker', 'unknown')
        logger.info(f"Training {board_type}_{num_players}p started on {worker}")
        return True
    return False


def kill_cmaes_on_node(host: str) -> bool:
    """Stop CMA-ES processes on a node to free it for GPU work."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"ubuntu@{host}", "pkill -f 'HeuristicAI.*json' 2>/dev/null; pkill -f 'cmaes' 2>/dev/null; echo done"],
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to kill CMA-ES on {host}: {e}")
        return False


def kill_cpu_work_on_node(host: str, work_types: list[str] = None) -> dict:
    """Stop CPU-bound work on a node to free it for GPU work.

    Args:
        host: Node IP/hostname
        work_types: List of work types to kill. Options: 'cmaes', 'gauntlet', 'tournament', 'all'
                   Defaults to ['all']

    Returns:
        dict with 'success' and 'killed' counts per type
    """
    if work_types is None:
        work_types = ['all']

    # Build kill commands based on work types
    kill_cmds = []
    if 'all' in work_types or 'cmaes' in work_types:
        kill_cmds.extend([
            "pkill -9 -f 'HeuristicAI.*json'",
            "pkill -9 -f 'cmaes_distributed'",
            "pkill -9 -f 'run_cpu_cmaes'",
        ])
    if 'all' in work_types or 'gauntlet' in work_types:
        kill_cmds.extend([
            "pkill -9 -f 'two_stage_gauntlet'",
            "pkill -9 -f 'baseline_gauntlet'",
        ])
    if 'all' in work_types or 'tournament' in work_types:
        kill_cmds.extend([
            "pkill -9 -f 'run_model_elo_tournament'",
        ])

    # Build compound command
    cmd = "; ".join([f"{c} 2>/dev/null || true" for c in kill_cmds])
    cmd += "; echo done"

    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
             f"ubuntu@{host}", cmd],
            capture_output=True, text=True, timeout=20
        )
        success = "done" in result.stdout
        return {'success': success, 'host': host}
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout killing CPU work on {host}")
        return {'success': False, 'host': host, 'error': 'timeout'}
    except Exception as e:
        logger.error(f"Failed to kill CPU work on {host}: {e}")
        return {'success': False, 'host': host, 'error': str(e)}


def kill_cpu_work_parallel(nodes: list[NodeCapabilities], work_types: list[str] = None) -> int:
    """Kill CPU-bound work on multiple nodes in parallel.

    Returns number of successful kills.
    """
    import concurrent.futures

    hosts = [n.host for n in nodes if n.host and n.host not in ('localhost', '127.0.0.1')]
    if not hosts:
        return 0

    logger.info(f"Killing CPU work on {len(hosts)} nodes in parallel...")

    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(kill_cpu_work_on_node, host, work_types): host for host in hosts}
        for future in concurrent.futures.as_completed(futures, timeout=30):
            try:
                result = future.result()
                if result.get('success'):
                    success_count += 1
                    logger.debug(f"Killed CPU work on {result.get('host')}")
            except Exception as e:
                logger.debug(f"Kill failed: {e}")

    return success_count


def start_cpu_cmaes_on_node(host: str, board_type: str, num_players: int) -> bool:
    """Start CPU CMA-ES on a specific CPU-rich node."""
    try:
        cmd = f"cd ~/ringrift/ai-service && nohup python3 scripts/cmaes_distributed.py --board {board_type} --players {num_players} >> /tmp/cmaes.log 2>&1 &"
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"ubuntu@{host}", cmd],
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to start CPU CMA-ES on {host}: {e}")
        return False


def start_gpu_cmaes_on_node(host: str, board_type: str, num_players: int) -> bool:
    """Start GPU-accelerated CMA-ES on a GPU-rich node.

    Uses run_gpu_cmaes.py which is 10-100x faster than CPU CMA-ES.
    """
    try:
        cmd = f"cd ~/ringrift/ai-service && nohup python3 scripts/run_gpu_cmaes.py --board {board_type} --num-players {num_players} --generations 50 --population-size 20 --games-per-eval 50 >> /tmp/gpu_cmaes.log 2>&1 &"
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"ubuntu@{host}", cmd],
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to start GPU CMA-ES on {host}: {e}")
        return False


def wake_up_cpu_nodes(cpu_nodes: list[NodeCapabilities]) -> list[NodeCapabilities]:
    """Try to wake up/reconnect to CPU-rich nodes.

    Returns list of nodes that are now available.
    """
    available = []

    for node in cpu_nodes:
        if not node.host:
            continue

        # Try to reach the node
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                 f"ubuntu@{node.host}", "echo alive"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and "alive" in result.stdout:
                logger.info(f"CPU node {node.node_id} is reachable")
                available.append(node)
        except Exception:
            logger.debug(f"CPU node {node.node_id} not reachable")

    return available


def start_cmaes_on_node(host: str, board_type: str, num_players: int) -> bool:
    """Start CMA-ES on a specific CPU-rich node (alias for backward compat)."""
    return start_cpu_cmaes_on_node(host, board_type, num_players)


def rebalance_cluster(nodes: list[NodeCapabilities], dry_run: bool = False) -> int:
    """Rebalance work across the cluster.

    Strategy:
    1. GPU-heavy nodes with idle GPUs -> Training or GPU CMA-ES
    2. CPU-only nodes -> CPU CMA-ES, hybrid selfplay
    3. Before assigning CPU work to GPU nodes, try to wake up CPU nodes first
    4. GPU-heavy nodes running CPU-only CMA-ES -> Stop and start GPU work instead
    """
    changes = 0

    # Categorize nodes
    gpu_heavy = [n for n in nodes if n.gpu_tier == 'heavy']
    gpu_medium = [n for n in nodes if n.gpu_tier in ('medium', 'light')]
    cpu_only = [n for n in nodes if n.gpu_tier == 'none']

    # Find misrouted nodes (GPU-heavy but running CPU-only work)
    # Include nodes running any CPU-bound work: CMA-ES, gauntlets, or tournaments
    misrouted = [n for n in gpu_heavy if n.is_gpu_idle and n.is_cpu_busy and
                 (n.cmaes_running or n.gauntlet_running or n.tournament_running)]

    # Find GPU-heavy nodes with idle GPUs (not training)
    gpu_idle = [n for n in gpu_heavy if n.is_gpu_idle and n.training_jobs == 0]

    # Find CPU-only nodes that could take work
    cpu_available = [n for n in cpu_only if n.cpu_percent < 70]

    configs = get_training_configs()
    config_idx = 0

    logger.info(f"Cluster state: {len(gpu_idle)} GPU-idle heavy nodes, "
                f"{len(misrouted)} misrouted (CPU work on GPU nodes), "
                f"{len(cpu_available)} available CPU nodes")

    # STEP 1: Try to wake up and use CPU-only nodes for CPU work FIRST
    # Before using GPU nodes for CPU work, always try CPU nodes first
    if misrouted and cpu_available:
        logger.info("Attempting to move CPU work from GPU nodes to CPU nodes...")

        # Wake up CPU nodes
        awake_cpu = wake_up_cpu_nodes(cpu_available)

        for cpu_node in awake_cpu[:len(misrouted)]:
            if not cpu_node.host:
                continue

            board, players = configs[config_idx % len(configs)]
            config_idx += 1

            if dry_run:
                logger.info(f"[DRY RUN] Would start CPU CMA-ES on {cpu_node.node_id}")
                changes += 1
            else:
                logger.info(f"Starting CPU CMA-ES on {cpu_node.node_id} (freeing GPU nodes)")
                if start_cpu_cmaes_on_node(cpu_node.host, board, players):
                    changes += 1

    # STEP 2: Stop CPU-bound work on GPU nodes in parallel
    if misrouted:
        if dry_run:
            for node in misrouted[:6]:
                logger.info(f"[DRY RUN] Would stop CPU work on {node.node_id} and start GPU work")
                changes += 1
        else:
            # Kill all CPU-bound work types (CMA-ES, gauntlets, tournaments) in parallel
            logger.info(f"Stopping CPU work on {len(misrouted[:6])} misrouted GPU nodes...")
            killed = kill_cpu_work_parallel(misrouted[:6], work_types=['all'])
            changes += killed
            time.sleep(2)

    # STEP 3: Start GPU work on idle GPU-heavy nodes
    for node in gpu_idle[:6]:  # Max 6 per run
        if dry_run:
            logger.info(f"[DRY RUN] Would start GPU work on {node.node_id}")
            changes += 1
            continue

        board, players = configs[config_idx % len(configs)]
        config_idx += 1

        # Use policy to determine best work type
        best_work = get_best_work_type(node.node_id, ["training", "gpu_cmaes", "tournament"])

        if best_work == "training" and is_work_allowed(node.node_id, "training") and trigger_training_on_node(node.node_id, board, players):
            logger.info(f"Started training on {node.node_id} (policy: allowed)")
            changes += 1
            time.sleep(1)
            continue

        # Priority 2: GPU CMA-ES (10-100x faster than CPU)
        if node.host and is_work_allowed(node.node_id, "gpu_cmaes"):
            logger.info(f"Starting GPU CMA-ES on {node.node_id} (policy: allowed)")
            if start_gpu_cmaes_on_node(node.host, board, players):
                changes += 1
                time.sleep(1)

    # STEP 4: Ensure CPU-only nodes are utilized (if still available)
    remaining_cpu = [n for n in cpu_available if n.cpu_percent < 50]
    for node in remaining_cpu[:2]:
        if node.cmaes_running:
            continue  # Already running

        # Check policy allows CPU CMA-ES on this node
        if not is_work_allowed(node.node_id, "cpu_cmaes"):
            logger.debug(f"Skipping {node.node_id}: cpu_cmaes not allowed by policy")
            continue

        if dry_run:
            logger.info(f"[DRY RUN] Would start CPU CMA-ES on {node.node_id}")
            changes += 1
            continue

        if node.host:
            board, players = configs[config_idx % len(configs)]
            config_idx += 1

            logger.info(f"Starting CPU CMA-ES on available CPU node {node.node_id} (policy: allowed)")
            if start_cpu_cmaes_on_node(node.host, board, players):
                changes += 1

    return changes


def main():
    parser = argparse.ArgumentParser(description="Smart Work Router")
    parser.add_argument("--report", action="store_true", help="Show utilization report")
    parser.add_argument("--detect-external", action="store_true", help="Detect external work via SSH")
    parser.add_argument("--rebalance", action="store_true", help="Rebalance work across cluster")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300, help="Daemon check interval (seconds)")
    parser.add_argument("--quiet", action="store_true", help="Reduce log output (only show warnings)")
    parser.add_argument("--kill-misrouted", action="store_true", help="Kill CPU work on GPU-heavy nodes")
    parser.add_argument("--work-types", type=str, default="all", help="Work types to kill: all,cmaes,gauntlet,tournament")
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    if args.daemon:
        logger.info(f"Starting Smart Work Router daemon (interval: {args.interval}s)")
        while True:
            try:
                nodes = get_cluster_status()
                if nodes:
                    # Detect external work periodically
                    for node in nodes[:10]:  # Check top 10 nodes
                        if node.host and node.gpu_tier == 'heavy' and node.is_gpu_idle:
                            external = detect_external_work(node.host)
                            node.cmaes_running = external['cmaes_running']

                    changes = rebalance_cluster(nodes)
                    if changes > 0:
                        logger.info(f"Made {changes} rebalancing changes")
            except Exception as e:
                logger.error(f"Daemon error: {e}")

            time.sleep(args.interval)

    elif args.kill_misrouted:
        # Explicit kill of CPU work on GPU-heavy nodes
        nodes = get_cluster_status()
        if nodes:
            # Detect external work first
            logger.info("Detecting external work on GPU-heavy nodes...")
            gpu_heavy = [n for n in nodes if n.gpu_tier == 'heavy']
            for node in gpu_heavy:
                if node.host and node.host not in ('localhost', '127.0.0.1'):
                    external = detect_external_work(node.host)
                    node.cmaes_running = external['cmaes_running']
                    node.tournament_running = external['tournament_running']
                    node.gauntlet_running = external['gauntlet_running']

            # Find misrouted nodes
            misrouted = [n for n in gpu_heavy if n.is_gpu_idle and n.is_cpu_busy]
            if not misrouted:
                logger.info("No misrouted nodes found")
            else:
                work_types = args.work_types.split(',')
                logger.info(f"Found {len(misrouted)} misrouted nodes, killing work types: {work_types}")

                if args.dry_run:
                    for node in misrouted:
                        logger.info(f"[DRY RUN] Would kill {work_types} on {node.node_id}")
                else:
                    killed = kill_cpu_work_parallel(misrouted, work_types=work_types)
                    logger.info(f"Killed CPU work on {killed}/{len(misrouted)} nodes")

    elif args.report or (not args.rebalance and not args.kill_misrouted):
        nodes = get_cluster_status()
        if nodes:
            report = generate_utilization_report(nodes, detect_external=args.detect_external)
            print(report)

    elif args.rebalance:
        nodes = get_cluster_status()
        if nodes:
            # Always detect external work when rebalancing
            for node in nodes:
                if node.host and node.host not in ('localhost', '127.0.0.1'):
                    external = detect_external_work(node.host)
                    node.cmaes_running = external['cmaes_running']
                    node.tournament_running = external['tournament_running']

            changes = rebalance_cluster(nodes, dry_run=args.dry_run)
            logger.info(f"Rebalancing complete: {changes} changes")


if __name__ == "__main__":
    main()
