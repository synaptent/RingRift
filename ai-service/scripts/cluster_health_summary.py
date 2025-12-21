#!/usr/bin/env python3
"""Cluster Health Summary CLI - Lane 5 Observability (December 2025).

Provides a concise overview of cluster health for daily operations:
- Node status (online/offline, GPU utilization)
- Training pipeline status (games pending, models in progress)
- Data quality metrics (parity pass rate, canonical gate status)
- Recent performance (Elo trends, win rates)

Usage:
    # Quick summary (default)
    python scripts/cluster_health_summary.py

    # Detailed output
    python scripts/cluster_health_summary.py --verbose

    # JSON output for automation
    python scripts/cluster_health_summary.py --json

    # Monitor mode (refresh every 60s)
    python scripts/cluster_health_summary.py --watch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# SSH options for cluster access
SSH_OPTS = "-o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes"

# Known cluster nodes
GH200_NODES = [
    "lambda-gh200-a", "lambda-gh200-c", "lambda-gh200-d", "lambda-gh200-e",
    "lambda-gh200-f", "lambda-gh200-g", "lambda-gh200-h", "lambda-gh200-i",
    "lambda-gh200-k", "lambda-gh200-l",
]
H100_NODES = ["lambda-h100", "lambda-2xh100"]


@dataclass
class NodeHealth:
    """Health status of a single node."""
    hostname: str
    online: bool = False
    gpu_util: float = 0.0
    gpu_memory_pct: float = 0.0
    active_processes: int = 0
    process_types: list[str] = field(default_factory=list)


@dataclass
class PipelineHealth:
    """Health status of the training pipeline."""
    games_pending_training: int = 0
    training_in_progress: bool = False
    training_config: str = ""
    last_training_time: str = ""
    models_in_staging: int = 0
    models_in_production: int = 0


@dataclass
class DataQuality:
    """Data quality metrics."""
    parity_pass_rate: float = 1.0
    parity_games_checked: int = 0
    canonical_gate_passed: bool = True
    canonical_violations: int = 0


@dataclass
class PerformanceMetrics:
    """Recent performance metrics."""
    best_elo: dict[str, float] = field(default_factory=dict)
    elo_trends: dict[str, float] = field(default_factory=dict)
    recent_win_rates: dict[str, float] = field(default_factory=dict)


@dataclass
class ClusterHealth:
    """Complete cluster health summary."""
    timestamp: str = ""
    nodes: list[NodeHealth] = field(default_factory=list)
    pipeline: PipelineHealth = field(default_factory=PipelineHealth)
    data_quality: DataQuality = field(default_factory=DataQuality)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Summary stats
    nodes_online: int = 0
    nodes_total: int = 0
    nodes_active: int = 0
    avg_gpu_util: float = 0.0
    health_status: str = "unknown"  # healthy, degraded, critical


async def _ssh_command(hostname: str, command: str) -> tuple[int, str]:
    """Execute command on remote host via SSH."""
    user = "ubuntu"
    ssh_cmd = f"ssh {SSH_OPTS} {user}@{hostname} '{command}'"

    try:
        proc = await asyncio.create_subprocess_shell(
            ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(
            proc.communicate(),
            timeout=10.0
        )
        return (proc.returncode or 0, stdout.decode().strip())
    except asyncio.TimeoutError:
        return (-1, "")
    except Exception:
        return (-1, "")


async def check_node_health(hostname: str) -> NodeHealth:
    """Check health of a single node."""
    health = NodeHealth(hostname=hostname)

    # Check if online
    rc, _ = await _ssh_command(hostname, "echo OK")
    if rc != 0:
        return health

    health.online = True

    # Get GPU utilization
    rc, output = await _ssh_command(
        hostname,
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits 2>/dev/null | head -1"
    )
    if rc == 0 and output:
        try:
            parts = output.split(",")
            if len(parts) >= 3:
                health.gpu_util = float(parts[0].strip())
                mem_used = float(parts[1].strip())
                mem_total = float(parts[2].strip())
                health.gpu_memory_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
        except (ValueError, IndexError):
            pass

    # Count active processes
    rc, output = await _ssh_command(
        hostname,
        "ps aux | grep -E 'train|selfplay|unified_ai_loop|tournament' | grep -v grep | wc -l"
    )
    if rc == 0:
        try:
            health.active_processes = int(output)
        except ValueError:
            pass

    # Identify process types
    rc, output = await _ssh_command(
        hostname,
        "ps aux | grep -E 'train|selfplay|unified_ai_loop|tournament' | grep -v grep | "
        "awk '{print $11}' | xargs -I{} basename {} 2>/dev/null | sort -u | head -5"
    )
    if rc == 0 and output:
        health.process_types = [p for p in output.split("\n") if p][:5]

    return health


async def get_pipeline_health() -> PipelineHealth:
    """Get training pipeline health from state files."""
    health = PipelineHealth()

    # Check for state files locally (these are synced from cluster)
    state_file = Path("data/unified_loop_state.json")
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)

            health.training_in_progress = state.get("training_in_progress", False)
            health.training_config = state.get("training_config", "")

            # Count games pending
            configs = state.get("configs", {})
            for cfg in configs.values():
                health.games_pending_training += cfg.get("games_since_training", 0)

            if state.get("training_started_at"):
                ts = state["training_started_at"]
                if isinstance(ts, float):
                    health.last_training_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        except (json.JSONDecodeError, KeyError):
            pass

    return health


def get_data_quality() -> DataQuality:
    """Get data quality metrics from canonical validation logs."""
    quality = DataQuality()

    # Check parity validation log
    parity_log = Path("logs/parity_validation.log")
    if parity_log.exists():
        try:
            with open(parity_log) as f:
                lines = f.readlines()[-100:]  # Last 100 lines

            passed = sum(1 for l in lines if "PASS" in l)
            failed = sum(1 for l in lines if "FAIL" in l)
            total = passed + failed
            if total > 0:
                quality.parity_pass_rate = passed / total
                quality.parity_games_checked = total
        except Exception:
            pass

    # Check canonical gate status
    gate_log = Path("logs/canonical_gate.log")
    if gate_log.exists():
        try:
            with open(gate_log) as f:
                lines = f.readlines()[-50:]

            violations = sum(1 for l in lines if "VIOLATION" in l or "non-canonical" in l.lower())
            quality.canonical_violations = violations
            quality.canonical_gate_passed = violations == 0
        except Exception:
            pass

    return quality


def get_performance_metrics() -> PerformanceMetrics:
    """Get recent performance metrics from ELO tracking."""
    metrics = PerformanceMetrics()

    # Check ELO state file
    elo_file = Path("data/elo_state.json")
    if elo_file.exists():
        try:
            with open(elo_file) as f:
                elo_state = json.load(f)

            ratings = elo_state.get("ratings", {})
            for config, rating in ratings.items():
                if isinstance(rating, dict):
                    metrics.best_elo[config] = rating.get("elo", 1500)
                else:
                    metrics.best_elo[config] = rating
        except (json.JSONDecodeError, KeyError):
            pass

    return metrics


async def collect_cluster_health(verbose: bool = False) -> ClusterHealth:
    """Collect complete cluster health summary."""
    health = ClusterHealth(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    all_nodes = GH200_NODES + H100_NODES
    health.nodes_total = len(all_nodes)

    # Check all nodes in parallel
    tasks = [check_node_health(hostname) for hostname in all_nodes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, NodeHealth):
            health.nodes.append(result)
            if result.online:
                health.nodes_online += 1
                if result.active_processes > 0:
                    health.nodes_active += 1

    # Calculate average GPU utilization
    online_nodes = [n for n in health.nodes if n.online]
    if online_nodes:
        health.avg_gpu_util = sum(n.gpu_util for n in online_nodes) / len(online_nodes)

    # Get other health metrics
    health.pipeline = await get_pipeline_health()
    health.data_quality = get_data_quality()
    health.performance = get_performance_metrics()

    # Determine overall health status
    if health.nodes_online == 0:
        health.health_status = "critical"
    elif health.nodes_online < health.nodes_total / 2:
        health.health_status = "degraded"
    elif not health.data_quality.canonical_gate_passed:
        health.health_status = "degraded"
    elif health.avg_gpu_util < 10 and health.nodes_active == 0:
        health.health_status = "idle"
    else:
        health.health_status = "healthy"

    return health


def format_health_summary(health: ClusterHealth, verbose: bool = False) -> str:
    """Format health summary for terminal display."""
    lines = []

    # Header
    status_color = {
        "healthy": "\033[92m",  # Green
        "degraded": "\033[93m",  # Yellow
        "critical": "\033[91m",  # Red
        "idle": "\033[94m",  # Blue
    }.get(health.health_status, "")
    reset = "\033[0m"

    lines.append("=" * 60)
    lines.append(f"CLUSTER HEALTH SUMMARY - {health.timestamp}")
    lines.append(f"Status: {status_color}{health.health_status.upper()}{reset}")
    lines.append("=" * 60)

    # Node summary
    lines.append("")
    lines.append(f"NODES: {health.nodes_online}/{health.nodes_total} online, "
                 f"{health.nodes_active} active, "
                 f"Avg GPU: {health.avg_gpu_util:.1f}%")

    if verbose:
        lines.append("-" * 40)
        for node in sorted(health.nodes, key=lambda n: n.hostname):
            status = "ONLINE" if node.online else "OFFLINE"
            if node.online:
                procs = ", ".join(node.process_types[:3]) or "idle"
                lines.append(f"  {node.hostname}: {status} "
                            f"(GPU: {node.gpu_util:.0f}%, {procs})")
            else:
                lines.append(f"  {node.hostname}: {status}")

    # Pipeline summary
    lines.append("")
    lines.append("PIPELINE:")
    training_status = "IN PROGRESS" if health.pipeline.training_in_progress else "idle"
    lines.append(f"  Training: {training_status}"
                 f"{f' ({health.pipeline.training_config})' if health.pipeline.training_config else ''}")
    lines.append(f"  Games pending: {health.pipeline.games_pending_training}")

    # Data quality
    lines.append("")
    lines.append("DATA QUALITY:")
    parity_status = "PASS" if health.data_quality.parity_pass_rate >= 0.99 else "CHECK"
    lines.append(f"  Parity: {health.data_quality.parity_pass_rate:.1%} "
                 f"({health.data_quality.parity_games_checked} games)")
    gate_status = "PASS" if health.data_quality.canonical_gate_passed else "VIOLATIONS"
    lines.append(f"  Canonical gate: {gate_status}")

    # Performance
    if health.performance.best_elo:
        lines.append("")
        lines.append("BEST ELO:")
        for config, elo in sorted(health.performance.best_elo.items()):
            lines.append(f"  {config}: {elo:.0f}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_json(health: ClusterHealth) -> str:
    """Format health summary as JSON."""
    return json.dumps({
        "timestamp": health.timestamp,
        "status": health.health_status,
        "nodes": {
            "online": health.nodes_online,
            "total": health.nodes_total,
            "active": health.nodes_active,
            "avg_gpu_util": health.avg_gpu_util,
            "details": [
                {
                    "hostname": n.hostname,
                    "online": n.online,
                    "gpu_util": n.gpu_util,
                    "processes": n.process_types,
                }
                for n in health.nodes
            ]
        },
        "pipeline": {
            "training_in_progress": health.pipeline.training_in_progress,
            "training_config": health.pipeline.training_config,
            "games_pending": health.pipeline.games_pending_training,
        },
        "data_quality": {
            "parity_pass_rate": health.data_quality.parity_pass_rate,
            "parity_games_checked": health.data_quality.parity_games_checked,
            "canonical_gate_passed": health.data_quality.canonical_gate_passed,
        },
        "performance": {
            "best_elo": health.performance.best_elo,
        },
    }, indent=2)


async def main_async(args):
    """Async main function."""
    while True:
        health = await collect_cluster_health(verbose=args.verbose)

        if args.json:
            print(format_json(health))
        else:
            # Clear screen in watch mode
            if args.watch:
                print("\033[2J\033[H", end="")
            print(format_health_summary(health, verbose=args.verbose))

        if not args.watch:
            break

        await asyncio.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(
        description="Cluster health summary for daily operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed node status",
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "-w", "--watch",
        action="store_true",
        help="Monitor mode - refresh periodically",
    )
    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=60,
        help="Refresh interval in seconds (default: 60)",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
