"""Cluster Dashboard - Unified status view.

Consolidates cluster_monitor.py, cluster_status.sh, monitor_*.sh into one tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NodeStatus:
    """Status of a single cluster node."""
    node_id: str
    host: str
    online: bool = False
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    role: str = "unknown"
    gpu_name: str = ""
    error: str = ""


@dataclass
class ClusterStatus:
    """Overall cluster status."""
    total_nodes: int = 0
    online_nodes: int = 0
    leader: str | None = None
    nodes: list[NodeStatus] = field(default_factory=list)
    total_selfplay_jobs: int = 0
    total_training_jobs: int = 0
    avg_gpu_util: float = 0.0


def get_node_status(host: str, port: int = 8770, timeout: int = 5) -> dict | None:
    """Get status from a single node's P2P orchestrator endpoint."""
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/status"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def get_cluster_status(entry_point: str = "localhost:8770") -> ClusterStatus:
    """Get status of entire cluster via P2P orchestrator.

    Args:
        entry_point: Host:port of any P2P orchestrator node.

    Returns:
        ClusterStatus with all node information.
    """
    host, port = entry_point.split(":") if ":" in entry_point else (entry_point, "8770")

    status = ClusterStatus()

    # Try to get status from entry point
    data = get_node_status(host, int(port))
    if not data:
        return status

    # Parse self info
    self_info = data.get("self", {})
    status.leader = data.get("leader_id") or data.get("effective_leader_id")

    # Add self node
    self_status = NodeStatus(
        node_id=self_info.get("node_id", "unknown"),
        host=self_info.get("host", host),
        online=True,
        gpu_percent=self_info.get("gpu_percent", 0),
        gpu_memory_percent=self_info.get("gpu_memory_percent", 0),
        cpu_percent=self_info.get("cpu_percent", 0),
        memory_percent=self_info.get("memory_percent", 0),
        disk_percent=self_info.get("disk_percent", 0),
        selfplay_jobs=self_info.get("selfplay_jobs", 0),
        training_jobs=self_info.get("training_jobs", 0),
        role=data.get("role", "follower"),
        gpu_name=self_info.get("gpu_name", ""),
    )
    status.nodes.append(self_status)

    # Add peer nodes
    peers = data.get("peers", {})
    for peer_id, peer_info in peers.items():
        # Check if peer is alive
        last_hb = peer_info.get("last_heartbeat", 0)
        import time
        is_alive = (time.time() - last_hb) < 180  # 3 min timeout

        peer_status = NodeStatus(
            node_id=peer_id,
            host=peer_info.get("host", ""),
            online=is_alive,
            gpu_percent=peer_info.get("gpu_percent", 0),
            gpu_memory_percent=peer_info.get("gpu_memory_percent", 0),
            cpu_percent=peer_info.get("cpu_percent", 0),
            memory_percent=peer_info.get("memory_percent", 0),
            disk_percent=peer_info.get("disk_percent", 0),
            selfplay_jobs=peer_info.get("selfplay_jobs", 0),
            training_jobs=peer_info.get("training_jobs", 0),
            role=peer_info.get("role", "follower"),
            gpu_name=peer_info.get("gpu_name", ""),
        )
        status.nodes.append(peer_status)

    # Calculate totals
    status.total_nodes = len(status.nodes)
    status.online_nodes = sum(1 for n in status.nodes if n.online)
    status.total_selfplay_jobs = sum(n.selfplay_jobs for n in status.nodes)
    status.total_training_jobs = sum(n.training_jobs for n in status.nodes)

    online_with_gpu = [n for n in status.nodes if n.online and n.gpu_name]
    if online_with_gpu:
        status.avg_gpu_util = sum(n.gpu_percent for n in online_with_gpu) / len(online_with_gpu)

    return status


def print_cluster_status(status: ClusterStatus, verbose: bool = False) -> None:
    """Print cluster status in a readable table format."""
    print(f"\n{'='*80}")
    print("  RINGRIFT CLUSTER STATUS")
    print(f"{'='*80}")
    print(f"  Nodes: {status.online_nodes}/{status.total_nodes} online")
    print(f"  Leader: {status.leader or 'None'}")
    print(f"  Selfplay Jobs: {status.total_selfplay_jobs}")
    print(f"  Training Jobs: {status.total_training_jobs}")
    print(f"  Avg GPU Util: {status.avg_gpu_util:.1f}%")
    print(f"{'='*80}\n")

    # Sort nodes: online first, then by node_id
    sorted_nodes = sorted(status.nodes, key=lambda n: (not n.online, n.node_id))

    # Print table header
    print(f"{'Node':<20} {'Status':<8} {'GPU%':<6} {'VRAM%':<6} {'CPU%':<6} {'Disk%':<6} {'Jobs':<8} {'Role':<10}")
    print("-" * 80)

    for node in sorted_nodes:
        status_str = "ONLINE" if node.online else "OFFLINE"

        jobs_str = f"{node.selfplay_jobs}sp"
        if node.training_jobs:
            jobs_str += f"+{node.training_jobs}tr"

        role_str = node.role.upper() if node.role == "leader" else node.role

        print(
            f"{node.node_id:<20} "
            f"{status_str:<8} "
            f"{node.gpu_percent:>5.1f} "
            f"{node.gpu_memory_percent:>5.1f} "
            f"{node.cpu_percent:>5.1f} "
            f"{node.disk_percent:>5.1f} "
            f"{jobs_str:<8} "
            f"{role_str:<10}"
        )

    print()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Cluster Dashboard")
    parser.add_argument("--host", default="localhost", help="P2P orchestrator host")
    parser.add_argument("--port", type=int, default=8770, help="P2P orchestrator port")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    entry = f"{args.host}:{args.port}"
    status = get_cluster_status(entry)

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(status), indent=2))
    else:
        print_cluster_status(status, verbose=args.verbose)


if __name__ == "__main__":
    main()
