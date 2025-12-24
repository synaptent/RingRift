#!/usr/bin/env python3
"""Example usage of ClusterMonitor for programmatic access.

This demonstrates how to integrate cluster monitoring into your own scripts
for alerts, automation, and custom reporting.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.distributed.cluster_monitor import ClusterMonitor


def main():
    """Demonstrate various ClusterMonitor features."""

    print("=" * 80)
    print("ClusterMonitor Example Usage")
    print("=" * 80)

    # Initialize monitor
    monitor = ClusterMonitor(ssh_timeout=10)

    print("\n1. Quick cluster health check")
    print("-" * 80)
    status = monitor.get_cluster_status(
        include_game_counts=True,
        include_training_status=True,
        include_disk_usage=True,
    )

    print(f"Active nodes: {status.active_nodes}/{status.total_nodes}")
    print(f"Total games: {status.total_games:,}")
    print(f"Nodes training: {status.nodes_training}")
    print(f"Avg disk usage: {status.avg_disk_usage:.1f}%")

    print("\n2. Check for low disk space alerts")
    print("-" * 80)
    disk_threshold = 85.0
    for host_name, node in status.nodes.items():
        if node.reachable and node.disk_usage_percent > disk_threshold:
            print(f"⚠️  {host_name}: {node.disk_usage_percent:.1f}% disk usage "
                  f"({node.disk_free_gb:.0f}GB free)")

    print("\n3. List active training nodes")
    print("-" * 80)
    for host_name, node in status.nodes.items():
        if node.training_active:
            print(f"🏋️  {host_name}: {len(node.training_processes)} processes")
            for proc in node.training_processes[:3]:  # Show first 3
                print(f"    - PID {proc['pid']}: {proc['command'][:60]}...")

    print("\n4. Game distribution across nodes")
    print("-" * 80)
    nodes_with_games = [
        (name, node.total_games)
        for name, node in status.nodes.items()
        if node.total_games > 0
    ]
    nodes_with_games.sort(key=lambda x: x[1], reverse=True)

    for host_name, game_count in nodes_with_games[:10]:  # Top 10
        print(f"{host_name:25} {game_count:>12,} games")

    print("\n5. Check specific node in detail")
    print("-" * 80)
    # Check one node with all details
    active_hosts = monitor.get_active_hosts()
    if active_hosts:
        test_host = active_hosts[0]
        print(f"Querying {test_host}...")

        node = monitor.get_node_status(
            test_host,
            include_game_counts=True,
            include_training_status=True,
            include_disk_usage=True,
            include_sync_status=False,
        )

        if node.reachable:
            print(f"  Status: ONLINE")
            print(f"  Response: {node.response_time_ms:.0f}ms")
            print(f"  Role: {node.role}")
            print(f"  GPU: {node.gpu}")
            print(f"  Games: {node.total_games:,}")
            if node.game_counts:
                print(f"  Breakdown:")
                for config, count in sorted(node.game_counts.items()):
                    print(f"    {config}: {count:,}")
            print(f"  Disk: {node.disk_usage_percent:.1f}% "
                  f"({node.disk_free_gb:.0f}GB free)")
            print(f"  Training: {'YES' if node.training_active else 'NO'}")
        else:
            print(f"  Status: OFFLINE ({node.error})")

    print("\n6. Detect unreachable nodes")
    print("-" * 80)
    offline_nodes = [
        name for name, node in status.nodes.items()
        if not node.reachable
    ]
    if offline_nodes:
        print(f"⚠️  {len(offline_nodes)} unreachable nodes:")
        for host in offline_nodes:
            error = status.nodes[host].error or "unknown error"
            print(f"  - {host}: {error}")
    else:
        print("✅ All nodes reachable!")

    print("\n7. Full dashboard view")
    print("-" * 80)
    monitor.print_dashboard(status)


if __name__ == "__main__":
    main()
