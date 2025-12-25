#!/usr/bin/env python3
"""Cluster Health CLI - Command line interface for cluster management.

Provides commands to inspect and manage the cluster health daemon.

Usage:
    # Show cluster status
    python scripts/cluster_health_cli.py status

    # Show all nodes with health
    python scripts/cluster_health_cli.py nodes --all

    # Force recovery on a node
    python scripts/cluster_health_cli.py recover lambda-gh200-s

    # Deploy SSH key to a node
    python scripts/cluster_health_cli.py deploy-key lambda-gh200-s

    # Restart P2P on all nodes
    python scripts/cluster_health_cli.py restart-p2p --all

    # Start selfplay on underutilized nodes
    python scripts/cluster_health_cli.py optimize

    # Watch mode
    python scripts/cluster_health_cli.py watch --interval 30
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add ai-service to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.providers import Provider
from app.coordination.health_check_orchestrator import (
    HealthCheckOrchestrator,
    NodeHealthState,
)
from app.coordination.recovery_orchestrator import RecoveryOrchestrator, RecoveryAction
from app.coordination.utilization_optimizer import UtilizationOptimizer


def format_health_state(state: NodeHealthState) -> str:
    """Format health state with color/emoji."""
    symbols = {
        NodeHealthState.HEALTHY: "✓ HEALTHY",
        NodeHealthState.DEGRADED: "◐ DEGRADED",
        NodeHealthState.UNHEALTHY: "✗ UNHEALTHY",
        NodeHealthState.OFFLINE: "⊘ OFFLINE",
        NodeHealthState.PROVIDER_DOWN: "⊗ PROVIDER_DOWN",
        NodeHealthState.RETIRED: "◯ RETIRED",
    }
    return symbols.get(state, str(state))


async def cmd_status(args):
    """Show cluster status summary."""
    orchestrator = HealthCheckOrchestrator()

    print("Discovering instances...")
    await orchestrator.run_full_health_check()

    summary = await orchestrator.get_cluster_health()

    print()
    print("=" * 60)
    print("CLUSTER HEALTH STATUS")
    print("=" * 60)
    print(f"Timestamp: {summary.timestamp.isoformat()}")
    print()
    print(f"{'Total Nodes:':<20} {summary.total_nodes}")
    print(f"{'Healthy:':<20} {summary.healthy}")
    print(f"{'Degraded:':<20} {summary.degraded}")
    print(f"{'Unhealthy:':<20} {summary.unhealthy}")
    print(f"{'Offline:':<20} {summary.offline}")
    print(f"{'Retired:':<20} {summary.retired}")
    print()

    active = summary.total_nodes - summary.retired
    if active > 0:
        avail = summary.healthy + summary.degraded
        pct = avail / active * 100
        print(f"{'Availability:':<20} {pct:.1f}% ({avail}/{active} nodes)")

    print()
    print(f"{'Total GPUs:':<20} {summary.total_gpus} ({summary.available_gpus} available)")
    print(f"{'Total CPU Cores:':<20} {summary.total_cpu_cores} ({summary.available_cpu_cores} available)")
    print(f"{'Hourly Cost:':<20} ${summary.hourly_cost:.2f}")

    print()
    print("By Provider:")
    for provider, stats in sorted(summary.by_provider.items()):
        print(
            f"  {provider:<12} {stats['total']:>3} nodes, "
            f"{stats['healthy']:>2} healthy, {stats['available']:>2} available"
        )

    await orchestrator.stop()


async def cmd_nodes(args):
    """Show all nodes with health details."""
    orchestrator = HealthCheckOrchestrator()

    print("Checking nodes...")
    await orchestrator.run_full_health_check()

    # Filter nodes
    nodes = []
    for node_id, health in sorted(orchestrator.node_health.items()):
        if args.offline_only and health.state != NodeHealthState.OFFLINE:
            continue
        if args.unhealthy_only and health.state not in (
            NodeHealthState.UNHEALTHY,
            NodeHealthState.OFFLINE,
        ):
            continue
        if not args.all and health.state == NodeHealthState.RETIRED:
            continue
        if args.provider and health.provider and health.provider.value != args.provider:
            continue
        nodes.append((node_id, health))

    print()
    print(f"{'Node':<20} {'Provider':<10} {'State':<15} {'GPU%':>5} {'CPU%':>5} {'Mem%':>5}")
    print("-" * 70)

    for node_id, health in nodes:
        provider = health.provider.value if health.provider else "?"
        state = format_health_state(health.state)
        gpu = f"{health.gpu_percent:.0f}" if health.gpu_percent else "-"
        cpu = f"{health.cpu_percent:.0f}" if health.cpu_percent else "-"
        mem = f"{health.memory_percent:.0f}" if health.memory_percent else "-"

        print(f"{node_id:<20} {provider:<10} {state:<15} {gpu:>5} {cpu:>5} {mem:>5}")

    print()
    print(f"Total: {len(nodes)} nodes")

    await orchestrator.stop()


async def cmd_recover(args):
    """Attempt recovery on a node."""
    orchestrator = HealthCheckOrchestrator()
    recovery = RecoveryOrchestrator(health_orchestrator=orchestrator)

    print(f"Running health check for {args.node}...")
    await orchestrator.run_full_health_check()

    health = orchestrator.get_node_health(args.node)
    if not health:
        # Try partial match
        for nid in orchestrator.node_health:
            if args.node.lower() in nid.lower():
                args.node = nid
                health = orchestrator.node_health[nid]
                break

    if not health:
        print(f"Node '{args.node}' not found")
        await orchestrator.stop()
        return

    print(f"Current state: {format_health_state(health.state)}")

    if health.state == NodeHealthState.HEALTHY:
        print("Node is healthy, no recovery needed")
        await orchestrator.stop()
        return

    # Force specific action if provided
    force_action = None
    if args.action:
        try:
            force_action = RecoveryAction(args.action)
        except ValueError:
            print(f"Invalid action: {args.action}")
            print(f"Valid actions: {[a.value for a in RecoveryAction]}")
            await orchestrator.stop()
            return

    print(f"Attempting recovery on {args.node}...")
    result = await recovery.attempt_recovery(args.node, force_action=force_action)

    print()
    print(f"Action: {result.action.value}")
    print(f"Success: {'Yes' if result.success else 'No'}")
    print(f"Message: {result.message}")

    if result.next_action:
        print(f"Next action: {result.next_action.value}")

    await orchestrator.stop()


async def cmd_restart_p2p(args):
    """Restart P2P daemon on nodes."""
    orchestrator = HealthCheckOrchestrator()

    print("Discovering nodes...")
    await orchestrator.run_full_health_check()

    # Determine target nodes
    if args.all:
        targets = list(orchestrator.node_health.keys())
    else:
        targets = args.nodes

    if not targets:
        print("No nodes specified. Use --all or provide node names.")
        await orchestrator.stop()
        return

    print(f"Restarting P2P on {len(targets)} nodes...")

    from app.coordination.recovery_orchestrator import RecoveryOrchestrator

    recovery = RecoveryOrchestrator(health_orchestrator=orchestrator)

    for node_id in targets:
        health = orchestrator.get_node_health(node_id)
        if not health or not health.instance:
            print(f"  {node_id}: not found")
            continue

        if not health.ssh_healthy:
            print(f"  {node_id}: SSH not reachable")
            continue

        result = await recovery.attempt_recovery(node_id, force_action=RecoveryAction.RESTART_P2P)
        status = "OK" if result.success else "FAILED"
        print(f"  {node_id}: {status}")

    await orchestrator.stop()


async def cmd_deploy_key(args):
    """Deploy SSH key to a node."""
    # Read public key
    key_path = Path(args.key_file).expanduser()
    if not key_path.exists():
        print(f"Key file not found: {key_path}")
        return

    with open(key_path) as f:
        public_key = f.read().strip()

    if not public_key.startswith(("ssh-rsa", "ssh-ed25519", "ecdsa-")):
        print("Invalid public key format")
        return

    orchestrator = HealthCheckOrchestrator()
    recovery = RecoveryOrchestrator(health_orchestrator=orchestrator)

    print("Discovering nodes...")
    await orchestrator.run_full_health_check()

    print(f"Deploying key to {args.node}...")
    success = await recovery.deploy_ssh_key(args.node, public_key)

    if success:
        print("Key deployed successfully")
    else:
        print("Key deployment failed")

    await orchestrator.stop()


async def cmd_optimize(args):
    """Run utilization optimization."""
    orchestrator = HealthCheckOrchestrator()
    optimizer = UtilizationOptimizer(health_orchestrator=orchestrator)

    print("Running health check...")
    await orchestrator.run_full_health_check()

    print("Finding underutilized nodes...")
    underutilized = await optimizer.get_underutilized_nodes()

    if not underutilized:
        print("No underutilized nodes found")
        await orchestrator.stop()
        return

    print(f"Found {len(underutilized)} underutilized nodes")

    if args.dry_run:
        print("\nDry run - would spawn jobs on:")
        for node_id, health in underutilized:
            board = optimizer._select_board_for_node(health)
            print(f"  {node_id}: {board.value}")
        await orchestrator.stop()
        return

    print("\nSpawning selfplay jobs...")
    results = await optimizer.optimize_cluster()

    print()
    for result in results:
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.node_id}: {result.message}")

    successful = sum(1 for r in results if r.success)
    print(f"\nSpawned {successful}/{len(results)} jobs")

    await orchestrator.stop()


async def cmd_watch(args):
    """Watch cluster status continuously."""
    import os

    orchestrator = HealthCheckOrchestrator()

    try:
        while True:
            # Clear screen
            os.system("clear" if os.name == "posix" else "cls")

            await orchestrator.run_full_health_check()
            summary = await orchestrator.get_cluster_health()

            print("=" * 60)
            print(f"CLUSTER STATUS - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)

            active = summary.total_nodes - summary.retired
            if active > 0:
                avail = summary.healthy + summary.degraded
                pct = avail / active * 100
                print(f"Availability: {pct:.1f}% ({avail}/{active} nodes)")

            print(
                f"Healthy: {summary.healthy} | "
                f"Degraded: {summary.degraded} | "
                f"Unhealthy: {summary.unhealthy} | "
                f"Offline: {summary.offline}"
            )
            print(f"GPUs: {summary.available_gpus}/{summary.total_gpus} available")
            print(f"Cost: ${summary.hourly_cost:.2f}/hr")

            print()
            print(f"{'Node':<20} {'State':<12} {'GPU%':>5} {'CPU%':>5}")
            print("-" * 50)

            for node_id, health in sorted(orchestrator.node_health.items()):
                if health.state == NodeHealthState.RETIRED:
                    continue
                state = health.state.value[:10]
                gpu = f"{health.gpu_percent:.0f}" if health.gpu_percent else "-"
                cpu = f"{health.cpu_percent:.0f}" if health.cpu_percent else "-"
                print(f"{node_id:<20} {state:<12} {gpu:>5} {cpu:>5}")

            print()
            print(f"(Refreshing every {args.interval}s, Ctrl+C to exit)")

            await asyncio.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        await orchestrator.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Cluster Health CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status
    status_parser = subparsers.add_parser("status", help="Show cluster status")
    status_parser.set_defaults(func=cmd_status)

    # nodes
    nodes_parser = subparsers.add_parser("nodes", help="List nodes with health")
    nodes_parser.add_argument("--all", "-a", action="store_true", help="Include retired nodes")
    nodes_parser.add_argument("--offline-only", action="store_true", help="Show only offline nodes")
    nodes_parser.add_argument("--unhealthy-only", action="store_true", help="Show only unhealthy nodes")
    nodes_parser.add_argument("--provider", "-p", help="Filter by provider")
    nodes_parser.set_defaults(func=cmd_nodes)

    # recover
    recover_parser = subparsers.add_parser("recover", help="Attempt recovery on node")
    recover_parser.add_argument("node", help="Node to recover")
    recover_parser.add_argument("--action", help="Force specific action")
    recover_parser.set_defaults(func=cmd_recover)

    # restart-p2p
    p2p_parser = subparsers.add_parser("restart-p2p", help="Restart P2P daemon")
    p2p_parser.add_argument("nodes", nargs="*", help="Nodes to restart")
    p2p_parser.add_argument("--all", action="store_true", help="Restart on all nodes")
    p2p_parser.set_defaults(func=cmd_restart_p2p)

    # deploy-key
    key_parser = subparsers.add_parser("deploy-key", help="Deploy SSH key")
    key_parser.add_argument("node", help="Target node")
    key_parser.add_argument("--key-file", default="~/.ssh/id_cluster.pub", help="Public key file")
    key_parser.set_defaults(func=cmd_deploy_key)

    # optimize
    opt_parser = subparsers.add_parser("optimize", help="Run utilization optimization")
    opt_parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be done")
    opt_parser.set_defaults(func=cmd_optimize)

    # watch
    watch_parser = subparsers.add_parser("watch", help="Watch cluster status")
    watch_parser.add_argument("--interval", type=int, default=30, help="Refresh interval")
    watch_parser.set_defaults(func=cmd_watch)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Run command
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
