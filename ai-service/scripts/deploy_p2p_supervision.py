#!/usr/bin/env python3
"""
Deploy P2P supervision to cluster nodes.

This script:
1. Copies p2p_supervisor.py to each node
2. Adds cron @reboot entry for auto-start
3. Kills existing P2P processes
4. Starts supervised P2P
5. Verifies it's running

Usage:
    # Deploy to all nodes
    python scripts/deploy_p2p_supervision.py

    # Deploy to specific nodes
    python scripts/deploy_p2p_supervision.py --nodes lambda-gh200-1,lambda-gh200-2

    # Dry run
    python scripts/deploy_p2p_supervision.py --dry-run

    # Skip cron setup (just restart P2P)
    python scripts/deploy_p2p_supervision.py --skip-cron
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.ssh import SSHClient, SSHConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


async def deploy_to_node(
    node_name: str,
    node_config: dict,
    skip_cron: bool,
    dry_run: bool,
) -> Tuple[str, bool, str]:
    """Deploy P2P supervision to a single node."""
    try:
        # Skip coordinator nodes
        if node_config.get('role') == 'coordinator' and node_name in ['local-mac', 'mac-studio']:
            return (node_name, True, "SKIPPED: Coordinator node")

        # Skip if P2P is disabled
        if not node_config.get('p2p_enabled', True):
            return (node_name, True, "SKIPPED: P2P disabled")

        # Skip if node is not ready
        if node_config.get('status') != 'ready':
            return (node_name, False, f"SKIPPED: Status is {node_config.get('status')}")

        # Create SSH client
        logger.info(f"[{node_name}] Connecting...")
        ssh_config = SSHConfig(
            host=node_config.get('ssh_host', ''),
            port=node_config.get('ssh_port', 22),
            user=node_config.get('ssh_user', 'root'),
            key_path=node_config.get('ssh_key'),
            tailscale_ip=node_config.get('tailscale_ip'),
            work_dir=node_config.get('ringrift_path', '~/ringrift/ai-service'),
        )
        client = SSHClient(ssh_config)

        # Test connection
        test_result = await client.run_async("echo connected", timeout=10)
        if test_result.returncode != 0:
            return (node_name, False, f"Connection failed: {test_result.stderr}")

        node_path = node_config.get('ringrift_path', '~/ringrift/ai-service')
        venv_activate = node_config.get('venv_activate', 'source venv/bin/activate')

        if dry_run:
            return (node_name, True, f"DRY-RUN: Would deploy supervision to {node_path}")

        # Step 1: Ensure scripts directory exists
        await client.run_async(f"mkdir -p {node_path}/scripts {node_path}/logs", timeout=10)

        # Step 2: Kill existing P2P processes
        logger.info(f"[{node_name}] Stopping existing P2P...")
        await client.run_async("pkill -f p2p_orchestrator || true", timeout=10)
        await client.run_async("pkill -f p2p_supervisor || true", timeout=10)
        await asyncio.sleep(2)

        # Step 3: Add cron entry if not skipped
        if not skip_cron:
            logger.info(f"[{node_name}] Setting up cron @reboot...")

            # Remove any existing P2P cron entries
            await client.run_async(
                "crontab -l 2>/dev/null | grep -v p2p_supervisor | grep -v p2p_orchestrator | crontab - || true",
                timeout=10
            )

            # Add new cron entry for supervised P2P
            cron_cmd = (
                f"@reboot cd {node_path} && {venv_activate} && "
                f"python scripts/p2p_supervisor.py --node-id {node_name} "
                f">> logs/p2p_supervisor.log 2>&1"
            )

            add_cron_result = await client.run_async(
                f'(crontab -l 2>/dev/null; echo "{cron_cmd}") | crontab -',
                timeout=20
            )
            if add_cron_result.returncode != 0:
                logger.warning(f"[{node_name}] Failed to add cron: {add_cron_result.stderr}")

        # Step 4: Start supervised P2P in background
        logger.info(f"[{node_name}] Starting supervised P2P...")
        start_cmd = (
            f"cd {node_path} && {venv_activate} && "
            f"nohup python scripts/p2p_supervisor.py --node-id {node_name} "
            f"> logs/p2p_supervisor.log 2>&1 &"
        )
        start_result = await client.run_async(start_cmd, timeout=30)

        if start_result.returncode != 0:
            return (node_name, False, f"Failed to start supervisor: {start_result.stderr}")

        # Step 5: Wait and verify
        await asyncio.sleep(5)

        verify_result = await client.run_async("pgrep -f p2p_supervisor", timeout=10)
        if verify_result.returncode != 0 or not verify_result.stdout.strip():
            # Check if P2P orchestrator is running directly (supervisor may have started it)
            p2p_result = await client.run_async("pgrep -f p2p_orchestrator", timeout=10)
            if p2p_result.returncode == 0 and p2p_result.stdout.strip():
                return (node_name, True, f"P2P running (PID: {p2p_result.stdout.strip()})")
            return (node_name, False, "Supervisor not running after start")

        return (node_name, True, f"Supervisor running (PID: {verify_result.stdout.strip()})")

    except Exception as e:
        logger.error(f"[{node_name}] Error: {e}")
        return (node_name, False, f"Exception: {str(e)}")


async def deploy_all(
    nodes: List[str],
    skip_cron: bool,
    dry_run: bool,
    max_parallel: int = 10
) -> Dict[str, Tuple[bool, str]]:
    """Deploy to all specified nodes in parallel."""
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'distributed_hosts.yaml'
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    hosts = config.get('hosts', {})

    # Filter to specified nodes or all P2P-enabled nodes
    if nodes:
        target_nodes = [(n, hosts[n]) for n in nodes if n in hosts]
    else:
        target_nodes = [
            (name, cfg) for name, cfg in hosts.items()
            if cfg.get('p2p_enabled', True) and cfg.get('status') == 'ready'
        ]

    logger.info(f"Deploying to {len(target_nodes)} nodes...")

    # Run in parallel with limit
    semaphore = asyncio.Semaphore(max_parallel)

    async def limited_deploy(name, cfg):
        async with semaphore:
            return await deploy_to_node(name, cfg, skip_cron, dry_run)

    tasks = [limited_deploy(name, cfg) for name, cfg in target_nodes]
    results = await asyncio.gather(*tasks)

    return {r[0]: (r[1], r[2]) for r in results}


def main():
    parser = argparse.ArgumentParser(
        description='Deploy P2P supervision to cluster nodes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--nodes',
        help='Comma-separated list of nodes to deploy to (default: all P2P-enabled)'
    )
    parser.add_argument(
        '--skip-cron',
        action='store_true',
        help='Skip cron setup (just restart P2P)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without doing it'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=10,
        help='Max parallel deployments (default: 10)'
    )
    args = parser.parse_args()

    nodes = args.nodes.split(',') if args.nodes else []

    logger.info("=" * 60)
    logger.info("P2P Supervision Deployment")
    logger.info(f"  Nodes: {', '.join(nodes) if nodes else 'ALL P2P-enabled'}")
    logger.info(f"  Skip cron: {args.skip_cron}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info("=" * 60)

    results = asyncio.run(deploy_all(
        nodes=nodes,
        skip_cron=args.skip_cron,
        dry_run=args.dry_run,
        max_parallel=args.parallel
    ))

    # Print summary
    success_count = sum(1 for ok, _ in results.values() if ok)
    fail_count = len(results) - success_count

    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)

    for node_name in sorted(results.keys()):
        success, message = results[node_name]
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {node_name}: {message}")

    print("=" * 60)
    print(f"Total: {len(results)} nodes, {success_count} OK, {fail_count} FAILED")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == '__main__':
    main()
