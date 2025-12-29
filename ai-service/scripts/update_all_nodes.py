#!/usr/bin/env python3
"""
Update all cluster nodes to the latest code from GitHub.

Usage:
    python scripts/update_all_nodes.py [--commit HASH] [--restart-p2p] [--dry-run]
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

from app.core.ssh import SSHClient, SSHConfig, SSHResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Node path mappings by provider
PATH_MAPPINGS = {
    'runpod': '/workspace/ringrift/ai-service',
    'vast': '~/ringrift/ai-service',  # Default for vast
    'vast_workspace': '/workspace/ringrift/ai-service',  # Some vast nodes
    'nebius': '~/ringrift/ai-service',
    'vultr': '/root/ringrift/ai-service',
    'hetzner': '/root/ringrift/ai-service',
    'mac-studio': '~/Development/RingRift/ai-service',
    'local-mac': '/Users/armand/Development/RingRift/ai-service',
}


def get_node_path(node_name: str, node_config: dict) -> str | None:
    """Get the ringrift path for a node based on config or provider.

    Returns None if node explicitly has no path (e.g., proxy-only nodes).
    """
    # Use explicit path from config if available
    if 'ringrift_path' in node_config:
        path = node_config['ringrift_path']
        # Return None if explicitly set to null/None
        if path is None:
            return None
        return path

    # Determine provider from node name
    for provider in ['runpod', 'vast', 'nebius', 'vultr', 'hetzner']:
        if node_name.startswith(provider):
            return PATH_MAPPINGS[provider]

    # Default fallback
    return '~/ringrift/ai-service'


async def check_p2p_running(client, node_name: str, node_path: str) -> bool:
    """Check if P2P orchestrator is running on the node."""
    result = await client.run_async("pgrep -f p2p_orchestrator", timeout=10)
    if result.returncode == 0 and result.stdout.strip():
        logger.info(f"[{node_name}] P2P orchestrator is running (PID: {result.stdout.strip()})")
        return True
    return False


async def update_node(
    node_name: str,
    node_config: dict,
    commit_hash: str,
    restart_p2p: bool,
    dry_run: bool
) -> Tuple[str, bool, str]:
    """
    Update a single node to the latest code.

    Returns:
        (node_name, success, message)
    """
    try:
        # Skip coordinator nodes (local-mac, mac-studio)
        if node_config.get('role') == 'coordinator' and node_name in ['local-mac', 'mac-studio']:
            return (node_name, True, "SKIPPED: Coordinator node")

        # Skip if node is not ready
        if node_config.get('status') != 'ready':
            return (node_name, False, f"SKIPPED: Status is {node_config.get('status')}")

        # Create SSH client from config
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

        node_path = get_node_path(node_name, node_config)
        logger.info(f"[{node_name}] Using path: {node_path}")

        # Check if P2P is running
        p2p_was_running = await check_p2p_running(client, node_name, node_path)

        if dry_run:
            msg = f"DRY-RUN: Would update {node_path}"
            if p2p_was_running and restart_p2p:
                msg += " and restart P2P"
            return (node_name, True, msg)

        # Update git repository
        logger.info(f"[{node_name}] Updating git repository...")

        # Stash any local changes
        stash_cmd = f"cd {node_path} && git stash"
        stash_result = await client.run_async(stash_cmd, timeout=30)
        if stash_result.returncode != 0:
            logger.warning(f"[{node_name}] Git stash failed (may be nothing to stash): {stash_result.stderr}")

        # Pull latest code
        pull_cmd = f"cd {node_path} && git pull origin main"
        pull_result = await client.run_async(pull_cmd, timeout=60)
        if pull_result.returncode != 0:
            return (node_name, False, f"Git pull failed: {pull_result.stderr}")

        # Verify commit
        verify_cmd = f"cd {node_path} && git rev-parse --short HEAD"
        verify_result = await client.run_async(verify_cmd, timeout=10)
        current_commit = verify_result.stdout.strip() if verify_result.returncode == 0 else "unknown"

        logger.info(f"[{node_name}] Updated to commit {current_commit}")

        # Restart P2P if it was running and requested
        if p2p_was_running and restart_p2p:
            logger.info(f"[{node_name}] Restarting P2P orchestrator...")

            # Kill existing P2P
            kill_cmd = "pkill -f p2p_orchestrator"
            await client.run_async(kill_cmd, timeout=10)

            # Wait for graceful shutdown
            await asyncio.sleep(2)

            # Build proper P2P start command
            venv_activate = node_config.get('venv_activate', 'source venv/bin/activate')

            # Build P2P arguments
            p2p_args = [
                f"--node-id {node_name}",
                "--port 8770",
                f"--ringrift-path {node_path}",
                "--kill-duplicates",  # Kill any stale processes
            ]

            # Add advertise-host for Tailscale nodes
            tailscale_ip = node_config.get('tailscale_ip')
            if tailscale_ip:
                p2p_args.append(f"--advertise-host {tailscale_ip}")

            # Add relay-peers for NAT-blocked nodes (use p2p_voters as relay)
            if node_config.get('nat_blocked') or node_config.get('force_relay_mode'):
                # Use non-NAT-blocked voters as relay peers
                relay_peers = ['vultr-a100-20gb:8770', 'nebius-h100-3:8770']
                p2p_args.append(f"--relay-peers {','.join(relay_peers)}")

            # Build known peers list from p2p_voters
            known_peers = [
                'vultr-a100-20gb:8770',
                'nebius-h100-3:8770',
                'hetzner-cpu1:8770',
                'nebius-backbone-1:8770',
            ]
            p2p_args.append(f"--peers {','.join(known_peers)}")

            p2p_args_str = ' '.join(p2p_args)

            start_cmd = (
                f"cd {node_path} && {venv_activate} && "
                f"mkdir -p logs && "
                f"nohup python scripts/p2p_orchestrator.py {p2p_args_str} "
                f"> logs/p2p.log 2>&1 &"
            )
            start_result = await client.run_async(start_cmd, timeout=15)

            if start_result.returncode != 0:
                logger.warning(f"[{node_name}] P2P restart may have failed: {start_result.stderr}")
            else:
                # Give it a moment to start
                await asyncio.sleep(3)
                # Verify it's running
                if await check_p2p_running(client, node_name, node_path):
                    return (node_name, True, f"Updated to {current_commit}, P2P restarted")
                else:
                    return (node_name, True, f"Updated to {current_commit}, P2P restart failed")

        return (node_name, True, f"Updated to {current_commit}")

    except Exception as e:
        logger.error(f"[{node_name}] Error: {e}")
        return (node_name, False, f"Exception: {str(e)}")


async def update_all_nodes(
    commit_hash: str,
    restart_p2p: bool,
    dry_run: bool,
    max_parallel: int = 10
) -> Dict[str, Tuple[bool, str]]:
    """
    Update all cluster nodes in parallel.

    Returns:
        Dict mapping node_name -> (success, message)
    """
    # Load distributed hosts config
    config_path = Path(__file__).parent.parent / 'config' / 'distributed_hosts.yaml'

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    hosts = config.get('hosts', {})
    logger.info(f"Found {len(hosts)} hosts in configuration")

    # Create update tasks
    tasks = []
    for node_name, node_config in hosts.items():
        task = update_node(node_name, node_config, commit_hash, restart_p2p, dry_run)
        tasks.append(task)

    # Run updates in parallel with concurrency limit
    results = {}
    semaphore = asyncio.Semaphore(max_parallel)

    async def run_with_semaphore(task):
        async with semaphore:
            return await task

    logger.info(f"Starting updates (max {max_parallel} parallel)...")
    completed_tasks = await asyncio.gather(*[run_with_semaphore(t) for t in tasks])

    for node_name, success, message in completed_tasks:
        results[node_name] = (success, message)

    return results


def print_summary(results: Dict[str, Tuple[bool, str]]):
    """Print a summary of update results."""
    successful = []
    failed = []
    skipped = []
    p2p_restarted = []

    for node_name, (success, message) in results.items():
        if "SKIPPED" in message:
            skipped.append((node_name, message))
        elif success:
            successful.append((node_name, message))
            if "P2P restarted" in message or "P2P restart" in message:
                p2p_restarted.append(node_name)
        else:
            failed.append((node_name, message))

    print("\n" + "="*80)
    print("UPDATE SUMMARY")
    print("="*80)

    print(f"\n✅ Successfully updated: {len(successful)} nodes")
    for node_name, message in successful:
        print(f"  - {node_name}: {message}")

    if p2p_restarted:
        print(f"\n🔄 P2P restarted: {len(p2p_restarted)} nodes")
        for node_name in p2p_restarted:
            print(f"  - {node_name}")

    if skipped:
        print(f"\n⏭️  Skipped: {len(skipped)} nodes")
        for node_name, message in skipped:
            print(f"  - {node_name}: {message}")

    if failed:
        print(f"\n❌ Failed: {len(failed)} nodes")
        for node_name, message in failed:
            print(f"  - {node_name}: {message}")

    print(f"\n📊 Total: {len(results)} nodes")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Update all cluster nodes to latest code")
    parser.add_argument(
        '--commit',
        default='88ca80cb2',
        help='Git commit hash to update to (default: 88ca80cb2)'
    )
    parser.add_argument(
        '--restart-p2p',
        action='store_true',
        help='Restart P2P orchestrator if it was running'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually updating'
    )
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=10,
        help='Maximum number of parallel updates (default: 10)'
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    logger.info(f"Target commit: {args.commit}")
    logger.info(f"Restart P2P: {args.restart_p2p}")

    # Run updates
    results = asyncio.run(update_all_nodes(
        args.commit,
        args.restart_p2p,
        args.dry_run,
        args.max_parallel
    ))

    # Print summary
    print_summary(results)

    # Exit with error if any updates failed
    failed_count = sum(1 for success, _ in results.values() if not success)
    if failed_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
