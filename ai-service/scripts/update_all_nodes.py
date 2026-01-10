#!/usr/bin/env python3
"""
Update all cluster nodes to the latest code from GitHub.

Usage:
    # Safe mode (RECOMMENDED) - preserves quorum during rolling updates
    python scripts/update_all_nodes.py --safe-mode --restart-p2p

    # Sync config files (for files in .gitignore like distributed_hosts.yaml)
    python scripts/update_all_nodes.py --sync-config --restart-p2p

    # Legacy mode (not recommended for production)
    python scripts/update_all_nodes.py [--commit HASH] [--restart-p2p] [--dry-run]

January 3, 2026 - Sprint 16.2:
    Added --safe-mode flag for quorum-safe rolling updates. This mode uses
    QuorumSafeUpdateCoordinator to batch updates by node type and verify
    cluster health between batches.

    Problem: Simultaneous P2P restarts on 10+ nodes caused quorum loss cascade.
    Solution: Safe mode updates non-voters in parallel, then voters one at a time.

January 9, 2026 - Config Sync:
    Added --sync-config flag to sync non-git-tracked config files (like
    distributed_hosts.yaml which is in .gitignore) to all nodes. This prevents
    configuration drift that can cause P2P cluster inconsistencies.

    Problem: distributed_hosts.yaml is gitignored, so git pull doesn't update it.
    Solution: Explicitly sync config files via SCP when --sync-config is set.
"""

import argparse
import asyncio
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.ssh import SSHClient, SSHConfig
from app.config.cluster_config import get_p2p_voters, get_cluster_nodes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Config files to sync (relative to ai-service directory)
# These are files in .gitignore that need explicit sync
CONFIG_FILES_TO_SYNC = [
    'config/distributed_hosts.yaml',
]

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


def get_p2p_voter_peers() -> list[str]:
    """Get P2P peer list from distributed_hosts.yaml voters.

    Jan 7, 2026: Replaces hardcoded peer list. Dynamically builds peer list
    from config to ensure consistency with actual voter configuration.

    Returns:
        List of voter peer addresses in "IP:8770" format for bootstrap.
        Falls back to empty list if config unavailable (p2p_orchestrator
        will use its own config-based bootstrap in that case).
    """
    try:
        voters = get_p2p_voters()
        nodes = get_cluster_nodes()
        peers = []

        for voter_name in voters:
            node = nodes.get(voter_name)
            if node and node.tailscale_ip:
                peers.append(f"{node.tailscale_ip}:8770")

        # Return up to 5 voters as bootstrap peers (sufficient for discovery)
        return peers[:5]
    except Exception as e:
        logger.warning(f"Failed to load P2P voters from config: {e}")
        return []  # Fall back to config-based bootstrap in p2p_orchestrator


async def check_p2p_running(client, node_name: str, node_path: str) -> bool:
    """Check if P2P orchestrator is running on the node."""
    result = await client.run_async("pgrep -f p2p_orchestrator", timeout=10)
    if result.returncode == 0 and result.stdout.strip():
        logger.info(f"[{node_name}] P2P orchestrator is running (PID: {result.stdout.strip()})")
        return True
    return False


async def sync_config_files(
    client,
    node_name: str,
    node_path: str,
    local_base_path: Path,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """
    Sync non-git-tracked config files to a node via SCP.

    These files are in .gitignore so git pull doesn't update them.
    This function explicitly syncs them to prevent configuration drift.

    Args:
        client: SSH client for the node
        node_name: Name of the node
        node_path: Remote path to ringrift/ai-service
        local_base_path: Local path to ai-service directory
        dry_run: Preview mode without making changes

    Returns:
        (success, message)
    """
    synced_files = []
    failed_files = []

    for config_file in CONFIG_FILES_TO_SYNC:
        local_file = local_base_path / config_file
        if not local_file.exists():
            logger.warning(f"[{node_name}] Config file not found locally: {config_file}")
            failed_files.append(config_file)
            continue

        remote_file = f"{node_path}/{config_file}"

        if dry_run:
            logger.info(f"[{node_name}] DRY-RUN: Would sync {config_file}")
            synced_files.append(config_file)
            continue

        # Use SCP to copy the file
        # The SSHClient doesn't have direct SCP support, so we use cat + ssh
        try:
            with open(local_file, 'r') as f:
                content = f.read()

            # Ensure parent directory exists
            parent_dir = '/'.join(remote_file.rsplit('/', 1)[:-1])
            await client.run_async(f"mkdir -p {parent_dir}", timeout=10)

            # Write content via heredoc
            # Escape any single quotes in the content
            escaped_content = content.replace("'", "'\"'\"'")
            cmd = f"cat > {remote_file} << 'CONFIGEOF'\n{content}\nCONFIGEOF"
            result = await client.run_async(cmd, timeout=30)

            if result.returncode == 0:
                logger.info(f"[{node_name}] Synced {config_file}")
                synced_files.append(config_file)
            else:
                logger.warning(f"[{node_name}] Failed to sync {config_file}: {result.stderr}")
                failed_files.append(config_file)

        except Exception as e:
            logger.error(f"[{node_name}] Error syncing {config_file}: {e}")
            failed_files.append(config_file)

    if failed_files:
        return (False, f"Config sync partial: {len(synced_files)} OK, {len(failed_files)} failed")
    elif synced_files:
        return (True, f"Config synced: {', '.join(synced_files)}")
    else:
        return (True, "No config files to sync")


async def update_node(
    node_name: str,
    node_config: dict,
    commit_hash: str,
    restart_p2p: bool,
    dry_run: bool,
    include_coordinators: bool = False,
    sync_config: bool = False,
) -> Tuple[str, bool, str]:
    """
    Update a single node to the latest code.

    Args:
        node_name: Name of the node
        node_config: Node configuration dictionary
        commit_hash: Target git commit
        restart_p2p: Whether to restart P2P orchestrator
        dry_run: Preview mode without making changes
        include_coordinators: Include coordinator nodes in updates (default: False for local-mac only)
        sync_config: Sync non-git-tracked config files (like distributed_hosts.yaml)

    Returns:
        (node_name, success, message)
    """
    try:
        # Skip local-mac (this machine) unless explicitly included
        # Note: mac-studio IS updated since it's a separate machine
        if node_name == 'local-mac' and not include_coordinators:
            return (node_name, True, "SKIPPED: Local machine (use --include-coordinators)")

        # Skip if node is not ready
        if node_config.get('status') != 'ready':
            return (node_name, False, f"SKIPPED: Status is {node_config.get('status')}")

        # Create SSH client from config
        # Prefer Tailscale IP over SSH proxy hosts (more reliable)
        tailscale_ip = node_config.get('tailscale_ip')
        ssh_host = node_config.get('ssh_host', '')
        ssh_port = node_config.get('ssh_port', 22)

        # If we have a Tailscale IP, use it as the primary host (port 22)
        # SSH proxy hosts like ssh2.vast.ai:12345 are unreliable
        if tailscale_ip:
            primary_host = tailscale_ip
            primary_port = 22
            logger.info(f"[{node_name}] Using Tailscale IP: {tailscale_ip}")
        else:
            primary_host = ssh_host
            primary_port = ssh_port
            logger.info(f"[{node_name}] Using SSH host: {ssh_host}:{ssh_port}")

        ssh_config = SSHConfig(
            host=primary_host,
            port=primary_port,
            user=node_config.get('ssh_user', 'root'),
            key_path=node_config.get('ssh_key'),
            tailscale_ip=tailscale_ip,  # Keep for fallback
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

        # Sync config files if requested (for files in .gitignore)
        config_sync_msg = ""
        if sync_config:
            local_base = Path(__file__).parent.parent  # ai-service directory
            sync_success, sync_msg = await sync_config_files(
                client, node_name, node_path, local_base, dry_run
            )
            if sync_success:
                config_sync_msg = f", {sync_msg}"
                logger.info(f"[{node_name}] {sync_msg}")
            else:
                logger.warning(f"[{node_name}] {sync_msg}")
                config_sync_msg = f", {sync_msg}"

        # Restart P2P if it was running and requested
        if p2p_was_running and restart_p2p:
            logger.info(f"[{node_name}] Restarting P2P orchestrator...")

            # Kill existing P2P
            kill_cmd = "pkill -f p2p_orchestrator"
            await client.run_async(kill_cmd, timeout=10)

            # Wait for graceful shutdown
            await asyncio.sleep(2)

            # Build proper P2P start command
            # Jan 7, 2026: Handle venv activation robustly
            # - Use explicit config value if provided
            # - ':' means no-op (pyenv/conda/system Python)
            # - Default to source venv/bin/activate with existence check
            venv_activate = node_config.get('venv_activate')
            if venv_activate is None:
                # Default: try to source venv, but test if it exists first
                venv_activate = (
                    f"if [ -f {node_path}/venv/bin/activate ]; then "
                    f"source {node_path}/venv/bin/activate; "
                    f"fi"
                )
            elif venv_activate == ':':
                # Explicit no-op for pyenv/conda/system Python
                venv_activate = ':'

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

            # Build known peers list dynamically from config p2p_voters
            # Jan 7, 2026: Replaces hardcoded peer list for consistency
            known_peers = get_p2p_voter_peers()
            if known_peers:
                p2p_args.append(f"--peers {','.join(known_peers)}")
            # If no peers from config, p2p_orchestrator uses its own bootstrap

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
                    return (node_name, True, f"Updated to {current_commit}{config_sync_msg}, P2P restarted")
                else:
                    return (node_name, True, f"Updated to {current_commit}{config_sync_msg}, P2P restart failed")

        return (node_name, True, f"Updated to {current_commit}{config_sync_msg}")

    except Exception as e:
        logger.error(f"[{node_name}] Error: {e}")
        return (node_name, False, f"Exception: {str(e)}")


async def update_all_nodes(
    commit_hash: str,
    restart_p2p: bool,
    dry_run: bool,
    max_parallel: int = 10,
    include_coordinators: bool = False,
    sync_config: bool = False,
) -> Dict[str, Tuple[bool, str]]:
    """
    Update all cluster nodes in parallel.

    Args:
        commit_hash: Target git commit hash
        restart_p2p: Whether to restart P2P orchestrator
        dry_run: Preview mode without making changes
        max_parallel: Maximum concurrent updates
        include_coordinators: Include coordinator nodes (local-mac)
        sync_config: Sync non-git-tracked config files (like distributed_hosts.yaml)

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
        task = update_node(
            node_name, node_config, commit_hash, restart_p2p, dry_run,
            include_coordinators=include_coordinators,
            sync_config=sync_config,
        )
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
    parser = argparse.ArgumentParser(
        description="Update all cluster nodes to latest code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Safe mode (RECOMMENDED for production)
    python scripts/update_all_nodes.py --safe-mode --restart-p2p

    # Safe mode with custom convergence timeout
    python scripts/update_all_nodes.py --safe-mode --restart-p2p --convergence-timeout 180

    # Update only non-voter nodes (safest, no quorum risk)
    python scripts/update_all_nodes.py --safe-mode --skip-voters --restart-p2p

    # Dry run to preview batches
    python scripts/update_all_nodes.py --safe-mode --restart-p2p --dry-run

    # Legacy mode (not recommended, shows warning)
    python scripts/update_all_nodes.py --restart-p2p
"""
    )
    parser.add_argument(
        '--commit',
        default='main',
        help='Git commit/branch to update to (default: main)'
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
    parser.add_argument(
        '--include-coordinators',
        action='store_true',
        help='Include coordinator nodes (local-mac) in updates'
    )

    # January 3, 2026 - Sprint 16.2: Quorum-safe rolling updates
    parser.add_argument(
        '--safe-mode',
        action='store_true',
        help='Use quorum-safe rolling updates (RECOMMENDED for production)'
    )
    parser.add_argument(
        '--batch-delay',
        type=int,
        default=30,
        help='Seconds between voter batches in safe mode (default: 30)'
    )
    parser.add_argument(
        '--skip-voters',
        action='store_true',
        help='Only update non-voter nodes (safest, no quorum risk)'
    )
    parser.add_argument(
        '--skip-non-voters',
        action='store_true',
        help='Only update voter nodes (for testing voter updates)'
    )
    parser.add_argument(
        '--convergence-timeout',
        type=int,
        default=120,
        help='Seconds to wait for cluster convergence (default: 120)'
    )

    # January 9, 2026: Config file sync for gitignored files
    parser.add_argument(
        '--sync-config',
        action='store_true',
        help='Sync non-git-tracked config files (like distributed_hosts.yaml)'
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    logger.info(f"Target commit: {args.commit}")
    logger.info(f"Restart P2P: {args.restart_p2p}")
    logger.info(f"Include coordinators: {args.include_coordinators}")
    if args.sync_config:
        logger.info(f"Sync config: {CONFIG_FILES_TO_SYNC}")

    # January 3, 2026 - Sprint 16.2: Use QuorumSafeUpdateCoordinator in safe mode
    if args.safe_mode:
        logger.info("SAFE MODE: Using quorum-safe rolling updates")

        try:
            from scripts.cluster_update_coordinator import (
                QuorumSafeUpdateCoordinator,
                UpdateCoordinatorConfig,
            )
        except ImportError as e:
            logger.error(f"Failed to import QuorumSafeUpdateCoordinator: {e}")
            logger.error("Please ensure scripts/cluster_update_coordinator.py exists")
            sys.exit(1)

        config = UpdateCoordinatorConfig(
            convergence_timeout=args.convergence_timeout,
            voter_update_delay=args.batch_delay,
            max_parallel_non_voters=args.max_parallel,
            dry_run=args.dry_run,
        )
        coordinator = QuorumSafeUpdateCoordinator(config=config)

        try:
            result = asyncio.run(coordinator.update_cluster(
                target_commit=args.commit,
                restart_p2p=args.restart_p2p,
                dry_run=args.dry_run,
                skip_voters=args.skip_voters,
                skip_non_voters=args.skip_non_voters,
            ))

            # Print safe mode summary
            print("\n" + "="*80)
            print("QUORUM-SAFE UPDATE SUMMARY")
            print("="*80)

            if result.success:
                print(f"\n✅ Update completed successfully")
                print(f"   Batches completed: {result.batches_completed}")
                print(f"   Nodes updated: {result.nodes_updated}")
                if result.nodes_skipped:
                    print(f"   Nodes skipped: {result.nodes_skipped}")
                print(f"   Duration: {result.duration_seconds:.1f}s")
            else:
                print(f"\n❌ Update failed")
                print(f"   Error: {result.error_message}")
                if result.failed_batch:
                    print(f"   Failed batch: {result.failed_batch}")
                if result.rollback_performed:
                    print(f"   Rollback: Performed successfully")

            print("="*80 + "\n")

            if not result.success:
                sys.exit(1)

        except KeyboardInterrupt:
            logger.warning("Update interrupted by user")
            sys.exit(130)
        except Exception as e:
            logger.error(f"Safe mode update failed: {e}")
            sys.exit(1)

    else:
        # Legacy mode - warn about quorum risk
        if args.restart_p2p:
            warnings.warn(
                "Running without --safe-mode with --restart-p2p can cause quorum loss. "
                "Consider using: python scripts/update_all_nodes.py --safe-mode --restart-p2p",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "⚠️  WARNING: Running without --safe-mode. "
                "Simultaneous P2P restarts may cause quorum loss. "
                "Use --safe-mode for production updates."
            )

        # Run legacy parallel updates
        results = asyncio.run(update_all_nodes(
            args.commit,
            args.restart_p2p,
            args.dry_run,
            args.max_parallel,
            args.include_coordinators,
            args.sync_config,
        ))

        # Print summary
        print_summary(results)

        # Exit with error if any updates failed
        failed_count = sum(1 for success, _ in results.values() if not success)
        if failed_count > 0:
            sys.exit(1)


if __name__ == '__main__':
    main()
