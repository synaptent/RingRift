#!/usr/bin/env python3
"""
P2P Orchestrator Process Supervisor.

Keeps the P2P orchestrator running with automatic restart on crash.
Designed for use with cron @reboot or systemd.

Usage:
    # Run with auto-detected node-id from hostname
    python scripts/p2p_supervisor.py

    # Run with explicit node-id
    python scripts/p2p_supervisor.py --node-id lambda-gh200-1

    # Dry run (show command that would be executed)
    python scripts/p2p_supervisor.py --dry-run

Environment:
    RINGRIFT_P2P_SUPERVISOR_MAX_RESTARTS: Max restarts before giving up (default: 10)
    RINGRIFT_P2P_SUPERVISOR_RESTART_WINDOW: Window for counting restarts in seconds (default: 600)
    RINGRIFT_P2P_SUPERVISOR_INITIAL_BACKOFF: Initial backoff in seconds (default: 5)
    RINGRIFT_P2P_SUPERVISOR_MAX_BACKOFF: Max backoff in seconds (default: 300)
"""

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Configuration via environment
MAX_RESTARTS = int(os.environ.get('RINGRIFT_P2P_SUPERVISOR_MAX_RESTARTS', 10))
RESTART_WINDOW = int(os.environ.get('RINGRIFT_P2P_SUPERVISOR_RESTART_WINDOW', 600))
INITIAL_BACKOFF = int(os.environ.get('RINGRIFT_P2P_SUPERVISOR_INITIAL_BACKOFF', 5))
MAX_BACKOFF = int(os.environ.get('RINGRIFT_P2P_SUPERVISOR_MAX_BACKOFF', 300))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [P2P-SUPERVISOR] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_hostname() -> str:
    """Get the current hostname."""
    return socket.gethostname()


def load_cluster_config() -> dict:
    """Load the distributed hosts configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'distributed_hosts.yaml'
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_node_config(node_id: str, config: dict) -> Optional[dict]:
    """Get configuration for a specific node."""
    hosts = config.get('hosts', {})
    return hosts.get(node_id)


def detect_node_id(config: dict) -> Optional[str]:
    """Detect node ID from hostname or Tailscale IP."""
    hostname = get_hostname()
    hosts = config.get('hosts', {})

    # First try exact hostname match
    for node_id, node_config in hosts.items():
        if node_id == hostname:
            return node_id
        # Also check if hostname is in ssh_host or tailscale_ip
        if node_config.get('ssh_host') == hostname:
            return node_id
        if node_config.get('tailscale_ip') == hostname:
            return node_id

    # Try partial match (e.g., 'lambda-gh200-1' from 'lambda-gh200-1.internal')
    for node_id in hosts:
        if hostname.startswith(node_id):
            return node_id

    return None


def build_p2p_command(node_id: str, node_config: dict, config: dict) -> list:
    """Build the P2P orchestrator command with all necessary arguments."""
    ringrift_path = node_config.get('ringrift_path', '~/ringrift/ai-service')
    # Expand ~ in path
    ringrift_path = os.path.expanduser(ringrift_path)

    script_path = os.path.join(ringrift_path, 'scripts', 'p2p_orchestrator.py')

    cmd = [
        sys.executable,  # Use current Python interpreter
        script_path,
        f'--node-id', node_id,
        '--port', '8770',
        '--ringrift-path', ringrift_path,
        '--kill-duplicates',
    ]

    # Add advertise-host for Tailscale nodes
    tailscale_ip = node_config.get('tailscale_ip')
    if tailscale_ip:
        cmd.extend(['--advertise-host', tailscale_ip])

    # Add relay-peers for NAT-blocked nodes
    if node_config.get('nat_blocked') or node_config.get('force_relay_mode'):
        relay_peers = []
        # Find non-NAT-blocked voters from config
        p2p_voters = config.get('p2p_voters', [])
        hosts = config.get('hosts', {})
        for voter in p2p_voters:
            voter_config = hosts.get(voter, {})
            if not voter_config.get('nat_blocked'):
                # Use Tailscale IP if available, otherwise ssh_host
                peer_host = voter_config.get('tailscale_ip') or voter_config.get('ssh_host')
                if peer_host:
                    relay_peers.append(f"{peer_host}:8770")
        if relay_peers:
            cmd.extend(['--relay-peers', ','.join(relay_peers[:3])])  # Limit to 3 relay peers

    # Build known peers list from p2p_voters
    peers = []
    p2p_voters = config.get('p2p_voters', [])
    hosts = config.get('hosts', {})
    for voter in p2p_voters:
        if voter == node_id:
            continue  # Don't include self
        voter_config = hosts.get(voter, {})
        peer_host = voter_config.get('tailscale_ip') or voter_config.get('ssh_host')
        if peer_host:
            peers.append(f"{peer_host}:8770")
    if peers:
        cmd.extend(['--peers', ','.join(peers[:5])])  # Limit to 5 initial peers

    return cmd


def run_p2p_supervised(node_id: str, node_config: dict, config: dict, dry_run: bool = False):
    """Run the P2P orchestrator with supervision and auto-restart."""
    restarts = []
    backoff = INITIAL_BACKOFF
    proc = None

    # Handle signals for graceful shutdown
    shutdown_requested = False

    def signal_handler(sig, frame):
        nonlocal shutdown_requested, proc
        sig_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else str(sig)
        logger.info(f"Received {sig_name}, shutting down...")
        shutdown_requested = True
        if proc and proc.poll() is None:
            logger.info("Sending SIGTERM to P2P process...")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("P2P didn't terminate in 30s, sending SIGKILL")
                proc.kill()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    cmd = build_p2p_command(node_id, node_config, config)
    logger.info(f"P2P command: {' '.join(cmd)}")

    if dry_run:
        logger.info("DRY RUN - would execute the above command")
        return

    while not shutdown_requested:
        # Clean old restart times
        now = time.time()
        restarts = [t for t in restarts if now - t < RESTART_WINDOW]

        if len(restarts) >= MAX_RESTARTS:
            logger.error(
                f"Too many restarts ({len(restarts)} in {RESTART_WINDOW}s), "
                f"giving up. Check logs for root cause."
            )
            sys.exit(1)

        # Start P2P orchestrator
        logger.info(f"Starting P2P orchestrator for node {node_id}...")
        start_time = time.time()

        try:
            # Feb 2026: Set PYTHONPATH and cwd so supervisor-restarted P2P can
            # find scripts.* modules. Without this, restarts fail with
            # "ModuleNotFoundError: No module named 'scripts'" on Lambda nodes.
            proc_env = os.environ.copy()
            ringrift_path = proc_env.get(
                "RINGRIFT_PATH",
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            # Feb 28, 2026: Use unconditional assignment, not setdefault.
            # setdefault doesn't override when PYTHONPATH exists (even empty ""),
            # causing "ModuleNotFoundError: No module named 'scripts'" on ALL
            # Lambda restarts. This was the root cause of 0% training success
            # on 7 Lambda GH200 nodes (441 consecutive failures).
            proc_env["PYTHONPATH"] = ringrift_path

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=proc_env,
                cwd=ringrift_path,
            )

            # Stream output to our log
            while True:
                if shutdown_requested:
                    break
                line = proc.stdout.readline()
                if line:
                    # Log P2P output without timestamp (it has its own)
                    print(line.rstrip())
                elif proc.poll() is not None:
                    break

            exit_code = proc.wait()
            runtime = time.time() - start_time

        except Exception as e:
            logger.exception(f"Exception running P2P: {e}")
            exit_code = -1
            runtime = time.time() - start_time

        if shutdown_requested:
            logger.info("Shutdown complete")
            return

        # Handle exit
        if exit_code == 0 and runtime > 60:
            # Clean exit after running for a while - probably intentional shutdown
            logger.info(f"P2P exited cleanly (code 0) after {runtime:.0f}s")
            # Reset backoff for clean exits
            backoff = INITIAL_BACKOFF
            # Still restart since we want to keep it running
            logger.info("Restarting after clean exit...")
        elif exit_code == 0 and runtime <= 60:
            # Quick clean exit - might be a startup failure
            logger.warning(f"P2P exited quickly (code 0, {runtime:.1f}s) - possible startup issue")
        else:
            # Crash
            logger.error(f"P2P crashed (code {exit_code}) after {runtime:.1f}s")

        # Record restart
        restarts.append(time.time())
        restart_count = len(restarts)

        logger.info(f"Restarting in {backoff}s... (restart {restart_count}/{MAX_RESTARTS} in window)")
        time.sleep(backoff)

        # Exponential backoff
        backoff = min(backoff * 2, MAX_BACKOFF)


def main():
    parser = argparse.ArgumentParser(
        description='P2P Orchestrator Process Supervisor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--node-id',
        help='Node ID (auto-detected from hostname if not specified)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show command that would be executed without running it'
    )
    args = parser.parse_args()

    # Load config
    config = load_cluster_config()
    if not config:
        logger.error("Failed to load cluster config")
        sys.exit(1)

    # Determine node ID
    node_id = args.node_id
    if not node_id:
        node_id = detect_node_id(config)
        if not node_id:
            logger.error(
                f"Could not detect node ID from hostname '{get_hostname()}'. "
                "Please specify --node-id explicitly."
            )
            sys.exit(1)
        logger.info(f"Auto-detected node ID: {node_id}")

    # Get node config
    node_config = get_node_config(node_id, config)
    if not node_config:
        logger.error(f"No configuration found for node '{node_id}'")
        sys.exit(1)

    # Check if P2P is enabled for this node
    if not node_config.get('p2p_enabled', True):
        logger.info(f"P2P is disabled for node {node_id}")
        sys.exit(0)

    # Log startup info
    logger.info(f"=" * 60)
    logger.info(f"P2P Supervisor starting")
    logger.info(f"  Node ID: {node_id}")
    logger.info(f"  Hostname: {get_hostname()}")
    logger.info(f"  Max restarts: {MAX_RESTARTS} in {RESTART_WINDOW}s")
    logger.info(f"  Backoff: {INITIAL_BACKOFF}s -> {MAX_BACKOFF}s")
    logger.info(f"=" * 60)

    # Run supervised
    run_p2p_supervised(node_id, node_config, config, args.dry_run)


if __name__ == '__main__':
    main()
