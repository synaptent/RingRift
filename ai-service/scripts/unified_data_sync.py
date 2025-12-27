#!/usr/bin/env python3
"""Unified Data Sync Service - CLI Entry Point.

This is the primary entry point for all data synchronization. It consolidates:
- streaming_data_collector.py (continuous incremental sync) [DEPRECATED]
- collector_watchdog.py (health monitoring) [DEPRECATED]
- sync_all_data.py (batch sync) [DEPRECATED]

All functionality from deprecated scripts is preserved here.

Usage:
    # Run as daemon (replaces streaming_data_collector.py)
    python scripts/unified_data_sync.py

    # With custom config
    python scripts/unified_data_sync.py --config config/unified_loop.yaml

    # With custom hosts (SSoT defaults to distributed_hosts.yaml)
    python scripts/unified_data_sync.py --hosts config/distributed_hosts.yaml

    # One-shot sync (replaces sync_all_data.py)
    python scripts/unified_data_sync.py --once

    # With watchdog (replaces collector_watchdog.py)
    python scripts/unified_data_sync.py --watchdog

    # Dry run
    python scripts/unified_data_sync.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

# Determine paths
SCRIPT_DIR = Path(__file__).resolve().parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent

# Add ai-service to path for imports
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.config.ports import UNIFIED_SYNC_API_PORT
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("unified_data_sync")
DEFAULT_HOSTS_PATH = "config/distributed_hosts.yaml"
LEGACY_HOSTS_PATH = "config/remote_hosts.yaml"

def run_with_watchdog(service_args: list, check_interval: int = 30, max_restarts: int = 10):
    """Run the service with watchdog monitoring.

    Restarts the service if it crashes or becomes unresponsive.
    """
    import urllib.error
    import urllib.request

    restart_count = 0
    process = None
    http_port = UNIFIED_SYNC_API_PORT

    # Extract http-port from args if specified
    for i, arg in enumerate(service_args):
        if arg == "--http-port" and i + 1 < len(service_args):
            http_port = int(service_args[i + 1])

    def start_service():
        nonlocal process
        cmd = [sys.executable, str(SCRIPT_DIR / "unified_data_sync.py"), *service_args]
        logger.info(f"Starting service: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(AI_SERVICE_ROOT),
        )
        return process

    def check_health():
        try:
            url = f"http://localhost:{http_port}/health"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
            return False

    def stop_service():
        nonlocal process
        if process:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process = None

    # Signal handling
    shutdown_requested = False

    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        logger.info("Shutdown requested")
        shutdown_requested = True
        stop_service()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main watchdog loop
    consecutive_failures = 0
    process = start_service()
    time.sleep(10)  # Give it time to start

    while not shutdown_requested:
        # Check if process is alive
        if process.poll() is not None:
            logger.warning(f"Service exited with code {process.returncode}")
            restart_count += 1
            if restart_count >= max_restarts:
                logger.error(f"Max restarts ({max_restarts}) exceeded")
                break
            logger.info(f"Restarting service (attempt {restart_count}/{max_restarts})")
            time.sleep(5)
            process = start_service()
            time.sleep(10)
            consecutive_failures = 0
            continue

        # Check health
        if check_health():
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            logger.warning(f"Health check failed ({consecutive_failures}/3)")

            if consecutive_failures >= 3:
                logger.warning("Service unresponsive, restarting")
                stop_service()
                restart_count += 1
                if restart_count >= max_restarts:
                    logger.error(f"Max restarts ({max_restarts}) exceeded")
                    break
                time.sleep(5)
                process = start_service()
                time.sleep(10)
                consecutive_failures = 0

        time.sleep(check_interval)

    stop_service()
    logger.info("Watchdog stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Data Sync Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script consolidates all data sync functionality:
  - streaming_data_collector.py (DEPRECATED)
  - collector_watchdog.py (DEPRECATED)
  - sync_all_data.py (DEPRECATED)

Examples:
    # Run as daemon
    python scripts/unified_data_sync.py

    # With watchdog monitoring
    python scripts/unified_data_sync.py --watchdog

    # One-shot sync
    python scripts/unified_data_sync.py --once

    # Verbose output
    python scripts/unified_data_sync.py -v
        """
    )
    parser.add_argument("--config", type=str, default="config/unified_loop.yaml", help="Config file")
    parser.add_argument("--hosts", type=str, default=DEFAULT_HOSTS_PATH, help="Hosts file (SSoT)")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="Check what would sync")
    parser.add_argument("--interval", type=int, help="Override poll interval")
    parser.add_argument("--http-port", type=int, default=UNIFIED_SYNC_API_PORT, help="HTTP API port")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--force", action="store_true", help="Force run even if sync_disabled=true")

    # Watchdog options
    parser.add_argument("--watchdog", action="store_true", help="Run with watchdog monitoring")
    parser.add_argument("--watchdog-interval", type=int, default=30, help="Watchdog check interval")
    parser.add_argument("--max-restarts", type=int, default=10, help="Max restarts before giving up")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # If watchdog mode, run as supervisor
    if args.watchdog:
        service_args = []
        if args.config != "config/unified_loop.yaml":
            service_args.extend(["--config", args.config])
        if args.hosts != DEFAULT_HOSTS_PATH:
            service_args.extend(["--hosts", args.hosts])
        if args.interval:
            service_args.extend(["--interval", str(args.interval)])
        if args.http_port != UNIFIED_SYNC_API_PORT:
            service_args.extend(["--http-port", str(args.http_port)])
        if args.verbose:
            service_args.append("-v")

        run_with_watchdog(
            service_args,
            check_interval=args.watchdog_interval,
            max_restarts=args.max_restarts,
        )
        return

    # Otherwise, run the service directly
    try:
        from app.distributed.unified_data_sync import (
            UnifiedDataSyncService,
        )
    except ImportError as e:
        logger.error(f"Failed to import unified_data_sync module: {e}")
        sys.exit(1)

    config_path = AI_SERVICE_ROOT / args.config
    hosts_path = AI_SERVICE_ROOT / args.hosts
    if args.hosts == DEFAULT_HOSTS_PATH and not hosts_path.exists():
        legacy_hosts = AI_SERVICE_ROOT / LEGACY_HOSTS_PATH
        if legacy_hosts.exists():
            hosts_path = legacy_hosts

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Check if sync is disabled on this machine
    import yaml
    with open(config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    di_config = raw_config.get("data_ingestion", {})
    if di_config.get("sync_disabled", False):
        logger.warning("=" * 60)
        logger.warning("DATA SYNC DISABLED on this machine (sync_disabled=true)")
        logger.warning("This machine is configured as orchestrator only.")
        logger.warning("Run unified_data_sync.py on a machine with sufficient storage.")
        logger.warning("=" * 60)
        logger.warning("Exiting. To override, use --force flag or set sync_disabled=false")
        if not args.force:
            sys.exit(0)
        logger.info("--force flag set, continuing anyway...")

    # Create service
    try:
        service = UnifiedDataSyncService.from_config(config_path, hosts_path)
    except Exception as e:
        logger.error(f"Failed to create service: {e}")
        sys.exit(1)

    if args.interval:
        service.config.poll_interval_seconds = args.interval

    service.http_port = args.http_port

    if args.dry_run:
        logger.info("Dry run - checking hosts...")
        for host in service.hosts.values():
            status = "ephemeral" if host.is_ephemeral else "persistent"
            logger.info(f"  {host.name}: {host.ssh_user}@{host.ssh_host}:{host.ssh_port} ({status})")
        logger.info("Components enabled:")
        logger.info(f"  Manifest replication: {service._manifest_replicator is not None}")
        logger.info(f"  P2P fallback: {service._p2p_fallback is not None}")
        logger.info(f"  Content dedup: {service._content_deduplicator is not None}")
        logger.info(f"  WAL: {service._ingestion_wal is not None}")
        # Aggregation mode status
        if service.config.enable_aggregation:
            is_agg = service._is_aggregator_node()
            logger.info(f"  Aggregation enabled: YES (this node is aggregator: {is_agg})")
            logger.info(f"    Aggregator node: {service.config.aggregator_node}")
            logger.info(f"    Aggregator DB: {service.config.aggregator_db}")
            logger.info(f"    Replica nodes: {len(service.config.replica_nodes)}")
            for replica in service.config.replica_nodes:
                logger.info(f"      - {replica.get('host')}: {replica.get('path')}")
        else:
            logger.info("  Aggregation enabled: NO")
        return

    # Signal handlers
    def signal_handler(sig, frame):
        logger.info("Shutdown requested")
        service.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    if args.once:
        asyncio.run(service.run_collection_cycle())
        logger.info(f"One-shot complete: {service._last_cycle_games} games synced")
    else:
        asyncio.run(service.run())


if __name__ == "__main__":
    main()
