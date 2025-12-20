#!/usr/bin/env python3
"""Data Aggregator Service - Centralized data collection and cloud sync.

This script runs on the designated data aggregation node (Mac Studio) to:
1. Collect game data from all cluster nodes (via unified_data_sync.py)
2. Store data on local high-capacity storage (OWC external drive)
3. Periodically sync to AWS S3 for cloud backup

Architecture:
    Cluster Nodes (GH200, Vast.ai, etc.)
            |
            v (rsync/P2P)
    Mac Studio (Data Aggregator)
      - /Volumes/RingRift-Data/selfplay_repository/
            |
            v (aws s3 sync)
    AWS S3 (ringrift-models-20251214)

Usage:
    # Run as daemon (primary mode)
    python scripts/data_aggregator.py

    # One-shot sync
    python scripts/data_aggregator.py --once

    # S3 sync only
    python scripts/data_aggregator.py --s3-only

    # Status check
    python scripts/data_aggregator.py --status

Requirements:
    - AWS CLI configured with credentials
    - OWC drive mounted at /Volumes/RingRift-Data
    - SSH access to cluster nodes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Determine paths
SCRIPT_DIR = Path(__file__).resolve().parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Unified resource checking (canonical source for resource limits)
try:
    from app.utils.resource_guard import get_disk_usage as resource_get_disk_usage
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    resource_get_disk_usage = None

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("data_aggregator")
# Default configuration
DEFAULT_CONFIG = {
    'aggregator': {
        'enabled': True,
        'local_storage': {
            'enabled': True,
            'base_path': '/Volumes/RingRift-Data',
        },
        's3_storage': {
            'enabled': True,
            'bucket': 'ringrift-models-20251214',
            'region': 'us-east-1',
            'sync_interval_minutes': 60,
        },
    },
    'data_collection': {
        'local_sync_dir': '/Volumes/RingRift-Data/selfplay_repository/raw/cluster_synced',
        'poll_interval_seconds': 60,
    },
}


class DataAggregator:
    """Centralized data aggregation service."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Statistics
        self._stats = {
            'cluster_syncs': 0,
            's3_syncs': 0,
            'total_games_collected': 0,
            'total_bytes_to_s3': 0,
            'last_cluster_sync': None,
            'last_s3_sync': None,
            'errors': [],
        }

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration and storage availability."""
        agg_config = self.config.get('aggregator', {})
        local_storage = agg_config.get('local_storage', {})

        if local_storage.get('enabled', False):
            base_path = Path(local_storage.get('base_path', '/Volumes/RingRift-Data'))
            if not base_path.exists():
                logger.warning(f"Local storage path not found: {base_path}")
                logger.warning("OWC drive may not be mounted")

        s3_storage = agg_config.get('s3_storage', {})
        if s3_storage.get('enabled', False):
            # Check AWS CLI availability
            try:
                result = subprocess.run(['aws', '--version'], capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning("AWS CLI not available, S3 sync disabled")
                    agg_config['s3_storage']['enabled'] = False
            except FileNotFoundError:
                logger.warning("AWS CLI not found, S3 sync disabled")
                agg_config['s3_storage']['enabled'] = False

    async def run_cluster_sync(self) -> int:
        """Run data sync from cluster nodes.

        Returns number of games synced.
        """
        logger.info("Starting cluster data sync...")

        # Get sync directory from config
        sync_dir = self.config.get('data_collection', {}).get(
            'local_sync_dir',
            '/Volumes/RingRift-Data/selfplay_repository/raw/cluster_synced'
        )
        sync_path = Path(sync_dir)
        sync_path.mkdir(parents=True, exist_ok=True)

        # Build command for unified_data_sync with custom output dir
        cmd = [
            sys.executable,
            str(AI_SERVICE_ROOT / 'scripts' / 'unified_data_sync.py'),
            '--once',
            '--force',  # Override sync_disabled since we're the aggregator
        ]

        try:
            # Set environment to use custom sync directory
            env = os.environ.copy()
            env['RINGRIFT_SYNC_DIR'] = str(sync_path)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(AI_SERVICE_ROOT),
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=600  # 10 minute timeout
            )

            if process.returncode == 0:
                # Parse output for game count
                output = stdout.decode()
                games = 0
                if 'games synced' in output.lower():
                    try:
                        # Extract number from "X games synced"
                        for line in output.split('\n'):
                            if 'games synced' in line.lower():
                                parts = line.split()
                                for _i, p in enumerate(parts):
                                    if p.isdigit():
                                        games = int(p)
                                        break
                    except (ValueError, IndexError):
                        pass

                self._stats['cluster_syncs'] += 1
                self._stats['total_games_collected'] += games
                self._stats['last_cluster_sync'] = datetime.now().isoformat()
                logger.info(f"Cluster sync complete: {games} games")
                return games
            else:
                error_msg = stderr.decode()[:200]
                logger.error(f"Cluster sync failed: {error_msg}")
                self._stats['errors'].append({
                    'time': datetime.now().isoformat(),
                    'type': 'cluster_sync',
                    'error': error_msg,
                })
                return 0

        except asyncio.TimeoutError:
            logger.error("Cluster sync timed out")
            return 0
        except Exception as e:
            logger.error(f"Cluster sync error: {e}")
            return 0

    async def run_s3_sync(self) -> dict[str, int]:
        """Sync data to AWS S3.

        Returns dict with upload/delete counts.
        """
        s3_config = self.config.get('aggregator', {}).get('s3_storage', {})
        if not s3_config.get('enabled', False):
            logger.debug("S3 sync disabled")
            return {'uploaded': 0, 'deleted': 0}

        bucket = s3_config.get('bucket', 'ringrift-models-20251214')
        logger.info(f"Starting S3 sync to s3://{bucket}/...")

        stats = {'uploaded': 0, 'deleted': 0}

        # Sync models
        models_path = AI_SERVICE_ROOT / 'models'
        if models_path.exists():
            uploaded, deleted = await self._s3_sync_directory(
                models_path, f's3://{bucket}/models/', ['*.pth', '*.pt']
            )
            stats['uploaded'] += uploaded
            stats['deleted'] += deleted

        # Sync game databases
        games_path = AI_SERVICE_ROOT / 'data' / 'games'
        if games_path.exists():
            uploaded, deleted = await self._s3_sync_directory(
                games_path, f's3://{bucket}/data/games/', ['*.db']
            )
            stats['uploaded'] += uploaded
            stats['deleted'] += deleted

        # Sync state files
        data_path = AI_SERVICE_ROOT / 'data'
        if data_path.exists():
            uploaded, deleted = await self._s3_sync_directory(
                data_path, f's3://{bucket}/data/state/', ['*.json']
            )
            stats['uploaded'] += uploaded
            stats['deleted'] += deleted

        self._stats['s3_syncs'] += 1
        self._stats['last_s3_sync'] = datetime.now().isoformat()
        logger.info(f"S3 sync complete: {stats['uploaded']} uploaded, {stats['deleted']} deleted")

        return stats

    async def _s3_sync_directory(
        self,
        local_path: Path,
        s3_uri: str,
        include_patterns: list,
    ) -> tuple:
        """Sync a directory to S3."""
        cmd = [
            'aws', 's3', 'sync',
            str(local_path),
            s3_uri,
            '--exclude', '*',
        ]

        for pattern in include_patterns:
            cmd.extend(['--include', pattern])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=600
            )

            output = stdout.decode()
            uploaded = output.count('upload:')
            deleted = output.count('delete:')

            if uploaded > 0 or deleted > 0:
                logger.debug(f"  {local_path.name}: {uploaded} uploaded, {deleted} deleted")

            return uploaded, deleted

        except Exception as e:
            logger.error(f"S3 sync error for {local_path}: {e}")
            return 0, 0

    def get_status(self) -> dict[str, Any]:
        """Get aggregator status."""
        agg_config = self.config.get('aggregator', {})
        local_storage = agg_config.get('local_storage', {})
        s3_storage = agg_config.get('s3_storage', {})

        status = {
            'running': self._running,
            'stats': self._stats.copy(),
            'config': {
                'local_storage_enabled': local_storage.get('enabled', False),
                'local_storage_path': local_storage.get('base_path', ''),
                's3_enabled': s3_storage.get('enabled', False),
                's3_bucket': s3_storage.get('bucket', ''),
            },
        }

        # Check disk space using unified resource guard
        if local_storage.get('enabled', False):
            base_path = Path(local_storage.get('base_path', ''))
            if base_path.exists():
                try:
                    if HAS_RESOURCE_GUARD and resource_get_disk_usage:
                        used_pct, free_gb, total_gb = resource_get_disk_usage(str(base_path))
                        status['disk'] = {
                            'total_gb': round(total_gb, 1),
                            'used_gb': round(total_gb - free_gb, 1),
                            'free_gb': round(free_gb, 1),
                            'percent_used': round(used_pct, 1),
                        }
                    else:
                        # Fallback to raw shutil
                        usage = shutil.disk_usage(base_path)
                        status['disk'] = {
                            'total_gb': round(usage.total / (1024**3), 1),
                            'used_gb': round(usage.used / (1024**3), 1),
                            'free_gb': round(usage.free / (1024**3), 1),
                            'percent_used': round(usage.used / usage.total * 100, 1),
                        }
                except Exception:
                    pass

        return status

    async def run(self):
        """Main aggregation loop."""
        self._running = True
        logger.info("Starting data aggregator service")

        agg_config = self.config.get('aggregator', {})
        data_config = self.config.get('data_collection', {})

        cluster_interval = data_config.get('poll_interval_seconds', 60)
        s3_interval = agg_config.get('s3_storage', {}).get('sync_interval_minutes', 60) * 60

        last_s3_sync = 0

        try:
            while self._running:
                cycle_start = time.time()

                # Run cluster sync
                try:
                    games = await self.run_cluster_sync()
                    if games > 0:
                        logger.info(f"Collected {games} games from cluster")
                except Exception as e:
                    logger.error(f"Cluster sync error: {e}")

                # Run S3 sync if interval elapsed
                if time.time() - last_s3_sync >= s3_interval:
                    try:
                        await self.run_s3_sync()
                        last_s3_sync = time.time()
                    except Exception as e:
                        logger.error(f"S3 sync error: {e}")

                # Wait for next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, cluster_interval - elapsed)

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=sleep_time
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass

        finally:
            self._running = False
            logger.info("Data aggregator stopped")

    def stop(self):
        """Request graceful shutdown."""
        self._running = False
        self._shutdown_event.set()


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from file or use defaults."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or DEFAULT_CONFIG
    return DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Data Aggregator Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs on the data aggregation node (Mac Studio) to:
1. Collect game data from all cluster nodes
2. Store data on local OWC drive
3. Sync to AWS S3 for cloud backup

Examples:
    # Run as daemon
    python scripts/data_aggregator.py

    # One-shot sync
    python scripts/data_aggregator.py --once

    # S3 sync only
    python scripts/data_aggregator.py --s3-only

    # Status check
    python scripts/data_aggregator.py --status
        """
    )
    parser.add_argument('--config', type=str, default='config/data_aggregator.yaml',
                        help='Config file path')
    parser.add_argument('--once', action='store_true',
                        help='Run one sync cycle and exit')
    parser.add_argument('--s3-only', action='store_true',
                        help='Only run S3 sync')
    parser.add_argument('--status', action='store_true',
                        help='Show aggregator status')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config_path = AI_SERVICE_ROOT / args.config
    config = load_config(config_path)

    # Create aggregator
    aggregator = DataAggregator(config)

    # Status check
    if args.status:
        status = aggregator.get_status()
        print(json.dumps(status, indent=2, default=str))
        return 0

    # Signal handlers
    def signal_handler(sig, frame):
        logger.info("Shutdown requested")
        aggregator.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    if args.s3_only:
        asyncio.run(aggregator.run_s3_sync())
    elif args.once:
        asyncio.run(aggregator.run_cluster_sync())
        asyncio.run(aggregator.run_s3_sync())
    else:
        asyncio.run(aggregator.run())

    return 0


if __name__ == "__main__":
    sys.exit(main())
