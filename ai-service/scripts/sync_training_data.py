#!/usr/bin/env python3
"""Sync training data from OWC/S3 before training.

This script ensures the best available training data is present locally
before training starts. It checks OWC external drive and S3 bucket for
training data and downloads the largest available dataset.

Usage:
    # Sync data for a specific config
    python scripts/sync_training_data.py --config hex8_2p

    # Sync data for all available configs
    python scripts/sync_training_data.py --all

    # Show manifest without syncing
    python scripts/sync_training_data.py --show-manifest

    # Force re-download even if local file exists
    python scripts/sync_training_data.py --config hex8_2p --force
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add ai-service root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.coordination.training_data_manifest import (
    DataSource,
    get_training_data_manifest,
)
from app.coordination.training_data_sync_daemon import (
    LOCAL_TRAINING_DIR,
    SyncResult,
    sync_training_data_for_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def show_manifest() -> None:
    """Display the training data manifest."""
    manifest = await get_training_data_manifest()

    print("\n" + "=" * 80)
    print("Training Data Manifest")
    print("=" * 80)

    if manifest.last_refresh:
        print(f"Last refreshed: {manifest.last_refresh}")
    print()

    summary = manifest.get_summary()
    if not summary:
        print("No training data found in manifest.")
        return

    # Sort by config key
    for config_key in sorted(summary.keys()):
        data = summary[config_key]
        sources = ", ".join(data["sources"])
        print(
            f"{config_key:15} | {data['count']:2} files | "
            f"Best: {data['best_size_mb']:>7.1f}MB from {data['best_source']:<5} | "
            f"Sources: {sources}"
        )

    print()
    print("=" * 80)


async def sync_config(config_key: str, force: bool = False) -> SyncResult:
    """Sync training data for a single config."""
    print(f"\nSyncing training data for {config_key}...")
    result = await sync_training_data_for_config(config_key, force=force)

    if result.success:
        if result.skipped_reason:
            print(f"  Skipped: {result.skipped_reason}")
        elif result.bytes_transferred > 0:
            size_mb = result.bytes_transferred / (1024 * 1024)
            print(
                f"  Downloaded {size_mb:.1f}MB from {result.source.value} "
                f"in {result.duration_seconds:.1f}s"
            )
            print(f"  Local path: {result.local_path}")
        else:
            print(f"  Using existing local file: {result.local_path}")
    else:
        print(f"  FAILED: {result.error}")

    return result


async def sync_all_configs(force: bool = False) -> None:
    """Sync training data for all configs in manifest."""
    manifest = await get_training_data_manifest()

    configs = manifest.get_configs()
    if not configs:
        print("No configs found in manifest.")
        return

    print(f"\nSyncing {len(configs)} configs...")

    successes = 0
    failures = 0
    skipped = 0
    total_bytes = 0

    for config_key in sorted(configs):
        result = await sync_config(config_key, force=force)
        if result.success:
            if result.skipped_reason:
                skipped += 1
            else:
                successes += 1
                total_bytes += result.bytes_transferred
        else:
            failures += 1

    print("\n" + "=" * 80)
    print("Sync Summary")
    print("=" * 80)
    print(f"Downloaded: {successes} configs ({total_bytes / (1024 * 1024):.1f}MB)")
    print(f"Skipped:    {skipped} configs (already up-to-date)")
    print(f"Failed:     {failures} configs")


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sync training data from OWC/S3 before training"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Config key to sync (e.g., hex8_2p)",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Sync all available configs",
    )
    parser.add_argument(
        "--show-manifest",
        "-m",
        action="store_true",
        help="Show manifest without syncing",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if local file exists",
    )
    parser.add_argument(
        "--refresh-manifest",
        "-r",
        action="store_true",
        help="Force refresh of manifest from all sources",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=LOCAL_TRAINING_DIR,
        help=f"Local training data directory (default: {LOCAL_TRAINING_DIR})",
    )

    args = parser.parse_args()

    # Force refresh if requested
    if args.refresh_manifest:
        print("Refreshing manifest from all sources...")
        manifest = await get_training_data_manifest(refresh_if_stale_hours=0)
        result = await manifest.refresh_all()
        print(f"Found: {result}")

    # Show manifest
    if args.show_manifest:
        await show_manifest()
        return 0

    # Sync specific config or all
    if args.config:
        result = await sync_config(args.config, force=args.force)
        return 0 if result.success else 1
    elif args.all:
        await sync_all_configs(force=args.force)
        return 0
    else:
        # Default: show manifest
        await show_manifest()
        print("\nUse --config <config_key> to sync a specific config, or --all to sync all.")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
