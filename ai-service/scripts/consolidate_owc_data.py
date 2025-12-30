#!/usr/bin/env python3
"""Consolidate training data from OWC external drive into canonical databases.

This script:
1. Discovers databases on the OWC drive (mac-studio:/Volumes/RingRift-Data)
2. Syncs relevant databases to local staging directory
3. Consolidates games into canonical_*.db format
4. Optionally syncs to cluster and exports to NPZ

Usage:
    # Discover what's available on OWC
    python scripts/consolidate_owc_data.py --discover

    # Full consolidation pipeline
    python scripts/consolidate_owc_data.py --consolidate

    # Full pipeline with cluster sync and NPZ export
    python scripts/consolidate_owc_data.py --full-pipeline

    # Specific configs only
    python scripts/consolidate_owc_data.py --consolidate --configs hexagonal_4p,square19_4p

Environment Variables:
    OWC_HOST: OWC host (default: mac-studio)
    OWC_USER: SSH user (default: armand)
    OWC_BASE_PATH: OWC mount path (default: /Volumes/RingRift-Data)
    RINGRIFT_DATA_DIR: Local data directory (default: data/games)

December 2025: Created for training data pipeline remediation.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

OWC_HOST = os.getenv("OWC_HOST", "mac-studio")
OWC_USER = os.getenv("OWC_USER", "armand")
OWC_BASE_PATH = os.getenv("OWC_BASE_PATH", "/Volumes/RingRift-Data")
OWC_SSH_KEY = os.getenv("OWC_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))

LOCAL_DATA_DIR = Path(os.getenv("RINGRIFT_DATA_DIR", "data/games"))
OWC_STAGING_DIR = LOCAL_DATA_DIR / "owc_imports"

# Critical configs that need data (December 2025)
CRITICAL_CONFIGS = [
    "hexagonal_4p",
    "square19_4p",
    "hexagonal_3p",
    "square19_3p",
    "square19_2p",
]

# Known OWC database paths with good training data
OWC_SOURCE_DATABASES = [
    "selfplay_repository/consolidated_archives/synced_20251213_182629_vast_2x5090_selfplay.db",
    "selfplay_repository/consolidated_archives/synced_20251213_172954_vast-5090-quad_selfplay.db",
    "selfplay_repository/consolidated_archives/synced_20251213_182629_lambda-h100_selfplay.db",
    "training_data/coordinator_backup/sq19_4p_selfplay.db",
    "training_data/coordinator_backup/sq19_2p_selfplay.db",
]


@dataclass
class OWCDatabase:
    """Information about a database on the OWC drive."""
    path: str
    size_bytes: int = 0
    configs: dict[str, int] = field(default_factory=dict)  # config_key -> game_count


@dataclass
class ConsolidationResult:
    """Result of consolidation for a config."""
    config_key: str
    success: bool
    games_before: int = 0
    games_after: int = 0
    games_added: int = 0
    error: str | None = None


# ============================================================================
# OWC Discovery
# ============================================================================


def run_ssh_command(command: str, timeout: int = 60) -> tuple[bool, str]:
    """Run SSH command on OWC host."""
    ssh_cmd = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-i", OWC_SSH_KEY,
        f"{OWC_USER}@{OWC_HOST}",
        command,
    ]

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def discover_owc_databases() -> list[OWCDatabase]:
    """Discover databases on OWC drive and their contents."""
    logger.info(f"Discovering databases on {OWC_HOST}:{OWC_BASE_PATH}...")

    databases: list[OWCDatabase] = []

    for rel_path in OWC_SOURCE_DATABASES:
        full_path = f"{OWC_BASE_PATH}/{rel_path}"
        db = OWCDatabase(path=rel_path)

        # Get file size
        success, output = run_ssh_command(f"stat -f%z '{full_path}' 2>/dev/null || stat -c%s '{full_path}' 2>/dev/null")
        if success:
            try:
                db.size_bytes = int(output)
            except ValueError:
                pass

        # Get game counts per config
        query = """
            SELECT board_type || '_' || num_players || 'p' as config, COUNT(*)
            FROM games
            WHERE winner IS NOT NULL
            GROUP BY board_type, num_players
        """
        success, output = run_ssh_command(
            f"sqlite3 '{full_path}' \"{query}\"",
            timeout=30,
        )

        if success and output:
            for line in output.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) == 2:
                        config_key, count = parts
                        try:
                            db.configs[config_key] = int(count)
                        except ValueError:
                            pass

        if db.configs:
            databases.append(db)
            logger.info(f"  {rel_path}: {sum(db.configs.values())} games")
            for config, count in sorted(db.configs.items()):
                if config.replace("_", "").replace("p", "").replace("2", "").replace("3", "").replace("4", "") in ["hexagonal", "square19"]:
                    logger.info(f"    - {config}: {count}")

    return databases


def get_owc_totals(databases: list[OWCDatabase]) -> dict[str, int]:
    """Get total game counts per config across all OWC databases."""
    totals: dict[str, int] = {}
    for db in databases:
        for config, count in db.configs.items():
            totals[config] = totals.get(config, 0) + count
    return totals


# ============================================================================
# Sync from OWC
# ============================================================================


def sync_database_from_owc(rel_path: str) -> Path | None:
    """Sync a single database from OWC to local staging."""
    OWC_STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # Create local filename from path
    local_name = rel_path.replace("/", "_")
    local_path = OWC_STAGING_DIR / local_name

    remote_path = f"{OWC_USER}@{OWC_HOST}:{OWC_BASE_PATH}/{rel_path}"

    logger.info(f"Syncing {rel_path} -> {local_path.name}...")

    rsync_cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e", f"ssh -i {OWC_SSH_KEY} -o StrictHostKeyChecking=no",
        remote_path,
        str(local_path),
    ]

    try:
        result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            logger.info(f"  Synced: {local_path.stat().st_size / 1024 / 1024:.1f} MB")
            return local_path
        else:
            logger.error(f"  Rsync failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        logger.error(f"  Rsync timed out")
        return None
    except Exception as e:
        logger.error(f"  Rsync error: {e}")
        return None


def sync_all_owc_databases(databases: list[OWCDatabase]) -> list[Path]:
    """Sync all relevant OWC databases to local staging."""
    synced: list[Path] = []

    for db in databases:
        # Only sync databases that have critical config games
        has_critical = any(
            config in CRITICAL_CONFIGS and count > 0
            for config, count in db.configs.items()
        )

        if not has_critical:
            logger.info(f"Skipping {db.path} (no critical config games)")
            continue

        local_path = sync_database_from_owc(db.path)
        if local_path:
            synced.append(local_path)

    return synced


# ============================================================================
# Consolidation
# ============================================================================


def get_local_canonical_count(config_key: str) -> int:
    """Get current game count in local canonical database."""
    parts = config_key.rsplit("_", 1)
    if len(parts) != 2:
        return 0

    board_type = parts[0]
    num_players = int(parts[1].replace("p", ""))

    canonical_path = LOCAL_DATA_DIR / f"canonical_{board_type}_{num_players}p.db"

    if not canonical_path.exists():
        return 0

    try:
        with sqlite3.connect(str(canonical_path)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM games WHERE winner IS NOT NULL")
            return cursor.fetchone()[0]
    except sqlite3.Error:
        return 0


def consolidate_config(config_key: str, source_dbs: list[Path]) -> ConsolidationResult:
    """Consolidate a single config from source databases."""
    result = ConsolidationResult(config_key=config_key, success=False)

    parts = config_key.rsplit("_", 1)
    if len(parts) != 2:
        result.error = f"Invalid config key: {config_key}"
        return result

    board_type = parts[0]
    num_players = int(parts[1].replace("p", ""))

    canonical_path = LOCAL_DATA_DIR / f"canonical_{board_type}_{num_players}p.db"
    result.games_before = get_local_canonical_count(config_key)

    logger.info(f"Consolidating {config_key} (currently {result.games_before} games)...")

    # Use the consolidation daemon's merge logic
    try:
        from app.coordination.data_consolidation_daemon import (
            DataConsolidationDaemon,
            ConsolidationConfig,
        )

        # Create daemon with staging dir as source
        config = ConsolidationConfig(
            data_dir=OWC_STAGING_DIR,
            canonical_dir=LOCAL_DATA_DIR,
            min_games_for_consolidation=1,  # Accept any number
        )

        daemon = DataConsolidationDaemon(config=config)

        # Run consolidation synchronously
        loop = asyncio.new_event_loop()
        try:
            stats = loop.run_until_complete(daemon._consolidate_config(board_type, num_players))

            result.games_after = get_local_canonical_count(config_key)
            result.games_added = result.games_after - result.games_before
            result.success = stats.success

            if not stats.success:
                result.error = stats.error

            logger.info(
                f"  {config_key}: +{result.games_added} games "
                f"({result.games_before} -> {result.games_after})"
            )

        finally:
            loop.close()

    except Exception as e:
        result.error = str(e)
        logger.error(f"  Consolidation error: {e}")

    return result


# ============================================================================
# Post-Consolidation
# ============================================================================


def sync_canonical_to_cluster(config_key: str) -> bool:
    """Sync canonical database to cluster nodes."""
    parts = config_key.rsplit("_", 1)
    if len(parts) != 2:
        return False

    board_type = parts[0]
    num_players = int(parts[1].replace("p", ""))

    canonical_path = LOCAL_DATA_DIR / f"canonical_{board_type}_{num_players}p.db"

    if not canonical_path.exists():
        logger.warning(f"Canonical database not found: {canonical_path}")
        return False

    logger.info(f"Syncing {canonical_path.name} to cluster...")

    try:
        # Use the sync infrastructure
        from app.coordination.auto_sync_daemon import AutoSyncDaemon

        daemon = AutoSyncDaemon.get_instance()

        # Trigger sync for this database
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(daemon.sync_file(str(canonical_path)))
            logger.info(f"  Synced {canonical_path.name} to cluster")
            return True
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"  Sync error: {e}")
        return False


def export_npz_for_config(config_key: str) -> bool:
    """Export training NPZ for a config."""
    parts = config_key.rsplit("_", 1)
    if len(parts) != 2:
        return False

    board_type = parts[0]
    num_players = int(parts[1].replace("p", ""))

    output_path = Path("data/training") / f"{config_key}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting NPZ for {config_key}...")

    try:
        cmd = [
            sys.executable,
            "scripts/export_replay_dataset.py",
            "--use-discovery",
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--output", str(output_path),
            "--allow-pending-gate",  # OWC data may not have parity gates
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode == 0 and output_path.exists():
            size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"  Exported: {output_path} ({size_mb:.1f} MB)")
            return True
        else:
            logger.error(f"  Export failed: {result.stderr[:500]}")
            return False

    except Exception as e:
        logger.error(f"  Export error: {e}")
        return False


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Consolidate OWC training data")
    parser.add_argument("--discover", action="store_true", help="Discover what's on OWC")
    parser.add_argument("--consolidate", action="store_true", help="Sync and consolidate")
    parser.add_argument("--full-pipeline", action="store_true", help="Full pipeline including cluster sync and NPZ export")
    parser.add_argument("--configs", type=str, help="Comma-separated configs (default: critical configs)")
    parser.add_argument("--skip-sync", action="store_true", help="Skip OWC sync (use existing staged files)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")

    args = parser.parse_args()

    if not any([args.discover, args.consolidate, args.full_pipeline]):
        parser.print_help()
        return 1

    # Parse configs
    configs = CRITICAL_CONFIGS
    if args.configs:
        configs = [c.strip() for c in args.configs.split(",")]

    # Discovery
    databases = discover_owc_databases()

    if not databases:
        logger.error("No databases found on OWC drive")
        return 1

    # Show totals
    totals = get_owc_totals(databases)
    logger.info("\n=== OWC Data Summary ===")
    for config in sorted(configs):
        owc_count = totals.get(config, 0)
        local_count = get_local_canonical_count(config)
        logger.info(f"  {config}: OWC={owc_count}, Local={local_count}, Potential=+{max(0, owc_count - local_count)}")

    if args.discover:
        return 0

    if args.dry_run:
        logger.info("\n[DRY RUN] Would sync and consolidate the above databases")
        return 0

    # Sync from OWC
    if not args.skip_sync:
        logger.info("\n=== Syncing from OWC ===")
        synced_dbs = sync_all_owc_databases(databases)
        logger.info(f"Synced {len(synced_dbs)} databases")
    else:
        synced_dbs = list(OWC_STAGING_DIR.glob("*.db")) if OWC_STAGING_DIR.exists() else []
        logger.info(f"Using {len(synced_dbs)} existing staged databases")

    if not synced_dbs:
        logger.error("No databases to consolidate")
        return 1

    # Consolidate
    logger.info("\n=== Consolidating ===")
    results: list[ConsolidationResult] = []

    for config in configs:
        result = consolidate_config(config, synced_dbs)
        results.append(result)

    # Summary
    logger.info("\n=== Consolidation Results ===")
    total_added = 0
    for r in results:
        status = "OK" if r.success else f"FAILED: {r.error}"
        logger.info(f"  {r.config_key}: +{r.games_added} ({r.games_before} -> {r.games_after}) [{status}]")
        total_added += r.games_added

    logger.info(f"\nTotal games added: {total_added}")

    # Full pipeline
    if args.full_pipeline:
        logger.info("\n=== Syncing to Cluster ===")
        for r in results:
            if r.success and r.games_added > 0:
                sync_canonical_to_cluster(r.config_key)

        logger.info("\n=== Exporting NPZ ===")
        for r in results:
            if r.success and r.games_added > 0:
                export_npz_for_config(r.config_key)

    return 0


if __name__ == "__main__":
    sys.exit(main())
