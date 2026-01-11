#!/usr/bin/env python3
"""Cluster-wide cleanup of orphan games (games without move data).

This script removes games that have metadata but no corresponding move data
across all cluster nodes, S3 buckets, and the OWC external drive.

January 2026: Created as part of data integrity enforcement initiative.

Usage:
    # Preview what would be cleaned (recommended first step)
    python scripts/cluster_cleanup_orphan_games.py --dry-run

    # Execute cleanup on all locations
    python scripts/cluster_cleanup_orphan_games.py --execute

    # Clean specific locations only
    python scripts/cluster_cleanup_orphan_games.py --execute --location local
    python scripts/cluster_cleanup_orphan_games.py --execute --location cluster
    python scripts/cluster_cleanup_orphan_games.py --execute --location s3
    python scripts/cluster_cleanup_orphan_games.py --execute --location owc

    # Verbose output
    python scripts/cluster_cleanup_orphan_games.py --execute --verbose

Locations:
    - local: Local databases in data/games/
    - cluster: All cluster nodes (Lambda, Vast.ai, RunPod, Nebius, Vultr)
    - s3: AWS S3 bucket ringrift-training-data
    - owc: OWC external drive on mac-studio

Requirements:
    - SSH keys configured for cluster access
    - AWS credentials for S3 access (optional)
    - SSH access to mac-studio for OWC drive (optional)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("cluster_cleanup")


# Minimum moves required for a game to be useful
MIN_MOVES_REQUIRED = 5

# S3 bucket and prefix for game databases
S3_BUCKET = "ringrift-training-data"
S3_PREFIX = "games/"

# OWC drive configuration on mac-studio
OWC_HOST = "mac-studio"
OWC_PATH = "/Volumes/RingRift-Data/selfplay_repository"

# Node path mappings by provider
NODE_PATH_MAPPINGS = {
    "runpod": "/workspace/ringrift/ai-service",
    "vast": "~/ringrift/ai-service",
    "nebius": "~/ringrift/ai-service",
    "vultr": "/root/ringrift/ai-service",
    "hetzner": "/root/ringrift/ai-service",
    "lambda": "~/ringrift/ai-service",
}


@dataclass
class CleanupStats:
    """Statistics for cleanup operation."""

    location: str = ""
    databases_processed: int = 0
    orphans_found: int = 0
    orphans_removed: int = 0
    games_preserved: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = True


def get_cluster_nodes() -> list[dict]:
    """Get cluster node configurations from distributed_hosts.yaml."""
    try:
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
        if not config_path.exists():
            logger.warning("distributed_hosts.yaml not found")
            return []

        with open(config_path) as f:
            config = yaml.safe_load(f)

        nodes = []
        # Note: YAML uses "hosts:" key, not "nodes:"
        hosts = config.get("hosts", {})
        if not hosts:
            logger.warning("No 'hosts' key found in distributed_hosts.yaml")
            return []

        for node_id, node_config in hosts.items():
            # Accept both "active" and "ready" status
            status = node_config.get("status", "")
            if status not in ("active", "ready"):
                continue
            # Skip local/coordinator nodes
            if node_id in ("local-mac", "mac-studio", "macbook-pro-3", "macbook-pro-4"):
                continue
            # Skip proxy-only nodes
            if node_config.get("role") == "proxy":
                continue

            # Determine path based on provider
            path = None
            for prefix, default_path in NODE_PATH_MAPPINGS.items():
                if node_id.startswith(prefix):
                    path = node_config.get("ringrift_path", default_path)
                    break

            if path:
                nodes.append({
                    "id": node_id,
                    "ssh_host": node_config.get("ssh_host") or node_config.get("tailscale_ip"),
                    "ssh_port": node_config.get("ssh_port", 22),
                    "ssh_user": node_config.get("ssh_user", "root"),
                    "path": path,
                })

        logger.info(f"Found {len(nodes)} cluster nodes")
        return nodes
    except Exception as e:
        logger.error(f"Error loading cluster config: {e}")
        return []


def find_local_databases(base_dir: Path) -> list[Path]:
    """Find all game databases in the local data directory."""
    databases = []
    games_dir = base_dir / "data" / "games"
    if games_dir.exists():
        databases.extend(games_dir.glob("*.db"))
        databases.extend(games_dir.glob("**/*.db"))

    # Filter out non-game databases
    databases = [
        db for db in databases
        if "jsonl" not in db.name
        and "sync" not in db.name
        and "elo" not in db.name
        and "registry" not in db.name
    ]
    return list(set(databases))


def count_orphan_games(db_path: Path) -> tuple[int, int]:
    """Count orphan games and total games in a database.

    Returns:
        Tuple of (orphan_count, total_games)
    """
    try:
        conn = sqlite3.connect(str(db_path))

        # Check if tables exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='games'"
        )
        if not cursor.fetchone():
            conn.close()
            return 0, 0

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'"
        )
        has_moves = cursor.fetchone() is not None

        # Count total games
        total = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]

        if not has_moves:
            # All games are orphans if no moves table
            conn.close()
            return total, total

        # Count orphan games (games with fewer than MIN_MOVES_REQUIRED moves)
        orphan_count = conn.execute(f"""
            SELECT COUNT(*) FROM games g
            LEFT JOIN (
                SELECT game_id, COUNT(*) as move_count
                FROM game_moves
                GROUP BY game_id
            ) m ON g.game_id = m.game_id
            WHERE m.move_count IS NULL OR m.move_count < {MIN_MOVES_REQUIRED}
        """).fetchone()[0]

        conn.close()
        return orphan_count, total

    except sqlite3.Error as e:
        logger.debug(f"Error counting orphans in {db_path}: {e}")
        return 0, 0


def cleanup_database(db_path: Path, dry_run: bool = True) -> tuple[int, int]:
    """Clean up orphan games from a database.

    Args:
        db_path: Path to the database
        dry_run: If True, don't actually delete

    Returns:
        Tuple of (orphans_removed, games_remaining)
    """
    try:
        conn = sqlite3.connect(str(db_path))

        # Check if tables exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'"
        )
        if not cursor.fetchone():
            conn.close()
            return 0, 0

        # Find orphan game IDs
        cursor = conn.execute(f"""
            SELECT g.game_id FROM games g
            LEFT JOIN (
                SELECT game_id, COUNT(*) as move_count
                FROM game_moves
                GROUP BY game_id
            ) m ON g.game_id = m.game_id
            WHERE m.move_count IS NULL OR m.move_count < {MIN_MOVES_REQUIRED}
        """)
        orphan_ids = [row[0] for row in cursor.fetchall()]

        if not orphan_ids:
            games_remaining = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            conn.close()
            return 0, games_remaining

        if not dry_run:
            # Delete in batches to avoid SQL variable limit
            batch_size = 500
            for i in range(0, len(orphan_ids), batch_size):
                batch = orphan_ids[i:i + batch_size]
                placeholders = ",".join("?" * len(batch))

                # Delete from all related tables first
                for table in ["game_moves", "game_initial_state", "game_state_snapshots",
                              "game_players", "game_choices", "game_history_entries"]:
                    try:
                        conn.execute(f"DELETE FROM {table} WHERE game_id IN ({placeholders})", batch)
                    except sqlite3.Error:
                        pass  # Table might not exist

                # Delete from games table
                conn.execute(f"DELETE FROM games WHERE game_id IN ({placeholders})", batch)

            conn.commit()

            # VACUUM to reclaim space
            try:
                conn.execute("VACUUM")
            except sqlite3.Error:
                pass  # VACUUM might fail in some cases

        games_remaining = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        conn.close()

        return len(orphan_ids), games_remaining

    except sqlite3.Error as e:
        logger.error(f"Error cleaning {db_path}: {e}")
        return 0, 0


def cleanup_local(base_dir: Path, dry_run: bool = True) -> CleanupStats:
    """Clean up orphan games from local databases."""
    stats = CleanupStats(location="local", dry_run=dry_run)

    databases = find_local_databases(base_dir)
    logger.info(f"Found {len(databases)} local databases")

    for db_path in databases:
        orphan_count, total = count_orphan_games(db_path)
        if orphan_count > 0:
            logger.info(f"  {db_path.name}: {orphan_count}/{total} orphans")
            removed, remaining = cleanup_database(db_path, dry_run)
            stats.orphans_found += orphan_count
            stats.orphans_removed += removed
            stats.games_preserved += remaining
            stats.databases_processed += 1
        else:
            stats.databases_processed += 1
            stats.games_preserved += total

    return stats


async def cleanup_cluster_node(node: dict, dry_run: bool = True) -> CleanupStats:
    """Clean up orphan games on a cluster node via SSH."""
    stats = CleanupStats(location=node["id"], dry_run=dry_run)

    ssh_host = node["ssh_host"]
    ssh_port = node["ssh_port"]
    ssh_user = node["ssh_user"]
    remote_path = node["path"]

    # Construct SSH command
    ssh_cmd = ["ssh", "-p", str(ssh_port), "-o", "ConnectTimeout=10",
               "-o", "StrictHostKeyChecking=no", f"{ssh_user}@{ssh_host}"]

    # Run the cleanup script on the remote node
    mode = "--dry-run" if dry_run else "--execute"
    remote_cmd = f"cd {remote_path} && python scripts/cleanup_games_without_moves.py {mode}"

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ssh_cmd + [remote_cmd],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            # Parse output for stats
            output = result.stdout
            logger.info(f"  {node['id']}: cleanup completed")
            stats.databases_processed = 1

            # Try to extract numbers from output
            for line in output.split("\n"):
                if "orphan" in line.lower() and "removed" in line.lower():
                    try:
                        # Try to extract number before "removed"
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.isdigit():
                                stats.orphans_removed += int(part)
                                break
                    except (ValueError, IndexError):
                        pass
        else:
            stats.errors.append(f"SSH failed: {result.stderr[:100]}")
            logger.warning(f"  {node['id']}: SSH failed - {result.stderr[:50]}")

    except subprocess.TimeoutExpired:
        stats.errors.append("SSH timeout")
        logger.warning(f"  {node['id']}: SSH timeout")
    except Exception as e:
        stats.errors.append(str(e))
        logger.warning(f"  {node['id']}: Error - {e}")

    return stats


async def cleanup_cluster(dry_run: bool = True) -> CleanupStats:
    """Clean up orphan games on all cluster nodes."""
    total_stats = CleanupStats(location="cluster", dry_run=dry_run)

    nodes = get_cluster_nodes()
    logger.info(f"Found {len(nodes)} cluster nodes")

    if not nodes:
        total_stats.errors.append("No cluster nodes found")
        return total_stats

    # Run cleanup on all nodes concurrently
    tasks = [cleanup_cluster_node(node, dry_run) for node in nodes]
    results = await asyncio.gather(*tasks)

    for result in results:
        total_stats.databases_processed += result.databases_processed
        total_stats.orphans_found += result.orphans_found
        total_stats.orphans_removed += result.orphans_removed
        total_stats.games_preserved += result.games_preserved
        total_stats.errors.extend(result.errors)

    return total_stats


def cleanup_s3(dry_run: bool = True) -> CleanupStats:
    """Clean up orphan games from S3 databases."""
    stats = CleanupStats(location="s3", dry_run=dry_run)

    try:
        import boto3
        s3 = boto3.client("s3")

        # List database files in S3
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
        db_keys = [
            obj["Key"] for obj in response.get("Contents", [])
            if obj["Key"].endswith(".db")
        ]

        logger.info(f"Found {len(db_keys)} databases in S3")

        for key in db_keys:
            try:
                # Download to temp file
                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                    temp_path = Path(f.name)

                s3.download_file(S3_BUCKET, key, str(temp_path))

                # Clean up
                orphan_count, total = count_orphan_games(temp_path)
                if orphan_count > 0:
                    logger.info(f"  s3://{S3_BUCKET}/{key}: {orphan_count}/{total} orphans")
                    removed, remaining = cleanup_database(temp_path, dry_run)
                    stats.orphans_found += orphan_count
                    stats.orphans_removed += removed
                    stats.games_preserved += remaining

                    if not dry_run and removed > 0:
                        # Re-upload cleaned database
                        s3.upload_file(str(temp_path), S3_BUCKET, key)
                        logger.info(f"    Uploaded cleaned {key}")

                stats.databases_processed += 1

                # Clean up temp file
                temp_path.unlink()

            except Exception as e:
                stats.errors.append(f"{key}: {e}")
                logger.warning(f"  Error processing {key}: {e}")

    except ImportError:
        stats.errors.append("boto3 not installed - skipping S3")
        logger.warning("boto3 not installed - skipping S3 cleanup")
    except Exception as e:
        stats.errors.append(str(e))
        logger.error(f"S3 error: {e}")

    return stats


def cleanup_owc(dry_run: bool = True) -> CleanupStats:
    """Clean up orphan games from OWC drive on mac-studio."""
    stats = CleanupStats(location="owc", dry_run=dry_run)

    # Find databases on OWC via SSH
    ssh_cmd = ["ssh", "-o", "ConnectTimeout=10", f"armand@{OWC_HOST}"]
    find_cmd = f"find {OWC_PATH} -name '*.db' -type f 2>/dev/null"

    try:
        result = subprocess.run(
            ssh_cmd + [find_cmd],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            stats.errors.append(f"SSH to {OWC_HOST} failed")
            logger.warning(f"Cannot reach OWC drive on {OWC_HOST}")
            return stats

        db_paths = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        logger.info(f"Found {len(db_paths)} databases on OWC")

        for remote_path in db_paths:
            try:
                # Download to temp file
                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                    temp_path = Path(f.name)

                scp_cmd = ["scp", f"armand@{OWC_HOST}:{remote_path}", str(temp_path)]
                subprocess.run(scp_cmd, capture_output=True, timeout=120)

                # Clean up
                orphan_count, total = count_orphan_games(temp_path)
                if orphan_count > 0:
                    db_name = Path(remote_path).name
                    logger.info(f"  owc:{db_name}: {orphan_count}/{total} orphans")
                    removed, remaining = cleanup_database(temp_path, dry_run)
                    stats.orphans_found += orphan_count
                    stats.orphans_removed += removed
                    stats.games_preserved += remaining

                    if not dry_run and removed > 0:
                        # Re-upload cleaned database
                        upload_cmd = ["scp", str(temp_path), f"armand@{OWC_HOST}:{remote_path}"]
                        subprocess.run(upload_cmd, capture_output=True, timeout=120)
                        logger.info(f"    Uploaded cleaned {db_name}")

                stats.databases_processed += 1

                # Clean up temp file
                temp_path.unlink()

            except subprocess.TimeoutExpired:
                stats.errors.append(f"{remote_path}: timeout")
            except Exception as e:
                stats.errors.append(f"{remote_path}: {e}")

    except subprocess.TimeoutExpired:
        stats.errors.append("SSH timeout")
        logger.warning(f"SSH timeout to {OWC_HOST}")
    except Exception as e:
        stats.errors.append(str(e))
        logger.error(f"OWC error: {e}")

    return stats


async def main():
    parser = argparse.ArgumentParser(
        description="Cluster-wide cleanup of orphan games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be cleaned (default)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform cleanup (required to modify data)",
    )
    parser.add_argument(
        "--location",
        choices=["local", "cluster", "s3", "owc", "all"],
        default="all",
        help="Location to clean (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dry_run = not args.execute

    if dry_run:
        logger.info("=== DRY RUN - No changes will be made ===")
    else:
        logger.info("=== EXECUTING CLEANUP - Changes will be made ===")

    base_dir = Path(__file__).parent.parent
    all_stats: list[CleanupStats] = []

    # Run cleanup for specified locations
    if args.location in ("local", "all"):
        logger.info("\n--- Local cleanup ---")
        stats = cleanup_local(base_dir, dry_run)
        all_stats.append(stats)

    if args.location in ("cluster", "all"):
        logger.info("\n--- Cluster cleanup ---")
        stats = await cleanup_cluster(dry_run)
        all_stats.append(stats)

    if args.location in ("s3", "all"):
        logger.info("\n--- S3 cleanup ---")
        stats = cleanup_s3(dry_run)
        all_stats.append(stats)

    if args.location in ("owc", "all"):
        logger.info("\n--- OWC cleanup ---")
        stats = cleanup_owc(dry_run)
        all_stats.append(stats)

    # Print summary
    print("\n" + "=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)

    total_orphans_found = 0
    total_orphans_removed = 0
    total_preserved = 0
    total_dbs = 0
    total_errors = 0

    for stats in all_stats:
        print(f"\n{stats.location.upper()}:")
        print(f"  Databases processed: {stats.databases_processed}")
        print(f"  Orphans found: {stats.orphans_found}")
        if not dry_run:
            print(f"  Orphans removed: {stats.orphans_removed}")
        print(f"  Games preserved: {stats.games_preserved}")
        if stats.errors:
            print(f"  Errors: {len(stats.errors)}")

        total_orphans_found += stats.orphans_found
        total_orphans_removed += stats.orphans_removed
        total_preserved += stats.games_preserved
        total_dbs += stats.databases_processed
        total_errors += len(stats.errors)

    print("\n" + "-" * 60)
    print("TOTALS:")
    print(f"  Databases: {total_dbs}")
    print(f"  Orphans found: {total_orphans_found}")
    if not dry_run:
        print(f"  Orphans removed: {total_orphans_removed}")
    print(f"  Games preserved: {total_preserved}")
    print(f"  Errors: {total_errors}")

    if dry_run and total_orphans_found > 0:
        print("\n" + "=" * 60)
        print("To actually perform cleanup, run with --execute flag")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
