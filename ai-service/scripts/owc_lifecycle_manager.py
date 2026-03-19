#!/usr/bin/env python3
"""OWC Drive Lifecycle Manager.

Automated disk space management for the OWC external drive on mac-studio.
Monitors free space, backs up data to S3, and cleans up old/redundant files.

Architecture:
    OWC Drive -> S3 Backup (verify) -> Local Deletion

Cleanup Priority (lowest value deleted first):
    1. Old model snapshots (timestamped canonical_*_YYYYMMDD_*.pth)
    2. Terminated/missing node selfplay repos
    3. Rotated gauntlet DB backups (*.bak)
    4. Old game copies (games/ dir)
    5. WAL/SHM journal files

Usage:
    # Check status (no deletions)
    python scripts/owc_lifecycle_manager.py --status

    # Dry run (show what would be cleaned)
    python scripts/owc_lifecycle_manager.py --dry-run

    # Run cleanup to reach target free space
    python scripts/owc_lifecycle_manager.py --target-free-gb 500

    # Daemon mode (check every 2 hours)
    python scripts/owc_lifecycle_manager.py --daemon

    # Force cleanup without S3 backup (for already-backed-up data)
    python scripts/owc_lifecycle_manager.py --skip-s3-verify

March 2026: Created after OWC drive filled to 100% causing 7-day pipeline stall.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OWC_BASE = Path("/Volumes/RingRift-Data")
S3_BUCKET = os.environ.get("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
DEFAULT_TARGET_FREE_GB = 500
DAEMON_INTERVAL_SECONDS = 7200  # 2 hours

# Nodes known to be terminated (update as needed)
TERMINATED_NODES = {
    "lambda-gh200-1", "lambda-gh200-2", "lambda-gh200-3",
    "lambda-gh200-4", "lambda-gh200-5",
    "lambda-gh200-training", "lambda-gh200-auto",
}

# Active nodes (update from distributed_hosts.yaml)
ACTIVE_NODES = {
    "lambda-gh200-8", "lambda-gh200-9", "lambda-gh200-10",
    "lambda-gh200-11", "lambda-gh200-12", "lambda-gh200-13",
    "lambda-gh200-14",
    "hetzner-cpu1", "hetzner-cpu2", "hetzner-cpu3",
    "mac-studio", "local-mac",
}


@dataclass
class CleanupCandidate:
    """A file or directory that can be cleaned up."""
    path: Path
    size_bytes: int
    priority: int  # Lower = cleaned first
    category: str
    description: str
    s3_prefix: str | None = None  # S3 path for backup verification
    needs_s3_backup: bool = True


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    freed_bytes: int = 0
    items_cleaned: int = 0
    items_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)


def get_owc_free_bytes() -> int | None:
    """Get free bytes on OWC drive."""
    if not OWC_BASE.exists():
        return None
    stat = os.statvfs(str(OWC_BASE))
    return stat.f_bavail * stat.f_frsize


def get_owc_total_bytes() -> int | None:
    """Get total bytes on OWC drive."""
    if not OWC_BASE.exists():
        return None
    stat = os.statvfs(str(OWC_BASE))
    return stat.f_blocks * stat.f_frsize


def bytes_to_gb(b: int) -> float:
    return b / (1024 ** 3)


def s3_object_exists(s3_path: str) -> bool:
    """Check if an object exists on S3."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_path],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except Exception:
        return False


def s3_sync_and_verify(local_path: Path, s3_prefix: str) -> bool:
    """Sync a local path to S3 and verify."""
    s3_uri = f"s3://{S3_BUCKET}/{s3_prefix}"
    try:
        if local_path.is_dir():
            cmd = [
                "aws", "s3", "sync", str(local_path), f"{s3_uri}/",
                "--storage-class", "STANDARD_IA", "--no-progress", "--quiet",
            ]
        else:
            cmd = [
                "aws", "s3", "cp", str(local_path), s3_uri,
                "--storage-class", "STANDARD_IA", "--no-progress", "--quiet",
            ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            logger.error(f"S3 sync failed for {local_path}: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"S3 sync timed out for {local_path}")
        return False
    except Exception as e:
        logger.error(f"S3 sync error for {local_path}: {e}")
        return False


def get_dir_size(path: Path) -> int:
    """Get total size of a directory."""
    total = 0
    try:
        for entry in os.scandir(str(path)):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(Path(entry.path))
    except (PermissionError, OSError):
        pass
    return total


def find_cleanup_candidates() -> list[CleanupCandidate]:
    """Find all files/directories that can be cleaned up."""
    candidates: list[CleanupCandidate] = []

    # Priority 1: Old model snapshots with timestamps
    models_dir = OWC_BASE / "canonical_models"
    if models_dir.exists():
        timestamp_pattern = re.compile(r"_\d{8}_\d{6}\.pth$")
        for f in models_dir.iterdir():
            if f.is_file() and timestamp_pattern.search(f.name):
                candidates.append(CleanupCandidate(
                    path=f,
                    size_bytes=f.stat().st_size,
                    priority=1,
                    category="old_model_snapshot",
                    description=f"Old model snapshot: {f.name}",
                    s3_prefix=f"owc/canonical_models/{f.name}",
                    needs_s3_backup=True,
                ))

    # Priority 2: Terminated node selfplay repos
    repo_dir = OWC_BASE / "selfplay_repository"
    if repo_dir.exists():
        for node_dir in repo_dir.iterdir():
            if node_dir.is_dir() and node_dir.name in TERMINATED_NODES:
                size = get_dir_size(node_dir)
                candidates.append(CleanupCandidate(
                    path=node_dir,
                    size_bytes=size,
                    priority=2,
                    category="terminated_node",
                    description=f"Terminated node data: {node_dir.name} ({bytes_to_gb(size):.1f}GB)",
                    s3_prefix=f"owc/selfplay_repository/{node_dir.name}",
                    needs_s3_backup=True,
                ))
        # Also check for nodes not in ACTIVE or TERMINATED (unknown)
        for node_dir in repo_dir.iterdir():
            if node_dir.is_dir() and node_dir.name not in ACTIVE_NODES and node_dir.name not in TERMINATED_NODES:
                size = get_dir_size(node_dir)
                candidates.append(CleanupCandidate(
                    path=node_dir,
                    size_bytes=size,
                    priority=2,
                    category="unknown_node",
                    description=f"Unknown node data: {node_dir.name} ({bytes_to_gb(size):.1f}GB)",
                    s3_prefix=f"owc/selfplay_repository/{node_dir.name}",
                    needs_s3_backup=True,
                ))

    # Priority 3: Rotated backup files (.bak)
    for search_dir in [OWC_BASE / "ai-service-live" / "data" / "games", OWC_BASE / "games"]:
        if search_dir.exists():
            for f in search_dir.rglob("*.bak"):
                if f.is_file():
                    candidates.append(CleanupCandidate(
                        path=f,
                        size_bytes=f.stat().st_size,
                        priority=3,
                        category="rotated_backup",
                        description=f"Rotated backup: {f.name} ({bytes_to_gb(f.stat().st_size):.1f}GB)",
                        needs_s3_backup=False,  # These are already backups of backups
                    ))

    # Priority 4: WAL/SHM journal files (transient, safe to delete)
    for search_dir in [OWC_BASE]:
        if search_dir.exists():
            for pattern in ["**/*-wal", "**/*-shm"]:
                for f in search_dir.glob(pattern):
                    if f.is_file():
                        candidates.append(CleanupCandidate(
                            path=f,
                            size_bytes=f.stat().st_size,
                            priority=4,
                            category="journal_file",
                            description=f"Journal file: {f.name}",
                            needs_s3_backup=False,
                        ))

    # Priority 5: selfplay_local JSONL files older than 30 days
    local_dir = OWC_BASE / "selfplay_local"
    if local_dir.exists():
        cutoff = time.time() - (30 * 86400)
        for f in local_dir.iterdir():
            if not f.is_file() or f.stat().st_mtime >= cutoff:
                continue
            if f.suffix == ".partial":
                # .partial files are incomplete writes — safe to delete without backup
                candidates.append(CleanupCandidate(
                    path=f,
                    size_bytes=f.stat().st_size,
                    priority=4,  # Higher priority (safe to delete)
                    category="incomplete_selfplay",
                    description=f"Incomplete selfplay: {f.name} ({bytes_to_gb(f.stat().st_size):.1f}GB)",
                    needs_s3_backup=False,
                ))
            elif f.suffix == ".jsonl":
                # Complete JSONL files MUST be backed up to S3 before deletion
                candidates.append(CleanupCandidate(
                    path=f,
                    size_bytes=f.stat().st_size,
                    priority=5,
                    category="old_selfplay_local",
                    description=f"Old selfplay JSONL: {f.name} ({bytes_to_gb(f.stat().st_size):.1f}GB)",
                    s3_prefix=f"owc/selfplay_local/{f.name}",
                    needs_s3_backup=True,
                ))

    # Sort by priority, then size (largest first within same priority)
    candidates.sort(key=lambda c: (c.priority, -c.size_bytes))
    return candidates


def show_status():
    """Show OWC drive status."""
    if not OWC_BASE.exists():
        logger.error(f"OWC drive not mounted at {OWC_BASE}")
        return

    free = get_owc_free_bytes()
    total = get_owc_total_bytes()
    used = total - free if total and free else 0
    pct = (used / total * 100) if total else 0

    print(f"\n{'='*60}")
    print(f"OWC Drive Status: {OWC_BASE}")
    print(f"{'='*60}")
    print(f"Total:     {bytes_to_gb(total):.1f} GB")
    print(f"Used:      {bytes_to_gb(used):.1f} GB ({pct:.1f}%)")
    print(f"Free:      {bytes_to_gb(free):.1f} GB")
    print()

    # Show top-level directory sizes
    print("Directory breakdown:")
    dirs = []
    for entry in sorted(OWC_BASE.iterdir()):
        if entry.is_dir():
            size = get_dir_size(entry)
            if size > 1024 ** 3:  # Only show > 1GB
                dirs.append((entry.name, size))
    dirs.sort(key=lambda x: -x[1])
    for name, size in dirs:
        print(f"  {name:40s} {bytes_to_gb(size):>8.1f} GB")

    # Show cleanup candidates
    candidates = find_cleanup_candidates()
    if candidates:
        total_reclaimable = sum(c.size_bytes for c in candidates)
        print(f"\nCleanup candidates ({len(candidates)} items, {bytes_to_gb(total_reclaimable):.1f} GB reclaimable):")
        by_category = {}
        for c in candidates:
            by_category.setdefault(c.category, []).append(c)
        for cat, items in sorted(by_category.items(), key=lambda x: x[1][0].priority):
            cat_total = sum(i.size_bytes for i in items)
            print(f"  [{items[0].priority}] {cat}: {len(items)} items, {bytes_to_gb(cat_total):.1f} GB")
    print()


def run_cleanup(
    target_free_gb: float = DEFAULT_TARGET_FREE_GB,
    dry_run: bool = False,
    skip_s3_verify: bool = False,
) -> CleanupResult:
    """Run cleanup to reach target free space."""
    result = CleanupResult()

    if not OWC_BASE.exists():
        result.errors.append(f"OWC drive not mounted at {OWC_BASE}")
        return result

    free = get_owc_free_bytes()
    target_free = int(target_free_gb * (1024 ** 3))

    if free >= target_free:
        logger.info(
            f"OWC has {bytes_to_gb(free):.1f}GB free, "
            f"target is {target_free_gb:.0f}GB. No cleanup needed."
        )
        return result

    needed = target_free - free
    logger.info(
        f"Need to free {bytes_to_gb(needed):.1f}GB "
        f"(current: {bytes_to_gb(free):.1f}GB, target: {target_free_gb:.0f}GB)"
    )

    candidates = find_cleanup_candidates()
    freed = 0

    for candidate in candidates:
        if freed >= needed:
            break

        action = "DRY RUN" if dry_run else "CLEANING"
        logger.info(f"[{action}] {candidate.description}")

        if candidate.needs_s3_backup:
            if skip_s3_verify:
                # Even with --skip-s3-verify, NEVER delete items that need
                # S3 backup without actually verifying. This flag only skips
                # the slow sync — it still won't delete unbackable data.
                # Only items with needs_s3_backup=False are safe to delete
                # without any verification (journal files, .bak files, etc.)
                logger.info(
                    f"  SKIPPING (needs S3 backup, --skip-s3-verify): "
                    f"{candidate.path.name}"
                )
                result.items_skipped += 1
                continue
            if candidate.s3_prefix:
                logger.info(f"  Backing up to S3: {candidate.s3_prefix}")
                if not dry_run:
                    if not s3_sync_and_verify(candidate.path, candidate.s3_prefix):
                        logger.warning(f"  S3 backup failed, SKIPPING deletion")
                        result.items_skipped += 1
                        result.errors.append(f"S3 backup failed: {candidate.path}")
                        continue

        if not dry_run:
            try:
                if candidate.path.is_dir():
                    shutil.rmtree(candidate.path)
                else:
                    candidate.path.unlink()
                freed += candidate.size_bytes
                result.freed_bytes += candidate.size_bytes
                result.items_cleaned += 1
                result.details.append(
                    f"Deleted {candidate.path} ({bytes_to_gb(candidate.size_bytes):.1f}GB)"
                )
                logger.info(
                    f"  Freed {bytes_to_gb(candidate.size_bytes):.1f}GB "
                    f"(total: {bytes_to_gb(freed):.1f}/{bytes_to_gb(needed):.1f}GB)"
                )
            except Exception as e:
                result.errors.append(f"Failed to delete {candidate.path}: {e}")
                logger.error(f"  Delete failed: {e}")
        else:
            freed += candidate.size_bytes
            result.freed_bytes += candidate.size_bytes
            result.items_cleaned += 1

    if freed < needed:
        remaining = needed - freed
        logger.warning(
            f"Could not free enough space. "
            f"Freed {bytes_to_gb(freed):.1f}GB, "
            f"still need {bytes_to_gb(remaining):.1f}GB"
        )

    return result


def daemon_loop(target_free_gb: float):
    """Run as a daemon, checking periodically."""
    logger.info(
        f"Starting OWC lifecycle daemon "
        f"(target: {target_free_gb:.0f}GB free, interval: {DAEMON_INTERVAL_SECONDS}s)"
    )
    while True:
        try:
            free = get_owc_free_bytes()
            if free is None:
                logger.warning("OWC drive not mounted, skipping cycle")
            else:
                target_free = int(target_free_gb * (1024 ** 3))
                if free < target_free:
                    logger.info(
                        f"OWC free space ({bytes_to_gb(free):.1f}GB) "
                        f"below target ({target_free_gb:.0f}GB), running cleanup"
                    )
                    result = run_cleanup(target_free_gb=target_free_gb)
                    logger.info(
                        f"Cleanup result: freed {bytes_to_gb(result.freed_bytes):.1f}GB, "
                        f"{result.items_cleaned} items, {len(result.errors)} errors"
                    )
                else:
                    logger.info(
                        f"OWC OK: {bytes_to_gb(free):.1f}GB free "
                        f"(target: {target_free_gb:.0f}GB)"
                    )
        except Exception as e:
            logger.error(f"Daemon cycle error: {e}", exc_info=True)

        time.sleep(DAEMON_INTERVAL_SECONDS)


def main():
    parser = argparse.ArgumentParser(description="OWC Drive Lifecycle Manager")
    parser.add_argument("--status", action="store_true", help="Show drive status")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned")
    parser.add_argument(
        "--target-free-gb", type=float, default=DEFAULT_TARGET_FREE_GB,
        help=f"Target free space in GB (default: {DEFAULT_TARGET_FREE_GB})",
    )
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument(
        "--skip-s3-verify", action="store_true",
        help="Skip slow S3 sync/verify. Items that need S3 backup will be "
             "SKIPPED (not deleted). Only safe items (journals, .bak files, "
             "incomplete writes) will be cleaned. Use for fast emergency cleanup.",
    )
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.daemon:
        daemon_loop(args.target_free_gb)
        return

    result = run_cleanup(
        target_free_gb=args.target_free_gb,
        dry_run=args.dry_run,
        skip_s3_verify=args.skip_s3_verify,
    )

    print(f"\n{'='*40}")
    print(f"Cleanup {'(DRY RUN) ' if args.dry_run else ''}Summary")
    print(f"{'='*40}")
    print(f"Items cleaned: {result.items_cleaned}")
    print(f"Items skipped: {result.items_skipped}")
    print(f"Space freed:   {bytes_to_gb(result.freed_bytes):.1f} GB")
    if result.errors:
        print(f"Errors:        {len(result.errors)}")
        for err in result.errors:
            print(f"  - {err}")
    if result.details:
        print("\nDetails:")
        for d in result.details:
            print(f"  {d}")

    free = get_owc_free_bytes()
    if free:
        print(f"\nOWC free space: {bytes_to_gb(free):.1f} GB")


if __name__ == "__main__":
    main()
