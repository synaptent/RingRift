#!/usr/bin/env python3
"""Disk monitoring and cleanup for RingRift cluster nodes.

Aggressive disk cleanup for space-constrained nodes (especially Vast.ai).
Runs as a cron job or standalone to prevent disk full scenarios.

Usage:
    # Check disk and clean if needed
    python scripts/disk_monitor.py --threshold 80

    # Force cleanup even if below threshold
    python scripts/disk_monitor.py --force

    # Dry run - show what would be cleaned
    python scripts/disk_monitor.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    path: str
    size_bytes: int
    deleted: bool
    reason: str


def get_disk_usage(path: str = "/") -> Tuple[int, int, float]:
    """Get disk usage stats. Returns (used_bytes, total_bytes, percent_used)."""
    stat = os.statvfs(path)
    total = stat.f_blocks * stat.f_frsize
    free = stat.f_bavail * stat.f_frsize
    used = total - free
    percent = (used / total) * 100 if total > 0 else 0
    return used, total, percent


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"


def find_old_files(directory: str, max_age_days: int, patterns: List[str]) -> List[Path]:
    """Find files older than max_age_days matching patterns."""
    cutoff = time.time() - (max_age_days * 86400)
    old_files = []

    for pattern in patterns:
        full_pattern = os.path.join(directory, pattern)
        for filepath in glob.glob(full_pattern, recursive=True):
            try:
                path = Path(filepath)
                if path.is_file() and path.stat().st_mtime < cutoff:
                    old_files.append(path)
            except (OSError, PermissionError):
                continue

    return old_files


def find_large_files(directory: str, min_size_mb: int = 100) -> List[Tuple[Path, int]]:
    """Find files larger than min_size_mb."""
    min_size = min_size_mb * 1024 * 1024
    large_files = []

    try:
        for root, dirs, files in os.walk(directory):
            # Skip .git directories
            dirs[:] = [d for d in dirs if d != '.git']

            for f in files:
                filepath = Path(root) / f
                try:
                    size = filepath.stat().st_size
                    if size > min_size:
                        large_files.append((filepath, size))
                except (OSError, PermissionError):
                    continue
    except (OSError, PermissionError):
        pass

    return sorted(large_files, key=lambda x: x[1], reverse=True)


def cleanup_temp_files(dry_run: bool = False) -> List[CleanupResult]:
    """Clean up temporary files."""
    results = []

    temp_patterns = [
        "/tmp/*.db",
        "/tmp/*.log",
        "/tmp/*.jsonl",
        "/tmp/selfplay_*",
        "/tmp/pytest-*",
        "/tmp/claude/*",
    ]

    for pattern in temp_patterns:
        for filepath in glob.glob(pattern):
            try:
                path = Path(filepath)
                if path.is_file():
                    size = path.stat().st_size
                    if not dry_run:
                        path.unlink()
                    results.append(CleanupResult(
                        path=str(path),
                        size_bytes=size,
                        deleted=not dry_run,
                        reason="temp_file"
                    ))
                elif path.is_dir() and time.time() - path.stat().st_mtime > 86400:
                    size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    if not dry_run:
                        shutil.rmtree(path)
                    results.append(CleanupResult(
                        path=str(path),
                        size_bytes=size,
                        deleted=not dry_run,
                        reason="temp_dir"
                    ))
            except (OSError, PermissionError):
                continue

    return results


def cleanup_old_logs(ringrift_path: str, max_age_days: int = 3, dry_run: bool = False) -> List[CleanupResult]:
    """Clean up old log files."""
    results = []
    log_dirs = [
        os.path.join(ringrift_path, "ai-service/logs"),
        "/var/log/ringrift",
    ]

    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            continue

        old_logs = find_old_files(log_dir, max_age_days, ["**/*.log", "**/*.jsonl"])
        for path in old_logs:
            try:
                size = path.stat().st_size
                if not dry_run:
                    path.unlink()
                results.append(CleanupResult(
                    path=str(path),
                    size_bytes=size,
                    deleted=not dry_run,
                    reason="old_log"
                ))
            except (OSError, PermissionError):
                continue

    return results


def cleanup_selfplay_data(ringrift_path: str, max_age_days: int = 7,
                          keep_min_gb: float = 1.0, dry_run: bool = False) -> List[CleanupResult]:
    """Clean up old selfplay data, keeping minimum amount for training."""
    results = []
    selfplay_dir = os.path.join(ringrift_path, "ai-service/data/selfplay")

    if not os.path.exists(selfplay_dir):
        return results

    # Find all selfplay files sorted by age (oldest first)
    all_files = []
    for pattern in ["**/*.jsonl", "**/*.db"]:
        for filepath in glob.glob(os.path.join(selfplay_dir, pattern), recursive=True):
            try:
                path = Path(filepath)
                mtime = path.stat().st_mtime
                size = path.stat().st_size
                all_files.append((path, mtime, size))
            except (OSError, PermissionError):
                continue

    all_files.sort(key=lambda x: x[1])  # Sort by mtime, oldest first

    # Calculate total size and keep threshold
    total_size = sum(f[2] for f in all_files)
    keep_bytes = int(keep_min_gb * 1024 * 1024 * 1024)

    # Delete old files until we're under the age limit or at minimum data
    cutoff = time.time() - (max_age_days * 86400)
    deleted_size = 0

    for path, mtime, size in all_files:
        if mtime >= cutoff:
            break  # Stop at files newer than cutoff

        # Keep minimum amount of data
        if total_size - deleted_size <= keep_bytes:
            break

        try:
            if not dry_run:
                path.unlink()
            deleted_size += size
            results.append(CleanupResult(
                path=str(path),
                size_bytes=size,
                deleted=not dry_run,
                reason="old_selfplay"
            ))
        except (OSError, PermissionError):
            continue

    return results


def cleanup_large_noncanonical_game_dbs(
    ringrift_path: str,
    *,
    dry_run: bool = False,
    min_size_mb: int = 256,
) -> List[CleanupResult]:
    """Clean up large non-canonical SQLite DBs under ai-service/data/games.

    This is specifically to protect small-disk nodes (e.g. Vast.ai 16GB overlay)
    from being bricked by multi-GB scratch DBs like `selfplay.db`.

    Safety rules:
    - Never delete `canonical_*.db`
    - Only deletes DBs matching selfplay-ish patterns
    - Only deletes when size >= min_size_mb
    """
    results: List[CleanupResult] = []
    games_dir = Path(ringrift_path) / "ai-service" / "data" / "games"
    if not games_dir.exists():
        return results

    patterns = [
        "selfplay*.db",
        "self_play*.db",
        "*selfplay*.db",
    ]
    min_size = int(min_size_mb) * 1024 * 1024

    seen: set[Path] = set()
    for pattern in patterns:
        for filepath in games_dir.glob(pattern):
            path = filepath
            if path in seen:
                continue
            seen.add(path)

            name = path.name
            if name.startswith("canonical_"):
                continue
            try:
                size = path.stat().st_size
            except (OSError, PermissionError):
                continue
            if size < min_size:
                continue

            # Remove the DB plus common SQLite sidecars if present.
            sidecars = [path, path.with_name(f"{name}-wal"), path.with_name(f"{name}-shm")]
            freed = 0
            deleted_any = False
            for candidate in sidecars:
                try:
                    if not candidate.exists():
                        continue
                    candidate_size = candidate.stat().st_size
                    if not dry_run:
                        candidate.unlink()
                    freed += candidate_size
                    deleted_any = True
                except (OSError, PermissionError):
                    continue

            if deleted_any:
                results.append(
                    CleanupResult(
                        path=str(path),
                        size_bytes=freed,
                        deleted=not dry_run,
                        reason="large_noncanonical_game_db",
                    )
                )

    return results


def cleanup_venv_cache(ringrift_path: str, dry_run: bool = False) -> List[CleanupResult]:
    """Clean up Python cache in venv."""
    results = []
    cache_dirs = [
        os.path.join(ringrift_path, "ai-service/venv/lib/python*/site-packages/*/__pycache__"),
        os.path.join(ringrift_path, "**/__pycache__"),
    ]

    for pattern in cache_dirs:
        for dirpath in glob.glob(pattern, recursive=True):
            try:
                path = Path(dirpath)
                if path.is_dir():
                    size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    if not dry_run:
                        shutil.rmtree(path)
                    results.append(CleanupResult(
                        path=str(path),
                        size_bytes=size,
                        deleted=not dry_run,
                        reason="pycache"
                    ))
            except (OSError, PermissionError):
                continue

    return results


def cleanup_deprecated_data(ringrift_path: str, dry_run: bool = False) -> List[CleanupResult]:
    """Clean up deprecated data directories."""
    results = []
    deprecated_dirs = [
        os.path.join(ringrift_path, "ai-service/data/deprecated*"),
        os.path.join(ringrift_path, "ai-service/data/selfplay_old*"),
        os.path.join(ringrift_path, "data"),  # Old location
    ]

    for pattern in deprecated_dirs:
        for dirpath in glob.glob(pattern):
            try:
                path = Path(dirpath)
                if path.is_dir():
                    size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    if not dry_run:
                        shutil.rmtree(path)
                    results.append(CleanupResult(
                        path=str(path),
                        size_bytes=size,
                        deleted=not dry_run,
                        reason="deprecated"
                    ))
            except (OSError, PermissionError):
                continue

    return results


def run_cleanup(ringrift_path: str, threshold: int = 80, force: bool = False,
                dry_run: bool = False, aggressive: bool = False) -> dict:
    """Run full disk cleanup if needed."""
    # Always measure disk usage on the volume that actually contains the RingRift
    # checkout/data. On macOS (APFS split volumes) and some container overlays,
    # checking "/" can under-report the real pressure where RingRift lives.
    used, total, percent = get_disk_usage(ringrift_path)
    free_gb = (total - used) / (1024 ** 3) if total > 0 else 0.0

    print(f"Disk usage: {format_size(used)} / {format_size(total)} ({percent:.1f}%)")

    if percent < threshold and not force:
        print(f"Disk usage {percent:.1f}% is below threshold {threshold}%, skipping cleanup")
        return {"cleaned": False, "reason": "below_threshold", "percent_used": percent}

    print(f"Running cleanup (threshold: {threshold}%, force: {force}, dry_run: {dry_run})...")

    all_results = []

    # Run cleanup steps in order of aggressiveness
    all_results.extend(cleanup_temp_files(dry_run))
    all_results.extend(cleanup_old_logs(ringrift_path, max_age_days=3, dry_run=dry_run))
    all_results.extend(cleanup_deprecated_data(ringrift_path, dry_run=dry_run))
    all_results.extend(cleanup_venv_cache(ringrift_path, dry_run=dry_run))

    if aggressive or percent > 90:
        print("Running aggressive cleanup (selfplay data)...")
        all_results.extend(cleanup_selfplay_data(
            ringrift_path,
            max_age_days=3 if percent > 95 else 7,
            keep_min_gb=0.5 if percent > 95 else 1.0,
            dry_run=dry_run
        ))
        # Last-resort protection for tiny disks: delete multi-GB non-canonical DBs
        # that can brick nodes (e.g. ai-service/data/games/selfplay.db).
        if force or percent > 95 or free_gb < 2.0:
            all_results.extend(
                cleanup_large_noncanonical_game_dbs(
                    ringrift_path,
                    dry_run=dry_run,
                    min_size_mb=256 if free_gb < 2.0 else 512,
                )
            )

    # Summarize results
    total_freed = sum(r.size_bytes for r in all_results if r.deleted)

    print(f"\nCleanup complete:")
    print(f"  Files processed: {len(all_results)}")
    print(f"  Space freed: {format_size(total_freed)}")

    # Show new disk usage (same volume as RingRift checkout)
    new_used, new_total, new_percent = get_disk_usage(ringrift_path)
    print(f"  New disk usage: {format_size(new_used)} / {format_size(new_total)} ({new_percent:.1f}%)")

    return {
        "cleaned": True,
        "files_cleaned": len([r for r in all_results if r.deleted]),
        "bytes_freed": total_freed,
        "old_percent": percent,
        "new_percent": new_percent,
        "dry_run": dry_run,
    }


def main():
    parser = argparse.ArgumentParser(description="Disk monitoring and cleanup")
    parser.add_argument("--threshold", type=int, default=80,
                        help="Disk usage percent threshold to trigger cleanup")
    parser.add_argument("--force", action="store_true",
                        help="Force cleanup even if below threshold")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be cleaned without deleting")
    parser.add_argument("--aggressive", action="store_true",
                        help="Use aggressive cleanup (removes more selfplay data)")
    parser.add_argument("--ringrift-path",
                        default=os.environ.get("RINGRIFT_DIR",
                                              os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        help="Path to RingRift directory")

    args = parser.parse_args()

    result = run_cleanup(
        ringrift_path=args.ringrift_path,
        threshold=args.threshold,
        force=args.force,
        dry_run=args.dry_run,
        aggressive=args.aggressive,
    )

    # Exit with error if still over threshold after cleanup
    if result.get("new_percent", 0) > args.threshold and not args.dry_run:
        print(f"\nWARNING: Still above threshold after cleanup!")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
