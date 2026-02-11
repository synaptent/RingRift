#!/usr/bin/env python3
"""OWC External Drive to S3 Mirror Script.

Comprehensive backup of the OWC external drive to AWS S3 with deduplication
to eliminate single point of failure (SPOF).

Architecture:
    OWC Drive (mac-studio) ----(aws s3 sync)---> S3 Bucket

Uses `aws s3 sync` which provides:
- Automatic deduplication via ETag/MD5 comparison
- Incremental uploads (only changed files)
- Efficient multi-part uploads for large files

Data Priority Tiers:
    Tier 1 (Critical): canonical_models, canonical_games, canonical_data
    Tier 2 (Important): consolidated, consolidated_training, model_checkpoints
    Tier 3 (Archival): cluster_games, cluster_aggregated, archived

Usage:
    # Full sync (all tiers)
    python scripts/owc_s3_mirror.py

    # Critical data only (fastest)
    python scripts/owc_s3_mirror.py --tier 1

    # Dry run
    python scripts/owc_s3_mirror.py --dry-run

    # Daemon mode (runs every 6 hours)
    python scripts/owc_s3_mirror.py --daemon

    # Estimate costs before syncing
    python scripts/owc_s3_mirror.py --estimate

December 2025: Created for comprehensive OWC backup.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Ensure AWS CLI is in PATH (pip-installed version on mac-studio)
_aws_paths = [
    os.path.expanduser("~/Library/Python/3.9/bin"),
    os.path.expanduser("~/Library/Python/3.10/bin"),
    os.path.expanduser("~/Library/Python/3.11/bin"),
    "/usr/local/bin",
    "/opt/homebrew/bin",
]
for _path in _aws_paths:
    if _path not in os.environ.get("PATH", "") and os.path.exists(_path):
        os.environ["PATH"] = f"{_path}:{os.environ.get('PATH', '')}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SyncDirectory:
    """A directory to sync to S3."""

    local_path: str  # Relative to OWC base
    s3_prefix: str   # S3 prefix within bucket
    tier: int        # Priority tier (1=critical, 2=important, 3=archival)
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class OWCMirrorConfig:
    """Configuration for OWC to S3 mirroring."""

    s3_bucket: str = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
    )
    aws_region: str = field(
        default_factory=lambda: os.getenv("AWS_REGION", "us-east-1")
    )
    owc_base: Path = Path("/Volumes/RingRift-Data")

    # Sync interval for daemon mode (6 hours)
    sync_interval_seconds: float = 21600.0

    # Parallel uploads
    max_concurrent_requests: int = 20

    # Storage class for archival data
    archival_storage_class: str = "STANDARD_IA"  # Infrequent Access for cost savings

    # Directories to sync, ordered by priority
    directories: list[SyncDirectory] = field(default_factory=lambda: [
        # Tier 1: Critical (smallest, most important)
        SyncDirectory(
            local_path="canonical_models",
            s3_prefix="owc/canonical_models",
            tier=1,
            include_patterns=["*.pth"],
            description="Canonical model checkpoints (~17GB)",
        ),
        SyncDirectory(
            local_path="canonical_data",
            s3_prefix="owc/canonical_data",
            tier=1,
            include_patterns=["*.npz"],
            description="Canonical training data (~18GB)",
        ),
        SyncDirectory(
            local_path="canonical_games",
            s3_prefix="owc/canonical_games",
            tier=1,
            include_patterns=["*.db"],
            exclude_patterns=["*-shm", "*-wal", "*-journal"],
            description="Canonical game databases (~53GB)",
        ),

        # Tier 2: Important (medium size, frequently used)
        SyncDirectory(
            local_path="consolidated",
            s3_prefix="owc/consolidated",
            tier=2,
            description="Consolidated data (~10GB)",
        ),
        SyncDirectory(
            local_path="consolidated_training",
            s3_prefix="owc/consolidated_training",
            tier=2,
            include_patterns=["*.npz"],
            description="Consolidated training NPZ (~161GB)",
        ),
        SyncDirectory(
            local_path="model_checkpoints",
            s3_prefix="owc/model_checkpoints",
            tier=2,
            include_patterns=["*.pth", "*.pt"],
            description="All model checkpoints (~330GB)",
        ),
        SyncDirectory(
            local_path="trained_models",
            s3_prefix="owc/trained_models",
            tier=2,
            include_patterns=["*.pth", "*.pt"],
            description="Production trained models (~299GB)",
        ),
        SyncDirectory(
            local_path="games",
            s3_prefix="owc/games",
            tier=2,
            include_patterns=["*.db"],
            exclude_patterns=["*-shm", "*-wal", "*-journal"],
            description="Local game databases (~324GB)",
        ),

        # Tier 3: Archival (large, infrequently accessed)
        SyncDirectory(
            local_path="cluster_games",
            s3_prefix="owc/cluster_games",
            tier=3,
            include_patterns=["*.db"],
            exclude_patterns=["*-shm", "*-wal", "*-journal"],
            description="Cluster game databases (~587GB)",
        ),
        SyncDirectory(
            local_path="cluster_aggregated",
            s3_prefix="owc/cluster_aggregated",
            tier=3,
            include_patterns=["*.db", "*.npz"],
            exclude_patterns=["*-shm", "*-wal", "*-journal"],
            description="Aggregated cluster data (~270GB)",
        ),
        SyncDirectory(
            local_path="archived",
            s3_prefix="owc/archived",
            tier=3,
            description="Archived data (~300GB)",
        ),
        SyncDirectory(
            local_path="cluster_collected_backup",
            s3_prefix="owc/cluster_collected_backup",
            tier=3,
            include_patterns=["*.db", "*.npz", "*.pth"],
            exclude_patterns=["*-shm", "*-wal", "*-journal", "*.log", "*.tmp"],
            description="Full cluster backup (~2.9TB)",
        ),
        SyncDirectory(
            local_path="selfplay_repository",
            s3_prefix="owc/selfplay_repository",
            tier=3,
            include_patterns=["*.db", "*.npz", "*.pth"],
            exclude_patterns=["*-shm", "*-wal", "*-journal", "*.log", "*.tmp"],
            description="Raw selfplay repository (~3.3TB)",
        ),
    ])


@dataclass
class SyncResult:
    """Result of a sync operation."""

    directory: str
    success: bool
    files_uploaded: int = 0
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    error: str = ""


@dataclass
class MirrorResult:
    """Complete result of mirroring operation."""

    success: bool
    total_files: int = 0
    total_bytes: int = 0
    duration_seconds: float = 0.0
    sync_results: list[SyncResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class OWCMirror:
    """OWC External Drive to S3 Mirror."""

    def __init__(self, config: OWCMirrorConfig | None = None):
        self.config = config or OWCMirrorConfig()
        self._last_sync: float = 0.0

    def estimate_costs(self) -> dict[str, Any]:
        """Estimate S3 storage costs for OWC data."""
        estimates = {
            "directories": [],
            "total_size_gb": 0,
            "monthly_storage_cost_usd": 0,
            "monthly_ia_storage_cost_usd": 0,
        }

        # S3 pricing (us-east-1)
        standard_per_gb = 0.023  # $/GB/month
        ia_per_gb = 0.0125  # $/GB/month for Infrequent Access

        for sync_dir in self.config.directories:
            local_path = self.config.owc_base / sync_dir.local_path
            if not local_path.exists():
                continue

            # Get directory size (cross-platform: macOS uses -sk, Linux uses -sb)
            try:
                # Try Linux-style first (bytes), fallback to macOS (kilobytes)
                result = subprocess.run(
                    ["du", "-sk", str(local_path)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    size_kb = int(result.stdout.split()[0])
                    size_gb = size_kb / (1024**2)

                    if sync_dir.tier <= 2:
                        monthly_cost = size_gb * standard_per_gb
                        storage_class = "STANDARD"
                    else:
                        monthly_cost = size_gb * ia_per_gb
                        storage_class = "STANDARD_IA"

                    estimates["directories"].append({
                        "path": sync_dir.local_path,
                        "tier": sync_dir.tier,
                        "size_gb": round(size_gb, 2),
                        "storage_class": storage_class,
                        "monthly_cost_usd": round(monthly_cost, 2),
                    })
                    estimates["total_size_gb"] += size_gb
                    if sync_dir.tier <= 2:
                        estimates["monthly_storage_cost_usd"] += monthly_cost
                    else:
                        estimates["monthly_ia_storage_cost_usd"] += monthly_cost

            except Exception as e:
                logger.warning(f"Could not estimate {sync_dir.local_path}: {e}")

        estimates["total_monthly_cost_usd"] = round(
            estimates["monthly_storage_cost_usd"] + estimates["monthly_ia_storage_cost_usd"],
            2
        )
        estimates["total_size_gb"] = round(estimates["total_size_gb"], 2)

        return estimates

    async def run_mirror(
        self,
        tier_filter: int | None = None,
        dry_run: bool = False,
        delete: bool = False,
    ) -> MirrorResult:
        """Run the mirror operation.

        Args:
            tier_filter: Only sync directories of this tier or lower
            dry_run: Show what would be synced without syncing
            delete: Delete files in S3 that don't exist locally
        """
        start_time = time.time()
        result = MirrorResult(success=True)

        # Filter directories by tier
        directories = self.config.directories
        if tier_filter is not None:
            directories = [d for d in directories if d.tier <= tier_filter]

        logger.info(
            f"Starting OWC->S3 mirror: {len(directories)} directories, "
            f"tier filter={tier_filter or 'all'}, dry_run={dry_run}"
        )

        for sync_dir in directories:
            sync_result = await self._sync_directory(sync_dir, dry_run, delete)
            result.sync_results.append(sync_result)

            if sync_result.success:
                result.total_files += sync_result.files_uploaded
                result.total_bytes += sync_result.bytes_transferred
            else:
                result.errors.append(sync_result.error)

        result.duration_seconds = time.time() - start_time
        result.success = len(result.errors) == 0

        self._last_sync = time.time()

        # Log summary
        logger.info(
            f"Mirror complete: {result.total_files} files, "
            f"{result.total_bytes / (1024**3):.2f}GB transferred, "
            f"{result.duration_seconds:.1f}s, "
            f"{'SUCCESS' if result.success else f'{len(result.errors)} errors'}"
        )

        return result

    async def _sync_directory(
        self,
        sync_dir: SyncDirectory,
        dry_run: bool,
        delete: bool,
    ) -> SyncResult:
        """Sync a single directory to S3."""
        start_time = time.time()
        local_path = self.config.owc_base / sync_dir.local_path

        if not local_path.exists():
            logger.warning(f"Directory not found: {local_path}")
            return SyncResult(
                directory=sync_dir.local_path,
                success=False,
                error=f"Directory not found: {local_path}",
            )

        s3_uri = f"s3://{self.config.s3_bucket}/{sync_dir.s3_prefix}/"

        # Build aws s3 sync command
        cmd = [
            "aws", "s3", "sync",
            str(local_path),
            s3_uri,
            "--no-progress",
            "--no-follow-symlinks",  # Avoid broken symlink issues
        ]

        # Add include patterns
        for pattern in sync_dir.include_patterns:
            cmd.extend(["--include", pattern])

        # Add exclude patterns
        for pattern in sync_dir.exclude_patterns:
            cmd.extend(["--exclude", pattern])

        # Use Infrequent Access for tier 3 (archival)
        if sync_dir.tier >= 3:
            cmd.extend(["--storage-class", self.config.archival_storage_class])

        # Delete files in S3 that don't exist locally
        # WARNING: This can cause data loss if local files were cleaned up
        if delete:
            logger.warning(
                f"--delete enabled for {sync_dir.local_path}. "
                "S3 files not present locally will be REMOVED from S3."
            )
            cmd.append("--delete")

        # Dry run
        if dry_run:
            cmd.append("--dryrun")

        logger.info(f"Syncing {sync_dir.description}: {local_path} -> {s3_uri}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=7200.0,  # 2 hour timeout for large directories
            )

            # AWS CLI returns exit code 2 for warnings (like skipped files)
            # This is OK if files were actually synced
            output = stdout.decode()
            file_count = output.count("upload:") + output.count("copy:")

            if process.returncode != 0:
                error_msg = stderr.decode().strip()
                # Exit code 2 with "warning:" is acceptable if we synced files
                # or if it's just about skipped symlinks/unreadable files
                if process.returncode == 2 and "warning:" in error_msg:
                    if file_count > 0 or "Skipping" in error_msg:
                        logger.warning(f"Sync completed with warnings for {sync_dir.local_path}: {error_msg}")
                    else:
                        logger.warning(f"Sync had warnings but no files for {sync_dir.local_path}: {error_msg}")
                else:
                    logger.error(f"Sync failed for {sync_dir.local_path}: {error_msg}")
                    return SyncResult(
                        directory=sync_dir.local_path,
                        success=False,
                        duration_seconds=time.time() - start_time,
                        error=error_msg,
                    )

            # file_count already computed above

            # Estimate bytes (rough estimate from directory size change tracking)
            bytes_transferred = 0

            logger.info(f"Synced {sync_dir.local_path}: {file_count} files")

            return SyncResult(
                directory=sync_dir.local_path,
                success=True,
                files_uploaded=file_count,
                bytes_transferred=bytes_transferred,
                duration_seconds=time.time() - start_time,
            )

        except asyncio.TimeoutError:
            return SyncResult(
                directory=sync_dir.local_path,
                success=False,
                duration_seconds=time.time() - start_time,
                error="Sync timed out after 2 hours",
            )
        except Exception as e:
            return SyncResult(
                directory=sync_dir.local_path,
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    async def update_manifest(self, result: MirrorResult) -> None:
        """Update the mirror manifest in S3."""
        manifest = {
            "timestamp": time.time(),
            "timestamp_iso": datetime.now().isoformat(),
            "source": "owc_mirror",
            "owc_path": str(self.config.owc_base),
            "result": {
                "success": result.success,
                "total_files": result.total_files,
                "total_bytes": result.total_bytes,
                "duration_seconds": result.duration_seconds,
                "directories_synced": [
                    {
                        "path": r.directory,
                        "success": r.success,
                        "files": r.files_uploaded,
                        "duration": r.duration_seconds,
                        "error": r.error if not r.success else None,
                    }
                    for r in result.sync_results
                ],
            },
        }

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(manifest, f, indent=2)
            temp_path = f.name

        try:
            cmd = [
                "aws", "s3", "cp", temp_path,
                f"s3://{self.config.s3_bucket}/owc/manifest.json",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

        finally:
            os.unlink(temp_path)

    def verify_prerequisites(self) -> list[str]:
        """Verify prerequisites for S3 mirroring."""
        errors = []

        # Check OWC drive mounted
        if not self.config.owc_base.exists():
            errors.append(f"OWC drive not mounted at {self.config.owc_base}")

        # Check AWS CLI
        try:
            result = subprocess.run(
                ["aws", "--version"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                errors.append("AWS CLI not found or not working")
        except Exception as e:
            errors.append(f"AWS CLI check failed: {e}")

        # Check AWS credentials
        try:
            result = subprocess.run(
                ["aws", "sts", "get-caller-identity"],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                errors.append("AWS credentials not configured")
        except Exception as e:
            errors.append(f"AWS credentials check failed: {e}")

        # Check S3 bucket access
        try:
            result = subprocess.run(
                ["aws", "s3", "ls", f"s3://{self.config.s3_bucket}/"],
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0:
                errors.append(f"Cannot access S3 bucket: {self.config.s3_bucket}")
        except Exception as e:
            errors.append(f"S3 bucket check failed: {e}")

        return errors


async def run_daemon(mirror: OWCMirror, config: OWCMirrorConfig) -> None:
    """Run mirror in daemon mode."""
    logger.info(f"Starting OWC mirror daemon (interval: {config.sync_interval_seconds}s)")

    while True:
        try:
            result = await mirror.run_mirror()
            await mirror.update_manifest(result)
        except Exception as e:
            logger.error(f"Mirror failed: {e}")

        await asyncio.sleep(config.sync_interval_seconds)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="OWC External Drive to S3 Mirror",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync all tiers
  python scripts/owc_s3_mirror.py

  # Sync critical data only (tier 1)
  python scripts/owc_s3_mirror.py --tier 1

  # Dry run to see what would be synced
  python scripts/owc_s3_mirror.py --dry-run

  # Estimate storage costs
  python scripts/owc_s3_mirror.py --estimate

  # Run as daemon (syncs every 6 hours)
  python scripts/owc_s3_mirror.py --daemon
        """
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        help="Only sync directories of this tier or lower (1=critical, 2=important, 3=archival)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode (continuous sync)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without syncing",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete files in S3 that don't exist locally. DANGEROUS: if local files "
             "were cleaned up, this will delete the S3 backups too. Use with --dry-run first.",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate S3 storage costs for OWC data",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=21600.0,
        help="Sync interval in seconds for daemon mode (default: 6 hours)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        help="Override S3 bucket name",
    )

    args = parser.parse_args()

    config = OWCMirrorConfig()
    config.sync_interval_seconds = args.interval
    if args.bucket:
        config.s3_bucket = args.bucket

    mirror = OWCMirror(config)

    # Verify prerequisites
    errors = mirror.verify_prerequisites()
    if errors:
        for error in errors:
            logger.error(f"Prerequisite check failed: {error}")
        sys.exit(1)

    if args.estimate:
        estimates = mirror.estimate_costs()
        print("\n=== S3 Storage Cost Estimate ===\n")
        print(f"{'Directory':<40} {'Tier':<6} {'Size (GB)':<12} {'Class':<15} {'$/month':<10}")
        print("-" * 90)
        for d in estimates["directories"]:
            print(f"{d['path']:<40} {d['tier']:<6} {d['size_gb']:<12.2f} {d['storage_class']:<15} ${d['monthly_cost_usd']:<10.2f}")
        print("-" * 90)
        print(f"{'TOTAL':<40} {'':<6} {estimates['total_size_gb']:<12.2f} {'':<15} ${estimates['total_monthly_cost_usd']:<10.2f}")
        print()
        return

    if args.daemon:
        await run_daemon(mirror, config)
    else:
        result = await mirror.run_mirror(
            tier_filter=args.tier,
            dry_run=args.dry_run,
            delete=args.delete,
        )

        if not args.dry_run:
            await mirror.update_manifest(result)

        sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    asyncio.run(main())
