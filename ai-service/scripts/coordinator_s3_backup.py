#!/usr/bin/env python3
"""Coordinator S3 Backup Script.

This script runs on the coordinator (mac-studio) to backup all cluster data to S3.
It syncs data that has been collected from cluster nodes via rsync.

Architecture:
    Cluster Nodes ---(rsync)---> Coordinator (OWC Drive)
    Coordinator ----(aws s3)---> S3 Bucket

Usage:
    # Run once
    python scripts/coordinator_s3_backup.py

    # Run in daemon mode (hourly)
    python scripts/coordinator_s3_backup.py --daemon

    # Dry run
    python scripts/coordinator_s3_backup.py --dry-run

December 2025: Created for centralized S3 backup.
"""

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
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass
class S3BackupConfig:
    """Configuration for S3 backup."""

    s3_bucket: str = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
    )
    aws_region: str = field(
        default_factory=lambda: os.getenv("AWS_REGION", "us-east-1")
    )

    # Paths
    owc_base: Path = Path("/Volumes/RingRift-Data")
    local_models: Path = ROOT / "models"
    local_games: Path = ROOT / "data" / "games"
    local_training: Path = ROOT / "data" / "training"

    # Sync interval for daemon mode
    sync_interval_seconds: float = 3600.0  # 1 hour


@dataclass
class BackupResult:
    """Result of a backup operation."""

    success: bool
    models_uploaded: int = 0
    databases_uploaded: int = 0
    npz_uploaded: int = 0
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class CoordinatorS3Backup:
    """Coordinator-based S3 backup for the entire cluster."""

    def __init__(self, config: S3BackupConfig | None = None):
        self.config = config or S3BackupConfig()
        self._last_backup: float = 0.0

    async def run_backup(self, dry_run: bool = False) -> BackupResult:
        """Run a full backup to S3."""
        start_time = time.time()
        result = BackupResult(success=True)

        logger.info(f"Starting S3 backup to s3://{self.config.s3_bucket}/")

        # 1. Backup models (highest priority)
        models_result = await self._backup_models(dry_run)
        result.models_uploaded = models_result["uploaded"]
        result.bytes_transferred += models_result["bytes"]
        result.errors.extend(models_result["errors"])

        # 2. Backup canonical databases
        dbs_result = await self._backup_databases(dry_run)
        result.databases_uploaded = dbs_result["uploaded"]
        result.bytes_transferred += dbs_result["bytes"]
        result.errors.extend(dbs_result["errors"])

        # 3. Backup training NPZ files
        npz_result = await self._backup_npz(dry_run)
        result.npz_uploaded = npz_result["uploaded"]
        result.bytes_transferred += npz_result["bytes"]
        result.errors.extend(npz_result["errors"])

        # 4. Update manifest
        if not dry_run:
            await self._update_manifest(result)

        result.duration_seconds = time.time() - start_time
        result.success = len(result.errors) == 0

        self._last_backup = time.time()

        logger.info(
            f"Backup complete: {result.models_uploaded} models, "
            f"{result.databases_uploaded} DBs, {result.npz_uploaded} NPZ files "
            f"({result.bytes_transferred / 1024 / 1024:.1f}MB) in {result.duration_seconds:.1f}s"
        )

        if result.errors:
            logger.warning(f"Backup had {len(result.errors)} errors")

        return result

    async def _backup_models(self, dry_run: bool) -> dict[str, Any]:
        """Backup model checkpoints."""
        result = {"uploaded": 0, "bytes": 0, "errors": []}

        # Find all canonical models
        model_dirs = [
            self.config.local_models,
        ]

        # Add OWC models if available
        owc_models = self.config.owc_base / "canonical_models"
        if owc_models.exists():
            model_dirs.append(owc_models)

        for model_dir in model_dirs:
            if not model_dir.exists():
                continue

            for model_file in model_dir.glob("canonical_*.pth"):
                if model_file.is_symlink():
                    continue

                s3_path = f"consolidated/models/{model_file.name}"

                try:
                    uploaded = await self._s3_sync_file(model_file, s3_path, dry_run)
                    if uploaded:
                        result["uploaded"] += 1
                        result["bytes"] += model_file.stat().st_size
                except Exception as e:
                    result["errors"].append(f"Model {model_file.name}: {e}")

        return result

    async def _backup_databases(self, dry_run: bool) -> dict[str, Any]:
        """Backup canonical game databases."""
        result = {"uploaded": 0, "bytes": 0, "errors": []}

        db_dirs = [
            self.config.local_games,
        ]

        # Add OWC databases if available
        owc_games = self.config.owc_base / "selfplay_repository" / "synced"
        if owc_games.exists():
            db_dirs.append(owc_games)

        for db_dir in db_dirs:
            if not db_dir.exists():
                continue

            for db_file in db_dir.glob("canonical_*.db"):
                if db_file.stat().st_size < 10000:  # Skip tiny DBs
                    continue

                s3_path = f"consolidated/databases/{db_file.name}"

                try:
                    uploaded = await self._s3_sync_file(db_file, s3_path, dry_run)
                    if uploaded:
                        result["uploaded"] += 1
                        result["bytes"] += db_file.stat().st_size
                except Exception as e:
                    result["errors"].append(f"Database {db_file.name}: {e}")

        return result

    async def _backup_npz(self, dry_run: bool) -> dict[str, Any]:
        """Backup training NPZ files."""
        result = {"uploaded": 0, "bytes": 0, "errors": []}

        npz_dirs = [
            self.config.local_training,
        ]

        # Add OWC training data if available
        owc_training = self.config.owc_base / "canonical_data"
        if owc_training.exists():
            npz_dirs.append(owc_training)

        for npz_dir in npz_dirs:
            if not npz_dir.exists():
                continue

            for npz_file in npz_dir.glob("*.npz"):
                s3_path = f"consolidated/training/{npz_file.name}"

                try:
                    uploaded = await self._s3_sync_file(npz_file, s3_path, dry_run)
                    if uploaded:
                        result["uploaded"] += 1
                        result["bytes"] += npz_file.stat().st_size
                except Exception as e:
                    result["errors"].append(f"NPZ {npz_file.name}: {e}")

        return result

    async def _s3_sync_file(
        self, local_path: Path, s3_path: str, dry_run: bool
    ) -> bool:
        """Sync a file to S3 if needed."""
        s3_uri = f"s3://{self.config.s3_bucket}/{s3_path}"

        # Check if file needs upload (size-based)
        local_size = local_path.stat().st_size

        try:
            # Get S3 object size
            cmd = [
                "aws", "s3api", "head-object",
                "--bucket", self.config.s3_bucket,
                "--key", s3_path,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()

            if process.returncode == 0:
                response = json.loads(stdout.decode())
                s3_size = response.get("ContentLength", 0)

                if local_size == s3_size:
                    # Same size, skip upload
                    return False

        except Exception:
            pass  # File doesn't exist in S3 or error, will upload

        if dry_run:
            logger.info(f"[DRY RUN] Would upload: {local_path.name}")
            return True

        # Upload file
        cmd = ["aws", "s3", "cp", str(local_path), s3_uri, "--only-show-errors"]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=600.0,  # 10 minute timeout for large files
        )

        if process.returncode != 0:
            raise RuntimeError(f"S3 upload failed: {stderr.decode()}")

        logger.info(f"Uploaded: {local_path.name} ({local_size / 1024 / 1024:.1f}MB)")
        return True

    async def _update_manifest(self, result: BackupResult) -> None:
        """Update the consolidated manifest in S3."""
        manifest = {
            "timestamp": time.time(),
            "timestamp_iso": datetime.now().isoformat(),
            "source": "coordinator",
            "backup_result": {
                "models_uploaded": result.models_uploaded,
                "databases_uploaded": result.databases_uploaded,
                "npz_uploaded": result.npz_uploaded,
                "bytes_transferred": result.bytes_transferred,
                "duration_seconds": result.duration_seconds,
                "success": result.success,
            },
        }

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(manifest, f, indent=2)
            temp_path = f.name

        try:
            cmd = [
                "aws", "s3", "cp", temp_path,
                f"s3://{self.config.s3_bucket}/consolidated/manifest.json",
                "--only-show-errors",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

        finally:
            os.unlink(temp_path)


async def run_daemon(backup: CoordinatorS3Backup, config: S3BackupConfig) -> None:
    """Run backup in daemon mode."""
    logger.info(f"Starting S3 backup daemon (interval: {config.sync_interval_seconds}s)")

    while True:
        try:
            await backup.run_backup()
        except Exception as e:
            logger.error(f"Backup failed: {e}")

        await asyncio.sleep(config.sync_interval_seconds)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Coordinator S3 Backup")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode (continuous)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3600.0,
        help="Sync interval in seconds (daemon mode)",
    )

    args = parser.parse_args()

    config = S3BackupConfig()
    config.sync_interval_seconds = args.interval

    backup = CoordinatorS3Backup(config)

    if args.daemon:
        await run_daemon(backup, config)
    else:
        result = await backup.run_backup(dry_run=args.dry_run)
        sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    asyncio.run(main())
