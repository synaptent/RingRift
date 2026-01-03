"""S3 Import Daemon - Downloads training data from AWS S3 (January 2026).

This daemon enables data recovery and bootstrap from S3:
- Pulls canonical databases from S3 for cluster restoration
- Downloads NPZ training files for new node bootstrap
- Retrieves model checkpoints for distributed training

Usage:
    from app.coordination.s3_import_daemon import S3ImportDaemon, get_s3_import_daemon

    daemon = get_s3_import_daemon()
    await daemon.import_from_s3(config_key="hex8_2p")  # Import specific config

Environment Variables:
    RINGRIFT_S3_BUCKET: S3 bucket name (default: ringrift-models-20251214)
    RINGRIFT_S3_REGION: AWS region (default: us-east-1)
    RINGRIFT_S3_IMPORT_ENABLED: Enable/disable daemon (default: true)
    RINGRIFT_S3_IMPORT_ON_STARTUP: Auto-import on startup (default: false)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.mixins.import_mixin import ImportDaemonMixin
from app.coordination.protocols import CoordinatorStatus

logger = logging.getLogger(__name__)


@dataclass
class S3ImportConfig:
    """Configuration for S3 import daemon."""

    bucket: str = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_S3_BUCKET", "ringrift-models-20251214"
        )
    )
    region: str = field(
        default_factory=lambda: os.environ.get("RINGRIFT_S3_REGION", "us-east-1")
    )
    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_S3_IMPORT_ENABLED", "true"
        ).lower() == "true"
    )
    import_on_startup: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_S3_IMPORT_ON_STARTUP", "false"
        ).lower() == "true"
    )

    # Check interval (for bootstrap mode)
    check_interval: float = 3600  # 1 hour

    # Local directories
    local_games_dir: str = "data/games"
    local_training_dir: str = "data/training"
    local_models_dir: str = "models"

    # S3 prefixes (matching S3PushDaemon structure)
    s3_games_prefix: str = "consolidated/games"
    s3_training_prefix: str = "consolidated/training"
    s3_models_prefix: str = "models"

    # Timeouts
    download_timeout: int = 600  # 10 minutes per file


@dataclass
class S3ImportStats:
    """Statistics for S3 import operations."""

    total_files_imported: int = 0
    total_bytes_imported: int = 0
    last_import_time: float = 0.0
    import_errors: int = 0
    last_error: str | None = None
    configs_imported: list[str] = field(default_factory=list)


@dataclass
class S3FileInfo:
    """Information about a file in S3."""

    key: str
    size: int
    last_modified: str
    etag: str


class S3ImportDaemon(HandlerBase, ImportDaemonMixin):
    """Daemon that imports training data from S3 for recovery and bootstrap.

    Provides on-demand import of canonical databases, training NPZ files,
    and model checkpoints from S3 backup.

    January 2026: Inherits from ImportDaemonMixin for file validation
    and atomic replacement functionality.
    """

    _instance: S3ImportDaemon | None = None

    # ImportDaemonMixin configuration
    IMPORT_LOG_PREFIX = "[S3Import]"
    IMPORT_VERIFY_CHECKSUMS = True

    def __init__(self, config: S3ImportConfig | None = None):
        """Initialize S3 import daemon.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or S3ImportConfig()
        super().__init__(name="s3_import", cycle_interval=self.config.check_interval)

        self._import_stats = S3ImportStats()
        self._imported_files: dict[str, str] = {}  # s3_key -> local_path
        self._s3_inventory: dict[str, S3FileInfo] = {}  # s3_key -> info
        self._base_path = Path(os.environ.get("RINGRIFT_BASE_PATH", "."))
        self._s3_available = True

    @classmethod
    def get_instance(cls) -> S3ImportDaemon:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def _check_aws_credentials(self) -> bool:
        """Check if AWS credentials are available."""
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get(
            "AWS_SECRET_ACCESS_KEY"
        ):
            return True

        aws_config = Path.home() / ".aws" / "credentials"
        if aws_config.exists():
            return True

        return False

    async def _list_s3_files(self, prefix: str) -> list[S3FileInfo]:
        """List files in S3 bucket with given prefix."""
        if not self._check_aws_credentials():
            logger.warning("[S3Import] AWS credentials not configured")
            return []

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "aws", "s3api", "list-objects-v2",
                    "--bucket", self.config.bucket,
                    "--prefix", prefix,
                    "--region", self.config.region,
                    "--output", "json",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                logger.warning(f"[S3Import] Failed to list S3: {result.stderr}")
                return []

            import json
            data = json.loads(result.stdout)
            files = []

            for item in data.get("Contents", []):
                files.append(S3FileInfo(
                    key=item["Key"],
                    size=item["Size"],
                    last_modified=item["LastModified"],
                    etag=item["ETag"].strip('"'),
                ))

            return files

        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[S3Import] S3 list error: {e}")
            return []

    async def _download_file(self, s3_key: str, local_path: Path) -> bool:
        """Download a file from S3.

        Args:
            s3_key: S3 object key
            local_path: Local destination path

        Returns:
            True if download succeeded
        """
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            s3_uri = f"s3://{self.config.bucket}/{s3_key}"

            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "aws", "s3", "cp",
                    s3_uri,
                    str(local_path),
                    "--region", self.config.region,
                ],
                capture_output=True,
                text=True,
                timeout=self.config.download_timeout,
            )

            if result.returncode == 0:
                file_size = local_path.stat().st_size

                # Validate downloaded file using ImportDaemonMixin
                validation = await self._validate_import(local_path)
                if not validation.valid:
                    logger.warning(
                        f"[S3Import] Downloaded file failed validation: "
                        f"{validation.error}"
                    )
                    self._import_stats.import_errors += 1
                    local_path.unlink(missing_ok=True)
                    return False

                self._imported_files[s3_key] = str(local_path)
                self._import_stats.total_files_imported += 1
                self._import_stats.total_bytes_imported += file_size
                logger.info(
                    f"[S3Import] Downloaded {s3_key} "
                    f"({file_size / (1024*1024):.1f} MB, validated)"
                )
                return True
            else:
                logger.warning(f"[S3Import] Download failed: {result.stderr}")
                self._import_stats.import_errors += 1
                return False

        except subprocess.TimeoutExpired:
            logger.warning(f"[S3Import] Download timed out for {s3_key}")
            self._import_stats.import_errors += 1
            return False
        except (OSError, IOError) as e:
            logger.warning(f"[S3Import] Download error: {e}")
            self._import_stats.import_errors += 1
            return False

    async def refresh_inventory(self) -> None:
        """Refresh the S3 inventory (list all available files)."""
        if not self._check_aws_credentials():
            self._s3_available = False
            return

        self._s3_inventory.clear()

        # List canonical databases
        db_files = await self._list_s3_files(self.config.s3_games_prefix)
        for f in db_files:
            self._s3_inventory[f.key] = f

        # List training NPZ files
        npz_files = await self._list_s3_files(self.config.s3_training_prefix)
        for f in npz_files:
            self._s3_inventory[f.key] = f

        # List models
        model_files = await self._list_s3_files(self.config.s3_models_prefix)
        for f in model_files:
            self._s3_inventory[f.key] = f

        self._s3_available = len(self._s3_inventory) > 0

        logger.info(f"[S3Import] Inventory refreshed: {len(self._s3_inventory)} files")

    async def import_from_s3(
        self,
        config_key: str | None = None,
        data_type: str = "all",  # "all", "games", "training", "models"
        force: bool = False,
    ) -> dict[str, Any]:
        """Import data from S3.

        Args:
            config_key: Optional config filter (e.g., "hex8_2p")
            data_type: Type of data to import
            force: Re-download even if file exists locally

        Returns:
            Dictionary with import results
        """
        if not self.config.enabled:
            return {"success": False, "error": "S3 import disabled"}

        if not self._check_aws_credentials():
            return {"success": False, "error": "AWS credentials not configured"}

        # Refresh inventory if empty
        if not self._s3_inventory:
            await self.refresh_inventory()

        results = {
            "success": True,
            "files_imported": 0,
            "bytes_imported": 0,
            "errors": [],
        }

        # Import canonical databases
        if data_type in ("all", "games"):
            for s3_key, info in self._s3_inventory.items():
                if not s3_key.startswith(self.config.s3_games_prefix):
                    continue
                if config_key and config_key not in s3_key:
                    continue

                filename = os.path.basename(s3_key)
                local_path = self._base_path / self.config.local_games_dir / filename

                if local_path.exists() and not force:
                    continue

                if await self._download_file(s3_key, local_path):
                    results["files_imported"] += 1
                    results["bytes_imported"] += info.size
                else:
                    results["errors"].append(f"Failed to download {s3_key}")

        # Import training NPZ files
        if data_type in ("all", "training"):
            for s3_key, info in self._s3_inventory.items():
                if not s3_key.startswith(self.config.s3_training_prefix):
                    continue
                if config_key and config_key not in s3_key:
                    continue

                filename = os.path.basename(s3_key)
                local_path = self._base_path / self.config.local_training_dir / filename

                if local_path.exists() and not force:
                    continue

                if await self._download_file(s3_key, local_path):
                    results["files_imported"] += 1
                    results["bytes_imported"] += info.size
                else:
                    results["errors"].append(f"Failed to download {s3_key}")

        # Import models
        if data_type in ("all", "models"):
            for s3_key, info in self._s3_inventory.items():
                if not s3_key.startswith(self.config.s3_models_prefix):
                    continue
                if config_key and config_key not in s3_key:
                    continue

                filename = os.path.basename(s3_key)
                local_path = self._base_path / self.config.local_models_dir / filename

                if local_path.exists() and not force:
                    continue

                if await self._download_file(s3_key, local_path):
                    results["files_imported"] += 1
                    results["bytes_imported"] += info.size
                else:
                    results["errors"].append(f"Failed to download {s3_key}")

        self._import_stats.last_import_time = time.time()

        if config_key and config_key not in self._import_stats.configs_imported:
            self._import_stats.configs_imported.append(config_key)

        # Emit event on successful import
        if results["files_imported"] > 0:
            self._emit_import_complete(results)

        return results

    async def import_for_recovery(self) -> dict[str, Any]:
        """Full recovery import - download all available data from S3.

        Use this after cluster disruption or when bootstrapping a new cluster.
        """
        logger.info("[S3Import] Starting full recovery import from S3")

        await self.refresh_inventory()

        results = await self.import_from_s3(data_type="all", force=False)

        logger.info(
            f"[S3Import] Recovery complete: "
            f"{results['files_imported']} files, "
            f"{results['bytes_imported'] / (1024*1024):.1f} MB"
        )

        return results

    async def import_missing_configs(self) -> dict[str, Any]:
        """Import data for configs that have no local data.

        Scans local directories and imports any configs available in S3
        but missing locally.
        """
        # Find configs available in S3
        await self.refresh_inventory()

        s3_configs = set()
        for s3_key in self._s3_inventory:
            # Extract config from filename like canonical_hex8_2p.db
            if "canonical_" in s3_key:
                parts = os.path.basename(s3_key).replace("canonical_", "").split(".")
                if parts:
                    config = parts[0]  # hex8_2p
                    s3_configs.add(config)

        # Find configs with local data
        local_configs = set()
        games_dir = self._base_path / self.config.local_games_dir
        if games_dir.exists():
            for db in games_dir.glob("canonical_*.db"):
                config = db.stem.replace("canonical_", "")
                local_configs.add(config)

        # Import missing configs
        missing = s3_configs - local_configs
        results = {
            "success": True,
            "missing_configs": list(missing),
            "imported": [],
            "errors": [],
        }

        for config in missing:
            import_result = await self.import_from_s3(config_key=config)
            if import_result.get("files_imported", 0) > 0:
                results["imported"].append(config)
            else:
                results["errors"].extend(import_result.get("errors", []))

        return results

    def _emit_import_complete(self, results: dict[str, Any]) -> None:
        """Emit S3_IMPORT_COMPLETE event."""
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.DATA_SYNC_COMPLETED,
                sync_type="s3_import",
                files_imported=results.get("files_imported", 0),
                bytes_imported=results.get("bytes_imported", 0),
                source="S3ImportDaemon",
            )
        except Exception as e:
            logger.debug(f"[S3Import] Could not emit event: {e}")

    async def _run_cycle(self) -> None:
        """Run one import cycle.

        In normal operation, this daemon is mostly passive and responds to
        explicit import requests. The cycle just refreshes inventory.
        """
        if not self.config.enabled:
            logger.debug("[S3Import] Disabled via config, skipping cycle")
            return

        # Refresh inventory periodically
        await self.refresh_inventory()

        # Bootstrap import on startup if enabled
        if self.config.import_on_startup and self._stats.cycles_completed == 0:
            logger.info("[S3Import] Startup import enabled, importing missing configs")
            await self.import_missing_configs()

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Get event subscriptions for this daemon."""
        return {
            "CLUSTER_RECOVERY_REQUESTED": self._on_recovery_requested,
            "NODE_BOOTSTRAP_REQUESTED": self._on_bootstrap_requested,
        }

    async def _on_recovery_requested(self, event: dict[str, Any]) -> None:
        """Handle cluster recovery request - import all data from S3."""
        logger.info("[S3Import] Recovery requested, starting full import")
        await self.import_for_recovery()

    async def _on_bootstrap_requested(self, event: dict[str, Any]) -> None:
        """Handle node bootstrap request - import missing configs."""
        logger.info("[S3Import] Bootstrap requested, importing missing configs")
        await self.import_missing_configs()

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="S3Import not running",
            )

        if not self._check_aws_credentials():
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="AWS credentials not configured",
                details={"s3_available": False},
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"S3 import ready, {len(self._s3_inventory)} files in inventory",
            details={
                "cycles_completed": self._stats.cycles_completed,
                "total_files_imported": self._import_stats.total_files_imported,
                "total_mb_imported": round(
                    self._import_stats.total_bytes_imported / (1024 * 1024), 2
                ),
                "s3_available": self._s3_available,
                "inventory_size": len(self._s3_inventory),
                "import_errors": self._import_stats.import_errors,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get current daemon statistics."""
        return {
            "total_files_imported": self._import_stats.total_files_imported,
            "total_bytes_imported": self._import_stats.total_bytes_imported,
            "total_mb_imported": round(
                self._import_stats.total_bytes_imported / (1024 * 1024), 2
            ),
            "last_import_time": self._import_stats.last_import_time,
            "import_errors": self._import_stats.import_errors,
            "last_error": self._import_stats.last_error,
            "inventory_size": len(self._s3_inventory),
            "configs_imported": self._import_stats.configs_imported,
            "s3_available": self._s3_available,
        }

    def get_inventory(self) -> dict[str, dict[str, Any]]:
        """Get current S3 inventory."""
        return {
            key: {
                "size": info.size,
                "last_modified": info.last_modified,
                "etag": info.etag,
            }
            for key, info in self._s3_inventory.items()
        }


def get_s3_import_daemon() -> S3ImportDaemon:
    """Get the singleton S3 import daemon instance."""
    return S3ImportDaemon.get_instance()


def reset_s3_import_daemon() -> None:
    """Reset the singleton instance (for testing)."""
    S3ImportDaemon.reset_instance()


# Factory function for daemon_runners.py
async def create_s3_import() -> None:
    """Create and run S3 import daemon (January 2026).

    Enables on-demand import of training data from S3 for cluster
    recovery and node bootstrap.
    """
    daemon = get_s3_import_daemon()
    await daemon.start()

    try:
        while daemon._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        await daemon.stop()
