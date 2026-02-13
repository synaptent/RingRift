"""S3 Node Sync Daemon - Automatic bi-directional S3 sync for cluster nodes.

This daemon runs on all cluster nodes to:
1. PUSH local game databases and models to S3 periodically
2. PULL training data (NPZ) and models from S3 before training
3. Track what data exists where via manifests

Architecture:
    Each node pushes to: s3://bucket/nodes/{node_id}/
    Consolidated data at: s3://bucket/consolidated/
    Coordinator pulls from nodes and creates consolidated view

Usage:
    # On any cluster node
    python -m app.coordination.s3_node_sync_daemon

    # Via environment
    RINGRIFT_S3_NODE_SYNC=true python scripts/master_loop.py

Configuration:
    RINGRIFT_S3_BUCKET - S3 bucket (default: ringrift-models-20251214)
    RINGRIFT_S3_SYNC_INTERVAL - Sync interval in seconds (default: 3600 = 1 hour)
    RINGRIFT_S3_PUSH_GAMES - Push game databases (default: true)
    RINGRIFT_S3_PUSH_MODELS - Push models (default: true)
    RINGRIFT_S3_PULL_NPZ - Pull NPZ files before training (default: true)

December 2025: Created for cluster-wide S3 backup and data distribution.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Add parent to path for imports
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.coordination.event_handler_utils import extract_config_key
from app.coordination.handler_base import HandlerBase
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult


def get_node_id() -> str:
    """Get unique node identifier."""
    # Try environment first
    node_id = os.getenv("RINGRIFT_NODE_ID")
    if node_id:
        return node_id

    # Try hostname
    import socket
    hostname = socket.gethostname()

    # Clean up common prefixes/suffixes
    for prefix in ["ip-", "instance-", "node-"]:
        if hostname.startswith(prefix):
            hostname = hostname[len(prefix):]

    return hostname


@dataclass
class S3NodeSyncConfig:
    """Configuration for S3 node sync daemon."""

    # S3 settings
    s3_bucket: str = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
    )
    aws_region: str = field(
        default_factory=lambda: os.getenv("AWS_REGION", "us-east-1")
    )

    # Sync settings
    sync_interval_seconds: float = field(
        default_factory=lambda: float(os.getenv("RINGRIFT_S3_SYNC_INTERVAL", "3600"))
    )

    # Push settings
    push_games: bool = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_PUSH_GAMES", "true").lower() == "true"
    )
    push_models: bool = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_PUSH_MODELS", "true").lower() == "true"
    )
    push_npz: bool = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_PUSH_NPZ", "true").lower() == "true"
    )

    # Pull settings
    pull_npz: bool = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_PULL_NPZ", "true").lower() == "true"
    )
    pull_models: bool = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_PULL_MODELS", "true").lower() == "true"
    )

    # Local paths
    games_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RINGRIFT_GAMES_DIR", "data/games"))
    )
    models_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RINGRIFT_MODELS_DIR", "models"))
    )
    npz_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RINGRIFT_NPZ_DIR", "data/training"))
    )

    # Bandwidth limit (KB/s, 0 = unlimited)
    bandwidth_limit_kbps: int = field(
        default_factory=lambda: int(os.getenv("RINGRIFT_S3_BANDWIDTH_LIMIT", "0"))
    )

    # Compression
    compress_uploads: bool = True

    # Retry settings
    retry_count: int = 3
    retry_delay_seconds: float = 30.0

    # Timeouts
    upload_timeout_seconds: float = 600.0
    download_timeout_seconds: float = 300.0


@dataclass
class S3SyncResult:
    """Result of an S3 sync operation.

    December 2025: Renamed from SyncResult to avoid collision with
    app.coordination.sync_constants.SyncResult (canonical).
    """

    success: bool
    uploaded_files: list[str] = field(default_factory=list)
    downloaded_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    bytes_transferred: int = 0


# Backward compatibility alias
SyncResult = S3SyncResult


@dataclass
class FileManifest:
    """Manifest of files on a node."""

    node_id: str
    timestamp: float
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    # files: {relative_path: {size, mtime, sha256, type}}


class S3NodeSyncDaemon(HandlerBase):
    """Daemon for bi-directional S3 sync on cluster nodes.

    Runs on every cluster node to ensure data is backed up to S3
    and training data is available when needed.

    January 2026: Migrated to HandlerBase for unified lifecycle management.
    """

    def __init__(self, config: S3NodeSyncConfig | None = None):
        self.config = config or S3NodeSyncConfig()
        self.node_id = get_node_id()

        # Initialize HandlerBase with cycle interval from config
        super().__init__(
            name=f"s3_node_sync_{self.node_id}",
            cycle_interval=self.config.sync_interval_seconds,
        )

        # Stats (in addition to HandlerBase stats)
        self._last_push_time: float = 0.0
        self._last_pull_time: float = 0.0
        self._push_count: int = 0
        self._pull_count: int = 0
        self._bytes_uploaded: int = 0
        self._bytes_downloaded: int = 0

        # Local manifest cache
        self._local_manifest: FileManifest | None = None

    # =========================================================================
    # HandlerBase Implementation
    # =========================================================================

    def _get_event_subscriptions(self) -> dict:
        """Return event subscriptions for HandlerBase.

        Enables event-driven sync instead of just interval-based:
        - TRAINING_COMPLETED: Push new model to S3 immediately
        - SELFPLAY_COMPLETE: Push games to S3 after significant batches
        - MODEL_PROMOTED: Trigger high-priority model backup
        """
        return {
            "training_completed": self._on_training_completed,
            "selfplay_complete": self._on_selfplay_complete,
            "model_promoted": self._on_model_promoted,
        }

    async def _on_start(self) -> None:
        """Hook called before main loop starts."""
        logger.info(
            f"S3NodeSyncDaemon starting on {self.node_id} "
            f"(bucket: {self.config.s3_bucket}, interval: {self.config.sync_interval_seconds}s)"
        )
        # Initial sync
        await self._run_push_cycle()

    async def _on_stop(self) -> None:
        """Hook called after main loop stops."""
        # Final push before shutdown
        logger.info("Running final S3 push before shutdown")
        await self._run_push_cycle()
        logger.info(f"S3NodeSyncDaemon stopped on {self.node_id}")

    async def _run_cycle(self) -> None:
        """Main work loop iteration - push local data to S3."""
        await self._run_push_cycle()

    def health_check(self) -> HealthCheckResult:
        """Check daemon health. Overrides HandlerBase for S3-specific checks."""
        # Get base health check
        base_health = super().health_check()

        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="S3NodeSyncDaemon not running",
            )

        # Check if we're syncing regularly
        time_since_push = time.time() - self._last_push_time if self._last_push_time > 0 else 0
        if time_since_push > self.config.sync_interval_seconds * 2:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"No push in {time_since_push:.0f}s",
                details={
                    "push_count": self._push_count,
                    "errors_count": self._stats.errors_count,
                    **base_health.details,
                },
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Healthy (pushes: {self._push_count}, errors: {self._stats.errors_count})",
            details={
                "node_id": self.node_id,
                "bytes_uploaded": self._bytes_uploaded,
                "bytes_downloaded": self._bytes_downloaded,
                **base_health.details,
            },
        )

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_training_completed(self, event: Any) -> None:
        """Handle TRAINING_COMPLETED event - trigger immediate S3 push.

        December 2025: Pushes new model to S3 immediately after training
        instead of waiting for the next interval-based sync.
        """
        try:
            payload = self._get_payload(event)
            config_key = self._extract_config_key(payload)

            logger.info(f"Training completed for {config_key}, triggering S3 push")

            # Use HandlerBase's safe task creation
            if self._running:
                self._safe_create_task(self._push_models(), context="S3 push models after training")
            self._record_success()

        except (KeyError, TypeError, ValueError) as e:
            self._record_error(f"Error handling TRAINING_COMPLETED event: {e}", e)

    def _on_selfplay_complete(self, event: Any) -> None:
        """Handle SELFPLAY_COMPLETE event - trigger S3 push for significant batches.

        December 2025: Only triggers sync for significant batches (>=100 games)
        to avoid excessive S3 operations from small selfplay runs.
        """
        try:
            payload = self._get_payload(event)
            games_count = payload.get("games_count") or payload.get("games_added", 0)
            config_key = self._extract_config_key(payload)

            # Only sync for significant batches (>=100 games)
            if games_count >= 100:
                logger.info(
                    f"Selfplay batch complete ({games_count} games for {config_key}), "
                    f"triggering S3 push"
                )
                if self._running:
                    self._safe_create_task(self._push_games(), context="S3 push games")
            else:
                logger.debug(f"Selfplay batch too small ({games_count} < 100), skipping S3 sync")
            self._record_success()

        except (KeyError, TypeError, ValueError) as e:
            self._record_error(f"Error handling SELFPLAY_COMPLETE event: {e}", e)

    def _on_model_promoted(self, event: Any) -> None:
        """Handle MODEL_PROMOTED event - high-priority model backup.

        December 2025: Immediately pushes promoted model to S3.
        Promoted models are critical and should be backed up ASAP.
        """
        try:
            payload = self._get_payload(event)
            config_key = self._extract_config_key(payload) or payload.get("board_type", "")

            logger.info(f"Model promoted ({config_key}), triggering high-priority S3 push")

            if self._running:
                self._safe_create_task(self._push_models(), context="S3 push promoted model")
            self._record_success()

        except (KeyError, TypeError, ValueError) as e:
            self._record_error(f"Error handling MODEL_PROMOTED event: {e}", e)

    async def _run_push_cycle(self) -> SyncResult:
        """Run a push cycle - upload local data to S3.

        January 2026: Reordered to upload manifest LAST, after all files are
        confirmed uploaded. This prevents race conditions where consolidation
        references files not yet in S3.
        """
        start_time = time.time()
        result = SyncResult(success=True)

        try:
            # Build local manifest first (still need it for file list)
            self._local_manifest = await self._build_local_manifest()

            # Push data files BEFORE manifest
            # Push game databases
            if self.config.push_games:
                games_result = await self._push_games()
                result.uploaded_files.extend(games_result.uploaded_files)
                result.errors.extend(games_result.errors)
                result.bytes_transferred += games_result.bytes_transferred

            # Push models
            if self.config.push_models:
                models_result = await self._push_models()
                result.uploaded_files.extend(models_result.uploaded_files)
                result.errors.extend(models_result.errors)
                result.bytes_transferred += models_result.bytes_transferred

            # Push NPZ files
            if self.config.push_npz:
                npz_result = await self._push_npz()
                result.uploaded_files.extend(npz_result.uploaded_files)
                result.errors.extend(npz_result.errors)
                result.bytes_transferred += npz_result.bytes_transferred

            # Push manifest LAST - after all files are confirmed uploaded
            # This ensures consolidation never references missing files
            await self._upload_manifest(self._local_manifest)

            result.duration_seconds = time.time() - start_time
            self._last_push_time = time.time()
            self._push_count += 1
            self._bytes_uploaded += result.bytes_transferred

            logger.info(
                f"Push cycle complete: {len(result.uploaded_files)} files, "
                f"{result.bytes_transferred / 1024 / 1024:.1f}MB in {result.duration_seconds:.1f}s"
            )

            if result.errors:
                logger.warning(f"Push errors: {result.errors}")
                result.success = False

        except (OSError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logger.error(f"Push cycle failed: {e}")
            result.success = False
            result.errors.append(str(e))
            self._record_error(f"Push cycle failed: {e}", e)

        return result

    async def _build_local_manifest(self) -> FileManifest:
        """Build manifest of local files."""
        manifest = FileManifest(
            node_id=self.node_id,
            timestamp=time.time(),
        )

        # Add game databases
        if self.config.games_dir.exists():
            for db_file in self.config.games_dir.glob("*.db"):
                if db_file.is_file():
                    manifest.files[f"games/{db_file.name}"] = {
                        "size": db_file.stat().st_size,
                        "mtime": db_file.stat().st_mtime,
                        "type": "database",
                    }

        # Add models
        if self.config.models_dir.exists():
            for model_file in self.config.models_dir.glob("*.pth"):
                if model_file.is_file() and not model_file.is_symlink():
                    manifest.files[f"models/{model_file.name}"] = {
                        "size": model_file.stat().st_size,
                        "mtime": model_file.stat().st_mtime,
                        "type": "model",
                    }

        # Add NPZ files
        if self.config.npz_dir.exists():
            for npz_file in self.config.npz_dir.glob("*.npz"):
                if npz_file.is_file():
                    manifest.files[f"training/{npz_file.name}"] = {
                        "size": npz_file.stat().st_size,
                        "mtime": npz_file.stat().st_mtime,
                        "type": "npz",
                    }

        return manifest

    async def _upload_manifest(self, manifest: FileManifest) -> None:
        """Upload manifest to S3."""
        manifest_path = f"nodes/{self.node_id}/manifest.json"

        manifest_data = {
            "node_id": manifest.node_id,
            "timestamp": manifest.timestamp,
            "timestamp_iso": datetime.fromtimestamp(manifest.timestamp).isoformat(),
            "files": manifest.files,
            "file_count": len(manifest.files),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(manifest_data, f, indent=2)
            temp_path = f.name

        try:
            await self._s3_upload(temp_path, manifest_path)
        finally:
            os.unlink(temp_path)

    async def _push_games(self) -> SyncResult:
        """Push game databases to S3."""
        result = SyncResult(success=True)

        if not self.config.games_dir.exists():
            return result

        for db_file in self.config.games_dir.glob("*.db"):
            if not db_file.is_file():
                continue

            # Skip very small databases (likely empty)
            if db_file.stat().st_size < 10000:  # < 10KB
                continue

            s3_path = f"nodes/{self.node_id}/games/{db_file.name}"

            # Check if needs upload (compare sizes)
            should_upload = await self._should_upload(db_file, s3_path)

            if should_upload:
                try:
                    uploaded = await self._s3_upload(str(db_file), s3_path)
                    if uploaded:
                        result.uploaded_files.append(db_file.name)
                        result.bytes_transferred += db_file.stat().st_size
                except (OSError, asyncio.TimeoutError) as e:
                    result.errors.append(f"{db_file.name}: {e}")

        return result

    async def _push_models(self) -> SyncResult:
        """Push model checkpoints to S3."""
        result = SyncResult(success=True)

        if not self.config.models_dir.exists():
            return result

        for model_file in self.config.models_dir.glob("*.pth"):
            if not model_file.is_file() or model_file.is_symlink():
                continue

            # Only push canonical models
            if not model_file.name.startswith("canonical_"):
                continue

            s3_path = f"nodes/{self.node_id}/models/{model_file.name}"

            should_upload = await self._should_upload(model_file, s3_path)

            if should_upload:
                try:
                    uploaded = await self._s3_upload(str(model_file), s3_path)
                    if uploaded:
                        result.uploaded_files.append(model_file.name)
                        result.bytes_transferred += model_file.stat().st_size
                except (OSError, asyncio.TimeoutError) as e:
                    result.errors.append(f"{model_file.name}: {e}")

        return result

    async def _push_npz(self) -> SyncResult:
        """Push NPZ training files to S3."""
        result = SyncResult(success=True)

        if not self.config.npz_dir.exists():
            return result

        for npz_file in self.config.npz_dir.glob("*.npz"):
            if not npz_file.is_file():
                continue

            s3_path = f"nodes/{self.node_id}/training/{npz_file.name}"

            should_upload = await self._should_upload(npz_file, s3_path)

            if should_upload:
                try:
                    uploaded = await self._s3_upload(str(npz_file), s3_path)
                    if uploaded:
                        result.uploaded_files.append(npz_file.name)
                        result.bytes_transferred += npz_file.stat().st_size
                except (OSError, asyncio.TimeoutError) as e:
                    result.errors.append(f"{npz_file.name}: {e}")

        return result

    async def _should_upload(self, local_path: Path, s3_path: str) -> bool:
        """Check if file should be uploaded (not in S3 or different size)."""
        try:
            # Check if exists in S3 with same size
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

            if process.returncode != 0:
                # File doesn't exist in S3
                return True

            # Parse response to get size
            response = json.loads(stdout.decode())
            s3_size = response.get("ContentLength", 0)
            local_size = local_path.stat().st_size

            # Upload if sizes differ
            return local_size != s3_size

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Dec 2025: Narrowed from bare Exception - S3 response parsing failed
            logger.warning(f"S3 head-object response parsing failed for {s3_path}: {e}")
            return True
        except (OSError, asyncio.TimeoutError, asyncio.CancelledError) as e:
            # Dec 2025: Network/process errors - assume upload needed
            logger.debug(f"S3 head-object failed for {s3_path} (will upload): {e}")
            return True

    async def _s3_upload(self, local_path: str, s3_path: str) -> bool:
        """Upload file to S3."""
        s3_uri = f"s3://{self.config.s3_bucket}/{s3_path}"

        cmd = ["aws", "s3", "cp", local_path, s3_uri]

        if self.config.bandwidth_limit_kbps > 0:
            # AWS CLI uses bytes/second for --expected-size
            cmd.extend(["--only-show-errors"])

        logger.debug(f"Uploading {local_path} to {s3_uri}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.upload_timeout_seconds,
            )

            if process.returncode != 0:
                logger.warning(f"S3 upload failed: {stderr.decode()}")
                return False

            return True

        except asyncio.TimeoutError:
            logger.warning(f"S3 upload timed out: {local_path}")
            process.kill()
            return False

    async def pull_training_data(self, config_key: str) -> SyncResult:
        """Pull training data from S3 for a specific config.

        This is called before training to ensure we have the latest
        consolidated training data.
        """
        result = SyncResult(success=True)

        # Download consolidated NPZ
        s3_path = f"consolidated/training/{config_key}.npz"
        local_path = self.config.npz_dir / f"{config_key}.npz"

        # Ensure directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            downloaded = await self._s3_download(s3_path, str(local_path))
            if downloaded:
                result.downloaded_files.append(config_key + ".npz")
                result.bytes_transferred += local_path.stat().st_size
                logger.info(f"Downloaded training data for {config_key}")
            else:
                logger.warning(f"No consolidated training data for {config_key} in S3")
        except (OSError, asyncio.TimeoutError) as e:
            result.errors.append(f"{config_key}.npz: {e}")

        self._pull_count += 1
        self._bytes_downloaded += result.bytes_transferred

        return result

    async def pull_model(self, model_name: str) -> SyncResult:
        """Pull a model from S3."""
        result = SyncResult(success=True)

        s3_path = f"consolidated/models/{model_name}"
        local_path = self.config.models_dir / model_name

        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            downloaded = await self._s3_download(s3_path, str(local_path))
            if downloaded:
                result.downloaded_files.append(model_name)
                result.bytes_transferred += local_path.stat().st_size
                logger.info(f"Downloaded model {model_name}")
        except (OSError, asyncio.TimeoutError) as e:
            result.errors.append(f"{model_name}: {e}")

        return result

    async def _s3_download(self, s3_path: str, local_path: str) -> bool:
        """Download file from S3."""
        s3_uri = f"s3://{self.config.s3_bucket}/{s3_path}"

        cmd = ["aws", "s3", "cp", s3_uri, local_path, "--only-show-errors"]

        logger.debug(f"Downloading {s3_uri} to {local_path}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.download_timeout_seconds,
            )

            if process.returncode != 0:
                error_msg = stderr.decode()
                if "404" in error_msg or "NoSuchKey" in error_msg:
                    return False
                logger.warning(f"S3 download failed: {error_msg}")
                return False

            return True

        except asyncio.TimeoutError:
            logger.warning(f"S3 download timed out: {s3_path}")
            process.kill()
            return False

    async def list_all_node_data(self) -> dict[str, FileManifest]:
        """List data from all nodes in S3."""
        manifests: dict[str, FileManifest] = {}

        # List all node directories
        cmd = [
            "aws", "s3", "ls",
            f"s3://{self.config.s3_bucket}/nodes/",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()

        if process.returncode != 0:
            logger.warning("Failed to list nodes in S3")
            return manifests

        # Parse node directories
        for line in stdout.decode().strip().split("\n"):
            if not line.strip():
                continue
            # Format: "PRE node_id/"
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "PRE":
                node_id = parts[1].rstrip("/")

                # Get manifest for this node
                manifest = await self._get_node_manifest(node_id)
                if manifest:
                    manifests[node_id] = manifest

        return manifests

    async def _get_node_manifest(self, node_id: str) -> FileManifest | None:
        """Get manifest for a specific node."""
        s3_path = f"nodes/{node_id}/manifest.json"

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            downloaded = await self._s3_download(s3_path, temp_path)
            if not downloaded:
                return None

            with open(temp_path) as f:
                data = json.load(f)

            return FileManifest(
                node_id=data["node_id"],
                timestamp=data["timestamp"],
                files=data["files"],
            )
        except (OSError, json.JSONDecodeError, KeyError, asyncio.TimeoutError) as e:
            logger.warning(f"Failed to get manifest for {node_id}: {e}")
            return None
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class S3ConsolidationDaemon:
    """Daemon that consolidates data from all nodes into unified view.

    Runs on coordinator only. Collects data from all node pushes and
    creates consolidated training data.
    """

    def __init__(self, config: S3NodeSyncConfig | None = None):
        self.config = config or S3NodeSyncConfig()
        self._running = False
        self._consolidation_interval = 3600.0  # 1 hour
        # Health tracking (Dec 2025)
        self._last_consolidation_time: float = 0.0
        self._consolidation_errors: int = 0
        self._nodes_consolidated: int = 0
        self._models_consolidated: int = 0
        self._npz_consolidated: int = 0

    async def start(self) -> None:
        """Start consolidation daemon."""
        self._running = True
        logger.info("S3ConsolidationDaemon starting")

        while self._running:
            try:
                await self._run_consolidation()
                self._last_consolidation_time = time.time()
                await asyncio.sleep(self._consolidation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation error: {e}")
                self._consolidation_errors += 1
                await asyncio.sleep(60.0)

    async def stop(self) -> None:
        """Stop consolidation daemon."""
        self._running = False

    def health_check(self) -> "HealthCheckResult":
        """Return health status for DaemonManager integration (Dec 2025).

        Implements the health check protocol required for daemon monitoring.

        Returns:
            HealthCheckResult with status and metrics
        """
        # Import from contracts (zero dependencies)
        from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

        # Check staleness - if no consolidation in 2x interval, mark degraded
        max_age = self._consolidation_interval * 2
        is_stale = (
            self._last_consolidation_time > 0
            and (time.time() - self._last_consolidation_time) > max_age
        )

        # Determine health status
        if not self._running:
            status = CoordinatorStatus.STOPPED
            healthy = False
            message = "S3ConsolidationDaemon stopped"
        elif is_stale:
            status = CoordinatorStatus.DEGRADED
            healthy = False
            message = f"Consolidation stale (last: {int(time.time() - self._last_consolidation_time)}s ago)"
        elif self._consolidation_errors > 5:
            status = CoordinatorStatus.DEGRADED
            healthy = False
            message = f"Excessive consolidation errors: {self._consolidation_errors}"
        else:
            status = CoordinatorStatus.RUNNING
            healthy = True
            message = "S3ConsolidationDaemon operational"

        return HealthCheckResult(
            healthy=healthy,
            status=status,
            message=message,
            details={
                "running": self._running,
                "last_consolidation_time": self._last_consolidation_time,
                "nodes_consolidated": self._nodes_consolidated,
                "models_consolidated": self._models_consolidated,
                "npz_consolidated": self._npz_consolidated,
                "consolidation_errors": self._consolidation_errors,
            },
        )

    async def _run_consolidation(self) -> None:
        """Consolidate data from all nodes."""
        sync_daemon = S3NodeSyncDaemon(self.config)

        # Get all node manifests
        manifests = await sync_daemon.list_all_node_data()
        logger.info(f"Found data from {len(manifests)} nodes")

        if not manifests:
            return

        # For each type of data, consolidate
        await self._consolidate_models(manifests)
        await self._consolidate_npz(manifests)

        # Create consolidated manifest
        await self._create_consolidated_manifest(manifests)

    async def _consolidate_models(self, manifests: dict[str, FileManifest]) -> None:
        """Consolidate models - keep latest version of each.

        January 2026: Updated to track skip counts and handle race conditions
        where manifest references models not yet uploaded to S3.
        """
        # Find all model files across nodes
        model_versions: dict[str, tuple[str, float]] = {}  # name -> (node, mtime)

        for node_id, manifest in manifests.items():
            for path, info in manifest.files.items():
                if info.get("type") == "model":
                    name = Path(path).name
                    mtime = info.get("mtime", 0)

                    if name not in model_versions or mtime > model_versions[name][1]:
                        model_versions[name] = (node_id, mtime)

        # Copy latest versions to consolidated with skip tracking
        copied_count = 0
        skipped_count = 0
        failed_count = 0

        for model_name, (node_id, _) in model_versions.items():
            src = f"nodes/{node_id}/models/{model_name}"
            dst = f"consolidated/models/{model_name}"
            success, status = await self._s3_copy(src, dst)

            if status == "copied":
                copied_count += 1
                logger.info(f"Consolidated model {model_name} from {node_id}")
            elif status == "skipped_missing":
                skipped_count += 1
            else:
                failed_count += 1

        self._models_consolidated += copied_count

        if skipped_count > 0 or failed_count > 0:
            logger.info(
                f"Model consolidation: {copied_count} copied, "
                f"{skipped_count} skipped (not yet uploaded), {failed_count} failed"
            )

    async def _consolidate_npz(self, manifests: dict[str, FileManifest]) -> None:
        """Consolidate NPZ files - keep latest version of each config.

        January 2026: Updated to track skip counts and handle race conditions
        where manifest references NPZ files not yet uploaded to S3.
        """
        npz_versions: dict[str, tuple[str, float]] = {}

        for node_id, manifest in manifests.items():
            for path, info in manifest.files.items():
                if info.get("type") == "npz":
                    name = Path(path).name
                    mtime = info.get("mtime", 0)

                    if name not in npz_versions or mtime > npz_versions[name][1]:
                        npz_versions[name] = (node_id, mtime)

        # Copy latest versions to consolidated with skip tracking
        copied_count = 0
        skipped_count = 0
        failed_count = 0

        for npz_name, (node_id, _) in npz_versions.items():
            src = f"nodes/{node_id}/training/{npz_name}"
            dst = f"consolidated/training/{npz_name}"
            success, status = await self._s3_copy(src, dst)

            if status == "copied":
                copied_count += 1
                logger.info(f"Consolidated NPZ {npz_name} from {node_id}")
            elif status == "skipped_missing":
                skipped_count += 1
            else:
                failed_count += 1

        self._npz_consolidated += copied_count

        if skipped_count > 0 or failed_count > 0:
            logger.info(
                f"NPZ consolidation: {copied_count} copied, "
                f"{skipped_count} skipped (not yet uploaded), {failed_count} failed"
            )

    async def _create_consolidated_manifest(
        self, manifests: dict[str, FileManifest]
    ) -> None:
        """Create a consolidated manifest of all data."""
        consolidated = {
            "timestamp": time.time(),
            "timestamp_iso": datetime.now().isoformat(),
            "node_count": len(manifests),
            "nodes": {},
            "summary": {
                "total_games": 0,
                "total_models": 0,
                "total_npz": 0,
            },
        }

        for node_id, manifest in manifests.items():
            games = sum(1 for p, i in manifest.files.items() if i.get("type") == "database")
            models = sum(1 for p, i in manifest.files.items() if i.get("type") == "model")
            npz = sum(1 for p, i in manifest.files.items() if i.get("type") == "npz")

            consolidated["nodes"][node_id] = {
                "last_seen": manifest.timestamp,
                "games": games,
                "models": models,
                "npz": npz,
            }

            consolidated["summary"]["total_games"] += games
            consolidated["summary"]["total_models"] += models
            consolidated["summary"]["total_npz"] += npz

        # Upload consolidated manifest
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(consolidated, f, indent=2)
            temp_path = f.name

        try:
            await self._s3_upload(temp_path, "consolidated/manifest.json")
            logger.info(f"Updated consolidated manifest: {consolidated['summary']}")
        finally:
            os.unlink(temp_path)

    async def _s3_key_exists(self, s3_key: str) -> bool:
        """Check if a key exists in S3.

        January 2026: Added to pre-validate S3 keys before copy operations,
        preventing 404 errors during consolidation when manifest is uploaded
        before model files.

        Returns:
            True if key exists, False otherwise.
        """
        cmd = [
            "aws", "s3api", "head-object",
            "--bucket", self.config.s3_bucket,
            "--key", s3_key,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()

        return process.returncode == 0

    async def _s3_copy(self, src: str, dst: str) -> tuple[bool, str]:
        """Copy file within S3.

        January 2026: Updated to pre-validate source existence and return
        status for better error handling during consolidation.

        Returns:
            Tuple of (success, status) where status is one of:
            - "copied" - file was successfully copied
            - "skipped_missing" - source file does not exist (expected race condition)
            - "failed" - copy failed for other reason
        """
        # Pre-validate source exists before attempting copy
        if not await self._s3_key_exists(src):
            logger.info(f"S3 copy skipped (source not yet uploaded): {src}")
            return (False, "skipped_missing")

        cmd = [
            "aws", "s3", "cp",
            f"s3://{self.config.s3_bucket}/{src}",
            f"s3://{self.config.s3_bucket}/{dst}",
            "--only-show-errors",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await process.communicate()

        if process.returncode != 0:
            logger.warning(f"S3 copy failed: {stderr.decode()}")
            return (False, "failed")

        return (True, "copied")

    async def _s3_upload(self, local_path: str, s3_path: str) -> bool:
        """Upload file to S3."""
        s3_uri = f"s3://{self.config.s3_bucket}/{s3_path}"

        cmd = ["aws", "s3", "cp", local_path, s3_uri, "--only-show-errors"]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await process.communicate()

        if process.returncode != 0:
            logger.warning(f"S3 upload failed: {stderr.decode()}")
            return False

        return True


# =============================================================================
# HandlerBase wrapper for S3ConsolidationDaemon (January 2026)
# =============================================================================


class S3ConsolidationHandler(HandlerBase):
    """HandlerBase wrapper for S3ConsolidationDaemon.

    January 2026: Added for unified daemon lifecycle management.
    Runs S3 consolidation on the coordinator node.
    """

    # Default cycle interval (1 hour for consolidation)
    DEFAULT_CYCLE_INTERVAL = 3600.0

    def __init__(
        self,
        config: S3NodeSyncConfig | None = None,
        cycle_interval: float = DEFAULT_CYCLE_INTERVAL,
    ):
        super().__init__(
            name="s3_consolidation",
            cycle_interval=cycle_interval,
        )
        self._daemon = S3ConsolidationDaemon(config)

    async def _run_cycle(self) -> None:
        """Run one consolidation cycle."""
        try:
            await self._daemon._run_consolidation()
            self._daemon._last_consolidation_time = time.time()
            logger.debug(
                f"[S3ConsolidationHandler] Cycle complete: "
                f"{self._daemon._nodes_consolidated} nodes, "
                f"{self._daemon._models_consolidated} models"
            )
        except Exception as e:
            logger.error(f"[S3ConsolidationHandler] Consolidation error: {e}")
            self._daemon._consolidation_errors += 1

    def _get_event_subscriptions(self) -> dict:
        """Get event subscriptions for consolidation."""
        return {
            "MODEL_PROMOTED": self._on_model_promoted,
            "TRAINING_COMPLETED": self._on_training_completed,
        }

    async def _on_model_promoted(self, event: dict) -> None:
        """Handle model promotion - trigger consolidation."""
        logger.info("[S3ConsolidationHandler] Model promoted, scheduling consolidation")
        # Just log - next cycle will pick it up

    async def _on_training_completed(self, event: dict) -> None:
        """Handle training completion."""
        from app.coordination.event_router import get_event_payload
        payload = get_event_payload(event)
        config_key = payload.get("config_key", "unknown")
        logger.info(f"[S3ConsolidationHandler] Training completed for {config_key}")

    def health_check(self) -> "HealthCheckResult":
        """Health check delegating to wrapped daemon."""
        return self._daemon.health_check()


# Convenience functions for use in training scripts

async def ensure_training_data_from_s3(config_key: str) -> bool:
    """Ensure training data is available, pulling from S3 if needed.

    Call this before training to ensure we have the latest consolidated
    training data from across the cluster.

    Args:
        config_key: Config key like "hex8_2p", "square8_3p"

    Returns:
        True if training data is available
    """
    config = S3NodeSyncConfig()
    local_path = config.npz_dir / f"{config_key}.npz"

    if local_path.exists():
        # Check if S3 has newer version
        daemon = S3NodeSyncDaemon(config)
        # For now, just use local if exists
        logger.info(f"Using local training data: {local_path}")
        return True

    # Need to pull from S3
    daemon = S3NodeSyncDaemon(config)
    result = await daemon.pull_training_data(config_key)

    return result.success and len(result.downloaded_files) > 0


def sync_ensure_training_data_from_s3(config_key: str) -> bool:
    """Synchronous wrapper for ensure_training_data_from_s3."""
    return asyncio.run(ensure_training_data_from_s3(config_key))


async def main() -> None:
    """Run daemon standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="S3 Node Sync Daemon")
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Run as consolidation daemon (coordinator only)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.consolidate:
        daemon = S3ConsolidationDaemon()
    else:
        daemon = S3NodeSyncDaemon()

    if args.once:
        if isinstance(daemon, S3NodeSyncDaemon):
            await daemon._run_push_cycle()
        else:
            await daemon._run_consolidation()
    else:
        try:
            await daemon.start()
        except KeyboardInterrupt:
            await daemon.stop()


if __name__ == "__main__":
    asyncio.run(main())
