"""Data Sync HTTP Handlers Mixin.

Provides HTTP endpoints for P2P data synchronization across the cluster.
Enables distributed game data, model, and training file sharing.

Usage:
    class P2POrchestrator(SyncHandlersMixin, ...):
        pass

Endpoints:
    POST /sync/start - Leader initiates cluster-wide data sync
    GET /sync/status - Get current sync status (plan, active jobs)
    POST /sync/push - Receive pushed data from GPU node (with disk space checks)
    POST /sync/receipt - Request sync receipt verification
    GET /sync/receipts - Get sync receipts statistics
    POST /sync/pull - Handle incoming request to pull files from a source node
    GET /sync/file - Stream a data file to a peer (with auth)
    POST /sync/job_update - Worker reports sync job status back to leader

Sync Mechanism (December 2025):
    - Push-based sync: GPU nodes push data before disk fills (Phase 3b)
    - Pull-based sync: Nodes pull from each other via rsync
    - Receipt verification: Ensures data integrity before cleanup
    - Multi-tier storage: Local -> OWC -> S3 fallback

Storage Tiers:
    - local: Primary storage on coordinator disk
    - owc: External OWC drive (mac-studio) when local >80%
    - s3: S3 bucket fallback when OWC unavailable

Created: December 28, 2025
Extracted from: scripts/p2p_orchestrator.py (lines 10312-11016)
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import shutil
import time
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import handler_timeout, HANDLER_TIMEOUT_DELIVERY

if TYPE_CHECKING:
    from scripts.p2p.models import DataManifest, SyncJob, SyncPlan


logger = logging.getLogger(__name__)

# Storage tier configuration (from Phase 3b)
# Disk thresholds from app.config.thresholds (canonical source)
try:
    from app.config.thresholds import DISK_CRITICAL_PERCENT, DISK_PRODUCTION_HALT_PERCENT
    CRITICAL_DISK_THRESHOLD = float(DISK_CRITICAL_PERCENT)  # Reject pushes above this
    _SYNC_DISK_THRESHOLD = float(DISK_PRODUCTION_HALT_PERCENT)
except ImportError:
    CRITICAL_DISK_THRESHOLD = 90.0  # Reject pushes above this
    _SYNC_DISK_THRESHOLD = 85.0
OWC_FALLBACK_THRESHOLD = 80.0   # Use OWC above this
OWC_MOUNT_PATHS = [
    Path("/Volumes/RingRift-Data"),
    Path("/Volumes/OWC-1"),
    Path("/mnt/owc"),
]


class SyncHandlersMixin(BaseP2PHandler):
    """Mixin providing data sync HTTP handlers.

    Inherits from BaseP2PHandler to use standardized response formatting
    (json_response, error_response) and common utilities.

    Requires the implementing class to have:
    - node_id: str
    - leader_id: str | None
    - sync_lock: Lock
    - sync_in_progress: bool
    - last_sync_time: float
    - auto_sync_interval: int
    - current_sync_plan: SyncPlan | None
    - active_sync_jobs: dict[str, SyncJob]
    - pending_sync_requests: list
    - peers: dict[str, NodeInfo]
    - peers_lock: Lock
    - self_info: NodeInfo
    - auth_token: str | None
    - start_cluster_sync() -> dict
    - get_data_directory() -> Path
    - _is_leader() -> bool
    - _proxy_to_leader(request) -> web.Response
    - _is_request_authorized(request) -> bool
    - _handle_sync_pull_request(...) -> dict
    """

    # Type hints for IDE support
    node_id: str
    leader_id: str | None
    sync_lock: Lock
    sync_in_progress: bool
    last_sync_time: float
    auto_sync_interval: int
    current_sync_plan: Any  # SyncPlan | None
    active_sync_jobs: dict[str, Any]  # dict[str, SyncJob]
    pending_sync_requests: list
    peers: dict[str, Any]  # dict[str, NodeInfo]
    peers_lock: Lock
    self_info: Any  # NodeInfo
    auth_token: str | None

    # =========================================================================
    # Sync Handlers
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_sync_start(self, request: web.Request) -> web.Response:
        """POST /sync/start - Leader initiates a cluster-wide data sync.

        Only the leader can start a sync. This collects manifests from all nodes,
        generates a sync plan, and dispatches rsync jobs to nodes.
        """
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)
            if not self._is_leader():
                return self.json_response({
                    "error": "Not the leader. Only leader can start cluster sync.",
                    "leader_id": self.leader_id,
                }, status=403)

            result = await self.start_cluster_sync()
            return self.json_response(result)
        except Exception as e:  # noqa: BLE001
            logger.error(f"in handle_sync_start: {e}")
            import traceback
            traceback.print_exc()
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_sync_status(self, request: web.Request) -> web.Response:
        """GET /sync/status - Get current sync status.

        Returns the current sync plan (if any), active sync jobs, and overall status.
        """
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                proxied = await self._proxy_to_leader(request)
                if proxied.status not in (502, 503):
                    return proxied

            with self.sync_lock:
                sync_plan_dict = self.current_sync_plan.to_dict() if self.current_sync_plan else None
                active_jobs_dict = {
                    job_id: job.to_dict()
                    for job_id, job in self.active_sync_jobs.items()
                }

            return self.json_response({
                "node_id": self.node_id,
                "is_leader": self._is_leader(),
                "sync_in_progress": self.sync_in_progress,
                "last_sync_time": self.last_sync_time,
                "auto_sync_interval": self.auto_sync_interval,
                "current_sync_plan": sync_plan_dict,
                "active_sync_jobs": active_jobs_dict,
                "pending_sync_requests": len(self.pending_sync_requests),
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"in handle_sync_status: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_sync_push(self, request: web.Request) -> web.Response:
        """POST /sync/push - Receive pushed data from GPU node.

        December 2025: Added for push-based sync with verified cleanup.

        GPU nodes push selfplay data to coordinator before disk fills.
        This endpoint:
        1. Checks coordinator disk space - rejects if >90% full
        2. Falls back to OWC external drive if local disk >80%
        3. Falls back to S3 if OWC unavailable
        4. Receives file content (inline for small files, or notification for large)
        5. Verifies checksum
        6. Stores file locally/OWC/S3
        7. Returns receipt confirming storage

        Request body:
        {
            "file_path": "data/games/selfplay_hex8.db",
            "checksum": "sha256hex...",
            "file_size": 12345678,
            "source_node": "nebius-h100-1",
            "content": "base64...",  # Optional for small files
            "pull_required": true     # If content not provided
        }

        Response:
        {
            "status": "received",
            "checksum_verified": true,
            "stored_at": "/path/to/file",
            "storage_tier": "local|owc|s3",
            "node_id": "coordinator-node"
        }
        """
        try:
            data = await request.json()
            file_path = data.get("file_path", "")
            expected_checksum = data.get("checksum", "")
            source_node = data.get("source_node", "")
            file_size = data.get("file_size", 0)
            content_b64 = data.get("content")
            pull_required = data.get("pull_required", False)

            if not file_path or not expected_checksum:
                return self.error_response(
                    "Missing file_path or checksum",
                    status=400
                )

            # =================================================================
            # Check disk space and determine storage tier
            # =================================================================

            # Check local disk usage
            try:
                local_data_dir = Path("data")
                if local_data_dir.exists():
                    usage = shutil.disk_usage(local_data_dir)
                    local_disk_usage = (usage.used / usage.total) * 100
                else:
                    local_disk_usage = 0.0
            except Exception as e:
                logger.warning(f"Could not check local disk: {e}")
                local_disk_usage = 0.0

            # Determine storage tier
            storage_tier = "local"
            owc_path: Path | None = None
            S3_BUCKET = os.environ.get("RINGRIFT_S3_BUCKET", "")
            S3_PREFIX = os.environ.get("RINGRIFT_S3_PREFIX", "sync-archive")

            if local_disk_usage >= CRITICAL_DISK_THRESHOLD:
                # Disk critically full - try OWC, then S3, then reject
                for mount_path in OWC_MOUNT_PATHS:
                    if mount_path.exists():
                        try:
                            owc_usage = shutil.disk_usage(mount_path)
                            if (owc_usage.used / owc_usage.total) * 100 < 90:
                                owc_path = mount_path
                                storage_tier = "owc"
                                break
                        except (OSError, PermissionError) as e:
                            logger.warning(f"Cannot check OWC mount {mount_path}: {e}")
                            continue

                if storage_tier == "local" and S3_BUCKET:
                    storage_tier = "s3"
                elif storage_tier == "local":
                    # No fallback available - reject push
                    return self.json_response({
                        "error": "Coordinator disk full (>90%), no fallback available",
                        "disk_usage_percent": local_disk_usage,
                        "suggestion": "Wait for cleanup or increase storage",
                    }, status=503)

            elif local_disk_usage >= OWC_FALLBACK_THRESHOLD:
                # Disk getting full - prefer OWC if available
                for mount_path in OWC_MOUNT_PATHS:
                    if mount_path.exists():
                        try:
                            owc_usage = shutil.disk_usage(mount_path)
                            if (owc_usage.used / owc_usage.total) * 100 < 80:
                                owc_path = mount_path
                                storage_tier = "owc"
                                break
                        except (OSError, PermissionError) as e:
                            logger.warning(f"Cannot check OWC mount {mount_path}: {e}")
                            continue

            # Determine destination path
            # Use relative path structure under data/
            if file_path.startswith("/"):
                # Absolute path - extract relative portion
                parts = file_path.split("/")
                # Find 'data' in path and use everything after
                try:
                    data_idx = parts.index("data")
                    rel_path = "/".join(parts[data_idx:])
                except ValueError:
                    rel_path = parts[-1]
            else:
                rel_path = file_path

            # Resolve to actual storage path based on tier
            if storage_tier == "owc" and owc_path:
                if rel_path.startswith("data/"):
                    local_path = owc_path / rel_path
                else:
                    local_path = owc_path / "data" / rel_path
            elif storage_tier == "s3":
                # For S3, we'll upload after decoding
                local_path = Path(f"/tmp/s3_upload/{rel_path}")
            else:
                # Local storage
                local_data_dir = Path("data")
                if rel_path.startswith("data/"):
                    local_path = Path(rel_path)
                else:
                    local_path = local_data_dir / rel_path

            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            checksum_verified = False
            stored_path = str(local_path)

            if content_b64:
                # Inline content - decode and verify
                try:
                    content = base64.b64decode(content_b64)
                except Exception as e:
                    return self.error_response(
                        f"Invalid base64 content: {e}",
                        status=400
                    )

                # Verify checksum
                computed_checksum = hashlib.sha256(content).hexdigest()
                if computed_checksum != expected_checksum:
                    return self.json_response({
                        "error": "Checksum mismatch",
                        "expected": expected_checksum[:16] + "...",
                        "got": computed_checksum[:16] + "...",
                    }, status=400)

                checksum_verified = True

                # Write file to appropriate storage tier
                if storage_tier == "s3":
                    # Upload to S3
                    try:
                        import boto3
                        s3_client = boto3.client('s3')
                        s3_key = f"{S3_PREFIX}/{rel_path}"
                        s3_client.put_object(
                            Bucket=S3_BUCKET,
                            Key=s3_key,
                            Body=content,
                            Metadata={
                                "source_node": source_node,
                                "checksum": expected_checksum,
                            }
                        )
                        stored_path = f"s3://{S3_BUCKET}/{s3_key}"
                        logger.info(
                            f"Uploaded pushed file to S3 from {source_node}: {stored_path} "
                            f"({file_size} bytes, checksum verified)"
                        )
                    except ImportError:
                        return self.json_response({
                            "error": "S3 storage configured but boto3 not available",
                        }, status=503)
                    except Exception as e:
                        return self.json_response({
                            "error": f"S3 upload failed: {e}",
                        }, status=500)
                else:
                    # Write to local/OWC storage
                    local_path.write_bytes(content)
                    stored_path = str(local_path)
                    logger.info(
                        f"Received pushed file from {source_node}: {local_path} "
                        f"({file_size} bytes, checksum verified, tier={storage_tier})"
                    )

            elif pull_required:
                # Large file - queue rsync pull from source node
                logger.info(
                    f"Received push notification from {source_node}: {file_path} "
                    f"({file_size} bytes, queuing rsync pull)"
                )

                # Queue the rsync pull request
                self._queue_rsync_pull(
                    source_node=source_node,
                    file_path=file_path,
                    local_path=str(local_path),
                    expected_checksum=expected_checksum,
                    file_size=file_size,
                )

            else:
                return self.error_response(
                    "No content and pull_required not set",
                    status=400
                )

            # Register in manifest if available
            try:
                from app.distributed.cluster_manifest import (
                    SyncReceipt,
                    get_cluster_manifest,
                )

                manifest = get_cluster_manifest()

                # Register incoming file in manifest
                receipt = SyncReceipt(
                    file_path=str(local_path),
                    file_checksum=expected_checksum,
                    synced_to=self.node_id,
                    synced_at=time.time(),
                    verified=checksum_verified,
                    file_size=file_size,
                    source_node=source_node,
                )
                manifest.register_sync_receipt(receipt)

            except Exception as e:
                logger.warning(f"Could not register sync receipt: {e}")

            return self.json_response({
                "status": "received",
                "checksum_verified": checksum_verified,
                "stored_at": stored_path,
                "storage_tier": storage_tier,
                "disk_usage_percent": local_disk_usage,
                "node_id": self.node_id,
            })

        except Exception as e:  # noqa: BLE001
            logger.error(f"in handle_sync_push: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_sync_receipt(self, request: web.Request) -> web.Response:
        """POST /sync/receipt - Request sync receipt verification.

        December 2025: Added for push-based sync with verified cleanup.

        GPU nodes call this to verify a file exists on coordinator
        with matching checksum. Used to confirm replication before
        local cleanup.

        Request body:
        {
            "file_path": "data/games/selfplay_hex8.db",
            "checksum": "sha256hex..."
        }

        Response:
        {
            "verified": true,
            "file_path": "data/games/selfplay_hex8.db",
            "checksum": "sha256hex...",
            "node_id": "coordinator-node",
            "timestamp": 1234567890.0
        }
        """
        try:
            data = await request.json()
            file_path = data.get("file_path", "")
            expected_checksum = data.get("checksum", "")

            if not file_path or not expected_checksum:
                return self.error_response(
                    "Missing file_path or checksum",
                    status=400
                )

            # Resolve to local path
            local_path = Path(file_path)
            if not local_path.is_absolute():
                # Try common base paths
                candidates = [
                    Path(file_path),
                    Path("data") / file_path,
                    Path(".") / file_path,
                ]
                local_path = None
                for candidate in candidates:
                    if candidate.exists():
                        local_path = candidate
                        break

            if local_path is None or not local_path.exists():
                return self.json_response({
                    "verified": False,
                    "reason": "not_found",
                    "file_path": file_path,
                })

            # Compute local checksum
            hasher = hashlib.sha256()
            with open(local_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            local_checksum = hasher.hexdigest()

            if local_checksum != expected_checksum:
                return self.json_response({
                    "verified": False,
                    "reason": "checksum_mismatch",
                    "file_path": file_path,
                    "expected": expected_checksum[:16] + "...",
                    "actual": local_checksum[:16] + "...",
                })

            return self.json_response({
                "verified": True,
                "file_path": file_path,
                "checksum": local_checksum,
                "node_id": self.node_id,
                "timestamp": time.time(),
            })

        except Exception as e:  # noqa: BLE001
            logger.error(f"in handle_sync_receipt: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_sync_receipts_status(self, request: web.Request) -> web.Response:
        """GET /sync/receipts - Get sync receipts statistics.

        December 2025: Added for push-based sync monitoring.

        Returns stats about sync receipts in the cluster manifest.
        """
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()
            stats = manifest.get_sync_stats()

            return self.json_response({
                "node_id": self.node_id,
                "is_leader": self._is_leader(),
                "sync_receipts": stats,
            })

        except ImportError:
            return self.json_response({
                "node_id": self.node_id,
                "error": "ClusterManifest not available",
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"in handle_sync_receipts_status: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_sync_pull(self, request: web.Request) -> web.Response:
        """POST /sync/pull - Handle incoming request to pull files from a source node.

        This is called by the leader to tell this node to pull files from another node.

        Request body:
        {
            "source_host": "192.168.1.100",
            "source_port": 8770,
            "source_node_id": "lambda-h100",
            "files": ["data/selfplay/sq8_2p/games_001.jsonl", ...]
        }
        """
        try:
            # Import check_disk_has_capacity from constants
            try:
                from scripts.p2p.constants import DEFAULT_PORT
            except ImportError:
                DEFAULT_PORT = 8770

            # Check disk capacity before accepting sync request
            has_capacity, disk_percent = self._check_disk_has_capacity()
            if not has_capacity:
                max_disk = float(os.environ.get("RINGRIFT_MAX_DISK_USAGE_PERCENT", str(_SYNC_DISK_THRESHOLD)))
                return self.json_response({
                    "error": f"Disk full ({disk_percent:.1f}% >= {max_disk}%)",
                    "disk_percent": disk_percent,
                    "threshold": max_disk
                }, status=507)  # 507 Insufficient Storage

            data = await request.json()
            source_node_id = data.get("source_node_id")
            files = data.get("files", [])

            if not source_node_id or not files:
                return self.error_response(
                    "Missing required fields: source_node_id, files",
                    status=400
                )

            # Prefer the local peer table for reachability (avoids leader guessing our routes).
            source_host = data.get("source_host")
            source_port = int(data.get("source_port", DEFAULT_PORT) or DEFAULT_PORT)
            with self.peers_lock:
                peer = self.peers.get(source_node_id)
            if source_node_id == self.node_id:
                peer = self.self_info
            if peer:
                source_host = peer.host
                source_port = peer.port

            if not source_host:
                return self.error_response(
                    "Missing required fields: source_host (or unknown source_node_id)",
                    status=400
                )

            logger.info(f"Received sync pull request: {len(files)} files from {source_node_id}")

            result = await self._handle_sync_pull_request(
                source_host=source_host,
                source_port=source_port,
                source_reported_host=(data.get("source_reported_host") or getattr(peer, "reported_host", "") or None),
                source_reported_port=(data.get("source_reported_port") or getattr(peer, "reported_port", 0) or None),
                source_node_id=source_node_id,
                files=files,
            )

            return self.json_response(result)
        except Exception as e:  # noqa: BLE001
            logger.error(f"in handle_sync_pull: {e}")
            import traceback
            traceback.print_exc()
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_sync_file(self, request: web.Request) -> web.StreamResponse:
        """GET /sync/file?path=<relative_path> - Stream a data file to a peer.

        Security:
        - Only serves files within `ai-service/data/**`.
        - Requires auth when RINGRIFT_CLUSTER_AUTH_TOKEN is set (even though it's a GET).
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return self.json_response({"error": "unauthorized"}, status=401)

            rel_path = (request.query.get("path") or "").lstrip("/")
            if not rel_path:
                return self.error_response("Missing required query param: path", status=400)

            data_dir = self.get_data_directory()
            data_dir.mkdir(parents=True, exist_ok=True)
            data_root = data_dir.resolve()
            full_path = (data_dir / rel_path)
            try:
                resolved = full_path.resolve()
                resolved.relative_to(data_root)
            except (AttributeError):
                return self.error_response("Invalid path", status=400)

            if not resolved.exists() or not resolved.is_file():
                return self.json_response({"error": "Not found"}, status=404)

            stat = resolved.stat()
            resp = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(stat.st_size),
                },
            )
            await resp.prepare(request)
            with open(resolved, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    await resp.write(chunk)
            await resp.write_eof()
            return resp
        except Exception as e:  # noqa: BLE001
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_sync_job_update(self, request: web.Request) -> web.Response:
        """POST /sync/job_update - Worker reports sync job status back to leader.

        Request body:
        {
            "job_id": "sync-123",
            "status": "completed|failed",
            "files_completed": 10,
            "bytes_transferred": 1048576,
            "error_message": "optional error message"
        }
        """
        try:
            data = await request.json()
            job_id = data.get("job_id")
            status = data.get("status")
            files_completed = data.get("files_completed", data.get("files_synced", 0))
            bytes_transferred = data.get("bytes_transferred", 0)
            error_message = data.get("error_message", data.get("error"))

            if not job_id or not status:
                return self.error_response(
                    "Missing required fields: job_id, status",
                    status=400
                )

            with self.sync_lock:
                if job_id in self.active_sync_jobs:
                    job = self.active_sync_jobs[job_id]
                    job.status = status
                    job.files_completed = int(files_completed or 0)
                    job.bytes_transferred = int(bytes_transferred or 0)
                    job.completed_at = time.time()
                    if error_message:
                        job.error_message = str(error_message)

                    logger.info(f"Sync job {job_id} {status}: {job.files_completed} files, {job.bytes_transferred} bytes")

                    # Update sync plan status if all jobs are done
                    if self.current_sync_plan:
                        all_done = all(
                            j.status in ("completed", "failed")
                            for j in self.current_sync_plan.sync_jobs
                        )
                        if all_done:
                            completed = sum(1 for j in self.current_sync_plan.sync_jobs if j.status == "completed")
                            failed = sum(1 for j in self.current_sync_plan.sync_jobs if j.status == "failed")
                            self.current_sync_plan.status = "completed" if failed == 0 else "partial"
                            self.current_sync_plan.completed_at = time.time()
                            self.sync_in_progress = False
                            self.last_sync_time = time.time()
                            logger.info(f"Cluster sync plan completed: {completed} succeeded, {failed} failed")

            return self.json_response({
                "success": True,
                "job_id": job_id,
                "status": status,
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"in handle_sync_job_update: {e}")
            return self.error_response(str(e), status=500)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _check_disk_has_capacity(self) -> tuple[bool, float]:
        """Check if disk has capacity for sync.

        Returns:
            Tuple of (has_capacity, disk_usage_percent)
        """
        try:
            max_disk = float(os.environ.get("RINGRIFT_MAX_DISK_USAGE_PERCENT", str(_SYNC_DISK_THRESHOLD)))
            data_dir = Path("data")
            if data_dir.exists():
                usage = shutil.disk_usage(data_dir)
                disk_percent = (usage.used / usage.total) * 100
                return disk_percent < max_disk, disk_percent
            return True, 0.0
        except ValueError as e:
            # Invalid env var format for float() - log and fail open
            logger.warning(f"Invalid RINGRIFT_MAX_DISK_USAGE_PERCENT value: {e}")
            return True, 0.0
        except OSError as e:
            # disk_usage() error - log and fail open
            logger.warning(f"Could not check disk usage: {e}")
            return True, 0.0

    def _queue_rsync_pull(
        self,
        source_node: str,
        file_path: str,
        local_path: str,
        expected_checksum: str,
        file_size: int,
    ) -> None:
        """Queue an rsync pull request for a large file.

        December 29, 2025: Implements the rsync pull queue for large files that
        can't be included inline in push notifications. The queue is processed
        by the sync coordinator using bandwidth-coordinated rsync.

        Args:
            source_node: Node ID that has the file
            file_path: Path to the file on the source node
            local_path: Where to save the file locally
            expected_checksum: SHA256 checksum for verification
            file_size: Size of the file in bytes
        """
        # Lazy init the queue
        if not hasattr(self, "_pending_rsync_pulls"):
            self._pending_rsync_pulls: list[dict[str, Any]] = []
            self._pending_rsync_pulls_lock = Lock()

        pull_request = {
            "source_node": source_node,
            "file_path": file_path,
            "local_path": local_path,
            "expected_checksum": expected_checksum,
            "file_size": file_size,
            "queued_at": time.time(),
            "status": "pending",
        }

        with self._pending_rsync_pulls_lock:
            # Check for duplicates (same source + file_path)
            for existing in self._pending_rsync_pulls:
                if (existing["source_node"] == source_node and
                    existing["file_path"] == file_path):
                    logger.debug(
                        f"Skipping duplicate rsync pull: {file_path} from {source_node}"
                    )
                    return

            self._pending_rsync_pulls.append(pull_request)
            queue_size = len(self._pending_rsync_pulls)

        logger.info(
            f"Queued rsync pull: {file_path} from {source_node} "
            f"({file_size / 1024 / 1024:.1f}MB, queue size: {queue_size})"
        )

        # Try to process queue immediately if sync coordinator is available
        try:
            from app.coordination.sync_bandwidth import get_coordinated_rsync

            # Fire and forget - sync coordinator will process the queue
            # This is a notification, actual transfer happens async
            rsync = get_coordinated_rsync()
            if rsync and hasattr(rsync, "queue_pull"):
                rsync.queue_pull(pull_request)
        except ImportError:
            # sync_bandwidth not available, queue will be processed later
            pass
        except Exception as e:
            logger.debug(f"Could not notify sync coordinator: {e}")

    def get_pending_rsync_pulls(self) -> list[dict[str, Any]]:
        """Get list of pending rsync pull requests.

        Returns:
            List of pending pull requests with source, path, size info
        """
        if not hasattr(self, "_pending_rsync_pulls"):
            return []

        with self._pending_rsync_pulls_lock:
            return list(self._pending_rsync_pulls)

    def clear_completed_rsync_pulls(self) -> int:
        """Remove completed or failed rsync pulls from the queue.

        Returns:
            Number of pulls removed
        """
        if not hasattr(self, "_pending_rsync_pulls"):
            return 0

        with self._pending_rsync_pulls_lock:
            original_count = len(self._pending_rsync_pulls)
            self._pending_rsync_pulls = [
                p for p in self._pending_rsync_pulls
                if p.get("status") == "pending"
            ]
            removed = original_count - len(self._pending_rsync_pulls)

        if removed > 0:
            logger.debug(f"Cleared {removed} completed rsync pulls from queue")

        return removed
