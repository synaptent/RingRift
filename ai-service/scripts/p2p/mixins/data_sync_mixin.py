"""Data Sync Mixin - cluster manifest, sync, and dedup helpers.

April 2026: Extracted from p2p_orchestrator.py (Phase 4 task 15).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any

import aiohttp

from scripts.p2p.constants import (
    DISK_CLEANUP_THRESHOLD,
    DISK_CRITICAL_THRESHOLD,
    DISK_WARNING_THRESHOLD,
)
from scripts.p2p.models import (
    ClusterDataManifest,
    DataSyncJob,
    ExternalStorageManifest,
    NodeDataManifest,
    NodeInfo,
)
from scripts.p2p.network import ClientTimeout, get_client_session
from scripts.p2p.p2p_mixin_base import P2PMixinBase

logger = logging.getLogger(__name__)


class DataSyncMixin(P2PMixinBase):
    """Mixin extracted from P2POrchestrator."""

    MIXIN_TYPE = "data_sync"

    sync_planner: Any
    data_sync_coordinator: Any
    sync: Any
    node_selector: Any
    current_sync_plan: Any
    sync_lock: Any
    last_sync_time: float
    leadership: Any
    cluster_data_manifest: Any
    last_manifest_collection: float
    manifest_collection_interval: float
    last_training_sync_time: float
    self_info: Any
    node_id: str
    peers: dict[str, Any]
    _peer_snapshot: Any

    def _collect_local_data_manifest(self) -> NodeDataManifest:
        """Collect manifest of all data files on this node.

        REFACTORED (Dec 2025): Delegates to SyncPlanner.collect_local_manifest().
        See scripts/p2p/managers/sync_planner.py for implementation.

        Scans the data directory for:
        - selfplay/ - Game replay files (.jsonl, .db)
        - models/ - Trained model files (.pt, .onnx)
        - training/ - Training data files (.npz)
        - games/ - Synced game databases (.db)

        Uses get_data_directory() to support both disk and ramdrive storage.
        """
        # Phase 2A: Delegate to SyncPlanner (Dec 2025)
        # This eliminates ~150 lines of duplicate code
        # Jan 23, 2026: Changed use_cache=False to True to reduce event loop blocking
        # The uncached version does heavy filesystem I/O (glob, stat, SQLite COUNT)
        # which can take 5-8 seconds and block the event loop, causing leader election failures
        return self.sync_planner.collect_local_manifest(use_cache=True)

    def _request_peer_manifest_sync(self, peer_id: str) -> NodeDataManifest | None:
        """Synchronous wrapper for requesting peer manifest.

        Used by SyncPlanner which expects a sync callback.
        Runs the async version in a new event loop.

        Args:
            peer_id: The peer's node ID to request from

        Returns:
            NodeDataManifest or None if request failed
        """
        # Look up peer info
        # Jan 2026: Use lock-free PeerSnapshot for read-only access
        peer_info = self._peer_snapshot.get_snapshot().get(peer_id)

        if not peer_info:
            logger.debug(f"Peer {peer_id} not found in peers dict")
            return None

        # Run async version in event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self._request_peer_manifest(peer_info), loop
            )
            return future.result(timeout=15)
        except RuntimeError:
            # No running loop - use asyncio.run
            try:
                return asyncio.run(self._request_peer_manifest(peer_info))
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to request manifest from {peer_id}: {e}")
                return None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to request manifest from {peer_id}: {e}")
            return None

    async def _request_peer_manifest(self, peer_info: NodeInfo) -> NodeDataManifest | None:
        """Request data manifest from a peer node."""
        try:
            # Keep manifest requests snappy: these are advisory and should not
            # stall leader loops or external callers (e.g. the improvement
            # daemon). Prefer faster failure and rely on periodic retries.
            timeout = ClientTimeout(total=10, sock_connect=3, sock_read=7)
            async with get_client_session(timeout) as session:
                for url in self._urls_for_peer(peer_info, "/data_manifest"):
                    try:
                        async with session.get(url, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                continue
                            data = await resp.json()
                        return NodeDataManifest.from_dict((data or {}).get("manifest", {}))
                    except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                        continue
        except Exception as e:  # noqa: BLE001
            logger.error(f"requesting manifest from {peer_info.node_id}: {e}")
        return None

    async def _collect_cluster_manifest(self) -> ClusterDataManifest:
        """Jan 29, 2026: Delegated to SyncOrchestrator.collect_cluster_manifest()."""
        return await self.sync.collect_cluster_manifest()

    async def _collect_external_storage_metadata(self) -> ExternalStorageManifest:
        """Collect metadata from external storage sources (OWC drive, S3 bucket).

        Jan 2026: Delegates to DataSyncCoordinator for unified cluster data visibility.

        Returns:
            ExternalStorageManifest with OWC and S3 metadata.
        """
        from scripts.p2p.models import ExternalStorageManifest

        # Delegate to DataSyncCoordinator
        metadata = await self.data_sync_coordinator.collect_external_storage_metadata()

        # Convert to ExternalStorageManifest
        external = ExternalStorageManifest(collected_at=metadata.collected_at)
        external.owc_available = metadata.owc_available
        external.owc_games_by_config = metadata.owc_games_by_config
        external.owc_total_games = metadata.owc_total_games
        external.owc_total_size_bytes = metadata.owc_total_size_bytes
        external.owc_last_scan = metadata.owc_last_scan
        external.owc_scan_error = metadata.owc_scan_error or ""
        external.s3_available = metadata.s3_available
        external.s3_games_by_config = metadata.s3_games_by_config
        external.s3_total_games = metadata.s3_total_games
        external.s3_total_size_bytes = metadata.s3_total_size_bytes
        external.s3_last_scan = metadata.s3_last_scan
        external.s3_bucket = metadata.s3_bucket
        external.s3_scan_error = metadata.s3_scan_error or ""

        return external

    def _extract_config_from_path(self, db_path: Path) -> str | None:
        """Extract config from path. Delegates to DataSyncCoordinator."""
        return self.data_sync_coordinator.extract_config_from_path(db_path)

    async def _execute_sync_plan(self) -> None:
        """Leader executes the sync plan by dispatching jobs to nodes.

        Delegates to SyncPlanner.execute_sync_plan() with _request_node_sync as callback.
        Dec 2025: Refactored to delegate to SyncPlanner for consolidated logic.
        """
        if not self.current_sync_plan:
            return

        # Delegate to SyncPlanner with our network request callback
        result = await self.sync_planner.execute_sync_plan(
            plan=self.current_sync_plan,
            execute_job_callback_async=self._request_node_sync,
        )

        # Update local state from SyncPlanner result
        with self.sync_lock:
            self.last_sync_time = time.time()

        if not result.get("success", False):
            logger.warning(f"Sync plan execution issue: {result.get('error', 'unknown')}")

    async def _request_node_sync(self, job: DataSyncJob) -> bool:
        """Request a target node to pull files from a source node."""
        target_peer = self.peers.get(job.target_node)
        if job.target_node == self.node_id:
            target_peer = self.self_info

        source_peer = self.peers.get(job.source_node)
        if job.source_node == self.node_id:
            source_peer = self.self_info

        if not target_peer or not source_peer:
            job.status = "failed"
            job.error_message = "Source or target peer not found"
            return False

        job.status = "running"
        job.started_at = time.time()

        try:
            # Local target: execute the pull directly (no HTTP round-trip).
            if job.target_node == self.node_id:
                result = await self._handle_sync_pull_request(
                    source_host=source_peer.host,
                    source_port=source_peer.port,
                    source_reported_host=(getattr(source_peer, "reported_host", "") or None),
                    source_reported_port=(getattr(source_peer, "reported_port", 0) or None),
                    source_node_id=job.source_node,
                    files=job.files,
                )
            else:
                payload = {
                    "job_id": job.job_id,
                    # Back-compat: target will prefer source_node_id lookup.
                    "source_host": source_peer.host,
                    "source_port": source_peer.port,
                    "source_node_id": job.source_node,
                    "files": job.files,
                }
                rh = (getattr(source_peer, "reported_host", "") or "").strip()
                rp = int(getattr(source_peer, "reported_port", 0) or 0)
                if rh and rp and (rh != source_peer.host or rp != source_peer.port):
                    payload["source_reported_host"] = rh
                    payload["source_reported_port"] = rp

                timeout = ClientTimeout(total=600)
                async with get_client_session(timeout) as session:
                    result = None
                    last_err: str | None = None
                    for url in self._urls_for_peer(target_peer, "/sync/pull"):
                        try:
                            async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                                if resp.status != 200:
                                    last_err = f"http_{resp.status}"
                                    continue
                                result = await resp.json()
                                break
                        except Exception as e:  # noqa: BLE001
                            last_err = str(e)
                            continue
                    if result is None:
                        job.status = "failed"
                        job.error_message = last_err or "sync_pull_failed"
                        # Note: SyncPlanner tracks jobs_failed count
                        return False

            ok = bool(result.get("success"))
            job.status = "completed" if ok else "failed"
            job.completed_at = time.time()
            job.bytes_transferred = int(result.get("bytes_transferred", 0) or 0)
            job.files_completed = int(result.get("files_completed", 0) or 0)
            if not ok:
                job.error_message = str(result.get("error") or "Unknown error")

            # Note: SyncPlanner tracks jobs_completed/jobs_failed counts

            if ok:
                logger.info(f"Sync job {job.job_id[:8]} completed: {job.source_node} -> {job.target_node}")
            else:
                logger.info(f"Sync job {job.job_id[:8]} failed: {job.error_message}")

            return ok

        except Exception as e:  # noqa: BLE001
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = time.time()
            # Note: SyncPlanner tracks jobs_failed count
            logger.info(f"Sync job {job.job_id[:8]} failed: {e}")
            return False

    async def _handle_sync_pull_request(
        self,
        source_host: str,
        source_port: int,
        source_node_id: str,
        files: list[str],
        source_reported_host: str | None = None,
        source_reported_port: int | None = None,
    ) -> dict[str, Any]:
        """Handle incoming request to pull files from a source node.

        Jan 28, 2026: Phase 18A - Delegates to SyncPlanner.
        """
        return await self.sync_planner.handle_sync_pull_request(
            source_host=source_host,
            source_port=source_port,
            source_node_id=source_node_id,
            files=files,
            source_reported_host=source_reported_host,
            source_reported_port=source_reported_port,
            data_dir=self.get_data_directory(),
            auth_headers_fn=self._auth_headers,
        )

    async def start_cluster_sync(self) -> dict[str, Any]:
        """
        Leader initiates a full cluster data sync.
        Returns status of the sync operation.
        """
        if not self.leadership.check_is_leader():
            return {"success": False, "error": "Not the leader"}

        # First, collect fresh manifests
        logger.info("Collecting cluster manifest for sync...")
        self.cluster_data_manifest = await self._collect_cluster_manifest()

        # Generate sync plan (using SyncPlanner manager for consolidated logic)
        self.current_sync_plan = self.sync_planner.generate_sync_plan(self.cluster_data_manifest)
        if not self.current_sync_plan:
            return {"success": True, "message": "No sync needed, all nodes in sync"}

        # Execute the plan
        await self._execute_sync_plan()

        return {
            "success": True,
            "plan_id": self.current_sync_plan.plan_id,
            "total_jobs": len(self.current_sync_plan.sync_jobs),
            "jobs_completed": self.current_sync_plan.jobs_completed,
            "jobs_failed": self.current_sync_plan.jobs_failed,
            "status": self.current_sync_plan.status,
        }

    def _should_sync_to_node(self, node: NodeInfo) -> bool:
        """Check if we should sync data TO this node based on disk space."""
        # Don't sync to nodes with critical disk usage
        if node.disk_percent >= DISK_CRITICAL_THRESHOLD:
            logger.info(f"Skipping sync to {node.node_id}: disk critical ({node.disk_percent:.1f}%)")
            return False
        # Warn but allow sync to nodes with warning-level disk
        if node.disk_percent >= DISK_WARNING_THRESHOLD:
            logger.warning(f"{node.node_id} disk at {node.disk_percent:.1f}%")
        return True

    def _get_training_nodes_plus_coordinator(self) -> list:
        """Get training nodes PLUS coordinator for selfplay data sync.

        Feb 2026: The coordinator needs SOME selfplay data for canonical DB
        consolidation and NPZ export, but it's a disk-constrained machine.
        Only sync to coordinator when disk is below 70% to avoid filling it up.
        GPU nodes use local JSONL→NPZ fallback (training_executor.py) when
        coordinator doesn't have data.
        """
        training_nodes = self.node_selector.get_training_primary_nodes()
        training_ids = {n.node_id for n in training_nodes}

        # Add coordinator as sync target only when disk is healthy.
        # The coordinator is disk-constrained and shouldn't be a bulk data sink.
        # Use DISK_WARNING_THRESHOLD (70%) not 90% to preserve disk space.
        if self.self_info and self.self_info.node_id not in training_ids:
            disk_pct = getattr(self.self_info, "disk_percent", 0)
            if disk_pct < DISK_WARNING_THRESHOLD:
                training_nodes.append(self.self_info)
                logger.debug(
                    f"Including coordinator {self.self_info.node_id} as selfplay "
                    f"sync target (disk={disk_pct:.0f}%)"
                )
            else:
                logger.info(
                    f"Coordinator {self.self_info.node_id} excluded from selfplay "
                    f"sync (disk={disk_pct:.0f}% >= {DISK_WARNING_THRESHOLD}%)"
                )

        return training_nodes

    async def _sync_selfplay_to_training_nodes(self) -> dict[str, Any]:
        """Sync selfplay data to training primary nodes AND coordinator.

        December 2025: Delegated to SyncPlanner.sync_selfplay_to_training_nodes()
        Feb 2026: Added coordinator as sync target for canonical DB consolidation.
        """
        if not self.leadership.check_is_leader():
            return {"success": False, "error": "Not the leader"}

        # Use stale manifest if available, otherwise will be collected fresh
        manifest = self.cluster_data_manifest
        if (time.time() - self.last_manifest_collection > self.manifest_collection_interval
                or not manifest):
            manifest = None  # Will be collected by SyncPlanner

        result = await self.sync_planner.sync_selfplay_to_training_nodes(
            get_training_nodes=self._get_training_nodes_plus_coordinator,
            should_sync_to_node=self._should_sync_to_node,
            should_cleanup_source=lambda node: node.disk_percent >= DISK_CLEANUP_THRESHOLD,
            collect_manifest=lambda: self.sync.collect_cluster_manifest(skip_external_storage=True),
            execute_sync_job=self._request_node_sync,
            cleanup_synced_files=self.sync.cleanup_synced_files,
            get_sync_router=self._get_sync_router,
            cluster_manifest=manifest,
        )

        # Update orchestrator state
        if result.get("success"):
            self.last_training_sync_time = time.time()
            # Refresh manifest after sync
            if not manifest:
                # Dec 2025: Add 5-minute timeout for manifest collection
                try:
                    self.cluster_data_manifest = await asyncio.wait_for(
                        self._collect_cluster_manifest(),
                        timeout=300.0  # 5 minutes max
                    )
                    self.last_manifest_collection = time.time()
                except asyncio.TimeoutError:
                    logger.warning("Post-sync manifest collection timed out after 5 minutes")

        return result

    def _init_data_deduplication(self):
        """Initialize data deduplication tracking."""
        self._synced_file_hashes: set[str] = set()  # Hash -> synced
        self._known_game_ids: set[str] = set()  # Game IDs we have
        self._dedup_stats = {
            "files_skipped": 0,
            "games_skipped": 0,
            "bytes_saved": 0,
            "last_cleanup": time.time(),
        }
        self._dedup_lock = threading.Lock()

    def _record_synced_file(self, file_hash: str, file_size: int):
        """Record a file as synced for deduplication.

        DATA DEDUPLICATION: Track file hashes we've synced to avoid
        re-syncing the same file from different peers.

        Args:
            file_hash: Hash of the synced file
            file_size: Size in bytes (for metrics)
        """
        if not hasattr(self, "_synced_file_hashes"):
            self._init_data_deduplication()

        with self._dedup_lock:
            self._synced_file_hashes.add(file_hash)

    def _is_file_already_synced(self, file_hash: str) -> bool:
        """Check if file was already synced based on hash.

        Args:
            file_hash: Hash to check

        Returns:
            True if file was already synced
        """
        if not hasattr(self, "_synced_file_hashes"):
            self._init_data_deduplication()

        if not file_hash:
            return False

        with self._dedup_lock:
            return file_hash in self._synced_file_hashes

    def _record_dedup_skip(self, file_count: int = 0, game_count: int = 0, bytes_saved: int = 0):
        """Record deduplication skip for metrics.

        Args:
            file_count: Number of files skipped
            game_count: Number of games skipped
            bytes_saved: Bytes saved by skipping
        """
        if not hasattr(self, "_dedup_stats"):
            self._init_data_deduplication()

        with self._dedup_lock:
            self._dedup_stats["files_skipped"] += file_count
            self._dedup_stats["games_skipped"] += game_count
            self._dedup_stats["bytes_saved"] += bytes_saved

    def _get_dedup_summary(self) -> dict:
        """Get deduplication metrics summary."""
        if not hasattr(self, "_dedup_stats"):
            self._init_data_deduplication()

        with self._dedup_lock:
            return {
                "files_skipped": self._dedup_stats.get("files_skipped", 0),
                "games_skipped": self._dedup_stats.get("games_skipped", 0),
                "bytes_saved_mb": round(self._dedup_stats.get("bytes_saved", 0) / (1024 * 1024), 2),
                "known_file_hashes": len(self._synced_file_hashes),
                "known_game_ids": len(self._known_game_ids),
            }

    def _get_data_summary_cached(self) -> dict[str, Any]:
        """Get cached data summary for /status endpoint.

        January 13, 2026: Added as part of unified data discovery infrastructure.
        Returns game counts from local canonical databases for quick access.
        For full multi-source data, use /data/summary endpoint.

        January 23, 2026: FIXED - Removed blocking SQLite fallback that was
        causing event loop blocks. Now only returns cached data or empty dict.
        The fallback to canonical DB scan should be done via async methods.

        Returns:
            Dict with total game counts per config from local canonical DBs
        """
        try:
            # Use cached game counts from selfplay scheduler if available
            if hasattr(self, "selfplay_scheduler") and self.selfplay_scheduler:
                counts = getattr(self.selfplay_scheduler, "_p2p_game_counts", None)
                if counts:
                    total = sum(counts.values())
                    return {
                        "total_games": total,
                        "by_config": dict(counts),
                        "source": "selfplay_scheduler_cache",
                        "config_count": len(counts),
                    }

            # FIXED Jan 23, 2026: Do NOT fall back to blocking SQLite scan here.
            # The sync method _seed_selfplay_scheduler_game_counts_sync() was
            # blocking the event loop for seconds. Return empty dict instead.
            # Game counts will be populated async via selfplay_scheduler.
            return {
                "total_games": 0,
                "by_config": {},
                "source": "none",
                "error": "No cached data - scheduler not initialized yet",
            }

        except Exception as e:  # noqa: BLE001
            return {
                "total_games": 0,
                "by_config": {},
                "source": "error",
                "error": str(e),
            }
