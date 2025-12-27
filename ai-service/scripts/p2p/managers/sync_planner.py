"""SyncPlanner: Data synchronization planning and execution for P2P cluster.

Extracted from p2p_orchestrator.py for better modularity.
Handles manifest collection, sync plan generation, and data distribution.

December 2025: Phase 2A extraction as part of god-class decomposition.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..models import (
        ClusterDataManifest,
        ClusterSyncPlan,
        DataFileInfo,
        DataSyncJob,
        NodeDataManifest,
        NodeInfo,
    )

logger = logging.getLogger(__name__)

# Event emission helper - imported lazily to avoid circular imports
_emit_event: Callable[[str, dict], None] | None = None


def _get_event_emitter() -> Callable[[str, dict], None] | None:
    """Get the event emitter function, initializing if needed."""
    global _emit_event
    if _emit_event is None:
        try:
            from app.coordination.event_router import emit_sync
            _emit_event = emit_sync
        except ImportError:
            # Event system not available
            pass
    return _emit_event


# Constants (match p2p/constants.py)
MANIFEST_JSONL_LINECOUNT_MAX_BYTES = 50 * 1024 * 1024  # 50MB threshold for sampling
MANIFEST_JSONL_SAMPLE_BYTES = 256 * 1024  # 256KB sample for large files
MANIFEST_JSONL_LINECOUNT_CHUNK_BYTES = 65536  # 64KB chunks for line counting
MAX_DISK_USAGE_PERCENT = 90.0


@dataclass
class SyncPlannerConfig:
    """Configuration for the SyncPlanner."""

    manifest_cache_age_seconds: int = 300  # 5 minutes
    manifest_collection_interval: int = 60  # 1 minute
    max_files_per_sync_job: int = 50
    sync_mtime_tolerance_seconds: int = 60  # Clock skew tolerance


@dataclass
class SyncStats:
    """Statistics for sync operations."""

    manifests_collected: int = 0
    sync_plans_generated: int = 0
    sync_jobs_created: int = 0
    sync_jobs_completed: int = 0
    sync_jobs_failed: int = 0
    bytes_synced: int = 0
    last_manifest_collection: float = 0.0
    last_sync_execution: float = 0.0


class SyncPlanner:
    """Manages data synchronization planning and execution for P2P cluster.

    This class is extracted from P2POrchestrator to handle:
    - Local manifest collection (scanning data directory)
    - Cluster manifest aggregation (collecting from all peers)
    - Sync plan generation (identifying missing files)
    - Sync plan execution (dispatching sync jobs)

    Follows the manager pattern established by StateManager:
    - Uses callbacks to access orchestrator state
    - Thread-safe with explicit lock passing
    - Testable in isolation with mock callbacks

    Usage:
        # In P2POrchestrator.__init__():
        self.sync_planner = SyncPlanner(
            node_id=self.node_id,
            data_directory=self.get_data_directory(),
            get_peers=lambda: self.peers,
            get_self_info=lambda: self.self_info,
            peers_lock=self.peers_lock,
            is_leader=lambda: self._is_leader(),
            config=SyncPlannerConfig(),
        )

        # Then call:
        manifest = self.sync_planner.collect_local_manifest()
        cluster = await self.sync_planner.collect_cluster_manifest()
        plan = self.sync_planner.generate_sync_plan(cluster)
    """

    def __init__(
        self,
        node_id: str,
        data_directory: Path,
        get_peers: Callable[[], dict[str, "NodeInfo"]],
        get_self_info: Callable[[], "NodeInfo"],
        peers_lock: threading.Lock,
        is_leader: Callable[[], bool],
        request_peer_manifest: Callable[[str], "NodeDataManifest | None"] | None = None,
        check_disk_capacity: Callable[[], tuple[bool, float]] | None = None,
        config: SyncPlannerConfig | None = None,
    ):
        """Initialize the SyncPlanner.

        Args:
            node_id: This node's unique identifier
            data_directory: Path to the data directory (supports ramdrive)
            get_peers: Callback to get current peer dict
            get_self_info: Callback to get this node's NodeInfo
            peers_lock: Lock for thread-safe peer access
            is_leader: Callback to check if this node is the leader
            request_peer_manifest: Callback to request manifest from a peer (optional)
            check_disk_capacity: Callback to check disk capacity (optional)
            config: Configuration options
        """
        self.node_id = node_id
        self.data_directory = data_directory
        self._get_peers = get_peers
        self._get_self_info = get_self_info
        self._peers_lock = peers_lock
        self._is_leader = is_leader
        self._request_peer_manifest = request_peer_manifest
        self._check_disk_capacity = check_disk_capacity or (lambda: (True, 0.0))
        self.config = config or SyncPlannerConfig()

        # Cached manifest
        self._cached_local_manifest: "NodeDataManifest | None" = None
        self._cached_manifest_time: float = 0.0

        # Cluster manifest (leader only)
        self._cluster_manifest: "ClusterDataManifest | None" = None

        # Current sync plan (leader only)
        self._current_sync_plan: "ClusterSyncPlan | None" = None

        # Active sync jobs
        self._active_sync_jobs: dict[str, "DataSyncJob"] = {}
        self._sync_lock = threading.Lock()
        self._sync_in_progress = False

        # Statistics
        self.stats = SyncStats()

    def _emit_sync_event(self, event_type: str, **kwargs) -> None:
        """Emit a sync lifecycle event if the event system is available.

        Args:
            event_type: One of DATA_SYNC_STARTED, DATA_SYNC_COMPLETED, DATA_SYNC_FAILED
            **kwargs: Additional event data
        """
        emitter = _get_event_emitter()
        if emitter is None:
            return

        payload = {
            "node_id": self.node_id,
            "timestamp": time.time(),
            **kwargs,
        }

        try:
            emitter(event_type, payload)
            logger.debug(f"Emitted {event_type}")
        except Exception as e:
            logger.debug(f"Failed to emit {event_type}: {e}")

    # ============================================
    # Local Manifest Collection
    # ============================================

    def collect_local_manifest(self, use_cache: bool = True) -> "NodeDataManifest":
        """Collect manifest of all data files on this node.

        Scans the data directory for:
        - selfplay/ - Game replay files (.jsonl, .db)
        - models/ - Trained model files (.pt, .onnx)
        - training/ - Training data files (.npz)
        - games/ - Synced game databases (.db)

        Args:
            use_cache: Whether to use cached manifest if still valid

        Returns:
            NodeDataManifest with all discovered files
        """
        # Import here to avoid circular imports at module level
        from ..models import DataFileInfo, NodeDataManifest

        # Check cache if requested
        if use_cache and self._cached_local_manifest:
            cache_age = time.time() - self._cached_manifest_time
            if cache_age < self.config.manifest_cache_age_seconds:
                return self._cached_local_manifest

        manifest = NodeDataManifest(
            node_id=self.node_id,
            collected_at=time.time(),
        )

        if not self.data_directory.exists():
            logger.info(f"Data directory not found: {self.data_directory}")
            return manifest

        files: list[DataFileInfo] = []

        # Scan for data files
        patterns = {
            "selfplay": ["selfplay/**/*.jsonl", "selfplay/**/*.db"],
            "model": ["models/**/*.pt", "models/**/*.onnx", "models/**/*.bin"],
            "training": ["training/**/*.npz"],
            "games": ["games/**/*.db"],
        }

        for file_type, globs in patterns.items():
            for pattern in globs:
                for file_path in self.data_directory.glob(pattern):
                    if not file_path.is_file():
                        continue

                    try:
                        stat = file_path.stat()
                        rel_path = str(file_path.relative_to(self.data_directory))

                        # Parse board_type and num_players from filename/path
                        board_type, num_players = self._parse_board_config(rel_path)

                        file_info = DataFileInfo(
                            path=rel_path,
                            size_bytes=stat.st_size,
                            modified_time=stat.st_mtime,
                            file_type=file_type,
                            board_type=board_type,
                            num_players=num_players,
                        )

                        # Count games for selfplay files
                        if file_type == "selfplay":
                            game_count = self._count_games_in_file(file_path, stat.st_size)
                            file_info.game_count = game_count
                            manifest.selfplay_games += game_count

                        files.append(file_info)

                        # Update summary stats
                        manifest.total_files += 1
                        manifest.total_size_bytes += stat.st_size

                        if file_type == "model":
                            manifest.model_count += 1
                        elif file_type == "training":
                            manifest.training_data_size += stat.st_size

                    except (OSError, ValueError) as e:
                        logger.debug(f"Error scanning file {file_path}: {e}")

        manifest.files = files

        # Update cache
        self._cached_local_manifest = manifest
        self._cached_manifest_time = time.time()
        self.stats.manifests_collected += 1
        self.stats.last_manifest_collection = time.time()

        logger.debug(
            f"Collected local manifest: {manifest.total_files} files, "
            f"{manifest.selfplay_games} games, "
            f"{manifest.total_size_bytes / (1024*1024):.1f} MB"
        )

        return manifest

    def _parse_board_config(self, path: str) -> tuple[str, int]:
        """Parse board_type and num_players from file path.

        Args:
            path: Relative file path

        Returns:
            Tuple of (board_type, num_players)
        """
        board_type = ""
        num_players = 0
        path_lower = path.lower()

        if "sq8" in path_lower or "square8" in path_lower:
            board_type = "square8"
        elif "sq19" in path_lower or "square19" in path_lower:
            board_type = "square19"
        elif "hex" in path_lower:
            board_type = "hexagonal"

        if "_2p" in path_lower or "2p_" in path_lower:
            num_players = 2
        elif "_3p" in path_lower or "3p_" in path_lower:
            num_players = 3
        elif "_4p" in path_lower or "4p_" in path_lower:
            num_players = 4

        return board_type, num_players

    def _count_games_in_file(self, file_path: Path, file_size: int) -> int:
        """Count games in a selfplay file (JSONL or SQLite).

        For JSONL files:
        - Small files (<=50MB): Count lines directly
        - Large files: Sample first 256KB to estimate

        For SQLite files:
        - Query games table for count

        Args:
            file_path: Path to the file
            file_size: File size in bytes

        Returns:
            Number of games (or estimated count)
        """
        if file_size == 0:
            return 0

        suffix = file_path.suffix.lower()

        if suffix == ".jsonl":
            return self._count_jsonl_games(file_path, file_size)
        elif suffix == ".db":
            return self._count_sqlite_games(file_path)

        return 0

    def _count_jsonl_games(self, file_path: Path, file_size: int) -> int:
        """Count or estimate lines in a JSONL file."""
        try:
            # For large files, estimate from sample
            if file_size > MANIFEST_JSONL_LINECOUNT_MAX_BYTES:
                with open(file_path, "rb") as f:
                    sample = f.read(MANIFEST_JSONL_SAMPLE_BYTES)
                    if not sample:
                        return 0
                    sample_lines = sample.count(b"\n")
                    if sample_lines == 0:
                        return 0
                    avg_bytes_per_line = len(sample) / sample_lines
                    return int(file_size / avg_bytes_per_line)

            # For small files, count directly
            with open(file_path, "rb") as f:
                line_count = 0
                last_byte = b""
                while True:
                    chunk = f.read(MANIFEST_JSONL_LINECOUNT_CHUNK_BYTES)
                    if not chunk:
                        break
                    line_count += chunk.count(b"\n")
                    last_byte = chunk[-1:]

            if file_size > 0 and last_byte != b"\n":
                line_count += 1

            return line_count

        except (OSError, ValueError):
            return 0

    def _count_sqlite_games(self, file_path: Path) -> int:
        """Count games in a SQLite database."""
        db_conn = None
        try:
            db_conn = sqlite3.connect(str(file_path), timeout=5)
            cursor = db_conn.execute("SELECT COUNT(*) FROM games")
            return cursor.fetchone()[0]
        except (sqlite3.Error, IndexError):
            return 0
        finally:
            if db_conn:
                db_conn.close()

    # ============================================
    # Cluster Manifest Aggregation
    # ============================================

    async def collect_cluster_manifest(self) -> "ClusterDataManifest | None":
        """Collect and aggregate manifests from all peers (leader only).

        Returns:
            ClusterDataManifest or None if not leader
        """
        from ..models import ClusterDataManifest

        if not self._is_leader():
            logger.debug("Not leader, skipping cluster manifest collection")
            return None

        manifest = ClusterDataManifest(collected_at=time.time())

        # Include our own manifest
        local_manifest = self.collect_local_manifest(use_cache=False)
        manifest.node_manifests[self.node_id] = local_manifest
        manifest.total_nodes += 1

        # Collect from peers
        with self._peers_lock:
            peers_snapshot = list(self._get_peers().values())

        for peer in peers_snapshot:
            if peer.node_id == self.node_id:
                continue
            if not peer.is_alive():
                continue

            try:
                peer_manifest = await self._collect_peer_manifest(peer)
                if peer_manifest:
                    manifest.node_manifests[peer.node_id] = peer_manifest
                    manifest.total_nodes += 1
            except Exception as e:
                logger.debug(f"Failed to collect manifest from {peer.node_id}: {e}")

        # Aggregate totals
        self._aggregate_cluster_totals(manifest)

        # Store for sync planning
        self._cluster_manifest = manifest
        self.stats.last_manifest_collection = time.time()

        logger.info(
            f"Collected cluster manifest: {manifest.total_nodes} nodes, "
            f"{manifest.total_files} files, "
            f"{manifest.total_selfplay_games} games"
        )

        return manifest

    async def _collect_peer_manifest(self, peer: "NodeInfo") -> "NodeDataManifest | None":
        """Request manifest from a single peer.

        Args:
            peer: The peer NodeInfo

        Returns:
            NodeDataManifest or None if request failed
        """
        if self._request_peer_manifest:
            # Use provided callback (async)
            if asyncio.iscoroutinefunction(self._request_peer_manifest):
                return await self._request_peer_manifest(peer.node_id)
            else:
                return self._request_peer_manifest(peer.node_id)

        # Default: no peer manifest collection available
        logger.debug(f"No peer manifest callback configured, skipping {peer.node_id}")
        return None

    def _aggregate_cluster_totals(self, manifest: "ClusterDataManifest") -> None:
        """Aggregate totals and analyze data distribution.

        Args:
            manifest: The cluster manifest to aggregate
        """
        all_files: set[str] = set()
        files_by_node: dict[str, set[str]] = {}

        for node_id, node_manifest in manifest.node_manifests.items():
            manifest.total_files += node_manifest.total_files
            manifest.total_size_bytes += node_manifest.total_size_bytes
            manifest.total_selfplay_games += node_manifest.selfplay_games
            manifest.files_by_node[node_id] = node_manifest.total_files

            # Track files for distribution analysis
            node_files = {f.path for f in node_manifest.files}
            files_by_node[node_id] = node_files
            all_files.update(node_files)

        manifest.unique_files = all_files

        # Find files missing from each node
        for file_path in all_files:
            missing_from = [
                node_id
                for node_id, node_files in files_by_node.items()
                if file_path not in node_files
            ]
            if missing_from:
                manifest.missing_from_nodes[file_path] = missing_from

    # ============================================
    # Sync Plan Generation
    # ============================================

    def generate_sync_plan(
        self,
        cluster_manifest: "ClusterDataManifest | None" = None,
    ) -> "ClusterSyncPlan | None":
        """Generate a sync plan from the cluster manifest.

        Creates sync jobs to distribute missing files across nodes.

        Args:
            cluster_manifest: Optional manifest to use (uses cached if None)

        Returns:
            ClusterSyncPlan or None if no sync needed
        """
        from ..models import ClusterSyncPlan, DataSyncJob

        manifest = cluster_manifest or self._cluster_manifest
        if not manifest:
            logger.info("No cluster manifest available, cannot generate sync plan")
            return None

        if not manifest.missing_from_nodes:
            logger.debug("All nodes have all files, no sync needed")
            return None

        plan = ClusterSyncPlan(
            plan_id=str(uuid.uuid4()),
            created_at=time.time(),
        )

        # For each missing file, find a source node and create sync job
        for file_path, missing_nodes in manifest.missing_from_nodes.items():
            source_node = self._find_source_for_file(file_path, manifest, missing_nodes)
            if not source_node:
                continue

            # Create sync jobs for each target node
            for target_node in missing_nodes:
                job = DataSyncJob(
                    job_id=str(uuid.uuid4()),
                    source_node=source_node,
                    target_node=target_node,
                    files=[file_path],
                    status="pending",
                )

                # Track file size
                node_manifest = manifest.node_manifests.get(source_node)
                if node_manifest:
                    file_info = node_manifest.files_by_path.get(file_path)
                    if file_info:
                        plan.total_bytes_to_sync += file_info.size_bytes

                plan.sync_jobs.append(job)
                plan.total_files_to_sync += 1

        self._current_sync_plan = plan
        self.stats.sync_plans_generated += 1

        logger.info(
            f"Generated sync plan: {len(plan.sync_jobs)} jobs, "
            f"{plan.total_files_to_sync} files, "
            f"{plan.total_bytes_to_sync / (1024*1024):.1f} MB total"
        )

        return plan

    def _find_source_for_file(
        self,
        file_path: str,
        manifest: "ClusterDataManifest",
        missing_nodes: list[str],
    ) -> str | None:
        """Find a source node that has the specified file.

        Args:
            file_path: The file to find
            manifest: The cluster manifest
            missing_nodes: Nodes that don't have the file

        Returns:
            Source node_id or None
        """
        for node_id, node_manifest in manifest.node_manifests.items():
            if node_id not in missing_nodes:
                if file_path in node_manifest.files_by_path:
                    return node_id
        return None

    # ============================================
    # Sync Plan Execution
    # ============================================

    async def execute_sync_plan(
        self,
        plan: "ClusterSyncPlan | None" = None,
        execute_job_callback: Callable[["DataSyncJob"], bool] | None = None,
    ) -> dict[str, Any]:
        """Execute the sync plan by dispatching jobs.

        Args:
            plan: Optional plan to execute (uses current if None)
            execute_job_callback: Callback to execute a single sync job

        Returns:
            Dict with execution results
        """
        plan = plan or self._current_sync_plan
        if not plan:
            return {"success": False, "error": "No sync plan available"}

        # Check disk capacity
        has_capacity, disk_percent = self._check_disk_capacity()
        if not has_capacity:
            logger.info(
                f"SKIPPING SYNC - Disk usage {disk_percent:.1f}% exceeds limit {MAX_DISK_USAGE_PERCENT}%"
            )
            return {"success": False, "error": "Disk capacity exceeded"}

        with self._sync_lock:
            if self._sync_in_progress:
                logger.info("Sync already in progress, skipping")
                return {"success": False, "error": "Sync already in progress"}
            self._sync_in_progress = True
            plan.status = "running"

        results = {
            "success": True,
            "jobs_total": len(plan.sync_jobs),
            "jobs_completed": 0,
            "jobs_failed": 0,
            "bytes_synced": 0,
        }

        # Emit sync started event
        self._emit_sync_event(
            "DATA_SYNC_STARTED",
            plan_id=plan.plan_id if hasattr(plan, "plan_id") else str(uuid.uuid4()),
            jobs_total=len(plan.sync_jobs),
        )

        try:
            # Group jobs by target node for efficiency
            jobs_by_target: dict[str, list["DataSyncJob"]] = {}
            for job in plan.sync_jobs:
                if job.target_node not in jobs_by_target:
                    jobs_by_target[job.target_node] = []
                jobs_by_target[job.target_node].append(job)

            # Execute jobs
            for target_node, jobs in jobs_by_target.items():
                for job in jobs:
                    try:
                        if execute_job_callback:
                            success = execute_job_callback(job)
                        else:
                            # Default: mark as pending for external execution
                            self._active_sync_jobs[job.job_id] = job
                            success = True  # Will be executed externally

                        if success:
                            job.status = "completed"
                            job.completed_at = time.time()
                            results["jobs_completed"] += 1
                            self.stats.sync_jobs_completed += 1
                        else:
                            job.status = "failed"
                            results["jobs_failed"] += 1
                            self.stats.sync_jobs_failed += 1

                    except Exception as e:
                        job.status = "failed"
                        job.error_message = str(e)
                        results["jobs_failed"] += 1
                        self.stats.sync_jobs_failed += 1
                        logger.error(f"Sync job {job.job_id} failed: {e}")

            plan.status = "completed" if results["jobs_failed"] == 0 else "partial"
            plan.jobs_completed = results["jobs_completed"]
            plan.jobs_failed = results["jobs_failed"]
            self.stats.last_sync_execution = time.time()

            # Emit sync completion event
            if results["jobs_failed"] == 0:
                self._emit_sync_event(
                    "DATA_SYNC_COMPLETED",
                    jobs_completed=results["jobs_completed"],
                    bytes_synced=results["bytes_synced"],
                )
            else:
                self._emit_sync_event(
                    "DATA_SYNC_FAILED",
                    jobs_completed=results["jobs_completed"],
                    jobs_failed=results["jobs_failed"],
                    error="Some sync jobs failed",
                )

        finally:
            with self._sync_lock:
                self._sync_in_progress = False

        return results

    # ============================================
    # Utility Methods
    # ============================================

    def get_cached_manifest(self) -> "NodeDataManifest | None":
        """Get the cached local manifest if still valid."""
        if not self._cached_local_manifest:
            return None

        cache_age = time.time() - self._cached_manifest_time
        if cache_age > self.config.manifest_cache_age_seconds:
            return None

        return self._cached_local_manifest

    def get_cluster_manifest(self) -> "ClusterDataManifest | None":
        """Get the current cluster manifest (leader only)."""
        return self._cluster_manifest

    def get_current_sync_plan(self) -> "ClusterSyncPlan | None":
        """Get the current sync plan."""
        return self._current_sync_plan

    def get_active_sync_jobs(self) -> dict[str, "DataSyncJob"]:
        """Get all active sync jobs."""
        return dict(self._active_sync_jobs)

    def clear_sync_jobs(self) -> None:
        """Clear completed and failed sync jobs."""
        with self._sync_lock:
            self._active_sync_jobs = {
                job_id: job
                for job_id, job in self._active_sync_jobs.items()
                if job.status == "pending"
            }

    def get_stats(self) -> dict[str, Any]:
        """Get sync planner statistics."""
        return {
            "manifests_collected": self.stats.manifests_collected,
            "sync_plans_generated": self.stats.sync_plans_generated,
            "sync_jobs_created": self.stats.sync_jobs_created,
            "sync_jobs_completed": self.stats.sync_jobs_completed,
            "sync_jobs_failed": self.stats.sync_jobs_failed,
            "bytes_synced": self.stats.bytes_synced,
            "last_manifest_collection": self.stats.last_manifest_collection,
            "last_sync_execution": self.stats.last_sync_execution,
            "cached_manifest_age": time.time() - self._cached_manifest_time if self._cached_manifest_time else None,
            "cluster_manifest_nodes": len(self._cluster_manifest.node_manifests) if self._cluster_manifest else 0,
            "active_sync_jobs": len(self._active_sync_jobs),
            "sync_in_progress": self._sync_in_progress,
        }
