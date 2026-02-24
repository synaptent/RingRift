"""SyncPlanner: Data synchronization planning and execution for P2P cluster.

Extracted from p2p_orchestrator.py for better modularity.
Handles manifest collection, sync plan generation, and data distribution.

December 2025: Phase 2A extraction as part of god-class decomposition.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from app.config.coordination_defaults import PeerDefaults, SQLiteDefaults
from scripts.p2p.db_helpers import p2p_db_connection
from scripts.p2p.p2p_mixin_base import EventSubscriptionMixin

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
# Dec 2025: Added thread-safe initialization to prevent race conditions
_publish_sync: Callable[[str, dict], None] | None = None
_event_emitter_lock = threading.Lock()

# Dec 2025: Import DataEventType for type-safe event emission
try:
    from app.distributed.data_events import DataEventType
except ImportError:
    DataEventType = None  # type: ignore[misc, assignment]

# Jan 2026: Import centralized timeouts from loops
try:
    from scripts.p2p.loops.loop_constants import LoopTimeouts
except ImportError:
    LoopTimeouts = None  # type: ignore[misc, assignment]


def _get_event_emitter() -> Callable[[str, dict], None] | None:
    """Get the event emitter function, initializing if needed (thread-safe).

    Uses publish_sync from event_router for synchronous event publication.
    This enables pipeline coordination to react to sync events.
    """
    global _publish_sync
    # Fast path
    if _publish_sync is not None:
        return _publish_sync

    # Slow path with lock
    with _event_emitter_lock:
        if _publish_sync is None:
            try:
                from app.coordination.event_router import publish_sync
                _publish_sync = publish_sync
            except ImportError:
                # Event system not available - running without coordination
                logger.debug("Event router not available, sync events will not be emitted")
    return _publish_sync


# Required event types for sync operations (used for validation)
REQUIRED_SYNC_EVENT_TYPES = [
    "DATA_SYNC_STARTED",
    "DATA_SYNC_COMPLETED",
    "DATA_SYNC_FAILED",
]


def _validate_event_types() -> bool:
    """Validate all required event types exist at startup.

    Dec 2025: Ensures sync events won't fail silently due to missing enum values.

    Returns:
        True if all required event types exist, False otherwise
    """
    if DataEventType is None:
        logger.debug("[SyncPlanner] DataEventType not available, skipping validation")
        return True  # Not a failure if module unavailable

    missing = []
    for event in REQUIRED_SYNC_EVENT_TYPES:
        if not hasattr(DataEventType, event):
            missing.append(event)
    if missing:
        logger.error(f"[SyncPlanner] Missing required event types: {missing}")
        return False
    logger.debug("[SyncPlanner] All required event types validated")
    return True


# Constants (match p2p/constants.py)
MANIFEST_JSONL_LINECOUNT_MAX_BYTES = 50 * 1024 * 1024  # 50MB threshold for sampling
MANIFEST_JSONL_SAMPLE_BYTES = 256 * 1024  # 256KB sample for large files
MANIFEST_JSONL_LINECOUNT_CHUNK_BYTES = 65536  # 64KB chunks for line counting
try:
    from app.config.thresholds import DISK_CRITICAL_PERCENT
    MAX_DISK_USAGE_PERCENT = float(DISK_CRITICAL_PERCENT)
except ImportError:
    MAX_DISK_USAGE_PERCENT = 90.0


@dataclass
class SyncPlannerConfig:
    """Configuration for the SyncPlanner.

    December 28, 2025: Now uses centralized PeerDefaults for timeout values.
    January 2026: Added max_concurrent_syncs to prevent memory pressure from rsync.
    """

    # Jan 2, 2026 (Sprint 8): Use centralized PeerDefaults.MANIFEST_TIMEOUT (60s = 1 minute)
    # Reduced from 300s to pick up new game data faster
    manifest_cache_age_seconds: int = int(PeerDefaults.MANIFEST_TIMEOUT)
    # Dec 28, 2025: Use centralized PeerDefaults.BOOTSTRAP_INTERVAL (60s = 1 minute)
    manifest_collection_interval: int = int(PeerDefaults.BOOTSTRAP_INTERVAL)
    max_files_per_sync_job: int = 50
    # Dec 28, 2025: Use centralized PeerDefaults.SUSPECT_TIMEOUT (30s)
    # Doubled to 60s for clock skew tolerance
    sync_mtime_tolerance_seconds: int = int(PeerDefaults.SUSPECT_TIMEOUT * 2)
    # Feb 2026: Limited to 1 to prevent OOM from parallel rsyncs
    # Each rsync can consume 7-10% RAM for large DBs; sequential keeps usage minimal
    max_concurrent_syncs: int = int(os.environ.get("RINGRIFT_MAX_CONCURRENT_SYNCS", "1"))
    # Jan 2026: Stagger large DB syncs by this many seconds
    large_db_sync_stagger_seconds: float = float(os.environ.get("RINGRIFT_LARGE_DB_SYNC_STAGGER", "60"))


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
    # Dec 2025: Track event emission for observability
    events_emitted: int = 0
    events_failed: int = 0
    last_event_error: str = ""


class SyncPlanner(EventSubscriptionMixin):
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

    Inherits from EventSubscriptionMixin for standardized event handling (Dec 2025).

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

    MIXIN_TYPE = "sync_planner"

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
        # Jan 2, 2026 (Sprint 9): Added lock to prevent race condition in cache check/write
        self._manifest_lock = threading.Lock()
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

        # Jan 2026: Semaphore for limiting concurrent sync operations (memory-aware)
        # Default of 2 keeps rsync RAM usage around 15-20% (each can use 7-10% for large DBs)
        self._sync_semaphore = asyncio.Semaphore(self.config.max_concurrent_syncs)
        # Track last large DB sync time for staggering
        self._last_large_sync_time: float = 0.0

        # Dec 2025: Validate event types at startup to catch config issues early
        _validate_event_types()

    def _emit_sync_event(self, event_type: "DataEventType", **kwargs) -> bool:
        """Emit a sync lifecycle event if the event system is available.

        Dec 2025: Now accepts DataEventType enum directly for type safety.
        Uses .value to get the actual event string (e.g., "sync_completed").

        Dec 2025 (P0-1 fix): Returns bool for caller to check success/failure.

        Args:
            event_type: DataEventType enum member (e.g., DataEventType.DATA_SYNC_COMPLETED)
            **kwargs: Additional event data

        Returns:
            True if event was emitted successfully, False otherwise.
        """
        emitter = _get_event_emitter()
        if emitter is None:
            return False

        # DataEventType not available - skip event emission
        if DataEventType is None:
            return False

        payload = {
            "node_id": self.node_id,
            "timestamp": time.time(),
            **kwargs,
        }

        # Dec 2025: Use .value to get actual event type string from enum
        # e.g., DataEventType.DATA_SYNC_COMPLETED.value == "sync_completed"
        actual_event_type = event_type.value

        # Dec 2025: Track event emission with retry for transient errors
        max_retries = 2
        for attempt in range(max_retries):
            try:
                emitter(actual_event_type, payload)
                self.stats.events_emitted += 1
                logger.debug(f"Emitted {actual_event_type} (from {event_type.name})")
                return True  # Success
            except (OSError, ConnectionError, TimeoutError) as e:
                # Transient errors - retry once (no sleep to avoid blocking event loop)
                if attempt < max_retries - 1:
                    logger.debug(f"Retry emit {actual_event_type}: {e}")
                    continue
                # Final failure
                self.stats.events_failed += 1
                self.stats.last_event_error = f"{actual_event_type}: {e}"
                logger.warning(f"[SyncPlanner] Failed to emit {actual_event_type} after {max_retries} attempts: {e}")
                return False
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                # Dec 2025: Narrowed from broad Exception - non-transient errors
                # RuntimeError: Event bus state errors
                # ValueError/TypeError: Invalid event data
                # AttributeError: Missing event attributes
                self.stats.events_failed += 1
                self.stats.last_event_error = f"{actual_event_type}: {e}"
                logger.warning(f"[SyncPlanner] Failed to emit {actual_event_type}: {e}")
                return False

        return False  # Should not reach here, but be explicit

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
        # Jan 2, 2026 (Sprint 9): Use lock to prevent race condition where multiple
        # threads read stale cache, both scan, and then race to update cache
        with self._manifest_lock:
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
        # Jan 2, 2026 (Sprint 9): Use lock to ensure atomic cache update
        with self._manifest_lock:
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
        try:
            with p2p_db_connection(file_path, timeout=SQLiteDefaults.READ_TIMEOUT) as db_conn:
                cursor = db_conn.execute("SELECT COUNT(*) FROM games")
                return cursor.fetchone()[0]
        except (sqlite3.Error, IndexError, TimeoutError):
            return 0

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
        # Jan 23, 2026: Changed use_cache=False to True to prevent event loop blocking
        # The uncached version can take 5-8 seconds and cause leader election failures
        # Also wrap in asyncio.to_thread() since this is an async method
        local_manifest = await asyncio.to_thread(
            self.collect_local_manifest, use_cache=True
        )
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
            except (OSError, ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
                # Dec 2025: Narrowed from broad Exception - network/peer communication errors
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
        execute_job_callback_async: Callable[["DataSyncJob"], Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the sync plan by dispatching jobs.

        Args:
            plan: Optional plan to execute (uses current if None)
            execute_job_callback: Sync callback to execute a single sync job
            execute_job_callback_async: Async callback (preferred for network calls)

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
            DataEventType.DATA_SYNC_STARTED,
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

            # Execute jobs with concurrency limiting (Jan 2026)
            async def execute_job_throttled(job: "DataSyncJob") -> bool:
                """Execute a sync job with semaphore-based throttling."""
                async with self._sync_semaphore:
                    # Stagger large DB syncs for memory protection
                    if hasattr(job, 'files') and job.files:
                        is_large = any("canonical" in f or "hexagonal" in f or "square19" in f for f in job.files)
                        if is_large:
                            since_last = time.time() - self._last_large_sync_time
                            if since_last < self.config.large_db_sync_stagger_seconds:
                                wait = self.config.large_db_sync_stagger_seconds - since_last
                                logger.debug(f"Staggering large DB sync by {wait:.1f}s")
                                await asyncio.sleep(wait)
                            self._last_large_sync_time = time.time()

                    if execute_job_callback_async:
                        return await execute_job_callback_async(job)
                    elif execute_job_callback:
                        return execute_job_callback(job)
                    else:
                        # Default: mark as pending for external execution
                        self._active_sync_jobs[job.job_id] = job
                        return True

            for target_node, jobs in jobs_by_target.items():
                for job in jobs:
                    try:
                        success = await execute_job_throttled(job)

                        if success:
                            # Only set completed if not already set by callback
                            if job.status != "completed":
                                job.status = "completed"
                                job.completed_at = time.time()
                            results["jobs_completed"] += 1
                            self.stats.sync_jobs_completed += 1
                        else:
                            if job.status != "failed":
                                job.status = "failed"
                            results["jobs_failed"] += 1
                            self.stats.sync_jobs_failed += 1

                    except (OSError, ConnectionError, TimeoutError, asyncio.TimeoutError, RuntimeError) as e:
                        # Dec 2025: Narrowed from broad Exception - sync job execution errors
                        # OSError: File system errors during sync
                        # ConnectionError/TimeoutError: Network issues
                        # RuntimeError: Sync callback state errors
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
                    DataEventType.DATA_SYNC_COMPLETED,
                    jobs_completed=results["jobs_completed"],
                    bytes_synced=results["bytes_synced"],
                )
            else:
                self._emit_sync_event(
                    DataEventType.DATA_SYNC_FAILED,
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
            # Dec 2025: Event emission tracking
            "events_emitted": self.stats.events_emitted,
            "events_failed": self.stats.events_failed,
            "last_event_error": self.stats.last_event_error if self.stats.events_failed > 0 else None,
        }

    # ============================================
    # Disk-based Manifest Cache
    # ============================================
    # December 2025: Moved from p2p_orchestrator.py for better cohesion.
    # Disk caching complements in-memory caching for startup performance.

    def get_manifest_cache_path(self) -> Path:
        """Get path for persistent manifest cache on disk."""
        return self.data_directory / ".manifest_cache.json"

    def save_manifest_to_cache(self, manifest: "NodeDataManifest") -> bool:
        """Save manifest to disk for faster startup.

        Persists the current manifest state so nodes can resume quickly after
        restart without needing to rescan all data files.

        Args:
            manifest: The NodeDataManifest to cache

        Returns:
            True if saved successfully, False otherwise
        """
        import json

        try:
            cache_path = self.get_manifest_cache_path()
            cache_data = {
                "version": 1,
                "saved_at": time.time(),
                "manifest": manifest.to_dict() if hasattr(manifest, "to_dict") else {},
            }
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)
            logger.debug(f"Saved manifest cache to {cache_path}")
            return True
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Failed to save manifest cache: {e}")
            return False

    def load_manifest_from_cache(self, max_age_seconds: int = 60) -> "NodeDataManifest | None":
        """Load manifest from disk cache if fresh enough.

        Returns cached manifest if it exists and is not too old, otherwise None.
        This speeds up startup by avoiding full data directory scans.

        Args:
            max_age_seconds: Maximum age of cache to consider valid (default 60s)

        Returns:
            NodeDataManifest if cache is valid, None otherwise
        """
        import json
        from ..models import NodeDataManifest

        try:
            cache_path = self.get_manifest_cache_path()
            if not cache_path.exists():
                return None

            with open(cache_path) as f:
                cache_data = json.load(f)

            # Check version
            if cache_data.get("version") != 1:
                logger.debug("Manifest cache version mismatch, ignoring")
                return None

            # Check age
            saved_at = cache_data.get("saved_at", 0)
            if time.time() - saved_at > max_age_seconds:
                logger.debug(f"Manifest cache too old ({int(time.time() - saved_at)}s > {max_age_seconds}s)")
                return None

            # Parse manifest
            manifest_dict = cache_data.get("manifest", {})
            if not manifest_dict:
                return None

            manifest = NodeDataManifest.from_dict(manifest_dict)
            logger.info(f"Loaded manifest from cache (age: {int(time.time() - saved_at)}s)")
            return manifest

        except (OSError, json.JSONDecodeError, TypeError, KeyError) as e:
            logger.debug(f"Failed to load manifest cache: {e}")
            return None

    def collect_local_manifest_cached(self, max_cache_age: int = 60) -> "NodeDataManifest":
        """Collect manifest with disk caching support.

        First tries to load from disk cache, then falls back to in-memory cache,
        then falls back to full scan. Saves result to disk cache after collection.

        Args:
            max_cache_age: Maximum age in seconds for cached manifest

        Returns:
            NodeDataManifest with all discovered files
        """
        # Try disk cache first (for startup scenarios)
        cached = self.load_manifest_from_cache(max_age_seconds=max_cache_age)
        if cached:
            # Update in-memory cache too
            self._cached_local_manifest = cached
            self._cached_manifest_time = time.time()
            return cached

        # Use normal collection (which includes in-memory caching)
        manifest = self.collect_local_manifest(use_cache=True)

        # Save to disk cache for next startup
        self.save_manifest_to_cache(manifest)

        return manifest

    # ============================================
    # Pull Request Handling
    # ============================================

    async def handle_sync_pull_request(
        self,
        source_host: str,
        source_port: int,
        source_node_id: str,
        files: list[str],
        source_reported_host: str | None = None,
        source_reported_port: int | None = None,
        *,
        data_dir: Path | None = None,
        auth_headers_fn: Callable[[], dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Handle incoming request to pull files from a source node.

        Pulls files over the P2P HTTP channel to avoid SSH/rsync dependencies.
        Uses configurable data directory to support both disk and ramdrive storage.

        Jan 28, 2026: Phase 18A - Migrated from p2p_orchestrator.py.

        Args:
            source_host: Source host IP or hostname
            source_port: Source port (typically 8770)
            source_node_id: Source node ID for logging
            files: List of relative file paths to pull
            source_reported_host: Alternative host (e.g., Tailscale IP)
            source_reported_port: Alternative port
            data_dir: Target data directory (defaults to orchestrator's data dir)
            auth_headers_fn: Function to get auth headers (defaults to orchestrator's)

        Returns:
            Dict with success status, bytes_transferred, files_completed, and error if any
        """
        try:
            from scripts.p2p.network import get_client_session
            from aiohttp import ClientTimeout
        except ImportError:
            return {
                "success": False,
                "error": "aiohttp not available",
                "bytes_transferred": 0,
                "files_completed": 0,
            }

        # Check disk capacity before pulling files
        try:
            from scripts.p2p.utils import check_disk_has_capacity, MAX_DISK_USAGE_PERCENT
            has_capacity, disk_percent = check_disk_has_capacity()
        except ImportError:
            # Fallback: assume we have capacity
            has_capacity, disk_percent = True, 0.0
            MAX_DISK_USAGE_PERCENT = 90

        if not has_capacity:
            return {
                "success": False,
                "error": f"Disk full ({disk_percent:.1f}% >= {MAX_DISK_USAGE_PERCENT}%)",
                "disk_percent": disk_percent,
                "bytes_transferred": 0,
                "files_completed": 0,
            }

        # Get data directory
        if data_dir is None and self._orchestrator:
            data_dir = self._orchestrator.get_data_directory()
        if data_dir is None:
            data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Get auth headers function
        if auth_headers_fn is None and self._orchestrator:
            auth_headers_fn = self._orchestrator._auth_headers
        if auth_headers_fn is None:
            auth_headers_fn = lambda: {}

        bytes_transferred = 0
        files_completed = 0
        errors: list[str] = []

        # Multi-path sources: prefer observed endpoint but allow a self-reported
        # endpoint (e.g. Tailscale) when the public route fails
        candidate_sources: list[tuple[str, int]] = []
        seen_sources: set[tuple[str, int]] = set()

        def _add_source(host: str | None, port: int | None) -> None:
            if not host:
                return
            h = str(host).strip()
            if not h:
                return
            try:
                p = int(port or 0)
            except ValueError:
                return
            if p <= 0:
                return
            key = (h, p)
            if key in seen_sources:
                return
            seen_sources.add(key)
            candidate_sources.append(key)

        _add_source(source_host, source_port)
        _add_source(source_reported_host, source_reported_port)

        # Constants
        try:
            from scripts.p2p.constants import HTTP_CONNECT_TIMEOUT, DEFAULT_PORT
        except ImportError:
            HTTP_CONNECT_TIMEOUT = 30
            DEFAULT_PORT = 8770

        timeout = ClientTimeout(total=None, sock_connect=HTTP_CONNECT_TIMEOUT, sock_read=600)

        async with get_client_session(timeout) as session:
            for rel_path in files:
                rel_path = (rel_path or "").lstrip("/")
                if not rel_path:
                    errors.append("empty_path")
                    continue

                # Security: keep all writes within data directory
                dest_path = (data_dir / rel_path)
                try:
                    data_root = data_dir.resolve()
                    dest_resolved = dest_path.resolve()
                    dest_resolved.relative_to(data_root)
                except (AttributeError, ValueError):
                    errors.append(f"invalid_path:{rel_path}")
                    continue

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = dest_path.with_name(dest_path.name + ".partial")

                last_err: str | None = None
                success = False

                for host, base_port in candidate_sources:
                    # Back-compat: if caller passed an SSH-like port (22), try DEFAULT_PORT too
                    ports_to_try: list[int] = []
                    try:
                        ports_to_try.append(int(base_port))
                    except (ValueError, AttributeError):
                        ports_to_try.append(DEFAULT_PORT)
                    if DEFAULT_PORT not in ports_to_try:
                        ports_to_try.append(DEFAULT_PORT)

                    for port in ports_to_try:
                        url = f"http://{host}:{port}/sync/file"
                        try:
                            async with session.get(
                                url,
                                params={"path": rel_path},
                                headers=auth_headers_fn(),
                            ) as resp:
                                if resp.status != 200:
                                    text = ""
                                    try:
                                        text = (await resp.text())[:200]
                                    except (KeyError, IndexError, AttributeError):
                                        text = ""
                                    last_err = f"{resp.status} {text}".strip()
                                    continue

                                with open(tmp_path, "wb") as out_f:
                                    async for chunk in resp.content.iter_chunked(1024 * 1024):
                                        out_f.write(chunk)
                                        bytes_transferred += len(chunk)

                                tmp_path.replace(dest_path)
                                files_completed += 1
                                success = True
                                break

                        except Exception as e:  # noqa: BLE001
                            last_err = str(e)
                            continue
                    if success:
                        break

                if not success:
                    errors.append(f"{rel_path}: {last_err or 'download_failed'}")
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except OSError:
                        pass

        # Update stats
        self.stats.bytes_synced += bytes_transferred
        if files_completed > 0:
            self.stats.sync_jobs_completed += 1
        if errors:
            self.stats.sync_jobs_failed += 1

        if errors:
            return {
                "success": False,
                "files_completed": files_completed,
                "bytes_transferred": bytes_transferred,
                "error": "; ".join(errors[:5]),
            }

        return {
            "success": True,
            "files_completed": files_completed,
            "bytes_transferred": bytes_transferred,
        }

    # ============================================
    # Selfplay to Training Nodes Sync
    # ============================================

    async def sync_selfplay_to_training_nodes(
        self,
        *,
        get_training_nodes: Callable[[], list["NodeInfo"]],
        should_sync_to_node: Callable[["NodeInfo"], bool],
        should_cleanup_source: Callable[["NodeInfo"], bool],
        collect_manifest: Callable[[], Any],
        execute_sync_job: Callable[["DataSyncJob"], Any],
        cleanup_synced_files: Callable[[str, list[str]], Any],
        get_sync_router: Callable[[], Any] | None = None,
        cluster_manifest: "ClusterDataManifest | None" = None,
        max_files_per_job: int = 50,
    ) -> dict[str, Any]:
        """Sync selfplay data to training primary nodes.

        This method orchestrates syncing selfplay data from source nodes
        to training-capable nodes, with disk-aware filtering and cleanup.

        Args:
            get_training_nodes: Callback to get training primary nodes
            should_sync_to_node: Callback to check if node has disk capacity
            should_cleanup_source: Callback to check if source needs cleanup
            collect_manifest: Async callback to collect cluster manifest
            execute_sync_job: Async callback to execute a sync job
            cleanup_synced_files: Async callback to cleanup files on source
            get_sync_router: Optional callback to get SyncRouter for quality routing
            cluster_manifest: Optional pre-collected cluster manifest
            max_files_per_job: Maximum files per sync job (default: 50)

        Returns:
            Dict with sync results:
            - success: bool
            - training_nodes: list of node IDs synced to
            - sync_jobs_created: number of jobs created
            - successful_syncs: number of successful syncs
            - sources_cleaned: number of sources cleaned up

        December 2025: Extracted from P2POrchestrator._sync_selfplay_to_training_nodes()
        """
        from ..models import DataSyncJob

        # Get training primary nodes
        training_nodes = get_training_nodes()
        if not training_nodes:
            return {"success": False, "error": "No training nodes available"}

        # Filter by disk space - try SyncRouter first for quality-based routing
        router = get_sync_router() if get_sync_router else None
        if router is not None:
            try:
                # Refresh capacity data before routing
                if hasattr(router, 'refresh_all_capacity'):
                    router.refresh_all_capacity()

                # Get sync targets with quality-based priority
                targets = router.get_sync_targets(
                    data_type="game",
                    exclude_nodes=[self.node_id],
                    max_targets=len(training_nodes),
                )
                if targets:
                    # Filter to training nodes only
                    eligible_training_nodes = [
                        n for n in training_nodes
                        if any(t.node_id == n.node_id for t in targets)
                    ]
                    if eligible_training_nodes:
                        logger.info(
                            f"SyncRouter: selected {len(eligible_training_nodes)} "
                            f"training nodes with quality-based routing"
                        )
                    else:
                        eligible_training_nodes = [
                            n for n in training_nodes if should_sync_to_node(n)
                        ]
                else:
                    eligible_training_nodes = [
                        n for n in training_nodes if should_sync_to_node(n)
                    ]
            except (AttributeError, ValueError, KeyError, RuntimeError) as e:
                # Dec 2025: Narrowed from broad Exception - SyncRouter errors
                # AttributeError: Missing router method
                # ValueError/KeyError: Invalid routing data
                # RuntimeError: Router state errors
                logger.debug(f"SyncRouter fallback: {e}")
                eligible_training_nodes = [
                    n for n in training_nodes if should_sync_to_node(n)
                ]
        else:
            eligible_training_nodes = [
                n for n in training_nodes if should_sync_to_node(n)
            ]

        if not eligible_training_nodes:
            return {"success": False, "error": "All training nodes have critical disk usage"}

        logger.info(f"Training sync: {len(eligible_training_nodes)} eligible training nodes")
        for node in eligible_training_nodes:
            gpu_power = node.gpu_power_score() if hasattr(node, 'gpu_power_score') else 0
            disk_pct = node.disk_percent if hasattr(node, 'disk_percent') else 0
            gpu_name = node.gpu_name if hasattr(node, 'gpu_name') else "unknown"
            logger.info(
                f"  - {node.node_id}: {gpu_name} (power={gpu_power}, disk={disk_pct:.1f}%)"
            )

        # Emit DATA_SYNC_STARTED event
        sync_start_time = time.time()
        self._emit_sync_event(
            DataEventType.DATA_SYNC_STARTED,
            sync_type="training_sync",
            target_nodes=[n.node_id for n in eligible_training_nodes],
        )

        # Collect cluster manifest if not provided
        manifest = cluster_manifest
        if not manifest:
            logger.info("Collecting fresh cluster manifest for training sync...")
            # Jan 2026: Use centralized timeout from LoopTimeouts
            manifest_timeout = 120.0  # Default fallback
            if LoopTimeouts is not None:
                manifest_timeout = LoopTimeouts.MANIFEST_COLLECTION
            try:
                manifest = await asyncio.wait_for(
                    collect_manifest(),
                    timeout=manifest_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Manifest collection timed out after {manifest_timeout}s, trying cached")
                # Try to get cached manifest as fallback (sync method)
                try:
                    cached = self.get_cached_manifest()
                    if cached:
                        # Cached manifest is local-only, but can still be useful
                        # for determining what files exist locally
                        logger.info("Using cached local manifest as partial fallback")
                        # Continue without full cluster manifest - sync may be incomplete
                        manifest = None  # Will fail gracefully below
                except (OSError, json.JSONDecodeError) as e:
                    # Cache file read/parse errors
                    logger.debug(f"Cache read error: {e}")
                    manifest = None
                except (ValueError, KeyError) as e:
                    # Invalid cached data structure
                    logger.debug(f"Cache data format error: {e}")
                    manifest = None

        if not manifest:
            return {"success": False, "error": "Failed to collect cluster manifest"}

        # Track source nodes that need cleanup after sync
        sources_to_cleanup: dict[str, list[str]] = {}

        # Find selfplay files that training nodes don't have
        sync_jobs: list[DataSyncJob] = []

        for target_node in eligible_training_nodes:
            target_manifest = manifest.node_manifests.get(target_node.node_id)
            target_files: set[str] = set()
            if target_manifest:
                target_files = set(target_manifest.files_by_path.keys())

            # Find source nodes with selfplay data this target doesn't have
            for source_id, source_manifest in manifest.node_manifests.items():
                if source_id == target_node.node_id:
                    continue

                # Check if source node needs disk cleanup
                source_node = self._get_peers().get(source_id)
                needs_cleanup = source_node and should_cleanup_source(source_node)

                # Find selfplay files to sync (with mtime comparison)
                files_to_sync = []
                for file_info in source_manifest.files:
                    if file_info.file_type != "selfplay":
                        continue

                    # Check if target needs this file
                    target_file_info = (
                        target_manifest.files_by_path.get(file_info.path)
                        if target_manifest else None
                    )

                    should_sync = False
                    if file_info.path not in target_files:
                        # Target doesn't have file at all
                        should_sync = True
                    elif (
                        target_file_info and
                        file_info.modified_time > target_file_info.modified_time + 60
                    ):
                        # Source is newer (60s tolerance for clock skew)
                        should_sync = True

                    if should_sync:
                        files_to_sync.append(file_info.path)

                if files_to_sync:
                    job_id = (
                        f"training_sync_{source_id}_to_{target_node.node_id}_"
                        f"{int(time.time())}"
                    )
                    job = DataSyncJob(
                        job_id=job_id,
                        source_node=source_id,
                        target_node=target_node.node_id,
                        files=files_to_sync[:max_files_per_job],
                        status="pending",
                    )
                    sync_jobs.append(job)
                    self._active_sync_jobs[job_id] = job
                    self.stats.sync_jobs_created += 1
                    logger.info(
                        f"Created training sync job: {len(files_to_sync)} files "
                        f"from {source_id} to {target_node.node_id}"
                    )

                    # Track files for cleanup if source has high disk usage
                    if needs_cleanup:
                        if source_id not in sources_to_cleanup:
                            sources_to_cleanup[source_id] = []
                        sources_to_cleanup[source_id].extend(
                            files_to_sync[:max_files_per_job]
                        )

        # Execute sync jobs with concurrency limiting (Jan 2026)
        # Use semaphore to prevent memory pressure from too many rsync processes
        successful_syncs = 0

        async def execute_with_throttle(job: "DataSyncJob") -> bool:
            """Execute sync job with semaphore-based throttling."""
            async with self._sync_semaphore:
                # Stagger large DB syncs to prevent memory spikes
                if job.files and any("canonical" in f or "hexagonal" in f or "square19" in f for f in job.files):
                    # Large DB - check if we need to stagger
                    since_last_large = time.time() - self._last_large_sync_time
                    if since_last_large < self.config.large_db_sync_stagger_seconds:
                        wait_time = self.config.large_db_sync_stagger_seconds - since_last_large
                        logger.debug(f"Staggering large DB sync by {wait_time:.1f}s for memory protection")
                        await asyncio.sleep(wait_time)
                    self._last_large_sync_time = time.time()

                try:
                    return await execute_sync_job(job)
                except (OSError, ConnectionError, TimeoutError, asyncio.TimeoutError, RuntimeError) as e:
                    logger.info(f"Sync job {job.job_id} failed: {e}")
                    job.error_message = str(e)
                    return False

        for job in sync_jobs:
            success = await execute_with_throttle(job)
            if success:
                job.status = "completed"
                job.completed_at = time.time()
                successful_syncs += 1
                self.stats.sync_jobs_completed += 1
            else:
                job.status = "failed"
                self.stats.sync_jobs_failed += 1

        # Cleanup source nodes with high disk usage after successful syncs
        cleanup_results = {}
        if successful_syncs > 0 and sources_to_cleanup:
            logger.info(
                f"Running post-sync cleanup on {len(sources_to_cleanup)} source nodes..."
            )
            for source_id, files in sources_to_cleanup.items():
                try:
                    success = await cleanup_synced_files(source_id, files)
                    cleanup_results[source_id] = success
                except (OSError, ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
                    # Dec 2025: Narrowed from broad Exception - cleanup errors
                    logger.debug(f"Cleanup failed for {source_id}: {e}")
                    cleanup_results[source_id] = False

        self.stats.last_sync_execution = time.time()

        # Emit DATA_SYNC_COMPLETED event
        if successful_syncs > 0 or not sync_jobs:
            self._emit_sync_event(
                DataEventType.DATA_SYNC_COMPLETED,
                sync_type="training_sync",
                duration_seconds=time.time() - sync_start_time,
                sync_jobs_created=len(sync_jobs),
                successful_syncs=successful_syncs,
                target_nodes=[n.node_id for n in eligible_training_nodes],
            )
        else:
            self._emit_sync_event(
                DataEventType.DATA_SYNC_FAILED,
                sync_type="training_sync",
                duration_seconds=time.time() - sync_start_time,
                sync_jobs_created=len(sync_jobs),
                successful_syncs=0,
                error="All sync jobs failed",
            )

        return {
            "success": True,
            "training_nodes": [n.node_id for n in eligible_training_nodes],
            "sync_jobs_created": len(sync_jobs),
            "successful_syncs": successful_syncs,
            "sources_cleaned": sum(1 for v in cleanup_results.values() if v),
        }

    # =========================================================================
    # Health Check (December 2025)
    # =========================================================================

    def health_check(self):
        """Check health status of SyncPlanner.

        Returns:
            HealthCheckResult with status, sync metrics, and error info
        """
        # Import from contracts (zero dependencies)
        from app.coordination.contracts import HealthCheckResult, CoordinatorStatus

        status = CoordinatorStatus.RUNNING
        is_healthy = True
        errors_count = 0
        last_error: str | None = None

        # Check sync stats for issues
        total_ops = (
            self.stats.manifests_collected
            + self.stats.sync_plans_generated
            + self.stats.sync_jobs_created
        )
        failed_jobs = self.stats.sync_jobs_failed

        if self.stats.sync_jobs_created > 0:
            failure_rate = failed_jobs / self.stats.sync_jobs_created
            if failure_rate > 0.5:
                status = CoordinatorStatus.ERROR
                is_healthy = False
                last_error = f"High sync failure rate: {failure_rate:.0%}"
                errors_count = failed_jobs
            elif failure_rate > 0.2:
                status = CoordinatorStatus.DEGRADED
                last_error = f"Elevated sync failure rate: {failure_rate:.0%}"
                errors_count = failed_jobs

        # Check last sync time - if no syncs in 30 minutes, degraded
        if self.stats.last_sync_execution > 0:
            time_since_last = time.time() - self.stats.last_sync_execution
            if time_since_last > 1800:  # 30 minutes
                if is_healthy:
                    status = CoordinatorStatus.DEGRADED
                    last_error = f"No sync in {time_since_last / 60:.0f} minutes"

        return HealthCheckResult(
            healthy=is_healthy,
            status=status if isinstance(status, str) else status,
            message=last_error or "SyncPlanner healthy",
            details={
                "operations_count": total_ops,
                "errors_count": errors_count,
                "manifests_collected": self.stats.manifests_collected,
                "sync_plans_generated": self.stats.sync_plans_generated,
                "sync_jobs_created": self.stats.sync_jobs_created,
                "sync_jobs_completed": self.stats.sync_jobs_completed,
                "sync_jobs_failed": self.stats.sync_jobs_failed,
                "bytes_synced": self.stats.bytes_synced,
                "last_sync_execution": self.stats.last_sync_execution,
            },
        )

    # =========================================================================
    # Event Subscriptions (December 2025 - uses EventSubscriptionMixin)
    # =========================================================================

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Return event subscriptions for EventSubscriptionMixin.

        Dec 28, 2025: Migrated to use EventSubscriptionMixin pattern.

        Returns:
            Dict mapping event names to handler methods
        """
        return {
            "LEADER_ELECTED": self._on_leader_elected,
            "NODE_RECOVERED": self._on_node_recovered,
            "HOST_ONLINE": self._on_host_online,
        }

    async def _on_leader_elected(self, event) -> None:
        """Handle LEADER_ELECTED events - clear cached manifests.

        When leadership changes, cached manifests may be stale since
        the new leader should re-collect from all nodes.
        """
        payload = self._extract_event_payload(event)
        new_leader = payload.get("leader_id", "")

        self._log_info(f"LEADER_ELECTED: {new_leader}, clearing cached manifests")

        # Clear cached manifests to force re-collection
        self._cached_local_manifest = None
        self._cached_manifest_time = 0.0
        self._cluster_manifest = None

    async def _on_node_recovered(self, event) -> None:
        """Handle NODE_RECOVERED events - invalidate manifest cache.

        When a node recovers, it may have new data that the leader
        needs to discover via manifest collection.
        """
        payload = self._extract_event_payload(event)
        node_id = payload.get("node_id", "") or payload.get("host", "")

        if not node_id:
            return

        self._log_info(f"NODE_RECOVERED: {node_id}, invalidating manifest cache")

        # Invalidate cluster manifest to trigger re-collection
        # Local manifest is kept since it's about this node
        self._cluster_manifest = None

    async def _on_host_online(self, event) -> None:
        """Handle HOST_ONLINE events - invalidate cluster manifest.

        When a new host comes online, it may have data that the leader
        should include in cluster-wide sync planning.
        """
        payload = self._extract_event_payload(event)
        node_id = payload.get("node_id", "")

        if not node_id:
            return

        self._log_info(f"HOST_ONLINE: {node_id}, invalidating cluster manifest")

        # Invalidate cluster manifest to trigger re-collection
        self._cluster_manifest = None
