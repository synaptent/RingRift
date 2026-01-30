"""Sync Orchestrator - Handles data synchronization across the cluster.

January 2026: Created as part of P2POrchestrator decomposition.

Responsibilities:
- Data sync coordination between nodes
- Manifest collection (local and remote)
- Node selection for sync targets
- Multi-transport failover (Tailscale -> SSH -> Base64)
- Deduplication tracking
- Model sync coordination
- Database sync coordination
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.p2p.orchestrators.base_orchestrator import BaseOrchestrator, HealthCheckResult

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator

logger = logging.getLogger(__name__)


class SyncOrchestrator(BaseOrchestrator):
    """Orchestrator for data synchronization across the cluster.

    This orchestrator handles all aspects of data sync in the P2P cluster:
    - Collecting and distributing data manifests
    - Coordinating selfplay data sync to training nodes
    - Managing model distribution
    - Tracking sync deduplication

    The actual sync logic is delegated to SyncPlanner and SyncRouter,
    but this orchestrator provides a unified interface and health monitoring.

    Usage:
        # In P2POrchestrator.__init__:
        self.sync = SyncOrchestrator(self)

        # Collect local manifest:
        manifest = self.sync.collect_local_manifest()

        # Check if sync needed:
        if self.sync.should_sync_to_node(node):
            ...
    """

    def __init__(self, p2p: "P2POrchestrator") -> None:
        """Initialize the sync orchestrator.

        Args:
            p2p: The parent P2POrchestrator instance.
        """
        super().__init__(p2p)

        # Sync statistics
        self._last_sync_time: float = 0.0
        self._sync_count: int = 0
        self._sync_errors: int = 0

    @property
    def name(self) -> str:
        """Return the name of this orchestrator."""
        return "sync"

    def health_check(self) -> HealthCheckResult:
        """Check the health of sync orchestrator.

        Returns:
            HealthCheckResult with sync status details.
        """
        try:
            issues = []

            # Check sync planner health
            sync_planner = getattr(self._p2p, "sync_planner", None)
            if sync_planner is None:
                issues.append("SyncPlanner not available")

            # Check for recent sync activity
            now = time.time()
            time_since_sync = now - self._last_sync_time
            if self._last_sync_time > 0 and time_since_sync > 3600:  # 1 hour
                issues.append(f"No sync activity for {time_since_sync/3600:.1f}h")

            # Check error rate
            if self._sync_count > 0:
                error_rate = self._sync_errors / self._sync_count
                if error_rate > 0.5:
                    issues.append(f"High sync error rate: {error_rate:.0%}")

            healthy = len(issues) == 0
            message = "Sync healthy" if healthy else "; ".join(issues)

            return HealthCheckResult(
                healthy=healthy,
                message=message,
                details={
                    "sync_count": self._sync_count,
                    "sync_errors": self._sync_errors,
                    "last_sync_time": self._last_sync_time,
                    "time_since_sync": time_since_sync if self._last_sync_time > 0 else None,
                    "issues": issues,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )

    # =========================================================================
    # Manifest Collection
    # =========================================================================

    def collect_local_manifest(self, use_cache: bool = True) -> Any:
        """Collect manifest of all data files on this node.

        Jan 29, 2026: Wrapper for SyncPlanner.collect_local_manifest().

        Args:
            use_cache: Whether to use cached manifest (recommended for performance)

        Returns:
            NodeDataManifest with local file information.
        """
        sync_planner = getattr(self._p2p, "sync_planner", None)
        if sync_planner is None:
            self._log_warning("SyncPlanner not available for manifest collection")
            return None
        return sync_planner.collect_local_manifest(use_cache=use_cache)

    def request_peer_manifest(self, peer_id: str) -> Any:
        """Request data manifest from a peer node.

        Jan 29, 2026: Wrapper for P2POrchestrator._request_peer_manifest_sync().

        Args:
            peer_id: The peer's node ID to request from

        Returns:
            NodeDataManifest or None if request failed
        """
        if hasattr(self._p2p, "_request_peer_manifest_sync"):
            return self._p2p._request_peer_manifest_sync(peer_id)
        return None

    # =========================================================================
    # Sync Decision Logic
    # =========================================================================

    def should_sync_to_node(self, node: Any) -> bool:
        """Check if we should sync data to a specific node.

        Jan 29, 2026: Wrapper for P2POrchestrator._should_sync_to_node().

        Args:
            node: NodeInfo of the target node

        Returns:
            True if we should sync to this node.
        """
        if hasattr(self._p2p, "_should_sync_to_node"):
            return self._p2p._should_sync_to_node(node)
        return False

    def should_cleanup_source(self, node: Any) -> bool:
        """Check if source data should be cleaned up after sync.

        Jan 29, 2026: Wrapper for P2POrchestrator._should_cleanup_source().

        Args:
            node: NodeInfo of the source node

        Returns:
            True if source cleanup is allowed.
        """
        if hasattr(self._p2p, "_should_cleanup_source"):
            return self._p2p._should_cleanup_source(node)
        return False

    # =========================================================================
    # Sync Execution
    # =========================================================================

    async def sync_selfplay_to_training_nodes(self) -> dict[str, Any]:
        """Sync selfplay data to training nodes.

        Jan 29, 2026: Wrapper for P2POrchestrator._sync_selfplay_to_training_nodes().

        Returns:
            Dict with sync results (files_synced, nodes_synced, errors)
        """
        self._sync_count += 1
        try:
            if hasattr(self._p2p, "_sync_selfplay_to_training_nodes"):
                result = await self._p2p._sync_selfplay_to_training_nodes()
                self._last_sync_time = time.time()
                return result
            return {"error": "sync method not available"}
        except Exception as e:
            self._sync_errors += 1
            self._log_error(f"Sync failed: {e}")
            return {"error": str(e)}

    # =========================================================================
    # Sync Status
    # =========================================================================

    def get_sync_status(self) -> dict[str, Any]:
        """Get current sync status.

        Returns:
            Dict with sync statistics and state.
        """
        sync_planner = getattr(self._p2p, "sync_planner", None)

        status = {
            "sync_count": self._sync_count,
            "sync_errors": self._sync_errors,
            "last_sync_time": self._last_sync_time,
            "sync_planner_available": sync_planner is not None,
        }

        if sync_planner is not None and hasattr(sync_planner, "get_status"):
            status["planner_status"] = sync_planner.get_status()

        return status

    def record_sync_success(self) -> None:
        """Record a successful sync operation."""
        self._last_sync_time = time.time()
        self._sync_count += 1

    def record_sync_error(self) -> None:
        """Record a sync error."""
        self._sync_count += 1
        self._sync_errors += 1

    # =========================================================================
    # Cleanup Operations
    # =========================================================================

    async def cleanup_synced_files(self, node_id: str, files: list[str]) -> bool:
        """Delete synced files from source node to free disk space.

        Jan 29, 2026: Implementation moved from P2POrchestrator._cleanup_synced_files().

        Only called after successful sync to training nodes.

        Args:
            node_id: The node ID to cleanup files on.
            files: List of file paths to delete.

        Returns:
            True if cleanup was successful or enqueued.
        """
        # Import aiohttp types here to avoid circular imports
        try:
            from aiohttp import ClientTimeout
            from app.core.http_client import get_client_session
        except ImportError:
            self._log_warning("Cannot cleanup files: aiohttp not available")
            return False

        # Get peer info from P2P
        peers_lock = getattr(self._p2p, "peers_lock", None)
        peers = getattr(self._p2p, "peers", {})

        if peers_lock is None:
            return False

        with peers_lock:
            node = peers.get(node_id)

        if not node or not node.is_alive():
            return False

        try:
            # Check for NAT-blocked nodes - use relay command
            if getattr(node, "nat_blocked", False):
                if hasattr(self._p2p, "_enqueue_relay_command_for_peer"):
                    cmd_id = await self._p2p._enqueue_relay_command_for_peer(
                        node,
                        "cleanup_files",
                        {"files": list(files or []), "reason": "post_sync_cleanup"},
                    )
                    if cmd_id:
                        self._log_info(f"Enqueued relay cleanup_files for {node_id} ({len(files)} files)")
                        return True
                    self._log_info(f"Relay queue full for {node_id}; skipping cleanup_files enqueue")
                    return False
                return False

            # Direct HTTP request
            timeout = ClientTimeout(total=60)
            async with get_client_session(timeout) as session:
                last_err: str | None = None

                # Get URLs from P2P
                urls_method = getattr(self._p2p, "_urls_for_peer", None)
                auth_headers_method = getattr(self._p2p, "_auth_headers", None)

                if urls_method is None:
                    self._log_warning("Cannot cleanup files: _urls_for_peer not available")
                    return False

                for url in urls_method(node, "/cleanup/files"):
                    try:
                        headers = auth_headers_method() if auth_headers_method else {}
                        async with session.post(
                            url,
                            json={"files": files, "reason": "post_sync_cleanup"},
                            headers=headers,
                        ) as resp:
                            if resp.status != 200:
                                last_err = f"http_{resp.status}"
                                continue
                            result = await resp.json()
                            freed_bytes = result.get("freed_bytes", 0)
                            self._log_info(f"Cleanup on {node_id}: freed {freed_bytes / 1e6:.1f}MB")
                            return True
                    except Exception as e:  # noqa: BLE001
                        last_err = str(e)
                        continue

                if last_err:
                    self._log_info(f"Cleanup files request failed on {node_id}: {last_err}")

        except Exception as e:  # noqa: BLE001
            self._log_error(f"Failed to cleanup files on {node_id}: {e}")

        return False

    # =========================================================================
    # Elo Summary
    # =========================================================================

    def get_local_elo_summary(self) -> dict[str, Any]:
        """Get summary of local ELO ratings for gossip propagation.

        Jan 29, 2026: Implementation moved from P2POrchestrator._get_local_elo_summary().

        DISTRIBUTED ELO: Share top models and their ratings via gossip so all
        nodes have visibility into model performance without querying the DB.

        LAZY LOADING: Defers ELO query until after startup (60s) to avoid
        slowing node initialization. Uses 10-minute cache to reduce DB load.

        Returns:
            Dict with top_models, total_models, last_update
        """
        import sqlite3

        now = time.time()
        cache_key = "_elo_summary_cache"
        cache_time_key = "_elo_summary_cache_time"
        startup_key = "_elo_startup_time"
        cached = getattr(self, cache_key, None)
        cached_time = getattr(self, cache_time_key, 0)

        # Track startup time for lazy loading
        if not hasattr(self, startup_key):
            setattr(self, startup_key, now)

        startup_time = getattr(self, startup_key, now)

        # LAZY LOADING: Don't query ELO during first 60s of startup
        if now - startup_time < 60:
            return {"top_models": [], "total_models": 0, "last_update": 0, "deferred": True}

        # Use 10-minute cache to reduce DB load
        if cached and now - cached_time < 600:
            return cached

        summary: dict[str, Any] = {
            "top_models": [],
            "total_models": 0,
            "last_update": 0,
        }

        try:
            from app.tournament import get_elo_database
            db = get_elo_database()

            # Get top 5 models by ELO (single optimized query)
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT participant_id, rating, games_played, last_update,
                           (SELECT COUNT(*) FROM elo_ratings) as total,
                           (SELECT MAX(last_update) FROM elo_ratings) as max_updated
                    FROM elo_ratings
                    ORDER BY rating DESC
                    LIMIT 5
                """)
                rows = cursor.fetchall()

                if rows:
                    summary["total_models"] = rows[0][4] if rows[0][4] else 0
                    summary["last_update"] = rows[0][5] if rows[0][5] else 0

                for row in rows:
                    summary["top_models"].append({
                        "model": row[0],
                        "elo": round(row[1]),
                        "games": row[2],
                    })

        except (KeyError, IndexError, AttributeError, ImportError, sqlite3.OperationalError):
            # Silently fail - ELO summary is optional
            pass

        # Cache the result
        setattr(self, cache_key, summary)
        setattr(self, cache_time_key, now)

        return summary

    # =========================================================================
    # Cluster Observability
    # =========================================================================

    def get_cluster_observability(self) -> dict[str, Any]:
        """Get cluster observability metrics for debugging.

        Jan 29, 2026: Implementation moved from P2POrchestrator._get_cluster_observability().

        December 30, 2025: Added to help diagnose idle GPU nodes and
        peer visibility discrepancies across the cluster.

        Returns:
            Dict with unhealthy_nodes, gossip_discovered_peers, cluster_job_distribution
        """
        result: dict[str, Any] = {}

        # 1. Unhealthy nodes from node_selector
        try:
            node_selector = getattr(self._p2p, "node_selector", None)
            if node_selector:
                unhealthy_set = getattr(node_selector, "_unhealthy_nodes", set())
                unhealthy_reasons = getattr(node_selector, "_unhealthy_reasons", {})
                result["unhealthy_nodes"] = {
                    "count": len(unhealthy_set),
                    "node_ids": list(unhealthy_set),
                    "reasons": dict(unhealthy_reasons),
                }
            else:
                result["unhealthy_nodes"] = {"error": "node_selector not available"}
        except Exception as e:  # noqa: BLE001
            result["unhealthy_nodes"] = {"error": str(e)}

        # 2. Gossip-discovered peers
        try:
            gossip_endpoints = getattr(self._p2p, "_gossip_learned_endpoints", {})
            result["gossip_discovered_peers"] = {
                "count": len(gossip_endpoints),
                "node_ids": list(gossip_endpoints.keys()),
            }
        except Exception as e:  # noqa: BLE001
            result["gossip_discovered_peers"] = {"error": str(e)}

        # 3. Cluster job distribution (for balance analysis)
        try:
            peers_lock = getattr(self._p2p, "peers_lock", None)
            peers = getattr(self._p2p, "peers", {})
            node_id = getattr(self._p2p, "node_id", "")
            self_info = getattr(self._p2p, "self_info", None)

            if peers_lock:
                with peers_lock:
                    job_distribution = {}
                    for pid, peer in peers.items():
                        if peer.is_alive():
                            job_distribution[pid] = {
                                "selfplay_jobs": int(getattr(peer, "selfplay_jobs", 0) or 0),
                                "training_jobs": int(getattr(peer, "training_jobs", 0) or 0),
                                "gpu_percent": float(getattr(peer, "gpu_percent", 0) or 0),
                            }
                    # Add self
                    if self_info:
                        job_distribution[node_id] = {
                            "selfplay_jobs": int(getattr(self_info, "selfplay_jobs", 0) or 0),
                            "training_jobs": int(getattr(self_info, "training_jobs", 0) or 0),
                            "gpu_percent": float(getattr(self_info, "gpu_percent", 0) or 0),
                        }

                # Compute summary stats
                if job_distribution:
                    all_jobs = [d["selfplay_jobs"] for d in job_distribution.values()]
                    avg_jobs = sum(all_jobs) / len(all_jobs) if all_jobs else 0
                    max_jobs = max(all_jobs) if all_jobs else 0
                    min_jobs = min(all_jobs) if all_jobs else 0
                    idle_count = sum(1 for j in all_jobs if j == 0)
                    result["cluster_job_distribution"] = {
                        "node_count": len(job_distribution),
                        "avg_selfplay_jobs": round(avg_jobs, 1),
                        "max_selfplay_jobs": max_jobs,
                        "min_selfplay_jobs": min_jobs,
                        "idle_nodes": idle_count,
                        "per_node": job_distribution,
                    }
                else:
                    result["cluster_job_distribution"] = {"error": "no peers available"}
            else:
                result["cluster_job_distribution"] = {"error": "peers_lock not available"}
        except Exception as e:  # noqa: BLE001
            result["cluster_job_distribution"] = {"error": str(e)}

        return result

    # =========================================================================
    # Cluster Activity
    # =========================================================================

    def calculate_cluster_activity_factor(self) -> float:
        """Calculate cluster activity factor for sync interval adjustment.

        Jan 29, 2026: Implementation moved from P2POrchestrator._calculate_cluster_activity_factor().

        CLUSTER ACTIVITY FACTOR:
        - < 1.0: Active cluster (training, selfplay) = faster sync
        - 1.0: Normal activity
        - > 1.0: Idle cluster = slower sync

        Returns:
            Activity factor (0.5 to 2.0)
        """
        now = time.time()

        # Check training activity (with defensive checks)
        training_active = False
        training_lock = getattr(self._p2p, "training_lock", None)
        training_jobs = getattr(self._p2p, "training_jobs", {})
        if training_lock and training_jobs:
            try:
                with training_lock:
                    for job in training_jobs.values():
                        if getattr(job, "status", None) == "running":
                            training_active = True
                            break
            except AttributeError:
                pass

        # Check selfplay activity (count active jobs)
        selfplay_count = 0
        jobs_lock = getattr(self._p2p, "jobs_lock", None)
        selfplay_jobs = getattr(self._p2p, "selfplay_jobs", {})
        if jobs_lock and selfplay_jobs:
            try:
                with jobs_lock:
                    for job in selfplay_jobs.values():
                        if getattr(job, "status", None) == "running":
                            selfplay_count += 1
            except AttributeError:
                pass

        # Check recent data generation from gossip
        recent_data = False
        gossip_states = getattr(self._p2p, "_gossip_node_states", {}) or {}
        for _node_id, state in gossip_states.items():
            if not isinstance(state, dict):
                continue
            last_game = state.get("last_game_time", 0)
            if now - last_game < 300:  # Game in last 5 min
                recent_data = True
                break

        # Calculate factor
        factor = 1.0

        if training_active:
            factor *= 0.5  # Much faster sync during training
        elif selfplay_count >= 5:
            factor *= 0.7  # Faster sync with active selfplay
        elif selfplay_count > 0:
            factor *= 0.85  # Slightly faster with some activity

        if recent_data:
            factor *= 0.9  # Faster sync when new data available

        # If completely idle (no jobs, stale data), slow down
        if selfplay_count == 0 and not training_active and not recent_data:
            factor *= 1.5

        return max(0.5, min(2.0, factor))

    # =========================================================================
    # Decentralized P2P Data Sync
    # =========================================================================

    async def p2p_data_sync(self) -> None:
        """DECENTRALIZED: Nodes sync data directly with peers without leader coordination.

        Jan 29, 2026: Extracted from P2POrchestrator._p2p_data_sync().

        P2P DATA SYNC with enhancements:
        - Health-based peer selection (avoids overloaded nodes)
        - Circuit breaker (skips unreliable peers)
        - Delta sync (only syncs files newer than last sync)
        - Model file prioritization (syncs models first)
        - ADAPTIVE INTERVALS: adjusts based on cluster activity and success rate
        """
        try:
            from scripts.p2p.node_info import NodeRole
        except ImportError:
            NodeRole = None

        p2p = self._p2p
        now = time.time()

        # ADAPTIVE INTERVAL: Uses activity-aware interval instead of fixed 5 min
        if hasattr(p2p, "_get_adaptive_sync_interval"):
            interval = p2p._get_adaptive_sync_interval("data")
        else:
            interval = 300  # 5 min default
        last_check = getattr(p2p, "_last_p2p_sync_check", 0)
        if now - last_check < interval:
            return
        p2p._last_p2p_sync_check = now

        # Skip if leader is actively managing sync (avoid conflicts)
        if NodeRole is not None and getattr(p2p, "role", None) == NodeRole.LEADER:
            return  # Leader uses centralized sync

        # Skip if a sync is already in progress
        if getattr(p2p, "sync_in_progress", False):
            return

        # Skip if under disk pressure
        self_info = getattr(p2p, "self_info", None)
        if self_info and getattr(self_info, "disk_percent", 0) > 85:
            return

        # Get our local manifest (use cache for speed)
        local_manifest = getattr(p2p, "local_data_manifest", None)
        if not local_manifest:
            try:
                sync_planner = getattr(p2p, "sync_planner", None)
                if sync_planner is None:
                    return
                # Jan 23, 2026: Wrap in asyncio.to_thread() to prevent event loop blocking
                # collect_local_manifest_cached() does file I/O and SQLite operations
                local_manifest = await asyncio.to_thread(
                    sync_planner.collect_local_manifest_cached, max_cache_age=600
                )
                manifest_lock = getattr(p2p, "manifest_lock", None)
                if manifest_lock:
                    with manifest_lock:
                        p2p.local_data_manifest = local_manifest
                else:
                    p2p.local_data_manifest = local_manifest
            except AttributeError:
                return

        # Get local file set with timestamps for delta sync
        local_files: dict[str, float] = {}
        for file_info in getattr(local_manifest, "files", []) or []:
            rel_path = getattr(file_info, "relative_path", "")
            if rel_path:
                local_files[rel_path] = getattr(file_info, "modified_at", 0)

        # Check peer manifests from gossip cache
        peer_manifests = getattr(p2p, "_gossip_peer_manifests", {})
        if not peer_manifests:
            return

        # Find files we're missing that peers have (with prioritization)
        files_to_sync: dict[str, list[tuple]] = {}  # peer_id -> [(file, priority)]
        file_hashes: dict[str, str] = {}  # file_path -> hash (for dedup tracking)
        last_sync_time = getattr(p2p, "_last_successful_p2p_sync", 0)

        for peer_id, peer_manifest in peer_manifests.items():
            if peer_id == p2p.node_id:
                continue

            # Check circuit breaker
            health = p2p._get_peer_health_score(peer_id) if hasattr(p2p, "_get_peer_health_score") else 50
            if health <= 0:
                continue

            peer_files = getattr(peer_manifest, "files", []) or []
            for file_info in peer_files:
                rel_path = getattr(file_info, "relative_path", "")
                modified_at = getattr(file_info, "modified_at", 0)
                file_hash = getattr(file_info, "file_hash", "")
                file_size = getattr(file_info, "size_bytes", 0)

                if not rel_path:
                    continue

                # Skip if we have this file with same or newer timestamp
                if rel_path in local_files and local_files[rel_path] >= modified_at:
                    continue

                # Skip if file is older than last sync (delta optimization)
                if modified_at < last_sync_time and rel_path in local_files:
                    continue

                # DATA DEDUPLICATION: Skip if we already synced this file (by hash)
                if file_hash and hasattr(p2p, "_is_file_already_synced") and p2p._is_file_already_synced(file_hash):
                    if hasattr(p2p, "_record_dedup_skip"):
                        p2p._record_dedup_skip(file_count=1, bytes_saved=file_size)
                    continue

                # Calculate priority (models > ELO/training DBs > training data > selfplay)
                priority = 0
                if "models/" in rel_path or rel_path.endswith(".pt") or rel_path.endswith(".onnx"):
                    priority = 100  # Highest priority for models
                elif rel_path.endswith(".db") and ("unified_elo" in rel_path or "elo_ratings" in rel_path):
                    priority = 90  # Very high priority for ELO database
                elif rel_path.endswith(".db") and ("canonical_" in rel_path or "consolidated_training" in rel_path or "training_pool" in rel_path):
                    priority = 80  # High priority for training databases
                elif "training/" in rel_path:
                    priority = 50
                elif rel_path.endswith(".db"):
                    priority = 30  # Medium priority for other databases
                else:
                    priority = 10

                if peer_id not in files_to_sync:
                    files_to_sync[peer_id] = []
                files_to_sync[peer_id].append((rel_path, priority, health))

                # Track hash for dedup recording after sync
                if file_hash:
                    file_hashes[rel_path] = file_hash

        if not files_to_sync:
            return

        # Select best peer using health score AND file count
        def peer_score(pid: str) -> float:
            files = files_to_sync[pid]
            h = p2p._get_peer_health_score(pid) if hasattr(p2p, "_get_peer_health_score") else 50
            file_score = sum(f[1] for f in files)  # Sum of priorities
            return h * 0.4 + file_score * 0.6

        best_peer = max(files_to_sync.keys(), key=peer_score)
        files_with_priority = files_to_sync[best_peer]

        # Sort by priority (highest first) and take top 10
        files_with_priority.sort(key=lambda x: x[1], reverse=True)
        files_to_request = [f[0] for f in files_with_priority[:10]]

        # Check if peer is alive
        peers_lock = getattr(p2p, "peers_lock", None)
        peers = getattr(p2p, "peers", {})
        if peers_lock:
            with peers_lock:
                peer = peers.get(best_peer)
        else:
            peer = peers.get(best_peer)

        if not peer or not peer.is_alive():
            return

        # Log and initiate sync
        total_missing = sum(len(f) for f in files_to_sync.values())
        model_files = sum(1 for f in files_to_request if "models/" in f or f.endswith(".pt"))
        peer_health = p2p._get_peer_health_score(best_peer) if hasattr(p2p, "_get_peer_health_score") else 0
        logger.info(
            f"P2P SYNC: Missing {total_missing} files, requesting {len(files_to_request)} "
            f"({model_files} models) from {best_peer} (health={peer_health:.0f})"
        )

        try:
            import uuid as uuid_mod
            # Import DataSyncJob from P2P
            try:
                from scripts.p2p.sync_types import DataSyncJob
            except ImportError:
                # Fallback: try to get it from p2p module
                DataSyncJob = getattr(p2p, "DataSyncJob", None)
                if DataSyncJob is None:
                    logger.debug("P2P SYNC: DataSyncJob not available, skipping")
                    return

            job = DataSyncJob(
                job_id=f"p2p_{uuid_mod.uuid4().hex[:8]}",
                source_node=best_peer,
                target_node=p2p.node_id,
                files=files_to_request,
            )

            p2p.sync_in_progress = True
            try:
                success = await p2p._request_node_sync(job)
                if hasattr(p2p, "_record_p2p_sync_result"):
                    p2p._record_p2p_sync_result(best_peer, success)
                if hasattr(p2p, "_record_sync_result_for_adaptive"):
                    p2p._record_sync_result_for_adaptive("data", success)  # ADAPTIVE INTERVAL

                if success:
                    logger.info(f"P2P SYNC: Completed {len(files_to_request)} files from {best_peer}")
                    p2p._last_successful_p2p_sync = now
                    # Invalidate manifest cache
                    sync_planner = getattr(p2p, "sync_planner", None)
                    if sync_planner:
                        cache_path = sync_planner.get_manifest_cache_path()
                        if cache_path.exists():
                            cache_path.unlink()
                    # Update metrics
                    if hasattr(p2p, "_p2p_sync_metrics"):
                        p2p._p2p_sync_metrics["bytes"] += job.bytes_transferred
                    # DATA DEDUPLICATION: Record synced file hashes
                    for fpath in files_to_request:
                        if fpath in file_hashes and hasattr(p2p, "_record_synced_file"):
                            p2p._record_synced_file(file_hashes[fpath], 0)
                else:
                    logger.info(f"P2P SYNC: Failed from {best_peer}: {job.error_message}")
            finally:
                p2p.sync_in_progress = False

        except Exception as e:  # noqa: BLE001
            logger.info(f"P2P SYNC: Error: {e}")
            if hasattr(p2p, "_record_sync_result_for_adaptive"):
                p2p._record_sync_result_for_adaptive("data", False)  # ADAPTIVE: record failure
            if hasattr(p2p, "_record_p2p_sync_result"):
                p2p._record_p2p_sync_result(best_peer, False)
            p2p.sync_in_progress = False

    # =========================================================================
    # GPU Selfplay Import
    # =========================================================================

    def import_gpu_selfplay_sync(self, validated_db: Path, canonical_db: Path) -> int:
        """Synchronous helper for importing GPU selfplay data to canonical database.

        Jan 29, 2026: Implementation moved from P2POrchestrator._import_gpu_selfplay_sync().

        This method copies games and moves from a validated selfplay database
        to the canonical training database, avoiding duplicates.

        Args:
            validated_db: Path to the source validated selfplay database.
            canonical_db: Path to the destination canonical database.

        Returns:
            Number of games imported.
        """
        try:
            from app.coordination.database_utils import safe_db_connection
        except ImportError:
            # Fallback context manager if database_utils not available
            from contextlib import contextmanager

            @contextmanager
            def safe_db_connection(db_path: Path):
                conn = sqlite3.connect(str(db_path), timeout=30.0)
                try:
                    yield conn
                finally:
                    conn.close()

        # Phase 3.4 Dec 29, 2025: Use context managers to prevent connection leaks
        with safe_db_connection(validated_db) as src_conn, \
             safe_db_connection(canonical_db) as dst_conn:

            # Ensure destination tables exist
            dst_conn.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    winner INTEGER,
                    move_count INTEGER,
                    game_time_ms INTEGER,
                    created_at REAL,
                    source TEXT DEFAULT 'selfplay'
                )
            """)
            dst_conn.execute("""
                CREATE TABLE IF NOT EXISTS moves (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    move_number INTEGER NOT NULL,
                    player INTEGER NOT NULL,
                    move_type TEXT NOT NULL,
                    from_pos TEXT,
                    to_pos TEXT,
                    direction TEXT,
                    captured_pos TEXT,
                    state_before TEXT,
                    policy_probs TEXT,
                    value_est REAL,
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                )
            """)
            dst_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_moves_game_id ON moves(game_id)
            """)
            dst_conn.commit()

            # Check source schema and copy games
            src_cursor = src_conn.cursor()
            src_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            src_tables = {row[0] for row in src_cursor.fetchall()}

            imported = 0
            if "games" in src_tables:
                # Get existing game IDs in destination to avoid duplicates
                dst_cursor = dst_conn.cursor()
                dst_cursor.execute("SELECT game_id FROM games")
                existing_ids = {row[0] for row in dst_cursor.fetchall()}

                # Copy games that don't already exist
                src_cursor.execute("SELECT * FROM games")
                src_columns = [desc[0] for desc in src_cursor.description]

                for row in src_cursor.fetchall():
                    game_id_idx = src_columns.index("game_id") if "game_id" in src_columns else 0
                    game_id = row[game_id_idx]

                    if game_id in existing_ids:
                        continue

                    # Insert game with proper column mapping
                    placeholders = ", ".join(["?"] * len(row))
                    columns = ", ".join(src_columns)
                    try:
                        dst_conn.execute(
                            f"INSERT OR IGNORE INTO games ({columns}) VALUES ({placeholders})",
                            row
                        )
                        imported += 1
                    except AttributeError:
                        continue

                # Copy moves for new games
                if "moves" in src_tables and imported > 0:
                    src_cursor.execute("SELECT * FROM moves")
                    move_columns = [desc[0] for desc in src_cursor.description]
                    move_placeholders = ", ".join(["?"] * len(move_columns))
                    move_col_str = ", ".join(move_columns)

                    for row in src_cursor.fetchall():
                        game_id_idx = move_columns.index("game_id") if "game_id" in move_columns else 1
                        game_id = row[game_id_idx]
                        if game_id not in existing_ids:
                            try:
                                dst_conn.execute(
                                    f"INSERT OR IGNORE INTO moves ({move_col_str}) VALUES ({move_placeholders})",
                                    row
                                )
                            except AttributeError:
                                continue

                dst_conn.commit()

        return imported
