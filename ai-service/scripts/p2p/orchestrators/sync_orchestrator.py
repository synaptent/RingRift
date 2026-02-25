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

from scripts.p2p.db_helpers import p2p_db_connection
from scripts.p2p.orchestrators.base_orchestrator import BaseOrchestrator, HealthCheckResult
from scripts.p2p.constants import (
    P2P_DATA_SYNC_BASE,
    P2P_DATA_SYNC_MAX,
    P2P_DATA_SYNC_MIN,
    P2P_MODEL_SYNC_BASE,
    P2P_MODEL_SYNC_MAX,
    P2P_MODEL_SYNC_MIN,
    P2P_SYNC_BACKOFF_FACTOR,
    P2P_SYNC_SPEEDUP_FACTOR,
    P2P_TRAINING_DB_SYNC_BASE,
    P2P_TRAINING_DB_SYNC_MAX,
    P2P_TRAINING_DB_SYNC_MIN,
)

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

        # Adaptive sync interval tracking (Jan 30, 2026)
        self._init_adaptive_sync_intervals()

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
        """Check if source node needs disk cleanup after sync.

        Feb 1, 2026: Inlined from removed P2POrchestrator._should_cleanup_source().

        Args:
            node: NodeInfo of the source node

        Returns:
            True if source cleanup is allowed (disk usage >= threshold).
        """
        try:
            from scripts.p2p.managers.memory_disk_manager import DISK_CLEANUP_THRESHOLD
        except ImportError:
            DISK_CLEANUP_THRESHOLD = 80  # noqa: N806 - Fallback
        return getattr(node, "disk_percent", 0) >= DISK_CLEANUP_THRESHOLD

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
        # Feb 2026: Use lock-free PeerSnapshot instead of peers_lock to avoid
        # blocking the event loop. peers_lock contention was contributing to
        # /status endpoint taking 30-60+ seconds.
        try:
            node_id = getattr(self._p2p, "node_id", "")
            self_info = getattr(self._p2p, "self_info", None)
            peer_snapshot = getattr(self._p2p, "_peer_snapshot", None)

            if peer_snapshot:
                peers_dict = peer_snapshot.get_snapshot()
                job_distribution = {}
                for pid, peer in peers_dict.items():
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
                # Fallback to lock-based access if snapshot not available
                peers_lock = getattr(self._p2p, "peers_lock", None)
                peers = getattr(self._p2p, "peers", {})
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
                        if self_info:
                            job_distribution[node_id] = {
                                "selfplay_jobs": int(getattr(self_info, "selfplay_jobs", 0) or 0),
                                "training_jobs": int(getattr(self_info, "training_jobs", 0) or 0),
                                "gpu_percent": float(getattr(self_info, "gpu_percent", 0) or 0),
                            }
                    if job_distribution:
                        all_jobs = [d["selfplay_jobs"] for d in job_distribution.values()]
                        avg_jobs = sum(all_jobs) / len(all_jobs) if all_jobs else 0
                        result["cluster_job_distribution"] = {
                            "node_count": len(job_distribution),
                            "avg_selfplay_jobs": round(avg_jobs, 1),
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
            interval = self.get_adaptive_sync_interval("data")
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
        if self_info and getattr(self_info, "disk_percent", 0) > 90:  # DISK_PRODUCTION_HALT_PERCENT
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
                    self.record_sync_result_for_adaptive("data", success)  # ADAPTIVE INTERVAL

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
                self.record_sync_result_for_adaptive("data", False)  # ADAPTIVE: record failure
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
        # Phase 3.4 Dec 29, 2025: Use context managers to prevent connection leaks
        # Feb 2026: Use p2p_db_connection for centralized fd limiting
        with p2p_db_connection(validated_db) as src_conn, \
             p2p_db_connection(canonical_db) as dst_conn:

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

    async def collect_cluster_manifest(self):
        """Leader-only: Collect manifests from all peers and build cluster view.

        Jan 29, 2026: Moved from P2POrchestrator._collect_cluster_manifest().

        Returns:
            ClusterDataManifest with aggregated cluster data state
        """
        import asyncio
        import time
        from scripts.p2p.protocols import ClusterDataManifest, NodeDataManifest

        p2p = self._p2p
        cluster_manifest = ClusterDataManifest(
            collected_at=time.time(),
        )

        # Collect from self
        local_manifest = await asyncio.to_thread(p2p._collect_local_data_manifest)
        with p2p.manifest_lock:
            p2p.local_data_manifest = local_manifest
        cluster_manifest.node_manifests[p2p.node_id] = local_manifest

        # Collect from peers in parallel.
        # Only probe peers that are currently alive and not retired; terminated
        # or long-dead nodes should not stall manifest collection. NAT-blocked
        # peers can't accept inbound /data_manifest, so they are excluded too.
        peers_snapshot = p2p._peer_snapshot.get_snapshot()
        peers = [
            p
            for p in peers_snapshot.values()
            if p.is_alive()
            and not bool(getattr(p, "retired", False))
            and not bool(getattr(p, "nat_blocked", False))
        ]

        tasks = [p2p._request_peer_manifest(peer) for peer in peers]
        # Add timeout to prevent hang if peers are unresponsive
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=45.0
            )
        except asyncio.TimeoutError:
            self._log_warning(
                f"Manifest collection timed out after 45s collecting from {len(peers)} peers. "
                "Proceeding with partial data."
            )
            results = []

        for peer, result in zip(peers, results, strict=False):
            if isinstance(result, NodeDataManifest):
                cluster_manifest.node_manifests[peer.node_id] = result

        # Compute cluster-wide statistics
        cluster_manifest.total_nodes = len(cluster_manifest.node_manifests)

        all_files: set[str] = set()
        for node_id, node_manifest in cluster_manifest.node_manifests.items():
            cluster_manifest.total_files += node_manifest.total_files
            cluster_manifest.total_size_bytes += node_manifest.total_size_bytes
            cluster_manifest.total_selfplay_games += node_manifest.selfplay_games
            cluster_manifest.files_by_node[node_id] = node_manifest.total_files

            for file_info in node_manifest.files:
                all_files.add(file_info.path)

        cluster_manifest.unique_files = all_files

        # Find files missing from nodes (for sync planning)
        # Pre-build file path sets per node to avoid O(n^2) set reconstruction.
        # Previously rebuilt {f.path for f in files} for every (file, node) pair,
        # blocking the event loop for 16+ minutes with 14 nodes × hundreds of files.
        node_file_sets = {
            node_id: {f.path for f in node_manifest.files}
            for node_id, node_manifest in cluster_manifest.node_manifests.items()
        }
        for file_path in all_files:
            nodes_without_file = [
                node_id for node_id in cluster_manifest.node_manifests
                if file_path not in node_file_sets[node_id]
            ]
            if nodes_without_file:
                cluster_manifest.missing_from_nodes[file_path] = nodes_without_file

        # Collect external storage metadata (OWC drive, S3 bucket)
        try:
            external_storage = await p2p._collect_external_storage_metadata()
            cluster_manifest.external_storage = external_storage
        except Exception as e:
            self._log_debug(f"External storage scan skipped: {e}")

        self._log_info(
            f"Cluster manifest: {cluster_manifest.total_nodes} nodes, "
            f"{len(cluster_manifest.unique_files)} unique files, "
            f"{cluster_manifest.total_selfplay_games} total games"
        )

        return cluster_manifest

    async def p2p_training_db_sync(self) -> None:
        """DECENTRALIZED: Sync training databases via P2P for improved training diversity.

        Jan 29, 2026: Moved from P2POrchestrator._p2p_training_db_sync().

        TRAINING DB P2P SYNC: Ensures all nodes have access to consolidated training
        data without relying on leader-coordinated sync. Prioritizes:
        - canonical_*.db (canonical training data)
        - consolidated_training*.db (merged training data)
        - training_pool*.db (training pool databases)

        ADAPTIVE INTERVALS: faster during training, slower when idle.
        """
        import time
        from scripts.p2p.protocols import DataSyncJob

        p2p = self._p2p
        now = time.time()

        # ADAPTIVE INTERVAL: Uses activity-aware interval (faster during training)
        interval = self.get_adaptive_sync_interval("training_db")
        last_check = getattr(p2p, "_last_p2p_training_db_sync", 0)
        if now - last_check < interval:
            return
        p2p._last_p2p_training_db_sync = now

        # Skip if sync is in progress
        if getattr(p2p, "sync_in_progress", False):
            return

        # Skip if under disk pressure
        if getattr(p2p.self_info, "disk_percent", 0) > 80:
            return

        # Get our local files
        local_manifest = getattr(p2p, "local_data_manifest", None)
        if not local_manifest:
            return

        local_dbs = set()
        local_db_sizes = {}
        for file_info in getattr(local_manifest, "files", []) or []:
            rel_path = getattr(file_info, "relative_path", "")
            if rel_path.endswith(".db"):
                local_dbs.add(rel_path)
                local_db_sizes[rel_path] = getattr(file_info, "size_bytes", 0)

        # Check peer manifests for training databases
        peer_manifests = getattr(p2p, "_gossip_peer_manifests", {})
        if not peer_manifests:
            return

        # Find training databases we're missing or have smaller versions of
        missing_dbs: dict[str, list[tuple]] = {}

        for peer_id, peer_manifest in peer_manifests.items():
            if peer_id == p2p.node_id:
                continue

            # Check circuit breaker
            health = p2p._get_peer_health_score(peer_id)
            if health <= 0:
                continue

            peer_files = getattr(peer_manifest, "files", []) or []
            for file_info in peer_files:
                rel_path = getattr(file_info, "relative_path", "")
                size = getattr(file_info, "size_bytes", 0)

                # Only sync training-related databases and ELO database
                if not rel_path.endswith(".db"):
                    continue
                if not ("canonical_" in rel_path or "consolidated_training" in rel_path or
                        "training_pool" in rel_path or "unified_elo" in rel_path or
                        "elo_ratings" in rel_path):
                    continue

                # Skip empty databases
                if size < 1024:
                    continue

                # Check if we don't have it or have a smaller version
                local_size = local_db_sizes.get(rel_path, 0)
                if local_size >= size:
                    continue

                if peer_id not in missing_dbs:
                    missing_dbs[peer_id] = []
                missing_dbs[peer_id].append((rel_path, size, health))

        if not missing_dbs:
            return

        # Pick healthiest peer with training DBs
        best_peer = max(missing_dbs.keys(), key=lambda pid: p2p._get_peer_health_score(pid))
        dbs_to_sync = [db[0] for db in missing_dbs[best_peer][:3]]

        # Check if peer is alive
        with p2p.peers_lock:
            peer = p2p.peers.get(best_peer)
        if not peer or not peer.is_alive():
            return

        self._log_info(f"TRAINING DB SYNC: Requesting {len(dbs_to_sync)} training DBs from {best_peer}")

        try:
            import uuid
            job = DataSyncJob(
                job_id=f"traindb_{uuid.uuid4().hex[:8]}",
                source_node=best_peer,
                target_node=p2p.node_id,
                files=dbs_to_sync,
            )

            p2p.sync_in_progress = True
            try:
                success = await p2p._request_node_sync(job)
                p2p._record_p2p_sync_result(best_peer, success)
                self.record_sync_result_for_adaptive("training_db", success)
                if success:
                    self._log_info(f"TRAINING DB SYNC: Got {len(dbs_to_sync)} training DBs from {best_peer}")
            finally:
                p2p.sync_in_progress = False
        except Exception as e:
            self._log_info(f"TRAINING DB SYNC: Error: {e}")
            self.record_sync_result_for_adaptive("training_db", False)
            p2p.sync_in_progress = False

    async def p2p_model_sync(self) -> None:
        """DECENTRALIZED: Sync model files via P2P for faster model distribution.

        Jan 30, 2026: Moved from P2POrchestrator._p2p_model_sync().

        MODEL P2P SYNC: Ensures all nodes have access to latest trained models
        without relying on leader-coordinated sync. Prioritizes:
        - Newer models (by timestamp)
        - Models for active board configurations
        - NNUE models (smaller, faster to sync)
        - ADAPTIVE INTERVALS: faster during training, slower when idle
        """
        import time
        from scripts.p2p.protocols import DataSyncJob

        p2p = self._p2p
        now = time.time()

        # ADAPTIVE INTERVAL: Uses activity-aware interval (faster during training)
        interval = self.get_adaptive_sync_interval("model")
        last_check = getattr(p2p, "_last_p2p_model_sync", 0)
        if now - last_check < interval:
            return
        p2p._last_p2p_model_sync = now

        # Skip if sync in progress
        if getattr(p2p, "sync_in_progress", False):
            return

        # Get model files from local manifest
        local_manifest = getattr(p2p, "local_data_manifest", None)
        if not local_manifest:
            return

        local_models = set()
        for file_info in getattr(local_manifest, "files", []) or []:
            rel_path = getattr(file_info, "relative_path", "")
            if rel_path and ("models/" in rel_path or rel_path.endswith((".pt", ".onnx", ".bin"))):
                local_models.add(rel_path)

        # Check peer manifests for models we're missing
        peer_manifests = getattr(p2p, "_gossip_peer_manifests", {})
        missing_models: dict[str, list[str]] = {}

        for peer_id, peer_manifest in peer_manifests.items():
            if peer_id == p2p.node_id:
                continue

            health = p2p._get_peer_health_score(peer_id)
            if health <= 0:
                continue

            peer_files = getattr(peer_manifest, "files", []) or []
            for file_info in peer_files:
                rel_path = getattr(file_info, "relative_path", "")
                if not rel_path:
                    continue
                if not ("models/" in rel_path or rel_path.endswith((".pt", ".onnx", ".bin"))):
                    continue
                if rel_path in local_models:
                    continue

                if peer_id not in missing_models:
                    missing_models[peer_id] = []
                missing_models[peer_id].append(rel_path)

        if not missing_models:
            return

        # Pick healthiest peer with models
        best_peer = max(missing_models.keys(), key=lambda pid: p2p._get_peer_health_score(pid))
        models_to_sync = missing_models[best_peer][:5]  # Max 5 models per cycle

        with p2p.peers_lock:
            peer = p2p.peers.get(best_peer)
        if not peer or not peer.is_alive():
            return

        self._log_info(f"MODEL SYNC: Requesting {len(models_to_sync)} models from {best_peer}")

        try:
            import uuid
            job = DataSyncJob(
                job_id=f"model_{uuid.uuid4().hex[:8]}",
                source_node=best_peer,
                target_node=p2p.node_id,
                files=models_to_sync,
            )

            p2p.sync_in_progress = True
            try:
                success = await p2p._request_node_sync(job)
                p2p._record_p2p_sync_result(best_peer, success)
                self.record_sync_result_for_adaptive("model", success)
                if success:
                    self._log_info(f"MODEL SYNC: Got {len(models_to_sync)} models from {best_peer}")
            finally:
                p2p.sync_in_progress = False
        except Exception as e:
            self._log_info(f"MODEL SYNC: Error: {e}")
            self.record_sync_result_for_adaptive("model", False)
            p2p.sync_in_progress = False

    # ==========================================================================
    # ADAPTIVE SYNC INTERVALS (Jan 30, 2026)
    # ==========================================================================
    # Dynamically adjust sync intervals based on:
    # - Cluster activity (training = more frequent model sync)
    # - Success/failure streaks (failures = back off, successes = speed up)
    # - Data freshness (new data in cluster = more frequent sync)
    # ==========================================================================

    def _init_adaptive_sync_intervals(self):
        """Initialize adaptive sync interval tracking."""
        self._adaptive_intervals = {
            "data": P2P_DATA_SYNC_BASE,
            "model": P2P_MODEL_SYNC_BASE,
            "training_db": P2P_TRAINING_DB_SYNC_BASE,
        }
        self._sync_success_streak = {
            "data": 0,
            "model": 0,
            "training_db": 0,
        }
        self._sync_failure_streak = {
            "data": 0,
            "model": 0,
            "training_db": 0,
        }
        self._last_interval_adjustment = 0

    def get_adaptive_sync_interval(self, sync_type: str) -> float:
        """Get the current adaptive interval for a sync type.

        ADAPTIVE SYNC INTERVALS: Intervals adjust based on:
        1. Cluster activity (training = faster sync for models)
        2. Success rate (failures = back off)
        3. Base/min/max bounds per sync type

        Args:
            sync_type: One of "data", "model", "training_db"

        Returns:
            Current interval in seconds
        """
        if not hasattr(self, "_adaptive_intervals"):
            self._init_adaptive_sync_intervals()

        # Get current interval
        current = self._adaptive_intervals.get(sync_type, P2P_DATA_SYNC_BASE)

        # Apply activity-based adjustment
        activity_factor = self.calculate_cluster_activity_factor()

        # Get bounds for this sync type
        if sync_type == "data":
            min_interval = P2P_DATA_SYNC_MIN
            max_interval = P2P_DATA_SYNC_MAX
        elif sync_type == "model":
            min_interval = P2P_MODEL_SYNC_MIN
            max_interval = P2P_MODEL_SYNC_MAX
        elif sync_type == "training_db":
            min_interval = P2P_TRAINING_DB_SYNC_MIN
            max_interval = P2P_TRAINING_DB_SYNC_MAX
        else:
            min_interval = 120
            max_interval = 600

        # Apply activity factor (0.5-1.0 = active cluster, 1.0-2.0 = idle cluster)
        adjusted = current * activity_factor

        # Clamp to bounds
        return max(min_interval, min(max_interval, adjusted))

    def record_sync_result_for_adaptive(self, sync_type: str, success: bool):
        """Record sync result to adjust adaptive intervals.

        ADAPTIVE INTERVAL ADJUSTMENT:
        - On success: reduce interval (speed up) up to min
        - On failure: increase interval (back off) up to max

        Args:
            sync_type: One of "data", "model", "training_db"
            success: Whether sync succeeded
        """
        if not hasattr(self, "_adaptive_intervals"):
            self._init_adaptive_sync_intervals()

        if success:
            self._sync_success_streak[sync_type] = self._sync_success_streak.get(sync_type, 0) + 1
            self._sync_failure_streak[sync_type] = 0

            # After 3 consecutive successes, speed up
            if self._sync_success_streak[sync_type] >= 3:
                current = self._adaptive_intervals[sync_type]
                new_interval = current * P2P_SYNC_SPEEDUP_FACTOR

                # Get min bound
                if sync_type == "data":
                    min_interval = P2P_DATA_SYNC_MIN
                elif sync_type == "model":
                    min_interval = P2P_MODEL_SYNC_MIN
                else:
                    min_interval = P2P_TRAINING_DB_SYNC_MIN

                self._adaptive_intervals[sync_type] = max(min_interval, new_interval)
                self._sync_success_streak[sync_type] = 0  # Reset streak
        else:
            self._sync_failure_streak[sync_type] = self._sync_failure_streak.get(sync_type, 0) + 1
            self._sync_success_streak[sync_type] = 0

            # On any failure, back off
            current = self._adaptive_intervals[sync_type]
            new_interval = current * P2P_SYNC_BACKOFF_FACTOR

            # Get max bound
            if sync_type == "data":
                max_interval = P2P_DATA_SYNC_MAX
            elif sync_type == "model":
                max_interval = P2P_MODEL_SYNC_MAX
            else:
                max_interval = P2P_TRAINING_DB_SYNC_MAX

            self._adaptive_intervals[sync_type] = min(max_interval, new_interval)

    def get_sync_interval_summary(self) -> dict:
        """Get summary of current adaptive sync intervals for monitoring."""
        if not hasattr(self, "_adaptive_intervals"):
            self._init_adaptive_sync_intervals()

        return {
            "data_interval": round(self.get_adaptive_sync_interval("data")),
            "model_interval": round(self.get_adaptive_sync_interval("model")),
            "training_db_interval": round(self.get_adaptive_sync_interval("training_db")),
            "activity_factor": round(self.calculate_cluster_activity_factor(), 2),
            "data_streak": {
                "success": self._sync_success_streak.get("data", 0),
                "failure": self._sync_failure_streak.get("data", 0),
            },
            "model_streak": {
                "success": self._sync_success_streak.get("model", 0),
                "failure": self._sync_failure_streak.get("model", 0),
            },
        }
