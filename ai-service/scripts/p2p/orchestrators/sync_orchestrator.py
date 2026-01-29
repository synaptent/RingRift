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

import logging
import time
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
