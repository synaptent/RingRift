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
