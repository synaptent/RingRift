"""WorkerPullController: Worker-side work claiming from leader.

January 2026: Extracted from p2p_orchestrator.py for better modularity.
Handles the pull-based work model where workers claim work from the leader.

This controller manages:
- Claiming single or batch work items from the leader
- Reporting work completion/failure results
- Tracking claiming statistics
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from aiohttp import ClientTimeout
    from scripts.p2p.models import PeerInfo

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class WorkerPullConfig:
    """Configuration for WorkerPullController.

    Attributes:
        claim_timeout: Timeout for single work claim (seconds)
        batch_claim_timeout: Timeout for batch work claim (seconds)
        report_timeout: Timeout for work result reporting (seconds)
        max_batch_size: Maximum items to claim in a batch
    """

    claim_timeout: float = 15.0
    batch_claim_timeout: float = 20.0
    report_timeout: float = 15.0
    max_batch_size: int = 10


@dataclass
class WorkerPullStats:
    """Statistics for WorkerPullController operations."""

    claims_attempted: int = 0
    claims_successful: int = 0
    claims_failed: int = 0
    batch_claims_attempted: int = 0
    batch_claims_successful: int = 0
    batch_items_claimed: int = 0
    results_reported: int = 0
    results_failed: int = 0
    last_claim_time: float = 0.0
    last_work_from_leader: float = 0.0


# ============================================================================
# Singleton management
# ============================================================================

_instance: WorkerPullController | None = None


def get_worker_pull_controller() -> WorkerPullController | None:
    """Get the singleton WorkerPullController instance."""
    return _instance


def set_worker_pull_controller(controller: WorkerPullController) -> None:
    """Set the singleton WorkerPullController instance."""
    global _instance
    _instance = controller


def reset_worker_pull_controller() -> None:
    """Reset the singleton WorkerPullController instance (for testing)."""
    global _instance
    _instance = None


def create_worker_pull_controller(
    config: WorkerPullConfig | None = None,
    orchestrator: Any | None = None,
) -> WorkerPullController:
    """Create and register a WorkerPullController instance.

    Args:
        config: Optional configuration
        orchestrator: P2P orchestrator reference (for callbacks)

    Returns:
        The created WorkerPullController instance
    """
    controller = WorkerPullController(config=config, orchestrator=orchestrator)
    set_worker_pull_controller(controller)
    return controller


# ============================================================================
# WorkerPullController
# ============================================================================


class WorkerPullController:
    """Controller for worker-side work claiming from the leader.

    This class handles:
    - Claiming work items from the leader's work queue
    - Batch claiming for improved efficiency
    - Reporting work results back to the leader
    """

    def __init__(
        self,
        config: WorkerPullConfig | None = None,
        orchestrator: Any | None = None,
    ):
        """Initialize WorkerPullController.

        Args:
            config: Configuration for the controller
            orchestrator: P2P orchestrator reference (for callbacks)
        """
        self.config = config or WorkerPullConfig()
        self._orchestrator = orchestrator
        self._stats = WorkerPullStats()

    @property
    def stats(self) -> WorkerPullStats:
        """Get current statistics."""
        return self._stats

    @property
    def last_work_from_leader(self) -> float:
        """Get timestamp of last work received from leader."""
        return self._stats.last_work_from_leader

    def set_orchestrator(self, orchestrator: Any) -> None:
        """Set the P2P orchestrator reference.

        Called during orchestrator initialization.
        """
        self._orchestrator = orchestrator

    def _get_node_id(self) -> str:
        """Get this node's ID from orchestrator."""
        return getattr(self._orchestrator, "node_id", "unknown")

    def _get_leader_id(self) -> str | None:
        """Get current leader ID from orchestrator."""
        return getattr(self._orchestrator, "leader_id", None)

    def _get_leader_peer(self) -> Any | None:
        """Get current leader peer info from orchestrator."""
        leader_id = self._get_leader_id()
        if not leader_id:
            return None

        peers_lock = getattr(self._orchestrator, "peers_lock", None)
        peers = getattr(self._orchestrator, "peers", {})

        if peers_lock:
            with peers_lock:
                return peers.get(leader_id)
        return peers.get(leader_id)

    def _url_for_peer(self, peer: Any, endpoint: str) -> str:
        """Build URL for a peer endpoint."""
        if hasattr(self._orchestrator, "_url_for_peer"):
            return self._orchestrator._url_for_peer(peer, endpoint)
        # Fallback
        host = getattr(peer, "host", "localhost")
        port = getattr(peer, "port", 8770)
        return f"http://{host}:{port}{endpoint}"

    def _auth_headers(self) -> dict[str, str]:
        """Get authentication headers from orchestrator."""
        if hasattr(self._orchestrator, "_auth_headers"):
            return self._orchestrator._auth_headers()
        return {}

    # ========================================================================
    # Work claiming
    # ========================================================================

    async def claim_work_from_leader(
        self, capabilities: list[str]
    ) -> dict[str, Any] | None:
        """Claim work from the leader's work queue.

        Args:
            capabilities: List of work types this node can handle

        Returns:
            Work item dict if claimed, None otherwise
        """
        import aiohttp
        from scripts.p2p.connection_pool import get_client_session

        self._stats.claims_attempted += 1

        leader_id = self._get_leader_id()
        node_id = self._get_node_id()

        if not leader_id or leader_id == node_id:
            return None

        leader_peer = self._get_leader_peer()
        if not leader_peer:
            return None

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.claim_timeout)
            async with get_client_session(timeout) as session:
                caps_str = ",".join(capabilities)
                url = self._url_for_peer(
                    leader_peer,
                    f"/work/claim?node_id={node_id}&capabilities={caps_str}"
                )
                async with session.get(url, headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        # Update timestamp on any leader response
                        self._stats.last_work_from_leader = time.time()
                        self._stats.last_claim_time = time.time()
                        data = await resp.json()
                        if data.get("status") == "claimed":
                            self._stats.claims_successful += 1
                            return data.get("work")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to claim work from leader: {e}")
            self._stats.claims_failed += 1

        return None

    async def claim_work_batch_from_leader(
        self, capabilities: list[str], max_items: int | None = None
    ) -> list[dict[str, Any]]:
        """Claim multiple work items from the leader's work queue.

        Batch claiming reduces HTTP round-trips and improves GPU utilization.

        Args:
            capabilities: List of work types this node can handle
            max_items: Maximum number of items to claim (default from config)

        Returns:
            List of work items, empty list if none available or error
        """
        import aiohttp
        from scripts.p2p.connection_pool import get_client_session

        self._stats.batch_claims_attempted += 1

        leader_id = self._get_leader_id()
        node_id = self._get_node_id()

        if not leader_id or leader_id == node_id:
            return []

        leader_peer = self._get_leader_peer()
        if not leader_peer:
            return []

        max_items = min(max_items or self.config.max_batch_size, self.config.max_batch_size)

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.batch_claim_timeout)
            async with get_client_session(timeout) as session:
                caps_str = ",".join(capabilities)
                url = self._url_for_peer(
                    leader_peer,
                    f"/work/claim_batch?node_id={node_id}&capabilities={caps_str}&max_items={max_items}"
                )
                async with session.get(url, headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        # Update timestamp on any leader response
                        self._stats.last_work_from_leader = time.time()
                        self._stats.last_claim_time = time.time()
                        data = await resp.json()
                        if data.get("status") == "claimed" and data.get("items"):
                            items = data.get("items", [])
                            self._stats.batch_claims_successful += 1
                            self._stats.batch_items_claimed += len(items)
                            return items
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to batch claim work from leader: {e}")

        return []

    # ========================================================================
    # Result reporting
    # ========================================================================

    async def report_work_result(
        self, work_item: dict[str, Any], success: bool
    ) -> bool:
        """Report work completion/failure to the leader.

        Args:
            work_item: The work item that was processed. May contain a "result"
                key with detailed results (e.g., gauntlet win_rates, Elo data)
                that should be forwarded to the leader for downstream processing.
            success: Whether the work completed successfully

        Returns:
            True if report was sent, False otherwise
        """
        import aiohttp
        from scripts.p2p.connection_pool import get_client_session

        leader_id = self._get_leader_id()
        node_id = self._get_node_id()

        if not leader_id or leader_id == node_id:
            return False

        work_id = work_item.get("work_id", "")
        if not work_id:
            return False

        leader_peer = self._get_leader_peer()
        if not leader_peer:
            return False

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.report_timeout)
            async with get_client_session(timeout) as session:
                if success:
                    url = self._url_for_peer(leader_peer, "/work/complete")
                    # Feb 2026: Include actual work result data (gauntlet win_rates,
                    # Elo data, etc.) so the leader can emit proper EVALUATION_COMPLETED
                    # events for the auto-promotion pipeline.
                    # Previously only {"node_id": node_id} was sent, discarding all results.
                    result_data = work_item.get("result", {})
                    if not isinstance(result_data, dict):
                        result_data = {}
                    result_data["node_id"] = node_id
                    payload = {"work_id": work_id, "result": result_data}
                else:
                    url = self._url_for_peer(leader_peer, "/work/fail")
                    error_msg = work_item.get("error", "execution_failed")
                    payload = {"work_id": work_id, "error": error_msg}

                async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        logger.debug(
                            f"Reported work {work_id} result: {'success' if success else 'failed'}"
                        )
                        self._stats.results_reported += 1
                        return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to report work result: {e}")
            self._stats.results_failed += 1

        return False

    # ========================================================================
    # Health check
    # ========================================================================

    def health_check(self) -> dict[str, Any]:
        """Return health check information for DaemonManager integration.

        Returns:
            Dict with health status and statistics
        """
        now = time.time()

        # Check if we're receiving work from leader
        time_since_last_work = now - self._stats.last_work_from_leader if self._stats.last_work_from_leader > 0 else float("inf")

        # Determine overall health
        if time_since_last_work > 600:  # 10 minutes
            status = "STALE"
        elif self._stats.claims_failed > self._stats.claims_successful:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        return {
            "status": status,
            "stats": {
                "claims_attempted": self._stats.claims_attempted,
                "claims_successful": self._stats.claims_successful,
                "claims_failed": self._stats.claims_failed,
                "batch_claims_attempted": self._stats.batch_claims_attempted,
                "batch_claims_successful": self._stats.batch_claims_successful,
                "batch_items_claimed": self._stats.batch_items_claimed,
                "results_reported": self._stats.results_reported,
                "results_failed": self._stats.results_failed,
            },
            "last_work_from_leader": self._stats.last_work_from_leader,
            "last_claim_time": self._stats.last_claim_time,
            "time_since_last_work": time_since_last_work if self._stats.last_work_from_leader > 0 else None,
        }
