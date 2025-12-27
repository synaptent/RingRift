"""NodeSelector: Node ranking and selection for job dispatch.

Extracted from p2p_orchestrator.py for better modularity.
Handles node ranking by GPU/CPU power and selection for various tasks.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Callable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..models import NodeInfo

# Default number of training nodes to select
TRAINING_NODE_COUNT = 5


class NodeSelector:
    """Selects and ranks nodes for job dispatch based on capabilities.

    Responsibilities:
    - Rank nodes by GPU processing power for training
    - Rank nodes by CPU processing power for data processing
    - Select best node for specific tasks (training, gauntlet)
    - Filter nodes by health, availability, and capabilities

    Usage:
        selector = NodeSelector(
            get_peers=lambda: orchestrator.peers,
            get_self_info=lambda: orchestrator.self_info,
            peers_lock=orchestrator.peers_lock,
        )

        # Get top GPU nodes for training
        training_nodes = selector.get_training_primary_nodes(count=5)

        # Get best GPU node for training
        best_gpu = selector.get_best_gpu_node_for_training()
    """

    def __init__(
        self,
        get_peers: Callable[[], dict[str, "NodeInfo"]],
        get_self_info: Callable[[], "NodeInfo"],
        peers_lock: threading.Lock | threading.RLock | None = None,
        get_training_jobs: Callable[[], dict[str, Any]] | None = None,
    ):
        """Initialize the NodeSelector.

        Args:
            get_peers: Callable that returns the current peers dict
            get_self_info: Callable that returns the current node's NodeInfo
            peers_lock: Optional lock for thread-safe peer access
            get_training_jobs: Optional callable that returns training jobs dict
        """
        self._get_peers = get_peers
        self._get_self_info = get_self_info
        self._peers_lock = peers_lock
        self._get_training_jobs = get_training_jobs
        # Track nodes marked as unhealthy via events (Dec 2025)
        self._unhealthy_nodes: set[str] = set()
        self._unhealthy_reasons: dict[str, str] = {}
        # December 2025: Initialize subscription state to prevent AttributeError
        self._subscribed = False
        self._subscription_lock = threading.Lock()

    def _get_all_nodes(self, include_self: bool = True) -> list["NodeInfo"]:
        """Get all nodes including self if requested."""
        if self._peers_lock:
            with self._peers_lock:
                nodes = list(self._get_peers().values())
        else:
            nodes = list(self._get_peers().values())

        if include_self:
            self_info = self._get_self_info()
            if self_info:
                nodes.append(self_info)

        return nodes

    # =========================================================================
    # GPU Node Selection
    # =========================================================================

    def get_training_primary_nodes(
        self, count: int = TRAINING_NODE_COUNT
    ) -> list["NodeInfo"]:
        """Get the top N nodes by GPU power for training priority.

        Returns nodes sorted by GPU processing power (highest first).
        These nodes should receive selfplay data first for training.

        Args:
            count: Number of nodes to return

        Returns:
            List of NodeInfo objects, sorted by GPU power (highest first)
        """
        all_nodes = self._get_all_nodes(include_self=True)

        # Filter to only GPU nodes that are alive and healthy
        gpu_nodes = [
            node
            for node in all_nodes
            if node.has_gpu and node.is_alive() and node.gpu_power_score() > 0
        ]

        # Sort by GPU power score (descending)
        gpu_nodes.sort(key=lambda n: n.gpu_power_score(), reverse=True)

        return gpu_nodes[:count]

    def get_training_nodes_ranked(self) -> list[dict[str, Any]]:
        """Get all GPU nodes with their power rankings for dashboard display.

        Returns:
            List of dicts with node info and power rankings
        """
        all_nodes = self._get_all_nodes(include_self=True)

        result = []
        for node in all_nodes:
            if node.has_gpu:
                result.append({
                    "node_id": node.node_id,
                    "gpu_name": node.gpu_name,
                    "gpu_power_score": node.gpu_power_score(),
                    "memory_gb": node.memory_gb,
                    "is_alive": node.is_alive(),
                    "is_healthy": node.is_healthy(),
                    "gpu_percent": node.gpu_percent,
                })

        # Sort by power score
        result.sort(key=lambda x: x["gpu_power_score"], reverse=True)
        return result

    def get_best_gpu_node_for_training(
        self, exclude_node_ids: set[str] | None = None
    ) -> "NodeInfo | None":
        """Get the best GPU node for neural network training.

        Prioritizes:
        1. GPU power score (H100 > GH200 > A10 > consumer GPUs)
        2. Low current load
        3. Not already running training

        Args:
            exclude_node_ids: Optional set of node IDs to exclude

        Returns:
            Best NodeInfo for training, or None if no suitable node
        """
        all_nodes = self._get_all_nodes(include_self=True)

        # Filter to GPU nodes that are healthy and not retired
        gpu_nodes = [
            n
            for n in all_nodes
            if n.has_gpu
            and n.is_alive()
            and n.is_healthy()
            and not getattr(n, "retired", False)
            and n.gpu_power_score() > 0
        ]

        if not gpu_nodes:
            return None

        # Exclude specified nodes (e.g., nodes already running training)
        if exclude_node_ids:
            available = [n for n in gpu_nodes if n.node_id not in exclude_node_ids]
            candidates = available if available else gpu_nodes
        else:
            # Check training jobs if getter is provided
            if self._get_training_jobs:
                training_jobs = self._get_training_jobs()
                nodes_with_training = {
                    j.worker_node
                    for j in training_jobs.values()
                    if hasattr(j, "worker_node")
                    and hasattr(j, "status")
                    and j.status in ("running", "queued")
                }
                available = [
                    n for n in gpu_nodes if n.node_id not in nodes_with_training
                ]
                candidates = available if available else gpu_nodes
            else:
                candidates = gpu_nodes

        # Sort by GPU power (descending), then load (ascending)
        candidates.sort(key=lambda n: (-n.gpu_power_score(), n.get_load_score()))
        return candidates[0] if candidates else None

    # =========================================================================
    # CPU Node Selection
    # =========================================================================

    def get_cpu_primary_nodes(self, count: int = 3) -> list["NodeInfo"]:
        """Get the top N nodes by CPU power for CPU-intensive tasks.

        Returns nodes sorted by CPU processing power (highest first).
        These nodes should receive CPU-intensive work like NPZ export,
        data aggregation, etc. Vast nodes are strongly preferred.

        Args:
            count: Number of nodes to return

        Returns:
            List of NodeInfo objects, sorted by CPU power (highest first)
        """
        all_nodes = self._get_all_nodes(include_self=True)

        # Filter to only alive and healthy nodes with CPU info
        cpu_nodes = [
            node
            for node in all_nodes
            if node.is_alive() and node.is_healthy() and node.cpu_power_score() > 0
        ]

        # Sort by CPU power score (descending) - vast nodes will rank highest
        cpu_nodes.sort(key=lambda n: (-n.cpu_power_score(), n.get_load_score()))

        return cpu_nodes[:count]

    def get_cpu_nodes_ranked(self) -> list[dict[str, Any]]:
        """Get all nodes with their CPU power rankings for dashboard display.

        Returns:
            List of dicts with node info and CPU power rankings
        """
        all_nodes = self._get_all_nodes(include_self=True)

        result = []
        for node in all_nodes:
            if node.cpu_count and node.cpu_count > 0:
                result.append({
                    "node_id": node.node_id,
                    "cpu_count": node.cpu_count,
                    "cpu_power_score": node.cpu_power_score(),
                    "cpu_percent": node.cpu_percent,
                    "memory_gb": node.memory_gb,
                    "is_alive": node.is_alive(),
                    "is_healthy": node.is_healthy(),
                    "has_gpu": node.has_gpu,
                })

        # Sort by CPU power score (descending)
        result.sort(key=lambda x: x["cpu_power_score"], reverse=True)
        return result

    def get_best_cpu_node_for_gauntlet(self) -> "NodeInfo | None":
        """Get the best CPU node for gauntlet/tournament work.

        Prioritizes Vast instances with high CPU count (200+ vCPUs).
        Gauntlets are CPU-bound and benefit from massive parallelism.

        Returns:
            Best NodeInfo for gauntlet, or None if no suitable node
        """
        all_nodes = self._get_all_nodes(include_self=True)

        # Filter to healthy nodes with high CPU count
        cpu_nodes = [
            n
            for n in all_nodes
            if n.is_alive()
            and n.is_healthy()
            and not getattr(n, "retired", False)
            and n.cpu_power_score() > 0
        ]

        if not cpu_nodes:
            return None

        # Strongly prefer Vast nodes (identified by "vast" in node_id or high CPU count)
        vast_nodes = [
            n
            for n in cpu_nodes
            if "vast" in n.node_id.lower() or n.cpu_count >= 64
        ]

        # Use vast nodes if available, otherwise fall back to any CPU node
        candidates = vast_nodes if vast_nodes else cpu_nodes

        # Sort by CPU power (descending), then load (ascending)
        candidates.sort(key=lambda n: (-n.cpu_power_score(), n.get_load_score()))
        return candidates[0] if candidates else None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_alive_gpu_nodes(self) -> list["NodeInfo"]:
        """Get all alive GPU nodes.

        Returns:
            List of NodeInfo for alive GPU nodes
        """
        all_nodes = self._get_all_nodes(include_self=True)
        return [n for n in all_nodes if n.has_gpu and n.is_alive()]

    def get_alive_nodes(self) -> list["NodeInfo"]:
        """Get all alive nodes.

        Returns:
            List of NodeInfo for alive nodes
        """
        all_nodes = self._get_all_nodes(include_self=True)
        return [n for n in all_nodes if n.is_alive()]

    def get_healthy_nodes(self) -> list["NodeInfo"]:
        """Get all healthy nodes.

        Returns:
            List of NodeInfo for healthy nodes
        """
        all_nodes = self._get_all_nodes(include_self=True)
        return [n for n in all_nodes if n.is_healthy()]

    def count_alive_peers(self) -> int:
        """Count alive peers (excluding self).

        Returns:
            Number of alive peers
        """
        if self._peers_lock:
            with self._peers_lock:
                peers = self._get_peers()
        else:
            peers = self._get_peers()

        return sum(1 for p in peers.values() if p.is_alive())

    # =========================================================================
    # Health State Management (Dec 2025)
    # =========================================================================

    def mark_node_unhealthy(self, node_id: str, reason: str = "") -> None:
        """Mark a node as unhealthy via event notification.

        Args:
            node_id: The ID of the unhealthy node
            reason: Optional reason for the unhealthy state
        """
        self._unhealthy_nodes.add(node_id)
        if reason:
            self._unhealthy_reasons[node_id] = reason

    def mark_node_healthy(self, node_id: str) -> None:
        """Mark a node as healthy (recovered).

        Args:
            node_id: The ID of the recovered node
        """
        self._unhealthy_nodes.discard(node_id)
        self._unhealthy_reasons.pop(node_id, None)

    def is_node_healthy(self, node_id: str) -> bool:
        """Check if a node is marked as healthy.

        Args:
            node_id: The ID of the node to check

        Returns:
            True if node is not in the unhealthy set
        """
        return node_id not in self._unhealthy_nodes

    def get_unhealthy_nodes(self) -> dict[str, str]:
        """Get all unhealthy nodes with reasons.

        Returns:
            Dict mapping node_id to reason
        """
        return {
            node_id: self._unhealthy_reasons.get(node_id, "")
            for node_id in self._unhealthy_nodes
        }

    # =========================================================================
    # Event Subscription (December 27, 2025)
    # =========================================================================

    def subscribe_to_events(self) -> None:
        """Subscribe to health-related events to track unhealthy nodes.

        December 27, 2025: Added to populate _unhealthy_nodes from events.
        This allows NodeSelector to filter out unhealthy nodes during selection.
        Uses double-check locking to prevent race conditions.
        """
        # Fast path - already subscribed
        if self._subscribed:
            return

        # Slow path with lock to prevent race conditions
        with self._subscription_lock:
            # Double-check after acquiring lock
            if self._subscribed:
                return

            try:
                from app.coordination.event_router import get_event_bus
                from app.distributed.data_events import DataEventType

                bus = get_event_bus()

                # Subscribe to HOST_OFFLINE to mark nodes as unhealthy
                if hasattr(DataEventType, "HOST_OFFLINE"):
                    bus.subscribe(DataEventType.HOST_OFFLINE, self._on_host_offline)
                    logger.info("[NodeSelector] Subscribed to HOST_OFFLINE")

                # Subscribe to NODE_RECOVERED to clear unhealthy status
                if hasattr(DataEventType, "NODE_RECOVERED"):
                    bus.subscribe(DataEventType.NODE_RECOVERED, self._on_node_recovered)
                    logger.info("[NodeSelector] Subscribed to NODE_RECOVERED")

                # Subscribe to HOST_ONLINE to clear offline nodes
                if hasattr(DataEventType, "HOST_ONLINE"):
                    bus.subscribe(DataEventType.HOST_ONLINE, self._on_host_online)
                    logger.info("[NodeSelector] Subscribed to HOST_ONLINE")

                self._subscribed = True
            except ImportError:
                logger.debug("[NodeSelector] Event router not available")
                self._subscribed = False  # Explicit reset on failure
            except (RuntimeError, AttributeError) as e:
                logger.warning(f"[NodeSelector] Failed to subscribe: {e}")
                self._subscribed = False  # Explicit reset on failure

    async def _on_host_offline(self, event) -> None:
        """Handle HOST_OFFLINE events - mark node as unhealthy."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            node_id = payload.get("node_id", "")
            reason = payload.get("reason", "host_offline")

            if not node_id:
                return

            self._unhealthy_nodes.add(node_id)
            self._unhealthy_reasons[node_id] = reason
            logger.info(f"[NodeSelector] Marked {node_id} as unhealthy: {reason}")

        except (AttributeError, KeyError, TypeError) as e:
            logger.debug(f"[NodeSelector] Error handling host offline: {e}")

    async def _on_host_online(self, event) -> None:
        """Handle HOST_ONLINE events - clear unhealthy status."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            node_id = payload.get("node_id", "")

            if not node_id:
                return

            if node_id in self._unhealthy_nodes:
                self._unhealthy_nodes.discard(node_id)
                self._unhealthy_reasons.pop(node_id, None)
                logger.info(f"[NodeSelector] Cleared unhealthy status for {node_id}")

        except (AttributeError, KeyError, TypeError) as e:
            logger.debug(f"[NodeSelector] Error handling host online: {e}")

    async def _on_node_recovered(self, event) -> None:
        """Handle NODE_RECOVERED events - clear unhealthy status."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            node_id = payload.get("node_id", "") or payload.get("host", "")

            if not node_id:
                return

            if node_id in self._unhealthy_nodes:
                self._unhealthy_nodes.discard(node_id)
                self._unhealthy_reasons.pop(node_id, None)
                logger.info(f"[NodeSelector] Cleared unhealthy status for recovered node {node_id}")

        except (AttributeError, KeyError, TypeError) as e:
            logger.debug(f"[NodeSelector] Error handling node recovered: {e}")

    # =========================================================================
    # Health Check (December 2025)
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Check health status of NodeSelector.

        Returns:
            Dict with status, node counts, and error info
        """
        status = "healthy"
        errors_count = 0
        last_error: str | None = None

        # Get node counts
        all_nodes = self._get_all_nodes(include_self=True)
        alive_nodes = [n for n in all_nodes if n.is_alive()]
        healthy_nodes = [n for n in all_nodes if n.is_healthy()]
        gpu_nodes = [n for n in alive_nodes if n.has_gpu]

        # Check for issues
        unhealthy_count = len(self._unhealthy_nodes)
        if unhealthy_count > 0:
            errors_count = unhealthy_count
            reasons = list(self._unhealthy_reasons.values())
            last_error = reasons[0] if reasons else f"{unhealthy_count} nodes marked unhealthy"

        # Degrade if too few nodes
        if len(alive_nodes) == 0:
            status = "unhealthy"
            last_error = "No alive nodes available"
            errors_count = 1
        elif len(gpu_nodes) == 0 and len(all_nodes) > 0:
            status = "degraded"
            if not last_error:
                last_error = "No GPU nodes available"
        elif unhealthy_count > len(all_nodes) // 2:
            status = "degraded"

        return {
            "status": status,
            "operations_count": len(all_nodes),
            "errors_count": errors_count,
            "last_error": last_error,
            "total_nodes": len(all_nodes),
            "alive_nodes": len(alive_nodes),
            "healthy_nodes": len(healthy_nodes),
            "gpu_nodes": len(gpu_nodes),
            "unhealthy_node_ids": list(self._unhealthy_nodes),
        }
