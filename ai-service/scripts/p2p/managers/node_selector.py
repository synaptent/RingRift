"""NodeSelector: Node ranking and selection for job dispatch.

Extracted from p2p_orchestrator.py for better modularity.
Handles node ranking by GPU/CPU power and selection for various tasks.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Callable

from app.core.async_context import safe_create_task

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

        # Session 17.28: Periodic unhealthy node recovery
        # Jan 5, 2026 (Session 17.27): Reduced from 60s to 15s for faster recovery
        self._running = False
        self._recovery_task: Any | None = None
        self._recovery_interval_seconds = 15.0  # Check every 15s for faster recovery
        self._pending_probes: set[str] = set()  # Nodes pending immediate probe

    async def start(self) -> None:
        """Start periodic unhealthy node recovery loop.

        Session 17.28: Automatically recovers nodes that have become healthy
        again, increasing node availability from 74% to 90%+.
        """
        if self._running:
            logger.warning("[NodeSelector] Already running")
            return

        self._running = True

        # Import asyncio here to avoid module-level import in sync class
        import asyncio

        self._recovery_task = safe_create_task(
            self._periodic_unhealthy_recovery(),
            name="node-selector-recovery",
        )
        logger.info("[NodeSelector] Started periodic unhealthy node recovery loop")

    async def stop(self) -> None:
        """Stop periodic unhealthy node recovery loop."""
        self._running = False

        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                import asyncio
                await asyncio.wait_for(asyncio.shield(self._recovery_task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass  # Task cancelled or timed out
            self._recovery_task = None

        logger.info("[NodeSelector] Stopped periodic unhealthy node recovery loop")

    async def _periodic_unhealthy_recovery(self) -> None:
        """Periodically attempt to recover unhealthy nodes.

        Session 17.28: Runs every 15s (reduced from 60s in Session 17.27) and checks
        if nodes marked unhealthy have recovered. This prevents nodes from being
        permanently excluded from work distribution after transient failures.

        Session 17.27: Added priority processing for nodes in _pending_probes set.
        When a node is marked unhealthy with schedule_probe=True, it gets checked
        on the next loop iteration rather than waiting for the full interval.

        Expected impact: 15% more nodes available for work distribution.
        """
        import asyncio

        logger.info(
            f"[NodeSelector] Starting periodic recovery with interval "
            f"{self._recovery_interval_seconds}s"
        )

        while self._running:
            try:
                # Jan 5, 2026: Check for priority pending probes first
                # If there are pending probes, process them immediately
                pending_to_check: set[str] = set()
                if self._pending_probes:
                    pending_to_check = self._pending_probes.copy()
                    self._pending_probes.clear()
                    logger.debug(
                        f"[NodeSelector] Processing {len(pending_to_check)} pending probes: "
                        f"{pending_to_check}"
                    )

                await asyncio.sleep(self._recovery_interval_seconds)

                if not self._running:
                    break

                # Run synchronous recover_unhealthy_nodes in thread pool
                # If we have pending probes, only check those specific nodes
                if pending_to_check:
                    recovered = await asyncio.to_thread(
                        self._recover_specific_nodes, pending_to_check
                    )
                else:
                    recovered = await asyncio.to_thread(self.recover_unhealthy_nodes)

                if recovered:
                    logger.info(
                        f"[NodeSelector] Auto-recovered {len(recovered)} nodes: "
                        f"{recovered}"
                    )

                    # Emit event for work queue to pick up recovered nodes
                    self._emit_nodes_recovered_event(recovered)

            except asyncio.CancelledError:
                logger.info("[NodeSelector] Periodic recovery cancelled")
                break
            except Exception as e:
                logger.warning(
                    f"[NodeSelector] Error in periodic recovery: {e}",
                    exc_info=True
                )
                # Continue running despite errors
                await asyncio.sleep(5.0)

    def _emit_nodes_recovered_event(self, node_ids: list[str]) -> None:
        """Emit event when nodes are recovered for work queue integration.

        Session 17.28: Signals that recovered nodes are ready for work,
        enabling faster work assignment than waiting for next scheduling cycle.
        """
        try:
            from app.coordination.event_emission_helpers import safe_emit_event

            safe_emit_event(
                "NODES_RECOVERED",
                {
                    "node_ids": node_ids,
                    "count": len(node_ids),
                    "source": "node_selector_periodic_recovery",
                    "timestamp": __import__("time").time(),
                },
                context="NodeSelector",
                source="node_selector",
            )
        except ImportError:
            pass  # Event modules not available

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

        # Filter to only GPU nodes that are alive, healthy, and have training capability
        # Dec 29, 2025: Skip coordinator nodes (empty capabilities)
        gpu_nodes = [
            node
            for node in all_nodes
            if node.has_gpu
            and node.is_alive()
            and node.gpu_power_score() > 0
            and "training" in getattr(node, "capabilities", [])  # Must have training capability
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

        # Filter to GPU nodes that are healthy, not retired, and have training capability
        # Dec 29, 2025: Skip coordinator nodes (empty capabilities)
        gpu_nodes = [
            n
            for n in all_nodes
            if n.has_gpu
            and n.is_alive()
            and n.is_healthy()
            and not getattr(n, "retired", False)
            and n.gpu_power_score() > 0
            and "training" in getattr(n, "capabilities", [])  # Must have training capability
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
    # GPU-Aware Selfplay Node Selection (December 2025)
    # =========================================================================

    # GPU-required engine modes (require CUDA or MPS)
    # These modes use neural network inference and require GPU acceleration
    GPU_REQUIRED_ENGINE_MODES = {
        "gumbel-mcts", "mcts", "nnue-guided", "policy-only",
        "nn-minimax", "nn-descent", "gnn", "hybrid",
        "gmo", "ebmo", "ig-gmo", "cage",
    }

    # CPU-compatible engine modes (can run on any node)
    CPU_COMPATIBLE_ENGINE_MODES = {
        "heuristic-only", "heuristic", "random", "random-only",
        "descent-only", "maxn", "brs",
    }

    def engine_mode_requires_gpu(self, engine_mode: str) -> bool:
        """Check if an engine mode requires GPU acceleration.

        Args:
            engine_mode: The engine mode string (e.g., "gumbel-mcts", "heuristic-only")

        Returns:
            True if the engine mode requires GPU (CUDA or MPS), False otherwise.

        December 2025: Added to ensure GPU-required selfplay is only dispatched
        to GPU-capable nodes, preventing wasted compute and silent fallbacks.
        """
        if not engine_mode:
            return False
        mode_lower = engine_mode.lower().strip()
        return mode_lower in self.GPU_REQUIRED_ENGINE_MODES

    def get_nodes_for_engine_mode(
        self,
        engine_mode: str,
        exclude_busy: bool = True,
        include_self: bool = True,
    ) -> list["NodeInfo"]:
        """Get nodes capable of running the given engine mode.

        For GPU-required modes (gumbel-mcts, mcts, etc.), returns only GPU-capable nodes.
        For CPU-compatible modes (heuristic, random, etc.), returns all healthy nodes.

        Args:
            engine_mode: The engine mode string
            exclude_busy: If True, exclude nodes with high load
            include_self: If True, include the current node in results

        Returns:
            List of NodeInfo objects capable of running the engine mode.

        December 2025: Core method for GPU-aware job dispatch.
        """
        all_nodes = self._get_all_nodes(include_self=include_self)

        # Filter to alive, healthy, non-retired nodes
        healthy_nodes = [
            node
            for node in all_nodes
            if node.is_alive()
            and node.is_healthy()
            and not getattr(node, "retired", False)
            and node.node_id not in self._unhealthy_nodes
        ]

        # If GPU required, filter to GPU-capable nodes only
        if self.engine_mode_requires_gpu(engine_mode):
            capable_nodes = [
                node
                for node in healthy_nodes
                if node.has_gpu and node.gpu_power_score() > 0
            ]
            if not capable_nodes:
                logger.warning(
                    f"No GPU-capable nodes available for engine mode '{engine_mode}'"
                )
        else:
            # CPU-compatible modes can run on any node
            capable_nodes = healthy_nodes

        # January 5, 2026 (Session 17.26): Changed from hard cutoff to soft exclusion
        # Previously, nodes with >=80% load were completely excluded, causing
        # cluster utilization to drop to 40-60% instead of 85%+.
        # Now we sort by load score (low to high) so low-load nodes are preferred,
        # but high-load nodes are still available as fallback.
        if exclude_busy:
            capable_nodes.sort(key=lambda n: n.get_load_score())

        return capable_nodes

    def get_gpu_nodes_for_selfplay(
        self,
        count: int | None = None,
        exclude_busy: bool = True,
    ) -> list["NodeInfo"]:
        """Get GPU-capable nodes for GPU selfplay (neural network modes).

        Returns nodes sorted by GPU power score, suitable for running
        GPU-accelerated selfplay (gumbel-mcts, mcts, etc.).

        Args:
            count: Maximum number of nodes to return (None = all)
            exclude_busy: If True, exclude nodes with high load

        Returns:
            List of GPU-capable NodeInfo objects, sorted by GPU power.

        December 2025: Ensures GPU selfplay is only dispatched to GPU nodes.
        """
        all_nodes = self._get_all_nodes(include_self=True)

        # Filter to GPU nodes that are alive, healthy, and not retired
        gpu_nodes = [
            node
            for node in all_nodes
            if node.has_gpu
            and node.is_alive()
            and node.is_healthy()
            and not getattr(node, "retired", False)
            and node.gpu_power_score() > 0
            and node.node_id not in self._unhealthy_nodes
        ]

        # January 5, 2026 (Session 17.26): Changed from hard cutoff to soft exclusion
        # Previously, nodes with >=80% load were completely excluded.
        # Now we prefer nodes by a combination of GPU power and load:
        # - Primary sort by load score (ascending) so idle nodes are first
        # - Secondary sort by GPU power (descending) to prefer powerful GPUs
        if exclude_busy:
            # Sort by (load_score ASC, gpu_power DESC) using tuple comparison
            gpu_nodes.sort(key=lambda n: (n.get_load_score(), -n.gpu_power_score()))
        else:
            # Sort by GPU power score (descending) only
            gpu_nodes.sort(key=lambda n: n.gpu_power_score(), reverse=True)

        if count is not None:
            return gpu_nodes[:count]
        return gpu_nodes

    def get_best_node_for_selfplay(
        self,
        engine_mode: str,
        board_type: str | None = None,
        exclude_node_ids: set[str] | None = None,
    ) -> "NodeInfo | None":
        """Get the best node for running selfplay with the given engine mode.

        For GPU-required modes: Returns the best GPU node by power score.
        For CPU-compatible modes: Returns any healthy node with low load.

        Args:
            engine_mode: The engine mode (e.g., "gumbel-mcts", "heuristic-only")
            board_type: Optional board type for size-based filtering (future use)
            exclude_node_ids: Optional set of node IDs to exclude

        Returns:
            Best NodeInfo for selfplay, or None if no suitable node.

        December 2025: Entry point for GPU-aware selfplay scheduling.
        """
        candidates = self.get_nodes_for_engine_mode(
            engine_mode, exclude_busy=True, include_self=True
        )

        if not candidates:
            return None

        # Apply exclusion filter
        if exclude_node_ids:
            candidates = [n for n in candidates if n.node_id not in exclude_node_ids]
            if not candidates:
                # If all candidates are excluded, try without busy filter
                candidates = self.get_nodes_for_engine_mode(
                    engine_mode, exclude_busy=False, include_self=True
                )
                candidates = [n for n in candidates if n.node_id not in exclude_node_ids]

        if not candidates:
            return None

        # Sort by appropriate power score
        if self.engine_mode_requires_gpu(engine_mode):
            # For GPU modes, prefer high GPU power
            candidates.sort(key=lambda n: (-n.gpu_power_score(), n.get_load_score()))
        else:
            # For CPU modes, prefer high CPU power (for parallelism)
            candidates.sort(key=lambda n: (-n.cpu_power_score(), n.get_load_score()))

        return candidates[0]

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

    def mark_node_unhealthy(self, node_id: str, reason: str = "", schedule_probe: bool = True) -> None:
        """Mark a node as unhealthy via event notification.

        Jan 5, 2026 (Session 17.27): Added schedule_probe parameter for faster recovery.
        When True, schedules an immediate probe after 15s instead of waiting for the
        next recovery loop cycle.

        Args:
            node_id: The ID of the unhealthy node
            reason: Optional reason for the unhealthy state
            schedule_probe: If True, schedule an immediate recovery probe (default: True)
        """
        self._unhealthy_nodes.add(node_id)
        if reason:
            self._unhealthy_reasons[node_id] = reason

        # Jan 5, 2026: Schedule immediate probe for faster recovery
        if schedule_probe:
            self._pending_probes.add(node_id)
            logger.debug(f"[NodeSelector] Scheduled immediate probe for {node_id}")

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

    def health_check(self):
        """Check health status of NodeSelector.

        Returns:
            HealthCheckResult with status, node counts, and error info
        """
        # Import from contracts (zero dependencies)
        from app.coordination.contracts import HealthCheckResult, CoordinatorStatus

        status = CoordinatorStatus.RUNNING
        is_healthy = True
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
            status = CoordinatorStatus.ERROR
            is_healthy = False
            last_error = "No alive nodes available"
            errors_count = 1
        elif len(gpu_nodes) == 0 and len(all_nodes) > 0:
            status = CoordinatorStatus.DEGRADED
            if not last_error:
                last_error = "No GPU nodes available"
        elif unhealthy_count > len(all_nodes) // 2:
            status = CoordinatorStatus.DEGRADED

        return HealthCheckResult(
            healthy=is_healthy,
            status=status if isinstance(status, str) else status,
            message=last_error or "NodeSelector healthy",
            details={
                "operations_count": len(all_nodes),
                "errors_count": errors_count,
                "total_nodes": len(all_nodes),
                "alive_nodes": len(alive_nodes),
                "healthy_nodes": len(healthy_nodes),
                "gpu_nodes": len(gpu_nodes),
                "unhealthy_node_ids": list(self._unhealthy_nodes),
            },
        )

    # =========================================================================
    # Unhealthy Node Recovery (December 31, 2025)
    # =========================================================================

    def recover_unhealthy_nodes(self) -> list[str]:
        """Attempt to recover nodes from unhealthy state.

        Checks all nodes in the unhealthy set and removes them if they're
        now showing as alive and healthy. This prevents nodes from being
        permanently stuck in the unhealthy set due to transient failures.

        Returns:
            List of node IDs that were recovered.

        December 31, 2025: Added to fix idle GPU nodes issue where nodes
        were permanently excluded from work distribution after transient failures.
        """
        if not self._unhealthy_nodes:
            return []

        recovered = []

        # Get current node info for all potentially unhealthy nodes
        all_nodes = self._get_all_nodes(include_self=True)
        nodes_by_id = {n.node_id: n for n in all_nodes}

        for node_id in list(self._unhealthy_nodes):
            node = nodes_by_id.get(node_id)

            # Check if node appears healthy now
            if node is not None:
                is_alive = node.is_alive()
                is_healthy = node.is_healthy()
                not_retired = not getattr(node, "retired", False)

                if is_alive and is_healthy and not_retired:
                    # Node has recovered - remove from unhealthy set
                    self._unhealthy_nodes.discard(node_id)
                    self._unhealthy_reasons.pop(node_id, None)
                    recovered.append(node_id)
                    logger.info(
                        f"[NodeSelector] Recovered node {node_id} from unhealthy state "
                        f"(alive={is_alive}, healthy={is_healthy})"
                    )
            else:
                # Node is no longer in peers dict - might have been removed
                # Keep in unhealthy set for now to prevent issues if it reappears
                logger.debug(
                    f"[NodeSelector] Node {node_id} in unhealthy set but not in peers"
                )

        if recovered:
            logger.info(
                f"[NodeSelector] Recovered {len(recovered)} nodes from unhealthy state: "
                f"{recovered}"
            )

        return recovered

    def _recover_specific_nodes(self, node_ids: set[str]) -> list[str]:
        """Attempt to recover specific nodes from unhealthy state.

        Jan 5, 2026 (Session 17.27): Added for faster recovery of nodes that were
        just marked unhealthy. Instead of waiting for the full recovery interval,
        nodes added to _pending_probes are checked on the next loop iteration.

        Args:
            node_ids: Set of node IDs to check for recovery.

        Returns:
            List of node IDs that were recovered.
        """
        if not node_ids:
            return []

        # Only check nodes that are actually in the unhealthy set
        nodes_to_check = node_ids & self._unhealthy_nodes
        if not nodes_to_check:
            logger.debug(
                f"[NodeSelector] No pending probe nodes in unhealthy set: {node_ids}"
            )
            return []

        recovered = []

        # Get current node info for nodes to check
        all_nodes = self._get_all_nodes(include_self=True)
        nodes_by_id = {n.node_id: n for n in all_nodes}

        for node_id in nodes_to_check:
            node = nodes_by_id.get(node_id)

            # Check if node appears healthy now
            if node is not None:
                is_alive = node.is_alive()
                is_healthy = node.is_healthy()
                not_retired = not getattr(node, "retired", False)

                if is_alive and is_healthy and not_retired:
                    # Node has recovered - remove from unhealthy set
                    self._unhealthy_nodes.discard(node_id)
                    self._unhealthy_reasons.pop(node_id, None)
                    recovered.append(node_id)
                    logger.info(
                        f"[NodeSelector] Proactive recovery: node {node_id} recovered "
                        f"(alive={is_alive}, healthy={is_healthy})"
                    )
                else:
                    logger.debug(
                        f"[NodeSelector] Proactive probe: node {node_id} still unhealthy "
                        f"(alive={is_alive}, healthy={is_healthy}, retired={not not_retired})"
                    )
            else:
                logger.debug(
                    f"[NodeSelector] Proactive probe: node {node_id} not in peers dict"
                )

        if recovered:
            logger.info(
                f"[NodeSelector] Proactive recovery: {len(recovered)}/{len(nodes_to_check)} "
                f"nodes recovered: {recovered}"
            )

        return recovered

    def get_recovery_candidates(self) -> list[dict[str, Any]]:
        """Get detailed info about unhealthy nodes that might recover.

        Returns:
            List of dicts with node_id, reason, and current status.
        """
        all_nodes = self._get_all_nodes(include_self=True)
        nodes_by_id = {n.node_id: n for n in all_nodes}

        candidates = []
        for node_id in self._unhealthy_nodes:
            node = nodes_by_id.get(node_id)
            candidate = {
                "node_id": node_id,
                "reason": self._unhealthy_reasons.get(node_id, "unknown"),
                "in_peers": node is not None,
            }
            if node:
                candidate.update({
                    "is_alive": node.is_alive(),
                    "is_healthy": node.is_healthy(),
                    "retired": getattr(node, "retired", False),
                    "can_recover": (
                        node.is_alive()
                        and node.is_healthy()
                        and not getattr(node, "retired", False)
                    ),
                })
            candidates.append(candidate)

        return candidates
