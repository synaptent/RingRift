"""Quorum-Aware Reconnection Manager for P2P Cluster.

Prioritizes reconnection to voter nodes when quorum is at risk, ensuring
the cluster can restore consensus quickly during network partitions.

December 30, 2025: Added for 48-hour autonomous operation reliability.
Prevents random reconnection order from delaying quorum restoration.

Usage:
    from app.coordination.quorum_recovery import (
        QuorumRecoveryManager,
        get_quorum_manager,
        ReconnectionPriority,
    )

    manager = get_quorum_manager()
    prioritized = manager.get_prioritized_reconnection_order(offline_nodes)

    for node_id in prioritized:
        await reconnect(node_id)

Integration with P2P Recovery Daemon:
    # In p2p_recovery_daemon.py
    from app.coordination.quorum_recovery import get_quorum_manager

    async def _attempt_recovery(self, offline_nodes: list[str]) -> None:
        manager = get_quorum_manager()
        prioritized = manager.get_prioritized_reconnection_order(offline_nodes)

        async for batch in manager.process_reconnections(prioritized, self._reconnect_node):
            logger.info(f"Reconnected batch: {batch}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable, AsyncIterator

from app.coordination.singleton_mixin import SingletonMixin

logger = logging.getLogger(__name__)


class ReconnectionPriority(IntEnum):
    """Priority levels for node reconnection.

    Lower numbers = higher priority (reconnected first).
    """

    VOTER_QUORUM_NEEDED = 0  # Voter node, and we need more for quorum
    VOTER_EXTRA = 1  # Voter node, but we already have quorum
    GPU_HIGH = 2  # GPU node with high-end GPU (H100, A100, GH200)
    GPU_STANDARD = 3  # GPU node with standard GPU (RTX, L40S)
    CPU_ONLY = 4  # CPU-only node


@dataclass
class NodeReconnectionInfo:
    """Information about a node pending reconnection."""

    node_id: str
    priority: ReconnectionPriority
    is_voter: bool = False
    has_gpu: bool = False
    gpu_type: str = ""
    last_seen: float = 0.0
    retry_count: int = 0


@dataclass
class QuorumRecoveryConfig:
    """Configuration for quorum recovery manager."""

    # Minimum voters needed for quorum
    quorum_size: int = 3

    # Maximum concurrent reconnection attempts
    max_concurrent_reconnections: int = 3

    # Delay between reconnection batches (seconds)
    batch_delay: float = 1.0

    # Maximum retries per node before giving up
    max_retries_per_node: int = 3

    # Reconnection timeout per node (seconds)
    reconnection_timeout: float = 30.0

    # High-end GPU types (prioritized for selfplay capacity)
    high_end_gpus: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "h100", "a100", "gh200", "l40s", "rtx5090", "rtx4090"
        })
    )

    @classmethod
    def from_defaults(cls) -> QuorumRecoveryConfig:
        """Create config from coordination defaults."""
        try:
            from app.config.coordination_defaults import (
                P2PDefaults,
                P2PRecoveryDefaults,
            )

            return cls(
                quorum_size=P2PDefaults.DEFAULT_QUORUM,
                max_concurrent_reconnections=getattr(
                    P2PRecoveryDefaults, "MAX_CONCURRENT_RECONNECTIONS", 3
                ),
            )
        except ImportError:
            return cls()


class QuorumRecoveryManager(SingletonMixin):
    """Prioritized reconnection manager for quorum restoration.

    Ensures voter nodes are reconnected first when quorum is at risk,
    followed by high-value GPU nodes, then standard nodes.

    Thread-safe singleton for use across the coordination layer.

    December 30, 2025: Added for 48-hour autonomous operation.
    """

    def __init__(self, config: QuorumRecoveryConfig | None = None):
        """Initialize the quorum recovery manager.

        Args:
            config: Optional configuration. If None, loads from defaults.
        """
        self._config = config or QuorumRecoveryConfig.from_defaults()
        self._voter_ids: set[str] = set()
        self._online_voters: set[str] = set()
        self._node_gpu_info: dict[str, str] = {}  # node_id -> gpu_type
        self._pending_reconnections: dict[str, NodeReconnectionInfo] = {}
        self._reconnection_in_progress: set[str] = set()
        self._last_quorum_check: float = 0.0
        self._stats = {
            "reconnections_attempted": 0,
            "reconnections_succeeded": 0,
            "reconnections_failed": 0,
            "quorum_restorations": 0,
        }

    # =========================================================================
    # Voter Management
    # =========================================================================

    def set_voters(self, voter_ids: list[str] | set[str]) -> None:
        """Set the list of voter node IDs.

        Args:
            voter_ids: List or set of node IDs that are voters.
        """
        self._voter_ids = set(voter_ids)
        logger.debug(f"QuorumRecovery: Set {len(self._voter_ids)} voters")

    def update_online_voters(self, online_voter_ids: list[str] | set[str]) -> None:
        """Update the set of currently online voters.

        Args:
            online_voter_ids: List or set of voter node IDs that are online.
        """
        self._online_voters = set(online_voter_ids) & self._voter_ids
        logger.debug(
            f"QuorumRecovery: {len(self._online_voters)}/{len(self._voter_ids)} voters online"
        )

    def register_node_gpu(self, node_id: str, gpu_type: str) -> None:
        """Register a node's GPU type for priority calculation.

        Args:
            node_id: The node identifier.
            gpu_type: GPU type string (e.g., "h100", "rtx4090", "none").
        """
        self._node_gpu_info[node_id] = gpu_type.lower()

    def load_from_cluster_config(self) -> None:
        """Load voter and GPU info from cluster configuration."""
        try:
            from app.config.cluster_config import (
                get_p2p_voters,
                load_cluster_config,
            )

            # Load voters
            voters = get_p2p_voters()
            self.set_voters(voters)

            # Load GPU info from hosts
            config = load_cluster_config()
            for host_id, host_info in config.hosts_raw.items():
                gpu = host_info.get("gpu", "none")
                if gpu and gpu != "none":
                    self._node_gpu_info[host_id] = gpu.lower()

            logger.info(
                f"QuorumRecovery: Loaded {len(self._voter_ids)} voters, "
                f"{len(self._node_gpu_info)} GPU nodes from config"
            )
        except Exception as e:
            logger.warning(f"QuorumRecovery: Failed to load cluster config: {e}")

    # =========================================================================
    # Quorum State
    # =========================================================================

    def has_quorum(self) -> bool:
        """Check if we currently have quorum.

        Returns:
            True if online voters >= quorum size.
        """
        return len(self._online_voters) >= self._config.quorum_size

    def needs_more_voters(self) -> bool:
        """Check if we need more voters to achieve/maintain quorum.

        Returns:
            True if we need more voters.
        """
        return len(self._online_voters) < self._config.quorum_size

    def voters_needed_for_quorum(self) -> int:
        """Get number of additional voters needed for quorum.

        Returns:
            Number of voters needed (0 if quorum is met).
        """
        needed = self._config.quorum_size - len(self._online_voters)
        return max(0, needed)

    # =========================================================================
    # Priority Calculation
    # =========================================================================

    def get_reconnection_priority(self, node_id: str) -> ReconnectionPriority:
        """Get the reconnection priority for a node.

        Priority order:
        1. Voter nodes when quorum is needed (highest)
        2. Voter nodes when quorum is met
        3. High-end GPU nodes (H100, A100, GH200, L40S, RTX5090/4090)
        4. Standard GPU nodes
        5. CPU-only nodes (lowest)

        Args:
            node_id: The node identifier.

        Returns:
            ReconnectionPriority enum value.
        """
        is_voter = node_id in self._voter_ids
        gpu_type = self._node_gpu_info.get(node_id, "none").lower()
        has_gpu = gpu_type and gpu_type != "none"

        if is_voter:
            if self.needs_more_voters():
                return ReconnectionPriority.VOTER_QUORUM_NEEDED
            return ReconnectionPriority.VOTER_EXTRA

        if has_gpu:
            if gpu_type in self._config.high_end_gpus:
                return ReconnectionPriority.GPU_HIGH
            return ReconnectionPriority.GPU_STANDARD

        return ReconnectionPriority.CPU_ONLY

    def get_prioritized_reconnection_order(
        self, offline_nodes: list[str] | set[str]
    ) -> list[str]:
        """Get nodes in prioritized reconnection order.

        Args:
            offline_nodes: List or set of offline node IDs.

        Returns:
            List of node IDs sorted by reconnection priority.
        """
        nodes_with_priority = []

        for node_id in offline_nodes:
            priority = self.get_reconnection_priority(node_id)
            nodes_with_priority.append((node_id, priority))

        # Sort by priority (lower = higher priority)
        nodes_with_priority.sort(key=lambda x: x[1])

        prioritized = [node_id for node_id, _ in nodes_with_priority]

        if prioritized:
            # Log priority breakdown
            voter_first = sum(
                1 for _, p in nodes_with_priority
                if p <= ReconnectionPriority.VOTER_EXTRA
            )
            gpu_nodes = sum(
                1 for _, p in nodes_with_priority
                if p in (ReconnectionPriority.GPU_HIGH, ReconnectionPriority.GPU_STANDARD)
            )
            logger.debug(
                f"QuorumRecovery: Prioritized {len(prioritized)} nodes "
                f"(voters={voter_first}, gpu={gpu_nodes})"
            )

        return prioritized

    # =========================================================================
    # Reconnection Processing
    # =========================================================================

    async def process_reconnections(
        self,
        offline_nodes: list[str] | set[str],
        reconnect_fn: Callable[[str], Awaitable[bool]],
        max_concurrent: int | None = None,
    ) -> AsyncIterator[list[str]]:
        """Process reconnections in priority order with concurrency control.

        Yields batches of successfully reconnected node IDs.

        Args:
            offline_nodes: Nodes to reconnect.
            reconnect_fn: Async function that attempts reconnection, returns True on success.
            max_concurrent: Maximum concurrent attempts (default from config).

        Yields:
            List of successfully reconnected node IDs per batch.
        """
        max_concurrent = max_concurrent or self._config.max_concurrent_reconnections
        prioritized = self.get_prioritized_reconnection_order(offline_nodes)

        if not prioritized:
            return

        logger.info(
            f"QuorumRecovery: Processing {len(prioritized)} reconnections "
            f"(max_concurrent={max_concurrent})"
        )

        # Process in batches
        for i in range(0, len(prioritized), max_concurrent):
            batch = prioritized[i : i + max_concurrent]
            batch_results = []

            # Create reconnection tasks
            tasks = []
            for node_id in batch:
                self._reconnection_in_progress.add(node_id)
                task = asyncio.create_task(
                    self._reconnect_with_timeout(node_id, reconnect_fn)
                )
                tasks.append((node_id, task))

            # Wait for batch to complete
            for node_id, task in tasks:
                try:
                    success = await task
                    self._stats["reconnections_attempted"] += 1

                    if success:
                        batch_results.append(node_id)
                        self._stats["reconnections_succeeded"] += 1

                        # Update online voters if this was a voter
                        if node_id in self._voter_ids:
                            self._online_voters.add(node_id)
                            if self.has_quorum():
                                self._stats["quorum_restorations"] += 1
                                logger.info("QuorumRecovery: Quorum restored!")
                    else:
                        self._stats["reconnections_failed"] += 1

                except Exception as e:
                    logger.warning(f"QuorumRecovery: Reconnection error for {node_id}: {e}")
                    self._stats["reconnections_failed"] += 1
                finally:
                    self._reconnection_in_progress.discard(node_id)

            if batch_results:
                yield batch_results

            # Delay between batches
            if i + max_concurrent < len(prioritized):
                await asyncio.sleep(self._config.batch_delay)

    async def _reconnect_with_timeout(
        self,
        node_id: str,
        reconnect_fn: Callable[[str], Awaitable[bool]],
    ) -> bool:
        """Attempt reconnection with timeout.

        Args:
            node_id: Node to reconnect.
            reconnect_fn: Async reconnection function.

        Returns:
            True if reconnection succeeded.
        """
        try:
            return await asyncio.wait_for(
                reconnect_fn(node_id),
                timeout=self._config.reconnection_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"QuorumRecovery: Reconnection timeout for {node_id}")
            return False
        except Exception as e:
            logger.warning(f"QuorumRecovery: Reconnection failed for {node_id}: {e}")
            return False

    # =========================================================================
    # Status and Health
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get current quorum recovery status.

        Returns:
            Dict with quorum state and statistics.
        """
        return {
            "has_quorum": self.has_quorum(),
            "online_voters": len(self._online_voters),
            "total_voters": len(self._voter_ids),
            "quorum_size": self._config.quorum_size,
            "voters_needed": self.voters_needed_for_quorum(),
            "reconnections_in_progress": len(self._reconnection_in_progress),
            "stats": dict(self._stats),
        }

    def health_check(self) -> dict[str, Any]:
        """Health check for daemon manager integration.

        Returns:
            Health check result dict.
        """
        try:
            from app.coordination.contracts import HealthCheckResult

            status = self.get_status()
            has_quorum = status["has_quorum"]

            if has_quorum:
                return HealthCheckResult.healthy(
                    f"Quorum met ({status['online_voters']}/{status['quorum_size']})",
                    **status,
                )
            else:
                return HealthCheckResult.degraded(
                    f"Quorum not met ({status['online_voters']}/{status['quorum_size']}), "
                    f"need {status['voters_needed']} more voters",
                    **status,
                )
        except ImportError:
            status = self.get_status()
            return {
                "healthy": status["has_quorum"],
                "status": "healthy" if status["has_quorum"] else "degraded",
                "details": status,
            }


# =============================================================================
# Singleton Accessors
# =============================================================================


def get_quorum_manager() -> QuorumRecoveryManager:
    """Get the singleton QuorumRecoveryManager instance.

    Returns:
        The global QuorumRecoveryManager instance.
    """
    manager = QuorumRecoveryManager.get_instance()

    # Auto-load config on first access
    if not manager._voter_ids:
        try:
            manager.load_from_cluster_config()
        except Exception as e:
            logger.debug(f"QuorumRecovery: Could not auto-load config: {e}")

    return manager


def reset_quorum_manager() -> None:
    """Reset the singleton instance (for testing)."""
    QuorumRecoveryManager.reset_instance()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "QuorumRecoveryManager",
    "QuorumRecoveryConfig",
    "ReconnectionPriority",
    "NodeReconnectionInfo",
    "get_quorum_manager",
    "reset_quorum_manager",
]
