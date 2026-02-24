"""Sync target selection for cluster data replication.

Extracted from cluster_manifest.py (December 2025) to improve modularity
and testability. Selects optimal nodes for data replication based on
capacity, policy, and priority.

Usage:
    from app.distributed.sync_target_selector import SyncTargetSelector

    selector = SyncTargetSelector(
        capacity_manager=manifest._capacity_manager,
        config_manager=manifest._config_manager,
        registry=manifest._registry,
        connection_factory=manifest._connection,
        hosts_config=manifest._hosts_config,
    )

    # Get candidate nodes for replicating a game
    targets = selector.get_replication_targets("game-123", min_copies=2)

    # Check if a node can receive data
    if selector.can_receive_data("node-1", DataType.GAME):
        # Sync to node-1
        pass
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Generator

from app.config.thresholds import DISK_SYNC_TARGET_PERCENT

if TYPE_CHECKING:
    from app.distributed.cluster_config_manager import (
        ClusterConfigManager,
        NodeSyncPolicy as ConfigNodeSyncPolicy,
    )
    from app.distributed.data_location_registry import DataLocationRegistry
    from app.distributed.node_capacity_manager import (
        NodeCapacity,
        NodeCapacityManager,
    )

logger = logging.getLogger(__name__)

# Constants
REPLICATION_TARGET_COUNT = 2  # Default replication count


class DataType(str, Enum):
    """Types of data tracked in the manifest."""

    GAME = "game"
    MODEL = "model"
    NPZ = "npz"
    CHECKPOINT = "checkpoint"


@dataclass
class NodeSyncPolicy:
    """Sync policy for a node."""

    node_id: str
    receive_games: bool = True
    receive_models: bool = True
    receive_npz: bool = True
    max_disk_usage_percent: float = float(DISK_SYNC_TARGET_PERCENT)
    excluded: bool = False
    exclusion_reason: str = ""


@dataclass
class SyncCandidateNode:
    """A potential node for syncing data.

    Note: Renamed from SyncTarget (Dec 2025) to avoid collision with
    app.coordination.sync_constants.SyncTarget which is for SSH connection specs.
    """

    node_id: str
    priority: int = 0  # Higher = sync first
    reason: str = ""
    capacity: Any = None  # NodeCapacity | None


# Backwards compatibility alias
SyncTarget = SyncCandidateNode


class SyncTargetSelector:
    """Selects sync targets based on capacity, policy, and priority.

    This class handles:
    - Sync policy evaluation per node
    - Replication target selection with priority sorting
    - Data type permission checks
    - Priority computation based on role, capacity, and storage type

    Thread-safe via the connection factory pattern.
    """

    def __init__(
        self,
        capacity_manager: NodeCapacityManager,
        config_manager: ClusterConfigManager,
        registry: DataLocationRegistry,
        connection_factory: Callable[[], Generator[sqlite3.Connection, None, None]],
        hosts_config: dict[str, Any] | None = None,
    ):
        """Initialize the sync target selector.

        Args:
            capacity_manager: NodeCapacityManager for capacity queries
            config_manager: ClusterConfigManager for sync policies
            registry: DataLocationRegistry for finding game locations
            connection_factory: Context manager factory for database connections
            hosts_config: Optional hosts configuration dict
        """
        self._capacity = capacity_manager
        self._config_manager = config_manager
        self._registry = registry
        self._connection = connection_factory
        self._hosts_config = hosts_config or {}

    def set_hosts_config(self, hosts_config: dict[str, Any]) -> None:
        """Update the hosts configuration.

        Args:
            hosts_config: New hosts configuration dict
        """
        self._hosts_config = hosts_config

    def get_sync_policy(self, node_id: str) -> NodeSyncPolicy:
        """Get sync policy for a node.

        Args:
            node_id: Node identifier

        Returns:
            NodeSyncPolicy for the node
        """
        # Delegate to config manager and convert to local NodeSyncPolicy type
        config_policy = self._config_manager.get_sync_policy(node_id)
        return NodeSyncPolicy(
            node_id=config_policy.node_id,
            receive_games=config_policy.receive_games,
            receive_models=config_policy.receive_models,
            receive_npz=config_policy.receive_npz,
            max_disk_usage_percent=config_policy.max_disk_usage_percent,
            excluded=config_policy.excluded,
            exclusion_reason=config_policy.exclusion_reason,
        )

    def can_receive_data(self, node_id: str, data_type: DataType) -> bool:
        """Check if a node can receive a specific type of data.

        Args:
            node_id: Node identifier
            data_type: Type of data to sync

        Returns:
            True if node can receive this data type
        """
        policy = self.get_sync_policy(node_id)

        if policy.excluded:
            return False

        # Check capacity
        capacity = self._capacity.get_node_capacity(node_id)
        if capacity and capacity.usage_percent >= policy.max_disk_usage_percent:
            return False

        # Check data type permissions
        if data_type == DataType.GAME:
            return policy.receive_games
        elif data_type == DataType.MODEL:
            return policy.receive_models
        elif data_type == DataType.NPZ:
            return policy.receive_npz
        elif data_type == DataType.CHECKPOINT:
            # Checkpoints follow model policy - training nodes need them
            return policy.receive_models

        return True

    def get_replication_targets(
        self,
        game_id: str,
        min_copies: int = REPLICATION_TARGET_COUNT,
        exclude_nodes: list[str] | None = None,
    ) -> list[SyncCandidateNode]:
        """Get candidate nodes for replicating a game.

        Args:
            game_id: Game to replicate
            min_copies: Desired minimum copies
            exclude_nodes: Nodes to exclude

        Returns:
            List of SyncCandidateNode sorted by priority
        """
        exclude_nodes_set = set(exclude_nodes or [])

        # Get current locations
        current_locations = self._registry.find_game(game_id)
        current_nodes = {loc.node_id for loc in current_locations}
        exclude_nodes_set.update(current_nodes)

        # Need more copies?
        copies_needed = min_copies - len(current_nodes)
        if copies_needed <= 0:
            return []

        targets: list[SyncCandidateNode] = []

        # Find candidate nodes
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT node_id FROM node_capacity")
            all_nodes = [row[0] for row in cursor.fetchall()]

        for node_id in all_nodes:
            if node_id in exclude_nodes_set:
                continue

            # Check if can receive games
            if not self.can_receive_data(node_id, DataType.GAME):
                continue

            # Get capacity and compute priority
            capacity = self._capacity.get_node_capacity(node_id)
            priority = self._compute_sync_priority(node_id, capacity)

            reason = self._get_sync_reason(node_id, capacity)

            targets.append(
                SyncCandidateNode(
                    node_id=node_id,
                    priority=priority,
                    reason=reason,
                    capacity=capacity,
                )
            )

        # Sort by priority (highest first)
        targets.sort(key=lambda t: t.priority, reverse=True)

        return targets[:copies_needed]

    def _compute_sync_priority(
        self,
        node_id: str,
        capacity: Any,  # NodeCapacity | None
    ) -> int:
        """Compute sync priority for a node.

        Higher priority = sync first.

        Factors:
        - Training nodes get higher priority
        - Nodes with more free space get higher priority
        - Ephemeral nodes get lower priority (data may be lost)
        """
        priority = 50  # Base priority

        # Adjust based on role
        host_config = self._hosts_config.get(node_id, {})
        role = host_config.get("role", "selfplay")

        if "training" in role:
            priority += 20
        elif role == "coordinator":
            priority -= 30

        # Adjust based on capacity
        if capacity:
            # More free space = higher priority
            if capacity.free_percent > 50:
                priority += 10
            elif capacity.free_percent < 20:
                priority -= 20

        # Ephemeral hosts get lower priority (we should sync FROM them, not TO)
        if host_config.get("storage_type") == "ephemeral":
            priority -= 10

        return priority

    def _get_sync_reason(
        self,
        node_id: str,
        capacity: Any,  # NodeCapacity | None
    ) -> str:
        """Get human-readable reason for sync target selection."""
        host_config = self._hosts_config.get(node_id, {})
        role = host_config.get("role", "selfplay")

        reasons = []

        if "training" in role:
            reasons.append("training node")

        if capacity:
            reasons.append(f"{capacity.free_percent:.1f}% free")

        return ", ".join(reasons) if reasons else "available"
