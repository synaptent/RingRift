"""Sync Router - Intelligent Data Routing Based on Node Capabilities.

Routes data (games, models, NPZ files) to appropriate nodes based on:
- Node role (training, selfplay, coordinator)
- Disk capacity and usage limits
- Storage type (persistent, ephemeral, NFS)
- Network topology (same provider, NFS sharing)
- Exclusion rules

Usage:
    from app.coordination.sync_router import SyncRouter, get_sync_router

    router = get_sync_router()

    # Get sync targets for game data
    targets = router.get_sync_targets(data_type="game", board_type="hex8")

    # Check if a node should receive specific data
    if router.should_sync_to_node("gpu-node-1", data_type="game"):
        sync_to_node(...)
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.distributed.cluster_manifest import (
    ClusterManifest,
    DataType,
    NodeCapacity,
    NodeSyncPolicy,
    SyncTarget,
    get_cluster_manifest,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Data classes
    "SyncRoute",
    "NodeSyncCapability",
    # Main class
    "SyncRouter",
    # Singleton accessors
    "get_sync_router",
    "reset_sync_router",
]


@dataclass
class SyncRoute:
    """A sync route with source and target information."""
    source_node: str
    target_node: str
    data_type: DataType
    priority: int = 0
    reason: str = ""
    estimated_size_bytes: int = 0
    bandwidth_limit_mbps: int | None = None


@dataclass
class NodeSyncCapability:
    """Sync capabilities for a node."""
    node_id: str
    can_receive_games: bool = True
    can_receive_models: bool = True
    can_receive_npz: bool = True
    is_training_node: bool = False
    is_priority_node: bool = False
    is_ephemeral: bool = False
    shares_nfs: bool = False
    provider: str = "unknown"
    disk_usage_percent: float = 0.0
    available_gb: float = 0.0


class SyncRouter:
    """Routes data to nodes based on capabilities and policies.

    Integrates with ClusterManifest for:
    - Exclusion rules and sync policies
    - Disk capacity checks
    - Replication tracking
    """

    def __init__(
        self,
        config_path: Path | None = None,
        manifest: ClusterManifest | None = None,
    ):
        """Initialize the sync router.

        Args:
            config_path: Path to distributed_hosts.yaml
            manifest: ClusterManifest instance (uses singleton if None)
        """
        self.node_id = socket.gethostname()
        self._manifest = manifest or get_cluster_manifest()

        # Load host configuration
        self._hosts_config: dict[str, Any] = {}
        self._sync_routing: dict[str, Any] = {}
        self._node_capabilities: dict[str, NodeSyncCapability] = {}
        self._load_config(config_path)

        logger.info(f"SyncRouter initialized: {len(self._node_capabilities)} nodes")

    def _load_config(self, config_path: Path | None = None) -> None:
        """Load configuration from distributed_hosts.yaml."""
        if config_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            config_path = base_dir / "config" / "distributed_hosts.yaml"

        if not config_path.exists():
            logger.warning(f"No config found at {config_path}")
            return

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            self._hosts_config = config.get("hosts", {})
            self._sync_routing = config.get("sync_routing", {})

            # Build node capabilities from hosts config
            self._build_node_capabilities()

        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def _build_node_capabilities(self) -> None:
        """Build node capability information from config."""
        priority_hosts = set(self._sync_routing.get("priority_hosts", []))

        for host_name, host_config in self._hosts_config.items():
            role = host_config.get("role", "selfplay")
            gpu = host_config.get("gpu", "")

            # Determine provider
            provider = "unknown"
            if "lambda" in host_name.lower():
                provider = "lambda"
            elif "vast" in host_name.lower():
                provider = "vast"
            elif "hetzner" in host_name.lower():
                provider = "hetzner"
            elif "aws" in host_name.lower():
                provider = "aws"
            elif any(x in host_name.lower() for x in ["mac", "mbp"]):
                provider = "mac"

            # Check if shares NFS (Lambda nodes with same provider)
            shares_nfs = provider == "lambda"

            # Check if ephemeral (Vast.ai)
            is_ephemeral = provider == "vast"

            # Check if training node
            is_training = "training" in role or host_name in priority_hosts

            # Get sync policy from manifest
            policy = self._manifest.get_sync_policy(host_name)

            cap = NodeSyncCapability(
                node_id=host_name,
                can_receive_games=policy.receive_games,
                can_receive_models=policy.receive_models,
                can_receive_npz=policy.receive_npz,
                is_training_node=is_training,
                is_priority_node=host_name in priority_hosts,
                is_ephemeral=is_ephemeral,
                shares_nfs=shares_nfs,
                provider=provider,
            )

            self._node_capabilities[host_name] = cap

    def get_sync_targets(
        self,
        data_type: str | DataType,
        board_type: str | None = None,
        num_players: int | None = None,
        exclude_nodes: list[str] | None = None,
        max_targets: int = 10,
    ) -> list[SyncTarget]:
        """Get candidate nodes for syncing data.

        Args:
            data_type: Type of data ("game", "model", "npz")
            board_type: Optional board type filter
            num_players: Optional num_players filter
            exclude_nodes: Nodes to exclude
            max_targets: Maximum number of targets to return

        Returns:
            List of SyncTarget sorted by priority
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type)

        exclude = set(exclude_nodes or [])
        exclude.add(self.node_id)  # Don't sync to self

        targets: list[SyncTarget] = []

        for node_id, cap in self._node_capabilities.items():
            if node_id in exclude:
                continue

            # Check if node can receive this data type
            if not self._can_receive_data_type(cap, data_type):
                continue

            # Check disk capacity
            if not self._check_node_capacity(node_id):
                continue

            # Skip if NFS sharing applies (data already visible)
            if self._shares_storage_with(node_id):
                continue

            # Compute priority
            priority = self._compute_target_priority(cap, data_type)
            reason = self._get_target_reason(cap)

            targets.append(SyncTarget(
                node_id=node_id,
                priority=priority,
                reason=reason,
            ))

        # Sort by priority (highest first)
        targets.sort(key=lambda t: t.priority, reverse=True)

        return targets[:max_targets]

    def _can_receive_data_type(
        self,
        cap: NodeSyncCapability,
        data_type: DataType,
    ) -> bool:
        """Check if a node can receive a specific data type."""
        if data_type == DataType.GAME:
            return cap.can_receive_games
        elif data_type == DataType.MODEL:
            return cap.can_receive_models
        elif data_type == DataType.NPZ:
            return cap.can_receive_npz
        return False

    def _check_node_capacity(self, node_id: str) -> bool:
        """Check if a node has capacity for more data."""
        return self._manifest.can_receive_data(node_id, DataType.GAME)

    def _shares_storage_with(self, target_node: str) -> bool:
        """Check if current node shares storage with target node.

        If both nodes are Lambda nodes with NFS, they share storage
        and don't need to sync between each other.
        """
        my_cap = self._node_capabilities.get(self.node_id)
        target_cap = self._node_capabilities.get(target_node)

        if not my_cap or not target_cap:
            return False

        # Both Lambda nodes share NFS
        if my_cap.shares_nfs and target_cap.shares_nfs:
            return True

        return False

    def _compute_target_priority(
        self,
        cap: NodeSyncCapability,
        data_type: DataType,
    ) -> int:
        """Compute sync priority for a target node.

        Higher priority = sync first.
        """
        priority = 50  # Base priority

        # Training nodes get highest priority for game/NPZ data
        if cap.is_training_node and data_type in (DataType.GAME, DataType.NPZ):
            priority += 30

        # Priority nodes get bonus
        if cap.is_priority_node:
            priority += 20

        # Ephemeral nodes get lower priority for receiving data
        # (we should sync FROM them, not TO them)
        if cap.is_ephemeral:
            priority -= 15

        # Prefer nodes with more available space
        if cap.disk_usage_percent < 50:
            priority += 10
        elif cap.disk_usage_percent > 70:
            priority -= 20

        return priority

    def _get_target_reason(self, cap: NodeSyncCapability) -> str:
        """Get human-readable reason for target selection."""
        reasons = []

        if cap.is_training_node:
            reasons.append("training")
        if cap.is_priority_node:
            reasons.append("priority")
        if cap.is_ephemeral:
            reasons.append("ephemeral")

        return ", ".join(reasons) if reasons else "available"

    def should_sync_to_node(
        self,
        target_node: str,
        data_type: str | DataType,
        source_node: str | None = None,
    ) -> bool:
        """Check if data should be synced to a specific node.

        Args:
            target_node: Target node ID
            data_type: Type of data
            source_node: Source node (defaults to current node)

        Returns:
            True if sync should proceed
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type)

        source = source_node or self.node_id

        # Can't sync to self
        if target_node == source:
            return False

        # Check node capabilities
        cap = self._node_capabilities.get(target_node)
        if not cap:
            # Unknown node - use manifest policy
            return self._manifest.can_receive_data(target_node, data_type)

        # Check data type permissions
        if not self._can_receive_data_type(cap, data_type):
            return False

        # Check capacity
        if not self._check_node_capacity(target_node):
            return False

        # Check NFS sharing
        if self._shares_storage_with(target_node):
            return False

        return True

    def get_optimal_source(
        self,
        game_id: str,
        target_node: str,
    ) -> str | None:
        """Find the optimal source node for syncing a game.

        Considers:
        - Network proximity (same provider)
        - Load balancing
        - Bandwidth availability

        Args:
            game_id: Game to sync
            target_node: Destination node

        Returns:
            Best source node ID or None if not found
        """
        # Find locations for this game
        locations = self._manifest.find_game(game_id)
        if not locations:
            return None

        target_cap = self._node_capabilities.get(target_node)
        target_provider = target_cap.provider if target_cap else "unknown"

        candidates: list[tuple[str, int]] = []

        for loc in locations:
            source = loc.node_id
            if source == target_node:
                continue

            source_cap = self._node_capabilities.get(source)
            score = 50  # Base score

            if source_cap:
                # Prefer same provider (lower latency)
                if source_cap.provider == target_provider:
                    score += 20

                # Avoid ephemeral nodes as sources if possible
                if source_cap.is_ephemeral:
                    score -= 10

                # Prefer nodes with low disk usage (less IO contention)
                if source_cap.disk_usage_percent < 50:
                    score += 10

            candidates.append((source, score))

        if not candidates:
            return None

        # Return highest scoring source
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def plan_replication(
        self,
        game_id: str,
        min_copies: int = 2,
    ) -> list[SyncRoute]:
        """Plan replication routes for a game.

        Args:
            game_id: Game to replicate
            min_copies: Minimum number of copies desired

        Returns:
            List of SyncRoute describing the sync plan
        """
        routes: list[SyncRoute] = []

        # Get current locations
        locations = self._manifest.find_game(game_id)
        current_nodes = {loc.node_id for loc in locations}

        copies_needed = min_copies - len(current_nodes)
        if copies_needed <= 0:
            return routes

        # Get targets
        targets = self._manifest.get_replication_targets(
            game_id,
            min_copies=min_copies,
            exclude_nodes=list(current_nodes),
        )

        for target in targets:
            # Find best source
            source = self.get_optimal_source(game_id, target.node_id)
            if not source:
                source = list(current_nodes)[0] if current_nodes else self.node_id

            routes.append(SyncRoute(
                source_node=source,
                target_node=target.node_id,
                data_type=DataType.GAME,
                priority=target.priority,
                reason=target.reason,
            ))

        return routes

    def get_node_capability(self, node_id: str) -> NodeSyncCapability | None:
        """Get sync capability information for a node."""
        return self._node_capabilities.get(node_id)

    def get_status(self) -> dict[str, Any]:
        """Get router status."""
        return {
            "node_id": self.node_id,
            "total_nodes": len(self._node_capabilities),
            "training_nodes": sum(
                1 for c in self._node_capabilities.values() if c.is_training_node
            ),
            "priority_nodes": sum(
                1 for c in self._node_capabilities.values() if c.is_priority_node
            ),
            "ephemeral_nodes": sum(
                1 for c in self._node_capabilities.values() if c.is_ephemeral
            ),
            "nfs_nodes": sum(
                1 for c in self._node_capabilities.values() if c.shares_nfs
            ),
        }


# Module-level singleton
_sync_router: SyncRouter | None = None


def get_sync_router() -> SyncRouter:
    """Get the singleton SyncRouter instance."""
    global _sync_router
    if _sync_router is None:
        _sync_router = SyncRouter()
    return _sync_router


def reset_sync_router() -> None:
    """Reset the singleton (for testing)."""
    global _sync_router
    _sync_router = None


__all__ = [
    "SyncRoute",
    "SyncRouter",
    "NodeSyncCapability",
    "get_sync_router",
    "reset_sync_router",
]
