"""Replication Manager - handles sync target selection and replication policies.

Extracted from ClusterManifest to improve maintainability.

December 2025 - ClusterManifest decomposition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from app.distributed.registries.base import BaseRegistry
from app.distributed.registries.node_inventory import NodeCapacity, MAX_DISK_USAGE_PERCENT

logger = logging.getLogger(__name__)

# Constants
REPLICATION_TARGET_COUNT = 2


class DataType(str, Enum):
    """Types of data tracked in the manifest."""
    GAME = "game"
    MODEL = "model"
    NPZ = "npz"
    CHECKPOINT = "checkpoint"


class NodeRole(str, Enum):
    """Node roles for sync targeting."""
    TRAINING = "training"
    SELFPLAY = "selfplay"
    COORDINATOR = "coordinator"
    STORAGE = "storage"
    EXCLUDED = "excluded"


@dataclass
class NodeSyncPolicy:
    """Sync policy for a node."""
    node_id: str
    receive_games: bool = True
    receive_models: bool = True
    receive_npz: bool = True
    max_disk_usage_percent: float = MAX_DISK_USAGE_PERCENT
    excluded: bool = False
    exclusion_reason: str = ""


@dataclass
class SyncCandidateNode:
    """A potential node for syncing data."""
    node_id: str
    priority: int = 0  # Higher = sync first
    reason: str = ""
    capacity: NodeCapacity | None = None


class ReplicationManager(BaseRegistry):
    """Manages sync target selection and replication policies.

    Determines where data should be replicated based on:
    - Node roles and capabilities
    - Disk capacity
    - Exclusion rules from configuration
    """

    def __init__(
        self,
        config_path: Path | None = None,
        node_inventory_manager=None,
        game_registry=None,
        **kwargs,
    ):
        """Initialize the replication manager.

        Args:
            config_path: Path to distributed_hosts.yaml
            node_inventory_manager: Optional NodeInventoryManager
            game_registry: Optional GameLocationRegistry
            **kwargs: Arguments passed to BaseRegistry
        """
        super().__init__(**kwargs)
        self._inventory_manager = node_inventory_manager
        self._game_registry = game_registry

        # Configuration
        self._hosts_config: dict[str, Any] = {}
        self._sync_policies: dict[str, NodeSyncPolicy] = {}
        self._priority_hosts: set[str] = set()
        self._max_disk_usage = MAX_DISK_USAGE_PERCENT

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: Path) -> None:
        """Load host configuration and build sync policies."""
        from app.config.cluster_config import load_cluster_config

        try:
            cluster_cfg = load_cluster_config(config_path)
            self._hosts_config = cluster_cfg.hosts_raw
            self._build_sync_policies_from_cluster_config(cluster_cfg)

        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def _build_sync_policies(self, config: dict[str, Any]) -> None:
        """Build node sync policies from configuration."""
        hosts = config.get("hosts", {})
        sync_routing = config.get("sync_routing", {})

        # Read max disk usage from config
        self._max_disk_usage = sync_routing.get(
            "max_disk_usage_percent", MAX_DISK_USAGE_PERCENT
        )

        # Auto-sync exclusion
        auto_sync = config.get("auto_sync", {})
        exclude_hosts = set(auto_sync.get("exclude_hosts", []))

        # Process excluded_hosts with detailed policies
        excluded_host_policies: dict[str, dict] = {}
        for entry in sync_routing.get("excluded_hosts", []):
            if isinstance(entry, dict):
                name = entry.get("name", "")
                if name:
                    exclude_hosts.add(name)
                    excluded_host_policies[name] = entry
            else:
                exclude_hosts.add(entry)

        # External storage overrides
        external_storage: dict[str, dict] = {}
        for entry in sync_routing.get("allowed_external_storage", []):
            if isinstance(entry, dict):
                host = entry.get("host", "")
                if host:
                    external_storage[host] = entry

        # Priority hosts
        self._priority_hosts = set(sync_routing.get("priority_hosts", []))

        # Build policies
        for host_name, host_config in hosts.items():
            policy = self._build_policy_for_host(
                host_name,
                host_config,
                exclude_hosts,
                excluded_host_policies,
                external_storage,
            )
            self._sync_policies[host_name] = policy

    def _build_sync_policies_from_cluster_config(self, cluster_cfg) -> None:
        """Build node sync policies from ClusterConfig object.

        Uses the consolidated cluster config helper instead of raw dict parsing.
        """
        from app.config.cluster_config import ClusterConfig

        if not isinstance(cluster_cfg, ClusterConfig):
            logger.warning("Invalid cluster config type, falling back to dict parsing")
            return

        hosts = cluster_cfg.hosts_raw
        sync_routing = cluster_cfg.sync_routing
        auto_sync = cluster_cfg.auto_sync

        # Read max disk usage from config
        self._max_disk_usage = sync_routing.max_disk_usage_percent

        # Auto-sync exclusion
        exclude_hosts = set(auto_sync.exclude_hosts)

        # Process excluded_hosts with detailed policies
        excluded_host_policies: dict[str, dict] = {}
        for host_name in sync_routing.excluded_hosts:
            exclude_hosts.add(host_name)

        # External storage overrides
        external_storage: dict[str, dict] = {}
        for storage in sync_routing.allowed_external_storage:
            external_storage[storage.host] = {
                "host": storage.host,
                "path": storage.path,
                "receive_games": storage.receive_games,
                "receive_npz": storage.receive_npz,
                "receive_models": storage.receive_models,
                "subdirs": storage.subdirs,
            }

        # Priority hosts
        self._priority_hosts = set(sync_routing.priority_hosts)

        # Build policies
        for host_name, host_config in hosts.items():
            policy = self._build_policy_for_host(
                host_name,
                host_config,
                exclude_hosts,
                excluded_host_policies,
                external_storage,
            )
            self._sync_policies[host_name] = policy

    def _build_policy_for_host(
        self,
        host_name: str,
        host_config: dict,
        exclude_hosts: set[str],
        excluded_policies: dict[str, dict],
        external_storage: dict[str, dict],
    ) -> NodeSyncPolicy:
        """Build sync policy for a single host."""
        role = host_config.get("role", "selfplay")

        policy = NodeSyncPolicy(
            node_id=host_name,
            max_disk_usage_percent=self._max_disk_usage,
        )

        # Coordinator nodes
        if role == "coordinator":
            policy.receive_games = False
            policy.receive_npz = False
            policy.receive_models = True
            policy.exclusion_reason = "coordinator node"

        # Explicit exclusion with detailed policy
        if host_name in excluded_policies:
            entry = excluded_policies[host_name]
            policy.receive_games = entry.get("receive_games", False)
            policy.receive_npz = entry.get("receive_npz", False)
            policy.receive_models = entry.get("receive_models", True)
            policy.exclusion_reason = entry.get("reason", "explicitly excluded")
        elif host_name in exclude_hosts:
            policy.receive_games = False
            policy.receive_npz = False
            policy.receive_models = True
            policy.exclusion_reason = "explicitly excluded"

        # Disabled nodes
        if not host_config.get("selfplay_enabled", True) and \
           not host_config.get("training_enabled", True):
            policy.receive_games = False
            policy.receive_npz = False
            policy.exclusion_reason = "selfplay and training disabled"

        # Mac machines - special handling
        if self._is_local_mac(host_name, host_config):
            policy.receive_games = False
            policy.receive_npz = False
            policy.receive_models = True
            policy.exclusion_reason = "local Mac machine"

            # External storage override
            if host_name in external_storage:
                override = external_storage[host_name]
                ext_path = override.get("path", "")
                if ext_path and Path(ext_path).exists():
                    policy.receive_games = override.get("receive_games", True)
                    policy.receive_npz = override.get("receive_npz", True)
                    policy.receive_models = override.get("receive_models", True)
                    policy.exclusion_reason = ""

        return policy

    def _is_local_mac(self, host_name: str, host_config: dict) -> bool:
        """Check if this is a local Mac machine."""
        if "mac" in host_name.lower() or "mbp" in host_name.lower():
            return True

        gpu = host_config.get("gpu", "")
        if "MPS" in gpu or "M1" in gpu or "M2" in gpu or "M3" in gpu:
            return True

        return False

    def get_sync_policy(self, node_id: str) -> NodeSyncPolicy:
        """Get sync policy for a node.

        Args:
            node_id: Node identifier

        Returns:
            NodeSyncPolicy for the node
        """
        return self._sync_policies.get(
            node_id,
            NodeSyncPolicy(node_id=node_id)
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

        # Check capacity if inventory manager available
        if self._inventory_manager:
            capacity = self._inventory_manager.get_node_capacity(node_id)
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
            return policy.receive_models  # Checkpoints follow model policy

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
        exclude_set = set(exclude_nodes or [])

        # Get current locations if game registry available
        if self._game_registry:
            current_locations = self._game_registry.find_game(game_id)
            current_nodes = {loc.node_id for loc in current_locations}
            exclude_set.update(current_nodes)

            # Need more copies?
            copies_needed = min_copies - len(current_nodes)
            if copies_needed <= 0:
                return []
        else:
            copies_needed = min_copies

        targets: list[SyncCandidateNode] = []

        # Get all known nodes
        if self._inventory_manager:
            all_nodes = self._inventory_manager.get_all_node_ids()
        else:
            all_nodes = list(self._sync_policies.keys())

        for node_id in all_nodes:
            if node_id in exclude_set:
                continue

            # Check if can receive games
            if not self.can_receive_data(node_id, DataType.GAME):
                continue

            # Get capacity and compute priority
            capacity = None
            if self._inventory_manager:
                capacity = self._inventory_manager.get_node_capacity(node_id)

            priority = self._compute_sync_priority(node_id, capacity)
            reason = self._get_sync_reason(node_id, capacity)

            targets.append(SyncCandidateNode(
                node_id=node_id,
                priority=priority,
                reason=reason,
                capacity=capacity,
            ))

        # Sort by priority (highest first)
        targets.sort(key=lambda t: t.priority, reverse=True)

        return targets[:copies_needed]

    def _compute_sync_priority(
        self,
        node_id: str,
        capacity: NodeCapacity | None,
    ) -> int:
        """Compute sync priority for a node.

        Higher priority = sync first.
        """
        priority = 50  # Base priority

        # Adjust based on role
        host_config = self._hosts_config.get(node_id, {})
        role = host_config.get("role", "selfplay")

        if "training" in role:
            priority += 20
        elif role == "coordinator":
            priority -= 30

        # Priority hosts
        if node_id in self._priority_hosts:
            priority += 15

        # Adjust based on capacity
        if capacity:
            if capacity.free_percent > 50:
                priority += 10
            elif capacity.free_percent < 20:
                priority -= 20

        # Ephemeral hosts get lower priority
        if host_config.get("storage_type") == "ephemeral":
            priority -= 10

        return priority

    def _get_sync_reason(
        self,
        node_id: str,
        capacity: NodeCapacity | None,
    ) -> str:
        """Get human-readable reason for sync target selection."""
        host_config = self._hosts_config.get(node_id, {})
        role = host_config.get("role", "selfplay")

        reasons = []

        if "training" in role:
            reasons.append("training node")

        if node_id in self._priority_hosts:
            reasons.append("priority host")

        if capacity:
            reasons.append(f"{capacity.free_percent:.1f}% free")

        return ", ".join(reasons) if reasons else "available"

    def get_nodes_needing_model(
        self,
        model_path: str,
        model_registry=None,
    ) -> list[str]:
        """Get nodes that need a model but don't have it.

        Args:
            model_path: Model path
            model_registry: Optional ModelRegistry

        Returns:
            List of node IDs
        """
        # Get nodes that already have the model
        if model_registry:
            locations = model_registry.find_model(model_path)
            has_model = {loc.node_id for loc in locations}
        else:
            has_model = set()

        # Find nodes that should have models but don't
        needing = []
        for node_id, policy in self._sync_policies.items():
            if not policy.receive_models:
                continue
            if node_id in has_model:
                continue
            needing.append(node_id)

        return needing

    def get_under_replicated_summary(
        self,
        game_registry=None,
        min_copies: int = REPLICATION_TARGET_COUNT,
    ) -> dict[str, Any]:
        """Get summary of under-replicated data.

        Args:
            game_registry: Optional GameLocationRegistry
            min_copies: Minimum required copies

        Returns:
            Summary dict
        """
        if game_registry is None:
            return {"under_replicated_games": 0, "min_copies": min_copies}

        under_replicated = game_registry.get_under_replicated_games(
            min_copies=min_copies,
            limit=10000,
        )

        return {
            "under_replicated_games": len(under_replicated),
            "min_copies": min_copies,
            "sample_games": under_replicated[:10],
        }
