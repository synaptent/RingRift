"""Cluster configuration manager for ClusterManifest.

This module provides the ClusterConfigManager class which handles loading
cluster configuration from distributed_hosts.yaml and building node
exclusion rules for sync policies.

Extracted from ClusterManifest for improved testability and separation
of concerns (December 2025).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.cluster_config import load_cluster_config

logger = logging.getLogger(__name__)

__all__ = [
    "ClusterConfigManager",
    "NodeSyncPolicy",
    "MAX_DISK_USAGE_PERCENT",
]

# Constants
try:
    from app.config.thresholds import DISK_SYNC_TARGET_PERCENT
    MAX_DISK_USAGE_PERCENT = DISK_SYNC_TARGET_PERCENT  # Don't sync to nodes above this usage
except ImportError:
    MAX_DISK_USAGE_PERCENT = 70  # Don't sync to nodes above this usage


@dataclass
class NodeSyncPolicy:
    """Sync policy for a node.

    Attributes
    ----------
    node_id:
        The node identifier (hostname).
    receive_games:
        Whether the node should receive game database files.
    receive_models:
        Whether the node should receive model files.
    receive_npz:
        Whether the node should receive NPZ training files.
    max_disk_usage_percent:
        Maximum disk usage before sync is blocked.
    excluded:
        Whether the node is completely excluded from sync.
    exclusion_reason:
        Human-readable reason for exclusion, if any.
    """

    node_id: str
    receive_games: bool = True
    receive_models: bool = True
    receive_npz: bool = True
    max_disk_usage_percent: float = MAX_DISK_USAGE_PERCENT
    excluded: bool = False
    exclusion_reason: str = ""


class ClusterConfigManager:
    """Manages cluster configuration and node sync policies.

    This class handles:
    - Loading cluster configuration from distributed_hosts.yaml
    - Building node exclusion rules based on role and config
    - Providing sync policy lookup for nodes
    - Tracking priority hosts for training data

    Extracted from ClusterManifest for testability.

    Usage
    -----
    ```python
    config_mgr = ClusterConfigManager()

    # Get sync policy for a node
    policy = config_mgr.get_sync_policy("runpod-h100")
    if policy.receive_games:
        # Node can receive game data
        ...

    # Check if host is a priority host
    if config_mgr.is_priority_host("nebius-h100-3"):
        # Prioritize sync to this host
        ...
    ```
    """

    def __init__(self, config_path: Path | None = None):
        """Initialize the cluster configuration manager.

        Parameters
        ----------
        config_path:
            Optional path to distributed_hosts.yaml. If None, uses
            the default location from cluster_config module.
        """
        self._hosts_config: dict[str, Any] = {}
        self._exclusion_rules: dict[str, NodeSyncPolicy] = {}
        self._max_disk_usage = MAX_DISK_USAGE_PERCENT
        self._priority_hosts: set[str] = set()
        self._load_config(config_path)

    def _load_config(self, config_path: Path | None = None) -> None:
        """Load host configuration and exclusion rules.

        Uses the consolidated cluster_config module instead of inline yaml loading.

        Parameters
        ----------
        config_path:
            Optional path to configuration file.
        """
        try:
            cluster_config = load_cluster_config(config_path)

            self._hosts_config = cluster_config.hosts_raw

            # Build exclusion rules from config (use raw section for backwards compat)
            raw_config = {
                "hosts": cluster_config.hosts_raw,
                "sync_routing": cluster_config.get_raw_section("sync_routing"),
                "auto_sync": cluster_config.get_raw_section("auto_sync"),
            }
            self._build_exclusion_rules(raw_config)

        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def _build_exclusion_rules(self, config: dict[str, Any]) -> None:
        """Build node exclusion rules from configuration.

        Parameters
        ----------
        config:
            Dictionary containing 'hosts', 'sync_routing', and 'auto_sync' sections.
        """
        hosts = config.get("hosts", {})

        # Get sync routing configuration
        sync_routing = config.get("sync_routing", {})

        # Read max disk usage from config
        self._max_disk_usage = sync_routing.get(
            "max_disk_usage_percent", MAX_DISK_USAGE_PERCENT
        )

        # Auto-sync exclusion from auto_sync section
        auto_sync = config.get("auto_sync", {})
        exclude_hosts = set(auto_sync.get("exclude_hosts", []))

        # Process sync_routing.excluded_hosts with detailed policies
        excluded_host_policies: dict[str, dict] = {}
        for entry in sync_routing.get("excluded_hosts", []):
            if isinstance(entry, dict):
                name = entry.get("name", "")
                if name:
                    exclude_hosts.add(name)
                    excluded_host_policies[name] = entry
            else:
                exclude_hosts.add(entry)

        # Process allowed_external_storage overrides
        external_storage_overrides: dict[str, dict] = {}
        for entry in sync_routing.get("allowed_external_storage", []):
            if isinstance(entry, dict):
                host = entry.get("host", "")
                if host:
                    external_storage_overrides[host] = entry

        # Priority hosts for training data
        self._priority_hosts = set(sync_routing.get("priority_hosts", []))

        for host_name, host_config in hosts.items():
            role = host_config.get("role", "selfplay")

            # Default policy
            policy = NodeSyncPolicy(
                node_id=host_name,
                max_disk_usage_percent=self._max_disk_usage,
            )

            # Check if coordinator (typically dev machines)
            if role == "coordinator":
                policy.receive_games = False
                policy.receive_npz = False
                policy.receive_models = True  # Still receive models
                policy.exclusion_reason = "coordinator node"

            # Check if explicitly excluded with detailed policy
            if host_name in excluded_host_policies:
                entry = excluded_host_policies[host_name]
                policy.receive_games = entry.get("receive_games", False)
                policy.receive_npz = entry.get("receive_npz", False)
                policy.receive_models = entry.get("receive_models", True)
                policy.exclusion_reason = entry.get("reason", "explicitly excluded")
            elif host_name in exclude_hosts:
                policy.receive_games = False
                policy.receive_npz = False
                policy.receive_models = True
                policy.exclusion_reason = "explicitly excluded"

            # Check selfplay_enabled/training_enabled flags
            if not host_config.get("selfplay_enabled", True) and not host_config.get(
                "training_enabled", True
            ):
                policy.receive_games = False
                policy.receive_npz = False
                policy.exclusion_reason = "selfplay and training disabled"

            # Mac machines - special handling
            if self._is_local_mac(host_name, host_config):
                # Exclude local Macs by default
                policy.receive_games = False
                policy.receive_npz = False
                policy.receive_models = True
                policy.exclusion_reason = "local Mac machine"

                # Check for external storage override (e.g., OWC drive)
                if host_name in external_storage_overrides:
                    override = external_storage_overrides[host_name]
                    # Only apply override if the external path exists
                    ext_path = override.get("path", "")
                    if ext_path and Path(ext_path).exists():
                        policy.receive_games = override.get("receive_games", True)
                        policy.receive_npz = override.get("receive_npz", True)
                        policy.receive_models = override.get("receive_models", True)
                        policy.exclusion_reason = ""
                        logger.info(
                            f"External storage override for {host_name}: {ext_path}"
                        )
                elif self._has_owc_external_drive(host_name, host_config):
                    # Fallback to legacy detection
                    policy.receive_games = True
                    policy.receive_npz = True
                    policy.exclusion_reason = ""

            self._exclusion_rules[host_name] = policy

    def _is_local_mac(self, host_name: str, host_config: dict) -> bool:
        """Check if this is a local Mac machine.

        Parameters
        ----------
        host_name:
            The hostname to check.
        host_config:
            Host configuration dictionary.

        Returns
        -------
        bool
            True if the host is identified as a local Mac.
        """
        # Check hostname patterns
        if "mac" in host_name.lower() or "mbp" in host_name.lower():
            return True

        # Check GPU field for MPS
        gpu = host_config.get("gpu", "")
        if "MPS" in gpu or "M1" in gpu or "M2" in gpu or "M3" in gpu:
            return True

        return False

    def _has_owc_external_drive(self, host_name: str, host_config: dict) -> bool:
        """Check if this Mac has an OWC external drive for sync.

        Parameters
        ----------
        host_name:
            The hostname to check.
        host_config:
            Host configuration dictionary.

        Returns
        -------
        bool
            True if the host has an OWC external drive configured.
        """
        # Mac Studio with external storage
        if "mac-studio" in host_name.lower():
            # Check for configured external drive path
            ringrift_path = host_config.get("ringrift_path", "")
            if "/Volumes/OWC" in ringrift_path or "/Volumes/External" in ringrift_path:
                return True

            # Check for sync_storage_path configuration
            sync_path = host_config.get("sync_storage_path", "")
            if "/Volumes/OWC" in sync_path:
                return True

        return False

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def max_disk_usage_percent(self) -> float:
        """Get the maximum disk usage percentage for sync."""
        return self._max_disk_usage

    @property
    def priority_hosts(self) -> set[str]:
        """Get the set of priority hosts for training data."""
        return self._priority_hosts.copy()

    @property
    def hosts_config(self) -> dict[str, Any]:
        """Get the raw hosts configuration dictionary."""
        return self._hosts_config.copy()

    def get_sync_policy(self, node_id: str) -> NodeSyncPolicy:
        """Get the sync policy for a node.

        Parameters
        ----------
        node_id:
            The node identifier to look up.

        Returns
        -------
        NodeSyncPolicy
            The sync policy for the node. If the node is not in the
            configuration, returns a default permissive policy.
        """
        if node_id in self._exclusion_rules:
            return self._exclusion_rules[node_id]

        # Default policy for unknown nodes
        return NodeSyncPolicy(
            node_id=node_id,
            max_disk_usage_percent=self._max_disk_usage,
        )

    def is_priority_host(self, node_id: str) -> bool:
        """Check if a node is a priority host for training data.

        Parameters
        ----------
        node_id:
            The node identifier to check.

        Returns
        -------
        bool
            True if the node is a priority host.
        """
        return node_id in self._priority_hosts

    def can_receive_data(
        self,
        node_id: str,
        data_type: str,
    ) -> bool:
        """Check if a node can receive a specific data type.

        Parameters
        ----------
        node_id:
            The node identifier to check.
        data_type:
            One of 'game', 'model', 'npz'.

        Returns
        -------
        bool
            True if the node can receive the specified data type.
        """
        policy = self.get_sync_policy(node_id)

        if policy.excluded:
            return False

        if data_type == "game":
            return policy.receive_games
        elif data_type == "model":
            return policy.receive_models
        elif data_type == "npz":
            return policy.receive_npz
        else:
            return True  # Unknown data type - allow by default

    def get_all_policies(self) -> dict[str, NodeSyncPolicy]:
        """Get all sync policies.

        Returns
        -------
        dict[str, NodeSyncPolicy]
            Mapping from node_id to sync policy.
        """
        return self._exclusion_rules.copy()

    def get_hosts_by_role(self, role: str) -> list[str]:
        """Get list of hosts with a specific role.

        Parameters
        ----------
        role:
            The role to filter by (e.g., 'training', 'selfplay', 'coordinator').

        Returns
        -------
        list[str]
            List of host names with the specified role.
        """
        hosts = []
        for host_name, host_config in self._hosts_config.items():
            if host_config.get("role", "selfplay") == role:
                hosts.append(host_name)
        return hosts

    def get_active_hosts(self) -> list[str]:
        """Get list of hosts that are marked as active/ready.

        Returns
        -------
        list[str]
            List of host names with status 'ready' or 'active'.
        """
        hosts = []
        for host_name, host_config in self._hosts_config.items():
            status = host_config.get("status", "ready")
            if status in ("ready", "active"):
                hosts.append(host_name)
        return hosts
