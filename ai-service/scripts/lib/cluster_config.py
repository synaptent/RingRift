#!/usr/bin/env python3
"""Cluster configuration loader for RingRift distributed training.

This module provides utilities to load and query the legacy
`config/cluster_nodes.yaml` inventory (or `cluster_nodes.local.yaml` if it
exists). New infrastructure should prefer `distributed_hosts.yaml` via the
unified loaders, but this module remains active compatibility surface for older
deployment helpers.

Usage:
    from scripts.lib.cluster_config import ClusterConfig

    config = ClusterConfig()
    for node in config.get_all_nodes():
        print(f"{node['name']}: {node['host']}")

    # Get nodes by group
    gh200_nodes = config.get_group('lambda_gh200_tailscale')

    # Get SSH command for a node
    ssh_cmd = config.get_ssh_command('gh200-a')
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def get_config_path() -> Path:
    """Get the cluster config file path, preferring local override."""
    ai_service_root = Path(__file__).resolve().parents[2]
    local_config = ai_service_root / "config" / "cluster_nodes.local.yaml"
    default_config = ai_service_root / "config" / "cluster_nodes.yaml"

    if local_config.exists():
        return local_config
    return default_config


class ClusterConfig:
    """Cluster configuration manager."""

    def __init__(self, config_path: Path | str | None = None):
        if config_path is None:
            config_path = get_config_path()
        self.config_path = Path(config_path)
        self._config: dict[str, Any] = {}
        self._nodes_by_name: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Cluster config not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f) or {}

        # Build node lookup index
        self._nodes_by_name = {}
        for group_name, group_data in self._config.items():
            if group_name == "config":
                continue
            if isinstance(group_data, dict) and "nodes" in group_data:
                for node in group_data["nodes"]:
                    node["_group"] = group_name
                    self._nodes_by_name[node["name"]] = node

    def get_all_nodes(self) -> list[dict[str, Any]]:
        """Get all nodes from all groups."""
        return list(self._nodes_by_name.values())

    def get_group(self, group_name: str) -> list[dict[str, Any]]:
        """Get all nodes in a specific group."""
        group_data = self._config.get(group_name, {})
        if isinstance(group_data, dict) and "nodes" in group_data:
            return group_data["nodes"]
        return []

    def get_group_names(self) -> list[str]:
        """Get all group names."""
        return [
            k for k, v in self._config.items()
            if k != "config" and isinstance(v, dict) and "nodes" in v
        ]

    def get_node(self, name: str) -> dict[str, Any] | None:
        """Get a specific node by name."""
        return self._nodes_by_name.get(name)

    def get_ssh_command(
        self,
        name: str,
        command: str = "",
        *,
        timeout: int | None = None,
        batch_mode: bool = True,
    ) -> str:
        """Get SSH command for a node."""
        node = self.get_node(name)
        if not node:
            raise ValueError(f"Unknown node: {name}")

        cfg = self._config.get("config", {})
        ssh_timeout = timeout or cfg.get("ssh_connect_timeout", 12)
        port = node.get("port", cfg.get("default_port", 22))

        opts = [f"-o ConnectTimeout={ssh_timeout}", "-o StrictHostKeyChecking=no"]
        if batch_mode:
            opts.append("-o BatchMode=yes")
        if port != 22:
            opts.append(f"-p {port}")

        opts_str = " ".join(opts)
        target = f"{node['user']}@{node['host']}"

        if command:
            return f"ssh {opts_str} {target} '{command}'"
        return f"ssh {opts_str} {target}"

    def get_config_option(self, key: str, default: Any = None) -> Any:
        """Get a configuration option."""
        return self._config.get("config", {}).get(key, default)

    def get_repo_paths(self) -> list[str]:
        """Get list of possible repository paths on remote nodes."""
        return self.get_config_option("repo_paths", [
            "~/ringrift",
            "~/RingRift",
            "/root/RingRift",
        ])


def print_nodes_as_bash_array() -> None:
    """Print all nodes in a format that can be sourced by bash scripts."""
    config = ClusterConfig()
    print("# Auto-generated from cluster_nodes.yaml")
    print("declare -A CLUSTER_NODES")
    print("declare -A CLUSTER_USERS")
    print("declare -A CLUSTER_PORTS")
    print("declare -a CLUSTER_GROUPS")
    print()

    for group_name in config.get_group_names():
        print(f'CLUSTER_GROUPS+=("{group_name}")')

    print()
    for node in config.get_all_nodes():
        name = node["name"]
        host = node["host"]
        user = node["user"]
        port = node.get("port", 22)
        print(f'CLUSTER_NODES["{name}"]="{host}"')
        print(f'CLUSTER_USERS["{name}"]="{user}"')
        print(f'CLUSTER_PORTS["{name}"]="{port}"')

    print()
    print(f'SSH_TIMEOUT={config.get_config_option("ssh_timeout", 25)}')
    print(f'SSH_CONNECT_TIMEOUT={config.get_config_option("ssh_connect_timeout", 12)}')


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--bash":
        print_nodes_as_bash_array()
    else:
        # Print summary
        config = ClusterConfig()
        print(f"Config loaded from: {config.config_path}")
        print(f"Groups: {', '.join(config.get_group_names())}")
        print(f"Total nodes: {len(config.get_all_nodes())}")
        for group in config.get_group_names():
            nodes = config.get_group(group)
            print(f"  {group}: {len(nodes)} nodes")
