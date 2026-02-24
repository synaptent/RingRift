"""Cluster Configuration Loader.

Reads static cluster configuration from config/cluster.yaml and provides
node metadata, fallback IPs, priorities, and alert thresholds to the
P2P orchestrator.

This integrates the centralized cluster.yaml config with the dynamic
P2P node discovery system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Default config path relative to ai-service root
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "cluster.yaml"


@dataclass
class NodeConfig:
    """Static configuration for a cluster node."""
    name: str
    host: str
    tailscale_ip: str | None = None
    ssh_user: str = "ubuntu"
    gpu_type: str = ""
    gpu_count: int = 1
    vram_gb: int = 0
    batch_multiplier: int = 32
    priority: int = 1  # 1=highest priority
    roles: list[str] = field(default_factory=lambda: ["selfplay"])
    status: str = "active"
    notes: str = ""

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def can_train(self) -> bool:
        return "training" in self.roles and self.is_active

    @property
    def can_selfplay(self) -> bool:
        return "selfplay" in self.roles and self.is_active


@dataclass
class AlertThresholds:
    """Alert threshold configuration."""
    gpu_util_low: int = 10
    gpu_util_critical: int = 0
    # Aligned with app.config.thresholds: DISK_PRODUCTION_HALT(85) / DISK_CRITICAL(90)
    disk_usage_warn: int = 80
    disk_usage_critical: int = 90
    memory_warn: int = 85
    offline_timeout_sec: int = 300


@dataclass
class JobDefaults:
    """Default job configuration."""
    selfplay_games_per_batch: int = 64
    selfplay_max_moves: int = 200
    selfplay_timeout_minutes: int = 30
    training_batch_size: int = 4096
    training_learning_rate: float = 0.001
    training_checkpoint_interval: int = 1000
    gauntlet_games_per_matchup: int = 100


@dataclass
class ClusterConfig:
    """Complete cluster configuration."""
    name: str = "ringrift-training"
    version: str = "2.0"
    nodes: dict[str, NodeConfig] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)
    alerts: AlertThresholds = field(default_factory=AlertThresholds)
    job_defaults: JobDefaults = field(default_factory=JobDefaults)

    def get_node(self, node_id: str) -> NodeConfig | None:
        """Get node config by ID."""
        return self.nodes.get(node_id)

    def get_tailscale_ip(self, node_id: str) -> str | None:
        """Get tailscale IP for a node (fallback when hostname fails)."""
        node = self.nodes.get(node_id)
        return node.tailscale_ip if node else None

    def get_active_nodes(self) -> list[NodeConfig]:
        """Get all active nodes."""
        return [n for n in self.nodes.values() if n.is_active]

    def get_training_nodes(self) -> list[NodeConfig]:
        """Get nodes capable of training."""
        return [n for n in self.nodes.values() if n.can_train]

    def get_selfplay_nodes(self) -> list[NodeConfig]:
        """Get nodes capable of selfplay."""
        return [n for n in self.nodes.values() if n.can_selfplay]

    def get_group_nodes(self, group_name: str) -> list[NodeConfig]:
        """Get nodes in a named group."""
        node_names = self.groups.get(group_name, [])
        return [self.nodes[n] for n in node_names if n in self.nodes]

    def get_priority_sorted_nodes(self, role: str = "selfplay") -> list[NodeConfig]:
        """Get nodes sorted by priority (1=highest)."""
        nodes = [n for n in self.nodes.values() if role in n.roles and n.is_active]
        return sorted(nodes, key=lambda n: n.priority)


def load_cluster_config(config_path: Path | None = None) -> ClusterConfig:
    """Load cluster configuration from YAML file.

    Args:
        config_path: Path to config file. Uses default if not specified.

    Returns:
        ClusterConfig with all node and alert settings.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # Return empty config if file doesn't exist
    if not config_path.exists():
        return ClusterConfig()

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not data:
        return ClusterConfig()

    # Parse cluster info
    cluster_info = data.get("cluster", {})
    config = ClusterConfig(
        name=cluster_info.get("name", "ringrift-training"),
        version=cluster_info.get("version", "2.0"),
    )

    # Parse alert thresholds
    alerts_data = data.get("alerts", {}).get("thresholds", {})
    config.alerts = AlertThresholds(
        gpu_util_low=alerts_data.get("gpu_util_low", 10),
        gpu_util_critical=alerts_data.get("gpu_util_critical", 0),
        disk_usage_warn=alerts_data.get("disk_usage_warn", 80),
        disk_usage_critical=alerts_data.get("disk_usage_critical", 90),
        memory_warn=alerts_data.get("memory_warn", 85),
        offline_timeout_sec=alerts_data.get("offline_timeout_sec", 300),
    )

    # Parse job defaults
    job_data = data.get("job_defaults", {})
    selfplay_data = job_data.get("selfplay", {})
    training_data = job_data.get("training", {})
    gauntlet_data = job_data.get("gauntlet", {})
    config.job_defaults = JobDefaults(
        selfplay_games_per_batch=selfplay_data.get("games_per_batch", 64),
        selfplay_max_moves=selfplay_data.get("max_moves", 200),
        selfplay_timeout_minutes=selfplay_data.get("timeout_minutes", 30),
        training_batch_size=training_data.get("batch_size", 4096),
        training_learning_rate=training_data.get("learning_rate", 0.001),
        training_checkpoint_interval=training_data.get("checkpoint_interval", 1000),
        gauntlet_games_per_matchup=gauntlet_data.get("games_per_matchup", 100),
    )

    # Parse nodes
    nodes_data = data.get("nodes", {})
    for node_name, node_data in nodes_data.items():
        config.nodes[node_name] = NodeConfig(
            name=node_name,
            host=node_data.get("host", node_name),
            tailscale_ip=node_data.get("tailscale_ip"),
            ssh_user=node_data.get("ssh_user", "ubuntu"),
            gpu_type=node_data.get("gpu_type", ""),
            gpu_count=node_data.get("gpu_count", 1),
            vram_gb=node_data.get("vram_gb", 0),
            batch_multiplier=node_data.get("batch_multiplier", 32),
            priority=node_data.get("priority", 1),
            roles=node_data.get("roles", ["selfplay"]),
            status=node_data.get("status", "active"),
            notes=node_data.get("notes", ""),
        )

    # Parse groups
    groups_data = data.get("groups", {})
    for group_name, group_info in groups_data.items():
        nodes_list = group_info.get("nodes", [])
        config.groups[group_name] = nodes_list

    return config


# Global cached config instance
_cached_config: ClusterConfig | None = None
_cached_config_mtime: float = 0.0


def get_cluster_config(force_reload: bool = False) -> ClusterConfig:
    """Get cluster config with caching.

    Automatically reloads if the config file has been modified.

    Args:
        force_reload: Force reload even if cached.

    Returns:
        Cached or freshly loaded ClusterConfig.
    """
    global _cached_config, _cached_config_mtime

    config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        if _cached_config is None:
            _cached_config = ClusterConfig()
        return _cached_config

    current_mtime = config_path.stat().st_mtime

    if force_reload or _cached_config is None or current_mtime > _cached_config_mtime:
        _cached_config = load_cluster_config(config_path)
        _cached_config_mtime = current_mtime

    return _cached_config


def get_webhook_urls() -> dict[str, str]:
    """Get alert webhook URLs from config/cluster.yaml.

    Returns dict with 'slack' and 'discord' keys if configured.
    Environment variable expansion is performed.
    """
    config_path = DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        data = yaml.safe_load(f)

    alerts = data.get("alerts", {})
    webhooks = {}

    # Expand environment variables
    slack = alerts.get("slack_webhook", "")
    if slack.startswith("${") and slack.endswith("}"):
        env_var = slack[2:-1]
        slack = os.environ.get(env_var, "")
    if slack:
        webhooks["slack"] = slack

    discord = alerts.get("discord_webhook", "")
    if discord.startswith("${") and discord.endswith("}"):
        env_var = discord[2:-1]
        discord = os.environ.get(env_var, "")
    if discord:
        webhooks["discord"] = discord

    return webhooks
