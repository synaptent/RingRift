"""Cluster configuration helpers for distributed operations.

This module provides a unified interface for loading and accessing cluster
configuration from distributed_hosts.yaml, eliminating duplicate yaml.safe_load
calls scattered across the codebase.

Usage:
    from app.config.cluster_config import (
        load_cluster_config,
        get_sync_routing,
        get_auto_sync_config,
        get_host_bandwidth_limit,
        get_host_provider,
        filter_hosts_by_status,
    )

    # Get sync routing config
    sync_config = get_sync_routing()
    max_disk = sync_config.max_disk_usage_percent

    # Get bandwidth limit for a host
    limit = get_host_bandwidth_limit("vast-12345")  # Returns 50 (MB/s)

    # Filter hosts by status
    ready_hosts = filter_hosts_by_status(["ready"])

Consolidates inline yaml.safe_load patterns from:
- app/coordination/sync_router.py
- app/coordination/auto_sync_daemon.py
- app/distributed/cluster_manifest.py
- app/distributed/registries/replication.py
- app/routes/cluster.py
- And 8+ other files
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.distributed.hosts import HostConfig

logger = logging.getLogger(__name__)

# Default config path relative to ai-service/
DEFAULT_CONFIG_PATH = "config/distributed_hosts.yaml"


@dataclass
class ExternalStorageConfig:
    """Configuration for external storage on a host."""

    host: str
    path: str
    receive_games: bool = True
    receive_npz: bool = True
    receive_models: bool = True
    subdirs: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalStorageConfig:
        """Create from dictionary."""
        return cls(
            host=data.get("host", ""),
            path=data.get("path", ""),
            receive_games=data.get("receive_games", True),
            receive_npz=data.get("receive_npz", True),
            receive_models=data.get("receive_models", True),
            subdirs=data.get("subdirs", {}),
        )


@dataclass
class SyncRoutingConfig:
    """Configuration for data sync routing.

    Extracted from the sync_routing section of distributed_hosts.yaml.
    """

    max_disk_usage_percent: float = 70.0
    target_disk_usage_percent: float = 60.0
    min_free_disk_percent: float = 15.0
    replication_target: int = 2
    excluded_hosts: list[str] = field(default_factory=list)
    priority_hosts: list[str] = field(default_factory=list)
    underserved_configs: list[str] = field(default_factory=list)
    allowed_external_storage: list[ExternalStorageConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyncRoutingConfig:
        """Create from sync_routing dictionary section."""
        external_storage = [
            ExternalStorageConfig.from_dict(entry)
            for entry in data.get("allowed_external_storage", [])
            if isinstance(entry, dict)
        ]
        return cls(
            max_disk_usage_percent=data.get("max_disk_usage_percent", 70.0),
            target_disk_usage_percent=data.get("target_disk_usage_percent", 60.0),
            min_free_disk_percent=data.get("min_free_disk_percent", 15.0),
            replication_target=data.get("replication_target", 2),
            excluded_hosts=data.get("excluded_hosts", []),
            priority_hosts=data.get("priority_hosts", []),
            underserved_configs=data.get("underserved_configs", []),
            allowed_external_storage=external_storage,
        )

    def is_host_excluded(self, host_name: str) -> bool:
        """Check if a host is in the excluded list."""
        return host_name in self.excluded_hosts

    def is_priority_host(self, host_name: str) -> bool:
        """Check if a host is in the priority list."""
        return host_name in self.priority_hosts

    def get_external_storage(self, host_name: str) -> ExternalStorageConfig | None:
        """Get external storage config for a host, if any."""
        for storage in self.allowed_external_storage:
            if storage.host == host_name:
                return storage
        return None


@dataclass
class AutoSyncConfig:
    """Configuration for automatic data synchronization.

    Extracted from the auto_sync section of distributed_hosts.yaml.
    """

    enabled: bool = True
    interval_seconds: int = 60
    gossip_interval_seconds: int = 15
    skip_nfs_sync: bool = True
    max_concurrent_syncs: int = 8
    min_games_to_sync: int = 10
    bandwidth_limit_mbps: int = 100
    exclude_hosts: list[str] = field(default_factory=list)
    host_bandwidth_overrides: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutoSyncConfig:
        """Create from auto_sync dictionary section."""
        return cls(
            enabled=data.get("enabled", True),
            interval_seconds=data.get("interval_seconds", 60),
            gossip_interval_seconds=data.get("gossip_interval_seconds", 15),
            skip_nfs_sync=data.get("skip_nfs_sync", True),
            max_concurrent_syncs=data.get("max_concurrent_syncs", 8),
            min_games_to_sync=data.get("min_games_to_sync", 10),
            bandwidth_limit_mbps=data.get("bandwidth_limit_mbps", 100),
            exclude_hosts=data.get("exclude_hosts", []),
            host_bandwidth_overrides=data.get("host_bandwidth_overrides", {}),
        )

    def get_bandwidth_limit(self, host_name: str) -> int:
        """Get bandwidth limit for a host (uses glob matching).

        Args:
            host_name: The host name to check

        Returns:
            Bandwidth limit in MB/s (from overrides or default)
        """
        for pattern, limit in self.host_bandwidth_overrides.items():
            if fnmatch.fnmatch(host_name, pattern):
                return limit
        return self.bandwidth_limit_mbps


@dataclass
class EloSyncConfig:
    """Configuration for Elo database synchronization."""

    coordinator: str = "mac-studio"
    sync_port: int = 8766
    sync_interval: int = 300
    divergence_threshold: int = 50
    transports: list[str] = field(default_factory=lambda: ["tailscale", "aria2", "http"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EloSyncConfig:
        """Create from elo_sync dictionary section."""
        return cls(
            coordinator=data.get("coordinator", "mac-studio"),
            sync_port=data.get("sync_port", 8766),
            sync_interval=data.get("sync_interval", 300),
            divergence_threshold=data.get("divergence_threshold", 50),
            transports=data.get("transports", ["tailscale", "aria2", "http"]),
        )


@dataclass
class ClusterConfig:
    """Complete cluster configuration from distributed_hosts.yaml."""

    sync_routing: SyncRoutingConfig
    auto_sync: AutoSyncConfig
    elo_sync: EloSyncConfig
    p2p_voters: list[str]
    hosts_raw: dict[str, dict[str, Any]]
    _raw_config: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClusterConfig:
        """Create from full YAML config dictionary."""
        return cls(
            sync_routing=SyncRoutingConfig.from_dict(data.get("sync_routing", {})),
            auto_sync=AutoSyncConfig.from_dict(data.get("auto_sync", {})),
            elo_sync=EloSyncConfig.from_dict(data.get("elo_sync", {})),
            p2p_voters=data.get("p2p_voters", []),
            hosts_raw=data.get("hosts", {}),
            _raw_config=data,
        )

    def get_raw_section(self, section: str) -> dict[str, Any]:
        """Get a raw config section by name."""
        return self._raw_config.get(section, {})


# Global cached config
_CLUSTER_CONFIG_CACHE: ClusterConfig | None = None


def _get_config_path() -> Path:
    """Get the path to distributed_hosts.yaml."""
    # Navigate from app/config/ to ai-service/
    ai_service_dir = Path(__file__).parent.parent.parent
    return ai_service_dir / DEFAULT_CONFIG_PATH


def load_cluster_config(
    config_path: str | Path | None = None,
    *,
    force_reload: bool = False,
) -> ClusterConfig:
    """Load cluster configuration from YAML file.

    Args:
        config_path: Optional explicit path to config file.
        force_reload: If True, ignore cache and reload from disk.

    Returns:
        ClusterConfig object with all configuration sections.
    """
    global _CLUSTER_CONFIG_CACHE

    if _CLUSTER_CONFIG_CACHE is not None and not force_reload and config_path is None:
        return _CLUSTER_CONFIG_CACHE

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed. Run 'pip install pyyaml'.")
        return ClusterConfig(
            sync_routing=SyncRoutingConfig(),
            auto_sync=AutoSyncConfig(),
            elo_sync=EloSyncConfig(),
            p2p_voters=[],
            hosts_raw={},
        )

    if config_path is None:
        config_path = _get_config_path()
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Cluster config not found: {config_path}")
        return ClusterConfig(
            sync_routing=SyncRoutingConfig(),
            auto_sync=AutoSyncConfig(),
            elo_sync=EloSyncConfig(),
            p2p_voters=[],
            hosts_raw={},
        )

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    config = ClusterConfig.from_dict(data)

    if config_path is None or config_path == _get_config_path():
        _CLUSTER_CONFIG_CACHE = config
        logger.debug(f"Loaded cluster config: {len(config.hosts_raw)} hosts")

    return config


def clear_cluster_config_cache() -> None:
    """Clear the cached cluster configuration."""
    global _CLUSTER_CONFIG_CACHE
    _CLUSTER_CONFIG_CACHE = None


def get_sync_routing(config_path: str | Path | None = None) -> SyncRoutingConfig:
    """Get sync routing configuration.

    Convenience function for accessing just the sync_routing section.
    """
    return load_cluster_config(config_path).sync_routing


def get_auto_sync_config(config_path: str | Path | None = None) -> AutoSyncConfig:
    """Get auto sync configuration.

    Convenience function for accessing just the auto_sync section.
    """
    return load_cluster_config(config_path).auto_sync


def get_elo_sync_config(config_path: str | Path | None = None) -> EloSyncConfig:
    """Get Elo sync configuration."""
    return load_cluster_config(config_path).elo_sync


def get_p2p_voters(config_path: str | Path | None = None) -> list[str]:
    """Get P2P voter node list."""
    return load_cluster_config(config_path).p2p_voters


def get_p2p_port() -> int:
    """Get P2P orchestrator port from environment or default.

    Centralizes the P2P port configuration to avoid hardcoded values
    across the codebase.

    Returns:
        P2P port number, default 8770

    Environment:
        RINGRIFT_P2P_PORT: Override the default port
    """
    import os
    return int(os.environ.get("RINGRIFT_P2P_PORT", "8770"))


def get_host_bandwidth_limit(host_name: str, config_path: str | Path | None = None) -> int:
    """Get bandwidth limit for a host in MB/s.

    Uses glob pattern matching from host_bandwidth_overrides.

    Args:
        host_name: The host name to check
        config_path: Optional config file path

    Returns:
        Bandwidth limit in MB/s
    """
    return load_cluster_config(config_path).auto_sync.get_bandwidth_limit(host_name)


def get_host_provider(host_name: str) -> str:
    """Infer provider from host name prefix.

    Args:
        host_name: The host name (e.g., "vast-12345", "runpod-h100")

    Returns:
        Provider name: "vast", "runpod", "nebius", "vultr", "hetzner", "lambda", "local"
    """
    name_lower = host_name.lower()

    # Check known prefixes
    if name_lower.startswith("vast-"):
        return "vast"
    if name_lower.startswith("runpod-"):
        return "runpod"
    if name_lower.startswith("nebius-"):
        return "nebius"
    if name_lower.startswith("vultr-"):
        return "vultr"
    if name_lower.startswith("hetzner-"):
        return "hetzner"
    if name_lower.startswith("lambda-"):
        return "lambda"

    # Check known local hosts
    if name_lower in ("mac-studio", "macbook", "localhost"):
        return "local"

    # Default to unknown
    return "unknown"


def filter_hosts_by_status(
    statuses: list[str],
    config_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Filter hosts by status field.

    Args:
        statuses: List of valid statuses (e.g., ["ready", "active"])
        config_path: Optional config file path

    Returns:
        Dict of host_name -> host_config for matching hosts
    """
    config = load_cluster_config(config_path)
    return {
        name: host
        for name, host in config.hosts_raw.items()
        if host.get("status", "ready").lower() in [s.lower() for s in statuses]
    }


def filter_hosts_by_provider(
    providers: list[str],
    config_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Filter hosts by provider.

    Args:
        providers: List of providers (e.g., ["vast", "runpod"])
        config_path: Optional config file path

    Returns:
        Dict of host_name -> host_config for matching providers
    """
    config = load_cluster_config(config_path)
    providers_lower = [p.lower() for p in providers]
    return {
        name: host
        for name, host in config.hosts_raw.items()
        if get_host_provider(name) in providers_lower
    }


def filter_hosts_by_role(
    roles: list[str],
    config_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Filter hosts by role field.

    Args:
        roles: List of valid roles (e.g., ["training", "selfplay"])
        config_path: Optional config file path

    Returns:
        Dict of host_name -> host_config for matching roles
    """
    config = load_cluster_config(config_path)
    roles_lower = [r.lower() for r in roles]
    return {
        name: host
        for name, host in config.hosts_raw.items()
        if host.get("role", "selfplay").lower() in roles_lower
    }


def get_ready_hosts(config_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Get all hosts with status='ready'.

    This is the preferred filter for job dispatch.
    """
    return filter_hosts_by_status(["ready"], config_path)


def get_priority_sync_targets(config_path: str | Path | None = None) -> list[str]:
    """Get priority hosts for data sync (receive data first)."""
    return load_cluster_config(config_path).sync_routing.priority_hosts


def get_underserved_configs(config_path: str | Path | None = None) -> list[str]:
    """Get board configs that need more selfplay data."""
    return load_cluster_config(config_path).sync_routing.underserved_configs


def is_host_sync_excluded(host_name: str, config_path: str | Path | None = None) -> bool:
    """Check if a host is excluded from sync operations."""
    return load_cluster_config(config_path).sync_routing.is_host_excluded(host_name)


# =============================================================================
# Node Dataclass and Helpers (December 2025)
# Consolidated from app/sync/cluster_hosts.py
# =============================================================================

@dataclass
class ClusterNode:
    """Represents a cluster node with connectivity info.

    Provides unified access to host configuration from distributed_hosts.yaml.
    """

    name: str
    tailscale_ip: str | None = None
    ssh_host: str | None = None
    ssh_user: str = "ubuntu"
    ssh_key: str | None = None
    ssh_port: int = 22
    ringrift_path: str = "~/ringrift/ai-service"
    status: str = "unknown"
    role: str = "unknown"
    memory_gb: int = 0
    cpus: int = 0
    gpu: str = ""
    gpu_vram_gb: int = 0  # GPU VRAM in GB (December 2025)
    bandwidth_mbps: int = 0  # Bandwidth limit in MB/s (December 2025)
    data_server_port: int = 8766
    data_server_url: str | None = None
    is_coordinator: bool = False  # Whether this is the Elo coordinator
    # Dec 2025: Storage routing fields
    use_external_storage: bool = False  # Route data to external storage
    external_storage_path: str | None = None  # External storage mount point
    skip_sync_receive: bool = False  # Skip receiving sync data
    storage_paths: dict[str, str] | None = None  # Custom storage paths per data type

    @property
    def best_ip(self) -> str | None:
        """Get best IP for connection (prefer Tailscale)."""
        for candidate in (self.tailscale_ip, self.ssh_host):
            if not candidate:
                continue
            host = str(candidate).strip()
            if not host:
                continue
            # Handle user@host format
            if "@" in host:
                host = host.split("@", 1)[1]
            return host
        return None

    @property
    def data_server_base_url(self) -> str | None:
        """Get base URL for the node's data server."""
        if self.data_server_url:
            return self.data_server_url
        ip = self.best_ip
        if not ip:
            return None
        return f"http://{ip}:{self.data_server_port}"

    @property
    def is_active(self) -> bool:
        """Check if node is marked as active."""
        return self.status not in ("terminated", "offline", "setup")

    @property
    def is_gpu_node(self) -> bool:
        """Check if node has a GPU."""
        return bool(self.gpu)

    @property
    def provider(self) -> str:
        """Get provider inferred from node name."""
        return get_host_provider(self.name)

    def get_storage_path(self, data_type: str) -> str:
        """Get storage path for a specific data type.

        Respects storage_paths config for custom routing, falls back to
        external_storage_path or default ringrift_path.

        Args:
            data_type: One of 'games', 'models', 'training_data', 'checkpoints',
                       'logs', 'sync_incoming', 'npz', 'databases'

        Returns:
            Path string for the specified data type.

        Dec 2025: Added for OWC external drive routing on mac-studio.
        """
        # Check for custom storage_paths first
        if self.storage_paths:
            if data_type in self.storage_paths:
                return self.storage_paths[data_type]
            # Handle aliases
            alias_map = {
                "npz": "training_data",
                "databases": "games",
            }
            if data_type in alias_map and alias_map[data_type] in self.storage_paths:
                return self.storage_paths[alias_map[data_type]]

        # Fall back to external storage if enabled
        if self.use_external_storage and self.external_storage_path:
            default_subdirs = {
                "games": "selfplay_repository",
                "models": "canonical_models",
                "training_data": "canonical_data",
                "npz": "canonical_data",
                "checkpoints": "model_checkpoints",
                "logs": "logs",
                "sync_incoming": "cluster_games",
                "databases": "selfplay_repository",
            }
            subdir = default_subdirs.get(data_type, data_type)
            return f"{self.external_storage_path}/{subdir}"

        # Default paths under ringrift_path
        base = self.ringrift_path
        default_paths = {
            "games": f"{base}/data/games",
            "models": f"{base}/models",
            "training_data": f"{base}/data/training",
            "npz": f"{base}/data/training",
            "checkpoints": f"{base}/checkpoints",
            "logs": f"{base}/logs",
            "sync_incoming": f"{base}/data/sync_incoming",
            "databases": f"{base}/data/games",
        }
        return default_paths.get(data_type, f"{base}/data/{data_type}")

    def should_receive_sync(self) -> bool:
        """Check if this node should receive sync data.

        Returns False if skip_sync_receive is set (e.g., for orchestrator nodes
        that should not accumulate training data on their local disk).

        Dec 2025: Added to prevent orchestrator disk fill-up.
        """
        return not self.skip_sync_receive


def get_cluster_nodes(config_path: str | Path | None = None) -> dict[str, ClusterNode]:
    """Get all cluster nodes from config.

    Returns:
        Dict mapping node name to ClusterNode object.
    """
    config = load_cluster_config(config_path)
    nodes: dict[str, ClusterNode] = {}

    # Get default data server port
    default_data_port = 8766
    try:
        from app.config.unified_config import get_config
        default_data_port = get_config().distributed.data_server_port
    except (ImportError, AttributeError, KeyError):
        pass

    # Get Elo coordinator name for is_coordinator flag
    elo_coordinator = config.elo_sync.coordinator

    for name, cfg in config.hosts_raw.items():
        nodes[name] = ClusterNode(
            name=name,
            tailscale_ip=cfg.get("tailscale_ip"),
            ssh_host=cfg.get("ssh_host"),
            ssh_user=cfg.get("ssh_user", "ubuntu"),
            ssh_key=cfg.get("ssh_key"),
            ssh_port=cfg.get("ssh_port", 22),
            ringrift_path=cfg.get("ringrift_path", "~/ringrift/ai-service"),
            status=cfg.get("status", "unknown"),
            role=cfg.get("role", "unknown"),
            memory_gb=cfg.get("memory_gb", 0),
            cpus=cfg.get("cpus", 0),
            gpu=cfg.get("gpu", ""),
            gpu_vram_gb=cfg.get("gpu_vram_gb", 0),
            bandwidth_mbps=cfg.get("bandwidth_mbps", 0),
            data_server_port=cfg.get("data_server_port", default_data_port),
            data_server_url=cfg.get("data_server_url"),
            is_coordinator=(name == elo_coordinator or cfg.get("is_coordinator", False)),
            # Dec 2025: Storage routing fields
            use_external_storage=cfg.get("use_external_storage", False),
            external_storage_path=cfg.get("external_storage_path"),
            skip_sync_receive=cfg.get("skip_sync_receive", False),
            storage_paths=cfg.get("storage_paths"),
        )

    return nodes


def get_active_nodes(config_path: str | Path | None = None) -> list[ClusterNode]:
    """Get all active (non-terminated) cluster nodes."""
    return [n for n in get_cluster_nodes(config_path).values() if n.is_active]


def get_gpu_nodes(config_path: str | Path | None = None) -> list[ClusterNode]:
    """Get all GPU-equipped cluster nodes."""
    return [n for n in get_cluster_nodes(config_path).values() if n.is_gpu_node and n.is_active]


def get_coordinator_node(config_path: str | Path | None = None) -> ClusterNode | None:
    """Get the Elo coordinator node."""
    elo_config = get_elo_sync_config(config_path)
    nodes = get_cluster_nodes(config_path)
    return nodes.get(elo_config.coordinator)


def get_nfs_hosts(config_path: str | Path | None = None) -> list[str]:
    """Get hosts that have NFS configured (excluded from sync).

    Returns:
        List of host names with NFS configured.
    """
    config = load_cluster_config(config_path)
    return [
        name for name, cfg in config.hosts_raw.items()
        if cfg.get("nfs_enabled", False) or cfg.get("has_nfs", False)
    ]


# =============================================================================
# Provider-based defaults (December 2025)
# Consolidated from sync_bandwidth.py and utilization_optimizer.py
# =============================================================================

# Default bandwidth limits by provider (in KB/s)
_PROVIDER_BANDWIDTH_DEFAULTS_KBS: dict[str, int] = {
    "lambda": 100_000,     # 100 MB/s - fast internal network
    "runpod": 100_000,     # 100 MB/s - good network
    "nebius": 50_000,      # 50 MB/s - conservative due to rate limits
    "vast": 50_000,        # 50 MB/s - varies by instance
    "vultr": 25_000,       # 25 MB/s - vGPU instances
    "hetzner": 100_000,    # 100 MB/s - dedicated servers
    "local": 100_000,      # 100 MB/s - local network
}


def get_node_bandwidth_kbs(node_name: str, config_path: str | Path | None = None) -> int:
    """Get bandwidth limit for a node in KB/s.

    Priority order:
    1. Explicit bandwidth_mbps in YAML config
    2. auto_sync.host_bandwidth_overrides pattern match
    3. Provider-based default
    4. Global default (100 MB/s)

    Args:
        node_name: The node/host name
        config_path: Optional config file path

    Returns:
        Bandwidth limit in KB/s
    """
    nodes = get_cluster_nodes(config_path)
    node = nodes.get(node_name)

    # 1. Check explicit config in node definition
    if node and node.bandwidth_mbps > 0:
        return node.bandwidth_mbps * 1000  # Convert MB/s to KB/s

    # 2. Check auto_sync overrides (uses glob patterns)
    auto_sync = get_auto_sync_config(config_path)
    override_limit = auto_sync.get_bandwidth_limit(node_name)
    if override_limit != auto_sync.bandwidth_limit_mbps:
        return override_limit * 1000

    # 3. Provider-based default
    provider = get_host_provider(node_name)
    if provider in _PROVIDER_BANDWIDTH_DEFAULTS_KBS:
        return _PROVIDER_BANDWIDTH_DEFAULTS_KBS[provider]

    # 4. Check for Tailscale IP (internal network = high bandwidth)
    if node and node.tailscale_ip:
        return 100_000  # 100 MB/s for Tailscale

    # 5. Global default
    return auto_sync.bandwidth_limit_mbps * 1000


def get_ready_nodes(config_path: str | Path | None = None) -> list[ClusterNode]:
    """Get all cluster nodes with status='ready'.

    Returns:
        List of ClusterNode objects with ready status.
    """
    return [
        n for n in get_cluster_nodes(config_path).values()
        if n.status == "ready"
    ]


def get_nodes_by_provider(
    provider: str,
    config_path: str | Path | None = None,
    *,
    only_active: bool = True,
) -> list[ClusterNode]:
    """Get all nodes for a specific provider.

    Args:
        provider: Provider name (vast, runpod, nebius, etc.)
        config_path: Optional config file path
        only_active: If True, only return active nodes

    Returns:
        List of ClusterNode objects for the provider.
    """
    nodes = get_cluster_nodes(config_path).values()
    result = [n for n in nodes if n.provider == provider.lower()]
    if only_active:
        result = [n for n in result if n.is_active]
    return result


def get_gpu_types(config_path: str | Path | None = None) -> dict[str, int]:
    """Get unique GPU types and their VRAM from cluster config.

    Returns:
        Dict mapping GPU model name to VRAM in GB.
    """
    gpu_types: dict[str, int] = {}

    for node in get_gpu_nodes(config_path):
        if not node.gpu or node.gpu == "none":
            continue

        # Handle "Nx GPU_MODEL" format (e.g., "2x RTX 5090")
        gpu_name = node.gpu
        if "x " in gpu_name:
            parts = gpu_name.split("x ", 1)
            if len(parts) == 2:
                gpu_name = parts[1]

        if gpu_name not in gpu_types:
            gpu_types[gpu_name] = node.gpu_vram_gb

    return gpu_types
