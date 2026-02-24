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
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.distributed.hosts import HostConfig

from app.config.thresholds import DISK_SYNC_TARGET_PERCENT

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

    max_disk_usage_percent: float = float(DISK_SYNC_TARGET_PERCENT)
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
            max_disk_usage_percent=data.get("max_disk_usage_percent", float(DISK_SYNC_TARGET_PERCENT)),
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
    max_concurrent_syncs: int = 1  # Feb 2026: Limited to 1 to prevent OOM from parallel rsyncs
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
            max_concurrent_syncs=data.get("max_concurrent_syncs", 1),
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


# =============================================================================
# Config Version Tracking (December 2025)
# For distributed config synchronization via gossip protocol
# =============================================================================


@dataclass
class ConfigVersion:
    """Tracks config freshness for distributed sync.

    Used by gossip protocol to detect when peer nodes have newer
    configurations and trigger automatic sync.
    """

    content_hash: str  # SHA256 of YAML content (first 16 chars)
    timestamp: float  # When config was last modified (file mtime)
    source_node: str  # Which node made the change

    @classmethod
    def from_yaml_path(
        cls, path: str | Path, source_node: str = "local"
    ) -> "ConfigVersion":
        """Compute version from YAML file.

        Args:
            path: Path to the YAML config file.
            source_node: Node ID that owns this config version.

        Returns:
            ConfigVersion with content hash and mtime.
        """
        path = Path(path)
        with open(path, "rb") as f:
            content = f.read()
        return cls(
            content_hash=hashlib.sha256(content).hexdigest()[:16],
            timestamp=os.path.getmtime(path),
            source_node=source_node,
        )

    def is_newer_than(self, other: "ConfigVersion") -> bool:
        """Returns True if self is newer than other."""
        return self.timestamp > other.timestamp

    def to_dict(self) -> dict[str, Any]:
        """Serialize for gossip state."""
        return {
            "hash": self.content_hash,
            "timestamp": self.timestamp,
            "source_node": self.source_node,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigVersion":
        """Deserialize from gossip state."""
        return cls(
            content_hash=data.get("hash", ""),
            timestamp=data.get("timestamp", 0.0),
            source_node=data.get("source_node", "unknown"),
        )


class ClusterConfigCache:
    """Singleton cache with automatic freshness tracking.

    Provides centralized config access with file-mtime-based invalidation.
    Used by gossip protocol to include config version in peer state.

    Thread-safe via lock protection on reload operations.
    """

    _instance: "ClusterConfigCache | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize cache (use get_instance() instead of direct construction)."""
        self._config: ClusterConfig | None = None
        self._version: ConfigVersion | None = None
        self._load_time: float = 0.0
        self._yaml_path: Path = _get_config_path()

    @classmethod
    def get_instance(cls) -> "ClusterConfigCache":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        with cls._lock:
            cls._instance = None

    def get_config(self, force_reload: bool = False) -> ClusterConfig:
        """Get config, auto-reloading if file changed.

        Args:
            force_reload: If True, reload regardless of mtime.

        Returns:
            ClusterConfig object.
        """
        if force_reload or self._needs_refresh():
            self._reload()
        if self._config is None:
            self._reload()
        return self._config  # type: ignore[return-value]

    def get_version(self) -> ConfigVersion:
        """Get current config version for gossip state.

        Returns:
            ConfigVersion with hash, timestamp, and source node.
        """
        if self._version is None:
            self._reload()
        return self._version  # type: ignore[return-value]

    def _needs_refresh(self) -> bool:
        """Check if config file has been modified since last load."""
        if self._config is None:
            return True
        try:
            current_mtime = os.path.getmtime(self._yaml_path)
            return current_mtime > self._load_time
        except OSError:
            # File may not exist or be inaccessible
            return False

    def _reload(self) -> None:
        """Reload config from disk."""
        with self._lock:
            self._config = load_cluster_config(force_reload=True)
            try:
                self._version = ConfigVersion.from_yaml_path(
                    self._yaml_path,
                    source_node=os.environ.get("RINGRIFT_NODE_ID", "unknown"),
                )
            except OSError:
                # File may not exist
                self._version = ConfigVersion(
                    content_hash="0" * 16,
                    timestamp=0.0,
                    source_node=os.environ.get("RINGRIFT_NODE_ID", "unknown"),
                )
            self._load_time = time.time()
            logger.debug(
                f"Config reloaded: hash={self._version.content_hash}, "
                f"mtime={self._version.timestamp:.0f}"
            )


def get_config_cache() -> ClusterConfigCache:
    """Get the global config cache singleton.

    Convenience function for accessing the cache.

    Returns:
        ClusterConfigCache singleton instance.
    """
    return ClusterConfigCache.get_instance()


def get_config_version() -> ConfigVersion:
    """Get the current config version.

    Convenience function for gossip protocol integration.

    Returns:
        ConfigVersion with hash and timestamp.
    """
    return ClusterConfigCache.get_instance().get_version()


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
    from app.config.ports import P2P_DEFAULT_PORT
    return P2P_DEFAULT_PORT


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
    tailscale_ipv6: str | None = None  # Jan 2, 2026: IPv6 Tailscale address (fd7a:...)
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
    # December 2025: GPU-aware job assignment fields
    cuda_capable: bool = False  # Explicit flag for CUDA capability
    selfplay_enabled: bool = True  # Whether selfplay jobs can run on this node
    training_enabled: bool = False  # Whether training jobs can run on this node
    preferred_workloads: list[str] | None = None  # Preferred workload types
    excluded_workloads: list[str] | None = None  # Excluded workload types
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
        """Get best IPv4 IP for connection (prefer Tailscale)."""
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
    def best_ipv6(self) -> str | None:
        """Get best IPv6 IP for connection (Jan 2, 2026).

        Returns Tailscale IPv6 address if available.
        """
        if self.tailscale_ipv6:
            return str(self.tailscale_ipv6).strip() or None
        return None

    def get_best_ip(self, prefer_ipv6: bool = False) -> str | None:
        """Get best IP for connection with IPv6 preference option.

        Jan 2, 2026: Added for dual-stack support.

        Args:
            prefer_ipv6: If True, prefer IPv6 over IPv4 when available.

        Returns:
            Best available IP address, or None if no IP available.
        """
        if prefer_ipv6 and self.best_ipv6:
            return self.best_ipv6
        return self.best_ip or self.best_ipv6

    @property
    def data_server_base_url(self) -> str | None:
        """Get base URL for the node's data server."""
        if self.data_server_url:
            return self.data_server_url
        ip = self.best_ip
        if not ip:
            return None
        # Jan 2, 2026: Handle IPv6 addresses (wrap in brackets)
        if ":" in ip and not ip.startswith("["):
            ip = f"[{ip}]"
        return f"http://{ip}:{self.data_server_port}"

    @property
    def is_active(self) -> bool:
        """Check if node is marked as active.

        December 2025: Added 'retired' status to inactive list to prevent
        sync operations from targeting retired nodes (e.g., nebius-h100-2).
        February 2026: Added 'archived' status for terminated/offline nodes
        preserved for configuration history.
        """
        return self.status not in ("terminated", "offline", "setup", "retired", "archived")

    @property
    def is_gpu_node(self) -> bool:
        """Check if node has a GPU."""
        return bool(self.gpu)

    @property
    def has_cuda_gpu(self) -> bool:
        """Check if node has a CUDA-capable GPU.

        December 2025: Added for GPU-aware job assignment.
        Uses explicit cuda_capable flag if set, otherwise infers from GPU presence.
        """
        # Explicit flag takes precedence
        if self.cuda_capable:
            return True
        # Infer from GPU presence and VRAM
        return bool(self.gpu) and self.gpu_vram_gb > 0

    @property
    def can_run_gpu_selfplay(self) -> bool:
        """Check if node can run GPU selfplay (neural network modes).

        December 2025: Requires both GPU capability AND selfplay enabled.
        """
        return (
            self.has_cuda_gpu
            and self.selfplay_enabled
            and "gpu-selfplay" not in (self.excluded_workloads or [])
        )

    @property
    def should_avoid_cpu_selfplay(self) -> bool:
        """Check if this node should avoid CPU-only selfplay.

        December 2025: High-end GPU nodes should prioritize GPU workloads.
        Avoid wasting expensive GPU compute on CPU-bound tasks.
        """
        # Nodes with >48GB VRAM should avoid CPU selfplay
        if self.gpu_vram_gb >= 48:
            return True
        # Check explicit exclusion
        if "cpu-selfplay" in (self.excluded_workloads or []):
            return True
        return False

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

        Returns False if:
        - skip_sync_receive is set (e.g., for orchestrator nodes)
        - node is not active (retired, terminated, offline, setup)

        Dec 2025: Added to prevent orchestrator disk fill-up.
        Dec 2025: Added is_active check to filter retired nodes.
        """
        return self.is_active and not self.skip_sync_receive


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
            tailscale_ipv6=cfg.get("tailscale_ipv6"),  # Jan 2, 2026: IPv6 support
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
            # December 2025: GPU-aware job assignment fields
            cuda_capable=cfg.get("cuda_capable", False),
            selfplay_enabled=cfg.get("selfplay_enabled", True),
            training_enabled=cfg.get("training_enabled", False),
            preferred_workloads=cfg.get("preferred_workloads"),
            excluded_workloads=cfg.get("excluded_workloads"),
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


# =============================================================================
# GPU-Aware Node Filtering (December 2025)
# For GPU-aware job assignment system
# =============================================================================


def get_gpu_capable_nodes(
    config_path: str | Path | None = None,
    *,
    only_active: bool = True,
    only_selfplay_enabled: bool = True,
) -> list[ClusterNode]:
    """Get nodes capable of running GPU selfplay (neural network modes).

    December 2025: Added for GPU-aware job assignment to prevent wasting
    GPU compute on CPU-only selfplay.

    Args:
        config_path: Optional config file path.
        only_active: If True, only return active nodes.
        only_selfplay_enabled: If True, only return nodes with selfplay enabled.

    Returns:
        List of ClusterNode objects that can run GPU selfplay.
    """
    nodes = get_cluster_nodes(config_path).values()
    result = []

    for node in nodes:
        if only_active and not node.is_active:
            continue
        if only_selfplay_enabled and not node.selfplay_enabled:
            continue
        if node.can_run_gpu_selfplay:
            result.append(node)

    return result


def get_cpu_only_nodes(
    config_path: str | Path | None = None,
    *,
    only_active: bool = True,
) -> list[ClusterNode]:
    """Get nodes without GPU capability (CPU-only selfplay).

    December 2025: Added for GPU-aware job assignment.

    Args:
        config_path: Optional config file path.
        only_active: If True, only return active nodes.

    Returns:
        List of ClusterNode objects without GPU capability.
    """
    nodes = get_cluster_nodes(config_path).values()
    result = []

    for node in nodes:
        if only_active and not node.is_active:
            continue
        if not node.has_cuda_gpu:
            result.append(node)

    return result


def get_nodes_for_engine_mode(
    engine_mode: str,
    config_path: str | Path | None = None,
    *,
    only_active: bool = True,
) -> list[ClusterNode]:
    """Get nodes suitable for a specific engine mode.

    December 2025: Added for GPU-aware job assignment.

    Args:
        engine_mode: Engine mode string (e.g., "gumbel-mcts", "heuristic").
        config_path: Optional config file path.
        only_active: If True, only return active nodes.

    Returns:
        List of ClusterNode objects suitable for the engine mode.
    """
    # GPU-required engine modes (neural network inference)
    GPU_REQUIRED_MODES = {
        "gumbel-mcts", "mcts", "nnue-guided", "policy-only",
        "nn-minimax", "nn-descent", "gnn", "hybrid",
        "gmo", "ebmo", "ig-gmo", "cage",
    }

    requires_gpu = engine_mode.lower() in GPU_REQUIRED_MODES

    if requires_gpu:
        return get_gpu_capable_nodes(config_path, only_active=only_active)
    else:
        # CPU-compatible modes can run on any node
        return get_active_nodes(config_path) if only_active else list(
            get_cluster_nodes(config_path).values()
        )


def filter_nodes_by_workload(
    workload_type: str,
    config_path: str | Path | None = None,
    *,
    only_active: bool = True,
) -> list[ClusterNode]:
    """Filter nodes by preferred/excluded workload configuration.

    December 2025: Added for workload-aware job assignment.

    Args:
        workload_type: Workload type (e.g., "gpu-selfplay", "training", "cpu-selfplay").
        config_path: Optional config file path.
        only_active: If True, only return active nodes.

    Returns:
        List of ClusterNode objects suitable for the workload type.
    """
    nodes = get_cluster_nodes(config_path).values()
    result = []

    for node in nodes:
        if only_active and not node.is_active:
            continue

        # Check excluded workloads
        if node.excluded_workloads and workload_type in node.excluded_workloads:
            continue

        # Nodes with preferred_workloads set are prioritized for those workloads
        # but can still run other workloads if not excluded
        result.append(node)

    # Sort by preference: nodes with this workload in preferred_workloads first
    def preference_key(n: ClusterNode) -> int:
        if n.preferred_workloads and workload_type in n.preferred_workloads:
            return 0  # Highest priority
        return 1  # Default priority

    result.sort(key=preference_key)
    return result


def get_training_nodes(
    config_path: str | Path | None = None,
    *,
    only_active: bool = True,
) -> list[ClusterNode]:
    """Get nodes enabled for training jobs.

    December 2025: Added for GPU-aware job assignment.

    Args:
        config_path: Optional config file path.
        only_active: If True, only return active nodes.

    Returns:
        List of ClusterNode objects enabled for training.
    """
    nodes = get_cluster_nodes(config_path).values()
    result = []

    for node in nodes:
        if only_active and not node.is_active:
            continue
        if node.training_enabled:
            result.append(node)

    return result


# =============================================================================
# Dynamic Config Updates (December 2025)
# For availability submodule: auto-provisioning and recovery
# =============================================================================


def add_or_update_node(
    node_name: str,
    node_config: dict[str, Any],
    config_path: str | Path | None = None,
) -> bool:
    """Add or update a node in the cluster configuration.

    December 2025: Added for availability submodule to dynamically register
    newly provisioned or recreated instances.

    Args:
        node_name: Name of the node to add/update.
        node_config: Dictionary with node configuration (ssh_host, gpu, etc.).
        config_path: Optional config file path.

    Returns:
        True if successful, False otherwise.
    """
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed - cannot update cluster config")
        return False

    if config_path is None:
        config_path = _get_config_path()
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.error(f"Cluster config not found: {config_path}")
        return False

    try:
        # Load existing config
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        # Ensure hosts section exists
        if "hosts" not in data:
            data["hosts"] = {}

        # Add or update the node
        if node_name in data["hosts"]:
            # Merge with existing config (preserve fields not in node_config)
            data["hosts"][node_name].update(node_config)
            logger.info(f"Updated node config: {node_name}")
        else:
            # Add new node
            data["hosts"][node_name] = node_config
            logger.info(f"Added new node to config: {node_name}")

        # Write back to file
        with open(config_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        # Clear cache to reload updated config
        clear_cluster_config_cache()

        return True

    except Exception as e:
        logger.error(f"Failed to update cluster config: {e}")
        return False


def update_node_status(
    node_name: str,
    status: str,
    config_path: str | Path | None = None,
    **extra_fields: Any,
) -> bool:
    """Update a node's status in the cluster configuration.

    December 2025: Convenience function for quick status changes.

    Args:
        node_name: Name of the node to update.
        status: New status (ready, offline, terminated, retired, setup).
        config_path: Optional config file path.
        **extra_fields: Additional fields to update (e.g., ssh_host, tailscale_ip).

    Returns:
        True if successful, False otherwise.
    """
    update = {"status": status}
    update.update(extra_fields)
    return add_or_update_node(node_name, update, config_path)


def remove_node(
    node_name: str,
    config_path: str | Path | None = None,
) -> bool:
    """Remove a node from the cluster configuration.

    December 2025: Added for cleanup of terminated instances.

    Args:
        node_name: Name of the node to remove.
        config_path: Optional config file path.

    Returns:
        True if successful, False otherwise.
    """
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed - cannot update cluster config")
        return False

    if config_path is None:
        config_path = _get_config_path()
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.error(f"Cluster config not found: {config_path}")
        return False

    try:
        # Load existing config
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        if "hosts" not in data or node_name not in data["hosts"]:
            logger.warning(f"Node not found in config: {node_name}")
            return True  # Already removed

        # Remove the node
        del data["hosts"][node_name]
        logger.info(f"Removed node from config: {node_name}")

        # Write back to file
        with open(config_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        # Clear cache to reload updated config
        clear_cluster_config_cache()

        return True

    except Exception as e:
        logger.error(f"Failed to remove node from config: {e}")
        return False


def cluster_node_to_dict(node: ClusterNode) -> dict[str, Any]:
    """Convert a ClusterNode to a dictionary for YAML serialization.

    December 2025: Added for availability submodule.

    Args:
        node: ClusterNode object to convert.

    Returns:
        Dictionary suitable for adding to cluster config.
    """
    result: dict[str, Any] = {
        "status": node.status,
    }

    # Only include non-default values
    if node.tailscale_ip:
        result["tailscale_ip"] = node.tailscale_ip
    if node.tailscale_ipv6:  # Jan 2, 2026: IPv6 support
        result["tailscale_ipv6"] = node.tailscale_ipv6
    if node.ssh_host:
        result["ssh_host"] = node.ssh_host
    if node.ssh_user != "ubuntu":
        result["ssh_user"] = node.ssh_user
    if node.ssh_key:
        result["ssh_key"] = node.ssh_key
    if node.ssh_port != 22:
        result["ssh_port"] = node.ssh_port
    if node.ringrift_path != "~/ringrift/ai-service":
        result["ringrift_path"] = node.ringrift_path
    if node.role != "unknown":
        result["role"] = node.role
    if node.memory_gb:
        result["memory_gb"] = node.memory_gb
    if node.cpus:
        result["cpus"] = node.cpus
    if node.gpu:
        result["gpu"] = node.gpu
    if node.gpu_vram_gb:
        result["gpu_vram_gb"] = node.gpu_vram_gb
    if node.bandwidth_mbps:
        result["bandwidth_mbps"] = node.bandwidth_mbps
    if node.cuda_capable:
        result["cuda_capable"] = node.cuda_capable
    if not node.selfplay_enabled:
        result["selfplay_enabled"] = node.selfplay_enabled
    if node.training_enabled:
        result["training_enabled"] = node.training_enabled
    if node.preferred_workloads:
        result["preferred_workloads"] = node.preferred_workloads
    if node.excluded_workloads:
        result["excluded_workloads"] = node.excluded_workloads
    if node.data_server_port != 8766:
        result["data_server_port"] = node.data_server_port
    if node.data_server_url:
        result["data_server_url"] = node.data_server_url
    if node.is_coordinator:
        result["is_coordinator"] = node.is_coordinator
    if node.use_external_storage:
        result["use_external_storage"] = node.use_external_storage
    if node.external_storage_path:
        result["external_storage_path"] = node.external_storage_path
    if node.skip_sync_receive:
        result["skip_sync_receive"] = node.skip_sync_receive
    if node.storage_paths:
        result["storage_paths"] = node.storage_paths

    return result
