"""
Unified Cluster Hosts Configuration

Provides a single interface for accessing cluster host information.
Reads from config/distributed_hosts.yaml as the primary source, with
fallback to config/cluster.yaml for newer deployments.

Usage:
    from scripts.lib.hosts import get_hosts, get_host, HostConfig

    # Get all hosts
    hosts = get_hosts()

    # Get specific host
    host = get_host("lambda-gh200-a")
    print(host.ssh_host)
    print(host.tailscale_ip)

    # Filter by role
    training_hosts = get_hosts(role="nn_training")

    # Get hosts with specific status
    active_hosts = get_hosts(status="ready")
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Find config directory
_SCRIPT_DIR = Path(__file__).parent
_AI_SERVICE_ROOT = _SCRIPT_DIR.parent.parent
_CONFIG_DIR = _AI_SERVICE_ROOT / "config"

DISTRIBUTED_HOSTS_PATH = _CONFIG_DIR / "distributed_hosts.yaml"
CLUSTER_YAML_PATH = _CONFIG_DIR / "cluster.yaml"


@dataclass
class HostConfig:
    """Configuration for a cluster host."""
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    ssh_key: str = "~/.ssh/id_cluster"
    tailscale_ip: str | None = None
    ringrift_path: str = "~/ringrift/ai-service"
    venv_activate: str = "source ~/ringrift/ai-service/venv/bin/activate"
    memory_gb: int = 0
    cpus: int = 0
    gpu: str = ""
    gpu_type: str = ""
    vram_gb: int = 0
    role: str = "selfplay"
    roles: list[str] = field(default_factory=list)
    status: str = "unknown"
    p2p_voter: bool = False
    vast_instance_id: str | None = None
    aws_instance_id: str | None = None
    worker_url: str | None = None
    worker_port: int | None = None
    notes: str = ""

    @property
    def is_vast(self) -> bool:
        """Check if this is a Vast.ai instance."""
        return self.vast_instance_id is not None or self.name.startswith("vast-")

    @property
    def is_aws(self) -> bool:
        """Check if this is an AWS instance."""
        return self.aws_instance_id is not None

    @property
    def is_lambda(self) -> bool:
        """Check if this is a Lambda Labs instance."""
        return self.name.startswith("lambda-")

    @property
    def effective_ssh_host(self) -> str:
        """Get the best SSH host to use (prefer Tailscale if available)."""
        return self.tailscale_ip or self.ssh_host

    @property
    def all_roles(self) -> list[str]:
        """Get all roles (combining legacy role and roles list)."""
        roles = list(self.roles) if self.roles else []
        if self.role and self.role not in roles:
            roles.append(self.role)
        return roles

    def has_role(self, role: str) -> bool:
        """Check if host has a specific role."""
        return role in self.all_roles or role in self.role


@dataclass
class EloSyncConfig:
    """ELO synchronization configuration."""
    coordinator: str = "mac-studio"
    sync_port: int = 8766
    sync_interval: int = 300
    divergence_threshold: int = 50
    transports: list[str] = field(default_factory=lambda: ["tailscale", "http"])


class HostsManager:
    """Manages cluster host configuration."""

    def __init__(
        self,
        distributed_hosts_path: Path | None = None,
        cluster_yaml_path: Path | None = None,
    ):
        self._distributed_path = distributed_hosts_path or DISTRIBUTED_HOSTS_PATH
        self._cluster_path = cluster_yaml_path or CLUSTER_YAML_PATH
        self._hosts: dict[str, HostConfig] | None = None
        self._elo_sync: EloSyncConfig | None = None
        self._raw_config: dict | None = None

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML files."""
        if self._raw_config is not None:
            return self._raw_config

        # Try distributed_hosts.yaml first (more complete)
        if self._distributed_path.exists():
            try:
                with open(self._distributed_path) as f:
                    self._raw_config = yaml.safe_load(f) or {}
                    return self._raw_config
            except Exception as e:
                print(f"Warning: Failed to load {self._distributed_path}: {e}")

        # Fall back to cluster.yaml
        if self._cluster_path.exists():
            try:
                with open(self._cluster_path) as f:
                    self._raw_config = yaml.safe_load(f) or {}
                    return self._raw_config
            except Exception as e:
                print(f"Warning: Failed to load {self._cluster_path}: {e}")

        self._raw_config = {}
        return self._raw_config

    def _parse_host(self, name: str, data: dict[str, Any]) -> HostConfig:
        """Parse a host configuration from raw data."""
        # Handle both distributed_hosts.yaml and cluster.yaml formats
        ssh_host = data.get("ssh_host") or data.get("host", name)

        # If host contains user@, extract it
        if "@" in str(ssh_host):
            user_part, ssh_host = ssh_host.split("@", 1)
            if not data.get("ssh_user"):
                data["ssh_user"] = user_part

        return HostConfig(
            name=name,
            ssh_host=ssh_host,
            ssh_user=data.get("ssh_user", "ubuntu"),
            ssh_port=data.get("ssh_port", 22),
            ssh_key=data.get("ssh_key", "~/.ssh/id_cluster"),
            tailscale_ip=data.get("tailscale_ip"),
            ringrift_path=data.get("ringrift_path", "~/ringrift/ai-service"),
            venv_activate=data.get("venv_activate", "source ~/ringrift/ai-service/venv/bin/activate"),
            memory_gb=data.get("memory_gb", 0),
            cpus=data.get("cpus", 0),
            gpu=data.get("gpu", data.get("gpu_type", "")),
            gpu_type=data.get("gpu_type", ""),
            vram_gb=data.get("vram_gb", 0),
            role=data.get("role", "selfplay"),
            roles=data.get("roles", []),
            status=data.get("status", "unknown"),
            p2p_voter=data.get("p2p_voter", False),
            vast_instance_id=data.get("vast_instance_id"),
            aws_instance_id=data.get("aws_instance_id"),
            worker_url=data.get("worker_url"),
            worker_port=data.get("worker_port"),
            notes=data.get("notes", ""),
        )

    def _load_hosts(self) -> dict[str, HostConfig]:
        """Load and parse all hosts."""
        if self._hosts is not None:
            return self._hosts

        config = self._load_config()
        self._hosts = {}

        # distributed_hosts.yaml uses "hosts" key
        hosts_data = config.get("hosts", {})

        # cluster.yaml uses "nodes" key
        if not hosts_data:
            hosts_data = config.get("nodes", {})

        for name, data in hosts_data.items():
            if data:  # Skip None entries
                self._hosts[name] = self._parse_host(name, data)

        return self._hosts

    def get_hosts(
        self,
        role: str | None = None,
        status: str | None = None,
        gpu_type: str | None = None,
        p2p_voter: bool | None = None,
        is_vast: bool | None = None,
    ) -> list[HostConfig]:
        """Get hosts matching filters.

        Args:
            role: Filter by role (e.g., "nn_training", "selfplay")
            status: Filter by status (e.g., "ready", "active")
            gpu_type: Filter by GPU type (e.g., "GH200", "H100")
            p2p_voter: Filter by P2P voter status
            is_vast: Filter Vast.ai instances

        Returns:
            List of matching HostConfig objects
        """
        hosts = list(self._load_hosts().values())

        if role is not None:
            hosts = [h for h in hosts if h.has_role(role)]

        if status is not None:
            hosts = [h for h in hosts if h.status == status]

        if gpu_type is not None:
            hosts = [h for h in hosts if gpu_type.lower() in h.gpu.lower() or gpu_type.lower() in h.gpu_type.lower()]

        if p2p_voter is not None:
            hosts = [h for h in hosts if h.p2p_voter == p2p_voter]

        if is_vast is not None:
            hosts = [h for h in hosts if h.is_vast == is_vast]

        return hosts

    def get_host(self, name: str) -> HostConfig | None:
        """Get a specific host by name."""
        return self._load_hosts().get(name)

    def get_host_names(self) -> list[str]:
        """Get all host names."""
        return list(self._load_hosts().keys())

    def get_elo_sync_config(self) -> EloSyncConfig:
        """Get ELO sync configuration."""
        if self._elo_sync is not None:
            return self._elo_sync

        config = self._load_config()
        elo_data = config.get("elo_sync", {})

        self._elo_sync = EloSyncConfig(
            coordinator=elo_data.get("coordinator", "mac-studio"),
            sync_port=elo_data.get("sync_port", 8766),
            sync_interval=elo_data.get("sync_interval", 300),
            divergence_threshold=elo_data.get("divergence_threshold", 50),
            transports=elo_data.get("transports", ["tailscale", "http"]),
        )

        return self._elo_sync

    def get_hosts_by_group(self, group_name: str) -> list[HostConfig]:
        """Get hosts in a named group (from cluster.yaml groups section)."""
        config = self._load_config()
        groups = config.get("groups", {})

        if group_name not in groups:
            return []

        group = groups[group_name]
        node_names = group.get("nodes", [])

        hosts = self._load_hosts()
        return [hosts[name] for name in node_names if name in hosts]

    def reload(self) -> None:
        """Force reload of configuration."""
        self._raw_config = None
        self._hosts = None
        self._elo_sync = None


# Global instance
_hosts_manager: HostsManager | None = None


def get_hosts_manager() -> HostsManager:
    """Get the global hosts manager."""
    global _hosts_manager
    if _hosts_manager is None:
        _hosts_manager = HostsManager()
    return _hosts_manager


def get_hosts(
    role: str | None = None,
    status: str | None = None,
    gpu_type: str | None = None,
    p2p_voter: bool | None = None,
    is_vast: bool | None = None,
) -> list[HostConfig]:
    """Get hosts matching filters.

    Args:
        role: Filter by role
        status: Filter by status
        gpu_type: Filter by GPU type
        p2p_voter: Filter by P2P voter status
        is_vast: Filter Vast.ai instances

    Returns:
        List of HostConfig objects
    """
    return get_hosts_manager().get_hosts(
        role=role,
        status=status,
        gpu_type=gpu_type,
        p2p_voter=p2p_voter,
        is_vast=is_vast,
    )


def get_host(name: str) -> HostConfig | None:
    """Get a specific host by name."""
    return get_hosts_manager().get_host(name)


def get_host_names() -> list[str]:
    """Get all host names."""
    return get_hosts_manager().get_host_names()


def get_elo_sync_config() -> EloSyncConfig:
    """Get ELO synchronization configuration."""
    return get_hosts_manager().get_elo_sync_config()


def get_hosts_by_group(group_name: str) -> list[HostConfig]:
    """Get hosts in a named group."""
    return get_hosts_manager().get_hosts_by_group(group_name)


# Convenience aliases for backwards compatibility
def load_distributed_hosts() -> dict[str, HostConfig]:
    """Load all hosts (backwards compatible name)."""
    return get_hosts_manager()._load_hosts()


def get_training_hosts() -> list[HostConfig]:
    """Get hosts suitable for training."""
    return get_hosts(role="nn_training")


def get_selfplay_hosts() -> list[HostConfig]:
    """Get hosts suitable for self-play."""
    return get_hosts(role="selfplay")


def get_active_hosts() -> list[HostConfig]:
    """Get hosts with ready/active status."""
    ready = get_hosts(status="ready")
    active = get_hosts(status="active")
    return list({h.name: h for h in ready + active}.values())


def get_p2p_voters() -> list[HostConfig]:
    """Get hosts that are P2P voters."""
    return get_hosts(p2p_voter=True)
