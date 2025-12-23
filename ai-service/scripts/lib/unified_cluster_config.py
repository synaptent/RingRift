"""Unified Cluster Configuration - Single source of truth for node configurations.

This module provides a unified interface for loading and validating cluster
node configurations. It consolidates the following config files:
- config/cluster_nodes.yaml (primary source)
- config/distributed_hosts.yaml (legacy, for compatibility)
- config/p2p_hosts.yaml (legacy, for compatibility)
- config/cluster.yaml (legacy, for compatibility)

Usage:
    from scripts.lib.unified_cluster_config import get_cluster_config, NodeConfig

    config = get_cluster_config()
    for node in config.get_nodes():
        print(f"{node.node_id}: {node.ssh_host} ({node.gpu_type})")

    # Get specific node
    node = config.get_node("lambda-gh200-e")
    if node:
        print(f"SSH: {node.ssh_user}@{node.ssh_host}")
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Config file paths (in priority order)
CONFIG_PATHS = [
    PROJECT_ROOT / "config" / "cluster_nodes.yaml",
    PROJECT_ROOT / "config" / "distributed_hosts.yaml",
    PROJECT_ROOT / "config" / "p2p_hosts.yaml",
    PROJECT_ROOT / "config" / "cluster.yaml",
]

# Cache settings
CONFIG_CACHE_TTL = 300  # 5 minutes
CONNECTIVITY_CACHE_TTL = 60  # 1 minute


@dataclass
class NodeConfig:
    """Configuration for a single cluster node."""
    node_id: str

    # SSH settings
    ssh_host: str = ""
    ssh_user: str = "ubuntu"
    ssh_key: str = ""
    ssh_port: int = 22

    # Network settings
    tailscale_ip: str = ""
    direct_ip: str = ""
    preferred_ip: str = ""  # Computed: which IP to use

    # Paths
    ringrift_path: str = "/home/ubuntu/ringrift/ai-service"
    venv_path: str = "/home/ubuntu/ringrift/ai-service/venv"
    data_path: str = "/home/ubuntu/ringrift/ai-service/data"

    # Resources
    gpu_type: str = ""
    vram_gb: int = 0
    memory_gb: int = 0
    cpu_cores: int = 0

    # Roles and capabilities
    roles: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)

    # Status
    status: str = "active"  # active, inactive, maintenance
    is_p2p_voter: bool = False
    is_training_node: bool = False
    is_controller: bool = False

    # Connectivity cache
    _last_connectivity_check: float = 0
    _is_reachable: bool | None = None

    def __post_init__(self):
        # Expand SSH key path
        if self.ssh_key and self.ssh_key.startswith("~"):
            self.ssh_key = os.path.expanduser(self.ssh_key)

        # Determine preferred IP
        if not self.preferred_ip:
            self.preferred_ip = self.tailscale_ip or self.direct_ip or self.ssh_host

        # Set SSH host if not specified
        if not self.ssh_host:
            self.ssh_host = self.preferred_ip

        # Determine roles from GPU type if not specified
        if not self.roles and self.gpu_type:
            gpu_lower = self.gpu_type.lower()
            if "h100" in gpu_lower or "gh200" in gpu_lower or "a100" in gpu_lower:
                self.roles = ["selfplay", "training", "gumbel"]
                self.is_training_node = True
            elif "4090" in gpu_lower or "3090" in gpu_lower:
                self.roles = ["selfplay", "training"]
            else:
                self.roles = ["selfplay"]

    def is_reachable(self, force_check: bool = False) -> bool:
        """Check if the node is reachable via network.

        Uses a cached value unless force_check is True or cache is stale.
        """
        now = time.time()
        if not force_check and self._is_reachable is not None:
            if now - self._last_connectivity_check < CONNECTIVITY_CACHE_TTL:
                return self._is_reachable

        # Quick TCP check on SSH port
        self._is_reachable = self._check_tcp_connectivity()
        self._last_connectivity_check = now
        return self._is_reachable

    def _check_tcp_connectivity(self, timeout: float = 2.0) -> bool:
        """Check TCP connectivity to SSH port."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((self.ssh_host, self.ssh_port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def get_ssh_command(self, command: str, timeout: int = 30) -> list[str]:
        """Build SSH command for this node."""
        cmd = [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
        ]
        if self.ssh_key:
            cmd.extend(["-i", self.ssh_key])
        if self.ssh_port != 22:
            cmd.extend(["-p", str(self.ssh_port)])

        cmd.append(f"{self.ssh_user}@{self.ssh_host}")
        cmd.append(command)
        return cmd

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "ssh": {
                "host": self.ssh_host,
                "user": self.ssh_user,
                "key": self.ssh_key,
                "port": self.ssh_port,
            },
            "network": {
                "tailscale_ip": self.tailscale_ip,
                "direct_ip": self.direct_ip,
                "preferred_ip": self.preferred_ip,
            },
            "paths": {
                "ringrift": self.ringrift_path,
                "venv": self.venv_path,
                "data": self.data_path,
            },
            "resources": {
                "gpu": self.gpu_type,
                "vram_gb": self.vram_gb,
                "memory_gb": self.memory_gb,
                "cpu_cores": self.cpu_cores,
            },
            "roles": self.roles,
            "capabilities": self.capabilities,
            "status": self.status,
            "is_p2p_voter": self.is_p2p_voter,
            "is_training_node": self.is_training_node,
            "is_controller": self.is_controller,
        }


class UnifiedClusterConfig:
    """Unified configuration loader for cluster nodes.

    Loads configuration from multiple YAML files and provides a single
    consistent interface. Handles schema validation, path expansion,
    and connectivity caching.
    """

    def __init__(self, config_paths: list[Path] | None = None):
        self.config_paths = config_paths or CONFIG_PATHS
        self.nodes: dict[str, NodeConfig] = {}
        self._load_time: float = 0
        self._lock = threading.RLock()

        # Load on init
        self.reload()

    def reload(self) -> None:
        """Reload configuration from YAML files."""
        with self._lock:
            self.nodes.clear()

            # Load from each config file
            for config_path in self.config_paths:
                if config_path.exists():
                    try:
                        self._load_config_file(config_path)
                    except Exception as e:
                        logger.error(f"Error loading {config_path}: {e}")

            self._load_time = time.time()
            logger.info(f"Loaded {len(self.nodes)} nodes from cluster configuration")

    def _load_config_file(self, path: Path) -> None:
        """Load a single config file."""
        import yaml

        with open(path) as f:
            config = yaml.safe_load(f) or {}

        # Handle different config file formats
        filename = path.name

        if filename == "cluster_nodes.yaml":
            self._parse_cluster_nodes(config)
        elif filename == "distributed_hosts.yaml":
            self._parse_distributed_hosts(config)
        elif filename == "p2p_hosts.yaml":
            self._parse_p2p_hosts(config)
        elif filename == "cluster.yaml":
            self._parse_cluster_yaml(config)

    def _parse_cluster_nodes(self, config: dict[str, Any]) -> None:
        """Parse cluster_nodes.yaml format (primary format)."""
        for node_id, node_cfg in config.get("nodes", {}).items():
            if node_id in self.nodes:
                continue  # Don't override existing

            ssh = node_cfg.get("ssh", {})
            paths = node_cfg.get("paths", {})
            resources = node_cfg.get("resources", {})

            self.nodes[node_id] = NodeConfig(
                node_id=node_id,
                ssh_host=ssh.get("host", ""),
                ssh_user=ssh.get("user", "ubuntu"),
                ssh_key=ssh.get("key", ""),
                ssh_port=ssh.get("port", 22),
                tailscale_ip=node_cfg.get("tailscale_ip", ""),
                direct_ip=node_cfg.get("direct_ip", ""),
                ringrift_path=paths.get("ringrift", "/home/ubuntu/ringrift/ai-service"),
                venv_path=paths.get("venv", "/home/ubuntu/ringrift/ai-service/venv"),
                data_path=paths.get("data", "/home/ubuntu/ringrift/ai-service/data"),
                gpu_type=resources.get("gpu", ""),
                vram_gb=resources.get("vram_gb", 0),
                memory_gb=resources.get("memory_gb", 0),
                cpu_cores=resources.get("cpu_cores", 0),
                roles=node_cfg.get("roles", []),
                status=node_cfg.get("status", "active"),
                is_p2p_voter=node_cfg.get("p2p_voter", False),
                is_controller=node_cfg.get("controller", False),
            )

    def _parse_distributed_hosts(self, config: dict[str, Any]) -> None:
        """Parse distributed_hosts.yaml format (legacy)."""
        for node_id, node_cfg in config.items():
            if isinstance(node_cfg, dict) and node_id not in self.nodes:
                self.nodes[node_id] = NodeConfig(
                    node_id=node_id,
                    ssh_host=node_cfg.get("ssh_host", node_cfg.get("host", "")),
                    ssh_user=node_cfg.get("ssh_user", node_cfg.get("user", "ubuntu")),
                    ssh_key=node_cfg.get("ssh_key", ""),
                    ringrift_path=node_cfg.get("ringrift_path", node_cfg.get("path", "")),
                    gpu_type=node_cfg.get("gpu", ""),
                    is_p2p_voter=node_cfg.get("p2p_voter", False),
                )

    def _parse_p2p_hosts(self, config: dict[str, Any]) -> None:
        """Parse p2p_hosts.yaml format (legacy)."""
        known_hosts = config.get("known_hosts", [])
        for host_entry in known_hosts:
            if isinstance(host_entry, str):
                # Simple host:port format
                parts = host_entry.split(":")
                host = parts[0]
                # Try to extract node_id from hostname
                node_id = host.split(".")[0] if "." in host else host
                if node_id not in self.nodes:
                    self.nodes[node_id] = NodeConfig(
                        node_id=node_id,
                        ssh_host=host,
                    )

    def _parse_cluster_yaml(self, config: dict[str, Any]) -> None:
        """Parse cluster.yaml format (legacy)."""
        nodes = config.get("nodes", {})
        for node_id, node_cfg in nodes.items():
            if isinstance(node_cfg, dict) and node_id not in self.nodes:
                self.nodes[node_id] = NodeConfig(
                    node_id=node_id,
                    ssh_host=node_cfg.get("host", ""),
                    ssh_user=node_cfg.get("user", "ubuntu"),
                    ssh_key=node_cfg.get("key", ""),
                    ringrift_path=node_cfg.get("path", ""),
                )

    def get_node(self, node_id: str) -> NodeConfig | None:
        """Get configuration for a specific node."""
        # Check if cache is stale
        if time.time() - self._load_time > CONFIG_CACHE_TTL:
            self.reload()

        return self.nodes.get(node_id)

    def get_nodes(self, status: str | None = None) -> list[NodeConfig]:
        """Get all nodes, optionally filtered by status."""
        # Check if cache is stale
        if time.time() - self._load_time > CONFIG_CACHE_TTL:
            self.reload()

        nodes = list(self.nodes.values())
        if status:
            nodes = [n for n in nodes if n.status == status]
        return nodes

    def get_training_nodes(self) -> list[NodeConfig]:
        """Get nodes capable of training."""
        return [n for n in self.get_nodes("active") if n.is_training_node]

    def get_gpu_nodes(self, gpu_type: str | None = None) -> list[NodeConfig]:
        """Get nodes with GPUs, optionally filtered by type."""
        nodes = [n for n in self.get_nodes("active") if n.gpu_type]
        if gpu_type:
            nodes = [n for n in nodes if gpu_type.lower() in n.gpu_type.lower()]
        return nodes

    def get_p2p_voters(self) -> list[NodeConfig]:
        """Get P2P voter nodes."""
        return [n for n in self.get_nodes("active") if n.is_p2p_voter]

    def get_controllers(self) -> list[NodeConfig]:
        """Get controller nodes (run unified_ai_loop)."""
        return [n for n in self.get_nodes("active") if n.is_controller]

    def get_ssh_config(self) -> dict[str, dict[str, Any]]:
        """Get SSH configuration for all nodes (for JobReaperDaemon)."""
        return {
            node_id: {
                "host": node.ssh_host,
                "user": node.ssh_user,
                "key": node.ssh_key,
                "port": node.ssh_port,
            }
            for node_id, node in self.nodes.items()
            if node.ssh_host
        }

    def validate_connectivity(self) -> dict[str, bool]:
        """Check connectivity to all active nodes."""
        results = {}
        for node in self.get_nodes("active"):
            results[node.node_id] = node.is_reachable(force_check=True)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get configuration statistics."""
        nodes = self.get_nodes()
        return {
            "total_nodes": len(nodes),
            "active_nodes": len([n for n in nodes if n.status == "active"]),
            "training_nodes": len([n for n in nodes if n.is_training_node]),
            "p2p_voters": len([n for n in nodes if n.is_p2p_voter]),
            "controllers": len([n for n in nodes if n.is_controller]),
            "gpu_types": list(set(n.gpu_type for n in nodes if n.gpu_type)),
            "load_time": self._load_time,
            "config_files": [str(p) for p in self.config_paths if p.exists()],
        }


# Singleton instance
_cluster_config: UnifiedClusterConfig | None = None
_config_lock = threading.Lock()


def get_cluster_config() -> UnifiedClusterConfig:
    """Get the singleton cluster configuration instance."""
    global _cluster_config
    if _cluster_config is None:
        with _config_lock:
            if _cluster_config is None:
                _cluster_config = UnifiedClusterConfig()
    return _cluster_config


def reload_cluster_config() -> None:
    """Force reload of cluster configuration."""
    config = get_cluster_config()
    config.reload()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "NodeConfig",
    "UnifiedClusterConfig",
    "get_cluster_config",
    "reload_cluster_config",
]
