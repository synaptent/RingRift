"""
Shared cluster host discovery and connectivity utilities.

Used by:
- scripts/elo_db_sync.py - Elo database synchronization
- scripts/aria2_data_sync.py - Model and data sync
- scripts/validate_cluster_elo.py - Elo validation
- app/training/elo_reconciliation.py - Elo drift reconciliation

Centralizes host configuration loading from distributed_hosts.yaml
to eliminate duplication across sync components.
"""

import json
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
HOSTS_CONFIG = ROOT / "config" / "distributed_hosts.yaml"

# Default ports
ELO_SYNC_PORT = 8766
DATA_SYNC_PORT = 8766
MODEL_SYNC_PORT = 8765


def _get_default_data_server_port() -> int:
    try:
        from app.config.unified_config import get_config
        return get_config().distributed.data_server_port
    except (ImportError, AttributeError, KeyError):
        return DATA_SYNC_PORT


@dataclass
class ClusterNode:
    """Represents a cluster node with connectivity info."""
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
    data_server_port: int = DATA_SYNC_PORT
    data_server_url: str | None = None

    @property
    def best_ip(self) -> str | None:
        """Get best IP for connection (prefer Tailscale)."""
        for candidate in (self.tailscale_ip, self.ssh_host):
            if not candidate:
                continue
            host = str(candidate).strip()
            if not host:
                continue
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


@dataclass
class EloSyncConfig:
    """Elo sync configuration from distributed_hosts.yaml."""
    coordinator: str = "mac-studio"
    sync_port: int = 8766
    sync_interval: int = 300
    divergence_threshold: int = 50
    transports: list[str] = field(default_factory=lambda: ["tailscale", "aria2", "http"])


def load_hosts_config() -> dict[str, Any]:
    """Load raw hosts config from distributed_hosts.yaml."""
    if not HOSTS_CONFIG.exists():
        return {}

    try:
        import yaml
        with open(HOSTS_CONFIG) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fallback: basic YAML parsing
        result = {"hosts": {}, "elo_sync": {}}
        current_section = None
        current_host = None

        with open(HOSTS_CONFIG) as f:
            for line in f:
                line = line.rstrip()
                if not line or line.startswith('#'):
                    continue

                if line == "hosts:":
                    current_section = "hosts"
                    continue
                elif line == "elo_sync:":
                    current_section = "elo_sync"
                    continue

                if current_section == "hosts":
                    if line.startswith('  ') and ':' in line and not line.startswith('    '):
                        current_host = line.strip().rstrip(':')
                        result["hosts"][current_host] = {}
                    elif current_host and line.startswith('    ') and ':' in line:
                        key, _, value = line.strip().partition(':')
                        value = value.strip().strip('"\'')
                        if value.isdigit():
                            value = int(value)
                        result["hosts"][current_host][key] = value

                elif (current_section == "elo_sync"
                        and line.startswith('  ') and ':' in line and not line.startswith('    ')):
                    key, _, value = line.strip().partition(':')
                    value = value.strip().strip('"\'')
                    if value.isdigit():
                        value = int(value)
                    result["elo_sync"][key] = value

        return result
    except (OSError, ValueError):
        return {}


def get_elo_sync_config() -> EloSyncConfig:
    """Get Elo sync configuration."""
    config = load_hosts_config().get("elo_sync", {})
    return EloSyncConfig(
        coordinator=config.get("coordinator", "mac-studio"),
        sync_port=config.get("sync_port", 8766),
        sync_interval=config.get("sync_interval", 300),
        divergence_threshold=config.get("divergence_threshold", 50),
        transports=config.get("transports", ["tailscale", "aria2", "http"]),
    )


def get_cluster_nodes() -> dict[str, ClusterNode]:
    """Get all cluster nodes from config."""
    hosts_config = load_hosts_config().get("hosts", {})
    nodes = {}
    data_server_port = _get_default_data_server_port()

    for name, cfg in hosts_config.items():
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
            data_server_port=cfg.get("data_server_port", data_server_port),
            data_server_url=cfg.get("data_server_url"),
        )

    return nodes


def get_active_nodes() -> list[ClusterNode]:
    """Get all active (non-terminated) cluster nodes."""
    return [n for n in get_cluster_nodes().values() if n.is_active]


def get_coordinator_node() -> ClusterNode | None:
    """Get the Elo coordinator node."""
    sync_config = get_elo_sync_config()
    nodes = get_cluster_nodes()
    return nodes.get(sync_config.coordinator)


def get_coordinator_address() -> tuple[str, int]:
    """Get coordinator IP and port."""
    sync_config = get_elo_sync_config()
    coord_node = get_coordinator_node()

    if coord_node and coord_node.best_ip:
        return coord_node.best_ip, sync_config.sync_port

    # Check environment variable fallback
    import os
    fallback_ip = os.environ.get("RINGRIFT_COORDINATOR_IP")
    if fallback_ip:
        return fallback_ip, sync_config.sync_port

    # No coordinator configured
    return None, sync_config.sync_port


def check_http_endpoint(ip: str, port: int, path: str = "/status", timeout: int = 5) -> dict | None:
    """Check if an HTTP endpoint is reachable and return response data."""
    try:
        url = f"http://{ip}:{port}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (OSError, ValueError, TimeoutError):
        return None


def check_node_reachable(node: ClusterNode, port: int = ELO_SYNC_PORT, timeout: int = 5) -> bool:
    """Check if a node's sync endpoint is reachable."""
    ip = node.best_ip
    if not ip:
        return False
    return check_http_endpoint(ip, port, "/status", timeout) is not None


def discover_reachable_nodes(port: int = ELO_SYNC_PORT, timeout: int = 5) -> list[tuple[ClusterNode, dict]]:
    """Discover all reachable nodes in parallel, returning node and status."""
    nodes = get_active_nodes()
    reachable = []

    def check_node(node):
        ip = node.best_ip
        if not ip:
            return None
        status = check_http_endpoint(ip, port, "/status", timeout)
        if status:
            return (node, status)
        return None

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(check_node, n): n for n in nodes}
        for future in as_completed(futures, timeout=timeout + 10):
            try:
                result = future.result()
                if result:
                    reachable.append(result)
            except (OSError, TimeoutError, ValueError):
                pass

    return reachable


def get_sync_urls(port: int = ELO_SYNC_PORT, path: str = "/db") -> list[str]:
    """Get URLs for all reachable sync endpoints."""
    reachable = discover_reachable_nodes(port)
    return [f"http://{node.best_ip}:{port}{path}" for node, _ in reachable if node.best_ip]


# Convenience functions for specific sync types
def get_elo_sync_urls() -> list[str]:
    """Get URLs for Elo database sync."""
    return get_sync_urls(ELO_SYNC_PORT, "/db")


def get_data_sync_urls(
    exclude_self: bool = True,
    reachable_only: bool = True,
    timeout: int = 5,
) -> list[str]:
    """Get URLs for data sync (games, training)."""
    if reachable_only:
        reachable = discover_reachable_nodes(_get_default_data_server_port(), timeout)
        nodes = [node for node, _ in reachable]
    else:
        nodes = get_active_nodes()

    hostname = socket.gethostname().lower()
    urls: list[str] = []

    for node in nodes:
        if exclude_self and node.name.lower() == hostname:
            continue
        base_url = node.data_server_base_url
        if base_url and base_url not in urls:
            urls.append(base_url)

    return urls
