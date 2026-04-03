"""
Shared cluster host discovery and connectivity utilities.

Used by:
- scripts/elo_db_sync.py - Elo database synchronization
- scripts/aria2_data_sync.py - Model and data sync
- scripts/validate_cluster_elo.py - Elo validation
- app/training/elo_reconciliation.py - Elo drift reconciliation

.. deprecated:: December 2025
    This module is being consolidated into app/config/cluster_config.py.
    Import ClusterNode and helper functions from there instead:

        from app.config.cluster_config import (
            ClusterNode,
            get_cluster_nodes,
            get_active_nodes,
            get_coordinator_node,
            get_elo_sync_config,
        )

    This module re-exports for backward compatibility.
"""
from __future__ import annotations


import json
import socket
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Re-export from cluster_config for backward compatibility
from app.config.cluster_config import (
    ClusterNode,
    EloSyncConfig,
    get_cluster_nodes as _get_cluster_nodes,
    get_active_nodes as _get_active_nodes,
    get_coordinator_node as _get_coordinator_node,
    get_elo_sync_config as _get_elo_sync_config,
    load_cluster_config as _load_cluster_config,
)

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


def load_hosts_config() -> dict[str, Any]:
    """Load raw hosts config from distributed_hosts.yaml.

    .. deprecated:: December 2025
        Use load_cluster_config() from app.config.cluster_config instead.
    """
    warnings.warn(
        "load_hosts_config() is deprecated. Use load_cluster_config() from "
        "app.config.cluster_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    config_path = _resolve_config_path()
    if not config_path.exists():
        return {}

    try:
        import yaml
    except ImportError:
        return {}

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}

    return data if isinstance(data, dict) else {}


def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve the effective config path for legacy cluster_hosts callers.

    The historical contract for this shim was that patching HOSTS_CONFIG
    changed all helper behavior. Keep honoring that even though the
    underlying implementation now lives in app.config.cluster_config.
    """
    return Path(config_path) if config_path is not None else Path(HOSTS_CONFIG)


def load_cluster_config(
    config_path: str | Path | None = None,
    *,
    force_reload: bool = False,
):
    """Load cluster config honoring the legacy HOSTS_CONFIG override."""
    return _load_cluster_config(_resolve_config_path(config_path), force_reload=force_reload)


def get_elo_sync_config(config_path: str | Path | None = None) -> EloSyncConfig:
    """Get Elo sync config honoring the legacy HOSTS_CONFIG override."""
    return _get_elo_sync_config(_resolve_config_path(config_path))


def get_cluster_nodes(config_path: str | Path | None = None) -> dict[str, ClusterNode]:
    """Get cluster nodes honoring the legacy HOSTS_CONFIG override."""
    resolved_path = _resolve_config_path(config_path)
    nodes = _get_cluster_nodes(resolved_path)
    raw_hosts = _load_cluster_config(resolved_path).hosts_raw
    default_data_port = _get_default_data_server_port()

    for name, node in nodes.items():
        if "data_server_port" not in raw_hosts.get(name, {}):
            node.data_server_port = default_data_port

    return nodes


def get_active_nodes(config_path: str | Path | None = None) -> list[ClusterNode]:
    """Get active nodes honoring the legacy HOSTS_CONFIG override."""
    return _get_active_nodes(_resolve_config_path(config_path))


def get_coordinator_node(config_path: str | Path | None = None) -> ClusterNode | None:
    """Get the coordinator node honoring the legacy HOSTS_CONFIG override."""
    return _get_coordinator_node(_resolve_config_path(config_path))


# NOTE: The following functions were removed Dec 27, 2025 (now imported from cluster_config):
# - ClusterNode dataclass
# - EloSyncConfig dataclass
# - get_elo_sync_config()
# - get_cluster_nodes()
# - get_active_nodes()
# - get_coordinator_node()
# See app/config/cluster_config.py for implementations.


def get_coordinator_address() -> tuple[str | None, int]:
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
