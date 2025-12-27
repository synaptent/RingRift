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
    get_cluster_nodes,
    get_active_nodes,
    get_coordinator_node,
    get_elo_sync_config,
    load_cluster_config,
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
    config = load_cluster_config()
    # Reconstruct the raw format for backward compatibility
    return {
        "hosts": config.hosts_raw,
        "elo_sync": {
            "coordinator": config.elo_sync.coordinator,
            "sync_port": config.elo_sync.sync_port,
            "sync_interval": config.elo_sync.sync_interval,
            "divergence_threshold": config.elo_sync.divergence_threshold,
            "transports": config.elo_sync.transports,
        },
    }


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
