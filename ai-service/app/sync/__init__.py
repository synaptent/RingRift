"""Cluster synchronization utilities for RingRift AI service.

This module provides utilities for coordinating data synchronization
across the distributed training cluster.
"""

from app.sync.cluster_hosts import (
    ClusterNode,
    EloSyncConfig,
    check_node_reachable,
    discover_reachable_nodes,
    get_active_nodes,
    get_cluster_nodes,
    get_coordinator_address,
    get_coordinator_node,
    get_data_sync_urls,
    get_elo_sync_config,
    get_elo_sync_urls,
    get_sync_urls,
    load_hosts_config,
)

__all__ = [
    "ClusterNode",
    "EloSyncConfig",
    "check_node_reachable",
    "discover_reachable_nodes",
    "get_active_nodes",
    "get_cluster_nodes",
    "get_coordinator_address",
    "get_coordinator_node",
    "get_data_sync_urls",
    "get_elo_sync_config",
    "get_elo_sync_urls",
    "get_sync_urls",
    "load_hosts_config",
]


def __dir__() -> list[str]:
    """Expose the intended sync surface for discoverability."""

    return sorted(set(globals()) | set(__all__))
