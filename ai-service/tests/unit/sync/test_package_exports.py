"""Focused tests for app.sync package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_sync_surface() -> None:
    module = importlib.import_module("app.sync")

    expected = [
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

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
