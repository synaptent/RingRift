"""Focused tests for app.testing package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_testing_surface() -> None:
    module = importlib.import_module("app.testing")

    expected = [
        "create_board_state",
        "create_game_state",
        "create_move",
        "create_player",
        "create_position",
        "create_ring_stack",
        "MockClusterState",
        "MockEventBus",
        "MockHostSyncState",
        "MockNodeResources",
        "MockTrainingState",
        "create_cluster_state",
        "create_coordination_db",
        "create_host_sync_state",
        "create_node_resources",
        "create_temp_db",
        "create_training_state",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
