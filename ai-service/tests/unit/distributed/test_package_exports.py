"""Focused tests for app.distributed package exports."""

from __future__ import annotations

import importlib
import warnings


def test_package_dir_lists_key_public_exports() -> None:
    module = importlib.import_module("app.distributed")

    expected = {
        "DistributedEvaluator",
        "QueueDistributedEvaluator",
        "discover_workers",
        "filter_healthy_workers",
        "parse_manual_workers",
        "wait_for_workers",
        "write_games_to_db",
        "HostConfig",
        "SSHExecutor",
        "MemoryTracker",
        "RemoteMemoryMonitor",
        "SyncCoordinator",
        "UnifiedWAL",
    }

    assert expected.issubset(set(module.__all__))
    for name in expected:
        assert name in dir(module)


def test_package_dir_excludes_removed_deprecated_cluster_symbols() -> None:
    module = importlib.import_module("app.distributed")

    for name in (
        "ClusterCoordinator",
        "TaskRole",
        "ProcessLimits",
        "TaskInfo",
        "check_and_abort_if_role_held",
    ):
        assert name not in module.__all__
        assert name not in dir(module)
        assert not hasattr(module, name)


def test_deprecated_cluster_coordinator_remains_direct_submodule_only() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("app.distributed.cluster_coordinator")

    assert hasattr(module, "ClusterCoordinator")
    assert any("deprecated" in str(item.message).lower() for item in caught)
