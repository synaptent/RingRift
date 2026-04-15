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


def test_package_dir_lists_lazy_deprecated_cluster_symbols() -> None:
    module = importlib.import_module("app.distributed")

    for name in (
        "ClusterCoordinator",
        "TaskRole",
        "ProcessLimits",
        "TaskInfo",
        "check_and_abort_if_role_held",
    ):
        assert name in module.__all__
        assert name in dir(module)


def test_lazy_deprecated_symbol_access_emits_warning() -> None:
    module = importlib.import_module("app.distributed")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = module.ClusterCoordinator

    warning_text = [str(item.message) for item in caught]
    assert any("deprecated" in text.lower() for text in warning_text)
    assert any("Q3 2026" in text for text in warning_text)
    assert any("task_coordinator.TaskCoordinator" in text for text in warning_text)
