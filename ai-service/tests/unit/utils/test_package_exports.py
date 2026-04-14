"""Focused tests for app.utils package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_utils_surface() -> None:
    module = importlib.import_module("app.utils")

    expected = [
        "safe_load_npz",
        "get_device",
        "get_device_info",
        "EnvConfig",
        "env",
        "get_bool",
        "get_float",
        "get_int",
        "get_list",
        "get_str",
        "GameDiscovery",
        "find_all_game_databases",
        "count_games_for_config",
        "get_game_counts_summary",
        "retry",
        "retry_async",
        "RetryConfig",
        "RETRY_STANDARD",
        "RETRY_SSH",
        "RETRY_HTTP",
        "check_disk_space_available",
        "ensure_disk_space",
        "get_available_disk_space",
        "ProgressReporter",
        "SoakProgressReporter",
        "OptimizationProgressReporter",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
