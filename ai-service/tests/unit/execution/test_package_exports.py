"""Focused tests for app.execution package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_execution_surface() -> None:
    module = importlib.import_module("app.execution")

    expected = [
        "BackendType",
        "BaseExecutor",
        "ExecutionResult",
        "ExecutorPool",
        "GameExecutor",
        "GameOutcome",
        "GameResult",
        "JobResult",
        "LocalBackend",
        "LocalExecutor",
        "OrchestratorBackend",
        "ParallelGameExecutor",
        "SSHBackend",
        "SSHExecutor",
        "SlurmBackend",
        "WorkerStatus",
        "get_backend",
        "run_command",
        "run_quick_game",
        "run_selfplay_batch",
        "run_ssh_command",
        "run_ssh_command_async",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
