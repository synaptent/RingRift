"""Focused tests for app.core package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_core_surface() -> None:
    module = importlib.import_module("app.core")

    expected = [
        "Codec",
        "Serializable",
        "SerializationError",
        "deserialize",
        "from_json",
        "register_codec",
        "serialize",
        "to_json",
        "FatalError",
        "RetryableError",
        "RingRiftError",
        "retry",
        "retry_async",
        "with_emergency_halt_check",
        "ShutdownManager",
        "get_shutdown_manager",
        "is_shutting_down",
        "on_shutdown",
        "request_shutdown",
        "shutdown_scope",
        "SingletonMeta",
        "SingletonMixin",
        "ThreadSafeSingletonMixin",
        "singleton",
        "TaskInfo",
        "TaskManager",
        "TaskState",
        "background_task",
        "get_task_manager",
        "get_logger",
        "setup_logging",
        "ConnectionInfo",
        "GPUInfo",
        "HealthStatus",
        "JobStatus",
        "NodeHealth",
        "NodeInfo",
        "NodeRole",
        "NodeState",
        "Provider",
        "ProviderInfo",
        "ResourceMetrics",
        "SSHClient",
        "SSHConfig",
        "SSHResult",
        "get_ssh_client",
        "run_ssh_command",
        "run_ssh_command_async",
        "run_ssh_command_sync",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
