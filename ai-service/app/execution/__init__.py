"""Unified execution framework for the RingRift AI service.

This package provides abstracted execution backends for running commands
locally, via SSH, or on cloud workers. All orchestrators should use this
instead of implementing their own SSH/subprocess logic.

Low-level execution:
    from app.execution import SSHExecutor, LocalExecutor, ExecutionResult

    # SSH execution
    executor = SSHExecutor(host="worker-1", user="ringrift")
    result = await executor.run("python scripts/run_selfplay.py")

    # Local execution
    executor = LocalExecutor()
    result = await executor.run("python scripts/train.py")

High-level orchestrator backends:
    from app.execution import get_backend, BackendType

    # Get configured backend (auto-detects from config)
    backend = get_backend()

    # Run selfplay across all available workers
    results = await backend.run_selfplay(games=100, board_type="square8", num_players=2)

    # Run tournament
    result = await backend.run_tournament(
        agent_ids=["random", "heuristic"],
        games_per_pairing=20,
    )
"""

from app.execution.executor import (
    ExecutionResult,
    BaseExecutor,
    LocalExecutor,
    SSHExecutor,
    ExecutorPool,
    run_command,
    run_command_async,
    run_ssh_command,
    run_ssh_command_async,
)
from app.execution.backends import (
    BackendType,
    WorkerStatus,
    JobResult,
    OrchestratorBackend,
    LocalBackend,
    SSHBackend,
    get_backend,
)

__all__ = [
    # Low-level executors
    "ExecutionResult",
    "BaseExecutor",
    "LocalExecutor",
    "SSHExecutor",
    "ExecutorPool",
    "run_command",
    "run_command_async",
    "run_ssh_command",
    "run_ssh_command_async",
    # High-level backends
    "BackendType",
    "WorkerStatus",
    "JobResult",
    "OrchestratorBackend",
    "LocalBackend",
    "SSHBackend",
    "get_backend",
]
