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

Game execution:
    from app.execution import GameExecutor, GameResult, run_quick_game

    # Run a single game
    executor = GameExecutor(board_type="square8", num_players=2)
    result = executor.run_game(
        player_configs=[
            {"ai_type": "mcts", "difficulty": 5},
            {"ai_type": "heuristic", "difficulty": 3},
        ],
    )

    # Quick game helper
    result = run_quick_game(p1_type="mcts", p2_type="heuristic")
"""

from app.execution.backends import (
    BackendType,
    JobResult,
    LocalBackend,
    OrchestratorBackend,
    SlurmBackend,
    SSHBackend,
    WorkerStatus,
    get_backend,
)
from app.execution.executor import (
    BaseExecutor,
    ExecutionResult,
    ExecutorPool,
    LocalExecutor,
    SSHExecutor,
    run_command,
    run_ssh_command,
    run_ssh_command_async,
)
from app.execution.game_executor import (
    GameExecutor,
    GameOutcome,
    GameResult,
    ParallelGameExecutor,
    run_quick_game,
    run_selfplay_batch,
)

__all__ = [
    # High-level backends
    "BackendType",
    "BaseExecutor",
    # Low-level executors
    "ExecutionResult",
    "ExecutorPool",
    "GameExecutor",
    # Game execution
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
